import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax
import jax.numpy as jnp
import numpy as np

from vita_jax.config import (DataConfig, ModelConfig, TextConfig, TrainConfig,
                             VITAConfig)
from vita_jax.policy import VITAPolicy
from vita_jax.vision import (ResNet18Features, load_resnet18_weights,
                             load_torchvision_resnet18)


def _state_dict_from_variables(variables):
    params = variables["params"]
    constants = variables["constants"]
    out = {}

    def conv(torch_key, node):
        out[torch_key + ".weight"] = np.asarray(
            node["kernel"]).transpose(3, 2, 0, 1)

    def bn(torch_key, node):
        out[torch_key + ".weight"] = np.asarray(node["scale"])
        out[torch_key + ".bias"] = np.asarray(node["bias"])
        out[torch_key + ".running_mean"] = np.asarray(node["mean"])
        out[torch_key + ".running_var"] = np.asarray(node["var"])

    conv("conv1", params["stem_conv"])
    bn("bn1", constants["stem_bn"])
    for layer in ("layer1", "layer2", "layer3", "layer4"):
        for block in range(2):
            name = f"{layer}_{block}"
            base = f"{layer}.{block}"
            conv(base + ".conv1", params[name]["conv1"])
            bn(base + ".bn1", constants[name]["bn1"])
            conv(base + ".conv2", params[name]["conv2"])
            bn(base + ".bn2", constants[name]["bn2"])
            if "downsample_conv" in params[name]:
                conv(base + ".downsample.0", params[name]["downsample_conv"])
                bn(base + ".downsample.1", constants[name]["downsample_bn"])
    return out


def _assert_tree_close(left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for a, b in zip(left_leaves, right_leaves):
        assert np.allclose(np.asarray(a), np.asarray(b))


def test_resnet_stem_max_pool_matches_torchvision_padding():
    pytest.importorskip("torch")
    import torch
    import torch.nn.functional as F
    x = np.arange(2 * 7 * 9, dtype=np.float32).reshape(1, 2, 7, 9)
    expected = F.max_pool2d(F.pad(torch.from_numpy(x), (1, 1, 1, 1),
                                  value=float("-inf")), 3, 2)
    from vita_jax.vision import max_pool_torchvision
    actual = max_pool_torchvision(jnp.asarray(x))
    assert np.array_equal(np.asarray(actual), expected.numpy())


def test_torchvision_mapping_uses_bn_names_and_injects_npz(tmp_path):
    module = ResNet18Features()
    image = jnp.zeros((1, 3, 32, 32), jnp.float32)
    source = module.init(jax.random.PRNGKey(0), image)
    target = module.init(jax.random.PRNGKey(1), image)
    state_dict = _state_dict_from_variables(target)

    params, constants = load_torchvision_resnet18(
        source["params"], source["constants"], state_dict)
    _assert_tree_close(params, target["params"])
    _assert_tree_close(constants, target["constants"])
    assert not np.allclose(np.asarray(source["params"]["stem_conv"]["kernel"]),
                           np.asarray(params["stem_conv"]["kernel"]))

    path = tmp_path / "resnet18.npz"
    np.savez(path, **state_dict)
    params_file, constants_file = load_resnet18_weights(
        str(path), source["params"], source["constants"])
    _assert_tree_close(params_file, target["params"])
    _assert_tree_close(constants_file, target["constants"])


def test_fresh_policy_init_injects_local_imagenet_weights(tmp_path):
    module = ResNet18Features()
    image = jnp.zeros((1, 3, 32, 32), jnp.float32)
    source = module.init(jax.random.PRNGKey(2), image)
    path = tmp_path / "resnet18.npz"
    np.savez(path, **_state_dict_from_variables(source))

    cfg = VITAConfig(
        model=ModelConfig(resnet18_imagenet_weights=str(path),
                          ppf_future_visual_weight=0.0,
                          ode_hidden_dim=16, ode_num_layers=2,
                          resize_shape=(32, 32), crop_shape=(24, 24),
                          state_dropout=0.0, ode_perturb_prob=0.0),
        text=TextConfig(encoder="synthetic", joint_planner=False,
                        generated_text_targets=False, max_length=8),
        data=DataConfig(data_dir="x", repo_id="x", ee_cache="x"),
        train=TrainConfig(steps=1, batch_size=2))
    policy = VITAPolicy.create(cfg)
    params, constants, _ = policy.init(
        jax.random.PRNGKey(3), batch_size=2, height=32, width=32,
        allow_random_vision=False)
    template = module.init(jax.random.PRNGKey(4), image)
    expected_params, expected_constants = load_torchvision_resnet18(
        template["params"], template["constants"],
        _state_dict_from_variables(source))
    _assert_tree_close(params["linen"]["vision"], expected_params)
    _assert_tree_close(constants["vision"], expected_constants)
    assert np.allclose(
        np.asarray(constants["vision"]["stem_bn"]["mean"]),
        np.asarray(source["constants"]["stem_bn"]["mean"]))
