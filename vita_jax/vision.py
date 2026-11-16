"""Shared ImageNet ResNet18 visual path.

The JAX model exposes ``layer3`` and ``layer4`` feature maps like the
torchvision ``IntermediateLayerGetter`` used by the active run.  BatchNorm is
the frozen variant: running statistics and affine values live in a non-optimized
``constants`` collection, while convolutions remain trainable.  Optional
torchvision weight loading maps a standard ``state_dict`` into that layout.
"""

from __future__ import annotations

from typing import Dict, Mapping, MutableMapping, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class FrozenBatchNorm(nn.Module):
    """BatchNorm with fixed running stats/affine values (torchvision wrapper)."""

    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        channels = x.shape[-1]
        mean = self.variable("constants", "mean", lambda: jnp.zeros(channels, jnp.float32))
        var = self.variable("constants", "var", lambda: jnp.ones(channels, jnp.float32))
        scale = self.variable("constants", "scale", lambda: jnp.ones(channels, jnp.float32))
        bias = self.variable("constants", "bias", lambda: jnp.zeros(channels, jnp.float32))
        inv = scale.value / jnp.sqrt(var.value + self.eps)
        return x * inv + (bias.value - mean.value * inv)


class ResidualBlock(nn.Module):
    channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.Conv(self.channels, (3, 3), (self.stride, self.stride),
                    padding=((1, 1), (1, 1)), use_bias=False, name="conv1")(x)
        y = FrozenBatchNorm(name="bn1")(y)
        y = nn.relu(y)
        y = nn.Conv(self.channels, (3, 3), (1, 1),
                    padding=((1, 1), (1, 1)), use_bias=False, name="conv2")(y)
        y = FrozenBatchNorm(name="bn2")(y)
        if self.stride != 1 or residual.shape[-1] != self.channels:
            residual = nn.Conv(self.channels, (1, 1), (self.stride, self.stride),
                               padding="VALID", use_bias=False,
                               name="downsample_conv")(residual)
            residual = FrozenBatchNorm(name="downsample_bn")(residual)
        return nn.relu(y + residual)


def max_pool_torchvision(x):
    """Torchvision maxpool for NCHW input with explicit symmetric padding."""
    x = x.transpose(0, 2, 3, 1)
    x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)),
                constant_values=-jnp.inf)
    x = nn.max_pool(x, (3, 3), (2, 2), padding="VALID")
    return x.transpose(0, 3, 1, 2)


class ResNet18Features(nn.Module):
    """ResNet18 that returns layer3 and layer4 maps for FASA."""

    @nn.compact
    def __call__(self, x):
        # Flax convolutions use NHWC; the policy/vision interface is NCHW.
        x = x.transpose(0, 2, 3, 1)
        x = nn.Conv(64, (7, 7), (2, 2), padding=((3, 3), (3, 3)),
                    use_bias=False, name="stem_conv")(x)
        x = FrozenBatchNorm(name="stem_bn")(x)
        x = nn.relu(x)
        # Match torchvision's explicit symmetric max-pool padding.
        x = max_pool_torchvision(x.transpose(0, 3, 1, 2))
        x = x.transpose(0, 2, 3, 1)
        stages = {}
        for layer, channels in (("layer1", 64), ("layer2", 128),
                                ("layer3", 256), ("layer4", 512)):
            for block in range(2):
                stride = 2 if (layer != "layer1" and block == 0) else 1
                x = ResidualBlock(channels, stride=stride,
                                  name=f"{layer}_{block}")(x)
            stages[layer] = x
        return (stages["layer3"].transpose(0, 3, 1, 2),
                stages["layer4"].transpose(0, 3, 1, 2))


# ---------------------------------------------------------------------------
# Optional torchvision weight conversion
# ---------------------------------------------------------------------------

def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return jax.tree_util.tree_map(lambda v: np.asarray(v), value)


def _torch_get(state_dict: Mapping, key: str):
    for candidate in (key, "module." + key, "vision." + key,
                      "resnet18." + key):
        if candidate in state_dict:
            return _as_numpy(state_dict[candidate])
    raise KeyError(key)


def _copy_conv_weights(params: MutableMapping, constants: MutableMapping,
                       torch_key: str, param_key: Tuple[str, ...],
                       constant_key: Tuple[str, ...], state_dict: Mapping,
                       torch_bn_prefix: Optional[str] = None):
    kernel = _torch_get(state_dict, torch_key)
    # PyTorch OIHW -> Flax HWIO
    kernel = kernel.transpose(2, 3, 1, 0)
    node = params
    for key in param_key[:-1]:
        node = node.setdefault(key, {})
    node[param_key[-1]] = {"kernel": kernel}
    prefix = (torch_bn_prefix if torch_bn_prefix is not None
              else torch_key.rsplit(".", 1)[0])
    for short, name in (("weight", "scale"), ("bias", "bias"),
                        ("running_mean", "mean"), ("running_var", "var")):
        key = f"{prefix}.{short}"
        value = _torch_get(state_dict, key)
        node = constants
        for part in constant_key[:-1]:
            node = node.setdefault(part, {})
        node.setdefault(constant_key[-1], {})[name] = value


def load_torchvision_resnet18(params: Mapping, constants: Mapping,
                              state_dict: Mapping):
    """Return ``(params, constants)`` with torchvision ResNet18 weights.

    Accepts a plain torchvision ``state_dict`` (with optional ``module.``,
    ``vision.`` or ``resnet18.`` prefixes).  The caller remains responsible for
    obtaining the ImageNet checkpoint; this function performs no download.
    """
    params = jax.tree_util.tree_map(lambda x: x, dict(params))
    constants = jax.tree_util.tree_map(lambda x: x, dict(constants))
    _copy_conv_weights(params, constants, "conv1.weight", ("stem_conv",),
                       ("stem_bn",), state_dict, torch_bn_prefix="bn1")
    for layer, blocks in (("layer1", 2), ("layer2", 2),
                          ("layer3", 2), ("layer4", 2)):
        for block in range(blocks):
            base = f"{layer}.{block}"
            name = f"{layer}_{block}"
            _copy_conv_weights(params, constants, base + ".conv1.weight",
                               (name, "conv1"), (name, "bn1"), state_dict,
                               torch_bn_prefix=base + ".bn1")
            _copy_conv_weights(params, constants, base + ".conv2.weight",
                               (name, "conv2"), (name, "bn2"), state_dict,
                               torch_bn_prefix=base + ".bn2")
            downsample = f"{base}.downsample.0.weight"
            try:
                _copy_conv_weights(params, constants, downsample,
                                   (name, "downsample_conv"),
                                   (name, "downsample_bn"), state_dict,
                                   torch_bn_prefix=base + ".downsample.1")
            except KeyError:
                pass
    return params, constants


def load_resnet18_state_dict(path: str) -> Dict[str, np.ndarray]:
    """Load a torchvision ``state_dict`` from ``.npz``, ``.pt`` or ``.pth``.

    No download is performed.  Plain dictionaries and common wrapper keys
    (``state_dict``/``model_state_dict``/``model``) are accepted.
    """
    path = str(path)
    suffix = path.rsplit(".", 1)[-1].lower()
    if suffix == "npz":
        with np.load(path) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency specific
        raise ImportError(
            "torch is required to read a non-npz ResNet18 state_dict") from exc
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        obj = torch.load(path, map_location="cpu")
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            value = obj.get(key)
            if isinstance(value, Mapping):
                obj = value
                break
    if not isinstance(obj, Mapping):
        raise ValueError(f"no state_dict found in {path!r}")
    return {str(key): _as_numpy(value) for key, value in obj.items()}


def load_resnet18_weights(path: str, params: Mapping, constants: Mapping):
    """Convert and inject a local torchvision ResNet18 checkpoint."""
    state_dict = load_resnet18_state_dict(path)
    return load_torchvision_resnet18(params, constants, state_dict)
