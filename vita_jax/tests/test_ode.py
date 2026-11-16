import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax
import jax.numpy as jnp

from vita_jax.config import ModelConfig
from vita_jax.ode import VITAODEHead
from vita_jax.pose import normalize_pose, state_to_pose14


def _cond(batch=2, dim=8):
    b = batch
    patches = jnp.zeros((b, 2, 3, 512, 2, 2))
    weights = jnp.ones((b, 2, 3, 1, 2, 2))
    return {"pooled": jnp.zeros((b, dim)), "patches": patches,
            "weights": weights}


def _stats():
    return (jnp.zeros((16,)), jnp.ones((16,)), jnp.zeros((14,)), jnp.ones((14,)))


def test_ode_path_shape_and_zero_velocity_start():
    cfg = ModelConfig(ode_hidden_dim=16, ode_num_layers=2, chunk_size=6,
                      ppf_future_visual_hidden=16)
    head = VITAODEHead(cfg, pooled_dim=8)
    cond = _cond()
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 2, 16))
    sm, ss, pm, ps = _stats()
    variables = head.init(jax.random.PRNGKey(1), cond, obs, sm, ss, pm, ps,
                          train=False)
    path = head.apply(variables, cond, obs, sm, ss, pm, ps, train=False)
    assert path.shape == (2, 6, 14)
    a0 = normalize_pose(state_to_pose14(obs[:, -1]), pm, ps)
    # The output layer is zero-initialized, so Euler leaves a0 unchanged.
    assert jnp.allclose(path[:, 0], a0, atol=1e-6)
    assert jnp.allclose(path, path[:, :1], atol=1e-6)


def test_ode_train_perturb_is_train_only():
    cfg = ModelConfig(ode_hidden_dim=16, ode_num_layers=2, chunk_size=4,
                      ode_perturb_prob=1.0, ode_perturb_std=0.1,
                      ppf_future_visual_hidden=16)
    head = VITAODEHead(cfg, pooled_dim=8)
    cond = _cond()
    obs = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 16))
    sm, ss, pm, ps = _stats()
    variables = head.init(jax.random.PRNGKey(3), cond, obs, sm, ss, pm, ps,
                          train=False)
    eval_path = head.apply(variables, cond, obs, sm, ss, pm, ps, train=False)
    train_path = head.apply(variables, cond, obs, sm, ss, pm, ps, train=True,
                            rngs={"perturb": jax.random.PRNGKey(4)})
    assert not jnp.allclose(eval_path, train_path)
