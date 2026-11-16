import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax
import jax.numpy as jnp

from vita_jax.fasa import ObjectChangeFusion


def test_fasa_no_motion_is_finite_and_zero_dynamic():
    module = ObjectChangeFusion(feat_dim=8, use_activation_prior=True,
                                mid_feat_dim=0, build_router=True, v2=True)
    feat = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 4, 4))
    gate = jax.nn.sigmoid(jax.random.normal(
        jax.random.PRNGKey(1), (2, 1, 4, 4)))
    variables = module.init(jax.random.PRNGKey(2), feat, feat, feat, gate, gate,
                            train=True)
    out = module.apply(variables, feat, feat, feat, gate, gate, train=False)
    for value in jax.tree_util.tree_leaves(out):
        if hasattr(value, "shape") and value.dtype == jnp.float32:
            assert jnp.all(jnp.isfinite(value))
    assert jnp.allclose(out["dynamic"], 0.0)
    assert jnp.all(out["identical_frames"])
    assert jnp.allclose(out["attention"], 0.0)
