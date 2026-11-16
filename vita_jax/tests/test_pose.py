import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax
import jax.numpy as jnp
import numpy as np

from vita_jax.pose import (normalize_pose, pose14_to_state16,
                           quat_wxyz_to_rotvec, rotvec_to_quat_wxyz,
                           state_to_pose14, unnormalize_pose)


def _random_quats(key, shape):
    q = jax.random.normal(key, (*shape, 4))
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    return q * jnp.where(q[..., :1] < 0, -1.0, 1.0)


def test_quat_rotvec_roundtrip():
    q = _random_quats(jax.random.PRNGKey(0), (8, 3))
    r = quat_wxyz_to_rotvec(q)
    q2 = rotvec_to_quat_wxyz(r)
    assert jnp.allclose(q2, q, atol=1e-6)
    assert r.shape == (8, 3, 3)


def test_state_pose_roundtrip_and_normalization():
    q = _random_quats(jax.random.PRNGKey(1), (5, 2))
    state = jnp.concatenate([
        jax.random.normal(jax.random.PRNGKey(2), (5, 3)),
        q[:, 0],
        jax.random.uniform(jax.random.PRNGKey(3), (5, 1)),
        jax.random.normal(jax.random.PRNGKey(4), (5, 3)),
        q[:, 1],
        jax.random.uniform(jax.random.PRNGKey(5), (5, 1)),
    ], axis=-1)
    pose = state_to_pose14(state)
    assert pose.shape == (5, 14)
    state2 = pose14_to_state16(pose)
    assert jnp.allclose(state2, state, atol=1e-6)
    mean = jnp.linspace(-0.5, 0.5, 14)
    std = jnp.linspace(0.5, 1.5, 14)
    norm = normalize_pose(pose, mean, std)
    back = unnormalize_pose(norm, mean, std)
    assert jnp.allclose(back, pose, atol=1e-6)
