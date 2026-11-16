"""Absolute 16-D EE state <-> 14-D xyz/rotvec/grip pose conversions.

The JAX implementation mirrors the PyTorch ODE head's canonicalization: a
quaternion is first flipped to ``w >= 0`` so the rotation vector angle lies in
``[0, pi]``.  This matches scipy's ``Rotation.as_rotvec`` convention used by
the dataset statistics and keeps ``a0``, the ODE target, and inference
unnormalization in one z-score space.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import jax
import jax.numpy as jnp

POSE_PER_ARM = 7
STATE_PER_ARM = 8


def _safe_norm(x, axis=-1, keepdims=True, eps=1e-8):
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims) + eps)


def quat_wxyz_to_rotvec(q: jax.Array) -> jax.Array:
    """``q[..., 4]`` in w,x,y,z -> ``rotvec[..., 3]``.

    The quaternion does not need to be normalized.  ``q`` and ``-q`` map to the
    same canonical rotation vector with angle in ``[0, pi]``.
    """
    q = jnp.asarray(q)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion must have 4 components, got {q.shape[-1]}")
    sign = jnp.where(q[..., 0:1] < 0, -jnp.ones_like(q[..., 0:1]), jnp.ones_like(q[..., 0:1]))
    q = q * sign
    w = q[..., 0]
    v = q[..., 1:4]
    v_norm = _safe_norm(v)
    angle = 2.0 * jnp.arctan2(v_norm, w[..., None])
    # The where branch avoids a division blow-up at the identity.  JAX still
    # evaluates the unused branch; clamp_min keeps its derivative finite.
    scale = jnp.where(v_norm > 1e-8, angle / jnp.maximum(v_norm, 1e-8), jnp.zeros_like(v_norm))
    return scale * v


def rotvec_to_quat_wxyz(r: jax.Array) -> jax.Array:
    """``rotvec[..., 3]`` -> unit quaternion ``[..., 4]`` in w,x,y,z."""
    r = jnp.asarray(r)
    if r.shape[-1] != 3:
        raise ValueError(f"rotvec must have 3 components, got {r.shape[-1]}")
    theta = _safe_norm(r, keepdims=True)
    half = 0.5 * theta
    scale = jnp.where(theta > 1e-8, jnp.sin(half) / jnp.maximum(theta, 1e-8),
                      jnp.full_like(theta, 0.5))
    return jnp.concatenate([jnp.cos(half), scale * r], axis=-1)


def state_to_pose14(state: jax.Array) -> jax.Array:
    """Raw 16-D dual-arm state -> absolute 14-D xyz/rotvec/grip pose.

    Layout: ``[left_xyz(3), left_qwxyz(4), left_grip(1),
              right_xyz(3), right_qwxyz(4), right_grip(1)]``.
    """
    state = jnp.asarray(state)
    if state.shape[-1] != 16:
        raise ValueError(f"state must have 16 components, got {state.shape[-1]}")
    parts = []
    for arm in range(2):
        base = arm * STATE_PER_ARM
        parts.append(state[..., base:base + 3])
        parts.append(quat_wxyz_to_rotvec(state[..., base + 3:base + 7]))
        parts.append(state[..., base + 7:base + 8])
    return jnp.concatenate(parts, axis=-1)


def pose14_to_state16(pose: jax.Array) -> jax.Array:
    """Absolute 14-D pose -> raw 16-D state with unit wxyz quaternions."""
    pose = jnp.asarray(pose)
    if pose.shape[-1] != 14:
        raise ValueError(f"pose must have 14 components, got {pose.shape[-1]}")
    parts = []
    for arm in range(2):
        base = arm * POSE_PER_ARM
        parts.append(pose[..., base:base + 3])
        parts.append(rotvec_to_quat_wxyz(pose[..., base + 3:base + 6]))
        parts.append(pose[..., base + 6:base + 7])
    return jnp.concatenate(parts, axis=-1)


def stats_pair(stats: Mapping[str, jax.Array] | None, dim: int, name: str
               ) -> Tuple[jax.Array, jax.Array]:
    if stats is None:
        return jnp.zeros((dim,), dtype=jnp.float32), jnp.ones((dim,), dtype=jnp.float32)
    mean = jnp.asarray(stats["mean"], dtype=jnp.float32).reshape(-1)
    std = jnp.asarray(stats["std"], dtype=jnp.float32).reshape(-1)
    if mean.shape[0] != dim or std.shape[0] != dim:
        raise ValueError(f"{name} stats must have {dim} dims, got {mean.shape[0]}/{std.shape[0]}")
    return mean, std


def normalize_pose(pose: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return (pose - mean.astype(pose.dtype)) / (std.astype(pose.dtype) + 1e-8)


def unnormalize_pose(pose: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return pose * std.astype(pose.dtype) + mean.astype(pose.dtype)


def normalize_state(state: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return (state - mean.astype(state.dtype)) / (std.astype(state.dtype) + 1e-8)
