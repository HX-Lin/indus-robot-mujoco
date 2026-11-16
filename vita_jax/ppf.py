"""Predicted-action-conditioned PPF future-visual auxiliary."""

from __future__ import annotations

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class PPFFutureVisualHead(nn.Module):
    action_cond_dim: int
    visual_dim: int
    hidden_dim: int = 512

    def setup(self):
        self.inp = nn.Dense(self.hidden_dim, name="input")
        self.output = nn.Dense(self.visual_dim, name="output")

    def __call__(self, action_cond, current_visual):
        x = jnp.concatenate([action_cond, current_visual], axis=-1)
        return self.output(jax.nn.silu(self.inp(x)))


def detached_delta(current_visual, future_visual):
    """Detached pooled ``z(t+4) - z(t)`` target."""
    return (jax.lax.stop_gradient(future_visual)
            - jax.lax.stop_gradient(current_visual))


def ppf_future_visual_loss(head, pred_path, current_visual, future_visual,
                           valid, horizon: int = 4) -> Tuple[jax.Array, dict]:
    """Training-only MSE with a per-sample future-horizon mask.

    ``pred_path`` is the differentiable normalized absolute ODE path; its first
    four poses are the action condition.  The head also receives the attached
    current visual feature, while the future delta target is detached.
    """
    if pred_path.ndim != 3 or pred_path.shape[1] < horizon:
        raise ValueError(
            f"PPF condition must be [B,T>={horizon},action_dim], got {pred_path.shape}")
    b = pred_path.shape[0]
    action_cond = pred_path[:, :horizon].reshape(b, -1)
    target = detached_delta(current_visual, future_visual)
    predicted = head(action_cond, current_visual)
    valid = jnp.asarray(valid).reshape(-1).astype(jnp.float32)
    if valid.shape[0] != b:
        raise ValueError("future validity must have one value per sample")
    per_sample = jnp.mean(jnp.square(predicted - target), axis=1)
    loss = jnp.sum(per_sample * valid) / jnp.maximum(jnp.sum(valid), 1.0)
    return loss, {
        "ppf_future_visual_loss": loss,
        "ppf_future_visual_valid_fraction": jnp.mean(valid),
    }
