"""Absolute 14-D ODE action head: FiLM velocity field + Euler integration."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import ModelConfig
from .pose import normalize_pose, normalize_state, state_to_pose14


class FiLMLayer(nn.Module):
    hidden_dim: int
    cond_dim: int

    def setup(self):
        self.linear = nn.Dense(self.hidden_dim, name="linear")
        self.film_proj = nn.Dense(2 * self.hidden_dim, name="film_proj")

    def __call__(self, h, cond, film=None):
        h = self.linear(h)
        if film is None:
            film = self.film_proj(cond)
        scale, shift = jnp.split(film, 2, axis=-1)
        return jax.nn.silu(scale * h + shift)

    def precompute(self, cond):
        return self.film_proj(cond)


class VITAVectorField(nn.Module):
    cond_dim: int
    action_dim: int = 14
    state_dim: int = 16
    hidden_dim: int = 256
    num_layers: int = 4
    vel_scale: float = 1.0

    def setup(self):
        self.inp = nn.Dense(self.hidden_dim, name="input")
        self.layers = [FiLMLayer(self.hidden_dim, self.cond_dim,
                                 name=f"film_{i}")
                       for i in range(self.num_layers)]
        self.out = nn.Dense(self.action_dim, name="output",
                            kernel_init=nn.initializers.zeros,
                            bias_init=nn.initializers.zeros)

    def __call__(self, a, s, cond, film=None):
        h = self.inp(jnp.concatenate([a, s], axis=-1))
        for i, layer in enumerate(self.layers):
            layer_film = film[:, i] if film is not None else None
            h = layer(h, cond, film=layer_film)
        return self.vel_scale * self.out(h)

    def precompute(self, cond):
        return jnp.stack([layer.precompute(cond) for layer in self.layers], axis=1)


class VITAODEHead(nn.Module):
    config: ModelConfig
    pooled_dim: int

    def setup(self):
        cfg = self.config
        if cfg.ode_spatial_attention:
            raise NotImplementedError(
                "ODE spatial-token cross-attention is disabled in the active run")
        if cfg.ode_relation_readout:
            raise NotImplementedError(
                "ODE relation readout is disabled in the active run")
        cond_in = self.pooled_dim
        if cfg.spatial_action_memory:
            cond_in += 512
        self.cond_in_dim = cond_in
        self.context_proj = nn.Dense(cfg.ode_hidden_dim, name="context_proj")
        self.field = VITAVectorField(
            cond_dim=cfg.ode_hidden_dim,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            hidden_dim=cfg.ode_hidden_dim,
            num_layers=cfg.ode_num_layers,
            name="velocity_field")

    def _process_cond(self, cond):
        cfg = self.config
        if isinstance(cond, Mapping):
            pooled = cond["pooled"]
            patches = cond.get("patches")
            weights = cond.get("weights")
            if cfg.spatial_action_memory:
                if patches is None or weights is None:
                    raise ValueError(
                        "spatial_action_memory requires cond patches and weights")
                b = pooled.shape[0]
                # [B, F*N, 512, H*W] and [B, F*N, 1, H*W]
                c = patches.shape[-3]
                grid = patches.shape[-2] * patches.shape[-1]
                flat_patches = patches.reshape(b, -1, c, grid)
                flat_weights = weights.reshape(b, -1, 1, grid)
                w_sum = jnp.maximum(jnp.sum(flat_weights, axis=-1, keepdims=True), 1e-6)
                w_norm = flat_weights / w_sum
                spatial_ctx = jnp.mean(
                    jnp.sum(flat_patches * w_norm, axis=-1), axis=1)
                cond_feat = jnp.concatenate([pooled, spatial_ctx], axis=-1)
            else:
                cond_feat = pooled
        else:
            cond_feat = cond
        if cond_feat.shape[-1] != self.cond_in_dim:
            raise ValueError(
                f"ODE cond has {cond_feat.shape[-1]} dims, expected {self.cond_in_dim}")
        return self.context_proj(cond_feat)

    def integrate(self, a0, s, cond_feat, n_steps: Optional[int] = None):
        cfg = self.config
        if n_steps is None:
            n_steps = cfg.chunk_size
        film = self.field.precompute(cond_feat)
        a = a0
        path = []
        for _ in range(int(n_steps)):
            if cfg.ode_solver == "euler":
                a = a + self.field(a, s, cond_feat, film=film)
            elif cfg.ode_solver == "rk4":
                k1 = self.field(a, s, cond_feat, film=film)
                k2 = self.field(a + 0.5 * k1, s, cond_feat, film=film)
                k3 = self.field(a + 0.5 * k2, s, cond_feat, film=film)
                k4 = self.field(a + k3, s, cond_feat, film=film)
                a = a + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            else:
                raise ValueError(f"unknown ODE solver {cfg.ode_solver!r}")
            path.append(a)
        return jnp.stack(path, axis=1)

    def __call__(self, cond, obs_state, state_mean, state_std,
                 pose_mean, pose_std, train: bool = False):
        cfg = self.config
        cond_feat = self._process_cond(cond)
        raw_state = obs_state if obs_state.ndim == 2 else obs_state[:, -1, :]
        s = normalize_state(raw_state, state_mean, state_std)
        a0 = normalize_pose(state_to_pose14(raw_state), pose_mean, pose_std)
        if train and cfg.ode_perturb_prob > 0:
            rng = self.make_rng("perturb")
            apply_perturb = jax.random.uniform(rng) < cfg.ode_perturb_prob
            rng, sub = jax.random.split(rng)
            noise = jax.random.normal(sub, a0.shape, dtype=a0.dtype)
            a0 = jnp.where(apply_perturb, a0 + cfg.ode_perturb_std * noise, a0)
        return self.integrate(a0, s, cond_feat)
