"""Clause/object/reference language conditioning and FASA router.

The T5 encoder itself is a separate parameter subtree (see ``text.py``); this
module owns the trainable alignment, gating, routing and grounding losses.  It
consumes token embeddings, so tests can use a deterministic synthetic encoder
while production runs pass Flan-T5-small encoder states.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import ModelConfig
from .fasa import resize_bilinear


def _normalize(x, axis=-1, eps=1e-8):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=axis, keepdims=True), eps)


def _masked_mean(tokens, mask):
    m = mask.astype(tokens.dtype)
    return (jnp.sum(tokens * m[..., None], axis=1)
            / jnp.maximum(jnp.sum(m, axis=1, keepdims=True), 1e-6))


class TextFeatureAttention(nn.Module):
    hidden_dim: int = 256

    def setup(self):
        self.query = nn.Dense(self.hidden_dim, name="query")
        self.key = nn.Dense(self.hidden_dim, name="key")
        self.value = nn.Dense(self.hidden_dim, name="value")
        # Zero-init output keeps the visual baseline exactly at initialization.
        self.output = nn.Dense(512, name="output",
                               kernel_init=nn.initializers.zeros,
                               bias_init=nn.initializers.zeros)

    def __call__(self, features, tokens, mask):
        # features: [B*N, H, W, 512]; tokens: [B*N, L, D]; mask: [B*N, L]
        b, h, w, c = features.shape
        patches = features.reshape(b, h * w, c)
        q = self.query(patches)
        k = self.key(tokens)
        v = self.value(tokens)
        scores = jnp.einsum("bpd,bld->bpl", q, k) / jnp.sqrt(self.hidden_dim)
        scores = jnp.where(mask[:, None, :].astype(bool), scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bpl,bld->bpd", attn, v)
        return self.output(context).reshape(b, h, w, 512)


class LanguageConditioner(nn.Module):
    config: ModelConfig

    def setup(self):
        cfg = self.config
        self.attention = TextFeatureAttention(cfg.language_hidden_dim,
                                              name="feature_attention")
        self.gate_query = nn.Dense(64, name="gate_query")
        self.gate_key = nn.Dense(64, name="gate_key")
        self.router_dense = nn.Dense(cfg.stage_router_hidden, name="router_dense")
        self.router_out = nn.Dense(cfg.n_clauses, name="router_out")
        self.target_log_scale = self.param(
            "target_log_scale", nn.initializers.constant(jnp.log(10.0)), ())
        self.target_bias = self.param(
            "target_bias", nn.initializers.constant(-3.0), ())
        self.source_null_bias = self.param(
            "source_null_bias", nn.initializers.zeros, ())

    # ------------------------------------------------------------------ keys
    def target_keys(self, tokens, mask):
        pooled = _masked_mean(tokens, mask)
        return _normalize(self.gate_key(pooled))

    def reference_keys(self, tokens, mask):
        return self.target_keys(tokens, mask)

    def target_logits(self, features, keys):
        """``features`` [B*N,512,H,W], ``keys`` [K,64] -> [B*N,K,H,W]."""
        b, c, h, w = features.shape
        patches = features.reshape(b, c, h * w).transpose(0, 2, 1)
        query = _normalize(self.gate_query(patches))
        logits = jnp.einsum("bpd,kd->bkp", query, keys)
        scale = jnp.exp(jnp.clip(self.target_log_scale, None, jnp.log(100.0)))
        return (logits * scale + self.target_bias).reshape(b, keys.shape[0], h, w)

    # ----------------------------------------------------------------- route
    def route(self, stage_feature):
        h = jax.nn.silu(self.router_dense(stage_feature))
        return jax.nn.softmax(self.router_out(h), axis=-1)

    def select_gate(self, logits, route_weights, n_cameras: int):
        """Route-weighted source gate, returns ``[B*N,1,H,W]``."""
        b_n, k, h, w = logits.shape
        b = b_n // n_cameras
        probs = jax.nn.sigmoid(logits).reshape(b, n_cameras, k, h, w)
        if not self.config.generated_text_targets:
            colour_order = ["red", "green", "blue"]
            order = list(self.config.language_order.split())
            if len(order) == 3 and set(order) == set(colour_order):
                indices = jnp.asarray([colour_order.index(c) for c in order])
                probs = jnp.take(probs, indices, axis=2)
        selected = jnp.einsum("bnkhw,bk->bnhw", probs, route_weights)
        return selected.reshape(b * n_cameras, 1, h, w)

    def select_reference_gate(self, logits, route_weights, n_cameras: int):
        b_n, k, h, w = logits.shape
        b = b_n // n_cameras
        probs = jax.nn.sigmoid(logits).reshape(b, n_cameras, k, h, w)
        selected = jnp.einsum("bnkhw,bk->bnhw", probs, route_weights)
        return selected.reshape(b * n_cameras, 1, h, w)

    def source_null_proposal(self, logits):
        flat = logits.reshape(logits.shape[0], logits.shape[1], -1)
        cells = flat.shape[-1]
        log_q = jax.nn.log_softmax(flat, axis=-1)
        null_logit = (jax.scipy.special.logsumexp(flat, axis=-1)
                      - jnp.log(cells) - self.source_null_bias)
        p = jax.nn.sigmoid(null_logit)
        qmax = jnp.max(jnp.exp(log_q), axis=-1, keepdims=True)
        shape = jnp.exp(log_q) / jnp.maximum(qmax, jnp.finfo(flat.dtype).tiny)
        return {
            "p": p,
            "null_logit": null_logit,
            "q": jnp.exp(log_q).reshape(logits.shape),
            "shape": shape.reshape(logits.shape),
            "log_spatial": log_q.reshape(logits.shape),
        }

    def select_source_null_gate(self, p, shape, route_weights, n_cameras: int):
        b = route_weights.shape[0]
        p = p.reshape(b, n_cameras, -1)
        shape = shape.reshape(b, n_cameras, -1, *shape.shape[-2:])
        weights = route_weights[:, None, :]
        selected_p = jnp.sum(p * weights, axis=-1)
        actual = jnp.sum(shape * p[..., None, None] * weights[..., None, None], axis=2)
        fallback = jnp.sum(shape * weights[..., None, None], axis=2)
        positive = selected_p > 0
        selected_shape = jnp.where(
            positive[..., None, None],
            actual / jnp.maximum(selected_p, 1e-8)[..., None, None],
            fallback)
        gate = selected_p[..., None, None] * selected_shape
        return (gate.reshape(b * n_cameras, 1, *shape.shape[-2:]),
                selected_p.reshape(b * n_cameras, 1),
                selected_shape.reshape(b * n_cameras, 1, *shape.shape[-2:]))

    # --------------------------------------------------------------- clauses
    def select_clause_tokens(self, tokens, mask, route_weights):
        """Blend clause token blocks without mixing padded embeddings."""
        b, k = route_weights.shape
        w = route_weights[:, :, None] * mask.astype(tokens.dtype)[None, :, :]
        denominator = jnp.sum(w, axis=1)
        mixed = jnp.sum(tokens[None, :, :, :] * w[..., None], axis=1)
        mixed = mixed / jnp.maximum(denominator[..., None], 1e-8)
        return mixed, denominator > 1e-8

    def feature_update(self, features, clause_tokens, clause_mask, route_weights):
        mixed, mixed_mask = self.select_clause_tokens(
            clause_tokens, clause_mask, route_weights)
        b = route_weights.shape[0]
        n = features.shape[0] // b
        mixed = jnp.repeat(mixed, n, axis=0)
        mixed_mask = jnp.repeat(mixed_mask, n, axis=0)
        out = self.attention(features.transpose(0, 2, 3, 1), mixed, mixed_mask)
        return out.transpose(0, 3, 1, 2)

    # ----------------------------------------------------------------- losses
    def stage_loss(self, route_weights, stage):
        target = jnp.clip(stage.reshape(-1).astype(jnp.int32), 0,
                          route_weights.shape[-1] - 1)
        return -jnp.mean(jnp.log(jnp.clip(
            jnp.take_along_axis(route_weights, target[:, None], axis=1), 1e-8, None)))

    def grounding_loss(self, logits, masks):
        """Colour localization loss aligned to the visual crop.

        ``logits`` [B*N,K,H,W], ``masks`` [B,N,K,Oh,Ow] (red/green/blue).
        The active source-null variant uses a joint spatial+null CE; the active
        run uses the KL + occupancy branch.
        """
        b_n, k, h, w = logits.shape
        n = masks.shape[1]
        b = b_n // n
        target = resize_bilinear(masks.reshape(b * n, k, *masks.shape[-2:]), (h, w))
        mass = jnp.sum(target, axis=(2, 3))
        visible = mass > 1e-6
        if self.config.fasa_source_null:
            proposal = self.source_null_proposal(logits)
            q = target.reshape(b_n, k, h * w)
            q = jnp.where(visible[..., None],
                          q / jnp.maximum(mass[..., None], 1e-8),
                          jnp.zeros_like(q))
            spatial_ce = -jnp.sum(q * proposal["log_spatial"].reshape(b_n, k, -1), axis=-1)
            positive_ce = spatial_ce - jax.nn.log_sigmoid(proposal["null_logit"])
            null_ce = -jax.nn.log_sigmoid(-proposal["null_logit"])
            loss = jnp.mean(jnp.where(visible, positive_ce, null_ce))
            return loss, {"target_joint_ce_loss": loss}
        q = target.reshape(b_n, k, h * w)
        q = q / jnp.maximum(mass[..., None], 1e-8)
        logp = jax.nn.log_sigmoid(logits).reshape(b_n, k, h * w)
        logp = logp - jax.scipy.special.logsumexp(logp, axis=-1, keepdims=True)
        kl = jnp.sum(q * (jnp.log(jnp.clip(q, 1e-8, None)) - logp), axis=-1)
        spatial = jnp.where(visible, kl, 0.0).sum() / jnp.maximum(visible.sum(), 1)
        occupancy = jnp.mean(
            jnp.maximum(target, 0) * -jax.nn.log_sigmoid(logits)
            + (1 - target) * -jax.nn.log_sigmoid(-logits))
        loss = spatial + occupancy
        return loss, {"target_kl_loss": spatial,
                      "target_occupancy_loss": occupancy}


def fasa_route_feature(router_stack, density):
    """Density-weighted description router feature ``[B,516]``."""
    density_sum = jnp.sum(density, axis=1, keepdims=True)
    positive = density_sum > 0
    safe = jnp.where(positive, density_sum, jnp.ones_like(density_sum))
    weights = jnp.where(positive, density / safe,
                        jnp.full_like(density, 1.0 / density.shape[1]))
    return jnp.sum(router_stack * weights[..., None], axis=1)
