"""FASA object-change-v2 source/reference fusion.

This is a JAX translation of the active PyTorch FASA branch:

* language-first per-description change evidence on the head camera builds a
  516-D router feature;
* area-normalised object magnitude density weights the descriptions into one
  stage feature;
* the routed source gate + routed reference gate select the current and previous
  support for every camera;
* dynamic evidence comes from ungated frame differences inside the support,
  static evidence from the current source gate, and destination/reference
  context is kept as a separate static term.

The legacy whole-image change refiner and FASA core/ring branch are not used by
the active run.  ``core_ring=True`` is rejected explicitly.  ``source_null`` is
implemented as source attenuation after the p-free computation (the active run
keeps it disabled), so it is not claimed to be bit-identical to the PyTorch
source-null decomposition.
"""

from __future__ import annotations

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import ModelConfig


def resize_bilinear(x, size):
    """Resize ``[B,C,H,W]`` to ``size=(H,W)`` with bilinear interpolation."""
    b, c, h, w = x.shape
    if (h, w) == tuple(size):
        return x
    y = jax.image.resize(x.transpose(0, 2, 3, 1),
                         (b, int(size[0]), int(size[1]), c), method="linear")
    return y.transpose(0, 3, 1, 2)


def max_pool_nchw(x, kernel: int = 3, stride: int = 1):
    y = x.transpose(0, 2, 3, 1)
    y = nn.max_pool(y, (kernel, kernel), (stride, stride), padding="SAME")
    return y.transpose(0, 3, 1, 2)


def safe_normalize(weights):
    denominator = jnp.sum(weights, axis=(2, 3), keepdims=True)
    positive = denominator > 0
    safe = jnp.where(positive, denominator, jnp.ones_like(denominator))
    return jnp.where(positive, weights / safe, jnp.zeros_like(weights))


def centroid(weights):
    b, _, h, w = weights.shape
    gy = jnp.linspace(0.0, 1.0, h, dtype=weights.dtype)
    gx = jnp.linspace(0.0, 1.0, w, dtype=weights.dtype)
    grid_y, grid_x = jnp.meshgrid(gy, gx, indexing="ij")
    normalised = safe_normalize(weights)
    cx = jnp.sum(normalised * grid_x[None, None], axis=(1, 2, 3))
    cy = jnp.sum(normalised * grid_y[None, None], axis=(1, 2, 3))
    return cx, cy


def _identity_content_init(feat_dim: int):
    def init(key, shape, dtype=jnp.float32):
        kernel = jnp.zeros(shape, dtype=dtype)
        eye = jnp.eye(feat_dim, dtype=dtype)
        return kernel.at[:feat_dim, :feat_dim].set(eye)
    return init


class ChangeRefine(nn.Module):
    in_channels: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = x.transpose(0, 2, 3, 1)
        x = nn.Conv(32, (3, 3), padding=((1, 1), (1, 1)), use_bias=False)(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), padding=((1, 1), (1, 1)), use_bias=False)(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = nn.relu(x)
        return nn.Conv(1, (1, 1))(x).transpose(0, 3, 1, 2)


class ObjectChangeFusion(nn.Module):
    feat_dim: int = 512
    use_activation_prior: bool = True
    mid_feat_dim: int = 256
    build_router: bool = True
    v2: bool = True
    source_null: bool = False

    def setup(self):
        if not self.v2:
            raise NotImplementedError("only FASA object-change v2 is ported")
        self.dynamic_content = nn.Dense(
            self.feat_dim, name="dynamic_content",
            kernel_init=_identity_content_init(self.feat_dim))
        self.reference_context = nn.Dense(self.feat_dim, name="reference_context")
        refine_in = 5 + (2 if self.use_activation_prior else 0) + (
            2 if self.mid_feat_dim > 0 else 0)
        self.change_refine = ChangeRefine(refine_in, name="change_refine")
        self.router_dense = None
        self.router_norm = None
        if self.build_router:
            self.router_dense = nn.Dense(
                self.feat_dim, name="router_dense",
                kernel_init=nn.initializers.lecun_normal())
            self.router_norm = nn.LayerNorm(name="router_norm")

    def __call__(self, feat_curr, feat_prev, fused, gate_curr, gate_prev,
                 reference_gate=None, prev_reference_gate=None,
                 mid_feat_curr=None, cosine_change: bool = True,
                 source_p=None, prev_source_p=None, source_shape=None,
                 prev_source_shape=None, train: bool = True):
        h, w = feat_curr.shape[-2:]
        size = (h, w)
        if self.source_null:
            if source_p is None or prev_source_p is None:
                raise ValueError("source_null requires explicit source p values")
            p_curr = source_p.reshape(-1, 1)
            p_prev = prev_source_p.reshape(-1, 1)
            raw_curr = source_shape if source_shape is not None else gate_curr
            raw_prev = prev_source_shape if prev_source_shape is not None else gate_prev
            raw_curr = resize_bilinear(raw_curr, size)
            raw_prev = resize_bilinear(raw_prev, size)
            p_union = jnp.maximum(p_curr, p_prev)
        else:
            p_curr = p_prev = p_union = None
            raw_curr = resize_bilinear(gate_curr, size)
            raw_prev = (resize_bilinear(gate_prev, size)
                        if gate_prev is not None else raw_curr)
        gate_curr, gate_prev = raw_curr, raw_prev
        ref_curr = (resize_bilinear(reference_gate, size)
                    if reference_gate is not None else None)
        ref_prev = (resize_bilinear(prev_reference_gate, size)
                    if prev_reference_gate is not None else None)

        support = jnp.maximum(gate_curr, gate_prev)
        support = max_pool_nchw(support, 3, 1)
        support = jnp.clip(support, 0.0, 1.0)
        confidence = jnp.max(gate_curr, axis=(2, 3), keepdims=True)
        support_coverage = jnp.mean(support, axis=(2, 3), keepdims=True)

        if cosine_change:
            dot = jnp.sum(feat_curr * feat_prev, axis=1, keepdims=True)
            norm = (jnp.sqrt(jnp.sum(feat_curr ** 2, axis=1, keepdims=True) + 1e-12)
                    * jnp.sqrt(jnp.sum(feat_prev ** 2, axis=1, keepdims=True) + 1e-12))
            raw_mean = jnp.clip(1.0 - dot / norm, 0.0, None)
            raw_max = raw_mean
            raw_std = jnp.zeros_like(raw_mean)
        else:
            delta = jnp.abs(feat_curr - feat_prev)
            raw_mean = jnp.mean(delta, axis=1, keepdims=True)
            raw_max = jnp.max(delta, axis=1, keepdims=True)
            raw_std = jnp.std(delta, axis=1, keepdims=True)

        identical = jnp.all(feat_curr == feat_prev, axis=1, keepdims=True)
        moving = (~identical).astype(raw_mean.dtype)
        raw_mean = raw_mean * moving
        raw_max = raw_max * moving
        raw_std = raw_std * moving

        support_sum = jnp.sum(support, axis=(2, 3))
        support_positive = support_sum > 0
        safe_support = jnp.where(support_positive, support_sum,
                                 jnp.ones_like(support_sum))
        density_numerator = jnp.sum(raw_mean * support, axis=(2, 3))
        magnitude_density = jnp.where(
            support_positive, density_numerator / safe_support,
            jnp.zeros_like(support_sum))
        dynamic_amplitude = magnitude_density / (1.0 + magnitude_density)
        router_magnitude = jnp.log1p(magnitude_density)

        gated_mean = raw_mean * support
        refine_inputs = [gated_mean, raw_max * support, raw_std * support, support,
                         jnp.broadcast_to(confidence, support.shape)]
        if self.use_activation_prior:
            activation = jnp.abs(feat_curr)
            refine_inputs.append(jnp.mean(activation, axis=1, keepdims=True) * support)
            refine_inputs.append(jnp.max(activation, axis=1, keepdims=True) * support)
        if self.mid_feat_dim > 0:
            if mid_feat_curr is None:
                refine_inputs.extend([jnp.zeros_like(support), jnp.zeros_like(support)])
            else:
                mid = resize_bilinear(mid_feat_curr, size)
                refine_inputs.append(jnp.mean(jnp.abs(mid), axis=1, keepdims=True) * support)
                refine_inputs.append(jnp.max(jnp.abs(mid), axis=1, keepdims=True) * support)
        refined = jax.nn.sigmoid(
            self.change_refine(jnp.concatenate(refine_inputs, axis=1), train=train))
        attention = safe_normalize(refined * gated_mean)
        content = jnp.sum(fused * attention, axis=(2, 3))

        presence = (magnitude_density > 0).reshape(-1).astype(content.dtype)
        dynamic = self.dynamic_content(jnp.concatenate(
            [content, router_magnitude.reshape(-1, 1)], axis=1)) * presence[:, None]

        static_weights = safe_normalize(gate_curr)
        source_static = jnp.sum(fused * static_weights, axis=(2, 3))
        source_cx, source_cy = centroid(static_weights)

        zero_gate = jnp.zeros_like(gate_curr)
        rw = safe_normalize(ref_curr if ref_curr is not None else zero_gate)
        pw = safe_normalize(ref_prev if ref_prev is not None else zero_gate)
        ref_static = jnp.sum(feat_curr * rw, axis=(2, 3))
        ref_prev = jnp.sum(feat_prev * pw, axis=(2, 3))
        rx, ry = centroid(rw)
        px, py = centroid(pw)
        ref_present = (jnp.sum(rw, axis=(2, 3)) + jnp.sum(pw, axis=(2, 3)) > 0)
        ref_present = ref_present.reshape(-1).astype(source_static.dtype)
        ref_context = self.reference_context(jnp.concatenate(
            [ref_static, ref_prev, rx[:, None], ry[:, None],
             px[:, None], py[:, None]], axis=1)) * ref_present[:, None]
        static = source_static + ref_context

        current_motion = raw_mean * gate_curr
        previous_motion = raw_mean * gate_prev
        curr_cx, curr_cy = centroid(current_motion)
        prev_cx, prev_cy = centroid(previous_motion)
        coords = jnp.concatenate([
            source_cx[:, None], source_cy[:, None],
            (curr_cx - prev_cx)[:, None], (curr_cy - prev_cy)[:, None]], axis=1)

        router = None
        if self.build_router:
            router_context = jnp.concatenate([
                dynamic, static, confidence.reshape(confidence.shape[0], -1),
                support_coverage.reshape(support_coverage.shape[0], -1),
                router_magnitude.reshape(-1, 1)], axis=1)
            router_core = self.router_norm(self.router_dense(router_context))
            router = jnp.concatenate([router_core, coords], axis=1)

        if self.source_null:
            dynamic = dynamic * p_union[:, None]
            content = content * p_union
            static = source_static * p_curr + ref_context
            coords = jnp.concatenate([
                (source_cx * p_curr)[:, None], (source_cy * p_curr)[:, None],
                (curr_cx * p_curr - prev_cx * p_prev)[:, None],
                (curr_cy * p_curr - prev_cy * p_prev)[:, None]], axis=1)
            if router is not None:
                router = jnp.concatenate([router[:, :self.feat_dim], coords], axis=1)
            confidence = confidence * p_curr[:, None]
            magnitude_density = magnitude_density * p_union
            dynamic_amplitude = dynamic_amplitude * p_union

        return {
            "dynamic": dynamic,
            "reference_static": ref_static,
            "motion_content": content,
            "static": static,
            "coords": coords,
            "router": router,
            "attention": attention,
            "static_weights": static_weights,
            "support": support,
            "magnitude_density": magnitude_density,
            "dynamic_amplitude": dynamic_amplitude,
            "confidence": confidence,
            "support_coverage": support_coverage,
            "identical_frames": identical,
        }


class FASAObjectChange(nn.Module):
    config: ModelConfig

    def setup(self):
        cfg = self.config
        if cfg.fasa_core_ring:
            raise NotImplementedError(
                "FASA core/ring is not part of the active run and is not ported")
        self.mid_conv = nn.Conv(512, (1, 1), use_bias=False, name="mid_conv")
        self.mid_bn = nn.BatchNorm(name="mid_bn")
        self.fusion_selected = ObjectChangeFusion(
            feat_dim=512,
            use_activation_prior=cfg.use_activation_prior,
            mid_feat_dim=256 if cfg.use_multiscale else 0,
            build_router=False,
            v2=True,
            source_null=cfg.fasa_source_null,
            name="selected")
        self.fusion_router = ObjectChangeFusion(
            feat_dim=512,
            use_activation_prior=cfg.use_activation_prior,
            mid_feat_dim=256 if cfg.use_multiscale else 0,
            build_router=True,
            v2=True,
            source_null=cfg.fasa_source_null,
            name="router")
        self.coord_proj = nn.Dense(cfg.fasa_coord_dim, name="coord_proj")

    def __call__(self, feat_curr, feat_prev, mid_curr=None,
                 gate_curr=None, gate_prev=None, reference_gate=None,
                 prev_reference_gate=None, feature_modulation=None,
                 source_p=None, prev_source_p=None, source_shape=None,
                 prev_source_shape=None, train: bool = True):
        """Return pooled per-camera FASA features and diagnostics.

        Shapes: features ``[B,N,512,H,W]``, mid features ``[B,N,256,H,W]``,
        gates ``[B,N,1,H,W]``.  The returned pooled tensor is ``[B,N,1060]``.
        """
        cfg = self.config
        b, n, c, h, w = feat_curr.shape
        fused = feat_curr
        if cfg.use_multiscale:
            if mid_curr is None:
                raise ValueError("use_multiscale requires layer3 features")
            mid = mid_curr.reshape(b * n, *mid_curr.shape[2:]).transpose(0, 2, 3, 1)
            mid = self.mid_conv(mid)
            mid = self.mid_bn(mid, use_running_average=not train)
            mid = nn.relu(mid).transpose(0, 3, 1, 2)
            mid = resize_bilinear(mid, (h, w)).reshape(b, n, c, h, w)
            fused = fused + mid
        if feature_modulation is not None:
            fused = fused + feature_modulation
        if gate_curr is None:
            raise ValueError("object-change FASA requires the routed source gate")

        outputs = []
        evidence_last = None
        for cam in range(n):
            ev = self.fusion_selected(
                feat_curr[:, cam], feat_prev[:, cam], fused[:, cam],
                gate_curr[:, cam], gate_prev[:, cam],
                reference_gate=(reference_gate[:, cam]
                                if reference_gate is not None else None),
                prev_reference_gate=(prev_reference_gate[:, cam]
                                     if prev_reference_gate is not None else None),
                mid_feat_curr=mid_curr[:, cam] if mid_curr is not None else None,
                cosine_change=cfg.fasa_cosine_change,
                source_p=source_p[:, cam] if source_p is not None else None,
                prev_source_p=(prev_source_p[:, cam]
                               if prev_source_p is not None else None),
                source_shape=(source_shape[:, cam]
                              if source_shape is not None else None),
                prev_source_shape=(prev_source_shape[:, cam]
                                   if prev_source_shape is not None else None),
                train=train)
            coords = self._spatial_softmax(fused[:, cam], prior=ev["static_weights"])
            gate_present = (jnp.sum(gate_curr[:, cam], axis=(1, 2, 3)) > 0)
            gate_present = gate_present.reshape(b, 1).astype(coords.dtype)
            if cfg.fasa_source_null and source_p is not None:
                coords = coords * source_p[:, cam].reshape(b, 1)
            coords = coords * gate_present
            outputs.append(jnp.concatenate(
                [ev["dynamic"], ev["static"], ev["coords"], coords], axis=1))
            evidence_last = ev
        return jnp.stack(outputs, axis=1), evidence_last

    def _spatial_softmax(self, feat_map, prior=None):
        b, c, h, w = feat_map.shape
        logits = feat_map.reshape(b, c, h * w)
        if prior is not None:
            p = resize_bilinear(prior, (h, w)).reshape(b, 1, h * w)
            logits = logits + jnp.log(jnp.clip(p, 1e-8, None))
        weights = jax.nn.softmax(logits, axis=2)
        gy = jnp.linspace(0.0, 1.0, h, dtype=feat_map.dtype)
        gx = jnp.linspace(0.0, 1.0, w, dtype=feat_map.dtype)
        grid_y, grid_x = jnp.meshgrid(gy, gx, indexing="ij")
        exp_x = jnp.sum(weights * grid_x.reshape(-1), axis=2)
        exp_y = jnp.sum(weights * grid_y.reshape(-1), axis=2)
        return self.coord_proj(jnp.concatenate([exp_x, exp_y], axis=1))

    def router_evidence(self, head_curr, head_prev, mid_curr, gates_curr,
                        gates_prev, refs_curr=None, refs_prev=None,
                        source_p=None, prev_source_p=None,
                        source_shape=None, prev_source_shape=None,
                        train: bool = True):
        """Per-description router features on the head camera, ``[B,K,516]``."""
        routers = []
        densities = []
        k = gates_curr.shape[1]
        for index in range(k):
            ev = self.fusion_router(
                head_curr, head_prev, head_curr,
                gates_curr[:, index], gates_prev[:, index],
                reference_gate=(refs_curr[:, index]
                                if refs_curr is not None else None),
                prev_reference_gate=(refs_prev[:, index]
                                     if refs_prev is not None else None),
                mid_feat_curr=mid_curr,
                cosine_change=self.config.fasa_cosine_change,
                source_p=source_p[:, index] if source_p is not None else None,
                prev_source_p=(prev_source_p[:, index]
                               if prev_source_p is not None else None),
                source_shape=(source_shape[:, index][:, None]
                              if source_shape is not None else None),
                prev_source_shape=(prev_source_shape[:, index][:, None]
                                   if prev_source_shape is not None else None),
                train=train)
            routers.append(ev["router"])
            densities.append(ev["magnitude_density"].reshape(-1))
        return jnp.stack(routers, axis=1), jnp.stack(densities, axis=1)
