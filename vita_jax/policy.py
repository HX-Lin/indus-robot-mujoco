"""Full JAX/Flax VITA absolute-ODE + predicted-action PPF policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import flax.core
import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import ModelConfig, VITAConfig
from .fasa import FASAObjectChange, max_pool_nchw, resize_bilinear
from .language import LanguageConditioner, fasa_route_feature
from .ode import VITAODEHead
from .pose import normalize_pose, normalize_state, state_to_pose14
from .ppf import PPFFutureVisualHead, ppf_future_visual_loss
from .text import TextBank, TextEncoder, build_text_encoder
from .vision import ResNet18Features, load_resnet18_weights


def _resize_images(x, size):
    lead = x.shape[:-3]
    flat = x.reshape((-1,) + x.shape[-3:])
    resized = resize_bilinear(flat, size)
    return resized.reshape(lead + resized.shape[-3:])


def _crop_images(x, top, left, crop_shape):
    # x [B,C,H,W] -> dynamic crop with static output size
    x = x.transpose(0, 2, 3, 1)
    x = jax.lax.dynamic_slice(x, (0, top, left, 0),
                              (x.shape[0], crop_shape[0], crop_shape[1], x.shape[-1]))
    return x.transpose(0, 3, 1, 2)



def assemble_objective(mse, l1, grounding_loss, route_loss, ppf_loss,
                       ppf_weight: float, train: bool,
                       flow_recon_weight: float = 0.5,
                       grounding_weight: float = 0.5,
                       stage_route_weight: float = 0.5):
    """Active-run objective: base pose/grounding/route, PPF train-only."""
    base = (mse + flow_recon_weight * l1
            + grounding_weight * grounding_loss
            + stage_route_weight * route_loss)
    if train and ppf_loss is not None:
        total = base + ppf_weight * ppf_loss
    else:
        total = base
    return total, base


class VITALinen(nn.Module):
    config: ModelConfig
    obs_dim: int
    text_dim: int

    def setup(self):
        cfg = self.config
        n = len(cfg.image_keys)
        self.vision = ResNet18Features(name="vision")
        self.fasa = FASAObjectChange(cfg, name="fasa")
        self.language = LanguageConditioner(cfg, name="language")
        self.ode = VITAODEHead(cfg, pooled_dim=self.obs_dim + self.text_dim,
                               name="ode")
        if cfg.ppf_future_visual_weight > 0:
            self.ppf = PPFFutureVisualHead(
                action_cond_dim=cfg.ppf_future_visual_horizon * cfg.action_dim,
                visual_dim=n * 512,
                hidden_dim=cfg.ppf_future_visual_hidden,
                name="ppf_future_visual")
        if cfg.fasa_cross_camera:
            self.cross_camera = nn.MultiHeadDotProductAttention(
                num_heads=4, qkv_features=cfg.fasa_per_cam_dim,
                out_features=cfg.fasa_per_cam_dim, name="cross_camera")
            self.cross_norm = nn.LayerNorm(name="cross_norm")
            self.camera_role = self.param(
                "camera_role", nn.initializers.zeros,
                (n, cfg.fasa_per_cam_dim))
        if cfg.fasa_camera_scale:
            self.camera_scale = self.param(
                "camera_scale", nn.initializers.ones, (n, 1))

    # ------------------------------------------------------------- visual
    def _augment(self, images, rng):
        """Photometric jitter shared across each camera's history."""
        b, s, n, c, h, w = images.shape
        keys = jax.random.split(rng, 4)
        shape = (b, 1, n, 1, 1, 1)
        bright = 0.45 + 1.15 * jax.random.uniform(keys[0], shape,
                                                  minval=0.0, maxval=1.0)
        contrast = 0.7 + 0.7 * jax.random.uniform(keys[1], shape,
                                                  minval=0.0, maxval=1.0)
        gain = 0.75 + 0.5 * jax.random.uniform(keys[2], (b, 1, n, c, 1, 1),
                                               minval=0.0, maxval=1.0)
        gamma = 0.6 + 0.8 * jax.random.uniform(keys[3], shape,
                                               minval=0.0, maxval=1.0)
        mean = jnp.mean(images, axis=(3, 4, 5), keepdims=True)
        images = ((images - mean) * contrast + mean) * bright
        images = jnp.power(jnp.clip(images * gain, 1e-4, 1.0), gamma)
        return images

    def _preprocess(self, images, train: bool):
        cfg = self.config
        b, s, n, c, h, w = images.shape
        images = _resize_images(images, cfg.resize_shape)
        h, w = images.shape[-2:]
        ch, cw = cfg.crop_shape
        if train:
            rng = self.make_rng("crop")
            top = jax.random.randint(rng, (), 0, h - ch + 1)
            left = jax.random.randint(rng, (), 0, w - cw + 1)
        else:
            top = (h - ch) // 2
            left = (w - cw) // 2
        images = _crop_images(images.reshape(b * s * n, c, h, w), top, left,
                              (ch, cw)).reshape(b, s, n, c, ch, cw)
        if train and cfg.photometric_aug:
            images = self._augment(images, self.make_rng("aug"))
        if cfg.input_norm:
            mu = jnp.mean(images, axis=(4, 5), keepdims=True)
            sd = jnp.std(images, axis=(4, 5), keepdims=True)
            images = (images - mu) / (sd + 1e-5) * 0.5 + 0.5
        return images, (top, left, h, w, ch, cw)

    def _deterministic_pooled(self, images):
        cfg = self.config
        b, n, c, h, w = images.shape
        images = _resize_images(images, cfg.resize_shape)
        h, w = images.shape[-2:]
        ch, cw = cfg.crop_shape
        top, left = (h - ch) // 2, (w - cw) // 2
        images = _crop_images(images.reshape(b * n, c, h, w), top, left, (ch, cw))
        if cfg.input_norm:
            mu = jnp.mean(images, axis=(2, 3), keepdims=True)
            sd = jnp.std(images, axis=(2, 3), keepdims=True)
            images = (images - mu) / (sd + 1e-5) * 0.5 + 0.5
        _, feat = self.vision(images)
        return jnp.mean(feat, axis=(2, 3)).reshape(b, n * 512)

    def _pool_raw(self, feat_maps, gate):
        if gate is None:
            return jnp.mean(feat_maps, axis=(2, 3)).reshape(feat_maps.shape[0], -1)
        gate = resize_bilinear(gate, feat_maps.shape[-2:])
        weights = gate / jnp.maximum(jnp.sum(gate, axis=(2, 3), keepdims=True), 1e-8)
        pooled = jnp.sum(feat_maps * weights, axis=(2, 3))
        scale = jnp.max(gate.reshape(gate.shape[0], -1), axis=1, keepdims=True)
        return (pooled * scale).reshape(feat_maps.shape[0], -1)

    def _crop_grounding(self, masks, crop_info):
        top, left, h, w, ch, cw = crop_info
        b, n, k = masks.shape[:3]
        masks = resize_bilinear(masks.reshape(b * n, k, *masks.shape[-2:]), (h, w))
        masks = _crop_images(masks, top, left, (ch, cw))
        return masks.reshape(b, n, k, ch, cw)

    @staticmethod
    def _reorder_router_probs(probs, cfg):
        if cfg.generated_text_targets:
            return probs
        colour_order = ["red", "green", "blue"]
        order = list(cfg.language_order.split())
        if len(order) == 3 and set(order) == set(colour_order):
            indices = jnp.asarray([colour_order.index(c) for c in order])
            return jnp.take(probs, indices, axis=2)
        return probs

    def __call__(self, batch, text, train: bool = False):
        cfg = self.config
        images = batch["images"]
        b, s, n, c, h, w = images.shape
        if s < 2:
            raise ValueError("FASA diff observations require two frames")
        images, crop_info = self._preprocess(images, train=train)
        ch, cw = images.shape[-2:]

        state_raw = batch["observation.state"]
        state_mean = batch["state_mean"]
        state_std = batch["state_std"]
        if train and cfg.state_dropout > 0:
            # PyTorch default is raw_zero: zero the measured state before
            # normalization, while the ODE start still uses the raw state.
            keep = (jax.random.uniform(self.make_rng("dropout"),
                                       (b, 1, 1)) >= cfg.state_dropout)
            state_obs = normalize_state(state_raw * keep, state_mean, state_std)
        else:
            state_obs = normalize_state(state_raw, state_mean, state_std)
        state_t = state_obs[:, -1]
        state_delta = state_t - state_obs[:, -2]
        if train and cfg.state_delta_dropout > 0:
            dkeep = (jax.random.uniform(self.make_rng("state_delta"),
                                        (b, 1)) >= cfg.state_delta_dropout)
            state_delta = state_delta * dkeep

        flat = images.reshape(b * s * n, c, ch, cw)
        mid_maps, feat_maps = self.vision(flat)
        feat_maps = feat_maps.reshape(b, s, n, 512, *feat_maps.shape[-2:])
        mid_maps = mid_maps.reshape(b, s, n, 256, *mid_maps.shape[-2:])
        curr = feat_maps[:, -1]
        prev = feat_maps[:, -2]
        mid_curr = mid_maps[:, -1]

        # Language first: encode target/reference keys, then localize.
        object_tokens = text["objects"]["tokens"]
        object_mask = text["objects"]["mask"]
        ref_tokens = text["references"]["tokens"]
        ref_mask = text["references"]["mask"]
        clause_tokens = text["clauses"]["tokens"]
        clause_mask = text["clauses"]["mask"]
        keys = self.language.target_keys(object_tokens, object_mask)
        ref_keys = self.language.reference_keys(ref_tokens, ref_mask)
        logits_curr = self.language.target_logits(
            curr.reshape(b * n, 512, *curr.shape[-2:]), keys)
        logits_prev = self.language.target_logits(
            prev.reshape(b * n, 512, *prev.shape[-2:]), keys)
        ref_logits_curr = self.language.target_logits(
            curr.reshape(b * n, 512, *curr.shape[-2:]), ref_keys)
        ref_logits_prev = self.language.target_logits(
            prev.reshape(b * n, 512, *prev.shape[-2:]), ref_keys)
        probs_curr = jax.nn.sigmoid(logits_curr).reshape(
            b, n, keys.shape[0], *curr.shape[-2:])
        probs_prev = jax.nn.sigmoid(logits_prev).reshape(
            b, n, keys.shape[0], *prev.shape[-2:])
        ref_probs_curr = jax.nn.sigmoid(ref_logits_curr).reshape(
            b, n, ref_keys.shape[0], *curr.shape[-2:])
        ref_probs_prev = jax.nn.sigmoid(ref_logits_prev).reshape(
            b, n, ref_keys.shape[0], *prev.shape[-2:])

        source_curr = source_prev = None
        if cfg.fasa_source_null:
            source_curr = self.language.source_null_proposal(logits_curr)
            source_prev = self.language.source_null_proposal(logits_prev)

        router_probs_curr = self._reorder_router_probs(probs_curr, cfg)
        router_probs_prev = self._reorder_router_probs(probs_prev, cfg)
        router_refs_curr = self._reorder_router_probs(ref_probs_curr, cfg)
        router_refs_prev = self._reorder_router_probs(ref_probs_prev, cfg)
        source_p_head = source_shape_head = None
        source_p_prev_head = source_shape_prev_head = None
        if cfg.fasa_source_null:
            source_p_head = source_curr["p"].reshape(b, n, -1)[:, 0]
            source_p_prev_head = source_prev["p"].reshape(b, n, -1)[:, 0]
            source_shape_head = source_curr["shape"].reshape(
                b, n, -1, *source_curr["shape"].shape[-2:])[:, 0]
            source_shape_prev_head = source_prev["shape"].reshape(
                b, n, -1, *source_prev["shape"].shape[-2:])[:, 0]
        router_stack, density = self.fasa.router_evidence(
            curr[:, 0], prev[:, 0], mid_curr[:, 0],
            router_probs_curr[:, 0][:, :, None],
            router_probs_prev[:, 0][:, :, None],
            refs_curr=router_refs_curr[:, 0][:, :, None],
            refs_prev=router_refs_prev[:, 0][:, :, None],
            source_p=source_p_head, prev_source_p=source_p_prev_head,
            source_shape=source_shape_head,
            prev_source_shape=source_shape_prev_head,
            train=train)
        stage_feat = fasa_route_feature(router_stack, density)
        route = self.language.route(stage_feat)

        selected_curr = self.language.select_gate(logits_curr, route, n)
        selected_prev = self.language.select_gate(logits_prev, route, n)
        selected_ref = self.language.select_reference_gate(
            ref_logits_curr, route, n)
        selected_ref_prev = self.language.select_reference_gate(
            ref_logits_prev, route, n)
        selected_p = selected_prev_p = selected_shape = selected_prev_shape = None
        if cfg.fasa_source_null:
            selected_curr, selected_p, selected_shape = (
                self.language.select_source_null_gate(
                    source_curr["p"], source_curr["shape"], route, n))
            selected_prev, selected_prev_p, selected_prev_shape = (
                self.language.select_source_null_gate(
                    source_prev["p"], source_prev["shape"], route, n))

        feature_mod = self.language.feature_update(
            curr.reshape(b * n, 512, *curr.shape[-2:]), clause_tokens,
            clause_mask, route).reshape(b, n, 512, *curr.shape[-2:])

        pooled, evidence = self.fasa(
            curr, prev, mid_curr,
            gate_curr=selected_curr.reshape(b, n, 1, *curr.shape[-2:]),
            gate_prev=selected_prev.reshape(b, n, 1, *curr.shape[-2:]),
            reference_gate=selected_ref.reshape(b, n, 1, *curr.shape[-2:]),
            prev_reference_gate=selected_ref_prev.reshape(
                b, n, 1, *curr.shape[-2:]),
            feature_modulation=feature_mod,
            source_p=(selected_p.reshape(b, n, 1)
                      if selected_p is not None else None),
            prev_source_p=(selected_prev_p.reshape(b, n, 1)
                           if selected_prev_p is not None else None),
            source_shape=(selected_shape.reshape(b, n, 1, *curr.shape[-2:])
                          if selected_shape is not None else None),
            prev_source_shape=(selected_prev_shape.reshape(
                b, n, 1, *curr.shape[-2:]) if selected_prev_shape is not None else None),
            train=train)

        if cfg.fasa_cross_camera:
            tokens = pooled + self.camera_role[None]
            attn = self.cross_camera(tokens, tokens)
            tokens = self.cross_norm(tokens + attn)
        else:
            tokens = pooled
        if cfg.fasa_camera_scale:
            tokens = tokens * self.camera_scale[None]
        f_t = tokens.reshape(b, -1)

        gate_curr_cam = selected_curr.reshape(b, n, 1, *curr.shape[-2:])
        gate_prev_cam = selected_prev.reshape(b, n, 1, *curr.shape[-2:])
        raw_t = self._pool_raw(curr.reshape(b * n, 512, *curr.shape[-2:]),
                               selected_curr).reshape(b, -1)
        raw_prev = self._pool_raw(prev.reshape(b * n, 512, *prev.shape[-2:]),
                                  selected_prev).reshape(b, -1)
        obs_features = jnp.concatenate(
            [f_t, raw_t - raw_prev, state_t, state_delta], axis=-1)
        if obs_features.shape[-1] != self.obs_dim:
            raise ValueError(
                f"observation dim {obs_features.shape[-1]} != configured {self.obs_dim}")

        language_vec, language_mask = self.language.select_clause_tokens(
            clause_tokens, clause_mask, route)
        language_vec = (
            jnp.sum(language_vec * language_mask[..., None], axis=1)
            / jnp.maximum(jnp.sum(language_mask, axis=1, keepdims=True), 1))
        pooled_cond = jnp.concatenate([obs_features, language_vec], axis=-1)

        # Two-frame target-neighbourhood memory.  The active ODE head consumes
        # the weighted global mean (spatial attention is disabled).
        patches = jnp.stack([prev, curr], axis=1)
        curr_neigh = max_pool_nchw(
            gate_curr_cam.reshape(b * n, 1, *gate_curr_cam.shape[-2:]), 3, 1
        ).reshape(b, n, 1, *gate_curr_cam.shape[-2:])
        prev_neigh = max_pool_nchw(
            gate_prev_cam.reshape(b * n, 1, *gate_prev_cam.shape[-2:]), 3, 1
        ).reshape(b, n, 1, *gate_prev_cam.shape[-2:])
        weights = jnp.stack([prev_neigh, curr_neigh], axis=1)
        cond = {"pooled": pooled_cond, "patches": patches, "weights": weights}

        pred_path = self.ode(
            cond, state_raw,
            state_mean=state_mean, state_std=state_std,
            pose_mean=batch["pose_mean"], pose_std=batch["pose_std"],
            train=train)

        gt_chunk = batch["action_ee_chunk"]
        if gt_chunk.ndim == 2:
            gt_chunk = gt_chunk.reshape(b, cfg.chunk_size, 16)
        gt_pose = normalize_pose(state_to_pose14(gt_chunk),
                                 batch["pose_mean"], batch["pose_std"])
        gt_pose = gt_pose[:, :pred_path.shape[1]]
        mse = jnp.mean(jnp.square(pred_path - gt_pose))
        l1 = jnp.mean(jnp.abs(pred_path - gt_pose))
        metrics = {"flow_loss": mse, "pose_loss": mse,
                   "flow_recon_loss": l1}

        ground_loss = jnp.zeros((), dtype=mse.dtype)
        if cfg.target_grounding:
            if batch.get("grounding_masks") is None:
                raise ValueError("target grounding requires colour sidecars")
            masks = self._crop_grounding(batch["grounding_masks"], crop_info)
            ground_loss, ground_metrics = self.language.grounding_loss(
                logits_curr, masks)
            metrics.update(ground_metrics)
            metrics["target_grounding_loss"] = ground_loss
        route_loss = jnp.zeros((), dtype=mse.dtype)
        if cfg.stage_route_weight > 0:
            if batch.get("stage") is None:
                raise ValueError("stage routing requires action-aligned stage labels")
            route_loss = self.language.stage_loss(route, batch["stage"])
            metrics["stage_route_loss"] = route_loss

        ppf_loss = None
        if cfg.ppf_future_visual_weight > 0:
            if batch.get("future_images") is None or batch.get("future_valid") is None:
                raise ValueError("PPF future-visual loss requires future RGB and validity")
            current_images = images[:, -1]
            current_visual = self._deterministic_pooled(current_images)
            future_visual = self._deterministic_pooled(batch["future_images"])
            ppf_loss, ppf_metrics = ppf_future_visual_loss(
                self.ppf, pred_path, current_visual, future_visual,
                batch["future_valid"], horizon=cfg.ppf_future_visual_horizon)
            metrics.update(ppf_metrics)
            metrics["ppf_future_visual_weighted_loss"] = (
                cfg.ppf_future_visual_weight * ppf_loss)
        total, base = assemble_objective(
            mse, l1, ground_loss, route_loss, ppf_loss,
            cfg.ppf_future_visual_weight, train,
            flow_recon_weight=cfg.flow_recon_weight,
            grounding_weight=cfg.target_grounding_weight,
            stage_route_weight=cfg.stage_route_weight)
        metrics["base_loss"] = base
        metrics["total_loss"] = total
        return {
            "loss": total,
            "metrics": metrics,
            "path": pred_path,
            "route": route,
            "stage_feature": stage_feat,
            "ppf_loss": ppf_loss,
            "grounding_logits": logits_curr,
        }


@dataclass
class VITAPolicy:
    config: VITAConfig
    text_encoder: TextEncoder
    text_bank: TextBank
    linen: VITALinen

    @classmethod
    def create(cls, config: VITAConfig, text_encoder: Optional[TextEncoder] = None,
               tokenizer=None) -> "VITAPolicy":
        config.validate()
        encoder = text_encoder or build_text_encoder(config.text)
        if encoder.output_dim <= 0:
            raise RuntimeError(getattr(encoder, "reason", "text encoder unavailable"))
        planner_enabled = bool(getattr(encoder, "joint_planner", False))
        if (config.text.encoder == "t5_flax" and config.text.joint_planner
                and not planner_enabled):
            raise RuntimeError(
                "the active t5_flax text path requires the joint seq2seq planner "
                "so the decoder loss can update the shared encoder parameters")
        text_config = config.text
        if not planner_enabled:
            # Synthetic/custom test-only encoders have no generative branch;
            # use the legacy clause texts rather than pretending to plan.
            text_config = replace(config.text, joint_planner=False,
                                  generated_text_targets=False)
        tokenizer = tokenizer or getattr(encoder, "tokenizer", None)
        bank = TextBank(text_config, tokenizer=tokenizer)
        n = len(config.model.image_keys)
        obs_dim = n * config.model.fasa_per_cam_dim + n * 512 + 2 * config.model.state_dim
        linen = VITALinen(config.model, obs_dim=obs_dim,
                          text_dim=encoder.output_dim)
        return cls(config=config, text_encoder=encoder, text_bank=bank, linen=linen)

    def dummy_batch(self, batch_size: int = 2, height: int = 32, width: int = 32):
        cfg = self.config.model
        n = len(cfg.image_keys)
        return {
            "images": jnp.zeros((batch_size, 2, n, 3, height, width), jnp.float32),
            "observation.state": jnp.zeros((batch_size, 2, 16), jnp.float32),
            "action_ee_chunk": jnp.zeros((batch_size, cfg.chunk_size, 16), jnp.float32),
            "future_images": jnp.zeros((batch_size, n, 3, height, width), jnp.float32),
            "future_valid": jnp.ones((batch_size,), jnp.bool_),
            "grounding_masks": jnp.zeros((batch_size, n, 3, height, width), jnp.float32),
            "stage": jnp.zeros((batch_size,), jnp.int32),
            "state_mean": jnp.zeros((16,), jnp.float32),
            "state_std": jnp.ones((16,), jnp.float32),
            "pose_mean": jnp.zeros((14,), jnp.float32),
            "pose_std": jnp.ones((14,), jnp.float32),
        }

    def init(self, rng, batch_size: int = 2, height: int = 32, width: int = 32,
             allow_random_vision: Optional[bool] = None):
        dummy = self.dummy_batch(batch_size, height, width)
        text_params = self.text_encoder.init_params(rng)
        text = self.encode_text(text_params)
        rngs = {"params": rng}
        for i, name in enumerate(("crop", "aug", "dropout",
                                  "state_delta", "perturb")):
            rngs[name] = jax.random.fold_in(rng, i + 1)
        variables = self.linen.init(rngs, dummy, text, train=True)
        params = {"linen": flax.core.unfreeze(variables["params"]),
                  "text": text_params}
        constants = flax.core.unfreeze(variables.get("constants", {}))
        batch_stats = flax.core.unfreeze(variables.get("batch_stats", {}))
        if allow_random_vision is None:
            allow_random_vision = self.config.text.encoder == "synthetic"
        weights = self.config.model.resnet18_imagenet_weights
        if weights:
            if "vision" not in params["linen"]:
                raise ValueError("initialized policy has no vision parameter tree")
            vision_params, vision_constants = load_resnet18_weights(
                weights, params["linen"]["vision"],
                constants.get("vision", {}))
            params["linen"]["vision"] = vision_params
            constants["vision"] = vision_constants
        elif not allow_random_vision:
            raise ValueError(
                "fresh VITA training requires a local ImageNet ResNet18 "
                "state_dict; pass --resnet18_imagenet_weights (random "
                "initialization is reserved for synthetic tests)")
        return params, constants, batch_stats

    def encode_text(self, text_params, train: bool = False, dropout_rng=None):
        arrays = self.text_bank.arrays
        out = {}
        keys = (jax.random.split(dropout_rng, 3)
                if train and dropout_rng is not None else (None, None, None))
        for index, group in enumerate(("clauses", "objects", "references")):
            ids = jnp.asarray(arrays.__getattribute__(group)["input_ids"])
            mask = jnp.asarray(arrays.__getattribute__(group)["attention_mask"])
            try:
                tokens = self.text_encoder.encode(
                    text_params, ids, mask, train=train,
                    dropout_rng=keys[index])
            except TypeError:
                tokens = self.text_encoder.encode(text_params, ids, mask)
            out[group] = {"tokens": tokens, "mask": mask.astype(bool)}
        return out

    def generate_plan_text(self, params, instruction=None):
        generate = getattr(self.text_encoder, "generate_plan_text", None)
        if generate is None:
            raise RuntimeError(
                "plan generation requires the joint seq2seq text encoder")
        instruction = instruction or self.config.text.task_instruction
        if not instruction:
            raise ValueError("plan generation requires a task instruction")
        return generate(params.get("text", {}), instruction)

    def set_generated_plan_text(self, raw):
        return self.text_bank.set_generated_plan(raw)

    def apply(self, params, constants, batch_stats, batch, train: bool,
              mutable=(), rng=None):
        text_rng = jax.random.fold_in(rng, 100) if train and rng is not None else None
        text = self.encode_text(params.get("text", {}), train=train,
                                dropout_rng=text_rng)
        variables = {"params": params["linen"], "constants": constants}
        if batch_stats:
            variables["batch_stats"] = batch_stats
        kwargs = {}
        if train:
            if rng is None:
                raise ValueError("training apply requires an rng")
            keys = jax.random.split(rng, 5)
            kwargs["rngs"] = {
                "crop": keys[0], "aug": keys[1], "dropout": keys[2],
                "state_delta": keys[3], "perturb": keys[4]}
        if mutable:
            out, mutated = self.linen.apply(
                variables, batch, text, train=train, mutable=mutable, **kwargs)
            return out, mutated
        return self.linen.apply(variables, batch, text, train=train, **kwargs)

    def loss(self, params, constants, batch_stats, batch, train: bool, rng=None,
             planner_index=0):
        planner_rng = None
        if train:
            if rng is None:
                raise ValueError("training loss requires an rng")
            rng, planner_rng = jax.random.split(rng)
            out, mutated = self.apply(params, constants, batch_stats, batch,
                                      train=True, mutable=("batch_stats",), rng=rng)
            new_batch_stats = mutated.get("batch_stats", batch_stats)
        else:
            out = self.apply(params, constants, batch_stats, batch, train=False)
            new_batch_stats = batch_stats
        loss = out["loss"]
        metrics = dict(out["metrics"])
        planning_loss = getattr(self.text_encoder, "planning_loss", None)
        weight = float(self.config.model.joint_plan_weight)
        if planning_loss is not None and weight > 0:
            planning_batch = self.text_bank.planning_batch(
                planner_index, self.config.text.joint_plan_batch_size)
            plan_loss = planning_loss(
                params.get("text", {}), planning_batch, train=train,
                dropout_rng=planner_rng)
            loss = loss + weight * plan_loss
            metrics["joint_plan_loss"] = plan_loss
            metrics["joint_plan_weighted_loss"] = weight * plan_loss
            metrics["total_loss"] = loss
        return loss, (metrics, new_batch_stats)
