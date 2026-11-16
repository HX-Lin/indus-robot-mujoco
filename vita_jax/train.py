"""Optax training loop, schedule, clipping and evaluation for the JAX VITA port."""

from __future__ import annotations

import dataclasses
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainConfig, VITAConfig
from .data import (build_datasets, collate, future_image_key,
                   normalization_arrays, SidecarData)
from .policy import VITAPolicy


def make_schedule(peak: float, train: TrainConfig):
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(peak),
        warmup_steps=int(train.warmup_steps),
        decay_steps=max(1, int(train.steps)),
    )


def make_optimizer(params, train: TrainConfig):
    policy_tx = optax.adamw(
        learning_rate=make_schedule(train.lr, train),
        b1=train.beta1, b2=train.beta2, eps=train.eps,
        weight_decay=train.weight_decay)
    text_tx = optax.adamw(
        learning_rate=make_schedule(train.language_lr, train),
        b1=train.beta1, b2=train.beta2, eps=train.eps,
        weight_decay=train.weight_decay)
    labels = {
        "linen": jax.tree_util.tree_map(lambda _: "policy", params["linen"]),
        "text": jax.tree_util.tree_map(lambda _: "text", params.get("text", {})),
    }
    transforms = {"policy": policy_tx, "text": text_tx}
    return optax.chain(
        optax.clip_by_global_norm(float(train.grad_clip)),
        optax.multi_transform(transforms, labels))


def prepare_batch(raw, config: VITAConfig, norm_arrays) -> Dict[str, jnp.ndarray]:
    model = config.model
    images = np.stack([np.asarray(raw[k]) for k in model.image_keys], axis=2)
    batch = {
        "images": images,
        "observation.state": np.asarray(raw["observation.state"]),
        "action_ee_chunk": np.asarray(raw["action_ee_chunk"]),
        "future_images": np.stack(
            [np.asarray(raw[future_image_key(k)]) for k in model.image_keys], axis=1),
        "future_valid": np.asarray(raw["observation.future_visual_valid"]),
        "state_mean": norm_arrays["observation.state"]["mean"],
        "state_std": norm_arrays["observation.state"]["std"],
        "pose_mean": norm_arrays["action_ee_chunk"]["mean"],
        "pose_std": norm_arrays["action_ee_chunk"]["std"],
    }
    if all(("grounding." + k) in raw for k in model.image_keys):
        batch["grounding_masks"] = np.stack(
            [np.asarray(raw["grounding." + k]) for k in model.image_keys], axis=1)
    if "observation.stage" in raw:
        batch["stage"] = np.asarray(raw["observation.stage"])
    return {k: jnp.asarray(v) for k, v in batch.items()}


def train(config: VITAConfig, policy: Optional[VITAPolicy] = None,
          dataset=None, norm_arrays=None) -> VITAPolicy:
    config.validate()
    if config.model.use_ppf_lite:
        raise NotImplementedError("legacy PPF-lite heads are not part of the active run")
    train_ds = dataset
    sidecar = None
    if train_ds is None:
        raise ValueError(
            "train() requires a dataset built with the LeRobot metadata FPS; "
            "use entrypoint.main or pass dataset= explicitly")
        sidecar = SidecarData(config.data, chunk_size=config.model.chunk_size)
    if norm_arrays is None:
        if sidecar is None:
            sidecar = SidecarData(config.data, chunk_size=config.model.chunk_size)
        norm_arrays = normalization_arrays(sidecar)
    if policy is None:
        policy = VITAPolicy.create(config)
    if (config.model.joint_plan_weight > 0
            and config.text.encoder == "t5_flax"
            and not hasattr(policy.text_encoder, "planning_loss")):
        raise RuntimeError(
            "joint_plan_weight > 0 requires the Flan-T5 joint planner; the "
            "configured text encoder has no teacher-forced decoder loss")

    rng = jax.random.PRNGKey(config.train.seed)
    params, constants, batch_stats = policy.init(rng)
    optimizer = make_optimizer(params, config.train)
    opt_state = optimizer.init(params)
    start_step = 0
    out_dir = Path(config.train.output_dir)
    if config.train.resume:
        params, constants, batch_stats, opt_state, start_step, _ = load_checkpoint(
            config.train.resume, params, constants, batch_stats, opt_state)
        optimizer = make_optimizer(params, config.train)
        warnings.warn(
            "resume loads model/optimizer pytrees; the cosine schedule restarts "
            "over the remaining steps from --steps as the fresh-run convention.",
            RuntimeWarning)

    drop_n = (config.data.drop_n_last_frames
              if config.data.drop_n_last_frames is not None
              else config.model.chunk_size - 1)
    sampler = train_ds.sampler(drop_n, seed=config.train.seed, shuffle=True)

    @jax.jit
    def step(params, constants, batch_stats, opt_state, batch, rng,
             planner_index):
        def loss_fn(params):
            return policy.loss(params, constants, batch_stats, batch,
                               train=True, rng=rng,
                               planner_index=planner_index)
        (loss, (metrics, new_batch_stats)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, new_batch_stats, new_opt_state, loss, metrics

    iterator = sampler.batches(config.train.batch_size)
    started = time.time()
    for step_index in range(start_step, int(config.train.steps)):
        try:
            indices = next(iterator)
        except StopIteration:
            iterator = sampler.batches(config.train.batch_size)
            indices = next(iterator)
        raw = collate([train_ds[i] for i in indices])
        batch = prepare_batch(raw, config, norm_arrays)
        rng, step_rng = jax.random.split(rng)
        params, batch_stats, opt_state, loss, metrics = step(
            params, constants, batch_stats, opt_state, batch, step_rng,
            step_index * config.text.joint_plan_batch_size)
        if config.train.log_freq and step_index % config.train.log_freq == 0:
            m = {k: float(np.asarray(v)) for k, v in metrics.items()
                 if np.asarray(v).ndim == 0}
            print(f"step {step_index + 1} loss {float(loss):.6f} "
                  f"({time.time() - started:.1f}s) {m}")
        if config.train.save_freq and (step_index + 1) % config.train.save_freq == 0:
            save_checkpoint(out_dir / "last", params, constants, batch_stats,
                            opt_state, step_index + 1,
                            {"config": dataclasses.asdict(config),
                             "norm": norm_arrays})
            print(f"saved {out_dir / 'last'}")
    save_checkpoint(out_dir / "last", params, constants, batch_stats, opt_state,
                    int(config.train.steps),
                    {"config": dataclasses.asdict(config), "norm": norm_arrays})
    return policy


def evaluate(config: VITAConfig, policy: VITAPolicy, dataset,
             norm_arrays, params, constants, batch_stats,
             max_batches: int = 0) -> Dict[str, float]:
    drop_n = (config.data.drop_n_last_frames
              if config.data.drop_n_last_frames is not None
              else config.model.chunk_size - 1)
    sampler = dataset.sampler(drop_n, seed=config.data.seed, shuffle=False)
    totals = {}
    count = 0
    for indices in sampler.batches(config.train.batch_size):
        raw = collate([dataset[i] for i in indices])
        batch = prepare_batch(raw, config, norm_arrays)
        loss, (metrics, _) = policy.loss(
            params, constants, batch_stats, batch, train=False,
            planner_index=count * config.text.joint_plan_batch_size)
        for key, value in metrics.items():
            value = float(np.asarray(value))
            totals[key] = totals.get(key, 0.0) + value
        totals["loss"] = totals.get("loss", 0.0) + float(loss)
        count += 1
        if max_batches and count >= max_batches:
            break
    return {key: value / max(1, count) for key, value in totals.items()}
