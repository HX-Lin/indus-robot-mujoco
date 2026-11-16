"""Command-line entrypoint for the JAX VITA port.

No data is mounted or transferred at import time.  Mount Google Drive only when
the user passes ``--mount_drive`` (or sets MOUNT_DRIVE in the notebook);
otherwise pass ordinary local/Drive paths.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from .config import DataConfig, ModelConfig, TextConfig, TrainConfig, VITAConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="JAX/Flax VITA ODE + PPF training")
    p.add_argument("--mount_drive", action="store_true",
                   help="Mount Google Drive when this command is executed. "
                        "Never performed by import.")
    p.add_argument("--mount_point", default="/content/drive")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--repo_id", required=True)
    p.add_argument("--ee_cache", required=True)
    p.add_argument("--stage_cache", required=True)
    p.add_argument("--grounding_cache", required=True)
    p.add_argument("--language_t5_model", default="")
    p.add_argument("--language_instruction", default=None,
                   help="Original task instruction used by the joint planner.")
    p.add_argument("--text_encoder", choices=("t5_flax", "synthetic"),
                   default="t5_flax")
    p.add_argument("--joint_planner", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Train the shared Flan-T5 encoder/decoder planner loss.")
    p.add_argument("--generated_text_targets",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Use literal generated-plan fields as FASA text inputs.")
    p.add_argument("--joint_plan_batch_size", type=int, default=8)
    p.add_argument("--planner_max_length", type=int, default=192)
    p.add_argument("--planner_labels_max_length", type=int, default=128)
    p.add_argument("--resnet18_imagenet_weights", default="",
                   help="Local torchvision ResNet18 ImageNet state_dict. "
                        "Required for fresh production training; never downloaded.")
    p.add_argument("--output_dir", default="vita_jax_runs/ode_ppf_pred_action")
    p.add_argument("--resume", default=None)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--language_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_freq", type=int, default=10000)
    p.add_argument("--eval_freq", type=int, default=0)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval_batches", type=int, default=0)
    p.add_argument("--val_split", type=float, default=0.0,
                   help="Active PyTorch run has val_freq=0, so this port requires 0")
    p.add_argument("--fps", type=float, required=True,
                   help="Dataset FPS from LeRobot metadata; timestamps depend on this")
    p.add_argument("--order", default="red green blue")
    p.add_argument("--target_descriptions", nargs=3, default=None)
    p.add_argument("--reference_texts", nargs=3, default=None)
    p.add_argument("--image_keys", nargs="+", default=list(ModelConfig().image_keys))
    p.add_argument("--drop_n_last_frames", type=int, default=None)
    p.add_argument("--no_jit", action="store_true")
    return p.parse_args(argv)


def build_config(args) -> VITAConfig:
    model = ModelConfig(
        image_keys=tuple(args.image_keys),
        resnet18_imagenet_weights=getattr(
            args, "resnet18_imagenet_weights", ""),
    )
    data = DataConfig(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        ee_cache=args.ee_cache,
        stage_cache=args.stage_cache,
        grounding_cache=args.grounding_cache,
        val_split=args.val_split,
        seed=args.seed,
        drop_n_last_frames=args.drop_n_last_frames,
        future_visual_horizon=model.ppf_future_visual_horizon,
    )
    text = TextConfig(
        encoder=args.text_encoder,
        model_path=args.language_t5_model,
        max_length=model.language_max_length,
        order=args.order,
        generated_text_targets=getattr(args, "generated_text_targets", True),
        joint_planner=getattr(args, "joint_planner", True),
        joint_plan_batch_size=getattr(args, "joint_plan_batch_size", 8),
        planner_max_length=getattr(args, "planner_max_length", 192),
        planner_labels_max_length=getattr(
            args, "planner_labels_max_length", 128),
        target_descriptions=(tuple(args.target_descriptions)
                             if args.target_descriptions else None),
        reference_texts=(tuple(args.reference_texts)
                         if args.reference_texts else None),
        task_instruction=getattr(args, "language_instruction", None),
    )
    train = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        language_lr=args.language_lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, warmup_steps=args.warmup_steps,
        seed=args.seed, save_freq=args.save_freq, eval_freq=args.eval_freq,
        output_dir=args.output_dir, resume=args.resume,
        eval_batches=args.eval_batches, jit=not args.no_jit,
    )
    return VITAConfig(model=model, data=data, text=text, train=train).validate()


def main(argv=None):
    args = parse_args(argv)
    if args.mount_drive:
        from google.colab import drive
        drive.mount(args.mount_point)
    config = build_config(args)
    from .data import SidecarData, build_datasets, normalization_arrays
    from .policy import VITAPolicy
    from .train import evaluate, train

    train_ds, val_ds = build_datasets(config.data, config.model, fps=args.fps)
    sidecar = SidecarData(config.data, chunk_size=config.model.chunk_size)
    norm = normalization_arrays(sidecar)
    policy = VITAPolicy.create(config)
    if args.eval_only:
        if not config.train.resume:
            raise ValueError("--eval_only requires --resume")
        rng = __import__("jax").random.PRNGKey(config.train.seed)
        params, constants, batch_stats = policy.init(rng)
        from .checkpoint import load_checkpoint
        from .train import make_optimizer
        optimizer = make_optimizer(params, config.train)
        opt_state = optimizer.init(params)
        params, constants, batch_stats, _, _, _ = load_checkpoint(
            config.train.resume, params, constants, batch_stats, opt_state)
        if (config.text.encoder == "t5_flax" and config.text.joint_planner
                and config.text.generated_text_targets):
            raw_plan = policy.generate_plan_text(params)
            policy.set_generated_plan_text(raw_plan)
            print(f"generated plan: {raw_plan}")
        metrics = evaluate(config, policy, val_ds or train_ds, norm,
                           params, constants, batch_stats,
                           max_batches=config.train.eval_batches)
        print(metrics)
        return metrics
    trained = train(config, policy=policy, dataset=train_ds, norm_arrays=norm)
    if val_ds is not None and config.train.eval_freq:
        # The final evaluation is intentionally separate from the training
        # objective; reload the saved last checkpoint to make that explicit.
        print("saved final checkpoint; run --eval_only --resume for evaluation")
    return trained


if __name__ == "__main__":
    main()
