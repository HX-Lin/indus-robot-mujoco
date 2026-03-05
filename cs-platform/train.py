"""
通用策略训练脚本 — 通过 --policy 参数选择模型

支持所有已注册的策略模型（ACT, VITA, ...），复用 LeRobot 数据集。

用法:
    python train.py --policy act --steps 100000
    python train.py --policy vita --steps 100000 --chunk_size 16
    python train.py --policy act --resume outputs/act/last
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle, dataset_to_policy_features
from lerobot.configs.types import FeatureType

from policies import get_policy_class, list_policies


def parse_args():
    # 第一轮: 只解析 --policy，获取策略类
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--policy", type=str, default="act")
    pre_args, _ = pre_parser.parse_known_args()

    # 获取策略类并让它注册自己的参数
    policy_cls = get_policy_class(pre_args.policy)

    # 第二轮: 完整解析（包含策略特有参数）
    parser = argparse.ArgumentParser(
        description=f"Train {pre_args.policy} policy on LeRobot dataset"
    )
    # 通用参数
    parser.add_argument("--policy", type=str, default="act",
                        help=f"Policy to use. Available: {list_policies()}")
    parser.add_argument("--data_dir", type=str, default="datasets_lerobot")
    parser.add_argument("--repo_id", type=str, default="local/mr2010-welding")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--save_freq", type=int, default=10000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: outputs/<policy_name>)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--val_freq", type=int, default=1000,
                        help="Validate every N steps (0 to disable)")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of episodes for validation")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--action_type", type=str, default="action",
                        choices=["action", "action.ee_delta"],
                        help="Action type: 'action' (joint abs) or 'action.ee_delta'")

    # 策略特有参数
    policy_cls.add_args(parser)

    args = parser.parse_args()

    # 设置默认 output_dir
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.policy}"

    # 设置默认 device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def to_float(x):
    return x.item() if isinstance(x, torch.Tensor) else float(x)


def save_loss_plot(train_history, val_history, output_dir):
    """保存训练/验证 loss 曲线图"""
    if not train_history:
        return

    # 确定 metrics 的 key 列表（从第一条记录中获取）
    metric_keys = list(train_history[0][1].keys())
    n_metrics = min(len(metric_keys), 4)  # 最多画 4 个子图

    fig, axes = plt.subplots(1, max(n_metrics, 1), figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for i, key in enumerate(metric_keys[:n_metrics]):
        t_steps = [h[0] for h in train_history]
        t_vals = [h[1].get(key, 0) for h in train_history]
        axes[i].plot(t_steps, t_vals, 'b-', alpha=0.7, label='train', linewidth=0.8)

        if val_history:
            v_steps = [h[0] for h in val_history]
            v_vals = [h[1].get(key, 0) for h in val_history]
            axes[i].plot(v_steps, v_vals, 'r-o', label='val', markersize=4, linewidth=1.5)

        axes[i].set_title(key)
        axes[i].set_xlabel('Step')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(f'{Path(output_dir).name}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(Path(output_dir) / 'loss_curve.png', dpi=150)
    plt.close(fig)


def validate(policy, val_loader, device, action_type="action"):
    """Compute average validation loss."""
    total_metrics = {}
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            # Remap action key if needed
            if action_type != "action" and action_type in batch:
                batch["action"] = batch.pop(action_type)
                pad_key = f"{action_type}_is_pad"
                if pad_key in batch:
                    batch["action_is_pad"] = batch.pop(pad_key)

            loss, loss_dict = policy.compute_loss(batch)
            for k, v in loss_dict.items():
                total_metrics[k] = total_metrics.get(k, 0) + to_float(v)
            total_metrics["loss"] = total_metrics.get("loss", 0) + to_float(loss)
            n_batches += 1

    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in total_metrics.items()}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load dataset metadata ────────────────────────────────────────
    print("\nLoading dataset...")
    ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.data_dir)
    fps = ds_meta.fps

    # 创建策略实例（用于获取 delta_timestamps 配置）
    policy_cls = get_policy_class(args.policy)
    policy = policy_cls()

    # delta_timestamps: 策略决定需要多少帧的 action chunk
    delta_timestamps = policy.get_delta_timestamps(fps)
    # 如果使用非默认 action type，重映射 key
    if args.action_type != "action" and "action" in delta_timestamps:
        delta_timestamps[args.action_type] = delta_timestamps.pop("action")

    # ── Train/Val split by episode ──
    total_episodes = ds_meta.total_episodes
    all_eps = list(range(total_episodes))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_eps)

    use_val = args.val_freq > 0 and args.val_split > 0
    if use_val:
        n_val = max(1, int(total_episodes * args.val_split))
        val_eps = sorted(all_eps[:n_val])
        train_eps = sorted(all_eps[n_val:])
    else:
        train_eps = all_eps
        val_eps = []

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.data_dir,
        video_backend="torchcodec",
        delta_timestamps=delta_timestamps,
    )
    ds_meta = dataset.meta
    print(f"  Total episodes: {total_episodes}")
    print(f"  Train episodes: {len(train_eps)}")

    # ── 2. Select features ─────────────────────────────────────────────
    all_features = dataset_to_policy_features(ds_meta.features)
    output_features = {}
    for k, v in all_features.items():
        if v.type is FeatureType.ACTION and k == args.action_type:
            output_features["action"] = v
            break
    input_features = {
        k: v for k, v in all_features.items()
        if v.type is not FeatureType.ACTION
        and (v.type is FeatureType.VISUAL or k == "observation.state")
    }

    print(f"\n  Input features:")
    for k, v in input_features.items():
        print(f"    {k}: type={v.type}, shape={v.shape}")
    print(f"  Output features:")
    for k, v in output_features.items():
        print(f"    {k}: type={v.type}, shape={v.shape}")

    # ── 3. Build policy ────────────────────────────────────────────────
    print(f"\nCreating {args.policy} policy...")
    policy.build(args, ds_meta.stats, input_features, output_features)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── 4. Optimizer ───────────────────────────────────────────────────
    optimizer = policy.get_optimizer()

    # LR Scheduler (if policy provides one)
    scheduler = None
    if hasattr(policy, 'get_scheduler'):
        scheduler = policy.get_scheduler(optimizer, args.steps)
        print(f"  LR Scheduler: cosine with warmup")

    # ── 5. Resume from checkpoint ──────────────────────────────────────
    start_step = 0
    if args.resume:
        resume_dir = Path(args.resume)
        if resume_dir.is_symlink():
            resume_dir = resume_dir.resolve()
        print(f"\nResuming from {resume_dir}...")
        policy = policy_cls.load(resume_dir, ds_meta.stats, device)
        optimizer = policy.get_optimizer()
        opt_path = resume_dir / "optimizer.pt"
        if opt_path.exists():
            opt_state = torch.load(opt_path, map_location=device, weights_only=True)
            optimizer.load_state_dict(opt_state["optimizer"])
            start_step = opt_state["step"]
            print(f"  Resumed at step {start_step}")

    # ── 6. Dataloaders ─────────────────────────────────────────────────
    chunk_size = policy.chunk_size
    drop_n = chunk_size - 1

    # Val loader
    val_loader = None
    if use_val and val_eps:
        val_sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=val_eps,
            drop_n_last_frames=drop_n,
            shuffle=False,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset, num_workers=args.num_workers, batch_size=args.batch_size,
            shuffle=False, sampler=val_sampler,
            pin_memory=device.type == "cuda", drop_last=False,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        print(f"  Val episodes:   {len(val_eps)}")
    print(f"  Frames:   {dataset.num_frames}")
    print(f"  FPS:      {fps}")

    # Train loader
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=train_eps,
        drop_n_last_frames=drop_n,
        shuffle=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_size=args.batch_size,
        shuffle=False, sampler=sampler,
        pin_memory=device.type == "cuda", drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    dl_iter = cycle(dataloader)

    # ── 7. Training loop ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Starting training: steps {start_step + 1} -> {args.steps}")
    print(f"  Policy:      {args.policy}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Chunk size:  {chunk_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Save freq:   {args.save_freq}")
    print(f"  Grad clip:   {args.grad_clip}")
    if val_loader:
        print(f"  Val freq:    {args.val_freq}")
        print(f"  Patience:    {args.patience if args.patience > 0 else 'disabled'}")
    print(f"{'=' * 60}")

    policy.train()
    train_start = time.time()
    running_metrics = {}

    # Early stopping state
    best_val_loss = float("inf")
    patience_counter = 0

    # Loss history for plotting
    train_history = []  # [(step, metrics_dict), ...]
    val_history = []

    for step in range(start_step + 1, args.steps + 1):
        batch = next(dl_iter)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        # Remap action key if needed
        if args.action_type != "action" and args.action_type in batch:
            batch["action"] = batch.pop(args.action_type)
            pad_key = f"{args.action_type}_is_pad"
            if pad_key in batch:
                batch["action_is_pad"] = batch.pop(pad_key)

        loss, loss_dict = policy.compute_loss(batch)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), args.grad_clip
        )
        optimizer.step()

        # LR scheduler step
        if scheduler is not None:
            scheduler.step()

        # EMA update
        if hasattr(policy, 'ema') and policy.ema is not None:
            policy.ema.update()

        # Accumulate metrics
        for k, v in loss_dict.items():
            running_metrics[k] = running_metrics.get(k, 0) + to_float(v)
        running_metrics["loss"] = running_metrics.get("loss", 0) + to_float(loss)

        # ── Logging ──
        if step % args.log_freq == 0:
            n = args.log_freq
            avg_metrics = {k: v / n for k, v in running_metrics.items()}
            elapsed = time.time() - train_start
            steps_done = step - start_step
            steps_per_sec = steps_done / elapsed
            eta_sec = (args.steps - step) / steps_per_sec if steps_per_sec > 0 else 0

            # Format metrics string
            metrics_str = " ".join(f"{k}={v:.4f}" for k, v in avg_metrics.items())
            print(
                f"step {step:6d}/{args.steps} | {metrics_str} | "
                f"grad={to_float(grad_norm):.2f} | "
                f"{steps_per_sec:.1f} it/s | ETA {eta_sec / 60:.0f}min"
            )
            train_history.append((step, avg_metrics))
            running_metrics = {}

        # ── Save checkpoint ──
        if step % args.save_freq == 0 or step == args.steps:
            ckpt_dir = output_dir / f"checkpoint_{step:07d}"
            policy.save(ckpt_dir)
            torch.save(
                {"step": step, "optimizer": optimizer.state_dict()},
                ckpt_dir / "optimizer.pt",
            )
            last_link = output_dir / "last"
            if last_link.is_symlink() or last_link.exists():
                last_link.unlink()
            last_link.symlink_to(ckpt_dir.name)
            print(f"  >> Saved checkpoint to {ckpt_dir}")

        # ── Validation ──
        if val_loader and args.val_freq > 0 and step % args.val_freq == 0:
            # 验证时使用 EMA 权重
            use_ema = hasattr(policy, 'ema') and policy.ema is not None
            if use_ema:
                policy.ema.apply_shadow()

            val_metrics = validate(policy, val_loader, device, args.action_type)

            if use_ema:
                policy.ema.restore()
            val_loss = val_metrics.get("loss", float("inf"))
            improved = val_loss < best_val_loss

            marker = " *BEST*" if improved else ""
            val_str = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"[VAL] step {step:6d} | {val_str}{marker}")
            val_history.append((step, val_metrics))
            save_loss_plot(train_history, val_history, output_dir)

            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                best_dir = output_dir / "best"
                policy.save(best_dir)
                torch.save(
                    {"step": step, "optimizer": optimizer.state_dict(),
                     "best_val_loss": best_val_loss},
                    best_dir / "optimizer.pt",
                )
                print(f"  >> Saved best checkpoint (val_loss={best_val_loss:.4f}) to {best_dir}")
            else:
                patience_counter += 1
                if args.patience > 0:
                    print(f"  >> No improvement ({patience_counter}/{args.patience})")
                    if patience_counter >= args.patience:
                        print(f"\n{'=' * 60}")
                        print(f"Early stopping at step {step} (patience={args.patience} exhausted)")
                        print(f"Best val_loss={best_val_loss:.4f}")
                        print(f"{'=' * 60}")
                        break

            # Resume training mode
            policy.train()

    total_time = time.time() - train_start
    print(f"\n{'=' * 60}")
    print(f"Training complete in {total_time / 60:.1f} minutes")
    print(f"Final checkpoint: {output_dir / 'last'}")
    if val_loader and best_val_loss < float("inf"):
        print(f"Best val_loss: {best_val_loss:.4f} (checkpoint: {output_dir / 'best'})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(f"Policy Training — {args.policy.upper()} — MR2010 Welding")
    print("=" * 60)
    train(args)
