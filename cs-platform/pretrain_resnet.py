"""
ResNet 预蒸馏脚本 — 只训练 ResNet backbone + 蒸馏 head

Phase 1: 先让 ResNet 学会"看到工件在哪"（空间注意力 + 语义特征对齐），
         然后 Phase 2 用预训练权重初始化 VITA 完整训练。

用法:
    python pretrain_resnet.py \
        --heatmap_dir gdino_heatmaps \
        --data_dir datasets_lerobot --repo_id local/mr2010-welding \
        --steps 50000 --lr 1e-4 --batch_size 16 \
        --output_dir outputs/pretrain_resnet
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet pre-distillation for GDINO features")
    parser.add_argument("--data_dir", type=str, default="datasets_lerobot")
    parser.add_argument("--repo_id", type=str, default="local/mr2010-welding")
    parser.add_argument("--heatmap_dir", type=str, required=True,
                        help="Path to pre-extracted GDINO heatmaps directory")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="outputs/pretrain_resnet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resize_shape", type=int, nargs=2, default=[240, 320])
    parser.add_argument("--crop_shape", type=int, nargs=2, default=[224, 308])
    parser.add_argument("--distill_weight", type=float, default=1.0,
                        help="Spatial attention (heatmap BCE) loss weight")
    parser.add_argument("--feat_distill_weight", type=float, default=1.0,
                        help="Semantic feature (cosine sim) loss weight")
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


class ResNetDistillModel(nn.Module):
    """ResNet18 backbone + 蒸馏 head（与 VITA 中完全相同的结构）."""

    def __init__(self, image_keys, resize_shape=(240, 320), crop_shape=(224, 308)):
        super().__init__()
        import torchvision
        from torchvision.models._utils import IntermediateLayerGetter
        from torchvision.ops.misc import FrozenBatchNorm2d

        self.image_keys = image_keys
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape

        # ResNet18 backbone（与 VITA ResNetObserver 完全一致）
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(
            backbone, return_layers={"layer4": "feature_map"})

        # 蒸馏 head（结构与 VITA build 中一致）
        # attn_head: ResNet feat_map → 空间注意力预测
        self.attn_head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        # feat_proj: ResNet 512-ch → Swin-T 768-ch 语义特征对齐
        self.feat_proj = nn.Sequential(
            nn.Conv2d(512, 768, kernel_size=1, bias=False),
            nn.BatchNorm2d(768),
        )

    def forward(self, images):
        """Extract ResNet feature map from images.

        Args:
            images: (B, C, H, W) raw images
        Returns:
            feat_map: (B, 512, H', W') ResNet feature map
        """
        # Resize
        if images.shape[-2:] != tuple(self.resize_shape):
            images = F.interpolate(images, size=tuple(self.resize_shape),
                                   mode='bilinear', align_corners=False)
        # Center crop (deterministic for consistency)
        ch, cw = self.crop_shape
        _, _, H, W = images.shape
        if self.training:
            top = torch.randint(0, H - ch + 1, (1,)).item()
            left = torch.randint(0, W - cw + 1, (1,)).item()
        else:
            top = (H - ch) // 2
            left = (W - cw) // 2
        images = images[..., top:top + ch, left:left + cw]

        feat_map = self.backbone(images)["feature_map"]  # (B, 512, H', W')
        return feat_map


class OfflineHeatmapLoader:
    """Simplified offline heatmap loader for pre-distillation."""

    def __init__(self, heatmap_dir, image_keys):
        self.heatmaps = {}
        self.spatial_features = {}
        heatmap_dir = Path(heatmap_dir)

        with open(heatmap_dir / "meta.json") as f:
            meta = json.load(f)

        total_frames = meta["total_frames"]
        feat_h = meta["feat_h"]
        feat_w = meta["feat_w"]
        swin_dim = meta.get("swin_feat_dim", 768)

        for img_key in image_keys:
            key_dir = heatmap_dir / img_key.replace(".", "_")
            hm_path = key_dir / "heatmaps.bin"
            if hm_path.exists():
                self.heatmaps[img_key] = np.memmap(
                    str(hm_path), dtype=np.float16, mode='r',
                    shape=(total_frames, 1, feat_h, feat_w))
                print(f"  Loaded heatmaps: {hm_path} shape=({total_frames}, 1, {feat_h}, {feat_w})")

            sfeat_path = key_dir / "spatial_features.bin"
            if sfeat_path.exists():
                self.spatial_features[img_key] = np.memmap(
                    str(sfeat_path), dtype=np.float16, mode='r',
                    shape=(total_frames, swin_dim, feat_h, feat_w))
                print(f"  Loaded spatial features: {sfeat_path} shape=({total_frames}, {swin_dim}, {feat_h}, {feat_w})")

    def get_batch(self, image_key, indices, device):
        """Get heatmap + spatial feature batch by frame indices."""
        heatmaps = None
        spatial_feats = None

        if image_key in self.heatmaps:
            mmap = self.heatmaps[image_key]
            hms = []
            for idx in indices:
                idx = int(idx)
                if 0 <= idx < mmap.shape[0]:
                    hms.append(torch.from_numpy(mmap[idx].copy()).float())
            if hms:
                heatmaps = torch.stack(hms, dim=0).to(device)

        if image_key in self.spatial_features:
            mmap = self.spatial_features[image_key]
            feats = []
            for idx in indices:
                idx = int(idx)
                if 0 <= idx < mmap.shape[0]:
                    feats.append(torch.from_numpy(mmap[idx].copy()).float())
            if feats:
                spatial_feats = torch.stack(feats, dim=0).to(device)

        return heatmaps, spatial_feats


def save_loss_plot(history, output_dir):
    if not history:
        return
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    steps = [h[0] for h in history]

    # Total loss
    ax1.plot(steps, [h[1]["total_loss"] for h in history], 'b-', alpha=0.7)
    ax1.set_title("Total Loss")
    ax1.set_xlabel("Step")
    ax1.grid(True, alpha=0.3)

    # Distill loss (BCE)
    if "distill_loss" in history[0][1]:
        ax2.plot(steps, [h[1].get("distill_loss", 0) for h in history], 'r-', alpha=0.7)
        ax2.axhline(y=0.05, color='green', alpha=0.4, linestyle='--', linewidth=1)
        ax2.set_title("distill_loss (target <0.05)")
    ax2.set_xlabel("Step")
    ax2.grid(True, alpha=0.3)

    # Feature distill loss (cosine)
    if "feat_distill_loss" in history[0][1]:
        ax3.plot(steps, [h[1].get("feat_distill_loss", 0) for h in history], 'g-', alpha=0.7)
        ax3.axhline(y=0.05, color='green', alpha=0.4, linestyle='--', linewidth=1)
        ax3.set_title("feat_distill_loss (target <0.05)")
    ax3.set_xlabel("Step")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("ResNet Pre-distillation", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "loss_curve.png", dpi=150)
    plt.close(fig)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset metadata
    print("\nLoading dataset...")
    ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.data_dir)
    fps = ds_meta.fps

    # Load dataset (only need images + index for heatmap lookup)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.data_dir,
        video_backend="torchcodec",
    )

    # Image keys from heatmap meta
    with open(Path(args.heatmap_dir) / "meta.json") as f:
        heatmap_meta = json.load(f)
    image_keys = heatmap_meta["image_keys"]
    print(f"  Image keys: {image_keys}")

    # Load offline heatmaps
    print("\nLoading offline heatmaps...")
    heatmap_store = OfflineHeatmapLoader(args.heatmap_dir, image_keys)

    # Train/Val split
    total_episodes = ds_meta.total_episodes
    all_eps = list(range(total_episodes))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_eps)

    n_val = max(1, int(total_episodes * args.val_split))
    val_eps = sorted(all_eps[:n_val])
    train_eps = sorted(all_eps[n_val:])

    print(f"  Total episodes: {total_episodes}")
    print(f"  Train episodes: {len(train_eps)}")
    print(f"  Val episodes:   {len(val_eps)}")
    print(f"  Total frames:   {dataset.num_frames}")

    # Build model
    print("\nCreating ResNet distillation model...")
    model = ResNetDistillModel(
        image_keys=image_keys,
        resize_shape=tuple(args.resize_shape),
        crop_shape=tuple(args.crop_shape),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} (trainable: {n_trainable:,})")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.95, 0.999), weight_decay=1e-6)

    # LR scheduler
    from diffusers.optimization import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=args.steps)

    # Dataloaders
    train_sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=train_eps,
        drop_n_last_frames=0,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_size=args.batch_size,
        shuffle=False, sampler=train_sampler,
        pin_memory=device.type == "cuda", drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    dl_iter = cycle(train_loader)

    val_sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=val_eps,
        drop_n_last_frames=0,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        pin_memory=device.type == "cuda", drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Starting pre-distillation: steps 1 -> {args.steps}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  distill_weight:      {args.distill_weight}")
    print(f"  feat_distill_weight: {args.feat_distill_weight}")
    print(f"  Target: distill_loss < 0.05, feat_distill_loss < 0.05")
    print(f"{'=' * 60}")

    model.train()
    train_start = time.time()
    running_metrics = {}
    train_history = []
    best_val_loss = float("inf")
    img_key = image_keys[0]  # Use first camera for distillation

    for step in range(1, args.steps + 1):
        batch = next(dl_iter)

        # Get frame indices for heatmap lookup
        frame_indices = batch.get("index", None)
        if frame_indices is None:
            continue
        if frame_indices.dim() > 1:
            frame_indices = frame_indices.squeeze(-1)
        idx_list = frame_indices.tolist()

        # Get images
        images = batch[img_key].to(device, non_blocking=True)
        if images.dim() == 5:
            images = images[:, -1]  # Take last frame if multi-frame

        # Forward: extract ResNet feature map
        feat_map = model(images)  # (B, 512, H', W')
        fh, fw = feat_map.shape[2], feat_map.shape[3]

        # Get targets from offline store
        heatmaps, spatial_feats = heatmap_store.get_batch(img_key, idx_list, device)

        loss = torch.tensor(0.0, device=device)
        metrics = {}

        # 1. Spatial attention distillation (heatmap BCE loss)
        if heatmaps is not None:
            if heatmaps.shape[2] != fh or heatmaps.shape[3] != fw:
                heatmaps = F.interpolate(heatmaps, size=(fh, fw),
                                         mode='bilinear', align_corners=False)
            pred_attn = model.attn_head(feat_map)
            distill_loss = F.binary_cross_entropy_with_logits(pred_attn, heatmaps)
            loss = loss + args.distill_weight * distill_loss
            metrics["distill_loss"] = distill_loss.item()

        # 2. Semantic feature distillation (cosine similarity loss)
        if spatial_feats is not None:
            if spatial_feats.shape[2] != fh or spatial_feats.shape[3] != fw:
                spatial_feats = F.interpolate(spatial_feats, size=(fh, fw),
                                              mode='bilinear', align_corners=False)
            resnet_proj = model.feat_proj(feat_map)
            cos_sim = F.cosine_similarity(resnet_proj, spatial_feats, dim=1)
            feat_loss = 1.0 - cos_sim.mean()
            loss = loss + args.feat_distill_weight * feat_loss
            metrics["feat_distill_loss"] = feat_loss.item()

        metrics["total_loss"] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        # Accumulate metrics
        for k, v in metrics.items():
            running_metrics[k] = running_metrics.get(k, 0) + v

        # Logging
        if step % args.log_freq == 0:
            n = args.log_freq
            avg = {k: v / n for k, v in running_metrics.items()}
            elapsed = time.time() - train_start
            sps = step / elapsed
            eta = (args.steps - step) / sps if sps > 0 else 0
            m_str = " ".join(f"{k}={v:.4f}" for k, v in avg.items())
            print(f"step {step:6d}/{args.steps} | {m_str} | "
                  f"{sps:.1f} it/s | ETA {eta/60:.0f}min")
            train_history.append((step, avg))
            running_metrics = {}

        # Save checkpoint
        if step % args.save_freq == 0 or step == args.steps:
            ckpt_dir = output_dir / f"checkpoint_{step:07d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "backbone": model.backbone.state_dict(),
                "attn_head": model.attn_head.state_dict(),
                "feat_proj": model.feat_proj.state_dict(),
                "step": step,
            }, ckpt_dir / "pretrain.pt")
            # Save meta
            with open(ckpt_dir / "pretrain_meta.json", "w") as f:
                json.dump({
                    "step": step,
                    "resize_shape": list(args.resize_shape),
                    "crop_shape": list(args.crop_shape),
                    "image_keys": image_keys,
                }, f, indent=2)
            last_link = output_dir / "last"
            if last_link.is_symlink() or last_link.exists():
                last_link.unlink()
            last_link.symlink_to(ckpt_dir.name)
            print(f"  >> Saved checkpoint to {ckpt_dir}")

        # Validation
        if args.val_freq > 0 and step % args.val_freq == 0:
            model.eval()
            val_metrics = {}
            n_batches = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vi = vbatch.get("index", None)
                    if vi is None:
                        continue
                    if vi.dim() > 1:
                        vi = vi.squeeze(-1)
                    vimages = vbatch[img_key].to(device)
                    if vimages.dim() == 5:
                        vimages = vimages[:, -1]
                    vfeat = model(vimages)
                    vfh, vfw = vfeat.shape[2], vfeat.shape[3]

                    vhm, vsf = heatmap_store.get_batch(img_key, vi.tolist(), device)
                    vloss = 0.0
                    if vhm is not None:
                        if vhm.shape[2] != vfh or vhm.shape[3] != vfw:
                            vhm = F.interpolate(vhm, size=(vfh, vfw),
                                                mode='bilinear', align_corners=False)
                        vp = model.attn_head(vfeat)
                        vdl = F.binary_cross_entropy_with_logits(vp, vhm).item()
                        val_metrics["distill_loss"] = val_metrics.get("distill_loss", 0) + vdl
                        vloss += vdl
                    if vsf is not None:
                        if vsf.shape[2] != vfh or vsf.shape[3] != vfw:
                            vsf = F.interpolate(vsf, size=(vfh, vfw),
                                                mode='bilinear', align_corners=False)
                        vrp = model.feat_proj(vfeat)
                        vcs = F.cosine_similarity(vrp, vsf, dim=1)
                        vfl = (1.0 - vcs.mean()).item()
                        val_metrics["feat_distill_loss"] = val_metrics.get("feat_distill_loss", 0) + vfl
                        vloss += vfl
                    val_metrics["total_loss"] = val_metrics.get("total_loss", 0) + vloss
                    n_batches += 1

            if n_batches > 0:
                val_avg = {k: v / n_batches for k, v in val_metrics.items()}
                vl = val_avg["total_loss"]
                improved = vl < best_val_loss
                marker = " *BEST*" if improved else ""
                v_str = " ".join(f"{k}={v:.4f}" for k, v in val_avg.items())
                print(f"[VAL] step {step:6d} | {v_str}{marker}")

                if improved:
                    best_val_loss = vl
                    best_dir = output_dir / "best"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "backbone": model.backbone.state_dict(),
                        "attn_head": model.attn_head.state_dict(),
                        "feat_proj": model.feat_proj.state_dict(),
                        "step": step,
                    }, best_dir / "pretrain.pt")
                    with open(best_dir / "pretrain_meta.json", "w") as f:
                        json.dump({
                            "step": step,
                            "resize_shape": list(args.resize_shape),
                            "crop_shape": list(args.crop_shape),
                            "image_keys": image_keys,
                            "best_val_loss": best_val_loss,
                        }, f, indent=2)
                    print(f"  >> Saved best checkpoint (val_loss={best_val_loss:.4f}) to {best_dir}")

            save_loss_plot(train_history, output_dir)
            model.train()

    total_time = time.time() - train_start
    print(f"\n{'=' * 60}")
    print(f"Pre-distillation complete in {total_time / 60:.1f} minutes")
    print(f"Final checkpoint: {output_dir / 'last'}")
    print(f"Best val_loss: {best_val_loss:.4f} (checkpoint: {output_dir / 'best'})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("ResNet Pre-distillation for GDINO Features")
    print("=" * 60)
    train(args)
