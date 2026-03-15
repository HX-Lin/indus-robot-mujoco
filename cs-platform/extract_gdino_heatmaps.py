"""
离线提取 Grounding DINO 空间注意力热图 + Swin-T 语义特征

遍历 LeRobot 数据集中所有帧的图像:
  1. 使用 GDINO 检测目标物体 → 生成高斯热图 (空间注意力蒸馏)
  2. 提取 Swin-T backbone 最后一层特征 → pool 到 ResNet 尺寸 (语义特征蒸馏)

采用内存映射存储 (numpy memmap)，提取时流式写入，训练时按需读取，
内存占用极低 (< 100MB)，支持任意规模数据集。

用法:
    python extract_gdino_heatmaps.py \
        --data_dir datasets_lerobot \
        --repo_id local/mr2010-welding \
        --output_dir gdino_heatmaps \
        --batch_size 8 \
        --prompt "object."

输出结构:
    gdino_heatmaps/
        observation_images_overhead/
            heatmaps.bin         # memmap (N, 1, feat_h, feat_w) float16
            spatial_features.bin # memmap (N, 768, feat_h, feat_w) float16
            features.bin         # memmap (N, 768) float16
        observation_images_ee/
            heatmaps.bin, spatial_features.bin, features.bin
        meta.json                # 提取参数记录
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_gdino(checkpoint_path, device):
    """加载 Grounding DINO 模型."""
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

    import groundingdino
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.models import build_model
    from groundingdino.util.utils import clean_state_dict

    config_path = os.path.join(
        os.path.dirname(groundingdino.__file__),
        "config", "GroundingDINO_SwinT_OGC.py"
    )
    args = SLConfig.fromfile(config_path)
    args.device = device

    # 使用本地 BERT
    local_bert = os.path.expanduser(
        '~/.cache/huggingface/hub/models--bert-base-uncased')
    if os.path.isdir(local_bert):
        args.text_encoder_type = local_bert

    model = build_model(args)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_swin_features(gdino, images, pool_h, pool_w):
    """提取 Swin-T backbone 最后一层特征.

    Args:
        gdino: Grounding DINO model
        images: (B, C, H, W) 图像 tensor
        pool_h, pool_w: AdaptiveAvgPool 目标尺寸 (与 ResNet feat_map 对齐)
    Returns:
        global_features: (B, 768) 全局语义特征 (GAP)
        spatial_features: (B, 768, pool_h, pool_w) 空间语义特征
    """
    from groundingdino.util.misc import NestedTensor
    b = images.shape[0]
    device = images.device
    mask = torch.zeros(b, images.shape[2], images.shape[3],
                       dtype=torch.bool, device=device)
    nt = NestedTensor(images, mask)

    # Swin-T backbone forward → dict of NestedTensor
    # Level 2 (最深层): (B, 768, H/32, W/32)
    swin_outs = gdino.backbone[0](nt)
    last_feat = swin_outs[2].tensors  # (B, 768, H/32, W/32)

    # 全局特征: GAP → (B, 768)
    global_features = F.adaptive_avg_pool2d(last_feat, (1, 1)).flatten(1)

    # 空间特征: 对齐到 ResNet feat_map 尺寸
    spatial_features = F.adaptive_avg_pool2d(last_feat, (pool_h, pool_w))

    return global_features, spatial_features


@torch.no_grad()
def generate_heatmap(gdino, images, text_prompt, feat_h, feat_w,
                     box_threshold=0.25):
    """用 GDINO 检测目标，生成高斯注意力热图.

    Args:
        gdino: Grounding DINO model
        images: (B, C, H, W) 图像 tensor ∈ [0, 1]
        text_prompt: 文本提示 (需以 "." 结尾)
        feat_h, feat_w: 目标热图空间尺寸
        box_threshold: 检测置信度阈值
    Returns:
        heatmaps: (B, 1, feat_h, feat_w) ∈ [0, 1]
    """
    b = images.shape[0]
    device = images.device

    outputs = gdino(images, captions=[text_prompt] * b)
    pred_logits = outputs["pred_logits"].sigmoid()  # (B, nq, num_classes)
    pred_boxes = outputs["pred_boxes"]               # (B, nq, 4) cxcywh

    heatmaps = []
    for i in range(b):
        scores = pred_logits[i].max(dim=-1)[0]  # (nq,)
        mask = scores > box_threshold
        if mask.sum() == 0:
            heatmaps.append(torch.full((1, feat_h, feat_w), 0.1, device=device))
            continue

        boxes = pred_boxes[i][mask]    # (n, 4) cxcywh normalized
        confs = scores[mask]           # (n,)

        hm = torch.zeros(1, feat_h, feat_w, device=device)
        gy, gx = torch.meshgrid(
            torch.linspace(0, 1, feat_h, device=device),
            torch.linspace(0, 1, feat_w, device=device),
            indexing='ij'
        )
        for box, conf in zip(boxes, confs):
            cx, cy, bw, bh = box
            sigma_x = bw / 4 + 0.02
            sigma_y = bh / 4 + 0.02
            gauss = torch.exp(
                -((gx - cx)**2 / (2 * sigma_x**2) +
                  (gy - cy)**2 / (2 * sigma_y**2))
            )
            hm[0] += conf * gauss

        hm = hm.clamp(0, 1)
        heatmaps.append(hm)

    return torch.stack(heatmaps, dim=0)  # (B, 1, feat_h, feat_w)


def main():
    parser = argparse.ArgumentParser(
        description="Extract GDINO heatmaps + Swin-T features offline")
    parser.add_argument("--data_dir", type=str, default="datasets_lerobot")
    parser.add_argument("--repo_id", type=str, default="local/mr2010-welding")
    parser.add_argument("--output_dir", type=str, default="gdino_heatmaps")
    parser.add_argument("--gdino_checkpoint", type=str,
                        default="models/groundingdino_swint_ogc.pth")
    parser.add_argument("--prompt", type=str, default="object.",
                        help="GDINO text prompt (must end with '.')")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resize_shape", type=int, nargs=2, default=[240, 320])
    parser.add_argument("--crop_shape", type=int, nargs=2, default=[224, 308])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_keys", type=str, nargs="+",
                        default=["observation.images.overhead", "observation.images.ee"])
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.prompt.endswith("."):
        args.prompt = args.prompt + "."

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GDINO Heatmap + Feature Extraction (memmap)")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image keys: {args.image_keys}")

    # 加载数据集（不需要 delta_timestamps，直接逐帧访问）
    print("\nLoading dataset...")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.data_dir,
        video_backend="torchcodec",
    )
    total_frames = dataset.num_frames
    print(f"  Total frames: {total_frames}")

    # 加载 GDINO
    print("\nLoading Grounding DINO...")
    gdino = load_gdino(args.gdino_checkpoint, str(device))
    print("  GDINO loaded successfully")

    # 计算 feature map 尺寸（通过 ResNet18 实际跑一次获取精确值）
    print("\nComputing feature map size...")
    import torchvision
    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.ops.misc import FrozenBatchNorm2d

    _backbone = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        norm_layer=FrozenBatchNorm2d,
    )
    _backbone = IntermediateLayerGetter(_backbone, return_layers={"layer4": "feature_map"})
    _backbone.eval().to(device)
    ch, cw = args.crop_shape
    _dummy = torch.randn(1, 3, ch, cw, device=device)
    with torch.no_grad():
        _feat = _backbone(_dummy)["feature_map"]
    feat_h, feat_w = _feat.shape[2], _feat.shape[3]
    del _backbone, _dummy, _feat
    torch.cuda.empty_cache()
    print(f"  ResNet feature map size: {feat_h} x {feat_w}")

    # 验证 Swin-T 特征维度
    swin_dim = 768
    print(f"  Swin-T last layer: {swin_dim} channels")
    print(f"  Swin-T spatial features will be pooled to: {feat_h} x {feat_w}")

    # 创建 DataLoader（逐帧访问，不使用 sampler）
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # 对每个 image key 分别提取热图 + 特征，使用 memmap 流式写入
    for img_key in args.image_keys:
        print(f"\n{'=' * 60}")
        print(f"Extracting for: {img_key}")
        print(f"{'=' * 60}")

        key_dir = output_dir / img_key.replace(".", "_")
        key_dir.mkdir(parents=True, exist_ok=True)

        # 检查是否已完成 (meta 标记)
        done_flag = key_dir / "_done"
        if done_flag.exists():
            print(f"  Already extracted (found _done flag), skipping.")
            continue

        # 创建 memmap 文件: 预分配全部空间，按 index 随机写入
        hm_mmap = np.memmap(
            str(key_dir / "heatmaps.bin"), dtype=np.float16, mode='w+',
            shape=(total_frames, 1, feat_h, feat_w))
        feat_mmap = np.memmap(
            str(key_dir / "features.bin"), dtype=np.float16, mode='w+',
            shape=(total_frames, swin_dim))
        sfeat_mmap = np.memmap(
            str(key_dir / "spatial_features.bin"), dtype=np.float16, mode='w+',
            shape=(total_frames, swin_dim, feat_h, feat_w))

        hm_size = total_frames * 1 * feat_h * feat_w * 2 / 1024 / 1024
        feat_size = total_frames * swin_dim * 2 / 1024 / 1024
        sfeat_size = total_frames * swin_dim * feat_h * feat_w * 2 / 1024 / 1024
        print(f"  Memmap files created:")
        print(f"    heatmaps.bin:         {hm_size:.1f} MB  shape=({total_frames}, 1, {feat_h}, {feat_w})")
        print(f"    features.bin:         {feat_size:.1f} MB  shape=({total_frames}, {swin_dim})")
        print(f"    spatial_features.bin: {sfeat_size:.1f} MB  shape=({total_frames}, {swin_dim}, {feat_h}, {feat_w})")

        n_detected = 0
        n_total = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=img_key)):
            images = batch[img_key]  # (B, C, H, W)
            indices = batch["index"]  # (B, 1) or (B,)
            if indices.dim() > 1:
                indices = indices.squeeze(-1)

            # Resize + center crop (与 ResNetObserver 一致，但用 center crop)
            if images.shape[-2:] != tuple(args.resize_shape):
                images = F.interpolate(
                    images.float(), size=tuple(args.resize_shape),
                    mode='bilinear', align_corners=False
                )
            rh, rw = args.resize_shape
            top = (rh - ch) // 2
            left = (rw - cw) // 2
            images = images[..., top:top + ch, left:left + cw]
            images = images.to(device)

            # === 1. Swin-T 特征提取 ===
            global_feat, spatial_feat = extract_swin_features(
                gdino, images, feat_h, feat_w)

            # === 2. GDINO 检测 → 热图 ===
            heatmaps = generate_heatmap(
                gdino, images, args.prompt, feat_h, feat_w,
                box_threshold=args.box_threshold
            )

            # 转为 numpy half 并直接写入 memmap (零内存积压)
            hm_np = heatmaps.cpu().numpy().astype(np.float16)
            feat_np = global_feat.cpu().numpy().astype(np.float16)
            sfeat_np = spatial_feat.cpu().numpy().astype(np.float16)
            idx_np = indices.numpy()

            for i in range(len(idx_np)):
                idx = int(idx_np[i])
                hm_mmap[idx] = hm_np[i]
                feat_mmap[idx] = feat_np[i]
                sfeat_mmap[idx] = sfeat_np[i]
                if hm_np[i].max() > 0.15:
                    n_detected += 1
                n_total += 1

            # 每 500 batch flush 一次确保数据落盘
            if (batch_idx + 1) % 500 == 0:
                hm_mmap.flush()
                feat_mmap.flush()
                sfeat_mmap.flush()

        # 最终 flush
        hm_mmap.flush()
        feat_mmap.flush()
        sfeat_mmap.flush()
        del hm_mmap, feat_mmap, sfeat_mmap

        elapsed = time.time() - start_time
        detect_rate = n_detected / max(n_total, 1) * 100

        print(f"\n  Frames processed: {n_total}")
        print(f"  Detection rate: {detect_rate:.1f}%")
        print(f"  Time: {elapsed:.1f}s ({n_total / elapsed:.1f} frames/s)")

        # 写入完成标记
        done_flag.write_text(f"frames={n_total}\n")

    # 保存 meta 信息
    meta = {
        "format": "memmap",
        "dtype": "float16",
        "prompt": args.prompt,
        "box_threshold": args.box_threshold,
        "resize_shape": args.resize_shape,
        "crop_shape": args.crop_shape,
        "feat_h": feat_h,
        "feat_w": feat_w,
        "swin_feat_dim": swin_dim,
        "total_frames": total_frames,
        "image_keys": args.image_keys,
        "gdino_checkpoint": args.gdino_checkpoint,
        "shapes": {
            "heatmaps": [total_frames, 1, feat_h, feat_w],
            "features": [total_frames, swin_dim],
            "spatial_features": [total_frames, swin_dim, feat_h, feat_w],
        },
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! Memmap files saved to {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
