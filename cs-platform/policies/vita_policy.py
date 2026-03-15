"""
VITA Policy — Vision-to-Action Flow Matching Policy

从 VITA (Gao et al., ICLR 2026) 提取核心模块，适配 LeRobot 数据格式。

核心思想: 直接从视觉 latent 流向动作 latent (无噪声起点, 无条件注入)，
天然支持多模态动作分布，解决 ACT temporal ensemble 的 action collapse 问题。

依赖: pip install torchcfm timm einops
"""

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policies import register_policy
from policies.base_policy import BasePolicy


# ============================================================
# 核心组件 (提取自 VITA, 适配自包含)
# ============================================================

class SinusoidalPosEmbed(nn.Module):
    """Sinusoidal positional embedding for timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class FlowNetLayer(nn.Module):
    """Flow network layer with time-conditioned modulation."""
    def __init__(self, dim, mlp_ratio=4., dropout=0.0):
        super().__init__()
        from timm.layers import Mlp
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=dropout,
        )
        self.time_modulator = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.dim = dim
        nn.init.constant_(self.time_modulator[-1].weight, 0)
        nn.init.constant_(self.time_modulator[-1].bias, 0)

    def forward(self, x, t):
        gamma, scale, shift = self.time_modulator(t).view(x.shape[0], 3, self.dim).unbind(1)
        x_norm = self.norm(x).mul(scale.add(1)).add_(shift)
        return x + self.mlp(x_norm).mul_(gamma)


class SimpleFlowNet(nn.Module):
    """MLP flow network with time embedding — predicts velocity field."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 mlp_ratio=4.0, dropout=0.0, time_embed_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmbed(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )
        self.layers = nn.ModuleList([
            FlowNetLayer(hidden_dim, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)

    def forward(self, x, t):
        x = self.input_proj(x)
        t = self.time_embed(t)
        for block in self.layers:
            x = block(x, t)
        return self.out_proj(self.norm(x))


class FlowMatcher:
    """OT-CFM Flow Matcher — 训练和采样。"""
    def __init__(self, sigma=0.0, num_sampling_steps=6):
        import torchcfm.conditional_flow_matching as cfm
        self.fm = cfm.ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        self.num_sampling_steps = num_sampling_steps

    def compute_loss(self, model, target, start=None):
        x0 = torch.randn_like(target) if start is None else start
        timestep, xt, ut = self.fm.sample_location_and_conditional_flow(x0, target)
        vt = model(xt, timestep)
        loss = torch.mean((vt - ut) ** 2)
        return loss, {"flow_loss": loss.item()}

    def sample(self, model, shape, device, num_steps=None, start=None):
        if num_steps is None:
            num_steps = self.num_sampling_steps
        x = torch.randn(shape, device=device) if start is None else start
        dt = 1.0 / num_steps
        for t in range(num_steps):
            timestep = torch.ones(x.shape[0], device=device) * (t / num_steps)
            x = x + model(x, timestep) * dt
        return x


# ============================================================
# GDINO Distillation + Offline Heatmap Store
# ============================================================


class GDINODistillation(nn.Module):
    """Grounding DINO 空间注意力蒸馏模块.

    训练时：Grounding DINO (frozen) 检测目标物体 → 生成空间注意力热图 →
            ResNet feature_map 通过 attn_head 预测该热图 → BCE loss.
    推理时：完全不需要 GDINO，零额外推理开销.

    方案A：注意力热图蒸馏 — GDINO 的检测结果转化为高斯热图，
           ResNet 学习预测"哪些空间区域包含可操作目标"。
    """
    def __init__(self, gdino_config, gdino_checkpoint, resnet_feat_dim=512,
                 text_prompt="object", device="cpu", subsample_k=4):
        super().__init__()
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.models import build_model
        from groundingdino.util.utils import clean_state_dict
        import os as _os

        # 设置离线模式，避免联网下载 BERT
        _os.environ.setdefault('HF_HUB_OFFLINE', '1')
        _os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

        # 加载配置，将 text_encoder_type 替换为本地 BERT 路径
        args = SLConfig.fromfile(gdino_config)
        args.device = device
        local_bert = _os.path.expanduser(
            '~/.cache/huggingface/hub/models--bert-base-uncased')
        if _os.path.isdir(local_bert):
            args.text_encoder_type = local_bert

        # 构建模型并加载权重
        self.gdino = build_model(args)
        ckpt = torch.load(gdino_checkpoint, map_location="cpu")
        self.gdino.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
        self.gdino.eval()
        for p in self.gdino.parameters():
            p.requires_grad = False
        self.gdino.eval()

        # ResNet feature_map → 单通道空间注意力预测
        self.attn_head = nn.Sequential(
            nn.Conv2d(resnet_feat_dim, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        # GDINO 要求 caption 以 "." 结尾，否则 tokenizer 报错
        self.text_prompt = text_prompt if text_prompt.endswith(".") else text_prompt + "."
        self.box_threshold = 0.25
        self.subsample_k = subsample_k  # 每 batch 随机取 k 张过 GDINO

    @torch.no_grad()
    def get_attention_heatmap(self, images, feat_h, feat_w):
        """用 GDINO 检测目标，生成高斯注意力热图.

        Args:
            images: (K, C, H, W) 原始图像 (sub-sampled)
            feat_h, feat_w: ResNet feature_map 的空间尺寸
        Returns:
            heatmap: (K, 1, feat_h, feat_w) ∈ [0, 1]
        """
        self.gdino.eval()
        k, c, h, w = images.shape
        device = images.device

        # GDINO 推理
        outputs = self.gdino(images, captions=[self.text_prompt] * k)
        pred_logits = outputs["pred_logits"].sigmoid()  # (K, nq, num_classes)
        pred_boxes = outputs["pred_boxes"]               # (K, nq, 4) cxcywh normalized

        # 生成高斯热图
        heatmaps = []
        for i in range(k):
            scores = pred_logits[i].max(dim=-1)[0]  # (nq,)
            mask = scores > self.box_threshold
            if mask.sum() == 0:
                # 没检测到物体，用均匀热图 (弱监督)
                heatmaps.append(torch.full((1, feat_h, feat_w), 0.1, device=device))
                continue

            boxes = pred_boxes[i][mask]    # (n, 4) cxcywh
            confs = scores[mask]           # (n,)

            # 在 feature_map 坐标系上生成高斯
            hm = torch.zeros(1, feat_h, feat_w, device=device)
            gy, gx = torch.meshgrid(
                torch.linspace(0, 1, feat_h, device=device),
                torch.linspace(0, 1, feat_w, device=device),
                indexing='ij'
            )
            for box, conf in zip(boxes, confs):
                cx, cy, bw, bh = box
                sigma_x = bw / 4 + 0.02  # 高斯宽度 ∝ box 宽
                sigma_y = bh / 4 + 0.02
                gauss = torch.exp(
                    -((gx - cx)**2 / (2 * sigma_x**2) +
                      (gy - cy)**2 / (2 * sigma_y**2))
                )
                hm[0] += conf * gauss

            hm = hm.clamp(0, 1)
            heatmaps.append(hm)

        return torch.stack(heatmaps, dim=0)  # (K, 1, feat_h, feat_w)

    def forward(self, resnet_feat_map, heatmap_target):
        """计算蒸馏 loss.

        Args:
            resnet_feat_map: (K, 512, H_r, W_r) sub-sampled ResNet feature_map
            heatmap_target: (K, 1, H_r, W_r) GDINO 生成的热图
        Returns:
            distill_loss: scalar
        """
        pred_attn = self.attn_head(resnet_feat_map)  # (K, 1, H_r, W_r)
        distill_loss = F.binary_cross_entropy_with_logits(pred_attn, heatmap_target)
        return distill_loss


class OfflineHeatmapStore:
    """离线热图 + 语义特征加载器 (memmap 模式).

    使用 numpy memmap 按需读取，内存占用极低 (< 10MB)。
    数据按 frame index 直接索引，无需加载整个文件。

    文件格式:
      - heatmaps.bin:         (N, 1, feat_h, feat_w) float16
      - spatial_features.bin: (N, 768, feat_h, feat_w) float16
      - features.bin:         (N, 768) float16
      - meta.json:            包含 shapes 信息
    """
    def __init__(self, heatmap_dir, image_keys):
        self.heatmaps = {}          # {image_key: np.memmap}
        self.spatial_features = {}  # {image_key: np.memmap}
        heatmap_dir = Path(heatmap_dir)

        # 读取 meta 获取 shape 信息
        meta_path = heatmap_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {heatmap_dir}")
        import json
        with open(meta_path) as f:
            meta = json.load(f)

        total_frames = meta["total_frames"]
        feat_h = meta["feat_h"]
        feat_w = meta["feat_w"]
        swin_dim = meta.get("swin_feat_dim", 768)

        for img_key in image_keys:
            key_dir = heatmap_dir / img_key.replace(".", "_")
            # 加载热图 memmap (只读)
            hm_path = key_dir / "heatmaps.bin"
            if hm_path.exists():
                print(f"  Loading offline heatmaps (memmap): {hm_path}")
                self.heatmaps[img_key] = np.memmap(
                    str(hm_path), dtype=np.float16, mode='r',
                    shape=(total_frames, 1, feat_h, feat_w))
                print(f"    Shape: ({total_frames}, 1, {feat_h}, {feat_w})")
            else:
                print(f"  WARNING: heatmap file not found: {hm_path}")
            # 加载空间语义特征 memmap (只读)
            sfeat_path = key_dir / "spatial_features.bin"
            if sfeat_path.exists():
                print(f"  Loading offline spatial features (memmap): {sfeat_path}")
                self.spatial_features[img_key] = np.memmap(
                    str(sfeat_path), dtype=np.float16, mode='r',
                    shape=(total_frames, swin_dim, feat_h, feat_w))
                print(f"    Shape: ({total_frames}, {swin_dim}, {feat_h}, {feat_w})")

    def get(self, image_key, indices, device):
        """按 frame index 获取热图 batch.

        Returns:
            (valid_indices, heatmaps): valid_indices 是 batch 内有效的位置 list,
            heatmaps 是 (K, 1, feat_h, feat_w) float32. 若全部缺失返回 ([], None).
        """
        if image_key not in self.heatmaps:
            return [], None
        mmap = self.heatmaps[image_key]
        valid_pos = []
        hms = []
        for pos, idx in enumerate(indices):
            idx = int(idx)
            if 0 <= idx < mmap.shape[0]:
                valid_pos.append(pos)
                hms.append(torch.from_numpy(mmap[idx].copy()).float())
        if not hms:
            return [], None
        return valid_pos, torch.stack(hms, dim=0).to(device)

    def get_spatial_features(self, image_key, indices, device):
        """按 frame index 获取空间语义特征 batch.

        Returns:
            (valid_indices, features): features 是 (K, 768, feat_h, feat_w) float32.
        """
        if image_key not in self.spatial_features:
            return [], None
        mmap = self.spatial_features[image_key]
        valid_pos = []
        feats = []
        for pos, idx in enumerate(indices):
            idx = int(idx)
            if 0 <= idx < mmap.shape[0]:
                valid_pos.append(pos)
                feats.append(torch.from_numpy(mmap[idx].copy()).float())
        if not feats:
            return [], None
        return valid_pos, torch.stack(feats, dim=0).to(device)




class CNNActionEncoder(nn.Module):
    """1D CNN encoder: (B, pred_horizon, action_dim) -> (B, latent_dim)."""
    def __init__(self, pred_horizon, action_dim, latent_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = []
        in_dim = action_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            layers.extend([nn.Conv1d(in_dim, out_dim, 5, stride=2, padding=2), nn.ReLU()])
            in_dim = out_dim
        self.encoder = nn.Sequential(*layers)
        conv_len = pred_horizon // (2 ** num_layers)
        self.latent_proj = nn.Linear(hidden_dim * conv_len, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, actions):
        # actions: (B, pred_horizon, action_dim)
        x = actions.transpose(1, 2)  # (B, action_dim, pred_horizon)
        x = self.encoder(x)
        return self.latent_proj(x.flatten(1))


class SimpleActionDecoder(nn.Module):
    """MLP decoder: (B, latent_dim) -> (B, pred_horizon, action_dim)."""
    def __init__(self, latent_dim, dec_hidden_dim, pred_horizon, action_dim,
                 num_layers=4, dropout=0.0):
        super().__init__()
        from timm.layers import Mlp
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.input_proj = nn.Linear(latent_dim, dec_hidden_dim)
        self.layers = nn.ModuleList([
            Mlp(dec_hidden_dim, dec_hidden_dim, dec_hidden_dim, bias=True, drop=dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(dec_hidden_dim, pred_horizon * action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        x = self.input_proj(z)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x).view(-1, self.pred_horizon, self.action_dim)


class FlowAwareSpatialAttention(nn.Module):
    """FASA — Flow-Aware Spatial Attention.

    自监督空间注意力模块，利用帧间特征变化自动发现交互区域。
    核心洞察: 图像中变化最显著的区域 = 交互正在发生的区域 = 最需要关注的区域。

    完整模式 (有 prev feat_map):
      1. 变化图: |feat_curr - feat_prev| → change_map → 自监督注意力
      2. 动态区域加权池化 — 聚焦变化(交互)区域的特征 (512d)
      3. 静态区域加权池化 — 保留稳定参考上下文 (512d)
      4. 变化质心 + 方向: (cx, cy, dx, dy) — 变化在哪 + 变化趋势 (4d)
      5. Spatial Softmax 坐标 → 压缩 (coord_dim d) — 各特征位置

    降级模式 (无 prev feat_map, obs_horizon=1):
      1. Global AvgPool (512d) + Spatial Softmax (coord_dim d)
    """

    def __init__(self, feat_dim=512, coord_dim=32):
        super().__init__()
        self.feat_dim = feat_dim
        self.coord_dim = coord_dim
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 变化图精炼: 多尺度 Conv + 通道注意力，从原始变化图中提取任务相关的变化模式
        # 输入: 原始变化图(1ch) + 通道最大值图(1ch) + 通道标准差图(1ch) = 3ch
        self.change_refine = nn.Sequential(
            # 多尺度特征提取
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 通道压缩
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # Spatial Softmax 坐标压缩
        self.coord_proj = nn.Linear(feat_dim * 2, coord_dim)

        # 完整模式输出: dynamic(512) + static(512) + change_coords(4) + ssm_coords(coord_dim)
        self.output_dim = feat_dim * 2 + 4 + coord_dim  # 1060

    @property
    def output_dim_full(self):
        return self.full_dim

    @property
    def output_dim_fallback(self):
        return self.fallback_dim

    def _spatial_softmax(self, feat_map):
        """Per-channel spatial softmax → expected (x, y) coordinates.

        Args: feat_map (B, C, H, W)
        Returns: (B, coord_dim)
        """
        B, C, H, W = feat_map.shape
        # spatial softmax per channel
        logits = feat_map.reshape(B, C, -1)             # (B, C, H*W)
        weights = F.softmax(logits, dim=2)               # (B, C, H*W)
        # grid coords
        gy = torch.linspace(0, 1, H, device=feat_map.device)
        gx = torch.linspace(0, 1, W, device=feat_map.device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        grid_x_flat = grid_x.reshape(-1)                 # (H*W,)
        grid_y_flat = grid_y.reshape(-1)                 # (H*W,)
        # expected coordinates per channel
        exp_x = (weights * grid_x_flat).sum(dim=2)       # (B, C)
        exp_y = (weights * grid_y_flat).sum(dim=2)       # (B, C)
        coords = torch.cat([exp_x, exp_y], dim=1)        # (B, C*2)
        return self.coord_proj(coords)                    # (B, coord_dim)

    def _compute_change_attention(self, feat_curr, feat_prev):
        """计算帧间变化注意力图.

        Args:
            feat_curr: (B, C, H, W)
            feat_prev: (B, C, H, W)
        Returns:
            change_attn: (B, 1, H, W) 归一化的变化注意力权重
            change_map_raw: (B, 1, H, W) 精炼后的变化图 (for direction compute)
        """
        # 帧间特征差的 L1 范数 → 多统计量描述变化模式
        delta = (feat_curr - feat_prev).abs()             # (B, C, H, W)
        change_mean = delta.mean(dim=1, keepdim=True)     # (B, 1, H, W) 平均变化量
        change_max = delta.max(dim=1, keepdim=True)[0]    # (B, 1, H, W) 最大变化通道
        change_std = delta.std(dim=1, keepdim=True)       # (B, 1, H, W) 变化一致性
        change_input = torch.cat([change_mean, change_max, change_std], dim=1)  # (B, 3, H, W)

        # 多尺度 Conv 精炼: 学习任务相关的变化模式
        change_refined = self.change_refine(change_input)  # (B, 1, H, W)
        change_refined = torch.sigmoid(change_refined)     # 归一化到 [0, 1]

        # 归一化为权重 (sum=1)
        change_attn = change_refined / (change_refined.sum(dim=[2, 3], keepdim=True) + 1e-8)

        return change_attn, change_refined

    def _compute_change_coords(self, change_attn, feat_curr, feat_prev):
        """计算变化质心坐标 + 变化方向向量.

        Args:
            change_attn: (B, 1, H, W) 归一化变化权重
            feat_curr, feat_prev: (B, C, H, W)
        Returns:
            (B, 4) — [cx, cy, dx, dy]
            cx, cy: 变化区域的质心 (归一化到 [0,1])
            dx, dy: 帧间变化方向 (质心偏移量)
        """
        B, _, H, W = change_attn.shape
        gy = torch.linspace(0, 1, H, device=change_attn.device)
        gx = torch.linspace(0, 1, W, device=change_attn.device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')

        attn_flat = change_attn.squeeze(1)                # (B, H, W)

        # 当前帧变化质心
        cx = (attn_flat * grid_x).sum(dim=[1, 2])         # (B,)
        cy = (attn_flat * grid_y).sum(dim=[1, 2])         # (B,)

        # 变化方向: 用当前帧和前一帧各自的特征激活质心的差
        # 计算前一帧的特征激活质心 (用 feat magnitude 做权重)
        prev_mag = feat_prev.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
        prev_w = prev_mag / (prev_mag.sum(dim=[2, 3], keepdim=True) + 1e-8)
        prev_w_flat = prev_w.squeeze(1)
        prev_cx = (prev_w_flat * grid_x).sum(dim=[1, 2])
        prev_cy = (prev_w_flat * grid_y).sum(dim=[1, 2])

        curr_mag = feat_curr.abs().mean(dim=1, keepdim=True)
        curr_w = curr_mag / (curr_mag.sum(dim=[2, 3], keepdim=True) + 1e-8)
        curr_w_flat = curr_w.squeeze(1)
        curr_cx = (curr_w_flat * grid_x).sum(dim=[1, 2])
        curr_cy = (curr_w_flat * grid_y).sum(dim=[1, 2])

        dx = curr_cx - prev_cx  # 特征质心 x 方向偏移
        dy = curr_cy - prev_cy  # 特征质心 y 方向偏移

        return torch.stack([cx, cy, dx, dy], dim=1)       # (B, 4)

    def forward(self, feat_curr, feat_prev=None):
        """
        Args:
            feat_curr: (B, C, H, W) 当前帧特征图
            feat_prev: (B, C, H, W) 前一帧特征图, None 时降级(零填充变化相关部分)
        Returns:
            (B, output_dim) — 始终输出统一维度 = 512 + 512 + 4 + coord_dim
        """
        B = feat_curr.shape[0]
        # Spatial Softmax 坐标 (两种模式都用)
        ssm_coords = self._spatial_softmax(feat_curr)     # (B, coord_dim)

        if feat_prev is None:
            # === 降级模式: Global Pool 做 dynamic + static, 变化坐标为零 ===
            global_feat = self.pool(feat_curr).flatten(1)  # (B, 512)
            zero_coords = torch.zeros(B, 4, device=feat_curr.device)
            return torch.cat([global_feat, global_feat, zero_coords, ssm_coords], dim=1)

        # === 完整模式: 帧间变化感知 ===
        change_attn, change_refined = self._compute_change_attention(
            feat_curr, feat_prev)                          # (B, 1, H, W)

        # 1. 动态区域加权池化 — 聚焦变化(交互)区域
        dynamic_feat = (feat_curr * change_attn).sum(dim=[2, 3])   # (B, 512)

        # 2. 静态区域加权池化 — 保留稳定参考上下文
        stable_attn = 1.0 - change_attn
        stable_attn = stable_attn / (stable_attn.sum(dim=[2, 3], keepdim=True) + 1e-8)
        static_feat = (feat_curr * stable_attn).sum(dim=[2, 3])    # (B, 512)

        # 3. 变化质心 + 方向编码
        change_coords = self._compute_change_coords(
            change_attn, feat_curr, feat_prev)             # (B, 4)

        return torch.cat([dynamic_feat, static_feat, change_coords, ssm_coords], dim=1)


class ResNetObserver(nn.Module):
    """ResNet18 image feature extractor + state concatenation.

    支持特征差分模式 (temporal_diff=True):
      obs_horizon=2 时输出 [f_t, Δf, state_t, Δstate] 而非 flatten [f_{t-1}, f_t, ...]
      Δf = f_t - f_{t-1} 编码特征空间的"语义速度"
    """
    def __init__(self, state_dim, image_keys, obs_horizon=1, resize_shape=(240, 320),
                 crop_shape=(224, 308), temporal_diff=False, use_spatial_attn=False):
        super().__init__()
        import torchvision
        from torchvision.models._utils import IntermediateLayerGetter
        from torchvision.ops.misc import FrozenBatchNorm2d

        self.image_keys = image_keys
        self.state_dim = state_dim
        self.obs_horizon = obs_horizon
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.temporal_diff = temporal_diff and obs_horizon >= 2
        self.use_spatial_attn = use_spatial_attn

        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # FASA 空间注意力: 每个相机一个 FlowAwareSpatialAttention 模块
        if use_spatial_attn:
            self.fasa_modules = nn.ModuleList([
                FlowAwareSpatialAttention(feat_dim=512, coord_dim=32) for _ in image_keys
            ])
            self._per_cam_dim = self.fasa_modules[0].output_dim  # 1060
            # 推理时维护前一帧 feat_map 的缓存 (per camera)
            self._prev_feat_maps = None
        else:
            self._per_cam_dim = 512

    @property
    def output_dim(self):
        n_img_feat = len(self.image_keys) * self._per_cam_dim
        if self.temporal_diff:
            # f_t + Δf + state_t + Δstate
            return n_img_feat * 2 + self.state_dim * 2
        return (n_img_feat + self.state_dim) * self.obs_horizon

    @property
    def frame_feature_dim(self):
        """单帧 raw 特征维度 (N*per_cam_dim + state_dim)，用于 dynamics 跨帧预测."""
        return len(self.image_keys) * self._per_cam_dim + self.state_dim

    def _extract_features(self, images, b):
        """Run ResNet backbone on images, return pooled features."""
        if images.shape[-2:] != tuple(self.resize_shape):
            images = F.interpolate(images, size=tuple(self.resize_shape),
                                   mode='bilinear', align_corners=False)
        ch, cw = self.crop_shape
        _, _, H, W = images.shape
        if self.training:
            top = torch.randint(0, H - ch + 1, (1,)).item()
            left = torch.randint(0, W - cw + 1, (1,)).item()
        else:
            top = (H - ch) // 2
            left = (W - cw) // 2
        images = images[..., top:top + ch, left:left + cw]
        feat_map = self.backbone(images)["feature_map"]  # (B*N, 512, H', W')
        features = self.pool(feat_map)
        return features, feat_map

    def _pool_per_camera(self, feat_maps_curr, b, n, feat_maps_prev=None):
        """对每个相机的 feat_map 做池化，支持 FASA 空间注意力.

        Args:
            feat_maps_curr: (B*N, 512, H', W') — 当前帧 N个相机特征图
            b: batch size
            n: number of cameras
            feat_maps_prev: (B*N, 512, H', W') — 前一帧特征图, None 时 FASA 降级
        Returns:
            (B, N * per_cam_dim) — 每个相机的特征 concat
        """
        if self.use_spatial_attn:
            per_cam_curr = feat_maps_curr.reshape(b, n, *feat_maps_curr.shape[1:])
            per_cam_prev = None
            if feat_maps_prev is not None:
                per_cam_prev = feat_maps_prev.reshape(b, n, *feat_maps_prev.shape[1:])
            cam_feats = []
            for i in range(n):
                prev_i = per_cam_prev[:, i] if per_cam_prev is not None else None
                cam_feats.append(self.fasa_modules[i](per_cam_curr[:, i], prev_i))
            return torch.cat(cam_feats, dim=1)
        else:
            pooled = self.pool(feat_maps_curr).flatten(1)  # (B*N, 512)
            return pooled.reshape(b, -1)  # (B, N*512)

    def forward(self, batch):
        state = batch["observation.state"]
        b = state.shape[0]
        n = len(self.image_keys)
        img_list = [batch[k] for k in self.image_keys]
        sample = img_list[0]

        if self.temporal_diff and sample.dim() == 5:
            # === 特征差分模式 (obs_horizon >= 2) ===
            s = sample.shape[1]
            images = torch.stack(img_list, dim=2)  # (B, S, N, C, H, W)
            images_flat = images.reshape(b * s * n, *sample.shape[2:])

            _, all_feat_maps = self._extract_features(images_flat, b)
            # reshape → (B, S, N, 512, H', W')
            fmap_5d = all_feat_maps.reshape(b, s, n, *all_feat_maps.shape[1:])

            # FASA: 当前帧 + 前一帧参考
            frame_fmaps_curr = fmap_5d[:, -1].reshape(b * n, *all_feat_maps.shape[1:])
            frame_fmaps_prev = fmap_5d[:, -2].reshape(b * n, *all_feat_maps.shape[1:])

            if self.use_spatial_attn:
                # 当前帧用前一帧做 FASA 参考
                f_t = self._pool_per_camera(frame_fmaps_curr, b, n, frame_fmaps_prev)
                # 前一帧: 如果有更早帧则使用，否则 FASA 降级
                if s >= 3:
                    frame_fmaps_prev2 = fmap_5d[:, -3].reshape(b * n, *all_feat_maps.shape[1:])
                    f_prev = self._pool_per_camera(frame_fmaps_prev, b, n, frame_fmaps_prev2)
                else:
                    f_prev = self._pool_per_camera(frame_fmaps_prev, b, n, None)
            else:
                f_t = self._pool_per_camera(frame_fmaps_curr, b, n)
                f_prev = self._pool_per_camera(frame_fmaps_prev, b, n)

            delta_f = f_t - f_prev

            # state 差分
            state_t = state[:, -1, :]
            state_prev = state[:, -2, :]
            delta_state = state_t - state_prev

            # 缓存当前帧的 feature_map 和原始图像
            self._curr_feat_map = fmap_5d[:, -1, 0]      # (B, 512, H', W')
            self._curr_raw_images = img_list[0][:, -1]    # (B, C, H, W)

            # 缓存 t-1 帧的独立观测特征 (供 temporal 模块构建帧对)
            # f_prev 已包含 FASA 空间特征, state_prev 是 t-1 帧 state
            self._prev_frame_obs = torch.cat([f_prev, f_prev * 0, state_prev, state_prev * 0], dim=1).detach()

            return torch.cat([f_t, delta_f, state_t, delta_state], dim=1)

        # === 原始模式 (obs_horizon=1 或非 temporal_diff) ===
        state_flat = state.reshape(b, -1)

        if sample.dim() == 4:
            images = torch.stack(img_list, dim=1)
            images = images.reshape(b * n, *sample.shape[1:])
        elif sample.dim() == 5:
            s = sample.shape[1]
            images = torch.stack(img_list, dim=2)
            images = images.reshape(b * s * n, *sample.shape[2:])
        else:
            raise ValueError(f"Unexpected image dim: {sample.dim()}, shape: {sample.shape}")

        _, feat_maps = self._extract_features(images, b)

        if sample.dim() == 4:
            # obs_horizon=1: 推理时用缓存的前一帧做 FASA
            prev_fmaps = None
            if self.use_spatial_attn and self._prev_feat_maps is not None:
                prev_fmaps = self._prev_feat_maps
            features = self._pool_per_camera(feat_maps, b, n, prev_fmaps)
            # 更新前帧缓存 (推理时使用)
            if self.use_spatial_attn:
                self._prev_feat_maps = feat_maps.detach()
        else:
            # obs_horizon > 1 非 temporal_diff
            fmap_5d = feat_maps.reshape(b, -1, n, *feat_maps.shape[1:])  # (B, S, N, ...)
            s_steps = fmap_5d.shape[1]
            step_feats = []
            for t in range(s_steps):
                curr_fmaps = fmap_5d[:, t].reshape(b * n, *feat_maps.shape[1:])
                prev_fmaps = None
                if self.use_spatial_attn and t > 0:
                    prev_fmaps = fmap_5d[:, t - 1].reshape(b * n, *feat_maps.shape[1:])
                step_feats.append(self._pool_per_camera(curr_fmaps, b, n, prev_fmaps))
            features = torch.cat(step_feats, dim=1)

        # 缓存当前帧的 feature_map 和原始图像
        self._curr_feat_map = feat_maps[:b]
        if sample.dim() == 4:
            self._curr_raw_images = img_list[0]
        else:
            self._curr_raw_images = img_list[0][:, -1]

        return torch.cat([state_flat, features], dim=1)


# ============================================================
# Normalize / Unnormalize (与 LeRobot stats 兼容)
# ============================================================

class Normalize(nn.Module):
    def __init__(self, keys, stats):
        super().__init__()
        self.keys = keys
        for key in keys:
            if key in stats:
                mean = torch.tensor(stats[key]["mean"], dtype=torch.float32)
                std = torch.tensor(stats[key]["std"], dtype=torch.float32)
                self.register_buffer(f"mean_{key.replace('.', '_')}", mean)
                self.register_buffer(f"std_{key.replace('.', '_')}", std)

    @torch.no_grad()
    def forward(self, batch):
        batch = dict(batch)
        for key in self.keys:
            mean_key = f"mean_{key.replace('.', '_')}"
            std_key = f"std_{key.replace('.', '_')}"
            if hasattr(self, mean_key):
                mean = getattr(self, mean_key)
                std = getattr(self, std_key)
                batch[key] = (batch[key] - mean) / (std + 1e-8)
        return batch


class Unnormalize(nn.Module):
    def __init__(self, keys, stats):
        super().__init__()
        self.keys = keys
        for key in keys:
            if key in stats:
                mean = torch.tensor(stats[key]["mean"], dtype=torch.float32)
                std = torch.tensor(stats[key]["std"], dtype=torch.float32)
                self.register_buffer(f"mean_{key.replace('.', '_')}", mean)
                self.register_buffer(f"std_{key.replace('.', '_')}", std)

    @torch.no_grad()
    def forward(self, batch):
        batch = dict(batch)
        for key in self.keys:
            mean_key = f"mean_{key.replace('.', '_')}"
            std_key = f"std_{key.replace('.', '_')}"
            if hasattr(self, mean_key):
                mean = getattr(self, mean_key)
                std = getattr(self, std_key)
                batch[key] = batch[key] * std + mean
        return batch


# ============================================================
# VITA Policy Wrapper
# ============================================================

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, power=0.75):
        self.model = model
        self.power = power
        self.shadow = {}
        self.backup = {}
        self._step = 0

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        self._step += 1
        decay = 1 - (1 + self._step) ** -self.power
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1 - decay)

    def apply_shadow(self):
        """Apply EMA weights for inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original weights after inference."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "step": self._step, "power": self.power}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self._step = state_dict["step"]
        self.power = state_dict["power"]


@register_policy("vita")
class VITAPolicyWrapper(BasePolicy):

    def __init__(self):
        super().__init__()
        self._args = None
        self._action_queue = deque()
        self.ema = None
        self._obs_horizon = 1   # default, updated in build()
        self._image_keys = []   # default, updated in build()
        self._action_scale = 1.0  # default, updated in build()
        # Temporal ensemble (smooth action output)
        self._ensemble_buffer = None  # (chunk_size, action_dim) running weighted buffer
        self._temporal_ensemble_coeff = 0.01  # default, updated in build()
        # Sub-modules (set in build())
        self.observer = None
        self.obs_encoder = None
        self.flow_net = None
        self.flow_matcher = None
        self.action_encoder = None
        self.action_decoder = None
        self.normalize_inputs = None
        self.normalize_targets = None
        self.unnormalize_outputs = None
        self.gdino_distill = None  # GDINO 蒸馏模块

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("--chunk_size", type=int, default=16,
                            help="Prediction horizon (pred_horizon)")
        parser.add_argument("--action_horizon", type=int, default=8,
                            help="Number of actions to execute per inference")
        parser.add_argument("--latent_dim", type=int, default=512)
        parser.add_argument("--flow_hidden_dim", type=int, default=512)
        parser.add_argument("--flow_num_layers", type=int, default=4)
        parser.add_argument("--flow_mlp_ratio", type=float, default=4.0)
        parser.add_argument("--num_sampling_steps", type=int, default=6)
        parser.add_argument("--ae_hidden_dim", type=int, default=512)
        parser.add_argument("--ae_num_layers", type=int, default=4)
        parser.add_argument("--enc_recon_weight", type=float, default=0.5,
                            help="Action AE encoder reconstruction weight")
        parser.add_argument("--flow_recon_weight", type=float, default=0.5,
                            help="Flow-decoded action reconstruction weight")
        parser.add_argument("--consistency_weight", type=float, default=1.0,
                            help="Flow latent consistency loss weight")
        parser.add_argument("--obs_horizon", type=int, default=1,
                            help="Number of observation history frames (1=current only)")
        parser.add_argument("--action_scale", type=float, default=1.0,
                            help="Scale factor for action values (e.g. 1000 for ee_delta m->mm)")
        parser.add_argument("--resize_shape", type=int, nargs=2, default=[240, 320])
        parser.add_argument("--crop_shape", type=int, nargs=2, default=[224, 308])
        parser.add_argument("--temporal_ensemble_coeff", type=float, default=0.01,
                            help="Temporal ensemble exponential decay coefficient (smaller=smoother, 0=disable)")
        # Grounding DINO 蒸馏参数
        parser.add_argument("--gdino_checkpoint", type=str,
                            default="models/groundingdino_swint_ogc.pth",
                            help="Path to Grounding DINO checkpoint")
        parser.add_argument("--distill_weight", type=float, default=0.5,
                            help="Grounding DINO spatial distillation loss weight")
        parser.add_argument("--distill_prompt", type=str, default="object",
                            help="Text prompt for Grounding DINO (task-agnostic)")
        parser.add_argument("--distill_subsample", type=int, default=4,
                            help="Number of images sub-sampled per batch for GDINO")
        parser.add_argument("--use_gdino_distill", action="store_true",
                            help="Enable Grounding DINO spatial attention distillation")
        parser.add_argument("--heatmap_dir", type=str, default=None,
                            help="Path to pre-extracted GDINO heatmaps (offline distillation). "
                                 "If set, GDINO model is NOT loaded — zero extra GPU memory.")
        parser.add_argument("--feat_distill_weight", type=float, default=0.5,
                            help="Swin-T semantic feature distillation loss weight (cosine sim)")
        parser.add_argument("--use_spatial_attn", action="store_true",
                            help="Enable spatial attention pooling (uses pre-distilled attn_head)")

    def build(self, args, dataset_stats, input_features, output_features):
        self._args = args
        device = torch.device(args.device if hasattr(args, 'device') else
                              ("cuda" if torch.cuda.is_available() else "cpu"))

        # Determine dimensions from features
        state_dim = 0
        image_keys = []
        for k, v in input_features.items():
            if k == "observation.state":
                state_dim = v.shape[0]
            elif hasattr(v, 'type') and str(v.type) in ("FeatureType.VISUAL",):
                image_keys.append(k)

        # Fallback: scan features for image keys by name pattern
        if not image_keys:
            for k, v in input_features.items():
                if "image" in k.lower():
                    image_keys.append(k)

        action_dim = 0
        for k, v in output_features.items():
            action_dim = v.shape[0]
            break

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._image_keys = image_keys
        self._pred_horizon = args.chunk_size
        self._action_horizon = args.action_horizon
        self._obs_horizon = args.obs_horizon if hasattr(args, 'obs_horizon') else 1
        self._action_scale = args.action_scale if hasattr(args, 'action_scale') else 1.0
        self._use_temporal = False  # deprecated, kept for checkpoint compat
        self._temporal_ensemble_coeff = getattr(args, 'temporal_ensemble_coeff', 0.01)

        # 若 action_scale != 1，对 dataset_stats 中 action 的 mean/std 做相应缩放
        if self._action_scale != 1.0:
            import copy
            dataset_stats = copy.deepcopy(dataset_stats)
            if "action" in dataset_stats:
                s = self._action_scale
                dataset_stats["action"]["mean"] = dataset_stats["action"]["mean"] * s
                dataset_stats["action"]["std"] = dataset_stats["action"]["std"] * s
                dataset_stats["action"]["min"] = dataset_stats["action"]["min"] * s
                dataset_stats["action"]["max"] = dataset_stats["action"]["max"] * s

        # Build observer
        self._use_spatial_attn = getattr(args, 'use_spatial_attn', False)
        self.observer = ResNetObserver(
            state_dim=state_dim,
            image_keys=image_keys,
            obs_horizon=self._obs_horizon,
            resize_shape=tuple(args.resize_shape),
            crop_shape=tuple(args.crop_shape),
            temporal_diff=(self._obs_horizon >= 2),
            use_spatial_attn=self._use_spatial_attn,
        )
        obs_dim = self.observer.output_dim

        self.obs_encoder = nn.Linear(obs_dim, args.latent_dim)
        self.flow_net = SimpleFlowNet(
            input_dim=args.latent_dim,
            hidden_dim=args.flow_hidden_dim,
            output_dim=args.latent_dim,
            num_layers=args.flow_num_layers,
            mlp_ratio=args.flow_mlp_ratio,
        )
        self.flow_matcher = FlowMatcher(
            sigma=0.0,
            num_sampling_steps=args.num_sampling_steps,
        )

        # Action autoencoder
        self.action_encoder = CNNActionEncoder(
            pred_horizon=self._pred_horizon,
            action_dim=action_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.ae_hidden_dim,
        )
        self.action_decoder = SimpleActionDecoder(
            latent_dim=args.latent_dim,
            dec_hidden_dim=args.ae_hidden_dim,
            pred_horizon=self._pred_horizon,
            action_dim=action_dim,
            num_layers=args.ae_num_layers,
        )

        # Normalization — 仅对 state 做 z-score，图像已经是 [0,1] 不需要归一化
        norm_keys = ["observation.state"]
        self.normalize_inputs = Normalize(norm_keys, dataset_stats)
        self.normalize_targets = Normalize(["action"], dataset_stats)
        self.unnormalize_outputs = Unnormalize(["action"], dataset_stats)

        # Grounding DINO 蒸馏模块（仅训练时使用，推理时不需要）
        self._use_gdino_distill = getattr(args, 'use_gdino_distill', False)
        self._heatmap_store = None  # 离线热图加载器

        heatmap_dir = getattr(args, 'heatmap_dir', None)
        if self._use_gdino_distill and heatmap_dir is not None:
            # === 离线蒸馏模式：从 .pt 文件加载预提取热图 + 特征 ===
            print(f"  GDINO distillation: OFFLINE mode (heatmap_dir={heatmap_dir})")
            self._heatmap_store = OfflineHeatmapStore(heatmap_dir, image_keys)
            # attn_head: ResNet feat → 空间注意力预测
            self.gdino_distill = nn.Module()  # placeholder
            self.gdino_distill.attn_head = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1),
            )
            # feat_proj: ResNet 512-ch → Swin-T 768-ch 语义特征对齐
            self.gdino_distill.feat_proj = nn.Sequential(
                nn.Conv2d(512, 768, kernel_size=1, bias=False),
                nn.BatchNorm2d(768),
            )
        elif self._use_gdino_distill:
            # === 在线蒸馏模式：实时运行 GDINO ===
            import groundingdino
            import os
            gdino_config = os.path.join(
                os.path.dirname(groundingdino.__file__),
                "config", "GroundingDINO_SwinT_OGC.py"
            )
            gdino_path = getattr(args, 'gdino_checkpoint',
                                 'models/groundingdino_swint_ogc.pth')
            self.gdino_distill = GDINODistillation(
                gdino_config=gdino_config,
                gdino_checkpoint=gdino_path,
                resnet_feat_dim=512,
                text_prompt=getattr(args, 'distill_prompt', 'object'),
                device=str(device),
                subsample_k=getattr(args, 'distill_subsample', 4),
            )

        self.to(device)

        # 初始化 EMA
        self.ema = EMA(self, power=0.75)
        self.ema.register()

    @property
    def chunk_size(self):
        return self._args.chunk_size if self._args else 16

    def get_delta_timestamps(self, fps):
        timestamps = {
            "action": [i / fps for i in range(self.chunk_size)],
        }
        # 读取 obs_horizon（build() 前从 args 获取，build() 后用实例属性）
        obs_horizon = self._obs_horizon
        if obs_horizon <= 1 and self._args and hasattr(self._args, 'obs_horizon'):
            obs_horizon = self._args.obs_horizon
        # 添加观测历史时间戳（obs_horizon > 1 时）
        if obs_horizon > 1:
            obs_ts = [-(obs_horizon - 1 - i) / fps for i in range(obs_horizon)]
            # build() 前 _image_keys 为空，使用默认 key
            image_keys = self._image_keys or [
                "observation.images.overhead",
                "observation.images.ee",
            ]
            for key in image_keys:
                timestamps[key] = obs_ts
            timestamps["observation.state"] = obs_ts
        return timestamps

    def compute_loss(self, batch):
        # 缩放 action（如 ee_delta m -> mm）
        if self._action_scale != 1.0:
            batch["action"] = batch["action"] * self._action_scale
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Observation features -> latent
        obs_features = self.observer(batch)

        gt_actions = batch["action"]
        batch_size = obs_features.shape[0]

        # Encode ground-truth actions to latent
        action_latents = self.action_encoder(gt_actions)

        # Obs features -> latent
        obs_latents = self.obs_encoder(obs_features)

        # Flow matching loss
        flow_loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net, target=action_latents, start=obs_latents)
        loss = flow_loss

        # Encoder reconstruction loss (AE)
        enc_recon_weight = self._args.enc_recon_weight
        if enc_recon_weight > 0:
            actions_recon = self.action_decoder(action_latents)
            enc_recon_loss = F.l1_loss(actions_recon, gt_actions)
            loss = loss + enc_recon_weight * enc_recon_loss
            metrics["enc_recon_loss"] = enc_recon_loss.item()

        # Flow-decoded reconstruction loss (FLD)
        flow_recon_weight = self._args.flow_recon_weight
        consistency_weight = self._args.consistency_weight
        if flow_recon_weight > 0 or consistency_weight > 0:
            action_latents_pred = self.flow_matcher.sample(
                self.flow_net,
                shape=(batch_size, self._args.latent_dim),
                device=obs_latents.device,
                start=obs_latents,
            )

            if consistency_weight > 0:
                consistency_loss = F.mse_loss(action_latents_pred, action_latents)
                loss = loss + consistency_weight * consistency_loss
                metrics["consistency_loss"] = consistency_loss.item()

            if flow_recon_weight > 0:
                actions_flow_recon = self.action_decoder(action_latents_pred)
                flow_recon_loss = F.l1_loss(actions_flow_recon, gt_actions)
                loss = loss + flow_recon_weight * flow_recon_loss
                metrics["flow_recon_loss"] = flow_recon_loss.item()

        # Grounding DINO 空间注意力 + 语义特征蒸馏 loss
        if self._use_gdino_distill and self.gdino_distill is not None and self.training:
            feat_map = getattr(self.observer, '_curr_feat_map', None)
            if feat_map is not None:
                distill_w = getattr(self._args, 'distill_weight', 0.5)
                feat_distill_w = getattr(self._args, 'feat_distill_weight', 0.5)

                if self._heatmap_store is not None:
                    # === 离线蒸馏：从预提取热图 + 特征加载 ===
                    frame_indices = batch.get("index", None)
                    if frame_indices is not None:
                        if frame_indices.dim() > 1:
                            frame_indices = frame_indices.squeeze(-1)
                        img_key = self._image_keys[0] if self._image_keys else None
                        if img_key is not None:
                            idx_list = frame_indices.tolist()

                            # 1. 空间注意力蒸馏 (heatmap BCE loss)
                            valid_pos, heatmap = self._heatmap_store.get(
                                img_key, idx_list, feat_map.device)
                            if heatmap is not None and len(valid_pos) > 0:
                                sub_feat_map = feat_map[valid_pos]
                                fh, fw = sub_feat_map.shape[2], sub_feat_map.shape[3]
                                if heatmap.shape[2] != fh or heatmap.shape[3] != fw:
                                    heatmap = F.interpolate(
                                        heatmap, size=(fh, fw),
                                        mode='bilinear', align_corners=False)
                                pred_attn = self.gdino_distill.attn_head(sub_feat_map)
                                distill_loss = F.binary_cross_entropy_with_logits(
                                    pred_attn, heatmap)
                                loss = loss + distill_w * distill_loss
                                metrics["distill_loss"] = distill_loss.item()

                            # 2. 语义特征蒸馏 (cosine similarity loss)
                            if feat_distill_w > 0 and hasattr(self.gdino_distill, 'feat_proj'):
                                vp2, swin_feat = self._heatmap_store.get_spatial_features(
                                    img_key, idx_list, feat_map.device)
                                if swin_feat is not None and len(vp2) > 0:
                                    sub_feat_map2 = feat_map[vp2]
                                    fh2, fw2 = sub_feat_map2.shape[2], sub_feat_map2.shape[3]
                                    if swin_feat.shape[2] != fh2 or swin_feat.shape[3] != fw2:
                                        swin_feat = F.interpolate(
                                            swin_feat, size=(fh2, fw2),
                                            mode='bilinear', align_corners=False)
                                    # ResNet 512 → 768 对齐
                                    resnet_proj = self.gdino_distill.feat_proj(sub_feat_map2)
                                    # 逐像素 cosine similarity → 目标是 1.0
                                    cos_sim = F.cosine_similarity(
                                        resnet_proj, swin_feat, dim=1)  # (K, fh, fw)
                                    feat_loss = 1.0 - cos_sim.mean()
                                    loss = loss + feat_distill_w * feat_loss
                                    metrics["feat_distill_loss"] = feat_loss.item()
                else:
                    # === 在线蒸馏：实时运行 GDINO ===
                    raw_images = getattr(self.observer, '_curr_raw_images', None)
                    if raw_images is not None:
                        k = min(self.gdino_distill.subsample_k, feat_map.shape[0])
                        indices = torch.randperm(
                            feat_map.shape[0], device=feat_map.device)[:k]
                        sub_feat_map = feat_map[indices]
                        sub_images = raw_images[indices]
                        feat_h, feat_w = sub_feat_map.shape[2], sub_feat_map.shape[3]
                        heatmap = self.gdino_distill.get_attention_heatmap(
                            sub_images, feat_h, feat_w)
                        distill_loss = self.gdino_distill(sub_feat_map, heatmap)
                        loss = loss + distill_w * distill_loss
                        metrics["distill_loss"] = distill_loss.item()

        metrics["total_loss"] = loss.item()
        return loss, metrics

    @torch.no_grad()
    def _generate_actions(self, batch):
        """Generate action chunk from observation batch."""
        obs_features = self.observer(batch)
        obs_latents = self.obs_encoder(obs_features)

        action_latents_pred = self.flow_matcher.sample(
            self.flow_net,
            shape=(obs_latents.shape[0], self._args.latent_dim),
            device=obs_latents.device,
            start=obs_latents,
        )
        return self.action_decoder(action_latents_pred)

    def select_action(self, batch):
        self.eval()
        # Normalize inputs
        batch_norm = self.normalize_inputs(batch)

        # Generate full chunk prediction (chunk_size steps)
        actions = self._generate_actions(batch_norm)  # (1, chunk_size, action_dim)
        # Unnormalize
        actions = self.unnormalize_outputs({"action": actions})["action"]
        # 反缩放 action（如 mm -> m）
        if self._action_scale != 1.0:
            actions = actions / self._action_scale

        new_pred = actions[0]  # (chunk_size, action_dim)
        chunk_size = new_pred.shape[0]

        # Temporal ensemble: blend new prediction with existing buffer
        if self._temporal_ensemble_coeff > 0:
            if self._ensemble_buffer is None:
                # First call: initialize buffer with new prediction
                self._ensemble_buffer = new_pred.clone()
            else:
                # Shift buffer left by 1 (previous step's action was consumed)
                # remaining buffer covers positions [1:] from last step
                buf = self._ensemble_buffer[1:]  # (remaining, action_dim)
                remaining = buf.shape[0]

                # Build blended buffer of size chunk_size
                blended = new_pred.clone()
                if remaining > 0:
                    # Exponential decay weights: older predictions get less weight
                    coeff = self._temporal_ensemble_coeff
                    indices = torch.arange(remaining, device=new_pred.device, dtype=new_pred.dtype)
                    # w = exp(-coeff * i): weight for OLD prediction
                    # (1 - w): weight for NEW prediction
                    w = torch.exp(-coeff * indices).unsqueeze(-1)  # (remaining, 1)
                    blended[:remaining] = w * buf + (1 - w) * new_pred[:remaining]

                self._ensemble_buffer = blended
        else:
            # No temporal ensemble — use action_queue fallback
            if len(self._action_queue) == 0:
                for t in range(min(self._action_horizon, chunk_size)):
                    self._action_queue.append(actions[:, t])
            return self._action_queue.popleft()

        # Return current action (first element of buffer)
        current_action = self._ensemble_buffer[0:1].unsqueeze(0)  # (1, 1, action_dim) -> (1, action_dim)
        return current_action.squeeze(1)  # (1, action_dim)

    def reset(self):
        self._action_queue = deque()
        self._ensemble_buffer = None
        # 重置 FASA 前帧缓存
        if hasattr(self, 'observer') and self.observer is not None and hasattr(self.observer, '_prev_feat_maps'):
            self.observer._prev_feat_maps = None

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self._args.lr,
            betas=(0.95, 0.999),
            eps=1e-8,
            weight_decay=1e-6,
        )
        return optimizer

    def get_scheduler(self, optimizer, num_training_steps):
        """返回 cosine scheduler with warmup."""
        from diffusers.optimization import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps,
        )

    def load_pretrained_resnet(self, pretrain_path):
        """加载预蒸馏的 ResNet backbone + 蒸馏 head 权重.

        Args:
            pretrain_path: 预蒸馏 checkpoint 目录，包含 pretrain.pt
        """
        pretrain_path = Path(pretrain_path)
        if pretrain_path.is_symlink():
            pretrain_path = pretrain_path.resolve()
        pt_file = pretrain_path / "pretrain.pt"
        if not pt_file.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {pt_file}")

        print(f"  Loading pretrained ResNet weights from {pt_file}...")
        ckpt = torch.load(pt_file, map_location="cpu")

        # 1. 加载 backbone 权重
        backbone_state = ckpt["backbone"]
        missing, unexpected = self.observer.backbone.load_state_dict(
            backbone_state, strict=False)
        if missing:
            print(f"    backbone missing keys: {missing}")
        if unexpected:
            print(f"    backbone unexpected keys: {unexpected}")
        print(f"    Loaded backbone ({len(backbone_state)} params)")

        # 2. FASA 模块无需加载 attn_head — 它是自监督的，从 action loss 端到端学习
        # 如果启用了 use_spatial_attn (FASA), 其 change_refine 和 coord_proj 随训练学习
        if self._use_spatial_attn:
            print(f"    FASA modules: {len(self.observer.fasa_modules)} cameras — self-supervised, no pretrained weights needed")

        # 3. 加载 attn_head 到 gdino_distill（兼容旧模式）
        if "attn_head" in ckpt and hasattr(self, 'gdino_distill') and self.gdino_distill is not None and hasattr(self.gdino_distill, 'attn_head'):
            self.gdino_distill.attn_head.load_state_dict(ckpt["attn_head"])
            print(f"    Loaded attn_head into gdino_distill ({len(ckpt['attn_head'])} params)")

        # 4. 加载 feat_proj 权重
        if "feat_proj" in ckpt and hasattr(self, 'gdino_distill') and self.gdino_distill is not None and hasattr(self.gdino_distill, 'feat_proj'):
            self.gdino_distill.feat_proj.load_state_dict(ckpt["feat_proj"])
            print(f"    Loaded feat_proj ({len(ckpt['feat_proj'])} params)")

        print(f"    Pre-distillation step: {ckpt.get('step', 'unknown')}")
        print(f"    Backbone trainable — will fine-tune with policy")

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存时使用 EMA 权重
        if self.ema is not None:
            self.ema.apply_shadow()

        # 保存时排除 GDINO 权重（frozen，不需要保存）
        state = {k: v for k, v in self.state_dict().items()
                 if not k.startswith("gdino_distill.gdino")}
        torch.save(state, path / "model.pt")

        if self.ema is not None:
            torch.save(self.ema.state_dict(), path / "ema.pt")
            self.ema.restore()

        # Save config
        meta = {
            "policy_type": "vita",
            "chunk_size": self._args.chunk_size,
            "action_horizon": self._args.action_horizon,
            "latent_dim": self._args.latent_dim,
            "flow_hidden_dim": self._args.flow_hidden_dim,
            "flow_num_layers": self._args.flow_num_layers,
            "flow_mlp_ratio": self._args.flow_mlp_ratio,
            "num_sampling_steps": self._args.num_sampling_steps,
            "ae_hidden_dim": self._args.ae_hidden_dim,
            "ae_num_layers": self._args.ae_num_layers,
            "enc_recon_weight": self._args.enc_recon_weight,
            "flow_recon_weight": self._args.flow_recon_weight,
            "consistency_weight": self._args.consistency_weight,
            "resize_shape": list(self._args.resize_shape),
            "crop_shape": list(self._args.crop_shape),
            "obs_horizon": self._obs_horizon,
            "action_scale": self._action_scale,
            "state_dim": self._state_dim,
            "action_dim": self._action_dim,
            "image_keys": self._image_keys,
            "pred_horizon": self._pred_horizon,
            "use_temporal": False,
            "use_gdino_distill": self._use_gdino_distill,
            "temporal_ensemble_coeff": self._temporal_ensemble_coeff,
            "use_spatial_attn": self._use_spatial_attn,
        }
        with open(path / "policy_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Path, dataset_stats, device):
        path = Path(path)
        if path.is_symlink():
            path = path.resolve()

        with open(path / "policy_meta.json") as f:
            meta = json.load(f)

        # Reconstruct args namespace — 推理时禁用 GDINO 蒸馏（不需要）
        args = argparse.Namespace(**{
            "device": str(device),
            "lr": 1e-4,  # not used at inference
            "use_gdino_distill": False,  # 推理时不需要 GDINO
            **{k: v for k, v in meta.items()
               if k not in ("policy_type", "state_dim", "action_dim",
                             "image_keys", "pred_horizon", "use_gdino_distill")},
        })

        # Reconstruct features info (minimal)
        from types import SimpleNamespace
        input_features = {
            "observation.state": SimpleNamespace(shape=(meta["state_dim"],),
                                                 type="STATE"),
        }
        for ik in meta["image_keys"]:
            input_features[ik] = SimpleNamespace(shape=(3, 480, 640),
                                                  type="FeatureType.VISUAL")
        output_features = {
            "action": SimpleNamespace(shape=(meta["action_dim"],)),
        }

        policy = cls()
        policy.build(args, dataset_stats, input_features, output_features)
        policy.load_state_dict(torch.load(path / "model.pt", map_location=device),
                               strict=False)
        policy.to(device)
        policy.eval()
        return policy
