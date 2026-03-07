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
# 时间感知模块 (Self-Predictive Dynamics + Progress + Memory)
# ============================================================

class LatentDynamics(nn.Module):
    """自预测潜在动力学模型: 从当前 obs_latent 预测下一步 obs_latent."""
    def __init__(self, dim, hidden_dim=None, num_layers=3):
        super().__init__()
        hidden_dim = hidden_dim or dim
        layers = []
        in_dim = dim
        for i in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs_latent):
        """预测下一步的 obs_latent."""
        return self.net(obs_latent)


class MomentumTargetEncoder(nn.Module):
    """Momentum target encoder for self-predictive dynamics.

    维护 obs_encoder 的 EMA 副本，提供非平凡的预测目标。
    dynamics_model 预测 target_encoder(obs_{t}) 而非 obs_encoder(obs_{t}).detach()，
    避免恒等映射的 trivial solution。
    """
    def __init__(self, encoder, momentum=0.99):
        super().__init__()
        import copy
        self.target_encoder = copy.deepcopy(encoder)
        # 冻结 target encoder：不参与梯度更新
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.momentum = momentum

    @torch.no_grad()
    def update(self, encoder):
        """EMA 更新 target encoder 参数."""
        for p_target, p_online in zip(self.target_encoder.parameters(),
                                       encoder.parameters()):
            p_target.data.mul_(self.momentum).add_(p_online.data, alpha=1 - self.momentum)

    def forward(self, x):
        with torch.no_grad():
            return self.target_encoder(x)


class ProgressHead(nn.Module):
    """从 obs_latent 预测任务进度 p ∈ [0,1].

    训练时用 episode 内相对帧位置 (frame_index / episode_length) 作为监督标签，
    推理时仅从观测特征自主推断进度，不需要任何外部输入。
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs_latent):
        """
        Args:
            obs_latent: (B, dim)
        Returns:
            progress: (B, 1) ∈ [0, 1]
        """
        return self.net(obs_latent)


class MemoryBank(nn.Module):
    """轻量记忆模块: 用 key-value 内积读取历史上下文."""
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def read(self, query, memory):
        """
        Args:
            query: (B, dim) — 当前 obs_latent
            memory: (B, T, dim) — 历史 obs_latent 序列
        Returns:
            context: (B, dim)
        """
        q = self.query_proj(query).unsqueeze(1)     # (B, 1, dim)
        k = self.key_proj(memory)                    # (B, T, dim)
        v = self.value_proj(memory)                  # (B, T, dim)

        dim = q.shape[-1]
        attn = torch.bmm(q, k.transpose(-1, -2)) / (dim ** 0.5)  # (B, 1, T)
        attn = F.softmax(attn, dim=-1)
        context = torch.bmm(attn, v).squeeze(1)     # (B, dim)
        return self.out_proj(context)


class EnhancedFlowNetLayer(nn.Module):
    """Flow network layer with dual FiLM modulation (flow time + task progress)."""
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
        # 双路 FiLM 调制
        self.time_modulator = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.progress_modulator = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.dim = dim
        # Zero-init for stable training
        nn.init.constant_(self.time_modulator[-1].weight, 0)
        nn.init.constant_(self.time_modulator[-1].bias, 0)
        nn.init.constant_(self.progress_modulator[-1].weight, 0)
        nn.init.constant_(self.progress_modulator[-1].bias, 0)

    def forward(self, x, t, progress_emb=None):
        """
        Args:
            x: (B, dim)
            t: (B, dim) — flow timestep embedding
            progress_emb: (B, dim) — task progress embedding (optional)
        """
        gamma_t, scale_t, shift_t = self.time_modulator(t).view(
            x.shape[0], 3, self.dim).unbind(1)
        x_norm = self.norm(x).mul(scale_t.add(1)).add_(shift_t)

        if progress_emb is not None:
            gamma_p, scale_p, shift_p = self.progress_modulator(progress_emb).view(
                x.shape[0], 3, self.dim).unbind(1)
            x_norm = x_norm.mul(scale_p.add(1)).add_(shift_p)
            return x + self.mlp(x_norm).mul_(gamma_t).mul_(gamma_p.add(1))

        return x + self.mlp(x_norm).mul_(gamma_t)


class EnhancedFlowNet(nn.Module):
    """MLP flow network with dual FiLM — flow time + task progress."""
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
        self.progress_embed = nn.Sequential(
            SinusoidalPosEmbed(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )
        self.layers = nn.ModuleList([
            EnhancedFlowNetLayer(hidden_dim, mlp_ratio, dropout)
            for _ in range(num_layers)
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
        nn.init.normal_(self.progress_embed[1].weight, std=0.02)
        nn.init.normal_(self.progress_embed[3].weight, std=0.02)

    def forward(self, x, t, progress=None):
        """
        Args:
            x: (B, input_dim)
            t: (B,) — flow timestep
            progress: (B, 1) or (B,) — task progress ∈ [0,1], optional
        """
        x = self.input_proj(x)
        t_emb = self.time_embed(t)

        p_emb = None
        if progress is not None:
            if progress.dim() == 2:
                progress = progress.squeeze(-1)
            p_emb = self.progress_embed(progress)

        for block in self.layers:
            x = block(x, t_emb, p_emb)
        return self.out_proj(self.norm(x))


class EnhancedFlowMatcher:
    """OT-CFM Flow Matcher with progress conditioning."""
    def __init__(self, sigma=0.0, num_sampling_steps=6):
        import torchcfm.conditional_flow_matching as cfm
        self.fm = cfm.ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        self.num_sampling_steps = num_sampling_steps

    def compute_loss(self, model, target, start=None, progress=None):
        x0 = torch.randn_like(target) if start is None else start
        timestep, xt, ut = self.fm.sample_location_and_conditional_flow(x0, target)
        vt = model(xt, timestep, progress=progress)
        loss = torch.mean((vt - ut) ** 2)
        return loss, {"flow_loss": loss.item()}

    def sample(self, model, shape, device, num_steps=None, start=None, progress=None):
        if num_steps is None:
            num_steps = self.num_sampling_steps
        x = torch.randn(shape, device=device) if start is None else start
        dt = 1.0 / num_steps
        for t in range(num_steps):
            timestep = torch.ones(x.shape[0], device=device) * (t / num_steps)
            x = x + model(x, timestep, progress=progress) * dt
        return x


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


class ResNetObserver(nn.Module):
    """ResNet18 image feature extractor + state concatenation.

    支持特征差分模式 (temporal_diff=True):
      obs_horizon=2 时输出 [f_t, Δf, state_t, Δstate] 而非 flatten [f_{t-1}, f_t, ...]
      Δf = f_t - f_{t-1} 编码特征空间的"语义速度"
    """
    def __init__(self, state_dim, image_keys, obs_horizon=1, resize_shape=(240, 320),
                 crop_shape=(224, 308), temporal_diff=False):
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

        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    @property
    def output_dim(self):
        n_img_feat = len(self.image_keys) * 512
        if self.temporal_diff:
            # f_t + Δf + state_t + Δstate
            return n_img_feat * 2 + self.state_dim * 2
        return (n_img_feat + self.state_dim) * self.obs_horizon

    @property
    def frame_feature_dim(self):
        """单帧 raw 特征维度 (N*512 + state_dim)，用于 dynamics 跨帧预测."""
        return len(self.image_keys) * 512 + self.state_dim

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
        features = self.pool(self.backbone(images)["feature_map"])
        return features

    def forward(self, batch):
        state = batch["observation.state"]
        b = state.shape[0]
        n = len(self.image_keys)
        img_list = [batch[k] for k in self.image_keys]
        sample = img_list[0]

        if self.temporal_diff and sample.dim() == 5:
            # === 特征差分模式 ===
            # images: (B, S, C, H, W), S >= 2
            s = sample.shape[1]
            images = torch.stack(img_list, dim=2)  # (B, S, N, C, H, W)
            images_flat = images.reshape(b * s * n, *sample.shape[2:])  # (B*S*N, C, H, W)

            all_features = self._extract_features(images_flat, b)  # (B*S*N, 512, 1, 1)
            all_features = all_features.reshape(b, s, n * 512)    # (B, S, N*512)

            # 当前帧和上一帧特征
            f_t = all_features[:, -1, :]         # (B, N*512)
            f_prev = all_features[:, -2, :]      # (B, N*512)
            delta_f = f_t - f_prev               # (B, N*512) 特征差分

            # state 差分
            # state: (B, S, state_dim)
            state_t = state[:, -1, :]            # (B, state_dim)
            state_prev = state[:, -2, :]         # (B, state_dim)
            delta_state = state_t - state_prev   # (B, state_dim)

            # 缓存各帧独立 raw 特征到 batch，供 dynamics 跨帧预测使用
            # obs_features_prev = [f_prev, state_prev], obs_features_curr = [f_t, state_t]
            self._frame_features = {
                "prev": torch.cat([f_prev, state_prev], dim=1),   # (B, N*512+state_dim)
                "curr": torch.cat([f_t, state_t], dim=1),         # (B, N*512+state_dim)
            }

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

        features = self._extract_features(images, b)
        features = features.reshape(b, -1)

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
        self._use_temporal = False  # default, updated in build()
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
        # 时间感知模块 (set in build() when use_temporal=True)
        self.dynamics_model = None
        self.progress_head = None
        self.memory_bank = None
        self.memory_proj = None  # 融合 obs_latent + memory_context
        self.target_encoder = None  # momentum target encoder for dynamics
        # 推理时的记忆缓存
        self._memory_buffer = None

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
        # 时间感知模块参数
        parser.add_argument("--use_temporal", action="store_true",
                            help="Enable temporal-aware modules (dynamics, progress, memory)")
        parser.add_argument("--dynamics_weight", type=float, default=0.5,
                            help="Self-predictive dynamics loss weight")
        parser.add_argument("--progress_weight", type=float, default=0.1,
                            help="Progress prediction loss weight (MSE vs frame position)")
        parser.add_argument("--episode_length", type=int, default=700,
                            help="Episode length for computing progress label")
        parser.add_argument("--memory_size", type=int, default=8,
                            help="Number of history frames for memory bank (training)")

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
        self._use_temporal = getattr(args, 'use_temporal', False)

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

        # use_temporal 需要 obs_horizon >= 2 来做特征差分
        if self._use_temporal and self._obs_horizon < 2:
            self._obs_horizon = 2

        # Build observer
        self.observer = ResNetObserver(
            state_dim=state_dim,
            image_keys=image_keys,
            obs_horizon=self._obs_horizon,
            resize_shape=tuple(args.resize_shape),
            crop_shape=tuple(args.crop_shape),
            temporal_diff=self._use_temporal,
        )
        obs_dim = self.observer.output_dim

        # Obs encoder: project to latent space
        self.obs_encoder = nn.Linear(obs_dim, args.latent_dim)

        if self._use_temporal:
            # === 时间感知模块 ===
            # Flow network with dual FiLM (flow time + progress)
            self.flow_net = EnhancedFlowNet(
                input_dim=args.latent_dim,
                hidden_dim=args.flow_hidden_dim,
                output_dim=args.latent_dim,
                num_layers=args.flow_num_layers,
                mlp_ratio=args.flow_mlp_ratio,
            )
            self.flow_matcher = EnhancedFlowMatcher(
                sigma=0.0,
                num_sampling_steps=args.num_sampling_steps,
            )
            # Self-predictive dynamics: predict latent_t from latent_{t-1}
            # dynamics_encoder 将各帧 raw 特征映射到 latent_dim
            frame_dim = self.observer.frame_feature_dim
            self.dynamics_encoder = nn.Linear(frame_dim, args.latent_dim)
            # dynamics_model 从 latent_{t-1} 预测 latent_t
            self.dynamics_model = LatentDynamics(args.latent_dim)
            # Progress head
            self.progress_head = ProgressHead(args.latent_dim)
            # Memory bank
            self.memory_bank = MemoryBank(args.latent_dim)
            # 融合 obs_latent + memory_context → enriched_latent
            self.memory_proj = nn.Linear(args.latent_dim * 2, args.latent_dim)
        else:
            # === 原始模式 ===
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
        # use_temporal 强制 obs_horizon >= 2
        use_temporal = self._use_temporal
        if not use_temporal and self._args and getattr(self._args, 'use_temporal', False):
            use_temporal = True
        if use_temporal and obs_horizon < 2:
            obs_horizon = 2
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
        obs_latents = self.obs_encoder(obs_features)

        gt_actions = batch["action"]
        batch_size = obs_features.shape[0]

        # Encode ground-truth actions to latent
        action_latents = self.action_encoder(gt_actions)

        # === 时间感知模块的额外处理 ===
        progress = None
        if self._use_temporal:
            # 1. Self-predictive dynamics loss (跨帧预测)
            #    利用 obs_horizon=2 提供的 t-1 和 t 帧独立特征
            #    dynamics_model(latent_{t-1}) 预测 latent_t（不同帧 → 非平凡目标）
            frame_feats = getattr(self.observer, '_frame_features', None)
            if frame_feats is not None:
                # 各帧独立编码到 latent space
                latent_prev = self.dynamics_encoder(frame_feats["prev"])  # (B, latent_dim)
                latent_curr = self.dynamics_encoder(frame_feats["curr"])  # (B, latent_dim)
                # dynamics_model 从 t-1 预测 t
                latent_pred = self.dynamics_model(latent_prev)
                dynamics_loss = F.mse_loss(latent_pred, latent_curr.detach())
                pred_error = ((latent_pred - latent_curr.detach()) ** 2).mean(dim=-1, keepdim=True)
            else:
                # fallback: 推理时或 obs_horizon=1 时无跨帧特征
                dynamics_loss = torch.tensor(0.0, device=obs_latents.device)
                pred_error = torch.zeros(batch_size, 1, device=obs_latents.device)

            # 2. Progress prediction — 用 obs_latents 预测进度，frame_index 作为监督标签
            progress = self.progress_head(obs_latents)  # (B, 1)

            # 计算 progress 监督标签：frame_index / (episode_length - 1) ∈ [0, 1]
            if "frame_index" in batch:
                frame_idx = batch["frame_index"].float()  # (B,) or (B, S)
                if frame_idx.dim() > 1:
                    frame_idx = frame_idx[:, -1]  # 取最新帧的 index
                # episode 长度：从 dataset 获取或默认 700
                ep_len = getattr(self._args, 'episode_length', 700)
                progress_label = (frame_idx / max(ep_len - 1, 1)).unsqueeze(-1)  # (B, 1)
                progress_loss = F.mse_loss(progress, progress_label)
            else:
                progress_loss = torch.tensor(0.0, device=obs_latents.device)

            # 3. Memory bank (训练时用 batch 内其他样本模拟记忆)
            memory_size = getattr(self._args, 'memory_size', 8)
            mem_indices = torch.stack([
                torch.randperm(batch_size, device=obs_latents.device)[:memory_size]
                for _ in range(batch_size)
            ])  # (B, memory_size)
            memory = obs_latents.detach()[mem_indices]  # (B, memory_size, latent_dim)
            memory_context = self.memory_bank.read(obs_latents, memory)  # (B, latent_dim)

            # 融合 obs_latent + memory_context
            enriched_latent = self.memory_proj(
                torch.cat([obs_latents, memory_context], dim=-1))  # (B, latent_dim)
        else:
            enriched_latent = obs_latents

        # Flow matching loss
        flow_loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net, target=action_latents, start=enriched_latent,
            **({"progress": progress} if self._use_temporal else {})
        )
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
                device=enriched_latent.device,
                start=enriched_latent,
                **({"progress": progress} if self._use_temporal else {})
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

        # 时间感知额外 loss
        if self._use_temporal:
            dyn_w = getattr(self._args, 'dynamics_weight', 0.5)
            prog_w = getattr(self._args, 'progress_weight', 0.1)
            loss = loss + dyn_w * dynamics_loss
            loss = loss + prog_w * progress_loss
            metrics["dynamics_loss"] = dynamics_loss.item()
            metrics["progress_loss"] = progress_loss.item()
            metrics["progress_mean"] = progress.mean().item()

        metrics["total_loss"] = loss.item()
        return loss, metrics

    @torch.no_grad()
    def _generate_actions(self, batch):
        """Generate action chunk from observation batch."""
        obs_features = self.observer(batch)
        obs_latents = self.obs_encoder(obs_features)

        progress = None
        if self._use_temporal:
            # Progress prediction — 推理时纯靠 obs 推断进度
            progress = self.progress_head(obs_latents)

            # Memory bank read (推理时使用缓存的 memory buffer)
            if self._memory_buffer is not None and len(self._memory_buffer) > 0:
                memory = torch.stack(list(self._memory_buffer), dim=1)  # (1, T, dim)
                memory_context = self.memory_bank.read(obs_latents, memory)
            else:
                memory_context = torch.zeros_like(obs_latents)

            enriched_latent = self.memory_proj(
                torch.cat([obs_latents, memory_context], dim=-1))

            # 写入 memory buffer
            if self._memory_buffer is not None:
                self._memory_buffer.append(obs_latents.detach())
        else:
            enriched_latent = obs_latents

        action_latents_pred = self.flow_matcher.sample(
            self.flow_net,
            shape=(enriched_latent.shape[0], self._args.latent_dim),
            device=enriched_latent.device,
            start=enriched_latent,
            **({"progress": progress} if self._use_temporal else {})
        )
        return self.action_decoder(action_latents_pred)

    def select_action(self, batch):
        self.eval()
        # Normalize inputs
        batch_norm = self.normalize_inputs(batch)

        if len(self._action_queue) == 0:
            actions = self._generate_actions(batch_norm)
            # Only take action_horizon steps
            actions = actions[:, :self._action_horizon]
            # Unnormalize
            actions = self.unnormalize_outputs({"action": actions})["action"]
            # 反缩放 action（如 mm -> m）
            if self._action_scale != 1.0:
                actions = actions / self._action_scale
            # Add to queue: (action_horizon, 1, action_dim)
            for t in range(actions.shape[1]):
                self._action_queue.append(actions[:, t])

        return self._action_queue.popleft()  # (1, action_dim)

    def reset(self):
        self._action_queue = deque()
        # 重置推理时的 memory buffer
        if self._use_temporal:
            self._memory_buffer = deque(maxlen=getattr(self._args, 'memory_size', 8) * 10)

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

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存时使用 EMA 权重
        if self.ema is not None:
            self.ema.apply_shadow()

        torch.save(self.state_dict(), path / "model.pt")

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
            "use_temporal": self._use_temporal,
            "dynamics_weight": getattr(self._args, 'dynamics_weight', 0.5),
            "memory_size": getattr(self._args, 'memory_size', 8),
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

        # Reconstruct args namespace
        args = argparse.Namespace(**{
            "device": str(device),
            "lr": 1e-4,  # not used at inference
            **{k: v for k, v in meta.items()
               if k not in ("policy_type", "state_dim", "action_dim",
                             "image_keys", "pred_horizon")},
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
