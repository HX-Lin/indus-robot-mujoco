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
    """ResNet18 image feature extractor + state concatenation."""
    def __init__(self, state_dim, image_keys, resize_shape=(240, 320),
                 crop_shape=(224, 308)):
        super().__init__()
        import torchvision
        from torchvision.models._utils import IntermediateLayerGetter
        from torchvision.ops.misc import FrozenBatchNorm2d

        self.image_keys = image_keys
        self.state_dim = state_dim
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape

        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    @property
    def output_dim(self):
        return len(self.image_keys) * 512 + self.state_dim

    def forward(self, batch):
        state = batch["observation.state"]
        # state 可能是 (B, state_dim) 或 (B, S, state_dim)
        # 统一 flatten 到 (B, state_dim) 或 (B, S*state_dim)
        b = state.shape[0]
        state_flat = state.reshape(b, -1)  # (B, state_dim) 或 (B, S*state_dim)

        n = len(self.image_keys)

        # LeRobot 返回图像格式:
        #   训练时 dataloader batch: (B, C, H, W) — obs_horizon=1 时无 S 维
        #   多 obs_horizon: (B, S, C, H, W)
        # 我们 stack N 个相机后送入 ResNet
        img_list = [batch[k] for k in self.image_keys]  # 每个 (B, C, H, W) 或 (B, S, C, H, W)
        sample = img_list[0]

        if sample.dim() == 4:
            # (B, C, H, W) — 无 S 维, obs_horizon=1
            # stack N cameras: (B, N, C, H, W) -> (B*N, C, H, W)
            images = torch.stack(img_list, dim=1)  # (B, N, C, H, W)
            images = images.reshape(b * n, *sample.shape[1:])  # (B*N, C, H, W)
        elif sample.dim() == 5:
            # (B, S, C, H, W) — 有 S 维
            s = sample.shape[1]
            images = torch.stack(img_list, dim=2)  # (B, S, N, C, H, W)
            images = images.reshape(b * s * n, *sample.shape[2:])  # (B*S*N, C, H, W)
        else:
            raise ValueError(f"Unexpected image dim: {sample.dim()}, shape: {sample.shape}")

        # Resize
        if images.shape[-2:] != tuple(self.resize_shape):
            images = F.interpolate(images, size=tuple(self.resize_shape),
                                   mode='bilinear', align_corners=False)
        # Random crop (train) / center crop (eval)
        ch, cw = self.crop_shape
        _, _, H, W = images.shape
        if self.training:
            top = torch.randint(0, H - ch + 1, (1,)).item()
            left = torch.randint(0, W - cw + 1, (1,)).item()
        else:
            top = (H - ch) // 2
            left = (W - cw) // 2
        images = images[..., top:top + ch, left:left + cw]

        # Extract features: (B*N, 512, 1, 1) -> (B, N*512)
        features = self.pool(self.backbone(images)["feature_map"])  # (B*N, 512, 1, 1)
        features = features.reshape(b, -1)  # (B, N*512)

        return torch.cat([state_flat, features], dim=1)  # (B, state_dim + N*512)


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
        parser.add_argument("--resize_shape", type=int, nargs=2, default=[240, 320])
        parser.add_argument("--crop_shape", type=int, nargs=2, default=[224, 308])

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

        # Build observer
        self.observer = ResNetObserver(
            state_dim=state_dim,
            image_keys=image_keys,
            resize_shape=tuple(args.resize_shape),
            crop_shape=tuple(args.crop_shape),
        )
        obs_dim = self.observer.output_dim

        # Obs encoder: project to latent space
        self.obs_encoder = nn.Linear(obs_dim, args.latent_dim)

        # Flow network
        self.flow_net = SimpleFlowNet(
            input_dim=args.latent_dim,
            hidden_dim=args.flow_hidden_dim,
            output_dim=args.latent_dim,
            num_layers=args.flow_num_layers,
            mlp_ratio=args.flow_mlp_ratio,
        )

        # Flow matcher (OT-CFM)
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
        # （图像 std 极小 ~0.001，z-score 会放大 1000 倍摧毁 ResNet 特征）
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
        return {
            "action": [i / fps for i in range(self.chunk_size)],
        }

    def compute_loss(self, batch):
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Observation features -> latent
        obs_features = self.observer(batch)
        obs_latents = self.obs_encoder(obs_features)

        gt_actions = batch["action"]
        batch_size = obs_features.shape[0]

        # Encode ground-truth actions to latent
        action_latents = self.action_encoder(gt_actions)

        # Flow matching loss: obs_latents -> action_latents
        flow_loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net, target=action_latents, start=obs_latents
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

        if len(self._action_queue) == 0:
            actions = self._generate_actions(batch_norm)
            # Only take action_horizon steps
            actions = actions[:, :self._action_horizon]
            # Unnormalize
            actions = self.unnormalize_outputs({"action": actions})["action"]
            # Add to queue: (action_horizon, 1, action_dim)
            for t in range(actions.shape[1]):
                self._action_queue.append(actions[:, t])

        return self._action_queue.popleft()  # (1, action_dim)

    def reset(self):
        self._action_queue = deque()

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
            "state_dim": self._state_dim,
            "action_dim": self._action_dim,
            "image_keys": self._image_keys,
            "pred_horizon": self._pred_horizon,
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
        policy.load_state_dict(torch.load(path / "model.pt", map_location=device))
        policy.to(device)
        policy.eval()
        return policy
