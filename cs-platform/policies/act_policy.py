"""
ACT Policy 包装器 — 将 LeRobot 的 ACTPolicy 封装为 BasePolicy 接口

包装 lerobot.policies.act.modeling_act.ACTPolicy，
保持原有功能不变，通过统一接口接入训练/评估流程。
"""

import argparse
import json
from pathlib import Path

import torch

from policies import register_policy
from policies.base_policy import BasePolicy


@register_policy("act")
class ACTPolicyWrapper(BasePolicy):

    def __init__(self):
        super().__init__()
        self._policy = None  # LeRobot ACTPolicy instance
        self._config = None  # ACTConfig
        self._args = None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("--chunk_size", type=int, default=50)
        parser.add_argument("--backbone", type=str, default="resnet18",
                            choices=["resnet18", "resnet34"])
        parser.add_argument("--n_action_steps", type=int, default=1,
                            help="Actions per inference (1=temporal ensemble)")
        parser.add_argument("--temporal_ensemble_coeff", type=float, default=0.01)
        parser.add_argument("--kl_weight", type=float, default=10.0)
        parser.add_argument("--dim_model", type=int, default=512)

    def build(self, args, dataset_stats, input_features, output_features):
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        self._args = args
        device = torch.device(args.device if hasattr(args, 'device') else
                              ("cuda" if torch.cuda.is_available() else "cpu"))

        backbone_weights = {
            "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
            "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
        }

        self._config = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            n_obs_steps=1,
            use_vae=True,
            kl_weight=args.kl_weight,
            dim_model=args.dim_model,
            n_heads=8,
            dim_feedforward=3200,
            n_encoder_layers=4,
            n_decoder_layers=1,
            n_vae_encoder_layers=4,
            latent_dim=32,
            vision_backbone=args.backbone,
            pretrained_backbone_weights=backbone_weights[args.backbone],
            dropout=0.1,
            optimizer_lr=args.lr,
            optimizer_weight_decay=1e-4,
            optimizer_lr_backbone=args.lr,
        )

        self._policy = ACTPolicy(config=self._config, dataset_stats=dataset_stats)
        self._policy.to(device)

    @property
    def chunk_size(self):
        return self._args.chunk_size if self._args else 50

    def get_delta_timestamps(self, fps):
        return {
            "action": [i / fps for i in range(self.chunk_size)],
        }

    def compute_loss(self, batch):
        # ACT 的 forward 返回 (loss, loss_dict)
        loss, loss_dict = self._policy.forward(batch)
        return loss, loss_dict

    def select_action(self, batch):
        return self._policy.select_action(batch)

    def reset(self):
        self._policy.reset()

    def parameters(self, recurse=True):
        return self._policy.parameters(recurse=recurse)

    def get_optimizer(self):
        return torch.optim.AdamW(
            self._policy.get_optim_params(),
            lr=self._config.optimizer_lr,
            weight_decay=self._config.optimizer_weight_decay,
        )

    def train(self, mode=True):
        self._policy.train(mode)
        return self

    def eval(self):
        self._policy.eval()
        return self

    def to(self, device):
        self._policy.to(device)
        return self

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._policy.save_pretrained(path)
        # 保存额外的训练参数以便恢复
        meta = {
            "policy_type": "act",
            "chunk_size": self._args.chunk_size,
            "backbone": self._args.backbone,
            "n_action_steps": self._args.n_action_steps,
            "temporal_ensemble_coeff": self._args.temporal_ensemble_coeff,
            "kl_weight": self._args.kl_weight,
            "dim_model": self._args.dim_model,
        }
        with open(path / "policy_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Path, dataset_stats, device):
        from lerobot.policies.act.modeling_act import ACTPolicy

        path = Path(path)
        if path.is_symlink():
            path = path.resolve()

        policy = cls()
        policy._policy = ACTPolicy.from_pretrained(
            str(path), dataset_stats=dataset_stats,
        )
        policy._policy.to(device)
        policy._policy.eval()

        # 读取元信息（如果有）
        meta_path = path / "policy_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            policy._config = policy._policy.config
        else:
            policy._config = policy._policy.config

        return policy
