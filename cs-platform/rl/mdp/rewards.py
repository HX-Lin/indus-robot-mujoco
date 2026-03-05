"""Custom reward functions for the MR2010 welding RL environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def tracking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Gaussian reward for proximity to current seam target.

    reward = exp(-||ee - target||^2 / std^2)
    Returns: (num_envs,)
    """
    robot = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, 0]  # (N, 3)
    target = env.command_manager.get_command(command_name)  # (N, 3)
    dist_sq = torch.sum((ee_pos - target) ** 2, dim=-1)
    return torch.exp(-dist_sq / (std ** 2))


def progress_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward for advancing along the seam (delta progress per step).
    Returns: (num_envs,)
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd.progress_delta


def orientation_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for keeping tool pointing downward.

    reward = (cos_angle + 1) / 2, where cos_angle = dot(tool_z, -world_Z)
    Perfect downward → 1.0, horizontal → 0.5, upward → 0.0
    Returns: (num_envs,)
    """
    robot = env.scene[asset_cfg.name]
    site_quat = robot.data.site_quat_w[:, 0]  # (N, 4)

    w, x, y, z = site_quat[:, 0], site_quat[:, 1], site_quat[:, 2], site_quat[:, 3]
    tool_z_z = 1 - 2 * (x * x + y * y)
    cos_angle = -tool_z_z
    return (cos_angle + 1.0) / 2.0


def seam_completion_bonus(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float = 0.95,
) -> torch.Tensor:
    """Bonus reward when seam is nearly completed.
    Returns: (num_envs,)
    """
    cmd = env.command_manager.get_term(command_name)
    return (cmd.progress >= threshold).float()
