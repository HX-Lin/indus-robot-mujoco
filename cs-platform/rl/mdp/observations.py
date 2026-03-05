"""Custom observation functions for the MR2010 welding RL environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def ee_position(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector (tool_tip site) position in world frame.
    Returns: (num_envs, 3)
    """
    robot = env.scene[asset_cfg.name]
    return robot.data.site_pos_w[:, 0]


def ee_to_target(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Vector from EE to current seam target.
    Returns: (num_envs, 3)
    """
    robot = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, 0]  # (N, 3)
    target = env.command_manager.get_command(command_name)  # (N, 3)
    return target - ee_pos


def seam_progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Fraction of seam completed [0, 1].
    Returns: (num_envs, 1)
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd.progress.unsqueeze(-1)


def seam_target(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Current seam target position in world frame.
    Returns: (num_envs, 3)
    """
    return env.command_manager.get_command(command_name)


def ee_orientation_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Cosine of angle between tool z-axis and world -Z (downward).

    Welding torch should point downward. Returns cos(angle) where 1.0 = perfect.
    Returns: (num_envs, 1)
    """
    robot = env.scene[asset_cfg.name]
    site_quat = robot.data.site_quat_w[:, 0]  # (N, 4)

    # Extract z-axis from quaternion (w, x, y, z)
    w, x, y, z = site_quat[:, 0], site_quat[:, 1], site_quat[:, 2], site_quat[:, 3]
    # z-axis of rotation: third column of rotation matrix
    tool_z_x = 2 * (x * z + w * y)
    tool_z_y = 2 * (y * z - w * x)
    tool_z_z = 1 - 2 * (x * x + y * y)

    # Desired direction: -Z (downward)
    cos_angle = -tool_z_z  # dot product with (0, 0, -1)
    return cos_angle.unsqueeze(-1)
