"""Custom termination functions for the MR2010 welding RL environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def seam_completed(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float = 0.99,
) -> torch.Tensor:
    """Terminate when seam tracking is fully completed.
    Returns: (num_envs,) bool
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd.progress >= threshold


def ee_out_of_bounds(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (0.3, 2.2),
    y_range: tuple[float, float] = (-1.0, 1.0),
    z_range: tuple[float, float] = (0.0, 1.5),
) -> torch.Tensor:
    """Terminate if end-effector leaves the workspace.
    Returns: (num_envs,) bool
    """
    robot = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, 0]  # (N, 3)

    out = (
        (ee_pos[:, 0] < x_range[0]) | (ee_pos[:, 0] > x_range[1])
        | (ee_pos[:, 1] < y_range[0]) | (ee_pos[:, 1] > y_range[1])
        | (ee_pos[:, 2] < z_range[0]) | (ee_pos[:, 2] > z_range[1])
    )
    return out
