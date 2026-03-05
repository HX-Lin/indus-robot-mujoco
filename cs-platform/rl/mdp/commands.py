"""SeamTrackingCommand — 管理焊缝路径的 CommandTerm

在每个 episode reset 时预计算 S-curve 焊缝路径，跟踪当前目标 waypoint，
当 EE 接近目标时自动前进到下一个 waypoint。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# ── S-curve geometry (fixed, in STL mm) ──────────────────────────
ARC1_CENTER = (386.0, 200.0)
ARC2_CENTER = (174.0, 200.0)
ARC_RADIUS = 109.0


def _compute_scurve_stl(num_points: int, device: str) -> torch.Tensor:
    """Compute S-curve centerline in STL mm coordinates. → (num_points, 3)"""
    t = torch.linspace(0, 1, num_points, device=device)
    c1x, c1y = ARC1_CENTER
    c2x, c2y = ARC2_CENTER
    R = ARC_RADIUS

    theta1 = -math.pi * torch.clamp(2 * t, max=1.0)
    upper_x = c1x + R * torch.cos(theta1)
    upper_y = c1y + R * torch.sin(theta1)

    theta2 = math.pi * torch.clamp(2 * t - 1, min=0.0)
    lower_x = c2x + R * torch.cos(theta2)
    lower_y = c2y + R * torch.sin(theta2)

    sigma = torch.sigmoid(20.0 * (t - 0.5))

    pts = torch.zeros(num_points, 3, device=device)
    pts[:, 0] = (1 - sigma) * upper_x + sigma * lower_x
    pts[:, 1] = (1 - sigma) * upper_y + sigma * lower_y
    return pts


def _stl_to_world(
    stl_mm: torch.Tensor,
    body_pos: torch.Tensor,
    body_rotmat: torch.Tensor,
) -> torch.Tensor:
    """Transform STL mm → world meters (batched).

    Args:
        stl_mm:      (P, 3)
        body_pos:    (N, 3)
        body_rotmat: (N, 3, 3)
    Returns:
        (N, P, 3) world coordinates
    """
    pts_m = stl_mm * 0.001
    rotated = torch.einsum("nij,pj->npi", body_rotmat, pts_m)
    return body_pos.unsqueeze(1) + rotated


# ── CommandTerm ──────────────────────────────────────────────────

@dataclass(kw_only=True)
class SeamTrackingCommandCfg(CommandTermCfg):
    """Configuration for seam tracking command."""

    num_waypoints: int = 100
    """Number of waypoints sampled along the S-curve."""
    advance_threshold: float = 0.02
    """Distance (m) to advance to the next waypoint."""
    robot_entity_name: str = "robot"
    site_name: str = "tool_tip"
    # Fixed workpiece pose (default settled position)
    workpiece_pos: tuple[float, float, float] = (1.475, 0.075, 0.44)
    workpiece_quat_wxyz: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)

    def build(self, env: ManagerBasedRlEnv) -> SeamTrackingCommand:
        return SeamTrackingCommand(self, env)


class SeamTrackingCommand(CommandTerm):
    """Tracks a seam path and provides the current target waypoint."""

    cfg: SeamTrackingCommandCfg

    def __init__(self, cfg: SeamTrackingCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        _num_envs = env.num_envs
        _device = env.device

        # Resolve robot site
        self._robot_cfg = SceneEntityCfg(
            cfg.robot_entity_name, site_names=(cfg.site_name,),
        )
        self._robot_cfg.resolve(env.scene)

        # Pre-compute seam path in STL coordinates (shared by all envs)
        self._stl_pts = _compute_scurve_stl(cfg.num_waypoints, _device)

        # Compute fixed workpiece transform
        pos = torch.tensor(cfg.workpiece_pos, device=_device)
        self._body_pos = pos.unsqueeze(0).expand(_num_envs, -1)  # (N, 3)

        # quat (w,x,y,z) → rotation matrix
        w, x, y, z = cfg.workpiece_quat_wxyz
        rotmat = torch.tensor([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ], device=_device)
        self._body_rotmat = rotmat.unsqueeze(0).expand(_num_envs, -1, -1)

        # Seam waypoints in world frame: (N, P, 3)
        self._seam_world = _stl_to_world(
            self._stl_pts, self._body_pos, self._body_rotmat,
        )

        # Per-env state
        self._target_idx = torch.zeros(_num_envs, dtype=torch.long, device=_device)
        self._current_target = self._seam_world[:, 0].clone()  # (N, 3)
        self._progress = torch.zeros(_num_envs, device=_device)
        self._prev_progress = torch.zeros(_num_envs, device=_device)

    # ── Public API ───────────────────────────────────────────────

    @property
    def command(self) -> torch.Tensor:
        """Current target position in world frame. Shape: (num_envs, 3)."""
        return self._current_target

    @property
    def target_idx(self) -> torch.Tensor:
        return self._target_idx

    @property
    def progress(self) -> torch.Tensor:
        """Fraction of seam completed [0, 1]. Shape: (num_envs,)."""
        return self._progress

    @property
    def progress_delta(self) -> torch.Tensor:
        """Change in progress since last step. Shape: (num_envs,)."""
        return self._progress - self._prev_progress

    @property
    def seam_waypoints(self) -> torch.Tensor:
        """All seam waypoints. Shape: (num_envs, num_waypoints, 3)."""
        return self._seam_world

    # ── Required overrides ───────────────────────────────────────

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Reset seam progress for specified environments."""
        self._target_idx[env_ids] = 0
        self._current_target[env_ids] = self._seam_world[env_ids, 0]
        self._progress[env_ids] = 0.0
        self._prev_progress[env_ids] = 0.0

    def _update_command(self) -> None:
        """Advance target waypoint when EE is close enough."""
        robot = self._env.scene[self.cfg.robot_entity_name]
        ee_pos = robot.data.site_pos_w[:, 0]  # (N, 3)

        dist = torch.norm(ee_pos - self._current_target, dim=-1)
        advance = dist < self.cfg.advance_threshold

        max_idx = self.cfg.num_waypoints - 1
        new_idx = torch.clamp(self._target_idx + advance.long(), max=max_idx)
        self._target_idx = new_idx

        # Gather new targets
        idx_exp = new_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3)
        self._current_target = self._seam_world.gather(1, idx_exp).squeeze(1)

        # Update progress
        self._prev_progress = self._progress.clone()
        self._progress = new_idx.float() / max_idx

    def _update_metrics(self) -> None:
        robot = self._env.scene[self.cfg.robot_entity_name]
        ee_pos = robot.data.site_pos_w[:, 0]
        dist = torch.norm(ee_pos - self._current_target, dim=-1)

        self.metrics["tracking_error"] = dist.mean().item()
        self.metrics["seam_progress"] = self._progress.mean().item()
        self.metrics["target_idx"] = self._target_idx.float().mean().item()
