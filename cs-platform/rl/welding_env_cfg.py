"""MR2010 焊接 RL 环境配置

基于 mjlab 的 ManagerBasedRlEnvCfg，使用 MuJoCo Warp GPU 并行仿真。
机械臂任务：在 workpiece 表面沿 S-curve 焊缝进行轨迹跟踪。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.entity.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.scene.scene import SceneCfg
from mjlab.sim.sim import SimulationCfg, MujocoCfg
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.managers.observation_manager import ObservationTermCfg, ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

# Ensure rl/mdp is importable
_cs_dir = Path(__file__).resolve().parent.parent
if str(_cs_dir) not in sys.path:
    sys.path.insert(0, str(_cs_dir))

from rl.mdp.commands import SeamTrackingCommandCfg
from rl.mdp import observations as weld_obs
from rl.mdp import rewards as weld_rew
from rl.mdp.terminations import seam_completed, ee_out_of_bounds


# ── Assets ──────────────────────────────────────────────────────
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def _get_mr2010_spec() -> mujoco.MjSpec:
    """Load MR2010 robot spec with tool_tip site."""
    spec = mujoco.MjSpec.from_file(str(ASSETS_DIR / "MR2010.xml"))
    # Load STL assets
    for pattern in ("*.STL", "*.stl"):
        for stl_file in (ASSETS_DIR).glob(pattern):
            spec.assets[stl_file.name] = stl_file.read_bytes()

    # Add tool_tip site on Link6
    link6 = spec.worldbody.find_child("Link6")
    site = link6.add_site()
    site.name = "tool_tip"
    site.pos = [-0.089885, -0.030538, 0.38983]
    site.quat = [0.963075, 0.0766, -0.256027, 0.0326944]
    site.type = mujoco.mjtGeom.mjGEOM_SPHERE
    site.size = [0.005, 0, 0]
    return spec


def _scene_spec_fn(spec: mujoco.MjSpec) -> None:
    """Add ground plane, lights, platform, and workpiece to the scene spec."""
    # Ground
    spec.add_texture(
        name="ground", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=(0.2, 0.3, 0.4), rgb2=(0.1, 0.2, 0.3),
        width=300, height=300,
    )
    mat = spec.add_material(name="ground", texuniform=True, texrepeat=(4, 4))
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = "ground"
    spec.worldbody.add_body(name="terrain").add_geom(
        name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0, 0, 0.01), material="ground",
    )

    # Lights
    spec.visual.headlight.diffuse = (0.6, 0.6, 0.6)
    spec.visual.headlight.ambient = (0.3, 0.3, 0.3)
    spec.worldbody.add_light(
        pos=(0, 0, 3), dir=(0, 0, -1),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=True, diffuse=(0.8, 0.8, 0.8),
    )

    # Platform (reuse from mr2010_robot)
    from mr2010_robot import add_platform
    add_platform(spec)

    # Workpiece — static body (no freejoint) matching SeamTrackingCommandCfg defaults
    _add_static_workpiece(spec)


def _add_static_workpiece(spec: mujoco.MjSpec) -> None:
    """Add workpiece as a static visual body at the fixed pose used by SeamTrackingCommand."""
    workpiece_stl = ASSETS_DIR / "workpeice.STL"
    if not workpiece_stl.exists():
        return

    spec.add_mesh(
        name="workpiece",
        content_type="model/stl",
        file="workpeice.STL",
        scale=(0.001, 0.001, 0.001),
    )
    spec.assets["workpeice.STL"] = workpiece_stl.read_bytes()

    # Position/quat from settled simulation (must match SeamTrackingCommandCfg defaults)
    workpiece_body = spec.worldbody.add_body(name="workpiece")
    workpiece_body.pos = (1.475, 0.075, 0.44)
    workpiece_body.quat = (0.0, 1.0, 0.0, 0.0)
    # No freejoint — workpiece is static in RL env

    # Visual offset (0, 0, 0.0112) compensates for convex-hull collision gap,
    # matching mr2010_robot.add_workpiece() so workpiece sits flush on platform.
    workpiece_body.add_geom(
        name="workpiece_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="workpiece",
        contype=0,
        conaffinity=0,
        pos=(0, 0, 0.0112),
    )


# ── Robot entity ─────────────────────────────────────────────────

JOINT_NAMES = ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6")

# PD gains matching mr2010_robot.py
_ARMATURE = {"Joint1": 50, "Joint2": 50, "Joint3": 30, "Joint4": 10, "Joint5": 5, "Joint6": 2}
_OMEGA_N = 15.0
_ZETA = 5.0

ROBOT_CFG = EntityCfg(
    spec_fn=_get_mr2010_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
        actuators=tuple(
            IdealPdActuatorCfg(
                target_names_expr=(jn,),
                stiffness=_ARMATURE[jn] * _OMEGA_N ** 2,
                damping=2.0 * _ZETA * _ARMATURE[jn] * _OMEGA_N,
                effort_limit=500.0,
                armature=_ARMATURE[jn],
            )
            for jn in JOINT_NAMES
        ),
        soft_joint_pos_limit_factor=0.95,
    ),
)


# ── SceneEntityCfg factories ─────────────────────────────────────
# Each term needs its own SceneEntityCfg instance because resolve()
# mutates the object (sets joint_ids/site_ids). Sharing a single
# instance across terms causes "Inconsistent joint names and indices".

def _robot_entity() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=(".*",))

def _robot_site() -> SceneEntityCfg:
    return SceneEntityCfg("robot", site_names=("tool_tip",))


# ── Environment config ──────────────────────────────────────────

@dataclass(kw_only=True)
class WeldingEnvCfg(ManagerBasedRlEnvCfg):
    """MR2010 S-curve welding seam tracking environment."""

    decimation: int = 10          # 2ms * 10 = 20ms → 50Hz control
    episode_length_s: float = 20.0

    sim: SimulationCfg = field(default_factory=lambda: SimulationCfg(
        mujoco=MujocoCfg(
            timestep=0.002,
            iterations=10,
            ls_iterations=20,
        ),
    ))

    scene: SceneCfg = field(default_factory=lambda: SceneCfg(
        num_envs=2048,
        env_spacing=3.0,
        entities={"robot": ROBOT_CFG},
        spec_fn=_scene_spec_fn,
    ))

    # ── Observations ─────────────────────────────────────────────
    observations: dict = field(default_factory=lambda: {
        "policy": ObservationGroupCfg(
            terms={
                "joint_pos": ObservationTermCfg(
                    func=mdp.joint_pos_rel,
                    params={"asset_cfg": _robot_entity()},
                ),
                "joint_vel": ObservationTermCfg(
                    func=mdp.joint_vel_rel,
                    params={"asset_cfg": _robot_entity()},
                    clip=(-10.0, 10.0),
                ),
                "ee_position": ObservationTermCfg(
                    func=weld_obs.ee_position,
                    params={"asset_cfg": _robot_site()},
                ),
                "ee_to_target": ObservationTermCfg(
                    func=weld_obs.ee_to_target,
                    params={"command_name": "seam", "asset_cfg": _robot_site()},
                ),
                "seam_progress": ObservationTermCfg(
                    func=weld_obs.seam_progress,
                    params={"command_name": "seam"},
                ),
                "seam_target": ObservationTermCfg(
                    func=weld_obs.seam_target,
                    params={"command_name": "seam"},
                ),
                "ee_orientation": ObservationTermCfg(
                    func=weld_obs.ee_orientation_error,
                    params={"asset_cfg": _robot_site()},
                ),
                "last_action": ObservationTermCfg(
                    func=mdp.last_action,
                ),
            },
            concatenate_terms=True,
            enable_corruption=True,
        ),
    })

    # ── Actions ──────────────────────────────────────────────────
    actions: dict = field(default_factory=lambda: {
        "arm": mdp.JointPositionActionCfg(
            entity_name="robot",
            actuator_names=[".*"],
            scale=1.0,
            use_default_offset=True,
        ),
    })

    # ── Commands ─────────────────────────────────────────────────
    commands: dict = field(default_factory=lambda: {
        "seam": SeamTrackingCommandCfg(
            resampling_time_range=(100.0, 100.0),  # don't resample mid-episode
            num_waypoints=100,
            advance_threshold=0.02,
            robot_entity_name="robot",
            site_name="tool_tip",
        ),
    })

    # ── Rewards ──────────────────────────────────────────────────
    rewards: dict = field(default_factory=lambda: {
        "tracking": RewardTermCfg(
            func=weld_rew.tracking_reward,
            weight=2.0,
            params={
                "command_name": "seam",
                "std": 0.05,
                "asset_cfg": _robot_site(),
            },
        ),
        "progress": RewardTermCfg(
            func=weld_rew.progress_reward,
            weight=5.0,
            params={"command_name": "seam"},
        ),
        "orientation": RewardTermCfg(
            func=weld_rew.orientation_reward,
            weight=0.5,
            params={"asset_cfg": _robot_site()},
        ),
        "completion_bonus": RewardTermCfg(
            func=weld_rew.seam_completion_bonus,
            weight=10.0,
            params={"command_name": "seam", "threshold": 0.95},
        ),
        "action_rate": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.01,
        ),
        "joint_vel": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.001,
            params={"asset_cfg": _robot_entity()},
        ),
        "joint_torques": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-1e-5,
            params={"asset_cfg": _robot_entity()},
        ),
    })

    # ── Terminations ─────────────────────────────────────────────
    terminations: dict = field(default_factory=lambda: {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "nan_detection": TerminationTermCfg(
            func=mdp.nan_detection,
        ),
        "ee_out_of_bounds": TerminationTermCfg(
            func=ee_out_of_bounds,
            params={"asset_cfg": _robot_site()},
        ),
    })

    # ── Events (resets) ──────────────────────────────────────────
    events: dict = field(default_factory=lambda: {
        "reset_scene": EventTermCfg(
            func=mdp.reset_scene_to_default,
            mode="reset",
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": _robot_entity(),
            },
        ),
    })

    # ── Curriculum & viewer ──────────────────────────────────────
    curriculum: dict = field(default_factory=dict)
