"""
MR2010机械臂模仿学习环境
支持数据采集、可视化和训练
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import time

import numpy as np
import mujoco

from mr2010_robot import (
    JOINT_NAMES, NUM_JOINTS, create_base_scene,
)


# ========================== 数据采集 ==========================

@dataclass
class Trajectory:
    """单条轨迹数据

    基础字段 (始终记录):
        timestamps, joint_positions, joint_velocities, actions

    扩展字段 (可选，需要在 add_step 中传入):
        ee_positions: 末端执行器位置 (3,)
        ee_orientations: 末端执行器旋转矩阵 (3,3)
        joint_torques: 关节力矩 (NUM_JOINTS,)
        overhead_images: 俯视摄像头图像 (H, W, 3)
        ee_images: 末端摄像头图像 (H, W, 3)
    """
    timestamps: List[float] = field(default_factory=list)
    joint_positions: List[np.ndarray] = field(default_factory=list)
    joint_velocities: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    # 扩展字段
    ee_positions: List[np.ndarray] = field(default_factory=list)
    ee_orientations: List[np.ndarray] = field(default_factory=list)
    joint_torques: List[np.ndarray] = field(default_factory=list)
    actions_ee_delta: List[np.ndarray] = field(default_factory=list)  # 末端位姿增量 (6,): [dx,dy,dz,drx,dry,drz]
    overhead_images: List[np.ndarray] = field(default_factory=list)
    ee_images: List[np.ndarray] = field(default_factory=list)

    def add_step(self, t: float, qpos: np.ndarray, qvel: np.ndarray, action: np.ndarray,
                 ee_pos: np.ndarray = None, ee_mat: np.ndarray = None,
                 torques: np.ndarray = None, action_ee_delta: np.ndarray = None,
                 overhead_img: np.ndarray = None, ee_img: np.ndarray = None):
        self.timestamps.append(t)
        self.joint_positions.append(qpos.copy())
        self.joint_velocities.append(qvel.copy())
        self.actions.append(action.copy())
        if ee_pos is not None:
            self.ee_positions.append(ee_pos.copy())
        if ee_mat is not None:
            self.ee_orientations.append(ee_mat.copy())
        if torques is not None:
            self.joint_torques.append(torques.copy())
        if action_ee_delta is not None:
            self.actions_ee_delta.append(action_ee_delta.copy())
        if overhead_img is not None:
            self.overhead_images.append(overhead_img.copy())
        if ee_img is not None:
            self.ee_images.append(ee_img.copy())

    def to_dict(self) -> Dict[str, Any]:
        """转为JSON可序列化字典 (不含图像)"""
        d = {
            "timestamps": self.timestamps,
            "joint_positions": [q.tolist() for q in self.joint_positions],
            "joint_velocities": [v.tolist() for v in self.joint_velocities],
            "actions": [a.tolist() for a in self.actions],
        }
        if self.ee_positions:
            d["ee_positions"] = [p.tolist() for p in self.ee_positions]
        if self.ee_orientations:
            d["ee_orientations"] = [m.tolist() for m in self.ee_orientations]
        if self.joint_torques:
            d["joint_torques"] = [t.tolist() for t in self.joint_torques]
        return d

    def save_npz(self, filepath: str):
        """保存为NPZ格式 (支持图像等大型数组)"""
        arrays = {
            "timestamps": np.array(self.timestamps),
            "joint_positions": np.array(self.joint_positions),
            "joint_velocities": np.array(self.joint_velocities),
            "actions": np.array(self.actions),
        }
        if self.ee_positions:
            arrays["ee_positions"] = np.array(self.ee_positions)
        if self.ee_orientations:
            arrays["ee_orientations"] = np.array(self.ee_orientations)
        if self.joint_torques:
            arrays["joint_torques"] = np.array(self.joint_torques)
        if self.overhead_images:
            arrays["overhead_images"] = np.array(self.overhead_images, dtype=np.uint8)
        if self.ee_images:
            arrays["ee_images"] = np.array(self.ee_images, dtype=np.uint8)
        np.savez_compressed(filepath, **arrays)

    @classmethod
    def load_npz(cls, filepath: str) -> "Trajectory":
        """从NPZ文件加载"""
        d = np.load(filepath)
        traj = cls()
        traj.timestamps = d["timestamps"].tolist()
        traj.joint_positions = list(d["joint_positions"])
        traj.joint_velocities = list(d["joint_velocities"])
        traj.actions = list(d["actions"])
        if "ee_positions" in d:
            traj.ee_positions = list(d["ee_positions"])
        if "ee_orientations" in d:
            traj.ee_orientations = list(d["ee_orientations"])
        if "joint_torques" in d:
            traj.joint_torques = list(d["joint_torques"])
        if "overhead_images" in d:
            traj.overhead_images = list(d["overhead_images"])
        if "ee_images" in d:
            traj.ee_images = list(d["ee_images"])
        return traj

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        traj = cls()
        traj.timestamps = d["timestamps"]
        traj.joint_positions = [np.array(q) for q in d["joint_positions"]]
        traj.joint_velocities = [np.array(v) for v in d["joint_velocities"]]
        traj.actions = [np.array(a) for a in d["actions"]]
        if "ee_positions" in d:
            traj.ee_positions = [np.array(p) for p in d["ee_positions"]]
        if "ee_orientations" in d:
            traj.ee_orientations = [np.array(m) for m in d["ee_orientations"]]
        if "joint_torques" in d:
            traj.joint_torques = [np.array(t) for t in d["joint_torques"]]
        return traj

    def __len__(self):
        return len(self.timestamps)


class DataCollector:
    """数据采集器

    支持两种保存格式:
    - JSON: 仅保存数值数据 (兼容旧格式)
    - NPZ: 保存完整数据 (包含图像)
    """

    def __init__(self, save_dir: str = "datasets", save_format: str = "npz"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.save_format = save_format
        self.trajectories: List[Trajectory] = []
        self.current_trajectory: Optional[Trajectory] = None

    def start_episode(self):
        """开始新的轨迹采集"""
        self.current_trajectory = Trajectory()
        print("Started new trajectory recording")

    def record_step(self, t: float, qpos: np.ndarray, qvel: np.ndarray, action: np.ndarray,
                    **kwargs):
        """记录一步数据

        kwargs 支持: ee_pos, ee_mat, torques, overhead_img, ee_img
        """
        if self.current_trajectory is not None:
            self.current_trajectory.add_step(t, qpos, qvel, action, **kwargs)

    def end_episode(self, save: bool = True):
        """结束当前轨迹采集"""
        if self.current_trajectory is not None and len(self.current_trajectory) > 0:
            self.trajectories.append(self.current_trajectory)
            print(f"Trajectory recorded: {len(self.current_trajectory)} steps")
            if save:
                self.save_trajectory(self.current_trajectory)
        self.current_trajectory = None

    def save_trajectory(self, traj: Trajectory):
        """保存单条轨迹"""
        idx = len(self.trajectories)
        if self.save_format == "npz":
            filename = self.save_dir / f"traj_{idx:04d}.npz"
            traj.save_npz(str(filename))
        else:
            filename = self.save_dir / f"traj_{idx:04d}.json"
            with open(filename, "w") as f:
                json.dump(traj.to_dict(), f)
        print(f"Saved trajectory to {filename}")

    def save_all(self, filename: str = "dataset.json"):
        """保存所有轨迹的元数据索引 (JSON格式，不含图像)"""
        filepath = self.save_dir / filename
        data = {"trajectories": [t.to_dict() for t in self.trajectories]}
        with open(filepath, "w") as f:
            json.dump(data, f)
        print(f"Saved {len(self.trajectories)} trajectories to {filepath}")

    def load(self, filename: str = "dataset.json"):
        """加载数据集"""
        filepath = self.save_dir / filename
        with open(filepath, "r") as f:
            data = json.load(f)
        self.trajectories = [Trajectory.from_dict(t) for t in data["trajectories"]]
        print(f"Loaded {len(self.trajectories)} trajectories from {filepath}")


# ========================== 模仿学习环境 ==========================

class MR2010ImitationEnv:
    """MR2010模仿学习环境"""

    def __init__(self, render: bool = True, with_platform: bool = True, with_workpiece: bool = True):
        spec = create_base_scene(
            with_actuators=True,
            with_platform=with_platform,
            with_workpiece=with_workpiece,
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.render_enabled = render
        self.viewer = None
        self.collector = DataCollector()
        self.recording = False
        self.timestep = self.model.opt.timestep
        self.tool_tip_site_id = self.model.site("tool_tip").id

        self.reset()

    def reset(self) -> np.ndarray:
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        """获取观测（关节位置和速度）"""
        qpos = self.data.qpos[:NUM_JOINTS].copy()
        qvel = self.data.qvel[:NUM_JOINTS].copy()
        return np.concatenate([qpos, qvel])

    def get_ee_pos(self) -> np.ndarray:
        """返回tool_tip站点的笛卡尔坐标"""
        return self.data.site_xpos[self.tool_tip_site_id].copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """执行一步仿真"""
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        if self.recording:
            self.collector.record_step(
                self.data.time,
                self.data.qpos[:NUM_JOINTS],
                self.data.qvel[:NUM_JOINTS],
                action,
            )

        return self.get_obs(), 0.0, False, {}

    def start_recording(self):
        """开始录制"""
        self.recording = True
        self.collector.start_episode()

    def stop_recording(self):
        """停止录制"""
        self.recording = False
        self.collector.end_episode()

    def render(self):
        """渲染"""
        if self.viewer is None and self.render_enabled:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ========================== 交互式演示采集 ==========================

def run_interactive_demo():
    """运行交互式演示采集"""
    import mujoco.viewer

    spec = create_base_scene(with_actuators=True, with_platform=True, with_workpiece=True)
    model = spec.compile()
    data = mujoco.MjData(model)

    collector = DataCollector()
    recording = False
    paused = False

    print("\n=== MR2010 模仿学习数据采集 ===")
    print("控制说明:")
    print("  R: 开始/停止录制")
    print("  Space: 暂停/继续仿真")
    print("  Backspace: 重置环境")
    print("  S: 保存所有数据")
    print("  鼠标拖拽: 直接拖动机械臂")
    print("=" * 40)

    def key_callback(keycode):
        nonlocal recording, paused

        if keycode == ord('R') or keycode == ord('r'):
            if not recording:
                recording = True
                collector.start_episode()
                print("Recording started...")
            else:
                recording = False
                collector.end_episode()
                print("Recording stopped.")

        elif keycode == 32:  # Space
            paused = not paused
            print("Paused" if paused else "Resumed")

        elif keycode == 8:  # Backspace
            mujoco.mj_resetData(model, data)
            print("Environment reset")

        elif keycode == ord('S') or keycode == ord('s'):
            collector.save_all()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if not paused:
                mujoco.mj_step(model, data)

                if recording:
                    action = data.ctrl[:NUM_JOINTS] if model.nu >= NUM_JOINTS else np.zeros(NUM_JOINTS)
                    collector.record_step(
                        data.time,
                        data.qpos[:NUM_JOINTS],
                        data.qvel[:NUM_JOINTS],
                        action,
                    )

            viewer.sync()
            time.sleep(model.opt.timestep)

    if len(collector.trajectories) > 0:
        collector.save_all()
        print(f"\nTotal trajectories collected: {len(collector.trajectories)}")


# ========================== 主程序 ==========================

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if mode == "demo":
        run_interactive_demo()

    elif mode == "test":
        env = MR2010ImitationEnv(render=False)
        obs = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Initial obs: {obs}")

        for _ in range(100):
            obs, _, _, _ = env.step(np.zeros(NUM_JOINTS))

        print(f"Final obs: {obs}")
        env.close()
        print("Test passed!")

    elif mode == "view":
        import mujoco.viewer
        spec = create_base_scene(with_actuators=True, with_platform=True, with_workpiece=True)
        model = spec.compile()
        data = mujoco.MjData(model)
        print(f"Model: {model.ngeom} geoms, {model.njnt} joints, {model.nu} actuators")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)

    else:
        print(f"Usage: python il_env.py [demo|test|view]")
