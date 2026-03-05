"""
LeRobot 原生格式数据采集器

使用 LeRobotDataset.create() / add_frame() / save_episode() / finalize() API
直接生成 LeRobot 可加载的数据集，无需额外转换。

用法:
    collector = LeRobotNativeCollector(
        repo_id="local/mr2010-welding",
        root="datasets_lerobot",
        fps=50,
    )
    collector.add_trajectory(traj)   # 添加 Trajectory 对象
    collector.close()                # 必须调用，写入 parquet footer
"""

import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from il_env import Trajectory
from mr2010_robot import JOINT_NAMES, NUM_JOINTS


def build_features(has_images: bool = True, image_size: tuple = (480, 640)) -> dict:
    """构建 LeRobot features 定义

    注意: 不要包含 DEFAULT_FEATURES (timestamp, frame_index, episode_index, index, task_index)，
    LeRobotDataset.create() 会自动添加。

    Args:
        has_images: 是否包含图像特征
        image_size: 图像尺寸 (height, width)
    """
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (NUM_JOINTS,),
            "names": list(JOINT_NAMES),
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": (NUM_JOINTS,),
            "names": [f"{n}_vel" for n in JOINT_NAMES],
        },
        "observation.ee_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        },
        "observation.torque": {
            "dtype": "float32",
            "shape": (NUM_JOINTS,),
            "names": [f"{n}_torque" for n in JOINT_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (NUM_JOINTS,),
            "names": list(JOINT_NAMES),
        },
        "action.ee_delta": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["dx", "dy", "dz", "drx", "dry", "drz"],
        },
    }

    if has_images:
        h, w = image_size
        features["observation.images.overhead"] = {
            "dtype": "video",
            "shape": (3, h, w),
            "names": ["channel", "height", "width"],
        }
        features["observation.images.ee"] = {
            "dtype": "video",
            "shape": (3, h, w),
            "names": ["channel", "height", "width"],
        }

    return features


class LeRobotNativeCollector:
    """使用 LeRobot 原生 API 的数据采集器

    用法:
        collector = LeRobotNativeCollector(...)

        # 方式1: 添加已生成的 Trajectory 对象
        collector.add_trajectory(traj)

        # 方式2: 逐帧添加
        collector.start_episode()
        collector.add_step(t, qpos, qvel, action, ...)
        collector.end_episode()

        # 必须调用
        collector.close()
    """

    def __init__(
        self,
        repo_id: str = "local/mr2010-welding",
        root: str = "datasets_lerobot",
        fps: int = 50,
        task_description: str = "welding S-curve seam trajectory tracking",
        robot_type: str = "MR2010",
        has_images: bool = True,
        image_size: tuple = (480, 640),
        use_videos: bool = True,
        vcodec: str = "h264",
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.task_description = task_description
        self.has_images = has_images
        self.image_size = image_size
        self._episode_count = 0
        self._current_frames: list[dict] = []

        # 如果目标目录已存在，先删除
        root_path = Path(root)
        if root_path.exists():
            print(f"  Removing existing dataset at {root_path}")
            shutil.rmtree(root_path)

        features = build_features(has_images=has_images, image_size=image_size)

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type=robot_type,
            use_videos=use_videos if has_images else False,
            image_writer_processes=0,
            image_writer_threads=4,
            vcodec=vcodec,
        )

        print(f"  LeRobot dataset created at {root}")
        print(f"  repo_id={repo_id}, fps={fps}, images={has_images}")

    @property
    def num_episodes(self) -> int:
        return self._episode_count

    def add_trajectory(self, traj: Trajectory):
        """添加一条完整的 Trajectory 对象为一个 episode"""
        n_frames = len(traj)
        has_overhead = bool(traj.overhead_images)
        has_ee = bool(traj.ee_images)

        for i in range(n_frames):
            frame = {
                "task": self.task_description,
                "observation.state": np.array(traj.joint_positions[i], dtype=np.float32),
                "action": np.array(traj.actions[i], dtype=np.float32),
            }

            if traj.joint_velocities and i < len(traj.joint_velocities):
                frame["observation.velocity"] = np.array(traj.joint_velocities[i], dtype=np.float32)

            if traj.ee_positions and i < len(traj.ee_positions):
                frame["observation.ee_pos"] = np.array(traj.ee_positions[i], dtype=np.float32)

            if traj.joint_torques and i < len(traj.joint_torques):
                frame["observation.torque"] = np.array(traj.joint_torques[i], dtype=np.float32)

            if traj.actions_ee_delta and i < len(traj.actions_ee_delta):
                frame["action.ee_delta"] = np.array(traj.actions_ee_delta[i], dtype=np.float32)

            if self.has_images:
                if has_overhead and i < len(traj.overhead_images):
                    frame["observation.images.overhead"] = np.array(
                        traj.overhead_images[i], dtype=np.uint8
                    )
                if has_ee and i < len(traj.ee_images):
                    frame["observation.images.ee"] = np.array(
                        traj.ee_images[i], dtype=np.uint8
                    )

            self.dataset.add_frame(frame)

        self.dataset.save_episode()
        self._episode_count += 1

        print(f"  Episode {self._episode_count - 1}: {n_frames} frames"
              + (f", images={has_overhead and has_ee}" if self.has_images else ""))

    # ---- 逐帧接口 ----

    def start_episode(self):
        """开始新的 episode (逐帧模式)"""
        self._current_frames = []

    def add_step(
        self,
        t: float,
        qpos: np.ndarray,
        qvel: np.ndarray,
        action: np.ndarray,
        ee_pos: np.ndarray = None,
        torques: np.ndarray = None,
        overhead_img: np.ndarray = None,
        ee_img: np.ndarray = None,
    ):
        """添加一帧数据"""
        frame = {
            "task": self.task_description,
            "observation.state": np.array(qpos, dtype=np.float32),
            "observation.velocity": np.array(qvel, dtype=np.float32),
            "action": np.array(action, dtype=np.float32),
        }

        if ee_pos is not None:
            frame["observation.ee_pos"] = np.array(ee_pos, dtype=np.float32)

        if torques is not None:
            frame["observation.torque"] = np.array(torques, dtype=np.float32)

        if self.has_images:
            if overhead_img is not None:
                frame["observation.images.overhead"] = np.array(overhead_img, dtype=np.uint8)
            if ee_img is not None:
                frame["observation.images.ee"] = np.array(ee_img, dtype=np.uint8)

        self.dataset.add_frame(frame)

    def end_episode(self):
        """结束当前 episode 并保存"""
        self.dataset.save_episode()
        self._episode_count += 1

    def close(self):
        """关闭数据集 (写入 parquet footer，必须调用)"""
        self.dataset.finalize()
        print(f"\nDataset finalized: {self._episode_count} episodes")
