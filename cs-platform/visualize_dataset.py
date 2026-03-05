"""
数据集可视化脚本

支持可视化:
- 关节角度随时间变化
- 关节速度随时间变化
- 关节力矩随时间变化
- 末端执行器3D轨迹
- 摄像头图像 (如果有)

用法:
    python visualize_dataset.py --data_dir datasets --traj_idx 0
    python visualize_dataset.py --npz_file datasets/traj_0001.npz
    python visualize_dataset.py --data_dir datasets --show_all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from il_env import Trajectory
from mr2010_robot import JOINT_NAMES, NUM_JOINTS


def load_trajectory(path: Path) -> Trajectory:
    """从JSON或NPZ加载单条轨迹"""
    if path.suffix == ".npz":
        return Trajectory.load_npz(str(path))
    elif path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        # 单条轨迹JSON
        if "timestamps" in data:
            return Trajectory.from_dict(data)
        # dataset.json格式 (取第一条)
        if "trajectories" in data:
            return Trajectory.from_dict(data["trajectories"][0])
    raise ValueError(f"Unsupported file format: {path.suffix}")


def load_trajectory_by_index(data_dir: Path, idx: int) -> Trajectory:
    """按索引加载轨迹 (尝试NPZ优先，再JSON)"""
    npz_path = data_dir / f"traj_{idx + 1:04d}.npz"
    json_path = data_dir / f"traj_{idx + 1:04d}.json"
    if npz_path.exists():
        return Trajectory.load_npz(str(npz_path))
    elif json_path.exists():
        with open(json_path, "r") as f:
            return Trajectory.from_dict(json.load(f))
    raise FileNotFoundError(f"No trajectory file found for index {idx} in {data_dir}")


def list_trajectories(data_dir: Path) -> list:
    """列出目录中所有轨迹文件"""
    files = sorted(data_dir.glob("traj_*.npz")) + sorted(data_dir.glob("traj_*.json"))
    # 按文件名去重 (NPZ优先)
    seen = set()
    unique = []
    for f in files:
        stem = f.stem
        if stem not in seen:
            seen.add(stem)
            unique.append(f)
    return unique


def plot_joint_angles(traj: Trajectory, ax=None):
    """绘制关节角度随时间变化"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    t = np.array(traj.timestamps)
    qpos = np.array(traj.joint_positions)
    for j in range(min(NUM_JOINTS, qpos.shape[1])):
        ax.plot(t, np.degrees(qpos[:, j]), label=JOINT_NAMES[j])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Angle (deg)")
    ax.set_title("Joint Angles")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_joint_velocities(traj: Trajectory, ax=None):
    """绘制关节速度随时间变化"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    t = np.array(traj.timestamps)
    qvel = np.array(traj.joint_velocities)
    for j in range(min(NUM_JOINTS, qvel.shape[1])):
        ax.plot(t, np.degrees(qvel[:, j]), label=JOINT_NAMES[j])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Velocity (deg/s)")
    ax.set_title("Joint Velocities")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_joint_torques(traj: Trajectory, ax=None):
    """绘制关节力矩随时间变化"""
    if not traj.joint_torques:
        if ax is not None:
            ax.text(0.5, 0.5, "No torque data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="gray")
            ax.set_title("Joint Torques (N/A)")
        return
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    t = np.array(traj.timestamps)
    torques = np.array(traj.joint_torques)
    for j in range(min(NUM_JOINTS, torques.shape[1])):
        ax.plot(t, torques[:, j], label=JOINT_NAMES[j])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title("Joint Torques")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_ee_trajectory_3d(traj: Trajectory, ax=None):
    """绘制末端执行器3D轨迹"""
    if not traj.ee_positions:
        if ax is not None:
            ax.text2D(0.5, 0.5, "No EE position data", transform=ax.transAxes,
                      ha="center", va="center", fontsize=14, color="gray")
            ax.set_title("End-Effector Trajectory (N/A)")
        return
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
    ee_pos = np.array(traj.ee_positions)
    ax.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], "b-", linewidth=1.5)
    ax.scatter(*ee_pos[0], c="green", s=80, marker="o", label="Start")
    ax.scatter(*ee_pos[-1], c="red", s=80, marker="x", label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("End-Effector 3D Trajectory")
    ax.legend()


def plot_ee_position(traj: Trajectory, ax=None):
    """绘制末端执行器位置随时间变化"""
    if not traj.ee_positions:
        if ax is not None:
            ax.text(0.5, 0.5, "No EE position data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="gray")
            ax.set_title("End-Effector Position (N/A)")
        return
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    t = np.array(traj.timestamps)
    ee_pos = np.array(traj.ee_positions)
    ax.plot(t, ee_pos[:, 0], label="X")
    ax.plot(t, ee_pos[:, 1], label="Y")
    ax.plot(t, ee_pos[:, 2], label="Z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("End-Effector Position")
    ax.legend()
    ax.grid(True, alpha=0.3)


def show_camera_images(traj: Trajectory, step_idx: int = 0):
    """显示指定时间步的摄像头图像"""
    has_overhead = bool(traj.overhead_images)
    has_ee = bool(traj.ee_images)
    if not has_overhead and not has_ee:
        print("No camera images in trajectory.")
        return

    ncols = int(has_overhead) + int(has_ee)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    col = 0
    if has_overhead:
        axes[col].imshow(traj.overhead_images[step_idx])
        axes[col].set_title(f"Overhead Camera (step {step_idx})")
        axes[col].axis("off")
        col += 1
    if has_ee:
        axes[col].imshow(traj.ee_images[step_idx])
        axes[col].set_title(f"End-Effector Camera (step {step_idx})")
        axes[col].axis("off")

    plt.tight_layout()


def visualize_single(traj: Trajectory, title: str = "Trajectory"):
    """可视化单条轨迹的完整面板"""
    has_ee = bool(traj.ee_positions)
    has_torques = bool(traj.joint_torques)
    has_images = bool(traj.overhead_images) or bool(traj.ee_images)

    # Create subplot layout
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{title}  ({len(traj)} steps)", fontsize=14)

    # Row 1: joint angles, joint velocities
    ax1 = fig.add_subplot(2, 3, 1)
    plot_joint_angles(traj, ax1)

    ax2 = fig.add_subplot(2, 3, 2)
    plot_joint_velocities(traj, ax2)

    ax3 = fig.add_subplot(2, 3, 3)
    plot_joint_torques(traj, ax3)

    # Row 2: EE position, EE 3D, stats
    ax4 = fig.add_subplot(2, 3, 4)
    plot_ee_position(traj, ax4)

    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    plot_ee_trajectory_3d(traj, ax5)

    # Stats panel
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    stats_text = f"Steps: {len(traj)}\n"
    if traj.timestamps:
        duration = traj.timestamps[-1] - traj.timestamps[0]
        stats_text += f"Duration: {duration:.3f} s\n"
    if has_ee:
        ee_pos = np.array(traj.ee_positions)
        path_length = np.sum(np.linalg.norm(np.diff(ee_pos, axis=0), axis=1))
        stats_text += f"EE path length: {path_length:.4f} m\n"
        stats_text += f"EE X range: [{ee_pos[:, 0].min():.3f}, {ee_pos[:, 0].max():.3f}]\n"
        stats_text += f"EE Y range: [{ee_pos[:, 1].min():.3f}, {ee_pos[:, 1].max():.3f}]\n"
        stats_text += f"EE Z range: [{ee_pos[:, 2].min():.3f}, {ee_pos[:, 2].max():.3f}]\n"
    if has_torques:
        torques = np.array(traj.joint_torques)
        stats_text += f"Max torque: {np.abs(torques).max():.2f} N·m\n"
    if has_images:
        n_overhead = len(traj.overhead_images)
        n_ee = len(traj.ee_images)
        stats_text += f"Overhead images: {n_overhead}\n"
        stats_text += f"EE images: {n_ee}\n"

    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace")
    ax6.set_title("Statistics")

    plt.tight_layout()
    return fig


def visualize_dataset_summary(data_dir: Path):
    """可视化整个数据集的统计概览"""
    traj_files = list_trajectories(data_dir)
    if not traj_files:
        print(f"No trajectory files found in {data_dir}")
        return

    print(f"Found {len(traj_files)} trajectories in {data_dir}")

    # Load all trajectories (metadata only for large datasets)
    lengths = []
    durations = []
    all_ee_paths = []

    for f in traj_files:
        traj = load_trajectory(f)
        lengths.append(len(traj))
        if traj.timestamps and len(traj) > 1:
            durations.append(traj.timestamps[-1] - traj.timestamps[0])
        if traj.ee_positions:
            all_ee_paths.append(np.array(traj.ee_positions))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Dataset Summary ({len(traj_files)} trajectories)", fontsize=14)

    # Trajectory lengths histogram
    axes[0].hist(lengths, bins=20, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Trajectory Length (steps)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Trajectory Lengths")
    axes[0].axvline(np.mean(lengths), color="red", linestyle="--",
                    label=f"Mean: {np.mean(lengths):.0f}")
    axes[0].legend()

    # Durations histogram
    if durations:
        axes[1].hist(durations, bins=20, edgecolor="black", alpha=0.7, color="orange")
        axes[1].set_xlabel("Duration (s)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Trajectory Durations")
        axes[1].axvline(np.mean(durations), color="red", linestyle="--",
                        label=f"Mean: {np.mean(durations):.3f}s")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No duration data", transform=axes[1].transAxes,
                     ha="center", va="center")

    # 3D EE paths overlay
    if all_ee_paths:
        ax3d = fig.add_subplot(1, 3, 3, projection="3d")
        axes[2].set_visible(False)
        for i, ee_pos in enumerate(all_ee_paths):
            ax3d.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2],
                      alpha=0.5, linewidth=0.8)
        ax3d.set_xlabel("X (m)")
        ax3d.set_ylabel("Y (m)")
        ax3d.set_zlabel("Z (m)")
        ax3d.set_title("All EE Trajectories")
    else:
        axes[2].text(0.5, 0.5, "No EE position data", transform=axes[2].transAxes,
                     ha="center", va="center")

    plt.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trajectory dataset")
    parser.add_argument("--data_dir", type=str, default="datasets",
                        help="Dataset directory (default: datasets)")
    parser.add_argument("--traj_idx", type=int, default=0,
                        help="Trajectory index to visualize (default: 0)")
    parser.add_argument("--npz_file", type=str, default=None,
                        help="Directly specify a .npz or .json trajectory file")
    parser.add_argument("--show_all", action="store_true",
                        help="Show dataset summary instead of single trajectory")
    parser.add_argument("--show_images", action="store_true",
                        help="Also show camera images (if available)")
    parser.add_argument("--image_step", type=int, default=0,
                        help="Step index for camera image display (default: 0)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to file instead of showing")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.show_all:
        fig = visualize_dataset_summary(data_dir)
        if fig is not None:
            if args.save:
                fig.savefig(args.save, dpi=150, bbox_inches="tight")
                print(f"Saved to {args.save}")
            else:
                plt.show()
        return

    # Load single trajectory
    if args.npz_file:
        traj = load_trajectory(Path(args.npz_file))
        title = Path(args.npz_file).stem
    else:
        traj = load_trajectory_by_index(data_dir, args.traj_idx)
        title = f"traj_{args.traj_idx + 1:04d}"

    print(f"Loaded trajectory: {title}")
    print(f"  Steps: {len(traj)}")
    print(f"  Has EE positions: {bool(traj.ee_positions)}")
    print(f"  Has EE orientations: {bool(traj.ee_orientations)}")
    print(f"  Has joint torques: {bool(traj.joint_torques)}")
    print(f"  Has overhead images: {bool(traj.overhead_images)}")
    print(f"  Has EE images: {bool(traj.ee_images)}")

    fig = visualize_single(traj, title=title)

    if args.show_images:
        show_camera_images(traj, step_idx=args.image_step)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
