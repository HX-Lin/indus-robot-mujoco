"""
可视化数据集中的单个 episode
生成视频: 左侧两个摄像头图像上下拼接, 右侧关节角度随时间变化曲线

用法:
    python visualize_episode.py --data_dir datasets_lerobot --episode 0 --output episode_0.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from mr2010_robot import JOINT_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a single episode from LeRobot dataset")
    parser.add_argument("--data_dir", type=str, default="datasets_lerobot")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=25)
    return parser.parse_args()


def render_joint_plot(timestamps, joint_positions, current_idx, img_height, img_width):
    """渲染关节角度曲线图为 numpy 图像"""
    fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)

    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for j in range(6):
        ax.plot(timestamps, np.degrees(joint_positions[:, j]),
                color=colors[j], linewidth=1.2, label=JOINT_NAMES[j])

    # 当前时刻竖线
    t_now = timestamps[current_idx]
    ax.axvline(x=t_now, color="red", linewidth=1.5, linestyle="--", alpha=0.8)

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Joint Angle (deg)", fontsize=10)
    ax.set_title("Joint Angles", fontsize=12)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)

    # Render to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()  # RGBA -> RGB
    plt.close(fig)

    # Resize to exact target
    img = cv2.resize(img, (img_width, img_height))
    return img


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    ep_idx = args.episode

    # Load parquet data
    parquet_path = data_dir / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(parquet_path)
    ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index").reset_index(drop=True)

    if len(ep_df) == 0:
        print(f"Episode {ep_idx} not found in dataset")
        return

    num_frames = len(ep_df)
    print(f"Episode {ep_idx}: {num_frames} frames")

    # Extract joint positions
    timestamps = ep_df["timestamp"].values.astype(np.float32)
    joint_positions = np.array(ep_df["observation.state"].tolist(), dtype=np.float32)

    # Check image directories
    overhead_dir = data_dir / "images" / "observation.images.overhead" / f"episode-{ep_idx:06d}"
    ee_dir = data_dir / "images" / "observation.images.ee" / f"episode-{ep_idx:06d}"
    has_overhead = overhead_dir.exists()
    has_ee = ee_dir.exists()
    print(f"  Overhead images: {has_overhead}")
    print(f"  EE images: {has_ee}")

    if not has_overhead and not has_ee:
        print("No images found, cannot generate video")
        return

    # Read first image to get dimensions
    first_img = np.array(Image.open(overhead_dir / "frame-000000.png"))
    cam_h, cam_w = first_img.shape[:2]
    print(f"  Image size: {cam_w}x{cam_h}")

    # Video layout: left = cameras stacked vertically, right = joint plot
    # Left panel: cam_w x (cam_h * 2)
    # Right panel: same height as left
    left_w = cam_w
    left_h = cam_h * 2
    right_w = cam_w  # same width as camera
    right_h = left_h
    total_w = left_w + right_w
    total_h = left_h

    # Output path
    if args.output is None:
        args.output = f"episode_{ep_idx}.mp4"
    output_path = str(data_dir / args.output)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, args.fps, (total_w, total_h))

    print(f"  Video: {total_w}x{total_h} @ {args.fps}fps -> {output_path}")
    print(f"  Generating {num_frames} frames...")

    for i in range(num_frames):
        # Load camera images
        overhead_path = overhead_dir / f"frame-{i:06d}.png"
        ee_path = ee_dir / f"frame-{i:06d}.png"

        if has_overhead and overhead_path.exists():
            overhead_img = np.array(Image.open(overhead_path))
        else:
            overhead_img = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        if has_ee and ee_path.exists():
            ee_img = np.array(Image.open(ee_path))
        else:
            ee_img = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        # Add labels
        oh_labeled = overhead_img.copy()
        ee_labeled = ee_img.copy()
        cv2.putText(oh_labeled, "Front Camera", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(ee_labeled, "EE Camera", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Stack cameras vertically
        left_panel = np.vstack([oh_labeled, ee_labeled])

        # Render joint plot
        right_panel = render_joint_plot(timestamps, joint_positions, i, right_h, right_w)

        # Combine
        frame = np.hstack([left_panel, right_panel])

        # Add frame info
        t = timestamps[i]
        cv2.putText(frame, f"t={t:.2f}s  frame={i}/{num_frames}",
                    (10, total_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Write (convert RGB -> BGR for cv2)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if (i + 1) % 50 == 0 or i == 0:
            print(f"    Frame {i + 1}/{num_frames}")

    writer.release()
    print(f"\nVideo saved to: {output_path}")


if __name__ == "__main__":
    main()
