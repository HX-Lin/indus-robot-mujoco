"""
焊缝轨迹数据集批量生成脚本 — 输出 LeRobot v3.0 格式

生成沿 workpiece 表面 S 型弧线的焊接轨迹数据集。
每条轨迹开始前随机化 workpiece 位置和朝向。

用法:
    # 生成50条S型弧线轨迹（含图像, 50fps保存）
    python generate_dataset.py --num_trajs 50 --output_dir datasets_lerobot

    # 不渲染图像（仅状态数据，速度更快）
    python generate_dataset.py --num_trajs 50 --no_images --output_dir datasets_lerobot
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco

from mr2010_robot import (
    create_base_scene, NUM_JOINTS,
    PLATFORM_POS, WORKPIECE_POS_RANGE, WORKPIECE_ROT_RANGE,
    euler_to_quat,
)
from ik_solver import DampedLeastSquaresIK
from il_env import Trajectory
from lerobot_collector import LeRobotNativeCollector
from seam_generator import (
    SeamPath,
    generate_trajectory,
    create_scurve_seam_on_workpiece,
)


SETTLE_STEPS = 2000  # 仿真步数，让 workpiece 自然落到 platform 上


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate S-curve welding seam trajectory dataset (LeRobot v3.0 format)"
    )
    parser.add_argument(
        "--num_trajs", type=int, default=50,
        help="Number of trajectories to generate (default: 50)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="datasets_lerobot",
        help="Output directory (default: datasets_lerobot)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--num_points", type=int, default=100,
        help="Number of waypoints per seam (default: 100)",
    )
    parser.add_argument(
        "--sim_steps", type=int, default=50,
        help="Simulation steps per waypoint (default: 50)",
    )
    parser.add_argument(
        "--record_fps", type=int, default=50,
        help="Recording FPS for saved data (default: 50)",
    )
    parser.add_argument(
        "--no_images", action="store_true",
        help="Disable camera image rendering (images are enabled by default)",
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[480, 640],
        metavar=("H", "W"),
        help="Image size (height width) (default: 480 640)",
    )
    parser.add_argument(
        "--task", type=str, default="welding S-curve seam trajectory tracking",
        help="Task description (default: welding S-curve seam trajectory tracking)",
    )
    parser.add_argument(
        "--robot_type", type=str, default="MR2010",
        help="Robot type identifier (default: MR2010)",
    )
    parser.add_argument(
        "--approach_steps", type=int, default=500,
        help="Simulation steps for approach phase (init -> seam start) (default: 500)",
    )
    return parser.parse_args()


def settle_workpiece(model, data, steps=SETTLE_STEPS):
    """让 workpiece 自然落到 platform 上并稳定"""
    data.ctrl[:model.nu] = data.qpos[:model.nu]
    for _ in range(steps):
        mujoco.mj_step(model, data)


def randomize_workpiece(model, data, rng: np.random.Generator, max_retries: int = 10):
    """随机化 workpiece 的位置和朝向 (通过 freejoint qpos)

    修改 workpiece freejoint 的 qpos (3 pos + 4 quat)，
    然后重新仿真让它落到 platform 上稳定。
    如果工件滑落平台则重试。

    Returns:
        settled_qpos: 稳定后的完整 qpos 快照
    """
    wp_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "workpiece_joint")
    qpos_addr = model.jnt_qposadr[wp_jnt_id]
    wp_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workpiece")

    workpiece_base_pos = PLATFORM_POS
    surface_z = workpiece_base_pos[2] + 0.54

    for attempt in range(max_retries):
        # 随机位置 (相对于 platform)
        rel_x = rng.uniform(*WORKPIECE_POS_RANGE["x"])
        rel_y = rng.uniform(*WORKPIECE_POS_RANGE["y"])
        pos = np.array([
            workpiece_base_pos[0] + rel_x,
            workpiece_base_pos[1] + rel_y,
            surface_z,
        ])

        # 工件姿态 = 绕X轴翻转180° (正面朝上) + 绕Z轴随机yaw旋转
        # q_flip = (0, 1, 0, 0) 即绕X轴180°
        # q_final = q_yaw * q_flip (Hamilton乘法)
        yaw = rng.uniform(*WORKPIECE_ROT_RANGE["yaw"])
        q_yaw = np.array(euler_to_quat(0.0, 0.0, yaw))  # (w, x, y, z)
        q_flip = np.array([0.0, 1.0, 0.0, 0.0])          # (w, x, y, z)
        w1, x1, y1, z1 = q_yaw
        w2, x2, y2, z2 = q_flip
        quat = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

        # 重置仿真状态
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_addr:qpos_addr + 3] = pos
        data.qpos[qpos_addr + 3:qpos_addr + 7] = quat
        data.qpos[:NUM_JOINTS] = 0.0
        data.ctrl[:NUM_JOINTS] = 0.0
        mujoco.mj_forward(model, data)

        # 让 workpiece 稳定
        settle_workpiece(model, data)

        # 验证: 正面朝上 (body z-axis 朝下) 且未滑落平台
        wp_pos = data.xpos[wp_body_id]
        wp_z = data.xmat[wp_body_id].reshape(3, 3)[:, 2]
        face_up = wp_z[2] < -0.999  # strict: < 2.5° tilt, avoid platform protrusions
        on_platform = 0.3 < wp_pos[2] < 0.6

        if face_up and on_platform:
            return data.qpos.copy()

        print(f"    Retry {attempt + 1}: workpiece invalid "
              f"(z={wp_pos[2]:.3f}, face_up={face_up}), re-randomizing...")

    raise RuntimeError("Failed to place workpiece on platform after max retries")


def main():
    args = parse_args()
    render_images = not args.no_images

    # 计算 record_interval: 仿真频率 / 保存频率
    sim_dt = 0.002  # MuJoCo default timestep
    sim_fps = int(1.0 / sim_dt)
    record_interval = max(1, sim_fps // args.record_fps)
    actual_record_fps = sim_fps // record_interval

    print("=" * 60)
    print("Welding S-Curve Trajectory Dataset Generator (LeRobot v3.0)")
    print("=" * 60)
    print(f"  Trajectories:  {args.num_trajs}")
    print(f"  Seam type:     S-curve (on workpiece surface)")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Seed:          {args.seed}")
    print(f"  Waypoints:     {args.num_points}")
    print(f"  Sim steps/wp:  {args.sim_steps}")
    print(f"  Sim FPS:       {sim_fps}")
    print(f"  Record FPS:    {actual_record_fps} (interval={record_interval})")
    print(f"  Render images: {render_images}")
    print(f"  Image size:    {args.image_size}")
    print(f"  Approach steps:{args.approach_steps}")
    print(f"  Randomize WP:  True")
    print("=" * 60)

    # Create scene with platform + workpiece
    print("\nCreating scene (with platform + workpiece)...")
    spec = create_base_scene(
        with_actuators=True, with_platform=True, with_workpiece=True,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    ik_solver = DampedLeastSquaresIK(model, data)
    print(f"Model: {model.ngeom} geoms, {model.njnt} joints, {model.nu} actuators")

    # Setup LeRobot native data collector (fps = actual recording fps)
    collector = LeRobotNativeCollector(
        repo_id="local/mr2010-welding",
        root=args.output_dir,
        fps=actual_record_fps,
        task_description=args.task,
        robot_type=args.robot_type,
        has_images=render_images,
        image_size=tuple(args.image_size),
    )

    # Random generator
    rng = np.random.default_rng(args.seed)

    # Track statistics
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "traj_lengths": [],
        "durations": [],
        "mean_ik_errors": [],
        "max_ik_errors": [],
    }

    wp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workpiece")
    total_start_time = time.time()

    for traj_idx in range(args.num_trajs):
        stats["total"] += 1

        # Randomize workpiece position & settle
        settled_qpos = randomize_workpiece(model, data, rng)
        wp_pos = data.xpos[wp_id].copy()

        # Create S-curve seam on the newly positioned workpiece
        seam = create_scurve_seam_on_workpiece(model, data, rng)
        waypoints = seam.sample_points(args.num_points)

        print(f"\n[{traj_idx + 1}/{args.num_trajs}] "
              f"Generating S-curve seam trajectory...")
        print(f"  Workpiece pos: {wp_pos}")
        print(f"  Seam: {waypoints[0]} -> {waypoints[-1]}")

        # Restore settled state (reset robot joints but preserve workpiece position)
        data.qpos[:] = settled_qpos
        data.ctrl[:NUM_JOINTS] = 0.0
        mujoco.mj_forward(model, data)

        traj_start_time = time.time()

        try:
            traj, ik_errors = generate_trajectory(
                seam=seam,
                model=model,
                data=data,
                ik_solver=ik_solver,
                num_points=args.num_points,
                sim_steps_per_point=args.sim_steps,
                render_images=render_images,
                image_size=tuple(args.image_size),
                restore_qpos=settled_qpos,
                record_interval=record_interval,
                approach_steps=args.approach_steps,
            )

            traj_elapsed = time.time() - traj_start_time

            errors = np.array(ik_errors)

            duration = traj.timestamps[-1] - traj.timestamps[0] if len(traj) > 1 else 0.0

            print(f"  Steps: {len(traj)}, Duration: {duration:.3f}s, "
                  f"Gen time: {traj_elapsed:.1f}s")
            print(f"  IK error: mean={errors.mean():.5f}m, max={errors.max():.5f}m")

            # Quality gate: reject trajectories with max IK error > 5mm
            ik_max_threshold = 0.005  # 5mm
            if errors.max() > ik_max_threshold:
                stats["failed"] += 1
                print(f"  REJECTED: max IK error {errors.max()*1000:.1f}mm > {ik_max_threshold*1000:.0f}mm threshold")
                continue

            # Save trajectory
            collector.add_trajectory(traj)

            # Update stats
            stats["success"] += 1
            stats["traj_lengths"].append(len(traj))
            stats["durations"].append(duration)
            stats["mean_ik_errors"].append(float(errors.mean()))
            stats["max_ik_errors"].append(float(errors.max()))

        except Exception as e:
            stats["failed"] += 1
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_elapsed = time.time() - total_start_time

    # Finalize dataset
    print("\nSaving combined dataset...")
    collector.close()

    # Save metadata/statistics
    summary = {
        "config": {
            "num_trajs": args.num_trajs,
            "seam_type": "scurve",
            "seed": args.seed,
            "num_points": args.num_points,
            "sim_steps_per_point": args.sim_steps,
            "record_fps": actual_record_fps,
            "record_interval": record_interval,
            "randomize_workpiece": True,
        },
        "results": {
            "total_attempted": stats["total"],
            "successful": stats["success"],
            "failed": stats["failed"],
        },
        "statistics": {},
    }

    if stats["traj_lengths"]:
        summary["statistics"] = {
            "traj_length": {
                "mean": float(np.mean(stats["traj_lengths"])),
                "std": float(np.std(stats["traj_lengths"])),
                "min": int(np.min(stats["traj_lengths"])),
                "max": int(np.max(stats["traj_lengths"])),
            },
            "duration_s": {
                "mean": float(np.mean(stats["durations"])),
                "std": float(np.std(stats["durations"])),
                "min": float(np.min(stats["durations"])),
                "max": float(np.max(stats["durations"])),
            },
            "ik_error_mean_m": {
                "mean": float(np.mean(stats["mean_ik_errors"])),
                "std": float(np.std(stats["mean_ik_errors"])),
                "min": float(np.min(stats["mean_ik_errors"])),
                "max": float(np.max(stats["mean_ik_errors"])),
            },
            "ik_error_max_m": {
                "mean": float(np.mean(stats["max_ik_errors"])),
                "std": float(np.std(stats["max_ik_errors"])),
                "min": float(np.min(stats["max_ik_errors"])),
                "max": float(np.max(stats["max_ik_errors"])),
            },
        }

    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total time:     {total_elapsed:.1f}s")
    print(f"  Successful:     {stats['success']}/{stats['total']}")
    print(f"  Failed:         {stats['failed']}/{stats['total']}")
    if stats["traj_lengths"]:
        print(f"  Avg traj len:   {np.mean(stats['traj_lengths']):.0f} steps")
        print(f"  Avg duration:   {np.mean(stats['durations']):.3f}s")
        print(f"  Avg IK error:   {np.mean(stats['mean_ik_errors']):.5f}m")
        print(f"  Max IK error:   {np.max(stats['max_ik_errors']):.5f}m")
    print(f"  Output dir:     {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
