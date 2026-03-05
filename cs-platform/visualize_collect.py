"""
可视化数据采集脚本

在 MuJoCo viewer 中实时观看机械臂沿 workpiece S 型弧线运行焊接轨迹，
同时采集数据集。方便直观验证轨迹是否正确。

用法:
    # 可视化采集 3 条轨迹（默认）
    python visualize_collect.py

    # 指定轨迹数量和输出目录
    python visualize_collect.py --num_trajs 5 --output_dir datasets_vis

    # 仅可视化不保存数据
    python visualize_collect.py --no_save

    # 调整播放速度（倍率，<1 慢放，>1 快放）
    python visualize_collect.py --speed 0.5
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

from mr2010_robot import create_base_scene, NUM_JOINTS
from ik_solver import DampedLeastSquaresIK
from seam_generator import (
    SCurveSeam,
    create_scurve_seam_on_workpiece,
    get_workpiece_surface_bounds,
    _solve_ik_offline,
)
from il_env import Trajectory
from lerobot_collector import LeRobotDataCollector


SETTLE_STEPS = 2000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize S-curve welding trajectory collection"
    )
    parser.add_argument(
        "--num_trajs", type=int, default=3,
        help="Number of trajectories to collect (default: 3)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="datasets_vis",
        help="Output directory for dataset (default: datasets_vis)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--num_points", type=int, default=80,
        help="Number of waypoints per seam (default: 80)",
    )
    parser.add_argument(
        "--sim_steps", type=int, default=50,
        help="Simulation steps per waypoint (default: 50)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="Only visualize, do not save dataset",
    )
    parser.add_argument(
        "--no_images", action="store_true",
        help="Do not save camera images in dataset",
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[480, 640],
        metavar=("H", "W"),
        help="Image size for dataset (default: 480 640)",
    )
    return parser.parse_args()


def settle_workpiece(model, data, steps=SETTLE_STEPS):
    """让 workpiece 自然落到 platform 上并稳定"""
    data.ctrl[:model.nu] = data.qpos[:model.nu]
    for _ in range(steps):
        mujoco.mj_step(model, data)


def add_seam_markers(spec, seam, num_markers=30):
    """在场景中添加焊缝路径标记点（小球），方便可视化

    Args:
        spec: MjSpec (compile 前调用)
        seam: SCurveSeam 对象
        num_markers: 标记点数量
    """
    pts = seam.sample_points(num_markers)
    for i, p in enumerate(pts):
        body = spec.worldbody.add_body(name=f"seam_marker_{i}")
        body.pos = tuple(p)
        body.add_geom(
            name=f"seam_marker_geom_{i}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.003, 0, 0],
            rgba=(1.0, 0.2, 0.2, 0.8),
            contype=0,
            conaffinity=0,
        )


def main():
    args = parse_args()

    print("=" * 60)
    print("Visualize S-Curve Welding Trajectory Collection")
    print("=" * 60)
    print(f"  Trajectories:  {args.num_trajs}")
    print(f"  Speed:         {args.speed}x")
    print(f"  Save dataset:  {not args.no_save}")
    if not args.no_save:
        print(f"  Output dir:    {args.output_dir}")
    print("=" * 60)

    rng = np.random.default_rng(args.seed)

    # --- Phase 1: 创建场景，让 workpiece 稳定 ---
    print("\nCreating scene...")
    spec_tmp = create_base_scene(
        with_actuators=True, with_platform=True, with_workpiece=True,
    )
    model_tmp = spec_tmp.compile()
    data_tmp = mujoco.MjData(model_tmp)
    settle_workpiece(model_tmp, data_tmp)

    # 获取 workpiece 稳定后的位置
    wp_id = mujoco.mj_name2id(model_tmp, mujoco.mjtObj.mjOBJ_BODY, "workpiece")
    wp_pos = data_tmp.xpos[wp_id].copy()
    wp_quat = data_tmp.xquat[wp_id].copy()
    print(f"  Workpiece settled at: {wp_pos}")

    bounds = get_workpiece_surface_bounds(model_tmp, data_tmp)
    print(f"  Surface: X=[{bounds['x_min']:.3f}, {bounds['x_max']:.3f}], "
          f"Y=[{bounds['y_min']:.3f}, {bounds['y_max']:.3f}], Z={bounds['z']:.3f}")

    # LeRobot collector
    collector = None
    if not args.no_save:
        collector = LeRobotDataCollector(
            save_dir=args.output_dir,
            fps=int(1.0 / model_tmp.opt.timestep),
            task_description="welding S-curve seam trajectory tracking",
            robot_type="MR2010",
        )

    # 用于离线渲染数据集图像的 renderer
    dataset_renderer = None
    has_overhead = False
    has_ee_cam = False
    render_images = (not args.no_save) and (not args.no_images)
    if render_images:
        dataset_renderer = mujoco.Renderer(
            model_tmp, height=args.image_size[0], width=args.image_size[1],
        )
        for i in range(model_tmp.ncam):
            cam_name = model_tmp.camera(i).name
            if cam_name == "overhead_cam":
                has_overhead = True
            elif cam_name == "ee_cam":
                has_ee_cam = True

    # --- Phase 2: 逐条轨迹采集 ---
    for traj_idx in range(args.num_trajs):
        print(f"\n{'='*60}")
        print(f"Trajectory {traj_idx + 1}/{args.num_trajs}")
        print(f"{'='*60}")

        # 生成 S 型弧线
        seam = create_scurve_seam_on_workpiece(model_tmp, data_tmp, rng)
        pts = seam.sample_points(args.num_points)
        print(f"  S-curve: body_pos={seam.body_pos}, "
              f"t_range=[{seam.t_start:.2f}, {seam.t_end:.2f}]")
        print(f"  Start: {pts[0]}")
        print(f"  End:   {pts[-1]}")

        # 重新构建场景（带焊缝标记点）
        spec = create_base_scene(
            with_actuators=True, with_platform=True, with_workpiece=True,
        )
        add_seam_markers(spec, seam, num_markers=40)
        model = spec.compile()
        data = mujoco.MjData(model)

        # 将 workpiece 放到稳定位置（直接设置 freejoint qpos，跳过 settle）
        wp_id_new = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workpiece")
        # freejoint qpos = [x, y, z, qw, qx, qy, qz]
        # 找到 workpiece freejoint 在 qpos 中的偏移
        wp_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "workpiece_joint")
        wp_qpos_adr = model.jnt_qposadr[wp_jnt_id]
        data.qpos[wp_qpos_adr:wp_qpos_adr + 3] = wp_pos
        data.qpos[wp_qpos_adr + 3:wp_qpos_adr + 7] = wp_quat
        data.ctrl[:NUM_JOINTS] = data.qpos[:NUM_JOINTS]
        mujoco.mj_forward(model, data)

        settled_qpos = data.qpos.copy()

        # 离线 IK 求解
        print("  Solving IK...")
        waypoints = seam.sample_points(args.num_points)
        orientations = seam.sample_orientations(args.num_points)
        joint_targets, ik_errors = _solve_ik_offline(waypoints, model, orientations)
        joint_targets = np.array(joint_targets)
        ik_errors = np.array(ik_errors)
        print(f"  IK errors: mean={ik_errors.mean():.5f}m, max={ik_errors.max():.5f}m")

        # 密集插值
        total_steps = args.num_points * args.sim_steps
        t_wp = np.linspace(0, 1, args.num_points)
        t_dense = np.linspace(0, 1, total_steps)
        q_dense = np.zeros((total_steps, NUM_JOINTS))
        for j in range(NUM_JOINTS):
            q_dense[:, j] = np.interp(t_dense, t_wp, joint_targets[:, j])

        dt = model.opt.timestep
        site_id = model.site("tool_tip").id

        # 准备 Trajectory
        traj = Trajectory()

        # 恢复到稳定状态
        data.qpos[:] = settled_qpos
        data.qpos[:NUM_JOINTS] = q_dense[0]
        data.ctrl[:NUM_JOINTS] = q_dense[0]
        mujoco.mj_forward(model, data)

        # 打开 viewer 并实时播放
        print(f"  Playing trajectory ({total_steps} steps)...")
        print(f"  [Close viewer window to skip to next trajectory]")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            step_i = 0
            while viewer.is_running() and step_i < total_steps:
                # 设置关节角
                data.qpos[:NUM_JOINTS] = q_dense[step_i]
                data.ctrl[:NUM_JOINTS] = q_dense[step_i]
                # 保持 workpiece 位置
                data.qpos[wp_qpos_adr:wp_qpos_adr + 3] = wp_pos
                data.qpos[wp_qpos_adr + 3:wp_qpos_adr + 7] = wp_quat

                mujoco.mj_forward(model, data)

                # 计算速度
                if step_i > 0:
                    qvel = (q_dense[step_i] - q_dense[step_i - 1]) / dt
                else:
                    qvel = np.zeros(NUM_JOINTS)

                ee_pos = data.site_xpos[site_id].copy()
                ee_mat = data.site_xmat[site_id].reshape(3, 3).copy()
                torques = data.qfrc_actuator[:NUM_JOINTS].copy()

                # 离线渲染数据集图片
                overhead_img = None
                ee_img = None
                if dataset_renderer is not None:
                    # 需要用 model_tmp 的 renderer，但它和当前 model 不同
                    # 这里用当前 model 重建 renderer
                    pass

                traj.add_step(
                    t=step_i * dt,
                    qpos=q_dense[step_i].copy(),
                    qvel=qvel.copy(),
                    action=q_dense[step_i].copy(),
                    ee_pos=ee_pos,
                    ee_mat=ee_mat,
                    torques=torques,
                    overhead_img=overhead_img,
                    ee_img=ee_img,
                )

                viewer.sync()

                # 按 speed 调整等待时间
                time.sleep(dt / args.speed)
                step_i += 1

        # 打印轨迹统计
        ee_positions = np.array(traj.ee_positions)
        print(f"  Collected {len(traj)} steps")
        print(f"  EE X: [{ee_positions[:,0].min():.4f}, {ee_positions[:,0].max():.4f}]")
        print(f"  EE Y: [{ee_positions[:,1].min():.4f}, {ee_positions[:,1].max():.4f}]")
        print(f"  EE Z: [{ee_positions[:,2].min():.4f}, {ee_positions[:,2].max():.4f}]")

        # 保存到 LeRobot 数据集
        if collector is not None and len(traj) > 0:
            collector.add_episode(traj)
            print(f"  Saved episode {traj_idx + 1}")

    # 完成
    if collector is not None:
        collector.finalize()
        print(f"\nDataset saved to {args.output_dir}/")

    if dataset_renderer is not None:
        dataset_renderer.close()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
