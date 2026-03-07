"""
策略模型评估脚本 — 在 MuJoCo 仿真中闭环运行训练好的策略

支持所有已注册的策略模型（ACT, VITA, ...）。

流程:
  1. 通过 --policy 参数选择模型类型，加载 checkpoint
  2. 随机化 workpiece 位置，生成 ground-truth S-curve 焊缝
  3. 闭环运行策略（图像 + 关节状态 → 关节位置目标）
  4. 对比末端执行器轨迹与 ground-truth 焊缝，输出跟踪误差指标

用法:
    python evaluate.py --policy act --checkpoint outputs/act/last
    python evaluate.py --policy vita --checkpoint outputs/vita/best
    python evaluate.py --render   # 打开 MuJoCo viewer 可视化
"""

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mujoco
import torch

from mr2010_robot import create_base_scene, NUM_JOINTS
from seam_generator import create_scurve_seam_on_workpiece, _rotmat_to_axisangle
from ik_solver import DampedLeastSquaresIK


SETTLE_STEPS = 2000
IMAGE_SIZE = (480, 640)  # (H, W), must match training
VIDEO_SIZE = (480, 640)  # (H, W) per sub-view in video


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate policy in MuJoCo")
    parser.add_argument("--policy", type=str, default="act",
                        help="Policy type (act, vita, ...)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint dir (default: outputs/<policy>/best)")
    parser.add_argument("--data_dir", type=str, default="datasets_lerobot")
    parser.add_argument("--repo_id", type=str, default="local/mr2010-welding")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Max policy steps per episode (at control_freq)")
    parser.add_argument("--control_freq", type=int, default=50,
                        help="Policy control frequency in Hz")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer")
    parser.add_argument("--video_dir", type=str, default="outputs/eval_videos",
                        help="Directory to save evaluation videos")
    parser.add_argument("--action_type", type=str, default="action",
                        choices=["action", "action.ee_delta"],
                        help="Action type: 'action' (joint abs) or 'action.ee_delta' (EE pose delta)")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = f"outputs/{args.policy}/best"
    return args


def settle_workpiece(model, data, steps=SETTLE_STEPS):
    """Hold robot fixed while workpiece settles on platform."""
    data.ctrl[:model.nu] = data.qpos[:model.nu]
    for _ in range(steps):
        mujoco.mj_step(model, data)


def randomize_workpiece(model, data, rng):
    """Randomize workpiece position/orientation and settle."""
    from mr2010_robot import (
        PLATFORM_POS, WORKPIECE_POS_RANGE, WORKPIECE_ROT_RANGE, euler_to_quat,
    )

    for attempt in range(10):
        mujoco.mj_resetData(model, data)

        # Random position on platform
        rel_x = rng.uniform(*WORKPIECE_POS_RANGE["x"])
        rel_y = rng.uniform(*WORKPIECE_POS_RANGE["y"])
        wp_pos = (
            PLATFORM_POS[0] + rel_x,
            PLATFORM_POS[1] + rel_y,
            PLATFORM_POS[2] + 0.54,
        )

        # Random yaw + flip
        yaw = rng.uniform(*WORKPIECE_ROT_RANGE["yaw"])
        q_yaw = euler_to_quat(0.0, 0.0, yaw)
        q_flip = (0.0, 1.0, 0.0, 0.0)
        w1, x1, y1, z1 = q_yaw
        w2, x2, y2, z2 = q_flip
        quat = (
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        )

        # Set workpiece freejoint (after 6 robot joints)
        data.qpos[NUM_JOINTS] = wp_pos[0]
        data.qpos[NUM_JOINTS + 1] = wp_pos[1]
        data.qpos[NUM_JOINTS + 2] = wp_pos[2]
        data.qpos[NUM_JOINTS + 3] = quat[0]
        data.qpos[NUM_JOINTS + 4] = quat[1]
        data.qpos[NUM_JOINTS + 5] = quat[2]
        data.qpos[NUM_JOINTS + 6] = quat[3]

        settle_workpiece(model, data)

        # Validate: workpiece face-up and on platform
        wp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workpiece")
        wp_z_axis = data.xmat[wp_id].reshape(3, 3)[:, 2]
        on_platform = 0.3 < data.xpos[wp_id][2] < 0.6
        face_up = wp_z_axis[2] < -0.999  # strict: < 2.5° tilt, avoid platform protrusions

        if on_platform and face_up:
            return data.qpos.copy()

    raise RuntimeError("Failed to place workpiece after 10 attempts")


def render_cameras(renderer, data, device):
    """Render overhead + ee cameras, return as normalized (1, 3, H, W) tensors."""
    images = {}
    for cam_name, key in [("overhead_cam", "observation.images.overhead"),
                           ("ee_cam", "observation.images.ee")]:
        renderer.update_scene(data, camera=cam_name)
        img = renderer.render()  # (H, W, 3) uint8
        t = torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0  # (3, H, W)
        images[key] = t.unsqueeze(0).to(device)  # (1, 3, H, W)
    return images


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    print(f"Device: {device}")

    # ── 1. Load policy ─────────────────────────────────────────────
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_symlink():
        checkpoint = checkpoint.resolve()
    print(f"\nLoading {args.policy} policy from {checkpoint}...")

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from policies import get_policy_class

    ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.data_dir)

    policy_cls = get_policy_class(args.policy)
    policy = policy_cls.load(checkpoint, ds_meta.stats, device)
    policy.to(device)
    policy.eval()
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # ── 2. Create MuJoCo scene ────────────────────────────────────────
    print("\nCreating MuJoCo scene...")
    spec = create_base_scene(
        with_actuators=True, with_platform=True, with_workpiece=True,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1])
    site_id = model.site("tool_tip").id

    sim_dt = model.opt.timestep
    decimation = int(round(1.0 / (args.control_freq * sim_dt)))
    print(f"  Sim dt: {sim_dt}s, Control freq: {args.control_freq}Hz, Decimation: {decimation}")

    # IK solver for EE delta mode
    ik_solver = None
    if args.action_type == "action.ee_delta":
        ik_solver = DampedLeastSquaresIK(model, data)
        print(f"  Action type: EE delta (using IK solver)")

    # Optional viewer
    viewer = None
    if args.render:
        viewer = mujoco.viewer.launch_passive(model, data)

    # ── 3. Video setup ────────────────────────────────────────────────
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    # Third-person view renderer for video (separate from policy camera renderer)
    vid_renderer = mujoco.Renderer(model, height=VIDEO_SIZE[0], width=VIDEO_SIZE[1])

    # ── 4. Evaluation loop ────────────────────────────────────────────
    all_metrics = []

    for ep in range(args.num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{args.num_episodes}")
        print(f"{'='*60}")

        # Randomize workpiece and settle
        settled_qpos = randomize_workpiece(model, data, rng)

        # Generate ground-truth S-curve seam
        seam = create_scurve_seam_on_workpiece(model, data, rng)
        gt_points = seam.sample_points(200)  # dense ground-truth path
        print(f"  Seam start: [{gt_points[0][0]:.3f}, {gt_points[0][1]:.3f}, {gt_points[0][2]:.3f}]")
        print(f"  Seam end:   [{gt_points[-1][0]:.3f}, {gt_points[-1][1]:.3f}, {gt_points[-1][2]:.3f}]")

        # Restore settled state, reset robot joints to zero (match training approach start)
        data.qpos[:] = settled_qpos
        data.qpos[:NUM_JOINTS] = 0.0
        data.ctrl[:NUM_JOINTS] = 0.0
        mujoco.mj_forward(model, data)

        # Workpiece 固定: 记录稳定位姿，每步仿真后恢复
        wp_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "workpiece_joint")
        if wp_jnt_id >= 0:
            wp_qadr = model.jnt_qposadr[wp_jnt_id]
            wp_vadr = model.jnt_dofadr[wp_jnt_id]
            wp_qpos_fixed = settled_qpos[wp_qadr:wp_qadr + 7].copy()
            wp_qvel_fixed = np.zeros(6)
        else:
            wp_qadr = None

        # Reset policy state (clear action queue/ensemble + memory buffer)
        policy.reset()

        # Video writer: side-by-side [overhead | ee | third-person], 3x width
        video_path = video_dir / f"episode_{ep:02d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_w = VIDEO_SIZE[1] * 3
        video_h = VIDEO_SIZE[0]
        video_writer = cv2.VideoWriter(
            str(video_path), fourcc, args.control_freq, (video_w, video_h),
        )

        # Run closed-loop
        ee_trajectory = []
        joint_trajectory = []

        # History buffer for obs_horizon > 1
        obs_horizon = getattr(policy, '_obs_horizon', 1)
        history_buf = None
        if obs_horizon > 1:
            history_buf = {
                "observation.state": deque(maxlen=obs_horizon),
                "images": {k: deque(maxlen=obs_horizon) for k in
                           getattr(policy, '_image_keys', [])},
            }

        for step in range(args.max_steps):
            # Get current state
            qpos = data.qpos[:NUM_JOINTS].copy()
            state = torch.from_numpy(qpos).float().unsqueeze(0).to(device)  # (1, 6)

            # Render cameras
            images = render_cameras(renderer, data, device)

            # Build observation batch
            if history_buf is not None:
                # Push current frame into history
                history_buf["observation.state"].append(state)
                for k in history_buf["images"]:
                    history_buf["images"][k].append(images[k])

                # Fill buffer if not full (episode start)
                while len(history_buf["observation.state"]) < obs_horizon:
                    history_buf["observation.state"].appendleft(state)
                    for k in history_buf["images"]:
                        history_buf["images"][k].appendleft(images[k])

                # Stack history: (1, S, ...)
                batch = {
                    "observation.state": torch.stack(
                        list(history_buf["observation.state"]), dim=1),  # (1, S, 6)
                }
                for k in history_buf["images"]:
                    batch[k] = torch.stack(
                        list(history_buf["images"][k]), dim=1)  # (1, S, C, H, W)
            else:
                batch = {"observation.state": state}
                batch.update(images)

            # Get action from policy
            action = policy.select_action(batch)  # (1, 6)
            action_np = action.squeeze(0).cpu().numpy()

            # Apply action
            if args.action_type == "action.ee_delta":
                # EE delta mode: predict delta pose, compute target EE, IK to joint angles
                delta_pos = action_np[:3]
                current_ee_pos = data.site_xpos[site_id].copy()
                target_ee_pos = current_ee_pos + delta_pos
                # Set current qpos as IK initial guess
                ik_solver.data.qpos[:NUM_JOINTS] = data.qpos[:NUM_JOINTS].copy()
                q_target = ik_solver.solve(target_ee_pos)
                data.ctrl[:NUM_JOINTS] = q_target
            else:
                # Joint absolute mode: directly set joint position target
                data.ctrl[:NUM_JOINTS] = action_np
            for _ in range(decimation):
                mujoco.mj_step(model, data)
                # 固定 workpiece: 恢复稳定位姿，防止被机械臂推动
                if wp_qadr is not None:
                    data.qpos[wp_qadr:wp_qadr + 7] = wp_qpos_fixed
                    data.qvel[wp_vadr:wp_vadr + 6] = wp_qvel_fixed

            # Record EE position
            ee_pos = data.site_xpos[site_id].copy()
            ee_trajectory.append(ee_pos)
            joint_trajectory.append(data.qpos[:NUM_JOINTS].copy())

            # Write video frame: [overhead | ee | third-person]
            renderer.update_scene(data, camera="overhead_cam")
            overhead_frame = renderer.render().copy()  # (H, W, 3) RGB
            renderer.update_scene(data, camera="ee_cam")
            ee_frame = renderer.render().copy()
            vid_renderer.update_scene(data, camera="front_cam")
            third_frame = vid_renderer.render().copy()

            combined = np.concatenate([overhead_frame, ee_frame, third_frame], axis=1)
            # Add step/info text overlay
            cv2.putText(combined, f"Step {step}/{args.max_steps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, "overhead", (10, video_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(combined, "ee_cam", (VIDEO_SIZE[1] + 10, video_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(combined, "front", (VIDEO_SIZE[1] * 2 + 10, video_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            # OpenCV expects BGR
            video_writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            if viewer is not None:
                viewer.sync()

        video_writer.release()
        print(f"  Video saved: {video_path}")

        ee_trajectory = np.array(ee_trajectory)  # (max_steps, 3)

        # ── 4. Compute metrics (seam tracking only) ──────────────────
        # Determine approach end: first frame where EE is within 50mm of seam start
        seam_start = gt_points[0]
        approach_threshold = 0.05  # 50mm
        tracking_start = 0
        for i, ee_pt in enumerate(ee_trajectory):
            if np.linalg.norm(ee_pt - seam_start) < approach_threshold:
                tracking_start = i
                break
        else:
            tracking_start = len(ee_trajectory)  # never reached seam

        # Seam tracking phase only
        tracking_traj = ee_trajectory[tracking_start:]
        tracking_steps = len(tracking_traj)

        # Distance to seam (only tracking phase)
        if tracking_steps > 0:
            min_distances = []
            for ee_pt in tracking_traj:
                dists = np.linalg.norm(gt_points - ee_pt, axis=1)
                min_distances.append(dists.min())
            min_distances = np.array(min_distances)
        else:
            # Approach failed: use full trajectory's closest approach as reference
            all_dists_to_start = np.linalg.norm(ee_trajectory - seam_start, axis=1)
            min_distances = np.array([all_dists_to_start.min()])

        # Multi-level coverage: fraction of GT points covered by tracking EE
        coverage_thresholds = [0.0005, 0.001, 0.002, 0.005]  # 0.5mm, 1mm, 2mm, 5mm
        coverages = {}
        for thresh in coverage_thresholds:
            covered = np.zeros(len(gt_points), dtype=bool)
            for ee_pt in tracking_traj:
                dists = np.linalg.norm(gt_points - ee_pt, axis=1)
                covered |= (dists < thresh)
            coverages[thresh] = covered.sum() / len(covered) * 100

        # Reward: multi-level gated system
        # Gate: approach must reach seam (within 50mm), otherwise reward = 0
        mean_dist = min_distances.mean()
        approached = tracking_start < len(ee_trajectory)

        if not approached:
            reward = 0.0
            reward_level = "FAIL (approach)"
        else:
            # Seam tracking quality levels based on coverage
            cov5 = coverages.get(0.005, 0.0)     # 5mm
            cov2 = coverages.get(0.002, 0.0)     # 2mm
            cov1 = coverages.get(0.001, 0.0)     # 1mm
            cov05 = coverages.get(0.0005, 0.0)   # 0.5mm

            if cov05 >= 80:
                reward = 1.0                              # excellent
                reward_level = "S  (cov0.5mm≥80%)"
            elif cov1 >= 80:
                reward = 0.8 + 0.2 * (cov05 / 100.0)    # 0.80-1.00
                reward_level = "A  (cov1mm≥80%)"
            elif cov2 >= 80:
                reward = 0.6 + 0.2 * (cov1 / 100.0)     # 0.60-0.80
                reward_level = "B  (cov2mm≥80%)"
            elif cov2 >= 50:
                reward = 0.4 + 0.2 * (cov2 / 100.0)     # 0.40-0.60
                reward_level = "C  (cov2mm≥50%)"
            elif cov5 > 0:
                reward = 0.1 + 0.3 * (cov5 / 100.0)     # 0.10-0.40
                reward_level = "D  (cov5mm>0%)"
            else:
                reward = 0.05                              # reached but no coverage
                reward_level = "E  (no coverage)"

        metrics = {
            "approach_frame": tracking_start,
            "tracking_steps": tracking_steps,
            "mean_dist": mean_dist,
            "max_dist": min_distances.max(),
            "median_dist": np.median(min_distances),
            "coverages": coverages,
            "approached": approached,
            "reward": reward,
            "reward_level": reward_level,
        }
        all_metrics.append(metrics)

        print(f"\n  Approach → seam at frame {tracking_start}/{len(ee_trajectory)}")
        print(f"\n  Seam Tracking Results ({tracking_steps} steps):")
        print(f"    Mean distance:     {mean_dist*1000:.1f} mm")
        print(f"    Max distance:      {metrics['max_dist']*1000:.1f} mm")
        print(f"    Median distance:   {metrics['median_dist']*1000:.1f} mm")
        print(f"    Coverage (<0.5mm): {coverages[0.0005]:.1f}%")
        print(f"    Coverage (<1mm):   {coverages[0.001]:.1f}%")
        print(f"    Coverage (<2mm):   {coverages[0.002]:.1f}%")
        print(f"    Coverage (<5mm):   {coverages[0.005]:.1f}%")
        print(f"    Reward: {reward:.3f}  [{reward_level}]")

        # EE trajectory extent (tracking phase only)
        if tracking_steps > 0:
            print(f"    EE X range: [{tracking_traj[:,0].min():.4f}, {tracking_traj[:,0].max():.4f}]")
            print(f"    EE Y range: [{tracking_traj[:,1].min():.4f}, {tracking_traj[:,1].max():.4f}]")
            print(f"    EE Z range: [{tracking_traj[:,2].min():.4f}, {tracking_traj[:,2].max():.4f}]")

    # ── 5. Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Evaluation Summary ({args.num_episodes} episodes)")
    print(f"{'='*60}")

    # Success = reward reaches max (S level)
    success_metrics = [m for m in all_metrics if m["reward"] >= 1.0]
    n_success = len(success_metrics)

    rewards = [m["reward"] for m in all_metrics]
    print(f"  Success rate (S):     {n_success}/{args.num_episodes} ({n_success/args.num_episodes*100:.0f}%)")
    print(f"  Avg reward:           {np.mean(rewards):.3f}")
    print(f"  Best reward:          {np.max(rewards):.3f}")
    print(f"  Worst reward:         {np.min(rewards):.3f}")

    # Reward level distribution
    level_counts = {}
    for m in all_metrics:
        lvl = m["reward_level"].split("(")[0].strip()
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    print(f"\n  Reward distribution:")
    for lvl in ["S", "A", "B", "C", "D", "E", "FAIL"]:
        cnt = level_counts.get(lvl, 0)
        if cnt > 0:
            bar = "█" * cnt
            print(f"    {lvl:4s}: {cnt:2d}  {bar}")

    # Per-level stats
    approached_metrics = [m for m in all_metrics if m["approached"]]
    n_approached = len(approached_metrics)
    print(f"\n  Approach rate:         {n_approached}/{args.num_episodes} ({n_approached/args.num_episodes*100:.0f}%)")

    if n_approached > 0:
        a_dists = [m["mean_dist"] for m in approached_metrics]
        a_cov05 = [m["coverages"][0.0005] for m in approached_metrics]
        a_cov1 = [m["coverages"][0.001] for m in approached_metrics]
        a_cov2 = [m["coverages"][0.002] for m in approached_metrics]
        a_cov5 = [m["coverages"][0.005] for m in approached_metrics]

        print(f"  Avg mean distance:    {np.mean(a_dists)*1000:.1f} mm")
        print(f"  Avg coverage (<0.5mm):{np.mean(a_cov05):.1f}%")
        print(f"  Avg coverage (<1mm):  {np.mean(a_cov1):.1f}%")
        print(f"  Avg coverage (<2mm):  {np.mean(a_cov2):.1f}%")
        print(f"  Avg coverage (<5mm):  {np.mean(a_cov5):.1f}%")

    renderer.close()
    vid_renderer.close()
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(f"{args.policy.upper()} Policy Evaluation — MR2010 Welding")
    print("=" * 60)
    evaluate(args)
