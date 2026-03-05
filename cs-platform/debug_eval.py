"""
诊断评估问题：对比模型推理时每一步的 action 输出、实际 qpos、
以及训练数据中相同初始条件下的 ground-truth 轨迹
"""
import numpy as np
import mujoco
import torch
from pathlib import Path

from mr2010_robot import create_base_scene, NUM_JOINTS, JOINT_NAMES
from ik_solver import DampedLeastSquaresIK
from seam_generator import create_scurve_seam_on_workpiece, generate_trajectory

# ── Config ──────────────────────────────────────────────────
CHECKPOINT = "outputs/exp2_r18_c100_30k/best"  # best model
SEED = 123
MAX_STEPS = 700
CONTROL_FREQ = 50
IMAGE_SIZE = (480, 640)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(SEED)

# ── 1. Load model ───────────────────────────────────────────
print("Loading ACT policy...")
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy

ds_meta = LeRobotDatasetMetadata(repo_id="local/mr2010-welding", root="datasets_lerobot")
policy = ACTPolicy.from_pretrained(str(CHECKPOINT), dataset_stats=ds_meta.stats)
policy.to(device)
policy.eval()
print(f"  Chunk size: {policy.config.chunk_size}")
print(f"  n_action_steps: {policy.config.n_action_steps}")

# ── 2. Create MuJoCo scene ──────────────────────────────────
print("\nCreating MuJoCo scene...")
spec = create_base_scene(with_actuators=True, with_platform=True, with_workpiece=True)
model = spec.compile()
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1])
site_id = model.site("tool_tip").id
ik_solver = DampedLeastSquaresIK(model, data)

sim_dt = model.opt.timestep
decimation = int(round(1.0 / (CONTROL_FREQ * sim_dt)))
print(f"  sim_dt={sim_dt}, decimation={decimation}")

# ── 3. Setup episode (same as evaluate.py) ───────────────────
from mr2010_robot import PLATFORM_POS, WORKPIECE_POS_RANGE, WORKPIECE_ROT_RANGE, euler_to_quat

# Randomize workpiece (same code as evaluate.py)
for attempt in range(10):
    mujoco.mj_resetData(model, data)
    rel_x = rng.uniform(*WORKPIECE_POS_RANGE["x"])
    rel_y = rng.uniform(*WORKPIECE_POS_RANGE["y"])
    wp_pos = (PLATFORM_POS[0] + rel_x, PLATFORM_POS[1] + rel_y, PLATFORM_POS[2] + 0.54)
    yaw = rng.uniform(*WORKPIECE_ROT_RANGE["yaw"])
    q_yaw = euler_to_quat(0.0, 0.0, yaw)
    q_flip = (0.0, 1.0, 0.0, 0.0)
    w1, x1, y1, z1 = q_yaw
    w2, x2, y2, z2 = q_flip
    quat = (w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2)
    data.qpos[NUM_JOINTS] = wp_pos[0]
    data.qpos[NUM_JOINTS+1] = wp_pos[1]
    data.qpos[NUM_JOINTS+2] = wp_pos[2]
    data.qpos[NUM_JOINTS+3:NUM_JOINTS+7] = quat
    data.ctrl[:model.nu] = data.qpos[:model.nu]
    for _ in range(2000):
        mujoco.mj_step(model, data)
    wp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workpiece")
    wp_z_axis = data.xmat[wp_id].reshape(3,3)[:,2]
    if 0.3 < data.xpos[wp_id][2] < 0.6 and wp_z_axis[2] < -0.999:
        break

settled_qpos = data.qpos.copy()
seam = create_scurve_seam_on_workpiece(model, data, rng)
gt_points = seam.sample_points(200)
print(f"\nSeam start: {gt_points[0]}")
print(f"Seam end:   {gt_points[-1]}")

# ── 4. Generate GROUND-TRUTH trajectory (same as training) ──
print("\n--- Generating ground-truth trajectory (as in training data) ---")
data.qpos[:] = settled_qpos
data.ctrl[:NUM_JOINTS] = 0.0
mujoco.mj_forward(model, data)

gt_traj, ik_errors = generate_trajectory(
    seam=seam, model=model, data=data,
    ik_solver=ik_solver, num_points=100,
    sim_steps_per_point=50,
    render_images=False,
    restore_qpos=settled_qpos,
    record_interval=10,
    approach_steps=2000,
)

gt_states = np.array(gt_traj.joint_positions)   # (700, 6) actual qpos in training
gt_actions = np.array(gt_traj.actions)           # (700, 6) action in training
gt_ee = np.array(gt_traj.ee_positions)           # (700, 3)
print(f"  GT trajectory: {len(gt_traj)} frames")
print(f"  GT EE Z range: [{gt_ee[:,2].min():.4f}, {gt_ee[:,2].max():.4f}]")

# ── 5. Run policy closed-loop (same as evaluate.py) ─────────
print("\n--- Running policy in closed-loop ---")
data.qpos[:] = settled_qpos
data.qpos[:NUM_JOINTS] = 0.0
data.ctrl[:NUM_JOINTS] = 0.0
mujoco.mj_forward(model, data)
policy.reset()

eval_actions = []
eval_qpos = []
eval_ee = []

for step in range(MAX_STEPS):
    qpos = data.qpos[:NUM_JOINTS].copy()
    state = torch.from_numpy(qpos).float().unsqueeze(0).to(device)

    images = {}
    for cam_name, key in [("overhead_cam", "observation.images.overhead"),
                           ("ee_cam", "observation.images.ee")]:
        renderer.update_scene(data, camera=cam_name)
        img = renderer.render()
        t = torch.from_numpy(img.copy()).float().permute(2,0,1) / 255.0
        images[key] = t.unsqueeze(0).to(device)

    batch = {"observation.state": state}
    batch.update(images)

    action = policy.select_action(batch)
    action_np = action.squeeze(0).cpu().numpy()

    eval_actions.append(action_np.copy())
    eval_qpos.append(qpos.copy())

    data.ctrl[:NUM_JOINTS] = action_np
    for _ in range(decimation):
        mujoco.mj_step(model, data)

    ee_pos = data.site_xpos[site_id].copy()
    eval_ee.append(ee_pos)

eval_actions = np.array(eval_actions)  # (700, 6)
eval_qpos = np.array(eval_qpos)       # (700, 6)
eval_ee = np.array(eval_ee)           # (700, 3)

renderer.close()

# ── 6. Analysis ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("DIAGNOSTIC ANALYSIS")
print("=" * 70)

# 6a. Action range comparison
print("\n[1] Action range comparison (GT training vs Policy output)")
print(f"{'Joint':<10} {'GT_action min':>14} {'GT_action max':>14} {'Policy min':>14} {'Policy max':>14}")
for j in range(NUM_JOINTS):
    print(f"{JOINT_NAMES[j]:<10} {gt_actions[:,j].min():>14.4f} {gt_actions[:,j].max():>14.4f} "
          f"{eval_actions[:,j].min():>14.4f} {eval_actions[:,j].max():>14.4f}")

# 6b. Action trajectory shape - sample key frames
print("\n[2] Action trajectory at key frames")
key_frames = [0, 50, 100, 150, 199, 300, 450, 600, 699]
print(f"{'Frame':<8} {'Source':<10} {'J1':>8} {'J2':>8} {'J3':>8} {'J4':>8} {'J5':>8} {'J6':>8}")
for f in key_frames:
    if f < len(gt_actions):
        gt_a = gt_actions[f]
        ev_a = eval_actions[f]
        print(f"{f:<8} {'GT':10} {gt_a[0]:>8.4f} {gt_a[1]:>8.4f} {gt_a[2]:>8.4f} {gt_a[3]:>8.4f} {gt_a[4]:>8.4f} {gt_a[5]:>8.4f}")
        print(f"{'':8} {'Policy':10} {ev_a[0]:>8.4f} {ev_a[1]:>8.4f} {ev_a[2]:>8.4f} {ev_a[3]:>8.4f} {ev_a[4]:>8.4f} {ev_a[5]:>8.4f}")
        diff = np.abs(gt_a - ev_a)
        print(f"{'':8} {'|diff|':10} {diff[0]:>8.4f} {diff[1]:>8.4f} {diff[2]:>8.4f} {diff[3]:>8.4f} {diff[4]:>8.4f} {diff[5]:>8.4f}")

# 6c. State tracking: does qpos follow action?
print("\n[3] State tracking: qpos[t+1] vs action[t] (policy closed-loop)")
track_err = np.abs(eval_qpos[1:] - eval_actions[:-1])
print(f"{'Joint':<10} {'Mean err':>10} {'Max err':>10} {'Pct tracked':>12}")
for j in range(NUM_JOINTS):
    mean_e = track_err[:,j].mean()
    max_e = track_err[:,j].max()
    action_range = eval_actions[:,j].max() - eval_actions[:,j].min()
    pct = (1.0 - mean_e / (action_range + 1e-8)) * 100
    print(f"{JOINT_NAMES[j]:<10} {mean_e:>10.5f} {max_e:>10.5f} {pct:>11.1f}%")

# Same for GT
print("\n[4] State tracking: qpos[t+1] vs action[t] (GT training data)")
gt_track_err = np.abs(gt_states[1:] - gt_actions[:-1])
print(f"{'Joint':<10} {'Mean err':>10} {'Max err':>10}")
for j in range(NUM_JOINTS):
    print(f"{JOINT_NAMES[j]:<10} {gt_track_err[:,j].mean():>10.5f} {gt_track_err[:,j].max():>10.5f}")

# 6d. EE trajectory comparison
print("\n[5] EE trajectory comparison")
print(f"{'':15} {'GT':>30} {'Policy':>30}")
print(f"{'':15} {'X range':>10} {'Y range':>10} {'Z range':>10} {'X range':>10} {'Y range':>10} {'Z range':>10}")
gt_ranges = [f"[{gt_ee[:,i].min():.3f},{gt_ee[:,i].max():.3f}]" for i in range(3)]
ev_ranges = [f"[{eval_ee[:,i].min():.3f},{eval_ee[:,i].max():.3f}]" for i in range(3)]
print(f"{'Full traj':15} {gt_ranges[0]:>10} {gt_ranges[1]:>10} {gt_ranges[2]:>10} "
      f"{ev_ranges[0]:>10} {ev_ranges[1]:>10} {ev_ranges[2]:>10}")

# Approach phase (frames 0-199) vs Seam phase (frames 200-699)
for phase, s, e in [("Approach 0-199", 0, 200), ("Seam 200-699", 200, 700)]:
    gt_z = f"[{gt_ee[s:e,2].min():.3f},{gt_ee[s:e,2].max():.3f}]"
    ev_z = f"[{eval_ee[s:e,2].min():.3f},{eval_ee[s:e,2].max():.3f}]"
    print(f"  {phase:13} {'':>10} {'':>10} {gt_z:>10} {'':>10} {'':>10} {ev_z:>10}")

# 6e. Check if action diff pattern matches
print("\n[6] Step-to-step action change magnitude")
gt_delta = np.abs(np.diff(gt_actions, axis=0))
ev_delta = np.abs(np.diff(eval_actions, axis=0))
print(f"{'Phase':<20} {'GT mean Δaction':>16} {'Policy mean Δaction':>20}")
for phase, s, e in [("Approach (0-199)", 0, 200), ("Seam (200-699)", 200, min(699, len(gt_delta)))]:
    if e <= len(gt_delta) and e <= len(ev_delta):
        print(f"  {phase:<18} {gt_delta[s:e].mean():>16.6f} {ev_delta[s:e].mean():>20.6f}")

# 6f. EE distance to seam at key frames
print("\n[7] EE distance to seam at key frames")
print(f"{'Frame':<8} {'GT dist (mm)':>14} {'Policy dist (mm)':>18} {'GT EE Z':>10} {'Policy EE Z':>12}")
for f in [0, 50, 100, 150, 199, 250, 350, 450, 550, 650, 699]:
    if f < len(gt_ee) and f < len(eval_ee):
        gt_dists = np.linalg.norm(gt_points - gt_ee[f], axis=1)
        ev_dists = np.linalg.norm(gt_points - eval_ee[f], axis=1)
        print(f"{f:<8} {gt_dists.min()*1000:>14.1f} {ev_dists.min()*1000:>18.1f} "
              f"{gt_ee[f,2]:>10.4f} {eval_ee[f,2]:>12.4f}")

# 6g. Check normalization / dataset stats
print("\n[8] Dataset stats (normalization parameters)")
for key in ["observation.state", "action"]:
    if key in ds_meta.stats:
        s = ds_meta.stats[key]
        print(f"\n  {key}:")
        for stat_name in ["mean", "std", "min", "max"]:
            if stat_name in s:
                vals = s[stat_name].numpy() if hasattr(s[stat_name], 'numpy') else s[stat_name]
                print(f"    {stat_name}: {vals}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
