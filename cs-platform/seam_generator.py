"""
焊缝路径定义和轨迹自动生成
支持直线焊缝、圆弧焊缝、锯齿焊缝、S型弧线焊缝
"""

import numpy as np
import mujoco

from mr2010_robot import create_base_scene, NUM_JOINTS
from ik_solver import DampedLeastSquaresIK
from il_env import Trajectory


def _rotmat_to_axisangle(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 (3,3) -> 轴角向量 (3,)，向量方向为旋转轴，模长为旋转角度(rad)"""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2 * np.sin(angle))
    return axis * angle


class SeamPath:
    """焊缝路径基类"""

    def sample_points(self, num_points: int) -> np.ndarray:
        """返回路径上的3D坐标点 (N, 3)"""
        raise NotImplementedError

    def sample_orientations(self, num_points: int) -> np.ndarray:
        """返回每个点的朝向 (旋转矩阵) (N, 3, 3)"""
        # Default: tool points downward (-z direction)
        mats = np.zeros((num_points, 3, 3))
        for i in range(num_points):
            mats[i] = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ])
        return mats


class StraightSeam(SeamPath):
    """直线焊缝"""

    def __init__(self, start: np.ndarray, end: np.ndarray):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)

    def sample_points(self, num_points: int) -> np.ndarray:
        t = np.linspace(0, 1, num_points).reshape(-1, 1)
        return self.start + t * (self.end - self.start)


class ArcSeam(SeamPath):
    """圆弧焊缝"""

    def __init__(self, center: np.ndarray, radius: float,
                 start_angle: float, end_angle: float, height: float):
        self.center = np.asarray(center, dtype=float)
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.height = height

    def sample_points(self, num_points: int) -> np.ndarray:
        angles = np.linspace(self.start_angle, self.end_angle, num_points)
        points = np.zeros((num_points, 3))
        points[:, 0] = self.center[0] + self.radius * np.cos(angles)
        points[:, 1] = self.center[1] + self.radius * np.sin(angles)
        points[:, 2] = self.height
        return points


class ZigzagSeam(SeamPath):
    """锯齿焊缝 (在直线基础上横向摆动)"""

    def __init__(self, start: np.ndarray, end: np.ndarray,
                 amplitude: float = 0.02, frequency: int = 10):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.amplitude = amplitude
        self.frequency = frequency

    def sample_points(self, num_points: int) -> np.ndarray:
        t = np.linspace(0, 1, num_points)
        direction = self.end - self.start
        length = np.linalg.norm(direction)
        forward = direction / length
        up = np.array([0, 0, 1.0])
        lateral = np.cross(forward, up)
        lateral = lateral / (np.linalg.norm(lateral) + 1e-8)

        points = np.zeros((num_points, 3))
        for i, ti in enumerate(t):
            base = self.start + ti * direction
            offset = self.amplitude * np.sin(2 * np.pi * self.frequency * ti)
            points[i] = base + offset * lateral
        return points


class SCurveSeam(SeamPath):
    """S型弧线焊缝 — 精确匹配 workpiece 表面的 S 型凸起中心线

    workpiece 上的 S 型凸起由两个半圆盘组成（STL 局部坐标, mm）:
      上半盘: 圆心 (386, 200), R=109
      下半盘: 圆心 (174, 200), R=109

    S 型焊缝中心线沿两个半盘之间的 **内侧弧线** 走:
      上内弧: 圆心 (386, 200), R=109, θ: -π/2 → -π
              路径: (386, 91) → (277, 200)
      下内弧: 圆心 (174, 200), R=109, θ: 0 → π/2
              路径: (283, 200) → (174, 309)
      中间有 6mm 直线过渡段连接两弧。

    本类将 STL 局部坐标的 S 型中心线通过 workpiece body 的位姿变换
    映射到世界坐标系。

    Args:
        body_pos: workpiece body 在世界坐标系中的位置 (3,)
        body_rotmat: workpiece body 的旋转矩阵 (3, 3)
        t_start: 起始参数 [0, 1], 0 = S 曲线顶端, 1 = S 曲线底端
        t_end: 结束参数 [0, 1]
        standoff_mm: 焊接高度 (mm)，tool tip 悬浮在 workpiece 表面上方的距离
    """

    # S 型凸起几何参数 (STL mm)
    ARC1_CENTER = np.array([386.0, 200.0])  # 上半盘圆心
    ARC2_CENTER = np.array([174.0, 200.0])  # 下半盘圆心
    ARC_RADIUS = 109.0                       # 半径

    def __init__(self, body_pos: np.ndarray, body_rotmat: np.ndarray,
                 t_start: float = 0.0, t_end: float = 1.0,
                 standoff_mm: float = 7.0):
        self.body_pos = np.asarray(body_pos, dtype=float)
        self.body_rotmat = np.asarray(body_rotmat, dtype=float)
        self.t_start = t_start
        self.t_end = t_end
        self.standoff_mm = standoff_mm

    def _stl_centerline(self, num_points: int) -> np.ndarray:
        """生成 STL 局部坐标(mm)下的完整 S 型中心线

        两段半圆弧通过 sigmoid 函数平滑混合:
          上半圆弧: center(386,200), R=109, θ: 0 → -π
          下半圆弧: center(174,200), R=109, θ: 0 → π
        在 t=0.5 附近用 sigmoid 平滑过渡，避免直线段跳跃。
        """
        t = np.linspace(0, 1, num_points)

        # 上半圆弧: θ 从 0 到 -π, 在 t∈[0, 0.5] 区间扫完
        theta1 = -np.pi * np.minimum(2 * t, 1.0)
        upper_x = self.ARC1_CENTER[0] + self.ARC_RADIUS * np.cos(theta1)
        upper_y = self.ARC1_CENTER[1] + self.ARC_RADIUS * np.sin(theta1)

        # 下半圆弧: θ 从 0 到 π, 在 t∈[0.5, 1] 区间扫完
        theta2 = np.pi * np.maximum(2 * t - 1, 0.0)
        lower_x = self.ARC2_CENTER[0] + self.ARC_RADIUS * np.cos(theta2)
        lower_y = self.ARC2_CENTER[1] + self.ARC_RADIUS * np.sin(theta2)

        # Sigmoid 混合: 在 t=0.5 附近平滑从上弧过渡到下弧
        k = 20.0  # 过渡陡度, Δt ≈ 0.2 (约 20% 的路径长度用于过渡)
        sigma = 1.0 / (1.0 + np.exp(-k * (t - 0.5)))

        pts = np.zeros((num_points, 3))
        pts[:, 0] = (1.0 - sigma) * upper_x + sigma * lower_x
        pts[:, 1] = (1.0 - sigma) * upper_y + sigma * lower_y
        pts[:, 2] = -self.standoff_mm  # STL Z轴朝下(body翻转180°), 负值=表面上方
        return pts

    def _stl_to_world(self, stl_pts_mm: np.ndarray) -> np.ndarray:
        """将 STL mm 坐标转换为世界坐标 (m)"""
        return self.body_pos + (self.body_rotmat @ (stl_pts_mm * 0.001).T).T

    def sample_points(self, num_points: int) -> np.ndarray:
        # 生成密集中心线，然后截取 [t_start, t_end] 范围
        dense = self._stl_centerline(max(num_points * 4, 200))
        total = len(dense)

        # 将 t_start, t_end 映射到索引
        i_start = int(round(self.t_start * (total - 1)))
        i_end = int(round(self.t_end * (total - 1)))
        i_start, i_end = min(i_start, i_end), max(i_start, i_end)
        i_end = max(i_end, i_start + 1)

        segment = dense[i_start:i_end + 1]

        # 转换到世界坐标
        world_pts = self._stl_to_world(segment)

        # 重采样到 num_points
        if len(world_pts) == num_points:
            return world_pts
        t_orig = np.linspace(0, 1, len(world_pts))
        t_new = np.linspace(0, 1, num_points)
        result = np.zeros((num_points, 3))
        for d in range(3):
            result[:, d] = np.interp(t_new, t_orig, world_pts[:, d])
        return result


def _solve_ik_offline(waypoints, model, orientations=None):
    """
    离线求解所有路径点的IK，使用独立的MjData副本。

    Args:
        waypoints: (N, 3) 目标位置
        model: MuJoCo 模型
        orientations: (N, 3, 3) 目标旋转矩阵，None 则只做位置 IK
    """
    ik_data = mujoco.MjData(model)
    ik_solver = DampedLeastSquaresIK(model, ik_data, damping=0.01)

    joint_targets = []
    errors = []
    for i, wp in enumerate(waypoints):
        target_mat = orientations[i] if orientations is not None else None
        if i == 0:
            ik_data.qpos[:NUM_JOINTS] = 0.0
            mujoco.mj_forward(model, ik_data)
            q = ik_solver.solve(wp, target_mat=target_mat,
                                max_iter=500, tol=1e-5)
        else:
            q = ik_solver.solve(wp, target_mat=target_mat,
                                max_iter=200, tol=1e-4)
        joint_targets.append(q.copy())
        err = np.linalg.norm(ik_solver.get_ee_pos() - wp)
        errors.append(err)

    return joint_targets, errors


def generate_trajectory(
    seam: SeamPath,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik_solver: DampedLeastSquaresIK,
    num_points: int = 100,
    sim_steps_per_point: int = 50,
    render_images: bool = False,
    image_size: tuple = (240, 320),
    restore_qpos: np.ndarray = None,
    record_interval: int = 1,
    approach_steps: int = 2000,
) -> Trajectory:
    """
    从焊缝路径生成完整轨迹（含接近阶段）

    完整轨迹包含两个阶段:
      1. Approach: 从初始位置平滑移动到焊缝起点
      2. Tracking: 沿焊缝路径跟踪

    action[t] = state[t+1]（下一帧的目标关节位置），使模型学到
    "给定当前状态和图像，下一步应该移到哪里"。

    Args:
        seam: 焊缝路径
        model: MuJoCo模型
        data: MuJoCo数据
        ik_solver: IK求解器 (仅用于获取site_id)
        num_points: 焊缝路径采样点数
        sim_steps_per_point: 每个路径点仿真步数 (用于获取速度)
        render_images: 是否渲染摄像头图像
        image_size: 图像尺寸 (height, width)
        restore_qpos: 可选，初始 qpos 快照（用于恢复 workpiece 等物体位置，
                      避免 mj_resetData 重置 freejoint 状态）
        record_interval: 每隔多少仿真步保存一帧 (默认1=每步保存)。
                         例如仿真500Hz, record_interval=10 → 保存50fps
        approach_steps: 从初始位置到焊缝起点的接近阶段仿真步数。

    Returns:
        (Trajectory, ik_errors) 元组
    """
    traj = Trajectory()
    dt = model.opt.timestep
    site_id = ik_solver.site_id

    # Workpiece 固定: 记录稳定后的 freejoint 状态，每步仿真后恢复
    # 真实环境中 workpiece 有夹具固定，不会被机械臂推动
    wp_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "workpiece_joint")
    if wp_jnt_id >= 0:
        wp_qadr = model.jnt_qposadr[wp_jnt_id]
        wp_vadr = model.jnt_dofadr[wp_jnt_id]
        wp_qpos_settled = restore_qpos[wp_qadr:wp_qadr + 7].copy() if restore_qpos is not None else data.qpos[wp_qadr:wp_qadr + 7].copy()
        wp_qvel_settled = np.zeros(6)
    else:
        wp_qadr = None

    # Setup renderer for camera images
    renderer = None
    has_overhead = False
    has_ee_cam = False
    if render_images:
        renderer = mujoco.Renderer(model, height=image_size[0], width=image_size[1])
        # Check which cameras exist
        for i in range(model.ncam):
            cam_name = model.camera(i).name
            if cam_name == "overhead_cam":
                has_overhead = True
            elif cam_name == "ee_cam":
                has_ee_cam = True

    # Sample seam path
    waypoints = seam.sample_points(num_points)
    orientations = seam.sample_orientations(num_points)

    # Step 1: IK for all waypoints (with orientation constraint)
    joint_targets, ik_errors = _solve_ik_offline(waypoints, model, orientations)
    joint_targets = np.array(joint_targets)  # (num_points, NUM_JOINTS)

    # Step 2: Build full trajectory = approach + seam tracking
    # Get initial robot joint position
    if restore_qpos is not None:
        q_init = restore_qpos[:NUM_JOINTS].copy()
    else:
        q_init = np.zeros(NUM_JOINTS)

    # Approach phase: interpolate from initial position to seam start
    q_approach = np.zeros((approach_steps, NUM_JOINTS))
    for j in range(NUM_JOINTS):
        q_approach[:, j] = np.linspace(q_init[j], joint_targets[0, j], approach_steps)

    # Seam tracking phase: dense interpolation between waypoints
    seam_steps = num_points * sim_steps_per_point
    t_waypoints = np.linspace(0, 1, num_points)
    t_dense = np.linspace(0, 1, seam_steps)
    q_seam = np.zeros((seam_steps, NUM_JOINTS))
    for j in range(NUM_JOINTS):
        q_seam[:, j] = np.interp(t_dense, t_waypoints, joint_targets[:, j])

    # Concatenate: approach + seam tracking
    q_dense = np.concatenate([q_approach, q_seam], axis=0)
    total_steps = len(q_dense)

    # Step 3: Phase 1 — Mocap run: use IK trajectory to drive PD controller,
    # record actual qpos at each recorded frame as the "action" sequence
    if restore_qpos is not None:
        data.qpos[:] = restore_qpos
    else:
        mujoco.mj_resetData(model, data)
    data.qpos[:NUM_JOINTS] = q_dense[0]
    data.ctrl[:NUM_JOINTS] = q_dense[0]
    mujoco.mj_forward(model, data)

    record_dt = dt * record_interval  # time step between recorded frames

    # Collect mocap qpos (action targets for replay)
    mocap_qpos_list = []
    for i in range(total_steps):
        data.ctrl[:NUM_JOINTS] = q_dense[i]
        mujoco.mj_step(model, data)
        # 固定 workpiece: 恢复稳定位姿，防止被机械臂推动
        if wp_qadr is not None:
            data.qpos[wp_qadr:wp_qadr + 7] = wp_qpos_settled
            data.qvel[wp_vadr:wp_vadr + 6] = wp_qvel_settled
        if i % record_interval == 0:
            mocap_qpos_list.append(data.qpos[:NUM_JOINTS].copy())
    mocap_qpos = np.array(mocap_qpos_list)  # (N_frames, NUM_JOINTS)

    # Step 4: Phase 2 — Replay: use mocap qpos as control targets,
    # record actual qpos as "state" and mocap qpos as "action"
    if restore_qpos is not None:
        data.qpos[:] = restore_qpos
    else:
        mujoco.mj_resetData(model, data)
    data.qpos[:NUM_JOINTS] = mocap_qpos[0]
    data.ctrl[:NUM_JOINTS] = mocap_qpos[0]
    mujoco.mj_forward(model, data)

    frame_idx = 0
    prev_ee_pos = data.site_xpos[site_id].copy()
    prev_ee_mat = data.site_xmat[site_id].reshape(3, 3).copy()
    for i in range(total_steps):
        # Use mocap qpos as control target
        frame_for_ctrl = min(i // record_interval, len(mocap_qpos) - 1)
        data.ctrl[:NUM_JOINTS] = mocap_qpos[frame_for_ctrl]

        mujoco.mj_step(model, data)
        # 固定 workpiece: 恢复稳定位姿，防止被机械臂推动
        if wp_qadr is not None:
            data.qpos[wp_qadr:wp_qadr + 7] = wp_qpos_settled
            data.qvel[wp_vadr:wp_vadr + 6] = wp_qvel_settled

        if i % record_interval != 0:
            continue

        # Record actual joint position (state from replay dynamics)
        actual_qpos = data.qpos[:NUM_JOINTS].copy()

        # Compute finite-difference velocity
        if frame_idx > 0:
            prev_qpos = mocap_qpos[frame_idx - 1]
            qvel = (actual_qpos - prev_qpos) / record_dt
        else:
            qvel = np.zeros(NUM_JOINTS)

        # End-effector pose
        ee_pos = data.site_xpos[site_id].copy()
        ee_mat = data.site_xmat[site_id].reshape(3, 3).copy()

        # Joint torques
        torques = data.qfrc_actuator[:NUM_JOINTS].copy()

        # Camera images
        overhead_img = None
        ee_img = None
        if renderer is not None:
            if has_overhead:
                renderer.update_scene(data, camera="overhead_cam")
                overhead_img = renderer.render()
            if has_ee_cam:
                renderer.update_scene(data, camera="ee_cam")
                ee_img = renderer.render()

        # action = 关节角度绝对值 (下一帧 mocap qpos)
        next_frame = min(frame_idx + 1, len(mocap_qpos) - 1)

        # 末端位姿增量: delta_ee = [dx, dy, dz, drx, dry, drz]
        if frame_idx > 0:
            delta_pos = ee_pos - prev_ee_pos  # 位置增量 (3,)
            # 旋转增量: R_delta = R_curr * R_prev^T, 转为轴角
            R_delta = ee_mat @ prev_ee_mat.T
            delta_rot = _rotmat_to_axisangle(R_delta)  # (3,)
            action_ee_delta = np.concatenate([delta_pos, delta_rot])
        else:
            action_ee_delta = np.zeros(6)
        prev_ee_pos = ee_pos.copy()
        prev_ee_mat = ee_mat.copy()

        traj.add_step(
            t=i * dt,
            qpos=actual_qpos,
            qvel=qvel.copy(),
            action=mocap_qpos[next_frame].copy(),
            action_ee_delta=action_ee_delta,
            ee_pos=ee_pos,
            ee_mat=ee_mat,
            torques=torques,
            overhead_img=overhead_img,
            ee_img=ee_img,
        )
        frame_idx += 1

    if renderer is not None:
        renderer.close()

    return traj, ik_errors


def get_workpiece_surface_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """获取 workpiece 稳定后的表面 Z 高度 (face-up surface)"""
    wp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workpiece")
    return data.xpos[wp_id][2]


def get_workpiece_surface_bounds(
    model: mujoco.MjModel, data: mujoco.MjData,
) -> dict:
    """获取 workpiece 稳定后的表面边界 (world frame)

    workpiece STL 尺寸: 530x400x50 mm，face-up 表面是 STL Z=0 面。
    body 使用 quat=(0,1,0,0) 翻转 180° 绕 X，使 Z=0 面朝上。

    Returns:
        dict with keys: x_min, x_max, y_min, y_max, z (表面高度)
    """
    wp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workpiece")
    body_pos = data.xpos[wp_id].copy()
    R = data.xmat[wp_id].reshape(3, 3)
    scale = 0.001

    # STL 角点 (Z=0 面 = face-up surface)
    stl_corners = np.array([
        [0, 0, 0], [530, 0, 0], [0, 400, 0], [530, 400, 0],
    ])
    world_corners = body_pos + (R @ (stl_corners * scale).T).T

    return {
        "x_min": world_corners[:, 0].min(),
        "x_max": world_corners[:, 0].max(),
        "y_min": world_corners[:, 1].min(),
        "y_max": world_corners[:, 1].max(),
        "z": world_corners[:, 2].mean(),
    }


def create_scurve_seam_on_workpiece(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator = None,
) -> SCurveSeam:
    """在 workpiece 表面创建 S 型弧线焊缝（精确匹配 STL 凸起中心线）

    根据 workpiece body 稳定后的实际位姿，将 STL 坐标系下的 S 型
    凸起中心线映射到世界坐标系。可通过 t_start/t_end 随机截取一段。

    Args:
        model: 已编译的 MuJoCo 模型 (含 workpiece)
        data: 仿真数据 (workpiece 已稳定)
        rng: 随机数生成器
    """
    if rng is None:
        rng = np.random.default_rng()

    wp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workpiece")
    body_pos = data.xpos[wp_id].copy()
    body_rotmat = data.xmat[wp_id].reshape(3, 3).copy()

    # 默认覆盖完整 S 型曲线
    t_start = 0.0
    t_end = 1.0

    return SCurveSeam(body_pos, body_rotmat, t_start, t_end)


def create_random_straight_seam(rng: np.random.Generator = None) -> StraightSeam:
    """创建随机直线焊缝"""
    if rng is None:
        rng = np.random.default_rng()
    x_start = rng.uniform(0.5, 0.7)
    y_start = rng.uniform(-0.15, 0.15)
    z = rng.uniform(0.43, 0.50)
    length = rng.uniform(0.08, 0.20)
    angle = rng.uniform(-0.5, 0.5)

    start = np.array([x_start, y_start, z])
    end = np.array([
        x_start + length * np.cos(angle),
        y_start + length * np.sin(angle),
        z,
    ])
    return StraightSeam(start, end)


def create_random_arc_seam(rng: np.random.Generator = None) -> ArcSeam:
    """创建随机圆弧焊缝"""
    if rng is None:
        rng = np.random.default_rng()
    center = np.array([rng.uniform(0.55, 0.7), rng.uniform(-0.1, 0.1)])
    radius = rng.uniform(0.05, 0.12)
    start_angle = rng.uniform(0, np.pi)
    arc_span = rng.uniform(np.pi / 3, np.pi)
    z = rng.uniform(0.43, 0.50)
    return ArcSeam(center, radius, start_angle, start_angle + arc_span, z)


def create_random_zigzag_seam(rng: np.random.Generator = None) -> ZigzagSeam:
    """创建随机锯齿焊缝"""
    if rng is None:
        rng = np.random.default_rng()
    straight = create_random_straight_seam(rng)
    amplitude = rng.uniform(0.005, 0.02)
    frequency = rng.integers(5, 15)
    return ZigzagSeam(straight.start, straight.end, amplitude, int(frequency))


if __name__ == "__main__":
    print("=" * 60)
    print("Seam Generator - S-Curve Trajectory Test")
    print("=" * 60)

    print("\n[1] Creating scene with platform + workpiece...")
    spec = create_base_scene(with_actuators=True, with_platform=True, with_workpiece=True)
    model = spec.compile()
    data = mujoco.MjData(model)
    print(f"    Model: {model.ngeom} geoms, {model.njnt} joints, {model.nu} actuators")

    # Let workpiece settle on platform
    print("\n[2] Settling workpiece on platform...")
    data.ctrl[:model.nu] = data.qpos[:model.nu]
    for _ in range(2000):
        mujoco.mj_step(model, data)

    wp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workpiece")
    print(f"    Workpiece pos: {data.xpos[wp_id]}")

    # Save settled state
    settled_qpos = data.qpos.copy()

    bounds = get_workpiece_surface_bounds(model, data)
    print(f"    Surface bounds: X=[{bounds['x_min']:.3f}, {bounds['x_max']:.3f}], "
          f"Y=[{bounds['y_min']:.3f}, {bounds['y_max']:.3f}], Z={bounds['z']:.3f}")

    ik_solver = DampedLeastSquaresIK(model, data)
    rng = np.random.default_rng(42)

    print(f"\n[3] Generating S-curve trajectory on workpiece surface...")
    seam = create_scurve_seam_on_workpiece(model, data, rng)

    pts = seam.sample_points(50)
    print(f"    Seam body_pos: {seam.body_pos}")
    print(f"    Seam t_range: [{seam.t_start:.2f}, {seam.t_end:.2f}]")
    print(f"    Start: {pts[0]}")
    print(f"    Mid:   {pts[len(pts)//2]}")
    print(f"    End:   {pts[-1]}")

    # Restore settled state before trajectory generation
    data.qpos[:] = settled_qpos
    data.ctrl[:NUM_JOINTS] = 0.0
    mujoco.mj_forward(model, data)

    traj, ik_errors = generate_trajectory(
        seam=seam, model=model, data=data,
        ik_solver=ik_solver, num_points=50, sim_steps_per_point=20,
        restore_qpos=settled_qpos,
    )

    duration = traj.timestamps[-1] - traj.timestamps[0] if len(traj) > 1 else 0.0
    print(f"    Trajectory length: {len(traj)} steps")
    print(f"    Duration: {duration:.3f} s")

    # Check IK quality
    ik_errors = np.array(ik_errors)
    print(f"    IK errors: mean={ik_errors.mean():.5f} m, "
          f"max={ik_errors.max():.5f} m, min={ik_errors.min():.5f} m")
    if ik_errors.max() < 0.005:
        print(f"    Quality: EXCELLENT (max error < 5mm)")
    elif ik_errors.max() < 0.02:
        print(f"    Quality: GOOD (max error < 20mm)")
    elif ik_errors.max() < 0.05:
        print(f"    Quality: ACCEPTABLE (max error < 50mm)")
    else:
        print(f"    Quality: POOR (max error >= 50mm)")

    # Verify trajectory is on workpiece surface
    ee_positions = np.array(traj.ee_positions)
    print(f"\n[4] Trajectory verification:")
    print(f"    EE X range: [{ee_positions[:,0].min():.4f}, {ee_positions[:,0].max():.4f}]")
    print(f"    EE Y range: [{ee_positions[:,1].min():.4f}, {ee_positions[:,1].max():.4f}]")
    print(f"    EE Z range: [{ee_positions[:,2].min():.4f}, {ee_positions[:,2].max():.4f}]")

    in_x = (ee_positions[:,0] >= bounds["x_min"]) & (ee_positions[:,0] <= bounds["x_max"])
    in_y = (ee_positions[:,1] >= bounds["y_min"]) & (ee_positions[:,1] <= bounds["y_max"])
    z_close = np.abs(ee_positions[:,2] - bounds["z"]) < 0.05
    on_surface = in_x & in_y & z_close
    pct = 100.0 * on_surface.sum() / len(on_surface)
    print(f"    Points on workpiece surface: {on_surface.sum()}/{len(on_surface)} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("Test complete.")
    print("=" * 60)
