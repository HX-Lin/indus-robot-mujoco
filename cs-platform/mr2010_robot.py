"""
MR2010机械臂共用配置模块
包含机器人模型加载、场景创建辅助函数等共用代码
"""

from pathlib import Path
import mujoco

# ========================== 常量定义 ==========================

ASSETS_DIR: Path = Path(__file__).parent / "assets"
MR2010_XML: Path = ASSETS_DIR / "MR2010.xml"
PLATFORM_STL: Path = ASSETS_DIR / "platform.STL"
WORKPIECE_STL: Path = ASSETS_DIR / "workpeice.STL"

assert MR2010_XML.exists(), f"MR2010.xml not found at {MR2010_XML}"

JOINT_NAMES = ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6")
NUM_JOINTS = 6

# 工作台默认位置（机械臂前方）
PLATFORM_POS = (0.8, 0.0, 0.0)  # x前方, y中心, z地面
PLATFORM_SIZE = (0.6, 0.4, 0.4)  # 半长, 半宽, 半高

# 工件随机范围（相对于工作台位置 PLATFORM_POS）
# platform桌面世界范围: x=[0.80, 2.25], y=[-0.71, 0.88]
# workpiece尺寸: ~0.56m x 0.44m, 旋转后最大半径~0.35m
# 收紧范围: 避开platform突起设备，确保工件完全在平坦桌面区域内
WORKPIECE_POS_RANGE = {
    "x": (0.65, 0.70),   # 相对于PLATFORM_POS的x偏移 (5cm小范围)
    "y": (0.05, 0.10),    # 相对于PLATFORM_POS的y偏移 (5cm小范围)
    "z": (0.0, 0.0),      # 工作台表面（由surface_z决定）
}
WORKPIECE_ROT_RANGE = {
    "roll": (0.0, 0.0),     # 绕x轴旋转范围（锁定）
    "pitch": (0.0, 0.0),    # 绕y轴旋转范围（锁定）
    "yaw": (-3.14, 3.14),   # 绕z轴旋转范围（弧度）
}

# 控制器参数
ARMATURE = 0.1
NATURAL_FREQ = 200 * 2.0 * 3.1415926535  # 200Hz — 高刚度，确保关节快速跟踪目标
DAMPING_RATIO = 1.0                        # 临界阻尼，无振荡
STIFFNESS = ARMATURE * NATURAL_FREQ ** 2
DAMPING = 2.0 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ


# ========================== 资源加载 ==========================

def get_mr2010_assets() -> dict[str, bytes]:
    """加载机械臂STL资源文件"""
    assets = {}
    for pattern in ("*.STL", "*.stl"):
        for stl_file in MR2010_XML.parent.glob(pattern):
            assets[stl_file.name] = stl_file.read_bytes()
    return assets


def get_mr2010_spec() -> mujoco.MjSpec:
    """创建MR2010机械臂的MjSpec"""
    spec = mujoco.MjSpec.from_file(str(MR2010_XML))
    spec.assets = get_mr2010_assets()
    return spec


# ========================== 场景辅助函数 ==========================

def add_ground_plane(spec: mujoco.MjSpec, with_texture: bool = True):
    """添加地面平面"""
    if with_texture:
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
    else:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0, 0, 0.01), rgba=(0.9, 0.9, 0.9, 1),
        )


def add_lights(spec: mujoco.MjSpec):
    """添加场景灯光"""
    spec.visual.headlight.diffuse = (0.6, 0.6, 0.6)
    spec.visual.headlight.ambient = (0.3, 0.3, 0.3)

    spec.worldbody.add_light(
        pos=(0, 0, 3), dir=(0, 0, -1),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=True, diffuse=(0.8, 0.8, 0.8),
    )


def add_cameras(spec: mujoco.MjSpec):
    """添加额外的评估用摄像头"""
    # front_cam: 近距离前方视角，聚焦机械臂末端和工件区域
    cam = spec.worldbody.add_camera()
    cam.name = "front_cam"
    cam.pos = [2.2400, 0.3800, 1.6500]
    cam.quat = [0.701453, 0.089236, 0.209687, 0.675301]
    cam.fovy = 49


def add_tool_tip_site(spec: mujoco.MjSpec):
    """在Link6上添加工具末端site，用于IK求解"""
    link6 = spec.worldbody.find_child("Link6")
    site = link6.add_site()
    site.name = "tool_tip"
    site.pos = [-0.089885, -0.030538, 0.38983]
    site.quat = [0.963075, 0.0766, -0.256027, 0.0326944]
    site.type = mujoco.mjtGeom.mjGEOM_SPHERE
    site.size = [0.005, 0, 0]


def add_joint_properties(spec: mujoco.MjSpec):
    """为机械臂关节添加armature和阻尼，确保仿真稳定性

    工业机器人的减速器典型传动比 100:1~160:1，
    折算到关节侧的等效转动惯量(armature)远大于连杆惯量。
    这里使用较大armature值来模拟真实减速器效果。
    """
    # 按关节负载分配 armature
    # damping 设为 0: 阻尼完全由 PD 控制器的 kd 项提供，避免双重阻尼
    joint_params = {
        "Joint1": {"armature": 5.0, "damping": 0.0},   # 基座旋转
        "Joint2": {"armature": 5.0, "damping": 0.0},   # 肩关节
        "Joint3": {"armature": 3.0, "damping": 0.0},   # 肘关节
        "Joint4": {"armature": 1.0, "damping": 0.0},   # 腕1
        "Joint5": {"armature": 0.5, "damping": 0.0},   # 腕2
        "Joint6": {"armature": 0.2, "damping": 0.0},   # 腕3
    }
    for joint_name in JOINT_NAMES:
        link_name = joint_name.replace("Joint", "Link")
        body = spec.worldbody.find_child(link_name)
        joint = body.first_joint()
        if joint is not None and joint.name == joint_name:
            joint.armature = joint_params[joint_name]["armature"]
            joint.damping = joint_params[joint_name]["damping"]


def add_actuators(spec: mujoco.MjSpec):
    """
    为机械臂关节添加位置控制actuator (PD控制器)

    力矩 = kp * (ctrl - qpos) - kd * qvel
    其中 ctrl 是目标关节角度（位置目标）

    增益基于临界阻尼设计: kp = armature * ω_n², kd = 2 * ζ * armature * ω_n
    其中 ω_n = 20 rad/s (快速响应), ζ = 1.0 (临界阻尼)
    """
    # 先设置关节属性
    add_joint_properties(spec)

    # PD增益: 基于实际有效惯量（armature + 连杆惯量）设计
    # decimation=10, dt=0.002 => 控制周期 0.02s
    # 目标: 关节在 2-3 个控制周期 (0.04-0.06s) 内到达目标
    # 临界阻尼: zeta=1.0, 98% 上升时间 ≈ 4/omega_n
    # omega_n=100 => t_98% ≈ 0.04s (2个控制周期)
    omega_n = 100.0  # 目标自然频率 (rad/s)
    zeta = 1.0       # 临界阻尼

    # 实际有效惯量 (从质量矩阵对角线测量, 包含 armature + 连杆惯量)
    effective_inertia = {
        "Joint1": 20.0, "Joint2": 40.0, "Joint3": 12.0,
        "Joint4": 1.5,  "Joint5": 0.6,  "Joint6": 0.2,
    }

    # MR2010 额定力矩限制 (Nm), 防止无限力推穿碰撞体
    torque_limits = {
        "Joint1": 300, "Joint2": 300, "Joint3": 150,
        "Joint4": 50,  "Joint5": 50,  "Joint6": 30,
    }

    for joint_name in JOINT_NAMES:
        J = effective_inertia[joint_name]
        kp = J * omega_n ** 2           # 刚度
        kd = 2.0 * zeta * J * omega_n   # 阻尼
        tau_max = torque_limits[joint_name]
        act = spec.add_actuator(
            name=f"act_{joint_name}",
            target=joint_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            dyntype=mujoco.mjtDyn.mjDYN_NONE,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        )
        act.gainprm[0] = kp
        act.biasprm[1] = -kp   # -kp * qpos
        act.biasprm[2] = -kd   # -kd * qvel
        act.ctrlrange = (-3.14, 3.14)
        act.forcerange = (-tau_max, tau_max)


def add_platform(spec: mujoco.MjSpec, pos: tuple = None) -> None:
    """
    添加工作台到场景

    Args:
        spec: MjSpec对象
        pos: 工作台位置 (x, y, z)，默认使用PLATFORM_POS
    """
    import os

    if pos is None:
        pos = PLATFORM_POS

    # 加载简化后的platform mesh (原始537670面 -> 134417面)
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    platform_stl = os.path.join(assets_dir, "platform_simplified.stl")
    spec.add_mesh(
        name="platform",
        file=platform_stl,
        scale=(0.001, 0.001, 0.001),  # STL是毫米单位，转换为米
    )

    platform_body = spec.worldbody.add_body(name="platform")
    # 翻转后需要调整z位置，使底面在地面上
    # platform高度约0.636米，翻转后需要抬高
    # y方向偏移-0.7使platform居中于机械臂正前方
    platform_body.pos = (pos[0], pos[1] - 0.7, pos[2] + 0.64)
    # 组合旋转：绕x轴180度(翻转) + 绕z轴90度
    # quat = (0, 0.707, 0.707, 0) 约等于 (0, sqrt(2)/2, sqrt(2)/2, 0)
    platform_body.quat = (0, 0.7071, 0.7071, 0)
    # 视觉mesh（禁用碰撞）
    platform_body.add_geom(
        name="platform_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="platform",
        contype=0,
        conaffinity=0,
    )

    # 加载CoACD凸分解碰撞mesh
    convex_dir = os.path.join(assets_dir, "platform_convex")
    convex_files = sorted(
        f for f in os.listdir(convex_dir) if f.endswith(".stl")
    )
    for cf in convex_files:
        mesh_name = f"platform_col_{cf[:-4]}"
        spec.add_mesh(
            name=mesh_name,
            file=os.path.join(convex_dir, cf),
            scale=(0.001, 0.001, 0.001),
        )
        platform_body.add_geom(
            name=mesh_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            rgba=(0, 0, 0, 0),
            contype=3,
            conaffinity=3,
        )


def euler_to_quat(roll: float, pitch: float, yaw: float) -> tuple:
    """将欧拉角转换为四元数 (w, x, y, z)"""
    import math
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


def add_workpiece(
    spec: mujoco.MjSpec,
    pos: tuple = None,
    quat: tuple = None,
    platform_pos: tuple = None,
    platform_height: float = None,
    randomize: bool = False,
) -> None:
    """
    添加工件到场景

    Args:
        spec: MjSpec对象
        pos: 工件位置 (x, y, z)，如果randomize=True则忽略
        quat: 工件姿态四元数 (w, x, y, z)，如果randomize=True则忽略
        platform_pos: 工作台位置，用于计算工件相对位置
        platform_height: 工作台高度（2倍半高），默认使用PLATFORM_SIZE[2]*2
        randomize: 是否随机生成位置和姿态
    """
    import numpy as np

    if platform_pos is None:
        platform_pos = PLATFORM_POS

    # workpiece从桌面上方开始下落，freejoint让其自然落到platform上
    workpiece_base_pos = (platform_pos[0], platform_pos[1], platform_pos[2])
    surface_z = workpiece_base_pos[2] + 0.54

    if randomize:
        # 随机生成位置（相对于工作台）
        rel_x = np.random.uniform(*WORKPIECE_POS_RANGE["x"])
        rel_y = np.random.uniform(*WORKPIECE_POS_RANGE["y"])
        pos = (
            workpiece_base_pos[0] + rel_x,
            workpiece_base_pos[1] + rel_y,
            surface_z,
        )

        # 绕z轴随机旋转 + 绕x轴翻转180°（正面朝上）
        # q_final = q_yaw * q_flip (Hamilton乘法)
        yaw = np.random.uniform(*WORKPIECE_ROT_RANGE["yaw"])
        q_yaw = euler_to_quat(0.0, 0.0, yaw)  # (w, x, y, z)
        q_flip = (0.0, 1.0, 0.0, 0.0)          # 绕x轴180°
        w1, x1, y1, z1 = q_yaw
        w2, x2, y2, z2 = q_flip
        quat = (
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        )
    else:
        if pos is None:
            # 默认放在桌面中心（PLATFORM_POS是body锚点，桌面中心有偏移）
            cx = (WORKPIECE_POS_RANGE["x"][0] + WORKPIECE_POS_RANGE["x"][1]) / 2
            cy = (WORKPIECE_POS_RANGE["y"][0] + WORKPIECE_POS_RANGE["y"][1]) / 2
            pos = (workpiece_base_pos[0] + cx, workpiece_base_pos[1] + cy, surface_z)
        if quat is None:
            quat = (0, 1, 0, 0)  # 绕x轴180度，正面朝上

    # 添加工件mesh（STL文件单位是毫米，需要缩放到米）
    spec.add_mesh(
        name="workpiece",
        content_type="model/stl",
        file="workpeice.STL",
        scale=(0.001, 0.001, 0.001),  # mm -> m
    )
    # 加载STL文件
    if WORKPIECE_STL.exists():
        spec.assets["workpeice.STL"] = WORKPIECE_STL.read_bytes()

    # workpiece带freejoint，自然落到工作台上
    workpiece_body = spec.worldbody.add_body(name="workpiece")
    workpiece_body.pos = pos
    workpiece_body.quat = quat
    workpiece_body.add_freejoint(name="workpiece_joint")

    # 视觉mesh（禁用碰撞），向下偏移10mm补偿CoACD凸块膨胀间隙
    workpiece_body.add_geom(
        name="workpiece_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="workpiece",
        mass=0.5,
        contype=0,
        conaffinity=0,
        pos=(0, 0, 0.0112),  # 补偿凸块碰撞产生的视觉间隙
    )

    # 加载CoACD凸分解碰撞mesh
    import os
    convex_dir = os.path.join(os.path.dirname(__file__), "assets", "workpiece_convex")
    convex_files = sorted(
        f for f in os.listdir(convex_dir) if f.endswith(".stl")
    )
    for cf in convex_files:
        mesh_name = f"wp_col_{cf[:-4]}"
        spec.add_mesh(
            name=mesh_name,
            file=os.path.join(convex_dir, cf),
            scale=(0.001, 0.001, 0.001),
        )
        workpiece_body.add_geom(
            name=mesh_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            rgba=(0, 0, 0, 0),
            contype=3,
            conaffinity=3,
        )


def setup_robot_collision(spec: mujoco.MjSpec):
    """配置机械臂连杆的碰撞组

    使用 MuJoCo 碰撞组 + exclude 实现:
    - 机械臂 vs 机械臂(自碰撞): 启用 — 非相邻连杆之间检测碰撞
    - 相邻连杆: 排除 — STL mesh 在关节处重叠，用 exclude 禁用
    - 机械臂 vs 地面: 禁用 — 避免 Link1 与地面摩擦导致 Joint1 无法转动
    - 机械臂 vs 工作台/工件: 启用 — 物理上正确的碰撞行为

    碰撞组设计:
    - 地面/base_link: contype=1, conaffinity=1 (默认, group 1)
    - 机械臂 Link1-Link6: contype=2, conaffinity=2 (group 2, 互相碰撞)
    - 工作台/工件碰撞体: contype=3, conaffinity=3 (group 1+2, 与所有物体碰撞)

    碰撞矩阵 (contype_A & conaffinity_B) | (contype_B & conaffinity_A):
    - Link vs Link:      (2&2)|(2&2) = 2 → 碰撞 ✓ (非相邻)
    - Link vs 地面:      (2&1)|(1&2) = 0 → 不碰撞 ✓
    - Link vs base_link: (2&1)|(1&2) = 0 → 不碰撞 ✓
    - Link vs 工作台:    (2&3)|(3&2) = 2 → 碰撞 ✓
    """
    link_names = ("Link1", "Link2", "Link3", "Link4", "Link5", "Link6")

    for link_name in link_names:
        body = spec.worldbody.find_child(link_name)
        if body is None:
            continue
        geom = body.first_geom()
        while geom is not None:
            geom.contype = 2
            geom.conaffinity = 2
            geom = body.next_geom(geom)

    # 排除相邻连杆对的碰撞 (STL mesh 在关节处重叠)
    adjacent_pairs = list(zip(link_names[:-1], link_names[1:]))
    for body1, body2 in adjacent_pairs:
        spec.add_exclude(
            name=f"adj_{body1}_{body2}",
            bodyname1=body1,
            bodyname2=body2,
        )


def create_base_scene(
    with_actuators: bool = True,
    with_platform: bool = False,
    with_workpiece: bool = False,
    randomize_workpiece: bool = False,
) -> mujoco.MjSpec:
    """
    创建基础场景（包含机械臂、地面、灯光）

    Args:
        with_actuators: 是否添加actuator（模仿学习需要，强化学习由mjlab管理）
        with_platform: 是否添加工作台
        with_workpiece: 是否添加工件
        randomize_workpiece: 是否随机化工件位置和姿态

    Returns:
        配置好的MjSpec
    """
    spec = mujoco.MjSpec()
    spec.modelname = "MR2010 Scene"

    # 求解器配置: 高增益PD控制器需要更硬的接触约束
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 50
    spec.option.impratio = 100   # 接触阻抗比，默认1，提高后接触更硬，防止高力推穿

    add_lights(spec)
    add_ground_plane(spec)

    # 附加机械臂
    robot_spec = get_mr2010_spec()
    frame = spec.worldbody.add_frame()
    spec.attach(robot_spec, prefix="", frame=frame)

    # 添加工具末端site和摄像头（必须在attach之后、compile之前）
    add_tool_tip_site(spec)
    add_cameras(spec)

    # 配置碰撞组: 机械臂不与地面碰撞，但与platform/workpiece碰撞
    setup_robot_collision(spec)

    if with_actuators:
        add_actuators(spec)

    if with_platform:
        add_platform(spec)

    if with_workpiece:
        add_workpiece(spec, randomize=randomize_workpiece)

    return spec


# ========================== 测试 ==========================

if __name__ == "__main__":
    import sys
    import mujoco.viewer

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "base":
        # 仅机械臂
        spec = create_base_scene()
        print("Mode: base (robot only)")
    elif mode == "platform":
        # 机械臂 + 工作台
        spec = create_base_scene(with_platform=True)
        print("Mode: platform (robot + platform)")
    elif mode == "full":
        # 完整场景（机械臂 + 工作台 + 工件）
        spec = create_base_scene(with_platform=True, with_workpiece=True)
        print("Mode: full (robot + platform + workpiece)")
    elif mode == "random":
        # 随机工件位置
        spec = create_base_scene(
            with_platform=True, with_workpiece=True, randomize_workpiece=True
        )
        print("Mode: random (robot + platform + random workpiece)")
    else:
        print(f"Usage: python mr2010_robot.py [base|platform|full|random]")
        sys.exit(1)

    model = spec.compile()
    data = mujoco.MjData(model)

    # 初始化ctrl=qpos，让PD控制器从初始位置开始保持，避免重力引起振荡
    data.ctrl[:model.nu] = data.qpos[:model.nu]
    mujoco.mj_forward(model, data)

    print(f"Model: {model.ngeom} geoms, {model.njnt} joints, {model.nu} actuators")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
