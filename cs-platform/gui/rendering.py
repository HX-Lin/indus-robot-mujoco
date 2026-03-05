"""
MuJoCo 场景渲染共享工具函数

提供摄像头渲染、场景创建、参数转换等 GUI 共用功能。
"""

import sys
import math
import threading
from pathlib import Path

import numpy as np
import mujoco

# 确保能导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from mr2010_robot import create_base_scene, NUM_JOINTS, JOINT_NAMES

# 全局渲染锁（mujoco.Renderer 不是线程安全的）
_render_lock = threading.Lock()


def create_scene_and_settle(
    with_workpiece: bool = True,
    randomize: bool = False,
    settle_steps: int = 2000,
):
    """创建场景并让工件稳定落到平台上

    Returns:
        (model, data, settled_qpos)
    """
    spec = create_base_scene(
        with_actuators=True,
        with_platform=True,
        with_workpiece=with_workpiece,
        randomize_workpiece=randomize,
    )
    model = spec.compile()
    data = mujoco.MjData(model)

    # 初始化控制器，让PD控制器保持当前位置
    data.ctrl[:model.nu] = data.qpos[:model.nu]
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    settled_qpos = data.qpos.copy()
    return model, data, settled_qpos


def render_camera_view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """渲染单个摄像头视图，返回 RGB numpy 数组"""
    with _render_lock:
        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data, camera=camera_name)
        img = renderer.render().copy()
        renderer.close()
    return img


def render_all_cameras_grid(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cell_width: int = 320,
    cell_height: int = 240,
) -> np.ndarray:
    """渲染所有摄像头并拼成网格图"""
    from PIL import Image, ImageDraw, ImageFont

    ncam = model.ncam
    if ncam == 0:
        return np.zeros((cell_height, cell_width, 3), dtype=np.uint8)

    cols = min(ncam, 3)
    rows = math.ceil(ncam / cols)

    grid = np.zeros((rows * cell_height, cols * cell_width, 3), dtype=np.uint8)

    with _render_lock:
        renderer = mujoco.Renderer(model, height=cell_height, width=cell_width)
        for i in range(ncam):
            cam_name = model.camera(i).name
            renderer.update_scene(data, camera=cam_name)
            img = renderer.render()

            r, c = divmod(i, cols)
            grid[r * cell_height:(r + 1) * cell_height,
                 c * cell_width:(c + 1) * cell_width] = img
        renderer.close()

    # 添加摄像头名称标签
    pil_img = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_img)
    for i in range(ncam):
        cam_name = model.camera(i).name
        r, c = divmod(i, cols)
        x = c * cell_width + 5
        y = r * cell_height + 5
        draw.text((x + 1, y + 1), cam_name, fill=(0, 0, 0))
        draw.text((x, y), cam_name, fill=(255, 255, 0))

    return np.array(pil_img)


def set_robot_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_joints: list | np.ndarray,
):
    """设置机械臂关节角并更新状态"""
    data.qpos[:NUM_JOINTS] = qpos_joints
    data.ctrl[:NUM_JOINTS] = qpos_joints
    mujoco.mj_forward(model, data)


def apply_camera_params(
    model: mujoco.MjModel,
    cam_id: int,
    pos: list = None,
    quat: list = None,
    fovy: float = None,
):
    """修改摄像头参数"""
    if pos is not None:
        model.cam_pos[cam_id] = pos
    if quat is not None:
        model.cam_quat[cam_id] = quat
    if fovy is not None:
        model.cam_fovy[cam_id] = fovy


def get_camera_names(model: mujoco.MjModel) -> list[str]:
    """获取所有摄像头名称"""
    return [model.camera(i).name for i in range(model.ncam)]


def get_camera_info(model: mujoco.MjModel, cam_name: str) -> dict:
    """获取摄像头参数"""
    cam_id = model.camera(cam_name).id
    cam = model.camera(cam_id)
    body_id = cam.bodyid
    body_name = model.body(body_id).name if body_id >= 0 else "world"
    return {
        "name": cam_name,
        "id": cam_id,
        "pos": cam.pos.copy(),
        "quat": cam.quat.copy(),
        "fovy": model.cam_fovy[cam_id],
        "body": body_name,
    }


def generate_xml_snippet(model: mujoco.MjModel) -> str:
    """生成所有摄像头的 XML 片段"""
    lines = []
    for i in range(model.ncam):
        cam = model.camera(i)
        name = cam.name
        pos = cam.pos
        quat = cam.quat
        fovy = model.cam_fovy[i]
        body_id = cam.bodyid
        body_name = model.body(body_id).name if body_id >= 0 else "worldbody"

        pos_str = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
        quat_str = f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"

        lines.append(f'<!-- parent: {body_name} -->')
        lines.append(
            f'<camera name="{name}" pos="{pos_str}" '
            f'quat="{quat_str}" fovy="{fovy:.0f}"/>'
        )
        lines.append("")
    return "\n".join(lines)


def save_cameras_to_files(model: mujoco.MjModel) -> str:
    """将当前摄像头参数保存回源文件

    - MR2010.xml 中定义的摄像头 (overhead_cam, ee_cam) 直接更新 XML
    - 代码中添加的摄像头 (front_cam) 更新 mr2010_robot.py
    """
    import re

    xml_path = Path(__file__).parent.parent / "assets" / "MR2010.xml"
    py_path = Path(__file__).parent.parent / "mr2010_robot.py"

    xml_text = xml_path.read_text(encoding="utf-8")
    py_text = py_path.read_text(encoding="utf-8")

    xml_updated = []
    py_updated = []

    for i in range(model.ncam):
        cam = model.camera(i)
        name = cam.name
        pos = cam.pos
        quat = cam.quat
        fovy = int(round(model.cam_fovy[i]))

        pos_xml = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
        quat_xml = f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"

        # 尝试在 XML 中更新
        cam_pattern = r'<camera\s+name="' + re.escape(name) + r'"[^/]*/>'
        if re.search(cam_pattern, xml_text):
            new_tag = (
                f'<camera name="{name}" pos="{pos_xml}" '
                f'quat="{quat_xml}" fovy="{fovy}"/>'
            )
            xml_text = re.sub(cam_pattern, new_tag, xml_text)
            xml_updated.append(name)
            continue

        # 尝试在 Python 代码中更新
        name_lit = f'cam.name = "{name}"'
        idx = py_text.find(name_lit)
        if idx >= 0:
            after = py_text[idx + len(name_lit):]
            after = re.sub(
                r'cam\.pos\s*=\s*\[[^\]]*\]',
                f'cam.pos = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]',
                after, count=1,
            )
            after = re.sub(
                r'cam\.quat\s*=\s*\[[^\]]*\]',
                f'cam.quat = [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]',
                after, count=1,
            )
            after = re.sub(
                r'cam\.fovy\s*=\s*[\d.]+',
                f'cam.fovy = {fovy}',
                after, count=1,
            )
            py_text = py_text[:idx + len(name_lit)] + after
            py_updated.append(name)

    msgs = []
    if xml_updated:
        xml_path.write_text(xml_text, encoding="utf-8")
        msgs.append(f"MR2010.xml: {', '.join(xml_updated)}")
    if py_updated:
        py_path.write_text(py_text, encoding="utf-8")
        msgs.append(f"mr2010_robot.py: {', '.join(py_updated)}")

    if msgs:
        return "已保存: " + "; ".join(msgs)
    return "未找到可保存的摄像头"


def euler_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """欧拉角(度) -> 四元数 (w, x, y, z)"""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def quat_to_euler(quat: np.ndarray) -> tuple[float, float, float]:
    """四元数 (w, x, y, z) -> 欧拉角(度) (roll, pitch, yaw)"""
    w, x, y, z = quat

    # Roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def add_seam_markers(spec, seam, num_markers=30):
    """在场景中添加焊缝路径标记点（小球），方便可视化

    Args:
        spec: MjSpec (compile 前调用)
        seam: SeamPath 对象
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


def render_orbit_view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    lookat: list | np.ndarray,
    distance: float = 2.0,
    azimuth: float = 90.0,
    elevation: float = -30.0,
    width: int = 800,
    height: int = 600,
) -> np.ndarray:
    """使用自由轨道摄像头渲染场景

    Args:
        lookat: 相机注视点 [x, y, z]
        distance: 相机到注视点的距离
        azimuth: 水平旋转角度 (度)
        elevation: 俯仰角度 (度, 负值=俯视)
    """
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

    with _render_lock:
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            renderer.update_scene(data, camera=cam)
        except TypeError:
            # Fallback: render without free camera if MjvCamera not supported
            renderer.update_scene(data)
        img = renderer.render().copy()
        renderer.close()
    return img


# 机器人姿态预设
POSE_PRESETS = {
    "home":    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    "forward": [0.0, -0.5,  0.5,  0.0,  0.3,  0.0],
    "up":      [0.0, -1.2,  0.0,  0.0,  0.0,  0.0],
    "left":    [-1.57, -0.5, 0.5, 0.0,  0.0,  0.0],
    "right":   [1.57, -0.5,  0.5, 0.0,  0.0,  0.0],
}
