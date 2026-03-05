"""
MuJoCo 摄像头位置交互式调试工具

功能:
  - 实时渲染所有摄像头画面并保存到文件
  - 通过命令行交互式调整摄像头 pos / quat / fovy
  - 可切换机械臂姿态查看不同构型下的摄像头效果
  - 输出最终参数供粘贴到 MR2010.xml

用法:
    python debug_cameras.py                     # 使用默认场景
    python debug_cameras.py --qpos 0 -0.5 0.5 0 0.3 0  # 指定初始关节角
    python debug_cameras.py --output_dir /tmp/cam_debug  # 指定输出目录

交互命令:
    render / r          — 渲染所有摄像头并保存图像
    select <name>       — 选择要调整的摄像头
    pos <x> <y> <z>     — 设置当前摄像头位置
    quat <w> <x> <y> <z> — 设置当前摄像头四元数
    fovy <angle>        — 设置当前摄像头视场角
    qpos <j1> ... <j6>  — 设置机械臂关节角
    preset <name>       — 切换预设姿态 (home, forward, up, left, right)
    list                — 列出所有摄像头
    info                — 显示当前摄像头参数
    xml                 — 输出所有摄像头的 XML 片段
    help / h            — 显示帮助
    quit / q            — 退出
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

# 确保能导入项目模块
sys.path.insert(0, str(Path(__file__).parent))
from mr2010_robot import create_base_scene, NUM_JOINTS


# 预设关节角
POSE_PRESETS = {
    "home":    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    "forward": [0.0, -0.5,  0.5,  0.0,  0.3,  0.0],
    "up":      [0.0, -1.2,  0.0,  0.0,  0.0,  0.0],
    "left":    [-1.57, -0.5, 0.5, 0.0,  0.0,  0.0],
    "right":   [1.57, -0.5,  0.5, 0.0,  0.0,  0.0],
}


def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo Camera Debug Tool")
    parser.add_argument(
        "--qpos", type=float, nargs=6, default=None,
        help="Initial joint angles (6 values)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/tmp/cam_debug",
        help="Directory to save rendered images (default: /tmp/cam_debug)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Render width (default: 640)",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Render height (default: 480)",
    )
    return parser.parse_args()


def render_all_cameras(model, data, renderer, output_dir, width, height):
    """渲染所有摄像头并保存图像"""
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Rendering {model.ncam} cameras ({width}x{height})...")
    for i in range(model.ncam):
        cam_name = model.camera(i).name
        renderer.update_scene(data, camera=cam_name)
        img = renderer.render()
        path = output_dir / f"{cam_name}.png"
        Image.fromarray(img).save(str(path))

        # 简单图像统计
        total = img.shape[0] * img.shape[1]
        non_black = np.count_nonzero(img.sum(axis=2))
        print(f"    {cam_name}: saved -> {path}  (coverage: {100*non_black/total:.0f}%)")


def get_camera_info(model, cam_name):
    """获取摄像头参数"""
    cam_id = model.camera(cam_name).id
    cam = model.camera(cam_id)

    # 获取摄像头所在的 body
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


def print_camera_info(info):
    """打印摄像头信息"""
    print(f"\n  Camera: {info['name']} (id={info['id']})")
    print(f"    body:  {info['body']}")
    print(f"    pos:   {info['pos'][0]:.4f} {info['pos'][1]:.4f} {info['pos'][2]:.4f}")
    print(f"    quat:  {info['quat'][0]:.6f} {info['quat'][1]:.6f} {info['quat'][2]:.6f} {info['quat'][3]:.6f}")
    print(f"    fovy:  {info['fovy']:.1f}")


def generate_xml_snippet(model):
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
        lines.append(f'<camera name="{name}" pos="{pos_str}" quat="{quat_str}" fovy="{fovy:.0f}"/>')
        lines.append("")

    return "\n".join(lines)


def print_help():
    print("""
  Commands:
    render / r            Render all cameras and save images
    select <name>         Select camera to adjust
    pos <x> <y> <z>       Set camera local position
    quat <w> <x> <y> <z>  Set camera quaternion
    fovy <angle>          Set field of view (degrees)
    qpos <j1> ... <j6>    Set joint angles
    preset <name>         Apply pose preset (home/forward/up/left/right)
    list                  List all cameras
    info                  Show selected camera parameters
    xml                   Output XML snippets for all cameras
    help / h              Show this help
    quit / q              Exit
    """)


def main():
    args = parse_args()

    print("=" * 60)
    print("MuJoCo Camera Debug Tool")
    print("=" * 60)

    # 创建场景
    spec = create_base_scene(with_actuators=True, with_platform=True)
    model = spec.compile()
    data = mujoco.MjData(model)

    # 设置初始姿态
    if args.qpos:
        data.qpos[:NUM_JOINTS] = args.qpos
    else:
        data.qpos[:NUM_JOINTS] = POSE_PRESETS["forward"]
    data.ctrl[:NUM_JOINTS] = data.qpos[:NUM_JOINTS]
    mujoco.mj_forward(model, data)

    print(f"\n  Model: {model.ngeom} geoms, {model.njnt} joints, {model.ncam} cameras")
    print(f"  Output: {args.output_dir}")
    print(f"  Resolution: {args.width}x{args.height}")

    # 列出摄像头
    print(f"\n  Cameras:")
    for i in range(model.ncam):
        cam = model.camera(i)
        body_id = cam.bodyid
        body = model.body(body_id).name if body_id >= 0 else "world"
        print(f"    [{i}] {cam.name}  (on {body})")

    # 创建渲染器
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    # 初始渲染
    render_all_cameras(model, data, renderer, args.output_dir, args.width, args.height)

    # 交互循环
    selected_cam = model.camera(0).name if model.ncam > 0 else None
    print(f"\n  Selected camera: {selected_cam}")
    print("  Type 'help' for commands.\n")

    while True:
        try:
            line = input(f"  [{selected_cam}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd in ("quit", "q", "exit"):
                break

            elif cmd in ("help", "h"):
                print_help()

            elif cmd in ("render", "r"):
                mujoco.mj_forward(model, data)
                render_all_cameras(model, data, renderer, args.output_dir, args.width, args.height)

            elif cmd == "list":
                for i in range(model.ncam):
                    cam = model.camera(i)
                    body_id = cam.bodyid
                    body = model.body(body_id).name if body_id >= 0 else "world"
                    marker = " *" if cam.name == selected_cam else ""
                    print(f"    [{i}] {cam.name}  (on {body}){marker}")

            elif cmd == "select":
                name = parts[1]
                # 支持通过索引选择
                try:
                    idx = int(name)
                    name = model.camera(idx).name
                except ValueError:
                    pass
                selected_cam = name
                info = get_camera_info(model, selected_cam)
                print_camera_info(info)

            elif cmd == "info":
                if selected_cam:
                    info = get_camera_info(model, selected_cam)
                    print_camera_info(info)
                else:
                    print("    No camera selected")

            elif cmd == "pos":
                if len(parts) != 4:
                    print("    Usage: pos <x> <y> <z>")
                    continue
                cam_id = model.camera(selected_cam).id
                new_pos = [float(parts[1]), float(parts[2]), float(parts[3])]
                model.cam_pos[cam_id] = new_pos
                print(f"    pos -> {new_pos}")
                mujoco.mj_forward(model, data)
                render_all_cameras(model, data, renderer, args.output_dir, args.width, args.height)

            elif cmd == "quat":
                if len(parts) != 5:
                    print("    Usage: quat <w> <x> <y> <z>")
                    continue
                cam_id = model.camera(selected_cam).id
                new_quat = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                model.cam_quat[cam_id] = new_quat
                print(f"    quat -> {new_quat}")
                mujoco.mj_forward(model, data)
                render_all_cameras(model, data, renderer, args.output_dir, args.width, args.height)

            elif cmd == "fovy":
                if len(parts) != 2:
                    print("    Usage: fovy <angle>")
                    continue
                cam_id = model.camera(selected_cam).id
                model.cam_fovy[cam_id] = float(parts[1])
                print(f"    fovy -> {parts[1]}")
                mujoco.mj_forward(model, data)
                render_all_cameras(model, data, renderer, args.output_dir, args.width, args.height)

            elif cmd == "qpos":
                if len(parts) != 7:
                    print("    Usage: qpos <j1> <j2> <j3> <j4> <j5> <j6>")
                    continue
                qpos = [float(x) for x in parts[1:7]]
                data.qpos[:NUM_JOINTS] = qpos
                data.ctrl[:NUM_JOINTS] = qpos
                mujoco.mj_forward(model, data)

                site_id = model.site("tool_tip").id
                ee_pos = data.site_xpos[site_id]
                print(f"    qpos -> {qpos}")
                print(f"    tool_tip -> ({ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f})")
                render_all_cameras(model, data, renderer, args.output_dir, args.width, args.height)

            elif cmd == "preset":
                if len(parts) != 2 or parts[1] not in POSE_PRESETS:
                    print(f"    Available presets: {list(POSE_PRESETS.keys())}")
                    continue
                qpos = POSE_PRESETS[parts[1]]
                data.qpos[:NUM_JOINTS] = qpos
                data.ctrl[:NUM_JOINTS] = qpos
                mujoco.mj_forward(model, data)

                site_id = model.site("tool_tip").id
                ee_pos = data.site_xpos[site_id]
                print(f"    preset '{parts[1]}' -> qpos={qpos}")
                print(f"    tool_tip -> ({ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f})")
                render_all_cameras(model, data, renderer, args.output_dir, args.width, args.height)

            elif cmd == "xml":
                snippet = generate_xml_snippet(model)
                print(f"\n{snippet}")

            else:
                print(f"    Unknown command: {cmd}. Type 'help' for commands.")

        except Exception as e:
            print(f"    Error: {e}")

    renderer.close()
    print("\n  Final camera XML:")
    print(generate_xml_snippet(model))
    print("Done.")


if __name__ == "__main__":
    main()
