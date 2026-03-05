"""
MuJoCo 摄像头位置可视化调节器

基于 Gradio 的交互式 GUI，实时预览摄像头渲染效果，
用滑块调整 pos/euler/fovy 参数，输出 XML 片段。

用法:
    cd cs-platform
    python -m gui.camera_tuner
"""

import sys
from pathlib import Path

import numpy as np
import mujoco
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent))
from gui.rendering import (
    create_scene_and_settle,
    render_camera_view,
    render_all_cameras_grid,
    set_robot_pose,
    apply_camera_params,
    get_camera_names,
    get_camera_info,
    generate_xml_snippet,
    save_cameras_to_files,
    euler_to_quat,
    quat_to_euler,
    POSE_PRESETS,
    NUM_JOINTS,
)

# ========================== 全局状态 ==========================

_model = None
_data = None
_settled_qpos = None


def init_scene():
    """初始化场景（仅在启动时调用一次）"""
    global _model, _data, _settled_qpos
    print("正在初始化场景...")
    _model, _data, _settled_qpos = create_scene_and_settle(with_workpiece=True)
    set_robot_pose(_model, _data, POSE_PRESETS["forward"])
    print(f"场景就绪: {_model.ncam} 个摄像头, {_model.njnt} 个关节")


# ========================== 回调函数 ==========================


def on_camera_select(cam_name):
    """切换摄像头时，更新所有滑块到该摄像头的当前参数"""
    if _model is None or not cam_name:
        return [gr.Slider()] * 7 + [gr.Textbox()]

    info = get_camera_info(_model, cam_name)
    roll, pitch, yaw = quat_to_euler(info["quat"])

    return [
        gr.Slider(value=float(info["pos"][0])),  # pos_x
        gr.Slider(value=float(info["pos"][1])),  # pos_y
        gr.Slider(value=float(info["pos"][2])),  # pos_z
        gr.Slider(value=float(roll)),             # roll
        gr.Slider(value=float(pitch)),            # pitch
        gr.Slider(value=float(yaw)),              # yaw
        gr.Slider(value=float(info["fovy"])),     # fovy
        gr.Textbox(value=f"所属 body: {info['body']}"),
    ]


def on_pose_preset(preset_name):
    """切换机器人姿态预设，返回6个关节角滑块值"""
    if preset_name not in POSE_PRESETS:
        return [gr.Slider()] * 6
    joints = POSE_PRESETS[preset_name]
    return [gr.Slider(value=float(j)) for j in joints]


def render_and_update(
    cam_name,
    pos_x, pos_y, pos_z,
    roll, pitch, yaw,
    fovy,
    j1, j2, j3, j4, j5, j6,
):
    """核心渲染回调：应用所有参数并渲染"""
    if _model is None or not cam_name:
        empty = np.zeros((480, 640, 3), dtype=np.uint8)
        return empty, empty, ""

    # 应用关节角
    set_robot_pose(_model, _data, [j1, j2, j3, j4, j5, j6])

    # 应用摄像头参数
    cam_id = _model.camera(cam_name).id
    quat = euler_to_quat(roll, pitch, yaw)
    apply_camera_params(_model, cam_id,
                        pos=[pos_x, pos_y, pos_z],
                        quat=quat.tolist(),
                        fovy=fovy)
    mujoco.mj_forward(_model, _data)

    # 渲染
    main_view = render_camera_view(_model, _data, cam_name, 640, 480)
    grid_view = render_all_cameras_grid(_model, _data, 320, 240)
    xml = generate_xml_snippet(_model)

    return main_view, grid_view, xml


# ========================== GUI 布局 ==========================


def build_app():
    init_scene()
    cam_names = get_camera_names(_model)
    default_cam = cam_names[0] if cam_names else ""
    default_info = get_camera_info(_model, default_cam) if default_cam else None

    # 默认参数
    if default_info:
        d_pos = default_info["pos"]
        d_roll, d_pitch, d_yaw = quat_to_euler(default_info["quat"])
        d_fovy = default_info["fovy"]
        d_body = f"所属 body: {default_info['body']}"
    else:
        d_pos = [0, 0, 1]
        d_roll = d_pitch = d_yaw = 0.0
        d_fovy = 45.0
        d_body = ""

    d_joints = POSE_PRESETS["forward"]

    # 预渲染初始画面
    set_robot_pose(_model, _data, d_joints)
    mujoco.mj_forward(_model, _data)
    init_main = render_camera_view(_model, _data, default_cam, 640, 480)
    init_grid = render_all_cameras_grid(_model, _data, 320, 240)
    init_xml = generate_xml_snippet(_model)

    with gr.Blocks(title="MuJoCo 摄像头调节器") as app:
        gr.Markdown("# MuJoCo 摄像头位置调节器")
        gr.Markdown("实时调整摄像头参数，预览渲染效果，输出 XML 片段")

        # ---- 选择栏 ----
        with gr.Row():
            cam_dropdown = gr.Dropdown(
                choices=cam_names, value=default_cam,
                label="摄像头", scale=2,
            )
            preset_dropdown = gr.Dropdown(
                choices=list(POSE_PRESETS.keys()), value="forward",
                label="机器人姿态预设", scale=2,
            )
            body_info = gr.Textbox(
                value=d_body, label="摄像头挂载", interactive=False, scale=1,
            )

        # ---- 渲染预览 ----
        with gr.Row():
            main_image = gr.Image(
                value=init_main,
                label="选中摄像头视图", height=480, width=640,
            )
            grid_image = gr.Image(
                value=init_grid,
                label="所有摄像头网格", height=480,
            )

        # ---- 摄像头参数 ----
        gr.Markdown("### 摄像头参数")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**位置 (局部坐标)**")
                pos_x = gr.Slider(-2.0, 5.0, value=float(d_pos[0]),
                                  step=0.01, label="X")
                pos_y = gr.Slider(-3.0, 3.0, value=float(d_pos[1]),
                                  step=0.01, label="Y")
                pos_z = gr.Slider(-1.0, 5.0, value=float(d_pos[2]),
                                  step=0.01, label="Z")
            with gr.Column():
                gr.Markdown("**朝向 (欧拉角, 度)**")
                roll = gr.Slider(-180, 180, value=float(d_roll),
                                 step=0.5, label="Roll")
                pitch = gr.Slider(-180, 180, value=float(d_pitch),
                                  step=0.5, label="Pitch")
                yaw = gr.Slider(-180, 180, value=float(d_yaw),
                                step=0.5, label="Yaw")
            with gr.Column():
                gr.Markdown("**视场角**")
                fovy = gr.Slider(10, 120, value=float(d_fovy),
                                 step=1, label="FOV (度)")

        # ---- 关节角 ----
        gr.Markdown("### 机械臂关节角 (rad)")
        with gr.Row():
            j1 = gr.Slider(-3.14, 3.14, value=d_joints[0], step=0.01,
                           label="Joint1")
            j2 = gr.Slider(-3.14, 3.14, value=d_joints[1], step=0.01,
                           label="Joint2")
            j3 = gr.Slider(-3.14, 3.14, value=d_joints[2], step=0.01,
                           label="Joint3")
        with gr.Row():
            j4 = gr.Slider(-3.14, 3.14, value=d_joints[3], step=0.01,
                           label="Joint4")
            j5 = gr.Slider(-3.14, 3.14, value=d_joints[4], step=0.01,
                           label="Joint5")
            j6 = gr.Slider(-3.14, 3.14, value=d_joints[5], step=0.01,
                           label="Joint6")

        # ---- XML 输出 + 保存按钮 ----
        gr.Markdown("### XML 片段")
        xml_output = gr.Code(value=init_xml, language="html", label="Camera XML")
        with gr.Row():
            save_btn = gr.Button("保存到文件", variant="primary", scale=1)
            save_status = gr.Textbox(
                value="", label="保存状态", interactive=False, scale=3,
            )

        # ---- 所有参数滑块列表 ----
        all_params = [
            cam_dropdown,
            pos_x, pos_y, pos_z,
            roll, pitch, yaw,
            fovy,
            j1, j2, j3, j4, j5, j6,
        ]
        render_outputs = [main_image, grid_image, xml_output]

        # ---- 事件绑定 ----

        # 切换摄像头 → 更新滑块值 + 渲染
        cam_dropdown.change(
            fn=on_camera_select,
            inputs=[cam_dropdown],
            outputs=[pos_x, pos_y, pos_z, roll, pitch, yaw, fovy, body_info],
        ).then(
            fn=render_and_update,
            inputs=all_params,
            outputs=render_outputs,
        )

        # 切换姿态预设 → 更新关节角滑块 + 渲染
        preset_dropdown.change(
            fn=on_pose_preset,
            inputs=[preset_dropdown],
            outputs=[j1, j2, j3, j4, j5, j6],
        ).then(
            fn=render_and_update,
            inputs=all_params,
            outputs=render_outputs,
        )

        # 任意参数滑块变化 → 渲染
        for slider in [pos_x, pos_y, pos_z, roll, pitch, yaw, fovy,
                       j1, j2, j3, j4, j5, j6]:
            slider.release(
                fn=render_and_update,
                inputs=all_params,
                outputs=render_outputs,
            )

        # 保存按钮 → 写入源文件
        save_btn.click(
            fn=lambda: save_cameras_to_files(_model),
            inputs=[],
            outputs=[save_status],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", share=False)
