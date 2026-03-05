"""
MuJoCo 焊缝轨迹可视化器

基于 Gradio 的交互式 GUI，可视化配置焊缝路径参数，
预览焊缝在工件上的位置，求解 IK 并导出轨迹。

用法:
    cd cs-platform
    python -m gui.seam_visualizer
"""

import sys
import io
import os
import base64
import tempfile
import threading
from pathlib import Path

import numpy as np
import mujoco
import trimesh
import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from gui.rendering import (
    create_scene_and_settle,
    set_robot_pose,
    get_camera_names,
    POSE_PRESETS,
    NUM_JOINTS,
)
from mr2010_robot import create_base_scene
from seam_generator import (
    SeamPath,
    StraightSeam,
    ArcSeam,
    ZigzagSeam,
    SCurveSeam,
    generate_trajectory,
    _solve_ik_offline,
    create_scurve_seam_on_workpiece,
    get_workpiece_surface_bounds,
)
from ik_solver import DampedLeastSquaresIK

# ========================== matplotlib 中文字体 ==========================

_zh_font = None
for font_name in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei"]:
    try:
        matches = fm.findfont(fm.FontProperties(family=font_name), fallback_to_default=False)
        if matches and os.path.exists(matches):
            _zh_font = font_name
            break
    except (ValueError, FileNotFoundError):
        continue

if _zh_font:
    plt.rcParams["font.sans-serif"] = [_zh_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

# ========================== 全局状态 ==========================

_model = None
_data = None
_settled_qpos = None
_wp_pos = None
_wp_rotmat = None
_surface_bounds = None
_last_seam = None
_last_traj = None

# 带焊缝标记球的 model/data（保留供扩展）
_marker_model = None
_marker_data = None

# GLB 临时文件目录
_glb_dir = tempfile.mkdtemp(prefix="seam_glb_")



def init_scene():
    """初始化场景"""
    global _model, _data, _settled_qpos, _wp_pos, _wp_rotmat, _surface_bounds
    print("正在初始化焊缝可视化场景...")
    _model, _data, _settled_qpos = create_scene_and_settle(with_workpiece=True)

    wp_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, "workpiece")
    _wp_pos = _data.xpos[wp_id].copy()
    _wp_rotmat = _data.xmat[wp_id].reshape(3, 3).copy()
    _surface_bounds = get_workpiece_surface_bounds(_model, _data)

    set_robot_pose(_model, _data, POSE_PRESETS["home"])
    print(f"场景就绪: 工件位置 {_wp_pos}, 表面 Z={_surface_bounds['z']:.3f}")
    print(f"  表面范围: X=[{_surface_bounds['x_min']:.3f}, {_surface_bounds['x_max']:.3f}], "
          f"Y=[{_surface_bounds['y_min']:.3f}, {_surface_bounds['y_max']:.3f}]")


# ========================== MuJoCo → GLB 导出 ==========================


def _mujoco_scene_to_glb(mj_model, mj_data, seam_pts=None,
                          start_pt=None, end_pt=None):
    """将 MuJoCo 场景导出为 GLB 文件，返回文件路径

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data (已 forward)
        seam_pts: 焊缝标记点 (Nx3 array)，显示为红色小球
        start_pt: 起点 [x,y,z]，显示为绿色球
        end_pt: 终点 [x,y,z]，显示为蓝色球
    """
    # 地板颜色覆盖：与平台/工件区分
    _COLOR_OVERRIDES = {
        "terrain": np.array([160, 190, 160, 255], dtype=np.uint8),  # 浅绿灰
    }

    scene = trimesh.Scene()

    for i in range(mj_model.ngeom):
        gt = int(mj_model.geom_type[i])
        rgba = mj_model.geom_rgba[i]
        geom_name = mj_model.geom(i).name or ""

        # 跳过不可见的碰撞体
        if rgba[3] < 0.01:
            continue

        # 决定颜色：优先使用覆盖色
        if geom_name in _COLOR_OVERRIDES:
            color4 = _COLOR_OVERRIDES[geom_name]
        else:
            color4 = np.append((rgba[:3] * 255).astype(np.uint8), 255)

        if gt == 7:  # mesh
            mid = mj_model.geom_dataid[i]
            if mid < 0:
                continue
            va = mj_model.mesh_vertadr[mid]
            vn = mj_model.mesh_vertnum[mid]
            fa = mj_model.mesh_faceadr[mid]
            fn = mj_model.mesh_facenum[mid]
            verts = mj_model.mesh_vert[va:va + vn].copy()
            faces = mj_model.mesh_face[fa:fa + fn].copy()

            # 变换到世界坐标
            gp = mj_data.geom_xpos[i]
            gm = mj_data.geom_xmat[i].reshape(3, 3)
            verts_w = (gm @ verts.T).T + gp

            m = trimesh.Trimesh(vertices=verts_w, faces=faces, process=False)
            m.visual.vertex_colors = np.tile(color4, (vn, 1)).astype(np.uint8)
            scene.add_geometry(m, geom_name=f"geom_{i}")

        elif gt == 0:  # ground plane
            verts = np.array([[-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]])
            faces = np.array([[0, 1, 2], [0, 2, 3]])
            m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            m.visual.vertex_colors = np.tile(color4, (4, 1)).astype(np.uint8)
            scene.add_geometry(m, geom_name="ground")

    # 添加焊缝标记球（红色）
    if seam_pts is not None:
        for j, p in enumerate(seam_pts):
            s = trimesh.creation.icosphere(subdivisions=1, radius=0.004)
            s.visual.vertex_colors = [255, 50, 50, 220]
            T = np.eye(4)
            T[:3, 3] = p
            s.apply_transform(T)
            scene.add_geometry(s, geom_name=f"seam_{j}")

    # 起点标记（绿色）
    if start_pt is not None:
        s = trimesh.creation.icosphere(subdivisions=2, radius=0.008)
        s.visual.vertex_colors = [0, 255, 0, 255]
        T = np.eye(4)
        T[:3, 3] = start_pt
        s.apply_transform(T)
        scene.add_geometry(s, geom_name="start_marker")

    # 终点标记（蓝色）
    if end_pt is not None:
        s = trimesh.creation.icosphere(subdivisions=2, radius=0.008)
        s.visual.vertex_colors = [0, 100, 255, 255]
        T = np.eye(4)
        T[:3, 3] = end_pt
        s.apply_transform(T)
        scene.add_geometry(s, geom_name="end_marker")

    # 导出为文件（Tab1 Model3D 用）
    path = os.path.join(_glb_dir, f"scene_{id(scene) % 100000}.glb")
    scene.export(path, file_type="glb")

    # 同时导出为 base64（Tab2 Three.js 用）
    buf = io.BytesIO()
    scene.export(buf, file_type="glb")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return path, b64


def _export_scene_glb(seam_pts=None, start_pt=None, end_pt=None):
    """使用基础场景导出 GLB，返回 (file_path, base64_str)"""
    _data.qpos[:] = _settled_qpos
    set_robot_pose(_model, _data, POSE_PRESETS["home"])
    mujoco.mj_forward(_model, _data)
    return _mujoco_scene_to_glb(_model, _data, seam_pts, start_pt, end_pt)


# ========================== 焊缝构建 ==========================


# ========================== Tab 回调 ==========================


def on_solve_ik(num_points, sim_steps):
    """求解 IK 并显示结果"""
    global _last_traj
    if _last_seam is None:
        return None, "请先在「配置焊缝」Tab 中预览焊缝路径", ""

    seam = _last_seam
    waypoints = seam.sample_points(num_points)
    orientations = seam.sample_orientations(num_points)

    joint_targets, ik_errors = _solve_ik_offline(waypoints, _model, orientations)
    ik_errors = np.array(ik_errors)

    if ik_errors.max() < 0.005:
        quality = "EXCELLENT (max err < 5mm)"
    elif ik_errors.max() < 0.02:
        quality = "GOOD (max err < 20mm)"
    elif ik_errors.max() < 0.05:
        quality = "ACCEPTABLE (max err < 50mm)"
    else:
        quality = "POOR (max err >= 50mm)"

    info = (
        f"IK 求解完成 ({num_points} 个路径点)\n"
        f"平均误差: {ik_errors.mean():.5f} m\n"
        f"最大误差: {ik_errors.max():.5f} m\n"
        f"最小误差: {ik_errors.min():.5f} m\n"
        f"质量评级: {quality}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    axes[0, 0].set_visible(False)
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
               c=np.arange(len(waypoints)), cmap="viridis", s=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Seam Path 3D")

    axes[0, 1].plot(ik_errors * 1000, "r-", linewidth=1)
    axes[0, 1].axhline(y=5, color="g", linestyle="--", alpha=0.5, label="5mm")
    axes[0, 1].set_xlabel("Waypoint")
    axes[0, 1].set_ylabel("Error (mm)")
    axes[0, 1].set_title("IK Error")
    axes[0, 1].legend()

    joint_targets_arr = np.array(joint_targets)
    for j in range(NUM_JOINTS):
        axes[1, 0].plot(
            np.degrees(joint_targets_arr[:, j]),
            label=f"J{j+1}", linewidth=1,
        )
    axes[1, 0].set_xlabel("Waypoint")
    axes[1, 0].set_ylabel("Angle (deg)")
    axes[1, 0].set_title("Joint Angles")
    axes[1, 0].legend(fontsize=7)

    if len(joint_targets_arr) > 1:
        dq = np.diff(joint_targets_arr, axis=0)
        for j in range(NUM_JOINTS):
            axes[1, 1].plot(np.degrees(dq[:, j]), label=f"J{j+1}", linewidth=1)
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("dAngle (deg/step)")
    axes[1, 1].set_title("Joint Velocity")
    axes[1, 1].legend(fontsize=7)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    plot_img = np.array(Image.open(buf))

    _data.qpos[:] = _settled_qpos
    mujoco.mj_forward(_model, _data)
    ik_solver = DampedLeastSquaresIK(_model, _data)
    traj, _ = generate_trajectory(
        seam=seam, model=_model, data=_data,
        ik_solver=ik_solver,
        num_points=num_points,
        sim_steps_per_point=sim_steps,
        restore_qpos=_settled_qpos,
    )
    _last_traj = traj

    traj_info = (
        f"\n轨迹生成完成:\n"
        f"  步数: {len(traj)}\n"
        f"  持续时间: {traj.timestamps[-1] - traj.timestamps[0]:.3f} s"
    )

    return plot_img, info + traj_info, ""


def on_render_video(cam_name, fps, duration, res_scale):
    """渲染轨迹回放视频（帧采样 + 可调分辨率）"""
    if _last_traj is None or len(_last_traj) == 0:
        return None, "请先求解 IK 生成轨迹"

    traj = _last_traj
    n_total = len(traj)
    base_w, base_h = 640, 480
    scale = res_scale / 100.0
    width = int(base_w * scale) // 2 * 2   # 保证偶数
    height = int(base_h * scale) // 2 * 2

    # 目标帧数 = 时长 × FPS，从轨迹中等间隔采样
    target_frames = int(duration * fps)
    target_frames = max(target_frames, 2)
    if target_frames >= n_total:
        indices = np.arange(n_total)
    else:
        indices = np.linspace(0, n_total - 1, target_frames, dtype=int)

    renderer = mujoco.Renderer(_model, height=height, width=width)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False, prefix="traj_video_", dir=_glb_dir
    )
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, int(fps), (width, height))

    # 预设工件稳定状态，只更新关节角
    _data.qpos[:] = _settled_qpos
    mujoco.mj_forward(_model, _data)

    for i in indices:
        qpos = traj.joint_positions[i]
        _data.qpos[:NUM_JOINTS] = qpos
        _data.ctrl[:NUM_JOINTS] = qpos
        mujoco.mj_forward(_model, _data)

        renderer.update_scene(_data, camera=cam_name)
        img = renderer.render()
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer.release()
    renderer.close()

    # 恢复初始姿态
    _data.qpos[:] = _settled_qpos
    mujoco.mj_forward(_model, _data)

    actual_frames = len(indices)
    return tmp.name, (
        f"视频渲染完成: {actual_frames} 帧 (从 {n_total} 帧采样), "
        f"{int(fps)} FPS, 时长 {actual_frames/fps:.1f}s, "
        f"分辨率 {width}×{height}"
    )


def on_export_npz():
    """导出轨迹为 NPZ 文件"""
    if _last_traj is None or len(_last_traj) == 0:
        return None, "请先求解 IK 生成轨迹"

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False, prefix="traj_")
    _last_traj.save_npz(tmp.name)
    return tmp.name, f"轨迹已导出: {tmp.name} ({len(_last_traj)} 步)"


# ========================== Tab 2: 3D 选点 ==========================

# 容器 HTML（纯 HTML，不含 script）
_PICK_VIEWER_DIV = (
    '<div id="pick3d_container"'
    ' style="width:100%;height:600px;border:1px solid #444;'
    'border-radius:8px;overflow:hidden;position:relative;">'
    '<div id="pick3d_info"'
    ' style="position:absolute;top:8px;left:8px;color:#fff;font-size:13px;'
    'z-index:10;background:rgba(0,0,0,0.5);padding:4px 10px;'
    'border-radius:4px;pointer-events:none;">'
    'Three.js 加载中...</div></div>'
)

# Three.js 查看器 JS（通过 js_on_load 执行，保证真正运行）
_PICK_VIEWER_JS_TEMPLATE = r'''
var SURFACE_Z = __SURFACE_Z__;
var THREE, OrbitControls, GLTFLoader;
var scene, camera, renderer, controls, loader, currentModel;
var startMarker = null, endMarker = null, midMarker = null;

window._pickStart = null;
window._pickEnd = null;
window._pickMid = null;
if(!window._pendingGLB) window._pendingGLB = null;

function mkSphere(color){
  var g = new THREE.SphereGeometry(0.008,16,16);
  var m = new THREE.MeshPhongMaterial({color:color, emissive:color, emissiveIntensity:0.3});
  return new THREE.Mesh(g, m);
}

function updateStatus(){
  var L = [];
  if(window._pickStart) L.push('\u8d77\u70b9: ('+window._pickStart[0].toFixed(4)+', '+window._pickStart[1].toFixed(4)+')');
  if(window._pickMid) L.push('\u4e2d\u70b9: ('+window._pickMid[0].toFixed(4)+', '+window._pickMid[1].toFixed(4)+')');
  if(window._pickEnd) L.push('\u7ec8\u70b9: ('+window._pickEnd[0].toFixed(4)+', '+window._pickEnd[1].toFixed(4)+')');
  if(L.length===0) L.push('Ctrl+\u70b9\u51fb\u8bbe\u8d77\u70b9');
  var el = document.getElementById('pick3d_info');
  if(el) el.textContent = L.join(' | ');
  var sb = document.querySelector('#click_status_box textarea');
  if(sb){sb.value=L.join('\n');sb.dispatchEvent(new Event('input',{bubbles:true}));}
}

function loadGLB(b64){
  if(!loader){window._pendingGLB=b64;return;}
  if(currentModel){scene.remove(currentModel);currentModel=null;}
  try {
    var bin = atob(b64);
    var buf = new ArrayBuffer(bin.length);
    var view = new Uint8Array(buf);
    for(var i=0;i<bin.length;i++) view[i]=bin.charCodeAt(i);
    loader.parse(buf, '', function(gltf){
      currentModel=gltf.scene;
      scene.add(currentModel);
      console.log('GLB loaded, children:', currentModel.children.length);
    }, function(err){
      console.error('GLB parse error:', err);
    });
  } catch(e) {
    console.error('GLB decode error:', e);
  }
}

window.loadPickerGLB = loadGLB;
window._clearMarkers = function(){
  window._pickStart=null; window._pickEnd=null; window._pickMid=null;
  if(scene && startMarker){scene.remove(startMarker);startMarker=null;}
  if(scene && endMarker){scene.remove(endMarker);endMarker=null;}
  if(scene && midMarker){scene.remove(midMarker);midMarker=null;}
  updateStatus();
};

function initViewer(){
  var c = document.getElementById('pick3d_container');
  if(!c) return;
  var rect = c.getBoundingClientRect();
  var W = rect.width || 800, H = rect.height || 600;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  camera = new THREE.PerspectiveCamera(45, W/H, 0.01, 100);
  camera.up.set(0,0,1);
  camera.position.set(2.5, -1.5, 2.0);

  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  c.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(1.5, 0, 0.4);
  controls.enableDamping = true;
  controls.update();

  scene.add(new THREE.AmbientLight(0xffffff, 1.0));
  var d1 = new THREE.DirectionalLight(0xffffff, 1.2);
  d1.position.set(2, -2, 5); scene.add(d1);
  var d2 = new THREE.DirectionalLight(0xffffff, 0.8);
  d2.position.set(-2, 2, 3); scene.add(d2);
  var d3 = new THREE.DirectionalLight(0xffffff, 0.5);
  d3.position.set(0, 0, 6); scene.add(d3);

  loader = new GLTFLoader();
  var raycaster = new THREE.Raycaster();
  var pickPlane = new THREE.Plane(new THREE.Vector3(0,0,1), -SURFACE_Z);

  function pickPoint(e){
    if(!e.ctrlKey && !e.shiftKey && !e.altKey) return;
    e.preventDefault(); e.stopPropagation();
    var rc = renderer.domElement.getBoundingClientRect();
    var mouse = new THREE.Vector2(
      ((e.clientX-rc.left)/rc.width)*2-1,
      -((e.clientY-rc.top)/rc.height)*2+1
    );
    raycaster.setFromCamera(mouse, camera);

    var pt = null;
    if(currentModel){
      var hits = raycaster.intersectObject(currentModel, true);
      if(hits.length>0) pt = hits[0].point.clone();
    }
    if(!pt){
      pt = new THREE.Vector3();
      if(!raycaster.ray.intersectPlane(pickPlane, pt)) return;
    }

    if(e.ctrlKey){
      window._pickStart = [pt.x, pt.y];
      if(startMarker) scene.remove(startMarker);
      startMarker = mkSphere(0x00ff00);
      startMarker.position.set(pt.x, pt.y, SURFACE_Z+0.005);
      scene.add(startMarker);
    } else if(e.altKey){
      window._pickMid = [pt.x, pt.y];
      if(midMarker) scene.remove(midMarker);
      midMarker = mkSphere(0xffcc00);
      midMarker.position.set(pt.x, pt.y, SURFACE_Z+0.005);
      scene.add(midMarker);
    } else if(e.shiftKey){
      window._pickEnd = [pt.x, pt.y];
      if(endMarker) scene.remove(endMarker);
      endMarker = mkSphere(0x0066ff);
      endMarker.position.set(pt.x, pt.y, SURFACE_Z+0.005);
      scene.add(endMarker);
    }
    updateStatus();
  }
  renderer.domElement.addEventListener('click', pickPoint);

  (function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);})();

  new ResizeObserver(function(){
    var rr = c.getBoundingClientRect();
    if(rr.width<10) return;
    camera.aspect = rr.width/rr.height;
    camera.updateProjectionMatrix();
    renderer.setSize(rr.width, rr.height);
  }).observe(c);

  if(window._pendingGLB){loadGLB(window._pendingGLB);window._pendingGLB=null;}
  var info = document.getElementById('pick3d_info');
  if(info) info.textContent = 'Ctrl=\u8d77\u70b9 | Alt=\u4e2d\u70b9(\u5706\u5f27) | Shift=\u7ec8\u70b9';
}

var BASE = 'https://esm.sh/three@0.170.0';
Promise.all([
  import(BASE),
  import(BASE+'/examples/jsm/controls/OrbitControls.js'),
  import(BASE+'/examples/jsm/loaders/GLTFLoader.js')
]).then(function(mods){
  THREE = mods[0];
  OrbitControls = mods[1].OrbitControls;
  GLTFLoader = mods[2].GLTFLoader;
  initViewer();
}).catch(function(err){
  console.error('Three.js load failed:', err);
  var info = document.getElementById('pick3d_info');
  if(info) info.textContent = 'Three.js \u52a0\u8f7d\u5931\u8d25: ' + err.message;
});
'''

# JS：生成焊缝前，从 JS 变量注入坐标到隐藏组件（含中间点）
_PICK_COORDS_JS = """(seam_type, num_pts, sx, sy, mx, my, ex, ey) => {
    const s = window._pickStart;
    const m = window._pickMid;
    const e = window._pickEnd;
    return [seam_type, num_pts,
        s ? s[0] : null, s ? s[1] : null,
        m ? m[0] : null, m ? m[1] : null,
        e ? e[0] : null, e ? e[1] : null];
}"""

# JS：清除选点后，清除 JS 侧标记
_CLEAR_JS = """() => { if(window._clearMarkers) window._clearMarkers(); return []; }"""

# JS：加载新 GLB 到 Three.js viewer (base64 data)
_RELOAD_GLB_JS = """(b64) => {
    if(!b64) return;
    if(window.loadPickerGLB) { window.loadPickerGLB(b64); }
    else { window._pendingGLB = b64; }
}"""


def _circle_through_3_points(p1, p2, p3):
    """通过三点求圆心和半径 (2D)

    Returns (center_xy, radius) or None if collinear.
    """
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-10:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / D
    center = np.array([ux, uy])
    radius = np.linalg.norm(center - np.array([ax, ay]))
    return center, radius


class ClickSCurveSeam(SeamPath):
    """由起点和终点生成的 S 型焊缝 — 两段反向半圆弧

    几何结构（在 start→end 的局部坐标下）：
      - 第一段半圆弧: 圆心在 1/4 处, 从 start 到 midpoint, 凸出方向由 mid 点决定
      - 第二段半圆弧: 圆心在 3/4 处, 从 midpoint 到 end, 向反方向凸出
      - 每段半径 = |start - end| / 4

    Args:
        start: 起点 [x, y, z]
        end: 终点 [x, y, z]
        mid: 可选中间点 [x, y], 用于确定第一段弧线的凸出朝向
             如果 mid 在 start→end 连线的左侧，第一段向左凸出；反之向右
             如果不提供，默认第一段向 +perp（左侧）凸出
    """

    def __init__(self, start: np.ndarray, end: np.ndarray,
                 mid: np.ndarray = None):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.mid = np.asarray(mid, dtype=float) if mid is not None else None

    def sample_points(self, num_points: int) -> np.ndarray:
        s = self.start
        e = self.end
        d = e[:2] - s[:2]
        L = np.linalg.norm(d)
        if L < 1e-6:
            return np.tile(s, (num_points, 1))

        along = d / L
        perp = np.array([-along[1], along[0]])

        # 用 mid 点决定第一段弧线朝向
        # cross > 0 → mid 在 start→end 左侧（+perp 侧）
        # cross < 0 → mid 在右侧（-perp 侧）
        if self.mid is not None:
            v = self.mid[:2] - s[:2]
            cross = along[0] * v[1] - along[1] * v[0]
            if cross < 0:
                perp = -perp  # 翻转，让第一段弧线向 mid 所在侧凸出

        R = L / 4.0

        t = np.linspace(0, 1, num_points)
        pts = np.zeros((num_points, 3))

        # Z: 线性插值
        pts[:, 2] = s[2] + t * (e[2] - s[2])

        # 第一段半圆弧 (t ∈ [0, 0.5]):
        #   局部坐标圆心 = (L/4, 0), θ: π → 0, 经过 (L/4, R) 即 +perp 凸出
        mask1 = t <= 0.5
        theta1 = np.pi * (1.0 - 2.0 * t[mask1])
        lx1 = L / 4.0 + R * np.cos(theta1)
        ly1 = R * np.sin(theta1)
        pts[mask1, 0] = s[0] + lx1 * along[0] + ly1 * perp[0]
        pts[mask1, 1] = s[1] + lx1 * along[1] + ly1 * perp[1]

        # 第二段半圆弧 (t ∈ (0.5, 1.0]):
        #   局部坐标圆心 = (3L/4, 0), θ: π → 2π, 经过 (3L/4, -R) 即 -perp 凸出
        mask2 = t > 0.5
        theta2 = np.pi + np.pi * (2.0 * t[mask2] - 1.0)
        lx2 = 3.0 * L / 4.0 + R * np.cos(theta2)
        ly2 = R * np.sin(theta2)
        pts[mask2, 0] = s[0] + lx2 * along[0] + ly2 * perp[0]
        pts[mask2, 1] = s[1] + lx2 * along[1] + ly2 * perp[1]

        return pts


def on_generate_click_seam(seam_type_click, num_points_click,
                           sx, sy, mx, my, ex, ey):
    """根据 3D 选点坐标生成焊缝，返回 (glb_b64, info_text)"""
    global _last_seam

    surface_z = _surface_bounds["z"]

    # S曲线: 需要起点和终点，可选中间点确定弯曲方向
    if seam_type_click == "S曲线":
        if sx is None or sy is None or ex is None or ey is None:
            return "", "S曲线需要选取起点和终点\nCtrl+点击=起点, Shift+点击=终点\n可选: Alt+点击=中间点(确定弯曲方向)"
        start = np.array([float(sx), float(sy), surface_z])
        end = np.array([float(ex), float(ey), surface_z])
        dist = np.linalg.norm(end[:2] - start[:2])
        if dist < 0.005:
            return "", "起点和终点距离太近，无法生成 S 曲线"
        mid_2d = np.array([float(mx), float(my)]) if (mx is not None and my is not None) else None
        seam = ClickSCurveSeam(start, end, mid=mid_2d)
        _last_seam = seam
        num_pts = int(num_points_click)
        pts = seam.sample_points(num_pts)
        path_len = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        info = (
            f"焊缝已生成!\n"
            f"类型: S曲线 (两段反向半圆弧)\n"
            f"起点: ({start[0]:.4f}, {start[1]:.4f}, {start[2]:.4f})\n"
            f"终点: ({end[0]:.4f}, {end[1]:.4f}, {end[2]:.4f})\n"
            f"半圆弧半径: {dist/4:.4f} m\n"
            f"路径长度: {path_len:.4f} m | 采样点数: {num_pts}\n"
            f"\n焊缝已设置，可切换到「IK 预览」Tab 求解轨迹"
        )
        seam_pts = seam.sample_points(40)
        _, glb_b64 = _export_scene_glb(
            seam_pts=seam_pts,
            start_pt=start.tolist(),
            end_pt=end.tolist(),
        )
        return glb_b64, info

    # 其他类型需要起点和终点
    if sx is None or sy is None or ex is None or ey is None:
        return "", "请先 Ctrl+点击设置起点, Shift+点击设置终点"

    start = np.array([float(sx), float(sy), surface_z])
    end = np.array([float(ex), float(ey), surface_z])

    if seam_type_click == "直线":
        seam = StraightSeam(start, end)
    elif seam_type_click == "圆弧":
        if mx is not None and my is not None:
            # 三点定圆弧: 起点 → 中间点 → 终点
            mid_2d = np.array([float(mx), float(my)])
            result = _circle_through_3_points(start[:2], mid_2d, end[:2])
            if result is None:
                return "", "三点共线，无法生成圆弧，请调整中间点位置"
            center_xy, radius = result
            start_angle = np.arctan2(start[1] - center_xy[1], start[0] - center_xy[0])
            mid_angle = np.arctan2(mid_2d[1] - center_xy[1], mid_2d[0] - center_xy[0])
            end_angle = np.arctan2(end[1] - center_xy[1], end[0] - center_xy[0])

            # 用 CCW 角距判断绕行方向
            # d_sm = 从 start 逆时针到 mid 的角距
            # d_se = 从 start 逆时针到 end 的角距
            d_sm = (mid_angle - start_angle) % (2 * np.pi)
            d_se = (end_angle - start_angle) % (2 * np.pi)

            if d_sm < d_se:
                # 逆时针 start→end 经过 mid → 用 CCW 方向
                final_end = start_angle + d_se
            else:
                # 逆时针 start→end 不经过 mid → 用 CW 方向
                final_end = start_angle - (2 * np.pi - d_se)

            seam = ArcSeam(center_xy, radius, start_angle, final_end, surface_z)
        else:
            # 无中间点：退化为起点→终点的半圆弧
            center_xy = (start[:2] + end[:2]) / 2
            radius = np.linalg.norm(end[:2] - start[:2]) / 2
            if radius < 0.001:
                return "", "起点和终点太近，无法生成圆弧"
            start_angle = np.arctan2(start[1] - center_xy[1], start[0] - center_xy[0])
            end_angle = np.arctan2(end[1] - center_xy[1], end[0] - center_xy[0])
            seam = ArcSeam(center_xy, radius, start_angle, end_angle, surface_z)
    elif seam_type_click == "锯齿":
        seam = ZigzagSeam(start, end, amplitude=0.01, frequency=10)
    else:
        return "", "未知焊缝类型"

    _last_seam = seam
    num_pts = int(num_points_click)
    pts = seam.sample_points(num_pts)
    path_len = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

    mid_info = ""
    if seam_type_click == "圆弧" and mx is not None and my is not None:
        mid_info = f"中间点: ({float(mx):.4f}, {float(my):.4f}, {surface_z:.4f})\n"

    info = (
        f"焊缝已生成!\n"
        f"类型: {seam_type_click}\n"
        f"起点: ({start[0]:.4f}, {start[1]:.4f}, {start[2]:.4f})\n"
        f"{mid_info}"
        f"终点: ({end[0]:.4f}, {end[1]:.4f}, {end[2]:.4f})\n"
        f"路径长度: {path_len:.4f} m | 采样点数: {num_pts}\n"
        f"\n焊缝已设置，可切换到「IK 预览」Tab 求解轨迹"
    )

    seam_pts = seam.sample_points(40)
    _, glb_b64 = _export_scene_glb(
        seam_pts=seam_pts,
        start_pt=start.tolist(),
        end_pt=end.tolist(),
    )

    return glb_b64, info


def on_clear_points():
    """清除选点并重新导出基础场景"""
    _, glb_b64 = _export_scene_glb()
    return glb_b64, "已清除选点\nCtrl+点击=设起点, Shift+点击=设终点"


# ========================== GUI 布局 ==========================


def build_app():
    init_scene()

    bnd = _surface_bounds
    default_z = bnd["z"] if bnd else 0.55

    # 初始 GLB (base64 for Three.js)
    _data.qpos[:] = _settled_qpos
    set_robot_pose(_model, _data, POSE_PRESETS["home"])
    mujoco.mj_forward(_model, _data)
    _, init_click_b64 = _export_scene_glb()

    # 构造选点 JS（注入 surface_z）
    pick_viewer_js = _PICK_VIEWER_JS_TEMPLATE.replace(
        '__SURFACE_Z__', str(round(default_z, 4))
    )

    with gr.Blocks(
        title="MuJoCo 焊缝轨迹可视化器",
    ) as app:
        gr.Markdown("# MuJoCo 焊缝轨迹可视化器")
        gr.Markdown(
            "在 3D 视图中选取路径点 → IK 求解 → 导出轨迹"
        )

        with gr.Tabs():
            # ==================== Tab 1: 配置焊缝 ====================
            with gr.Tab("配置焊缝"):
                gr.Markdown(
                    "### 在 3D 视图中直接选取路径点\n"
                    "**Ctrl+左键** = 起点（绿色） | "
                    "**Alt+左键** = 中间点（黄色，圆弧/S曲线用） | "
                    "**Shift+左键** = 终点（蓝色）\n\n"
                    "普通左键拖拽 = 旋转视角 | 滚轮 = 缩放 | 右键拖拽 = 平移"
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        # 自定义 Three.js 3D 查看器（容器 + js_on_load 执行脚本）
                        gr.HTML(
                            value=_PICK_VIEWER_DIV,
                            js_on_load=pick_viewer_js,
                        )
                        # 隐藏的 Textbox 用于接收 GLB base64 并触发 JS 加载
                        pick_glb_out = gr.Textbox(visible=False)

                    with gr.Column(scale=1):
                        click_status = gr.Textbox(
                            value="Ctrl+点击=设起点\nAlt+点击=设中间点(圆弧)\nShift+点击=设终点",
                            label="选点状态", lines=5, interactive=False,
                            elem_id="click_status_box",
                        )

                        # 隐藏的 Number 组件：JS 写入坐标 → Python 读取
                        click_sx = gr.Number(visible=False, elem_id="pick_sx")
                        click_sy = gr.Number(visible=False, elem_id="pick_sy")
                        click_mx = gr.Number(visible=False, elem_id="pick_mx")
                        click_my = gr.Number(visible=False, elem_id="pick_my")
                        click_ex = gr.Number(visible=False, elem_id="pick_ex")
                        click_ey = gr.Number(visible=False, elem_id="pick_ey")

                        with gr.Row():
                            seam_type_click = gr.Radio(
                                ["S曲线", "直线", "圆弧", "锯齿"],
                                value="直线", label="焊缝类型",
                            )
                            num_points_click = gr.Slider(
                                20, 200, value=80, step=5,
                                label="采样点数",
                            )

                        gr.Markdown(
                            "*S曲线: 自动匹配工件凸起，无需选点*\n"
                            "*圆弧: 用 Alt+点击 设中间点确定弧形*"
                        )

                        generate_click_btn = gr.Button(
                            "生成焊缝", variant="primary",
                        )
                        clear_pts_btn = gr.Button(
                            "清除选点", variant="secondary",
                        )

                # 生成焊缝：先通过 JS 从 window._pickStart/_pickMid/_pickEnd 读取坐标
                generate_click_btn.click(
                    fn=on_generate_click_seam,
                    inputs=[seam_type_click, num_points_click,
                            click_sx, click_sy, click_mx, click_my,
                            click_ex, click_ey],
                    outputs=[pick_glb_out, click_status],
                    js=_PICK_COORDS_JS,
                ).then(
                    fn=None, inputs=[pick_glb_out], js=_RELOAD_GLB_JS,
                )

                # 清除选点
                clear_pts_btn.click(
                    fn=on_clear_points,
                    outputs=[pick_glb_out, click_status],
                    js=_CLEAR_JS,
                ).then(
                    fn=None, inputs=[pick_glb_out], js=_RELOAD_GLB_JS,
                )

                # 初始加载 GLB 到 Three.js viewer
                app.load(
                    fn=lambda: init_click_b64,
                    outputs=[pick_glb_out],
                ).then(
                    fn=None, inputs=[pick_glb_out], js=_RELOAD_GLB_JS,
                )

            # ==================== Tab 3: IK 预览 ====================
            with gr.Tab("IK 预览"):
                with gr.Row():
                    ik_num_points = gr.Slider(
                        20, 200, value=80, step=5,
                        label="路径点数",
                    )
                    ik_sim_steps = gr.Slider(
                        10, 100, value=50, step=5,
                        label="每点仿真步数 (sim_steps_per_point)",
                    )

                solve_btn = gr.Button("求解 IK 并生成轨迹", variant="primary")

                ik_info = gr.Textbox(label="IK 结果", lines=8, interactive=False)
                ik_plot = gr.Image(label="IK 分析图表", height=600)
                viewer_status = gr.Textbox(
                    label="状态", interactive=False, visible=True,
                )

                solve_btn.click(
                    fn=on_solve_ik,
                    inputs=[ik_num_points, ik_sim_steps],
                    outputs=[ik_plot, ik_info, viewer_status],
                )

                gr.Markdown("---")
                gr.Markdown("### 轨迹回放视频")
                with gr.Row():
                    video_cam = gr.Dropdown(
                        choices=get_camera_names(_model),
                        value=get_camera_names(_model)[0] if _model.ncam > 0 else "",
                        label="摄像头",
                    )
                    video_fps = gr.Slider(
                        5, 60, value=25, step=1,
                        label="帧率 (FPS)",
                    )
                    video_duration = gr.Slider(
                        1, 30, value=5, step=1,
                        label="视频时长 (秒)",
                    )
                    video_res = gr.Slider(
                        25, 100, value=50, step=25,
                        label="分辨率 (%)",
                    )
                render_video_btn = gr.Button(
                    "渲染回放视频", variant="secondary",
                )

                video_status = gr.Textbox(
                    label="视频状态", interactive=False,
                )
                video_player = gr.Video(label="轨迹回放")

                render_video_btn.click(
                    fn=on_render_video,
                    inputs=[video_cam, video_fps, video_duration, video_res],
                    outputs=[video_player, video_status],
                )

            # ==================== Tab 3: 导出 ====================
            with gr.Tab("导出"):
                gr.Markdown("### 导出轨迹数据")
                gr.Markdown("先在 **IK 预览** Tab 中求解 IK 生成轨迹，然后在此导出。")

                export_npz_btn = gr.Button("导出 NPZ 文件", variant="primary")
                export_file = gr.File(label="下载轨迹文件")
                export_info = gr.Textbox(label="导出状态", interactive=False)

                export_npz_btn.click(
                    fn=on_export_npz,
                    outputs=[export_file, export_info],
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", share=False, allowed_paths=[_glb_dir])
