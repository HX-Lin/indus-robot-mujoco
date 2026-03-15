# MR2010 工业机器人焊接仿真平台

基于 MuJoCo 的 MR2010 六轴工业机器人焊接仿真环境，支持模仿学习 (IL) 和强化学习 (RL) 两条训练路径。

模仿学习核心策略为 **VITA (Vision-to-Action Flow Matching)**，配合自研 **FASA (Flow-Aware Spatial Attention)** 自监督空间注意力模块，以及 Grounding DINO 视觉基础模型离线蒸馏的 ResNet-18 视觉编码器。

## 项目结构

```
cs-platform/
├── mr2010_robot.py          # 核心模块：机器人模型加载、场景创建
├── ik_solver.py             # 阻尼最小二乘 IK 求解器
├── seam_generator.py        # 焊缝路径定义 (直线/圆弧/锯齿/S曲线) + 轨迹生成
├── il_env.py                # 模仿学习环境 + Trajectory 数据类
├── lerobot_collector.py     # LeRobot v3.0 数据集采集器
├── generate_dataset.py      # 批量生成演示轨迹数据集
├── pretrain_resnet.py       # ResNet 预蒸馏 (Grounding DINO → ResNet18)
├── extract_gdino_heatmaps.py # 离线提取 GDINO 空间热图 + 语义特征
├── train.py                 # 模仿学习训练 (ACT / VITA 策略)
├── train_rl.py              # 强化学习训练 (PPO via mjlab)
├── evaluate.py              # 闭环评估 + 视频录制
├── visualize_collect.py     # 可视化采集过程
├── visualize_dataset.py     # 可视化 LeRobot 数据集
├── visualize_episode.py     # 可视化单条轨迹
├── debug_cameras.py         # 摄像头调试 CLI 工具
├── policies/                # 策略模块
│   ├── __init__.py          #   策略注册表
│   ├── base_policy.py       #   策略基类
│   ├── act_policy.py        #   ACT 策略 (Transformer + VAE)
│   └── vita_policy.py       #   VITA 策略 (Flow Matching + FASA)
├── gui/                     # 可视化 GUI 工具 (Gradio)
│   ├── rendering.py         #   共享渲染工具
│   ├── camera_tuner.py      #   摄像头位置调节器
│   └── seam_visualizer.py   #   焊缝轨迹可视化器
├── rl/                      # 强化学习 MDP 定义
│   ├── welding_env_cfg.py   #   RL 环境配置
│   └── mdp/                 #   观测/奖励/终止/指令函数
├── assets/                  # 模型资源
│   ├── MR2010.xml           #   机器人 MJCF 模型
│   ├── *.STL                #   机器人/平台/工件 mesh
│   ├── platform_convex/     #   平台凸分解碰撞体
│   └── workpiece_convex/    #   工件凸分解碰撞体
└── datasets_lerobot/        # 生成的数据集 (LeRobot v3.0 格式)

third_parties/
├── lerobot/                 # HuggingFace LeRobot (v0.4.4)
├── mjlab/                   # MuJoCo-Warp GPU 并行仿真框架
└── joycon-robotics/         # JoyCon 遥操作
```

## 环境安装

### 方式一：使用项目自带 venv（推荐用于 IL + RL 完整流程）

```bash
# 项目已包含预配置的虚拟环境
source cs_env/bin/activate

# 安装 mjlab (RL 训练需要)
cd third_parties/mjlab && pip install -e . && cd ../..

# 安装 lerobot (IL 训练需要)
cd third_parties/lerobot && pip install -e . && cd ../..
```

**核心依赖：**
- Python >= 3.10
- mujoco >= 3.4.0
- torch >= 2.7.0
- lerobot (v0.4.4, 用于 IL)
- torchcfm, timm, einops (VITA 策略额外依赖)
- rsl-rl-lib == 3.1.0 (用于 RL)
- mujoco-warp + warp-lang (用于 GPU 并行 RL)

### 方式二：使用 conda 环境（仅 IL 流程 + GUI 工具）

```bash
conda activate lerobot2   # 需要 mujoco >= 3.5.0
pip install gradio         # GUI 工具需要
```

## 快速开始

### 1. 验证场景

```bash
cd cs-platform

# 启动 MuJoCo 交互式 Viewer 查看完整场景
python mr2010_robot.py full
```

### 2. 生成数据集（模仿学习演示数据）

```bash
# 生成 160 条 S 曲线焊接轨迹（含摄像头图像）
python generate_dataset.py --num_trajs 160 --output_dir datasets_lerobot

# 快速生成（不含图像，速度更快）
python generate_dataset.py --num_trajs 160 --no_images
```

数据集特征：
| 特征 | 维度 | 说明 |
|------|------|------|
| `observation.state` | (6,) | 关节角度 |
| `observation.images.overhead` | (3,480,640) | 俯视摄像头 RGB |
| `observation.images.ee` | (3,480,640) | 末端摄像头 RGB |
| `action` | (6,) | 关节位置目标 |

### 3. 可视化采集过程

```bash
# 在 MuJoCo Viewer 中实时观看轨迹生成
python visualize_collect.py --num_trajs 3 --speed 0.5
```

### 4. 训练

#### 模仿学习 — VITA (推荐)

VITA 采用两阶段训练流程：

**阶段一：ResNet 预蒸馏**

从 Grounding DINO 蒸馏空间注意力和语义特征到 ResNet-18 backbone：

```bash
# 1. 离线提取 GDINO 特征 (仅需一次)
python extract_gdino_heatmaps.py

# 2. 预蒸馏训练
python pretrain_resnet.py --steps 40000 --batch_size 32
```

**阶段二：VITA 策略训练**

```bash
# 基础 VITA 训练
python train.py --policy vita \
  --pretrained_resnet outputs/pretrain_resnet/best \
  --steps 100000 --chunk_size 64 --batch_size 32 --num_workers 4 \
  --output_dir outputs/vita_v3

# VITA + FASA 空间注意力 (需要 obs_horizon >= 2)
python train.py --policy vita \
  --pretrained_resnet outputs/pretrain_resnet/best \
  --use_spatial_attn --obs_horizon 2 \
  --steps 100000 --chunk_size 64 --batch_size 32 --num_workers 4 \
  --output_dir outputs/vita_v4

# 从断点恢复训练
python train.py --resume outputs/vita_v3/last
```

VITA 模型架构：ResNet18 视觉编码器 (蒸馏增强) + OT-CFM Flow Matching (512d, 4层) + Action Autoencoder

FASA 在 VITA 基础上增加自监督空间注意力，利用帧间特征变化自动发现交互区域：
- 动态区域加权池化 — 聚焦变化(交互)区域
- 静态区域加权池化 — 保留稳定参考上下文
- 变化质心 + 方向编码 — 交互在哪 + 变化趋势
- Spatial Softmax 坐标 — 各通道空间关键点

#### 模仿学习 — ACT

```bash
# 训练 ACT 策略
python train.py --policy act --steps 100000 --batch_size 8 --chunk_size 50

# 从断点恢复训练
python train.py --resume outputs/act/last
```

ACT 模型架构：ResNet18 视觉编码器 + Transformer (512d, 8头, 4层编码器) + VAE (latent_dim=32)

#### 强化学习 (PPO)

```bash
# PPO 训练（2048 并行环境，GPU 加速）
python train_rl.py --num_envs 2048 --max_iterations 500

# 更多并行环境
python train_rl.py --num_envs 4096 --max_iterations 1000
```

RL 观测空间 (29维)：关节角/速度 + 末端位置 + 到目标向量 + 焊缝进度 + 朝向误差 + 上一步动作

RL 奖励设计：轨迹跟踪 (+2.0) + 进度推进 (+5.0) + 朝向正确 (+0.5) + 完成奖励 (+10.0) - 动作平滑惩罚 - 关节速度/力矩惩罚

### 5. 评估

```bash
# 闭环评估并录制视频
python evaluate.py --policy vita --checkpoint outputs/vita_v3/best --num_episodes 10

# 带交互式 Viewer
python evaluate.py --policy vita --checkpoint outputs/vita_v3/best --render
```

评估指标：
| 指标 | 说明 |
|------|------|
| 接近成功率 (%) | 末端是否到达焊缝起点 5mm 范围内 |
| 平均/中位数/最大跟踪误差 (mm) | 末端到焊缝最近点的欧氏距离 |
| 多级覆盖率 (%) | <0.5mm / <1mm / <2mm / <5mm 阈值下的覆盖比例 |
| 等级分布 | S/A/B/C/D/E/FAIL 根据平均跟踪误差分级 |

## GUI 可视化工具

### 摄像头位置调节器

交互式调整摄像头 pos / euler / fovy，实时预览渲染效果，导出 XML 片段。

```bash
cd cs-platform
python -m gui.camera_tuner
# 浏览器打开 http://localhost:7860
```

功能：
- 滑块调整摄像头位置 (XYZ) 和朝向 (Roll/Pitch/Yaw 欧拉角)
- 切换机器人姿态预设查看不同构型下的摄像头效果
- 自动生成可直接粘贴到 `MR2010.xml` 的 XML 片段

### 焊缝轨迹可视化器

可视化配置焊缝路径 → IK 求解预览 → 导出轨迹。

```bash
cd cs-platform
python -m gui.seam_visualizer
# 浏览器打开 http://localhost:7860
```

功能：
- 4 种焊缝类型可选：S曲线、直线、圆弧、锯齿
- 俯视图实时预览焊缝路径在工件上的位置
- 一键 IK 求解，查看误差分析图表
- 启动 3D Viewer 播放轨迹动画
- 导出 NPZ 轨迹文件或生成独立 Python 脚本

## 关键模块说明

### VITA 策略 (`policies/vita_policy.py`)

基于 Flow Matching 的视觉运动策略，核心组件：
- **OT-CFM Flow Matching**: 从视觉 latent 直接流向动作 latent，天然支持多模态
- **Action Autoencoder**: 1D-CNN 编码器 + MLP 解码器，压缩 action chunk 到 latent space
- **ResNet Observer**: 冻结的蒸馏 ResNet-18 backbone + 可选 FASA 空间注意力
- **FASA (Flow-Aware Spatial Attention)**: 帧间特征变化 → 自监督注意力图 → 加权池化 + 坐标编码
- **Temporal Ensemble**: ACT 风格的指数加权滑动平均，平滑连续 action chunk

### FASA 数据流

```
feat_map_t (B, 512, 7, 10)  +  feat_map_{t-1}
    │
    ├─ |feat_t - feat_{t-1}| → change_refine(Conv) → change_attn
    │
    ├─ 动态池化: feat_t * change_attn → sum → (B, 512)
    ├─ 静态池化: feat_t * (1 - change_attn) → sum → (B, 512)
    ├─ 变化坐标: weighted centroid + direction → (B, 4)
    └─ Spatial Softmax → coord_proj → (B, 32)

    ─── per camera: concat → (B, 1060) ───
```

### 焊缝路径 (`seam_generator.py`)

| 类型 | 类名 | 参数 |
|------|------|------|
| 直线 | `StraightSeam` | start, end (3D坐标) |
| 圆弧 | `ArcSeam` | center, radius, 角度范围, height |
| 锯齿 | `ZigzagSeam` | start, end, amplitude, frequency |
| S曲线 | `SCurveSeam` | 自动匹配工件 STL 凸起中心线 |

### IK 求解 (`ik_solver.py`)

阻尼最小二乘法 (Levenberg-Marquardt)，支持位置+姿态 6-DOF 约束：

```
dq = J^T (J J^T + λ²I)^{-1} × error
```

自适应阻尼，最大步长 0.5 rad/iter，关节限位裁剪。

### 场景创建 (`mr2010_robot.py`)

```python
from mr2010_robot import create_base_scene

spec = create_base_scene(
    with_actuators=True,     # PD 位置控制器
    with_platform=True,      # 工作台 (含凸分解碰撞体)
    with_workpiece=True,     # 工件 (freejoint, 自然落到平台)
    randomize_workpiece=True # 随机位置和yaw角
)
model = spec.compile()
```

## 许可证

本项目仅供研究和教育用途。
