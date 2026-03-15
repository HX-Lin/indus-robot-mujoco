# 论文框架文档

## 题目

**Temporal-Aware Flow Matching for Visuomotor Robot Policies via Self-Predictive Dynamics and Flow-Aware Spatial Attention**

（基于自预测动力学与流感知空间注意力的时间感知流匹配视觉运动机器人策略）

---

## 核心问题

标准的 flow matching 策略（如 VITA）存在两个根本性局限：

1. **时间盲目**——每步独立决策，无记忆，无法感知"当前处于任务的哪个阶段"
2. **空间信息丢失**——全局平均池化丢弃了空间结构，不知道交互区域在图像的哪个位置

这导致：
- 多阶段任务（接近→跟踪）中策略无法区分行为模式
- 缺乏精细空间定位能力，跟踪精度不足
- 无法适应当前 episode 的特定几何配置

## 核心贡献

提出 **Past-Present-Future (PPF)** 框架 + **FASA (Flow-Aware Spatial Attention)** 自监督空间注意力模块：

### 时间感知（三个自监督模块）
1. **Self-Predictive Latent Dynamics（未来）**：预测下一步观测潜变量，预测误差作为阶段转折指示器
2. **Progress Predictor + Dual FiLM（现在→全局）**：自监督学习任务进度，通过 FiLM 双路调制注入 flow network
3. **Episodic Memory Bank（过去）**：轻量 key-value 记忆模块，推理时积累观测历史

### 空间感知（自监督模块）
4. **FASA — Flow-Aware Spatial Attention（空间）**：利用帧间特征变化自动发现交互区域
   - 动态/静态双流池化：聚焦交互区域 + 保留场景上下文
   - 变化质心 + 方向编码：交互在哪 + 变化趋势
   - Spatial Softmax 坐标：各通道空间关键点
   - 无需外部教师模型，即插即用

### 视觉增强（离线预处理）
5. **Vision Foundation Model Pre-Distillation**：离线从 Grounding DINO 蒸馏到 ResNet-18 backbone，冻结后供 FASA 和策略使用

**完全自监督**，不需要任何人工标注的时间标签、阶段标签或空间标注。

---

## 论文结构

### 1. Introduction（引言）
- 机器人模仿学习中 flow matching 的优势
- 问题：时间盲目 + 空间信息丢失
- 现有方案的局限（显式标注、外部教师模型依赖）
- 提出 PPF + FASA 框架
- 贡献点总结（6点）

### 2. Related Work（相关工作）
- **2.1 Flow Matching for Robot Policies**: VITA, Diffusion Policy, π₀
- **2.2 Temporal Encoding in Imitation Learning**: obs stacking, ACT, TCN
- **2.3 Spatial Attention in Visuomotor Policies**: Spatial Softmax, Coordinate Attention, SE/CBAM, 蒸馏方法
- **2.4 Memory-Augmented Policies**: NTM, DNC, Transformer KV-cache, TTT
- **2.5 Self-Predictive Learning**: BYOL, VICReg, 前向动力学, 连接到 FASA

### 3. Preliminaries（预备知识）
- **3.1 OT-CFM**: 最优传输条件流匹配
- **3.2 VITA Architecture**: obs_latent → flow → action_latent → action_decoder
- **3.3 FiLM Conditioning**: 特征线性调制

### 4. Method（方法）
- **4.1 Overview**: 整体架构（时间 + 空间 + 记忆）
- **4.2 Temporal Feature Differencing**: 特征差分编码"语义速度"
- **4.3 Self-Predictive Latent Dynamics**: 预测误差 → 阶段转折指示器
- **4.4 Progress Predictor + Dual FiLM**: 序数排名 loss + 双路 FiLM
- **4.5 Episodic Memory Bank**: KV attention + 伪记忆训练
- **4.6 Flow-Aware Spatial Attention (FASA)**: ★新增核心方法
  - 帧间变化检测 → change_map
  - 动态/静态双流池化
  - 变化质心 + 方向编码 (cx, cy, dx, dy)
  - Spatial Softmax 坐标压缩
  - 优雅降级（无前一帧时）
  - 设计优势：自监督、即插即用、动态注意力
- **4.7 Vision Foundation Model Pre-Distillation**: 离线双蒸馏 + 冻结 backbone
- **4.8 Joint Training Objective**: 联合损失 + 梯度分析

### 5. Experimental Setup（实验设置）
- **5.1 Task**: MR2010 S-curve 焊缝跟踪（approach + tracking 两阶段）
- **5.2 Dataset**: 160 条专家轨迹，112,000 帧，50Hz
- **5.3 Baselines**: VITA / VITA+obs2 / VITA+progress / VITA+FASA / ACT
- **5.4 Metrics**: 接近成功率、跟踪误差、多级覆盖率、等级分布
- **5.5 Implementation**: chunk=64, latent=512, batch=32, 100k steps, FASA输出1060d/camera

### 6. Results（实验结果）
- **6.1 Main Results**: 定量对比表
- **6.2 Ablation Study**: F/D/P/M 各模块贡献
- **6.3 Analysis**:
  - Progress 预测曲线
  - Memory attention 热力图
  - Dynamics error 时间曲线
  - FASA 注意力可视化（approach vs tracking 不同关注模式）
  - 泛化性评估

### 7. Discussion（讨论）
- 通用性（不限于焊缝跟踪）
- vs Transformer 时间建模
- FASA vs 蒸馏空间注意力
- 计算开销（+5.3% 参数）
- 局限性（ordinal ranking 假设、伪记忆差异、FASA 冷启动、单任务评估）
- 未来方向

### 8. Conclusion（结论）

---

## 所需实验

| 实验 | 训练配置 | 备注 |
|------|---------|------|
| VITA Baseline | `--policy vita --obs_horizon 1` | 无时间/空间模块 |
| VITA + FASA only | `--policy vita --use_spatial_attn --obs_horizon 2` | 仅空间注意力 |
| PPF (完整) | `--policy vita --use_temporal --use_spatial_attn --obs_horizon 2` | 所有模块 |
| Ablation: VITA + Dynamics | 需要代码开关 | 仅 dynamics |
| Ablation: VITA + Progress | 需要代码开关 | 仅 progress |
| Ablation: VITA + Memory | 需要代码开关 | 仅 memory |
| obs_horizon=2 only | `--policy vita --obs_horizon 2` | 简单帧堆叠 |

## 所需图表

| 图号 | 内容 | 类型 |
|------|------|------|
| Fig.1 | PPF + FASA 整体架构图 | 系统框图 |
| Fig.2 | 焊缝跟踪任务示意图 | 3D 场景图 |
| Fig.3 | Temporal Feature Differencing 示意 | 数据流图 |
| Fig.4 | Dual FiLM 调制机制 | 模块细节图 |
| Fig.5 | Progress 预测曲线 | 折线图 |
| Fig.6 | Memory Attention 热力图 | 热力图 |
| Fig.7 | Dynamics Error 时间曲线 | 折线图 |
| Fig.8 | FASA 注意力可视化（approach vs tracking） | 热力图 |
| Fig.9 | 跟踪轨迹对比（PPF+FASA vs Baseline） | 3D 轨迹图 |
| Tab.1 | Main Results | 表格 |
| Tab.2 | Ablation Study (F/D/P/M) | 表格 |
| Tab.3 | 计算开销对比 | 表格 |
