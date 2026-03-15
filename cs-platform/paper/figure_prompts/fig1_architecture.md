# Fig.1 — PPF 整体架构图 (System Architecture Diagram)

## 图表类型
系统架构框图（technical block diagram），论文核心方法图，占据整页宽度（two-column width）。

## 内容描述
展示 PPF (Past-Present-Future) 框架的完整数据流架构，基于 VITA flow matching policy 扩展。

## 布局结构（从左到右）

### 左侧：输入
- 两个摄像头图像（overhead cam, ee cam），用小图标表示 480×640 RGB
- 关节角度向量 s_t ∈ R^6，用小向量图标

### 中部上方：Observation Encoder（灰色底色区域，标注"Base VITA"）
- ResNet-18 backbone（冻结，用雪花图标❄标注）
- 两个时间帧 t-1 和 t 的特征提取
- 特征差分模块：显示 f_t, Δf = f_t - f_{t-1}, s_t, Δs_t 四个向量拼接
- 线性投影 → obs_latent z_t^obs ∈ R^512

### 中部核心：三个时间感知模块（用三种不同颜色区分）

#### 模块1 - Self-Predictive Latent Dynamics（蓝色/蓝绿色，标注"Future"）
- 输入：z_t^obs
- 3层MLP → 预测 ẑ_{t+1}
- 输出：prediction error e_t = ||ẑ_{t+1} - z_t^obs||²
- 用虚线箭头标注 stop-gradient (sg)

#### 模块2 - Progress Predictor（橙色/黄色，标注"Present"）
- 输入：sg(z_t^obs) + e_t（从模块1来）
- 3层MLP + Sigmoid → progress p̂_t ∈ [0,1]
- 输出用虚线连接到 Flow Network 的 Dual FiLM

#### 模块3 - Episodic Memory Bank（绿色，标注"Past"）
- 记忆缓冲区：一排 latent 向量 [m_1, m_2, ..., m_T]
- Query/Key/Value 投影
- Softmax attention → context c_t
- 与 z_t^obs 拼接 → W_fuse → enriched latent z_t^enr

### 中部下方：Flow Network + Action Decoder（灰色底色区域）
- EnhancedFlowNet：4层 MLP，每层有 **Dual FiLM**
- Dual FiLM 详细展示：两条调制路径
  - 路径1：SinEmb(t) → FiLM params (γ_t, α_t, β_t)，标注"Flow Timestep"
  - 路径2：SinEmb(p̂) → FiLM params (γ_p, α_p, β_p)，标注"Task Progress"
- Flow 起点：z_t^enr（从 Memory Bank 来）
- Flow 终点：z^act（action latent）
- Action Decoder (MLP) → action chunk a_{t:t+H}

### 右侧：输出
- Action chunk 可视化：16步的关节角度序列

## 视觉风格
- 学术论文风格，清晰简洁
- 白色背景
- 模块用圆角矩形，不同颜色区分三个时间模块
- Base VITA 部分用浅灰色底色
- 数据流用实线箭头，梯度流/stop-gradient 用虚线箭头
- 关键公式标注在对应模块旁边
- 整体配色方案：蓝色系(Future) / 橙色系(Present) / 绿色系(Past) / 灰色(Base)
- 字体：无衬线体，公式用 LaTeX 渲染风格

## 尺寸
论文双栏全宽，建议宽高比 2:1，分辨率 300dpi
