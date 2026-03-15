# Fig.7 — Dynamics Error 时间曲线 (Dynamics Prediction Error Over Time)

## 图表类型
折线图（line plot），单栏宽度。

## 内容描述
展示 Self-Predictive Latent Dynamics 模型的预测误差 e_t 随 episode timestep 的变化，验证其在阶段转折处出现峰值。

## 布局结构

### 坐标系
- **X 轴**：Episode timestep（0 ~ 700）
- **Y 轴**：Prediction error e_t = ||ẑ_{t+1} - z_t^obs||²（对数刻度或线性刻度均可）
- 背景分区：
  - 左侧（0~200）浅蓝底色，标注 "Approach"
  - 右侧（200~700）浅橙底色，标注 "Tracking"
  - 分界线灰色虚线

### 曲线
- 4~6 条不同 episode 的曲线（不同颜色，半透明）
- 粗黑线为均值
- ±std 用灰色 shading

### 预期模式
- Approach 阶段：误差中等偏高（场景快速变化，但有规律）
- **Phase transition（~200步）：误差尖峰**（观测分布突变）
  - 用红色箭头或红色圆圈标注此处，附文字 "Phase Transition Peak"
- Tracking 阶段前期：误差下降，逐渐稳定
- Tracking 阶段中后期：误差保持低值（稳态跟踪，可预测）

### 附加元素
- 可在顶部加一个小窗口 (inset)，放大 transition 附近（150~250步）的误差变化

## 视觉风格
- 白色背景，matplotlib 学术风格
- 不同 episode 用蓝色渐变
- 尖峰区域用红色高亮标注
- 网格线浅灰虚线

## 数据来源
实验完成后从推理过程中记录每步的 prediction error。

## 尺寸
单栏宽度，宽高比约 1.5:1，分辨率 300dpi
