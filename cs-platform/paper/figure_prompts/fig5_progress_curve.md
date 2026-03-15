# Fig.5 — Progress 预测曲线 (Progress Prediction Visualization)

## 图表类型
折线图（line plot），单栏宽度。

## 内容描述
展示训练好的 Progress Predictor 在多个评估 episode 上预测的 progress 值随帧数变化的曲线，验证其能正确反映任务阶段。

## 布局结构

### 坐标系
- **X 轴**：Episode timestep（0 ~ 700）
- **Y 轴**：Predicted progress p̂_t（0.0 ~ 1.0）
- 背景分区：
  - 左侧区域（0~200）用浅蓝色底色，标注 "Approach Phase"
  - 右侧区域（200~700）用浅橙色底色，标注 "Tracking Phase"
  - 分界线用灰色虚线，标注 "Phase Transition"

### 曲线
- 4~6 条曲线，每条代表一个不同的评估 episode（不同颜色）
- 所有曲线应呈现类似趋势：
  - Approach 阶段：progress 从低值缓慢上升
  - 转折点附近：progress 快速增加
  - Tracking 阶段：progress 持续单调上升至接近 1.0
- 用图例标注 "Episode 1", "Episode 2", ...

### 附加元素
- 所有曲线的均值用粗黑线标注，±std 用灰色 shading
- 理想 progress（t/T 线性增长）用灰色虚线标注 "Linear baseline"

## 视觉风格
- 白色背景，matplotlib/seaborn 学术风格
- 不同 episode 用不同颜色但同一色系（如蓝色渐变）
- 网格线用浅灰虚线
- 字体大小适合论文阅读

## 数据来源
实验完成后从评估日志中提取。图中数据为占位示意。

## 尺寸
单栏宽度，宽高比约 1.5:1，分辨率 300dpi
