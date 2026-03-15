# Fig.3 — Temporal Feature Differencing 示意图

## 图表类型
数据流图（data flow diagram），单栏宽度。

## 内容描述
对比"朴素帧堆叠"和"特征差分"两种时间编码方式。

## 布局结构（上下两部分）

### 上半部分：Naive Frame Stacking（标注 ✗）
- 两帧图像 I_{t-1}, I_t → 各自通过 ResNet → f_{t-1}, f_t
- 拼接 [f_{t-1}, f_t, s_{t-1}, s_t] → 维度标注 (512N + d_s) × 2
- 用灰色/淡色表示，标注"信息冗余，维度随 S 线性增长"

### 下半部分：Temporal Feature Differencing（标注 ✓，推荐方案）
- 两帧图像 I_{t-1}, I_t → 各自通过 **共享** ResNet → f_{t-1}, f_t
- 计算差分：Δf = f_t - f_{t-1}（用减号图标）
- 计算状态差分：Δs = s_t - s_{t-1}
- 拼接 [f_t, Δf, s_t, Δs] → 维度标注 2(512N + d_s)
- 用彩色高亮表示
- Δf 旁边标注"semantic velocity"
- Δs 旁边标注"joint velocity"

### 右侧对比小注
- 文字标注维度对比：
  - Naive: (512N + d_s) × S，随 S 增长
  - Ours: 2(512N + d_s)，固定维度

## 视觉风格
- 白色背景，学术风格
- ResNet 用蓝色圆角矩形，共享权重用虚线连接
- 差分运算用红色/橙色减号符号
- 拼接操作用竖线分隔的长条矩形
- 数据流箭头清晰

## 尺寸
单栏宽度，宽高比约 1:1，分辨率 300dpi
