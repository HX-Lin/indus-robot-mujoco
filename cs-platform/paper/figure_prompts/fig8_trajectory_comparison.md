# Fig.8 — 跟踪轨迹对比 (Tracking Trajectory Comparison)

## 图表类型
3D 轨迹图 或 2D 投影轨迹图，双栏全宽。

## 内容描述
对比 PPF 和 Baseline VITA 在同一 episode 上的实际跟踪轨迹与 ground-truth 焊缝轨迹，直观展示精度差异。

## 布局结构（3 个子图并排）

### 子图 (a)：Ground Truth Seam
- S 形焊缝的 3D 曲线（红色实线）
- 标注起点和终点
- 坐标轴标注 x, y, z (mm)

### 子图 (b)：VITA Baseline
- Ground-truth 焊缝（红色实线，半透明）
- VITA 跟踪轨迹（蓝色实线）
- 误差区域用浅红色 shading 标注（ground-truth 与实际轨迹之间）
- 标注 mean tracking error 数值
- 明显可见跟踪偏差，特别在曲率大的区域

### 子图 (c)：PPF (Ours)
- Ground-truth 焊缝（红色实线，半透明）
- PPF 跟踪轨迹（绿色实线）
- 误差区域用浅绿色 shading（明显更窄）
- 标注 mean tracking error 数值
- 可见轨迹紧贴 ground-truth

### 附加子图（可选）：
- 底部加一个 1D 误差曲线对比
  - X 轴：tracking timestep
  - Y 轴：Euclidean distance (mm)
  - 蓝色线 = VITA，绿色线 = PPF
  - 标注阈值线：1mm, 2mm, 5mm（水平虚线）

## 视觉风格
- 白色背景
- 3D 视角一致（相同的 elevation 和 azimuth）
- Ground-truth 红色，Baseline 蓝色，PPF 绿色
- 误差 shading 半透明
- 坐标轴刻度统一

## 数据来源
实验完成后从评估日志中提取末端执行器位置轨迹和 ground-truth 焊缝坐标。

## 尺寸
双栏全宽，宽高比约 3:1（三个子图并排），分辨率 300dpi
