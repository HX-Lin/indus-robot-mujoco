# Fig.6 — Memory Attention 热力图 (Memory Attention Heatmap)

## 图表类型
热力图（heatmap），单栏宽度。

## 内容描述
可视化 Episodic Memory Bank 在推理过程中的 attention weights α，展示策略在不同时刻关注了哪些历史帧。

## 布局结构

### 主图：热力图
- **X 轴**：Memory buffer 位置（历史帧索引，从早到晚，0 ~ T）
- **Y 轴**：当前推理 timestep（0 ~ 700，从上到下）
- **颜色**：attention weight α 值，用 viridis 或 hot 色图
  - 深色 = 高注意力权重
  - 浅色 = 低注意力权重
- 每行对应当前 timestep 对所有历史帧的 attention 分布

### 预期模式
- Approach 阶段（前200步）：注意力较分散，或集中在最近几帧
- 转折点（~200步）：注意力可能回看 approach 起始帧
- Tracking 阶段（200~700步）：
  - 注意力集中在轨迹的关键转折帧
  - 或注意力集中在最近的若干帧（局部关注模式）
  - 可能对 approach 阶段的帧给予低权重

### 附加标注
- Y轴上标注 "Approach" 和 "Tracking" 分区
- 分界线用白色虚线
- Colorbar 在右侧，标注 "Attention Weight"

## 视觉风格
- 白色背景
- 热力图主体用 viridis 色图（或 Reds/Blues）
- 标注清晰
- 可在热力图旁边加一个小的 marginal histogram 展示平均 attention 分布

## 数据来源
实验完成后从推理过程中记录 attention weights。

## 尺寸
单栏宽度，宽高比约 1:1.2（略纵向），分辨率 300dpi
