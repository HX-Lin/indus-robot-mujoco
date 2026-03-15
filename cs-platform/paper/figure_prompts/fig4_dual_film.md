# Fig.4 — Dual FiLM 调制机制详细图

## 图表类型
模块细节图（module detail diagram），单栏宽度。

## 内容描述
展示 EnhancedFlowNetLayer 内部的 Dual FiLM 调制机制，对比标准单路 FiLM。

## 布局结构（左右对比）

### 左侧：Standard FiLM（VITA 原始）
- 输入 h
- LayerNorm → (1+α_t)⊙LN(h) + β_t
- MLP
- γ_t ⊙ MLP(·)
- 残差连接 h' = h + γ_t ⊙ MLP(·)
- 仅一条调制路径：SinEmb(t) → Linear → (γ_t, α_t, β_t)
- 标注 "Flow Timestep t ∈ [0,1]"

### 右侧：Dual FiLM（PPF 提出）
- 输入 h
- LayerNorm
- **第一路调制**（蓝色）：SinEmb(t) → Linear → (γ_t, α_t, β_t)
  - (1+α_t)⊙LN(h) + β_t → h_norm
  - 标注 "Flow Timestep"
- **第二路调制**（橙色）：SinEmb(p̂) → Linear → (γ_p, α_p, β_p)
  - (1+α_p)⊙h_norm + β_p → h_prog
  - 标注 "Task Progress"
- MLP(h_prog)
- 输出门控：(1+γ_p) ⊙ γ_t ⊙ MLP(·)
- 残差连接 h' = h + (1+γ_p) ⊙ γ_t ⊙ MLP(h_prog)
- 标注 "Zero-init" 在两个 Linear 旁边（用星号★标注）

### 关键标注
- 蓝色路径标注 "controls ODE integration position"
- 橙色路径标注 "controls behavioral mode"
- Zero-init 说明文字："初始化时 progress 模块输出为零，等价于标准 FiLM"

## 视觉风格
- 白色背景
- 左侧用灰色系（标准方法），右侧用彩色（我们的方法）
- 两条调制路径用不同颜色（蓝=flow time, 橙=progress）
- 运算符号（⊙, +）清晰标注
- 数据流自底向上

## 尺寸
单栏宽度，宽高比约 1:1.5（纵向较长），分辨率 300dpi
