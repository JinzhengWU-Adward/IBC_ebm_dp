# EBMConvMLP 底层细节分析

## 目录
1. [能量模型基础](#能量模型基础)
2. [模块分解](#模块分解)
3. [数学公式到代码映射](#数学公式到代码映射)
4. [可视化任务设计](#可视化任务设计)

---

## 能量模型基础

### 核心概念

**能量函数 (Energy Function)**: 
$$E_\theta(x, y) : \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}$$

- $x$: 输入图像
- $y$: 候选坐标
- $E_\theta(x, y)$: 能量值（越低越好）

**条件概率分布**:
$$p(y|x) = \frac{\exp(-E_\theta(x, y))}{Z(x)} = \frac{\exp(-E_\theta(x, y))}{\int \exp(-E_\theta(x, y')) dy'}$$

**训练目标 (InfoNCE Loss)**:
$$\mathcal{L} = -\log \frac{\exp(-E_\theta(x, y^+))}{\exp(-E_\theta(x, y^+)) + \sum_{i=1}^N \exp(-E_\theta(x, y_i^-))}$$

其中 $y^+$ 是正样本，$y_i^-$ 是负样本。

---

## 模块分解

### 1. CoordConv (坐标卷积)

**数学原理**:
在输入特征图上添加归一化的坐标通道，帮助网络学习空间位置信息。

**公式**:
- X坐标: $x_{coord}[i,j] = \frac{2i}{H-1} - 1 \in [-1, 1]$
- Y坐标: $y_{coord}[i,j] = \frac{2j}{W-1} - 1 \in [-1, 1]$

**代码实现**:
```python
# modules.py:11-28
y_coords = 2.0 * torch.arange(image_height) / (image_height - 1.0) - 1.0
x_coords = 2.0 * torch.arange(image_width) / (image_width - 1.0) - 1.0
coords = torch.stack((y_coords, x_coords), dim=0)
x = torch.cat((coords, x), dim=1)  # 拼接坐标通道
```

**输入**: `(B, C, H, W)` → **输出**: `(B, C+2, H, W)`

---

### 2. CNN (卷积神经网络)

**结构**:
- 输入: `(B, 3或5, 96, 96)`
- 卷积块: `[16, 32, 32]` 通道
- 残差块: 每个通道数对应一个 ResidualBlock

**ResidualBlock 公式**:
$$y = x + \text{Conv2}(\text{ReLU}(\text{Conv1}(\text{ReLU}(x))))$$

**代码实现**:
```python
# models.py:84-89
out = activation(x)
out = conv1(out)
out = activation(x)  # 注意：这里用的是 x 而不是 out
out = conv2(out)
return out + x
```

**输出**: `(B, 32, 96, 96)`

---

### 3. 1x1 卷积降维

**目的**: 将特征通道从 32 降到 16

**公式**:
$$f_{16} = \text{ReLU}(\text{Conv1x1}(f_{32}))$$

**代码实现**:
```python
# models.py:169, 177
self.conv = nn.Conv2d(32, 16, 1)  # 1x1 卷积
out = F.relu(self.conv(out))
```

**输出**: `(B, 16, 96, 96)`

---

### 4. SpatialSoftArgmax (空间软最大值)

**数学原理**:
对每个特征图计算加权平均位置，将 2D 特征图压缩为 1D 特征向量。

**公式**:
$$\text{softmax}(f_{ij}) = \frac{\exp(f_{ij})}{\sum_{i',j'} \exp(f_{i'j'})}$$

$$x_{mean} = \sum_{i,j} \text{softmax}(f_{ij}) \cdot x_{coord}[i,j]$$
$$y_{mean} = \sum_{i,j} \text{softmax}(f_{ij}) \cdot y_{coord}[i,j]$$

**代码实现**:
```python
# modules.py:66-86
softmax = F.softmax(x.view(-1, h * w), dim=-1)  # (B*C, H*W)
xc, yc = self._coord_grid(h, w, x.device)  # 归一化坐标网格
x_mean = (softmax * xc.flatten()).sum(dim=1)  # 加权平均
y_mean = (softmax * yc.flatten()).sum(dim=1)
return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)  # (B, C*2)
```

**输入**: `(B, 16, 96, 96)` → **输出**: `(B, 32)` (16个特征 × 2坐标)

---

### 5. 特征融合 (Feature Fusion)

**数学原理**:
将图像特征 $\phi(x)$ 与候选坐标 $y$ 拼接，形成联合表示。

**公式**:
$$f_{fused} = [\phi(x), y] \in \mathbb{R}^{D+2}$$

**代码实现**:
```python
# models.py:179
# out: (B, 32) - 图像特征
# y: (B, N, 2) - N个候选坐标
fused = torch.cat([
    out.unsqueeze(1).expand(-1, y.size(1), -1),  # (B, 1, 32) → (B, N, 32)
    y  # (B, N, 2)
], dim=-1)  # (B, N, 34)
```

**输出**: `(B, N, 34)` 其中 N 是候选坐标数量

---

### 6. MLP (多层感知机)

**数学原理**:
将融合特征映射到能量值。

**公式**:
$$E_\theta(x, y) = \text{MLP}([\phi(x), y])$$

对于每个候选坐标:
$$h_1 = \text{ReLU}(W_1 \cdot f_{fused} + b_1)$$
$$E = W_2 \cdot h_1 + b_2$$

**代码实现**:
```python
# models.py:181-183
fused = fused.reshape(B * N, D)  # (B*N, 34)
out = self.mlp(fused)  # (B*N, 1)
return out.view(B, N)  # (B, N) - 每个候选坐标的能量
```

**输入**: `(B*N, 34)` → **输出**: `(B, N)` - 每个候选坐标的能量值

---

### 7. InfoNCE 损失

**数学原理**:
将能量值转换为概率，使用交叉熵损失。

**公式**:
$$\text{logits} = -E_\theta(x, y)$$
$$p(y^+|x) = \text{softmax}(\text{logits})[k]$$
$$\mathcal{L} = -\log p(y^+|x) = \text{CrossEntropy}(\text{logits}, k)$$

其中 $k$ 是正样本的索引。

**代码实现**:
```python
# trainer.py:204-209
energy = self.model(input, targets)  # (B, N+1)
logits = -1.0 * energy  # 能量转logit
loss = F.cross_entropy(logits, ground_truth)  # ground_truth是正样本索引
```

---

### 8. 推理优化 (Derivative-Free Optimizer)

**数学原理**:
通过重要性采样迭代优化，找到能量最低的坐标。

**迭代公式**:
1. **计算概率**: $p_i = \frac{\exp(-E(x, y_i))}{\sum_j \exp(-E(x, y_j))}$
2. **重采样**: 按概率分布采样新候选
3. **添加噪声**: $y_{new} = y_{sampled} + \epsilon \cdot \mathcal{N}(0, 1)$
4. **噪声衰减**: $\epsilon_{t+1} = \epsilon_t \cdot \alpha$

**代码实现**:
```python
# optimizers.py:114-127
for i in range(self.iters):
    energies = ebm(x, samples)  # 计算能量
    probs = F.softmax(-1.0 * energies, dim=-1)  # 转概率
    idxs = torch.multinomial(probs, ...)  # 重要性采样
    samples = samples[idxs]  # 重采样
    samples = samples + torch.randn_like(samples) * noise_scale  # 加噪声
    noise_scale *= self.noise_shrink  # 噪声衰减
```

---

## 可视化任务设计

### 任务 1: 1D-1D 能量函数可视化

**目标**: 理解能量函数 $E(x, y)$ 的形状

**任务**: 给定一个固定的 1D 输入 $x$，可视化能量函数 $E(x, y)$ 关于 $y$ 的变化

**实现**:
- 输入: 1D 信号 $x \in \mathbb{R}^{100}$
- 输出: 能量函数 $E(x, y)$ 其中 $y \in [-1, 1]$
- 可视化: 绘制 $y$ vs $E(x, y)$ 曲线

---

### 任务 2: 2D 能量景观可视化

**目标**: 可视化 2D 坐标空间中的能量分布

**任务**: 给定一个固定的图像 $x$，在 2D 坐标空间 $[-1,1] \times [-1,1]$ 中计算能量值

**实现**:
- 输入: 固定图像 $x$
- 输出: 2D 能量图 $E(x, (y_1, y_2))$ 其中 $(y_1, y_2) \in [-1,1]^2$
- 可视化: 热力图或等高线图

---

### 任务 3: SpatialSoftArgmax 可视化

**目标**: 理解空间软最大值如何定位特征

**任务**: 可视化特征图及其对应的加权平均位置

**实现**:
- 输入: 特征图 $f \in \mathbb{R}^{H \times W}$
- 输出: 软最大值权重分布和计算出的位置
- 可视化: 
  - 特征图热力图
  - 软最大值权重分布
  - 计算出的 $(x_{mean}, y_{mean})$ 位置标记

---

### 任务 4: InfoNCE 损失可视化

**目标**: 理解正负样本的能量差异

**任务**: 可视化训练过程中正样本和负样本的能量分布

**实现**:
- 输入: 一批训练数据
- 输出: 正样本和负样本的能量值分布
- 可视化: 直方图对比，展示能量分离过程

---

### 任务 5: 优化过程可视化

**目标**: 可视化推理时的优化轨迹

**任务**: 在 2D 坐标空间中绘制优化迭代过程

**实现**:
- 输入: 一个测试图像
- 输出: 每次迭代的候选坐标分布
- 可视化: 
  - 2D 散点图，不同颜色表示不同迭代
  - 能量等高线图
  - 优化轨迹箭头

---

### 任务 6: 特征融合效果可视化

**目标**: 理解图像特征和坐标如何影响能量

**任务**: 固定图像特征，改变坐标，观察能量变化

**实现**:
- 输入: 固定图像特征 $\phi(x)$
- 输出: 不同坐标 $y$ 对应的能量值
- 可视化: 3D 表面图或 2D 热力图

---

### 任务 7: 简化 1D EBM 训练演示

**目标**: 在 1D 任务上完整演示 EBM 训练过程

**任务**: 
- 输入: 1D 信号 $x \in \mathbb{R}^{100}$
- 输出: 1D 坐标 $y \in [-1, 1]$
- 模型: 简化的 EBM（去掉 CNN，只用 MLP）

**实现**:
- 训练数据: 生成简单的 1D 回归任务
- 模型: $E(x, y) = \text{MLP}([\text{Flatten}(x), y])$
- 可视化: 训练过程中能量函数的变化

---

### 任务 8: 2D 坐标回归简化版

**目标**: 在 2D 任务上演示完整流程

**任务**:
- 输入: 简单的 2D 图像（如带有一个点的图像）
- 输出: 2D 坐标 $(y_1, y_2) \in [-1,1]^2$
- 模型: 简化的 EBM（使用简单的 CNN）

**实现**:
- 训练数据: 生成带有一个标记点的图像
- 模型: 完整的 EBMConvMLP（但可以简化 CNN）
- 可视化: 训练和推理的完整过程

