# Particle EBM 训练架构文档

本文档详细描述了 Particle 环境的 EBM（Energy-Based Model）训练架构，基于 PyTorch 复刻的 IBC（Implicit Behavioral Cloning）实现。

## 目录

1. [整体架构概览](#整体架构概览)
2. [数据流](#数据流)
3. [网络结构](#网络结构)
4. [训练流程](#训练流程)
5. [损失函数](#损失函数)
6. [采样器](#采样器)
7. [推理流程](#推理流程)
8. [超参数配置](#超参数配置)

---

## 整体架构概览

### 任务描述

Particle 环境是一个 2D 粒子导航任务：
- **状态空间**: 粒子位置 (x, y)、速度 (vx, vy)、两个目标位置
- **动作空间**: 2D 速度控制 (vx, vy)
- **目标**: 粒子依次访问两个目标点

### 核心思想

使用能量基模型（EBM）学习**隐式策略**：
- 不直接建模 π(a|s)（显式策略）
- 而是学习能量函数 E(s, a)，能量越低表示动作越优
- 推理时通过优化能量函数找到最优动作

### 架构流程图

```
┌──────────────────────────────────────────────────────────────┐
│                        训练阶段                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  数据集 (轨迹)                                                │
│       │                                                      │
│       ├─> 观测序列 (obs_seq)  ──┐                            │
│       │   [pos, vel, goal1, goal2] × 2                       │
│       │                           │                          │
│       └─> 真实动作 (action)      ────────> EBM 模型          │
│           [vx, vy]                │        ↓                 │
│                                   │     能量值 E_pos         │
│  ULA 采样器                       │        ↓                 │
│       │                           │                          │
│       └─> 负样本 (negatives)  ────┘                          │
│           随机初始化 + MCMC                                   │
│           [vx, vy] × 8               能量值 E_neg            │
│                                          ↓                   │
│                                   ┌──────────────┐           │
│                                   │  损失函数    │           │
│                                   ├──────────────┤           │
│                                   │ InfoNCE Loss │           │
│                                   │      +       │           │
│                                   │ Grad Penalty │           │
│                                   └──────┬───────┘           │
│                                          ↓                   │
│                                   反向传播 & 优化             │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                        推理阶段                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  当前观测 (obs_seq)                                          │
│       │                                                      │
│       └─> 无导数优化器 (Derivative-Free Optimizer)           │
│            │                                                 │
│            ├─> 采样多个候选动作 (16384 个)                    │
│            │   [vx, vy] × 16384                              │
│            │                                                 │
│            ├─> EBM 评估能量                                  │
│            │   E(obs, a) for all a                           │
│            │                                                 │
│            ├─> 重要性采样 (Importance Sampling)              │
│            │   按概率 p ∝ exp(-E) 重采样                      │
│            │                                                 │
│            ├─> 添加噪声 & 迭代 (3 次)                         │
│            │   噪声逐渐衰减 (0.33 → 0.165 → 0.0825)         │
│            │                                                 │
│            └─> 返回能量最低的动作                             │
│                a* = argmin_a E(obs, a)                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 数据流

### 1. 数据加载

**数据来源**: JSON 格式的轨迹文件
```json
{
  "trajectory": {
    "positions": [[x1, y1], [x2, y2], ...],  // 粒子位置序列
    "velocities": [[vx1, vy1], [vx2, vy2], ...]  // 粒子速度序列
  },
  "actions": [[ax1, ay1], [ax2, ay2], ...],  // 动作序列（控制速度）
  "first_goal_position": [gx1, gy1],
  "second_goal_position": [gx2, gy2]
}
```

**数据结构**:
- 每个轨迹包含 N 步（通常 200 步）
- 每步包含：位置、速度、动作、目标点

### 2. 数据预处理

#### 2.1 观测构建

对于每个时间步 t，观测包含：
```python
obs_t = [
    pos_agent,      # 粒子位置 (2D): [x, y]
    vel_agent,      # 粒子速度 (2D): [vx, vy]
    pos_first_goal, # 第一个目标 (2D): [gx1, gy1]
    pos_second_goal # 第二个目标 (2D): [gx2, gy2]
]
# 总维度: 2 + 2 + 2 + 2 = 8
```

#### 2.2 序列窗口

使用滑动窗口构建观测序列（obs_seq_len = 2）：
```python
obs_seq = [obs_t, obs_{t+1}]  # 形状: (2, 8)
action = action_t              # 形状: (2,)
```

#### 2.3 归一化

**观测归一化**（Z-score）:
```python
obs_normalized = (obs - obs_mean) / obs_std
```

**动作归一化**（Min-Max 到 [-1, 1]）:
```python
action_normalized = 2.0 * (action - action_min) / action_range - 1.0
```

### 3. 批次数据

**批次形状**:
```python
batch = {
    'obs_seq': (B, obs_seq_len, obs_dim),  # (512, 2, 8)
    'action': (B, action_dim)               # (512, 2)
}
```

---

## 网络结构

### 1. SequenceEBM 模型

整体架构：MLP + 残差块

```
输入: 
  - obs_seq: (B, 2, 8)  观测序列
  - action: (B, N, 2)   动作候选（N 个）

流程:
  obs_seq -> Flatten -> [obs_flat: (B, 16)]
  
  obs_flat -> Expand -> [obs_expanded: (B, N, 16)]
  action                [action: (B, N, 2)]
  
  Concat -> [x: (B, N, 18)]
  
  Reshape -> [x: (B*N, 18)]
  
  Projection -> [x: (B*N, 256)]
  
  ResidualBlock_1 -> [x: (B*N, 256)]
  
  Energy Projection -> [energy: (B*N, 1)]
  
  Reshape -> [energy: (B, N)]

输出: 能量值 (B, N)
```

### 2. 网络层详解

#### 2.1 投影层（Projection Layer）

```python
# 输入: obs_seq (B*N, obs_seq_len * obs_dim + action_dim)
#      = (B*N, 2*8 + 2) = (B*N, 18)
projection = nn.Linear(18, 256)

# 输出: (B*N, 256)
```

**作用**: 将拼接后的输入映射到隐藏空间

#### 2.2 残差块（MLPResidualBlock）

**结构**（Pre-Activation ResNet）:
```
x -> [LayerNorm] -> ReLU -> Dropout -> Linear(256→256)
  -> [LayerNorm] -> ReLU -> Dropout -> Linear(256→256)
  -> Add(x) -> output
```

**注意**:
- 使用 Pre-Activation 顺序（先归一化/激活，再卷积）
- 残差连接（跳跃连接）缓解梯度消失
- 可选的 LayerNorm（IBC 默认使用）

**代码实现**:
```python
class MLPResidualBlock(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1, norm_type='layer'):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim) if norm_type == 'layer' else None
        self.norm2 = nn.LayerNorm(hidden_dim) if norm_type == 'layer' else None
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        identity = x
        # 第一层
        out = self.norm1(x) if self.norm1 is not None else x
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.fc1(out)
        # 第二层
        out = self.norm2(out) if self.norm2 is not None else out
        out = self.activation(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        # 残差连接
        return out + identity
```

#### 2.3 能量投影层（Energy Projection）

```python
energy_projection = nn.Linear(256, 1)

# 输入: (B*N, 256)
# 输出: (B*N, 1) -> reshape -> (B, N)
```

**作用**: 将隐藏特征映射到标量能量值

### 3. 网络参数统计

**默认配置**（hidden_dim=256, num_residual_blocks=1）:

| 层名称 | 参数形状 | 参数数量 |
|--------|---------|---------|
| Projection | (18, 256) + (256,) | 4,864 |
| ResidualBlock.fc1 | (256, 256) + (256,) | 65,792 |
| ResidualBlock.fc2 | (256, 256) + (256,) | 65,792 |
| ResidualBlock.norm1 | (256,) × 2 | 512 |
| ResidualBlock.norm2 | (256,) × 2 | 512 |
| Energy Projection | (256, 1) + (1,) | 257 |
| **总计** | | **137,729** |

---

## 训练流程

### 1. 训练循环

```python
for iteration in range(num_iterations):
    # 1. 加载批次数据
    batch = next(dataloader)
    obs_seq = batch['obs_seq']  # (B, 2, 8)
    actions = batch['action']    # (B, 2)
    
    # 2. 计算正样本能量
    E_pos = model(obs_seq, actions.unsqueeze(1))  # (B, 1)
    
    # 3. 生成负样本（ULA 采样）
    negatives = ula_sampler.sample(
        x=obs_seq,
        ebm=model,
        num_samples=8  # num_counter_examples
    )  # (B, 8, 2)
    
    # 4. 计算负样本能量
    E_neg = model(obs_seq, negatives)  # (B, 8)
    
    # 5. 计算 InfoNCE 损失
    loss_infonce = compute_info_nce_loss(E_pos, E_neg)
    
    # 6. 计算梯度惩罚损失
    combined_actions = torch.cat([negatives, actions.unsqueeze(1)], dim=1)
    loss_grad = compute_gradient_penalty(model, obs_seq, combined_actions)
    
    # 7. 总损失
    loss = loss_infonce + loss_grad
    
    # 8. 反向传播 & 优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

### 2. 训练参数

| 参数名称 | 默认值 | 说明 |
|---------|-------|------|
| batch_size | 512 | 批次大小 |
| learning_rate | 1e-3 | 初始学习率 |
| num_iterations | 100,000 | 训练迭代次数 |
| num_counter_examples | 8 | 负样本数量 |
| lr_decay_steps | 100 | 学习率衰减步数 |
| lr_decay_rate | 0.99 | 学习率衰减率 |

### 3. 学习率调度

使用 **TensorFlow 风格的指数衰减**:

```python
lr = initial_lr * (decay_rate ^ (step / decay_steps))
```

**示例**:
- step = 0: lr = 1e-3
- step = 100: lr = 1e-3 × 0.99^1 = 9.9e-4
- step = 200: lr = 1e-3 × 0.99^2 = 9.801e-4
- step = 10000: lr = 1e-3 × 0.99^100 ≈ 3.66e-4

**特点**:
- 连续衰减（staircase=False）
- 衰减缓慢（每 100 步衰减 1%）

---

## 损失函数

### 1. InfoNCE 损失

**目标**: 对比学习，让正样本能量低于负样本能量

**公式**:
```
L_InfoNCE = -log( exp(-E_pos / τ) / (exp(-E_pos / τ) + Σ exp(-E_neg_i / τ)) )
```

等价于交叉熵损失:
```
L_InfoNCE = CrossEntropy([E_neg_1, ..., E_neg_N, E_pos], label=N)
```

**代码实现**:
```python
def compute_info_nce_loss(E_pos, E_neg, temperature=1.0):
    """
    Args:
        E_pos: (B, 1) 正样本能量
        E_neg: (B, num_negatives) 负样本能量
        temperature: softmax 温度
    
    Returns:
        loss: 标量损失
    """
    # 拼接能量: (B, num_negatives + 1)
    energies = torch.cat([E_neg, E_pos], dim=1)
    
    # Softmax（能量越低，概率越高）
    probs = F.softmax(-energies / temperature, dim=-1)
    
    # 正样本标签（最后一列）
    labels = torch.full((B,), num_negatives, dtype=torch.long)
    
    # 交叉熵损失
    loss = F.cross_entropy(-energies / temperature, labels)
    return loss
```

**直观理解**:
- 正样本能量应该 **低**（模型认为是好的动作）
- 负样本能量应该 **高**（模型认为是坏的动作）
- 通过对比学习，拉大正负样本的能量差距

### 2. 梯度惩罚损失

**目标**: 约束能量函数的梯度，防止梯度爆炸

**公式**:
```
L_grad = mean(max(0, ||∇_a E(s, a)|| - margin)^2)
```

**超参数**:
- `grad_margin = 1.0`: 梯度 margin（允许梯度小于 1.0）
- `grad_norm_type = 'inf'`: 使用 L∞ 范数（最大绝对值）
- `square_grad_penalty = True`: 对惩罚值平方

**代码实现**:
```python
def compute_gradient_penalty(model, obs_seq, actions, grad_margin=1.0):
    """
    Args:
        model: EBM 模型
        obs_seq: (B, obs_seq_len, obs_dim)
        actions: (B, num_samples, action_dim)
        grad_margin: 梯度 margin
    
    Returns:
        grad_loss: 标量梯度惩罚损失
    """
    # 展平 actions
    B, num_samples, action_dim = actions.shape
    actions_flat = actions.view(-1, action_dim).detach().requires_grad_(True)
    obs_seq_flat = obs_seq.unsqueeze(1).expand(B, num_samples, -1, -1)
    obs_seq_flat = obs_seq_flat.reshape(-1, obs_seq.size(1), obs_seq.size(2))
    
    # 计算能量
    energies = model(obs_seq_flat, actions_flat.unsqueeze(1)).squeeze(-1)
    
    # 计算梯度 dE/da
    grad = torch.autograd.grad(
        outputs=energies,
        inputs=actions_flat,
        grad_outputs=torch.ones_like(energies),
        create_graph=True
    )[0]
    
    # L∞ 范数
    grad_norms = torch.norm(grad, p=float('inf'), dim=1)  # (B*num_samples,)
    grad_norms = grad_norms.view(B, num_samples)
    
    # 应用 margin
    grad_norms = torch.clamp(grad_norms - grad_margin, min=0.0)
    
    # 平方并平均
    grad_loss = (grad_norms ** 2).mean()
    return grad_loss
```

**直观理解**:
- 限制能量函数的梯度不要太大
- 防止能量景观过于陡峭
- 有助于 MCMC 采样和梯度下降优化

### 3. 总损失

```python
loss = loss_infonce + loss_grad
```

**权重**:
- InfoNCE 损失权重: 1.0（默认）
- 梯度惩罚权重: 1.0（grad_loss_weight）

---

## 采样器

### 1. ULA 采样器（训练阶段）

**全称**: Unadjusted Langevin Algorithm（未校正朗之万算法）

**用途**: 训练时生成负样本（counter examples）

**更新规则**:
```
a_{t+1} = a_t - step_size × (0.5 × ∇_a E(s, a_t) + noise)
```

其中 `noise ~ N(0, noise_scale^2 I)`

**超参数**:
| 参数 | 默认值 | 说明 |
|------|-------|------|
| step_size | 0.1 | 初始步长 |
| num_steps | 100 | MCMC 迭代步数 |
| noise_scale | 1.0 | 噪声标准差 |
| step_size_final | 1e-5 | 最终步长 |
| step_size_power | 2.0 | 步长衰减幂次 |

**步长调度**（Polynomial Schedule）:
```python
progress = step / (num_steps - 1)
step_size_t = (step_size - step_size_final) × (1 - progress)^power + step_size_final
```

**采样流程**:
```python
# 1. 随机初始化
a_0 ~ Uniform([-1, 1]^2)

# 2. MCMC 迭代（100 步）
for t in range(100):
    # 计算梯度
    grad = ∇_a E(s, a_t)
    
    # 更新动作
    noise = N(0, 1.0^2)
    a_{t+1} = a_t - step_size_t × (0.5 × grad + noise)
    
    # 限制在边界内
    a_{t+1} = clip(a_{t+1}, -1, 1)
    
    # 更新步长
    step_size_t = polynomial_decay(t)

# 3. 返回最终样本
return a_100
```

**特点**:
- **梯度引导**: 沿能量梯度下降（找到低能量区域）
- **噪声探索**: 添加随机噪声（避免陷入局部最优）
- **步长衰减**: 开始大步探索，后期小步精细调整
- **负样本质量**: 生成的负样本更具挑战性（接近决策边界）

### 2. 无导数优化器（推理阶段）

**全称**: Derivative-Free Optimizer

**用途**: 推理时找到最优动作（argmin_a E(s, a)）

**算法**: 迭代重要性采样（Iterative Importance Sampling）

**超参数**:
| 参数 | 默认值 | 说明 |
|------|-------|------|
| num_samples | 16384 | 候选样本数（2^14） |
| num_iters | 3 | 优化迭代次数 |
| noise_scale | 0.33 | 初始噪声标准差 |
| noise_shrink | 0.5 | 噪声衰减率 |

**优化流程**:
```python
# 1. 初始化：随机采样
a ~ Uniform([-1, 1]^2) × 16384

# 2. 迭代优化（3 次）
for iter in range(3):
    # 计算能量
    E = EBM(s, a)  # (16384,)
    
    # 转换为概率（能量越低，概率越高）
    p = softmax(-E)  # (16384,)
    
    # 重要性采样
    a_resampled = resample(a, p, num_samples=16384)
    
    # 添加噪声
    noise = N(0, noise_scale^2)
    a = a_resampled + noise
    
    # 限制在边界内
    a = clip(a, -1, 1)
    
    # 噪声衰减
    noise_scale *= 0.5  # 0.33 -> 0.165 -> 0.0825

# 3. 返回能量最低的样本
a* = argmin_a E(s, a)
return a*
```

**特点**:
- **无需梯度**: 不依赖梯度信息（适用于不可微能量函数）
- **全局搜索**: 初始随机采样覆盖整个空间
- **自适应优化**: 通过重要性采样集中在低能量区域
- **噪声退火**: 逐渐减小噪声，从探索转向利用

**对比 ULA**:
| 特性 | ULA（训练） | 无导数优化器（推理） |
|------|------------|-------------------|
| 需要梯度 | ✓ | ✗ |
| 采样数量 | 少（8 个） | 多（16384 个） |
| 迭代次数 | 多（100 步） | 少（3 次） |
| 用途 | 生成负样本 | 找到最优动作 |
| 速度 | 较慢（梯度计算） | 较快（前向传播） |

---

## 推理流程

### 1. 推理示例

```python
# 1. 加载模型和归一化参数
model = SequenceEBM(...)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

norm_params = json.load(open('norm_params.json'))

# 2. 归一化当前观测
obs_seq = get_current_observation()  # (1, 2, 8)
obs_seq = (obs_seq - norm_params['obs_mean']) / norm_params['obs_std']

# 3. 创建无导数优化器
optimizer = DerivativeFreeOptimizer(
    bounds=[[-1, -1], [1, 1]],
    num_samples=16384,
    num_iters=3,
    noise_scale=0.33,
    noise_shrink=0.5
)

# 4. 推理最优动作
action_normalized, _ = optimizer.infer(obs_seq, model)

# 5. 反归一化动作
action = (action_normalized + 1.0) / 2.0 * action_range + action_min

# 6. 执行动作
env.step(action)
```

### 2. 推理性能

**计算复杂度**:
- 单次前向传播: O(B × N × hidden_dim^2)
- 推理迭代: 3 次 × 16384 样本 × 前向传播
- 总计: ~49152 次前向传播

**优化策略**:
1. **批量计算**: 16384 个样本并行评估
2. **GPU 加速**: 使用 CUDA 加速矩阵运算
3. **减少迭代**: num_iters = 3（平衡精度和速度）
4. **减少样本**: num_samples 可调（16384 → 4096）

**推理时间**（参考）:
- GPU (RTX 3090): ~10-20 ms/step
- CPU (Intel i9): ~100-200 ms/step

---

## 超参数配置

### 1. 网络超参数

```python
model = SequenceEBM(
    obs_dim=8,                  # 观测维度（固定）
    action_dim=2,               # 动作维度（固定）
    obs_seq_len=2,              # 观测序列长度（匹配 IBC）
    hidden_dim=256,             # 隐藏层维度（匹配 IBC width=256）
    num_residual_blocks=1,      # 残差块数量（匹配 IBC depth=2）
    dropout=0.0,                # Dropout 概率（IBC 默认 0.0）
    norm_type=None              # 归一化类型（IBC 默认 None）
)
```

### 2. 训练超参数

```python
train_particle_ebm(
    # 数据
    data_dir='data/_2d/particle',
    batch_size=512,             # 匹配 IBC
    
    # 优化器
    learning_rate=1e-3,         # 匹配 IBC
    lr_decay_steps=100,         # 匹配 IBC
    lr_decay_rate=0.99,         # 匹配 IBC
    
    # 训练
    num_iterations=100000,      # 匹配 IBC
    
    # 损失函数
    temperature=1.0,            # InfoNCE 温度
    num_counter_examples=8,     # 负样本数量（匹配 IBC）
    
    # 梯度惩罚
    add_grad_penalty=True,      # 匹配 IBC（grad_penalty.grad_margin=1.0）
    grad_margin=1.0,            # 匹配 IBC
    grad_norm_type='inf',       # 匹配 IBC（L∞ 范数）
    square_grad_penalty=True,   # 匹配 IBC
    grad_loss_weight=1.0,       # 梯度损失权重
    
    # ULA 采样器
    ula_step_size=0.1,          # 初始步长（匹配 IBC）
    ula_num_steps=100,          # MCMC 步数（匹配 IBC）
    ula_noise_scale=1.0,        # 噪声标准差（匹配 IBC）
    ula_step_size_final=1e-5,   # 最终步长（匹配 IBC）
    ula_step_size_power=2.0,    # 步长衰减幂次（匹配 IBC）
    
    # 其他
    save_interval=5000,
    eval_interval=10000,
    device='cuda'
)
```

### 3. 推理超参数

```python
optimizer = DerivativeFreeOptimizer(
    bounds=[[-1, -1], [1, 1]],  # 动作空间边界
    num_samples=16384,          # 候选样本数（匹配 IBC: 2^14）
    num_iters=3,                # 优化迭代次数（匹配 IBC）
    noise_scale=0.33,           # 初始噪声标准差（匹配 IBC）
    noise_shrink=0.5,           # 噪声衰减率（匹配 IBC）
    device='cuda'
)
```

### 4. 超参数对比（与 IBC 官方）

| 超参数 | 本实现 | IBC 官方 | 匹配 |
|--------|-------|---------|------|
| obs_seq_len | 2 | 2 | ✓ |
| hidden_dim | 256 | 256 | ✓ |
| num_residual_blocks | 1 | depth=2 | ✓ |
| dropout | 0.0 | 0.0 | ✓ |
| norm_type | None | None | ✓ |
| batch_size | 512 | 512 | ✓ |
| learning_rate | 1e-3 | 1e-3 | ✓ |
| lr_decay_steps | 100 | 100 | ✓ |
| lr_decay_rate | 0.99 | 0.99 | ✓ |
| num_counter_examples | 8 | 8 | ✓ |
| ula_step_size | 0.1 | 0.1 | ✓ |
| ula_num_steps | 100 | 100 | ✓ |
| grad_margin | 1.0 | 1.0 | ✓ |
| grad_norm_type | 'inf' | 'inf' | ✓ |
| num_samples (推理) | 16384 | 16384 | ✓ |
| num_iters (推理) | 3 | 3 | ✓ |

**结论**: 本实现与 IBC 官方配置 **完全匹配**

---

## 附录

### A. 关键文件列表

```
IBC_ebm_dp/
├── core/
│   ├── models.py              # SequenceEBM, MLPResidualBlock
│   ├── optimizers.py          # ULASampler, DerivativeFreeOptimizer
│   └── trainer.py             # 训练逻辑（未使用）
├── scripts/_2d/particle/
│   ├── particle_train.py      # 训练脚本
│   ├── particle_test.py       # 推理脚本
│   ├── particle_data_generate.py  # 数据生成
│   └── md/
│       ├── TRAINING_ARCHITECTURE.md  # 本文档
│       ├── DATA_GENERATION_COMPARISON.md
│       ├── EVAL_ALIGNMENT_ANALYSIS.md
│       └── ...
├── data/_2d/particle/
│   ├── traj/                  # 轨迹 JSON 文件
│   └── pic/                   # 轨迹可视化图片
└── models/_2d/particle/
    ├── checkpoints/           # 训练检查点
    └── norm_params.json       # 归一化参数
```

### B. 训练命令

```bash
# 快速测试（10K 迭代）
python particle_train.py --num_iterations 10000

# 完整训练（100K 迭代）
python particle_train.py --num_iterations 100000

# 自定义配置
python particle_train.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --batch_size 512 \
    --num_iterations 100000 \
    --device cuda
```

### C. 推理命令

```bash
# 测试训练好的模型
python particle_test.py \
    --model_path models/_2d/particle/checkpoints/final_model.pth \
    --num_test_episodes 100 \
    --visualize
```

### D. 参考文献

1. **Implicit Behavioral Cloning (IBC)**
   - Paper: Florence et al., "Implicit Behavioral Cloning", CoRL 2021
   - Code: https://github.com/google-research/ibc

2. **Energy-Based Models**
   - LeCun et al., "A Tutorial on Energy-Based Learning", 2006
   - Du & Mordatch, "Implicit Generation and Modeling with Energy-Based Models", NeurIPS 2019

3. **Langevin Dynamics**
   - Welling & Teh, "Bayesian Learning via Stochastic Gradient Langevin Dynamics", ICML 2011

4. **Contrastive Learning**
   - Oord et al., "Representation Learning with Contrastive Predictive Coding", 2018

---

## 版本信息

- **文档版本**: v1.0
- **创建日期**: 2025-11-25
- **作者**: VP-STO Team
- **对应代码**: `particle_train.py` (commit: latest)
- **对齐基准**: IBC 官方实现 (TensorFlow)

---

## 常见问题

### Q1: 为什么使用能量基模型而不是直接回归？

**A**: 
- **多模态建模**: 能量函数可以建模多峰分布（多个好的动作）
- **隐式学习**: 不需要显式建模概率分布，更灵活
- **更好的泛化**: 对比学习有助于学习更鲁棒的策略

### Q2: InfoNCE 损失与标准监督学习的区别？

**A**:
- **监督学习**: 直接拟合目标 `L = ||π(s) - a*||^2`
- **InfoNCE**: 对比正负样本 `L = -log(p(a*|s) / Σp(a|s))`
- **优势**: InfoNCE 更关注相对排序，而不是绝对值

### Q3: 为什么推理需要 16384 个样本？

**A**:
- **全局优化**: 需要覆盖整个动作空间
- **避免局部最优**: 更多样本增加找到全局最优的概率
- **IBC 经验**: 16384 是 IBC 论文中经验调优的结果

### Q4: 如何加速推理？

**A**:
1. 减少样本数: `num_samples = 4096`（精度略降）
2. 减少迭代: `num_iters = 2`（精度略降）
3. 使用 GPU: 并行计算所有样本
4. 模型量化: INT8 量化（速度提升 2-4×）

### Q5: 梯度惩罚的作用是什么？

**A**:
- **稳定训练**: 防止梯度爆炸
- **改善能量景观**: 让能量函数更平滑
- **提升采样效率**: 更平滑的能量函数更容易优化

---

**文档结束**

