# PyTorch 实现与 IBC 官方的关键差异

## 发现的问题

虽然配置参数看起来一致，但在实现细节上存在**关键差异**，这些差异可能导致效果截然不同。

---

## 差异 1: 梯度符号 ⚠️ **CRITICAL**

### IBC 官方 (`mcmc.py` 第 188-191 行)

```python
def gradient_wrt_act(energy_network, observations, actions, ...):
    with tf.GradientTape() as g:
        g.watch(actions)
        energies, _ = energy_network((observations, actions), ...)
    # My energy sign is flipped relative to Igor's code,
    # so -1.0 here.
    denergies_dactions = g.gradient(energies, actions) * -1.0  # ← 乘以 -1.0
    return denergies_dactions, energies
```

**关键注释："My energy sign is flipped relative to Igor's code, so -1.0 here"**

这意味着：
- IBC 计算的是 **`-dE/da`**
- 更新方向：`actions = actions - delta` 其中 `delta = stepsize * (-dE/da + noise)`
- 实际上是：`actions = actions + stepsize * (dE/da - noise)`

###  ULASampler (`optimizers.py` 第 434-456 行)

```python
# 计算梯度
grad = torch.autograd.grad(
    energies.sum(),
    samples,
    create_graph=False
)[0]  # ← 这是 dE/da（正梯度）

# ULA 更新
noise = torch.randn_like(samples) * self.noise_scale
delta = current_step_size * (0.5 * grad + noise)  # ← 使用正梯度
samples = samples - delta  # ← 减去delta
```

**分析：**
- PyTorch 计算的是 **`dE/da`**（正梯度）
- 更新：`samples = samples - stepsize * (0.5 * dE/da + noise)`
- 这与 IBC 的符号是**相反的**！

### 正确的梯度方向

**能量基模型的目标：找到低能量的动作**

IBC 的更新（按照第190行的注释理解）：
```
de_dact = -dE/da  # 负梯度
delta = stepsize * (0.5 * (-dE/da) + noise)
actions = actions - delta
      = actions - stepsize * (0.5 * (-dE/da) + noise)
      = actions + stepsize * (0.5 * dE/da - noise)
```

等等，这里有歧义。让我重新分析第248-259行：
```python
gradient_scale = 0.5
de_dact = (gradient_scale * l_lambda * de_dact +  # de_dact 已经是 -dE/da
           tf.random.normal(tf.shape(actions)) * l_lambda * noise_scale)
delta_actions = stepsize * de_dact
actions = actions - delta_actions  # 减去 delta
```

所以 IBC 的更新是：
```
de_dact_scaled = 0.5 * (-dE/da) + noise
actions = actions - stepsize * de_dact_scaled
        = actions - stepsize * (0.5 * (-dE/da) + noise)
        = actions + stepsize * (0.5 * dE/da - noise)
```

**这意味着 IBC 沿着正梯度方向移动（增加能量），加上噪声！**

这看起来不对...让我重新理解。实际上，IBC 的注释说"My energy sign is flipped"，这可能意味着：
- IBC 的网络输出是 **负能量**（即 logits）
- 因此 `dE/da` 实际上是 `d(-logits)/da = -d(logits)/da`
- 需要再乘以 -1 来得到 `d(logits)/da`

**结论：需要仔细验证 SequenceEBM 输出的是能量还是 logits！**

---

## 差异 2: delta_action_clip ⚠️ **IMPORTANT**

### IBC 官方 (`mcmc.py` 第 236-254 行)

```python
def langevin_step(..., delta_action_clip, ...):
    # This effectively scales the gradient as if the actions were
    # in a min-max range of -1 to 1.
    delta_action_clip = delta_action_clip * 0.5*(max_actions - min_actions)  # ← 缩放
    
    unclipped_de_dact = de_dact * 1.0
    grad_norms = compute_grad_norm(grad_norm_type, unclipped_de_dact)
    
    if grad_clip is not None:
        de_dact = tf.clip_by_value(de_dact, -grad_clip, grad_clip)
    
    gradient_scale = 0.5
    de_dact = (gradient_scale * l_lambda * de_dact +
               tf.random.normal(tf.shape(actions)) * l_lambda * noise_scale)
    delta_actions = stepsize * de_dact
    
    # Clip to box.
    delta_actions = tf.clip_by_value(delta_actions, -delta_action_clip,
                                     delta_action_clip)  # ← 裁剪 delta
    
    actions = actions - delta_actions
    actions = tf.clip_by_value(actions, min_actions, max_actions)
```

**关键点：**
- `delta_action_clip` 默认值：**0.1**
- 对于 `[-1, 1]` 范围：`delta_action_clip = 0.1 * 0.5 * (1 - (-1)) = 0.1 * 1.0 = 0.1`
- 每步的动作变化被限制在 **±0.1** 范围内

### PyTorch ULASampler (`optimizers.py` 第 445-462 行)

```python
def sample(...):
    # ULA 更新：匹配 IBC 逻辑
    noise = torch.randn_like(samples) * self.noise_scale
    delta = current_step_size * (0.5 * grad + noise)
    samples = samples - delta
    
    # 限制在边界内
    samples = samples.clamp(
        min=bounds_tensor[0, :],
        max=bounds_tensor[1, :]
    )  # ← 只裁剪最终结果，没有裁剪 delta！
```

**关键缺失：**
- ❌ **没有 `delta_action_clip`**
- ❌ **没有限制每步的变化量**
- 只裁剪最终位置到边界内

**影响：**
- PyTorch 实现中，每步可能跳得太远（特别是在初始步长0.1时）
- IBC 通过 `delta_action_clip=0.1` 确保每步最多移动 0.1
- 这会导致采样轨迹完全不同！

---

## 差异 3: 梯度裁剪 ⚠️

### IBC 官方
```python
if grad_clip is not None:
    de_dact = tf.clip_by_value(de_dact, -grad_clip, grad_clip)
```
- 默认 `grad_clip=None`（没有梯度裁剪）

### PyTorch ULASampler
```python
# 没有实现 grad_clip
```
- ✅ 这个差异影响较小（因为默认不裁剪）

---

## 差异 4: 噪声处理细节

### IBC 官方
```python
l_lambda = 1.0
de_dact = (gradient_scale * l_lambda * de_dact +
           tf.random.normal(tf.shape(actions)) * l_lambda * noise_scale)
```
- `l_lambda = 1.0`（恒定）
- `gradient_scale = 0.5`
- 噪声乘以 `l_lambda * noise_scale = 1.0 * 1.0 = 1.0`

### PyTorch ULASampler
```python
noise = torch.randn_like(samples) * self.noise_scale  # noise_scale = 1.0
delta = current_step_size * (0.5 * grad + noise)
```
- 噪声直接乘以 `noise_scale = 1.0`
- ✅ 这部分一致

---

## 差异 5: 步长调度

### IBC 官方 (`mcmc.py` 第 284-295 行)
```python
class PolynomialSchedule:
    def __init__(self, init, final, power, total_steps):
        self.init = init
        self.final = final
        self.power = power
        self.total_steps = total_steps
    
    def get_rate(self, step):
        if self.total_steps <= 1:
            return self.init
        progress = tf.minimum(1.0, tf.cast(step, tf.float32) / 
                            tf.cast(self.total_steps - 1, tf.float32))
        return ((self.init - self.final) *
                tf.pow((1.0 - progress), self.power) + self.final)
```

### PyTorch ULASampler (`optimizers.py` 第 367-380 行)
```python
def _get_step_size(self, step: int) -> float:
    if self.num_steps <= 1:
        return self.step_size
    
    progress = float(step) / float(self.num_steps - 1)
    rate = (self.step_size - self.step_size_final) * (
        (1.0 - progress) ** self.step_size_power
    ) + self.step_size_final
    return rate
```

- ✅ 公式完全一致
- ✅ 边界条件处理一致

---

## 总结：关键问题

### 🚨 **最严重的问题**

1. **缺少 `delta_action_clip`**
   - 影响：每步可能跳得太远，采样不稳定
   - 修复：添加 `delta_action_clip=0.1` 参数，限制每步变化量

2. **梯度符号可能不一致**
   - 需要验证：SequenceEBM 输出的是能量还是负能量（logits）
   - IBC 在梯度计算时乘以 -1.0，需要理解原因

### ⚠️ **次要问题**

3. **缺少 `grad_clip`**
   - 影响：较小（默认不使用）
   - 可选修复：添加梯度裁剪参数

---

## 修复建议

### 1. 修改 ULASampler 添加 delta_action_clip

```python
class ULASampler:
    def __init__(
        self,
        bounds: np.ndarray,
        step_size: float = 0.1,
        num_steps: int = 100,
        noise_scale: float = 1.0,
        step_size_final: float = 1e-5,
        step_size_power: float = 2.0,
        delta_action_clip: float = 0.1,  # ← 添加
        device: str = 'cpu'
    ):
        ...
        self.delta_action_clip = delta_action_clip
    
    def sample(self, ...):
        for step in range(self.num_steps):
            energies = ebm(x, samples)
            grad = torch.autograd.grad(energies.sum(), samples, ...)[0]
            
            # 计算 delta_action_clip（相对于动作范围）
            action_range = bounds_tensor[1, :] - bounds_tensor[0, :]
            delta_clip = self.delta_action_clip * 0.5 * action_range  # ← 添加
            
            with torch.no_grad():
                noise = torch.randn_like(samples) * self.noise_scale
                # 是否需要取负梯度？需要验证！
                de_dact = 0.5 * grad + noise  # 或 0.5 * (-grad) + noise
                delta = current_step_size * de_dact
                
                # 裁剪 delta（关键！）
                delta = delta.clamp(
                    min=-delta_clip,
                    max=delta_clip
                )  # ← 添加
                
                samples = samples - delta
                samples = samples.clamp(
                    min=bounds_tensor[0, :],
                    max=bounds_tensor[1, :]
                )
```

### 2. 验证梯度符号

检查训练损失是否正确下降：
- 如果损失持续上升或不收敛，可能是梯度符号问题
- 尝试在 ULA 更新中使用 `-grad` 代替 `grad`

### 3. 调试建议

在训练开始时打印：
```python
print(f"第一步 Langevin 采样:")
print(f"  初始能量: {energies[0].mean():.4f}")
print(f"  最终能量: {energies[-1].mean():.4f}")
print(f"  能量变化: {(energies[-1] - energies[0]).mean():.4f}")
```

期望：
- 能量应该下降（负变化）
- 如果能量上升，说明梯度方向错误

---

## 实验建议

1. **首先修复 `delta_action_clip`**
   - 这是最明显的差异
   - 应该能显著改善稳定性

2. **验证梯度符号**
   - 监控 Langevin 采样过程中的能量变化
   - 确认能量是下降的

3. **对比训练曲线**
   - 如果修复后效果仍差，检查其他训练超参数
   - 例如：批次大小、数据增强等

---

## 结论

**主要问题不在配置参数，而在实现细节！**

关键差异：
1. ❌ 缺少 `delta_action_clip`（每步变化量限制）
2. ⚠️ 梯度符号可能不一致（需要验证）

修复这两个问题后，效果应该会显著改善。

