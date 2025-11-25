# Particle 推理对齐检查

## IBC 官方推理流程

### 1. IbcPolicy._distribution() 流程

1. **初始化动作样本**：
   - 从 `action_sampling_spec` 中采样 `batch_size * num_action_samples` 个初始动作
   - 使用 `tensor_spec.sample_spec_nest` 从均匀分布采样

2. **Langevin MCMC 优化**：
   ```python
   action_samples = mcmc.langevin_actions_given_obs(
       self._actor_network,
       maybe_tiled_obs,
       action_samples,
       num_action_samples=self._num_action_samples,
       min_actions=self._action_sampling_spec.minimum,
       max_actions=self._action_sampling_spec.maximum,
       training=False,
       noise_scale=1.0
   )
   ```

3. **计算概率**：
   ```python
   probs = mcmc.get_probabilities(
       self._actor_network,
       batch_size,
       self._num_action_samples,
       maybe_tiled_obs,
       action_samples,
       training=False
   )
   ```
   - `get_probabilities` 的实现：
     ```python
     net_logits, _ = energy_network((observations, actions), training=training)
     net_logits = tf.reshape(net_logits, (batch_size, num_action_samples))
     probs = tf.nn.softmax(net_logits / temperature, axis=1)  # temperature=1.0
     probs = tf.reshape(probs, (-1,))
     ```

4. **创建分布并返回**：
   ```python
   distribution = MappedCategorical(probs=probs, mapped_values=action_samples)
   return policy_step.PolicyStep(distribution, policy_state)
   ```

### 2. GreedyPolicy 选择动作

IBC 使用 `GreedyPolicy` 包装 `IbcPolicy`：
```python
policy = greedy_policy.GreedyPolicy(collect_policy)
```

`GreedyPolicy` 会调用 `distribution.mode()` 而不是 `distribution.sample()`：
- `MappedCategorical.mode()` 返回概率最高的动作（argmax）
- 等价于：`action = action_samples[argmax(probs)]`

### 3. 关键参数

- `num_action_samples = 512`（从配置文件中）
- `noise_scale = 1.0`（推理时）
- `temperature = 1.0`（在 `get_probabilities` 中，但默认不传入，所以使用默认值 1.0）

## 当前实现的问题

### 问题 1: 动作选择方式 ❌

**当前实现**（`particle_test.py` 第 187-190 行）：
```python
dist = torch.distributions.Categorical(probs)
sampled_idx = dist.sample().item()
next_action = candidates[0, sampled_idx].cpu().numpy()
```

**IBC 实现**：
- 使用 `GreedyPolicy`，调用 `distribution.mode()`
- 等价于：`sampled_idx = probs.argmax(dim=1).item()`

**修复**：应该使用 `argmax` 而不是 `sample()`

### 问题 2: 观测序列更新方式 ⚠️

**当前实现**（`particle_test.py` 第 252-274 行）：
- 假设动作直接更新位置：`new_pos = next_action`
- 使用差分计算速度：`vel = new_pos - current_pos`

**IBC 环境实际更新**（`particle.py` 第 171-186 行）：
- 动作是 `pos_setpoint`（位置设定点）
- 使用 PD 控制器：
  ```python
  u_agent = k_p * (action - pos_agent) - k_v * vel_agent
  new_pos = pos_agent + vel_agent * dt
  new_vel = vel_agent + u_agent * dt
  ```
- 参数：`k_p=10.0`, `k_v=5.0`, `dt=0.005`, `repeat_actions=10`

**影响**：
- 当前简化版本可能不够准确
- 但完全模拟环境需要知道所有参数
- 对于测试目的，简化版本可能足够

### 问题 3: 温度参数 ✅

**当前实现**：`temperature=1.0`（正确）

**IBC 实现**：`get_probabilities` 默认 `temperature=1.0`（正确）

## 建议的修复

### 修复 1: 使用 argmax 选择动作

```python
# 当前（错误）：
dist = torch.distributions.Categorical(probs)
sampled_idx = dist.sample().item()

# 修复（正确）：
sampled_idx = probs.argmax(dim=1).item()  # 匹配 GreedyPolicy.mode()
```

### 修复 2: 考虑环境模拟（可选）

如果需要完全匹配 IBC 的行为，应该使用 PD 控制器更新观测：
```python
# 使用 PD 控制器更新位置和速度
k_p = 10.0
k_v = 5.0
dt = 0.005

u_agent = k_p * (next_action - current_pos) - k_v * current_vel
new_pos = current_pos + current_vel * dt
new_vel = current_vel + u_agent * dt
```

但对于测试目的，简化版本（直接使用动作作为新位置）可能足够。

