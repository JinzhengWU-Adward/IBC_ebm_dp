# 评估实现一致性分析

## 1. IBC 官方评估流程 (`eval_random_goals.sh` -> `policy_eval.py`)

### 1.1 环境交互流程
- 使用 `PyDriver` 运行环境
- 环境在 `reset()` 时初始化 `obs_log` 和 `act_log`
- 每次 `step(action)` 后：
  - `act_log.append({'pos_setpoint': action})`
  - 执行 PD 控制器更新位置和速度
  - `obs_log.append(obs)` （更新后的观测）

### 1.2 轨迹记录
- `obs_log` 长度 = 1 (初始) + n_steps (每次动作后) = n_steps + 1
- `act_log` 长度 = n_steps
- `obs_log[i]['pos_agent']` 是第 i 步后的位置
- `act_log[i]['pos_setpoint']` 是第 i 步的动作（目标位置）

### 1.3 可视化方式 (`particle_viz.py`)
```python
# 处理长度不匹配
if len(obs_log) != len(act_log):
    if len(obs_log) == len(act_log) + 1:
        obs_log_ = obs_log[:-1]  # 去掉最后一个观测

# 绘制轨迹（连接相邻位置）
for i in range(len(obs_log_)-1):
    pos_agent_k = obs_log_[i]['pos_agent']
    pos_agent_kplus1 = obs_log_[i+1]['pos_agent']
    ax.plot([pos_agent_k, pos_agent_kplus1], ...)  # 连接线段
```

**关键点**：
- `true_trajectory` 是从 `obs_log` 中提取的 `pos_agent` 序列
- 轨迹绘制使用 `obs_log[i]` 和 `obs_log[i+1]` 之间的线段
- 如果 `obs_log` 比 `act_log` 多 1，则去掉最后一个观测（因为最后一个观测没有对应的动作）

## 2. 我们的实现 (`particle_test.py`)

### 2.1 数据加载
- 从 JSON 文件加载 `positions` 和 `actions`
- `positions` 应该对应 `obs_log` 中的 `pos_agent` 序列
- `actions` 应该对应 `act_log` 中的 `pos_setpoint` 序列

### 2.2 数据生成 (`particle_data_generate.py`)
```python
# 在循环中记录位置
while not time_step.is_last() and step_count < max_steps:
    obs = time_step.observation
    pos_agent = obs['pos_agent']
    positions.append(pos_agent.copy())  # 记录当前观测的位置
    
    action = oracle.action(time_step).action
    actions.append(action.copy())
    
    time_step = env.step(action)  # 执行动作，更新环境
    
    step_count += 1

# 添加最后一步的位置
if not time_step.is_last() and len(positions) > 0:
    final_obs = time_step.observation
    positions.append(final_obs['pos_agent'].copy())
```

**问题分析**：
1. 在循环开始时记录位置，然后执行动作
2. 这意味着 `positions[0]` 是初始位置（reset 后的位置）
3. `positions[i]` 是第 i 步动作执行**之前**的位置
4. 最后添加的位置是最后一步动作执行**之后**的位置

**与 IBC 的差异**：
- IBC 的 `obs_log[0]` 是 reset 后的初始位置
- IBC 的 `obs_log[i+1]` 是第 i 步动作执行**之后**的位置
- 我们的 `positions[0]` 是初始位置（✓ 一致）
- 我们的 `positions[i+1]` 是第 i 步动作执行**之后**的位置（✓ 一致）

**结论**：数据生成逻辑应该是正确的，`positions` 应该对应 `obs_log` 中的 `pos_agent` 序列。

### 2.3 可视化实现 (`particle_test.py`)
```python
# 获取真实轨迹
true_positions = episode_data['positions']  # (T, 2)

# 可视化
ax1.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'g-', ...)
```

**问题**：
- 我们直接使用 `true_positions` 绘制轨迹
- 但 IBC 的可视化会处理长度不匹配（如果 `obs_log` 比 `act_log` 多 1，去掉最后一个）
- 我们的 `positions` 长度应该 = `actions` 长度 + 1（如果最后一步也被记录）

### 2.4 推理流程
```python
# 从 episode 开始构建初始观测序列
obs_seq_raw = np.stack([obs_vec_0, obs_vec_1])  # (2, 8)
obs_seq_norm = (obs_seq_raw - dataset.obs_mean) / dataset.obs_std

# 推理轨迹
pred_trajectory_norm, intermediate_states = infer_trajectory(...)

# 从 intermediate_states 构建预测位置序列
pred_positions = []
pred_positions.append(start_pos.copy())  # 起始位置

for state in intermediate_states[1:]:
    obs_seq = state['obs_seq'][0].cpu().numpy()
    last_obs = obs_seq[-1]
    pos_agent_norm = last_obs[:2]
    pos_agent_orig = pos_agent_norm * dataset.obs_std[:2] + dataset.obs_mean[:2]
    pred_positions.append(pos_agent_orig)
```

**问题分析**：
1. 我们使用 `start_pos` 作为起始位置（对应 `true_positions[0]`）
2. 从 `intermediate_states` 中提取每个步骤后的位置
3. 这应该与 IBC 的 `obs_log` 记录方式一致

## 3. 潜在问题

### 3.1 轨迹长度不匹配
- 如果 `true_positions` 的长度 = `actions` 长度 + 1，这是正常的
- 但可视化时，我们需要确保 `true_trajectory` 和 `pred_trajectory` 的长度一致

### 3.2 初始观测序列构建
- 我们使用 `true_positions[0]` 和 `true_positions[1]` 来构建初始观测序列
- 但 `true_positions[1]` 是第 0 步动作执行后的位置
- 在推理时，我们应该从 `true_positions[0]` 开始，然后预测第 0 步的动作，然后更新到 `true_positions[1]`
- 但我们的实现是：从 `true_positions[0]` 和 `true_positions[1]` 构建初始观测序列，然后预测第 1 步的动作

**这可能是一个问题**：我们的初始观测序列包含了未来信息（`true_positions[1]`），这不应该用于推理。

### 3.3 与 IBC 官方评估的差异
- IBC 使用 `PyDriver` 运行环境，环境自动记录 `obs_log` 和 `act_log`
- 我们是从 JSON 文件加载预记录的轨迹
- 在推理时，IBC 是实时与环境交互，我们是从预记录的数据中提取初始状态

## 4. 修复方案

### 4.1 初始观测序列构建 ✅ 已修复
**问题**：之前使用 `true_positions[0]` 和 `true_positions[1]` 构建初始观测序列，但 `true_positions[1]` 是未来信息（第 0 步动作执行后的位置），不应该用于推理。

**修复**：只使用 `true_positions[0]` 来构建初始观测序列，与 IBC 官方评估一致：
```python
# 只使用初始位置（reset 后的观测）
pos_agent_0 = true_positions[0].astype(np.float32)
vel_agent_0 = np.zeros(2, dtype=np.float32)  # reset 后初始速度为 0

obs_vec_0 = np.concatenate([
    pos_agent_0,                           # pos_agent
    vel_agent_0,                           # vel_agent (初始为 0)
    pos_first_goal.astype(np.float32),     # pos_first_goal
    pos_second_goal.astype(np.float32)     # pos_second_goal
])

# 如果 obs_seq_len=2，第二个观测使用第一个（历史堆叠）
# 这与 IBC 的 HistoryWrapper 行为一致
obs_vec_1 = obs_vec_0.copy()
```

**理由**：
- IBC 使用 `PyDriver` 从环境 `reset()` 后的初始观测开始推理
- 初始观测中，`pos_agent` 是 reset 后的位置，`vel_agent` 是 0
- 如果 `history_length=2`，`HistoryWrapper` 会在第一步时重复初始观测

### 4.2 轨迹对齐
确保 `true_trajectory` 和 `pred_trajectory` 的长度一致：
```python
# true_positions 长度 = n_steps + 1 (初始 + n_steps 动作后)
# pred_positions 长度 = 1 (初始) + n_steps (每次动作后) = n_steps + 1
# 应该是一致的
```

### 4.3 可视化一致性
确保可视化方式与 IBC 一致：
```python
# 如果 positions 长度 = actions 长度 + 1，这是正常的
# 可视化时，应该绘制 positions[i] 到 positions[i+1] 的线段
# 对应第 i 步动作执行后的轨迹段
```

