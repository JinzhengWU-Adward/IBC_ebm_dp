# 数据生成脚本对比分析

## 对比：`particle_data_generate.py` vs `collect_data.sh` (IBC 官方)

### 1. 策略配置

#### IBC 官方 (`collect_data.sh` → `policy_eval.py`)
```python
# policy_eval.py 第 305 行
policy = particle_oracles.ParticleOracle(env)
# 使用默认参数：
# - wait_at_first_goal = 1 (默认)
# - multimodal = False (默认)
# - goal_threshold = 0.01 (默认)
```

#### PyTorch 实现 (`particle_data_generate.py`)
```python
# 第 233-238 行
oracle = particle_oracles.ParticleOracle(
    env,
    wait_at_first_goal=1,
    multimodal=False,
    goal_threshold=0.01
)
```

**结论**：✅ **一致** - 策略参数完全相同

---

### 2. 环境配置

#### IBC 官方 (`policy_eval.py`)
```python
# 第 140 行
env = suite_gym.load('Particle-v0')
# 使用 gin 配置或默认配置
# 默认参数（从 particle.py 定义）：
# - n_steps = 50
# - n_dim = 2
# - hide_velocity = False
# - seed = None (通过 gin 配置)
# - dt = 0.005
# - repeat_actions = 10
# - k_p = 10.0
# - k_v = 5.0
# - goal_distance = 0.05
```

#### PyTorch 实现 (`particle_data_generate.py`)
```python
# 第 217-227 行
gym_env = particle.ParticleEnv(
    n_steps=n_steps,          # 50
    n_dim=n_dim,              # 2
    hide_velocity=False,
    seed=seed,                # 0 (固定种子) ⚠️ 差异
    dt=0.005,
    repeat_actions=10,
    k_p=10.0,
    k_v=5.0,
    goal_distance=0.05
)
```

**结论**：⚠️ **部分一致** - 除了 `seed` 参数外，其他参数完全一致
- **问题**：`particle_data_generate.py` 使用 `seed=0`（固定种子），而 IBC 官方使用 `seed=None`（随机种子）
- **影响**：固定种子会导致每次生成的数据完全相同，缺乏多样性

---

### 3. Episode 数量

#### IBC 官方 (`collect_data.sh`)
```bash
--num_episodes=200
--replicas=10
# 总共：200 * 10 = 2000 个 episodes
```

#### PyTorch 实现 (`particle_data_generate.py`)
```python
# 第 305 行
num_episodes=1000,  # 默认值
# 总共：1000 个 episodes
```

**结论**：⚠️ **不一致** - Episode 数量不同
- IBC 官方：2000 个 episodes（200 × 10 replicas）
- PyTorch 实现：1000 个 episodes

---

### 4. 数据格式

#### IBC 官方 (`collect_data.sh`)
```python
# policy_eval.py 第 332-337 行
observers.append(
    example_encoding_dataset.TFRecordObserver(
        dataset_path,
        policy.collect_data_spec,
        py_mode=True,
        compress_image=True))
# 输出：TFRecord 格式（.tfrecord 文件）
```

#### PyTorch 实现 (`particle_data_generate.py`)
```python
# 第 262-284 行
json_data = {
    'sample_id': ...,
    'start_position': ...,
    'target_position': ...,
    'first_goal_position': ...,
    'second_goal_position': ...,
    'trajectory': {...},
    'actions': ...,
    'config': {...}
}
# 输出：JSON 格式（.json 文件）和 PNG 图像
```

**结论**：⚠️ **格式不同** - 但这是预期的，因为 PyTorch 实现需要 JSON 格式

---

### 5. 环境包装

#### IBC 官方 (`policy_eval.py`)
```python
# 第 140 行
env = suite_gym.load('Particle-v0')
# 如果 history_length 设置，会添加 HistoryWrapper
if history_length:
    env = wrappers.HistoryWrapper(
        env, history_length=history_length, tile_first_step_obs=True)
```

#### PyTorch 实现 (`particle_data_generate.py`)
```python
# 第 230 行
env = suite_gym.wrap_env(gym_env)
# 没有 HistoryWrapper（因为我们在训练时自己构建序列）
```

**结论**：✅ **功能一致** - 都使用 `suite_gym` 包装，只是包装方式略有不同

---

## 关键问题总结

### 🔴 严重问题

1. **Seed 参数不一致**
   - **IBC 官方**：`seed=None`（随机，每次 reset 生成不同的目标点）
   - **PyTorch 实现**：`seed=0`（固定，每次 reset 生成相同的目标点）
   - **影响**：固定种子会导致数据缺乏多样性，可能影响模型泛化能力

### ⚠️ 需要注意的问题

2. **Episode 数量不同**
   - **IBC 官方**：2000 个 episodes
   - **PyTorch 实现**：1000 个 episodes
   - **影响**：数据量较少，但可以通过增加 `num_episodes` 参数调整

### ✅ 已正确实现

3. **策略参数**：完全一致
4. **环境参数**：除 seed 外完全一致
5. **数据内容**：观测、动作、轨迹结构一致

---

## 修复建议

### 修复 1：将 seed 改为 None（推荐）

```python
# particle_data_generate.py 第 309 行
generate_particle_dataset(
    output_dir=str(output_dir),
    num_episodes=1000,
    n_dim=2,
    n_steps=50,
    image_size=64,
    seed=None  # 改为 None，与 IBC 官方一致
)
```

### 修复 2：增加 Episode 数量（可选）

```python
# particle_data_generate.py 第 305 行
generate_particle_dataset(
    output_dir=str(output_dir),
    num_episodes=2000,  # 增加到 2000，匹配 IBC 官方
    n_dim=2,
    n_steps=50,
    image_size=64,
    seed=None
)
```

---

## 验证方法

生成数据后，检查：
1. 目标点是否随机分布（如果 seed=None）
2. 轨迹是否多样化（不同的起点、目标点组合）
3. 数据量是否足够（建议至少 2000 个 episodes）

