# Particle EBM 训练脚本对齐检查清单

## 检查顺序建议

### 1. 配置文件参数（`ibc/ibc/configs/particle/mlp_ebm_langevin.gin`）

#### 1.1 训练超参数
- [x] `batch_size = 512` ✅
- [x] `num_iterations = 100000` ✅
- [x] `learning_rate = 1e-3` ✅
- [ ] `decay_steps = 100` ❌ **当前使用 10000，应该改为 100**
- [x] `decay_rate = 0.99` ✅ (在 ExponentialDecay 中)
- [x] `sequence_length = 2` ✅
- [x] `eval_interval = 10000` ✅
- [x] `eval_episodes = 20` ✅

#### 1.2 网络架构参数
- [x] `MLPEBM.width = 256` ✅
- [x] `MLPEBM.depth = 2` ✅ (对应 1 个残差块)
- [x] `MLPEBM.rate = 0.0` ✅ (dropout)
- [x] `ResNetLayer.normalizer = None` ✅
- [x] `MLPEBM.activation = 'relu'` ✅

#### 1.3 EBM 训练参数
- [x] `ImplicitBCAgent.num_counter_examples = 8` ✅
- [x] `langevin_actions_given_obs.num_iterations = 100` ✅
- [x] `IbcPolicy.num_action_samples = 512` ✅ (用于推理)
- [x] `temperature = 1.0` ✅ (softmax_temperature)

#### 1.4 数据归一化参数
- [x] `compute_dataset_statistics.min_max_actions = True` ✅
- [x] `get_normalizers.nested_obs = True` ✅
- [x] `get_normalizers.num_samples = 5000` ✅

#### 1.5 梯度惩罚参数 ✅ **已实现**
- [x] `grad_penalty.grad_margin = 1.0` ✅
- [x] `grad_penalty.only_apply_final_grad_penalty = True` ✅ (在 `compute_gradient_penalty` 中实现)
- [x] `ImplicitBCAgent.run_full_chain_under_gradient = True` ✅ (通过 `create_graph=True` 实现)
- [x] `ImplicitBCAgent.add_grad_penalty = True` ✅

#### 1.6 其他参数
- [x] `ImplicitBCAgent.compute_mse = True` ✅ (仅用于监控，不影响训练)
- [x] `ImplicitBCAgent.ebm_loss_type = 'info_nce'` ✅

---

### 2. 学习率调度器（`ibc/ibc/train/get_agent.py`）

#### 2.1 IBC 实现
```python
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate, decay_steps=decay_steps, decay_rate=0.99)
```

**公式**: `lr = initial_lr * (decay_rate ^ (step / decay_steps))`

#### 2.2 当前实现
```python
def lr_lambda(step):
    return lr_decay_rate ** (step / lr_decay_steps)
```

**问题**:
- ❌ `decay_steps = 10000` 应该是 `100`
- ⚠️ 公式可能不完全一致（需要验证）

---

### 3. 损失函数（`ibc/ibc/agents/ibc_agent.py`）

#### 3.1 InfoNCE 损失 ✅
- 实现正确：`F.cross_entropy(-energies / temperature, labels)`

#### 3.2 梯度惩罚损失 ❌ **缺失**
IBC 的损失函数包含两部分：
1. InfoNCE 损失
2. 梯度惩罚损失（gradient penalty）

**IBC 实现**:
```python
if self._add_grad_penalty:
    grad_loss = gradient_loss.grad_penalty(
        self.cloning_network,
        self._grad_norm_type,
        batch_size,
        chain_data,
        maybe_tiled_obs,
        combined_true_counter_actions,
        training,
    )
    per_example_loss += grad_loss
```

**梯度惩罚公式**:
- 计算能量对动作的梯度：`dE/da`
- 计算梯度范数：`||dE/da||`
- 应用 margin：`max(0, ||dE/da|| - grad_margin)`
- 平方（如果 `square_grad_penalty=True`）：`(max(0, ||dE/da|| - 1.0))^2`
- 平均：`mean((max(0, ||dE/da|| - 1.0))^2)`

---

### 4. ULA 采样器（`ibc/ibc/agents/mcmc.py`）

#### 4.1 参数 ✅
- [x] `step_size = 0.1` ✅
- [x] `num_steps = 100` ✅
- [x] `noise_scale = 1.0` ✅
- [x] `step_size_final = 1e-5` ✅
- [x] `step_size_power = 2.0` ✅

#### 4.2 更新公式 ✅
- 已对齐到 IBC 的非标准公式

#### 4.3 梯度处理 ✅
- [x] `negatives.detach()` ✅ (对应 `stop_chain_grad=True`)

---

### 5. 数据归一化（`ibc/ibc/train/stats.py`）

#### 5.1 观测归一化 ✅
- Z-score 归一化：`(obs - mean) / std`
- 使用 5000 个样本计算统计量

#### 5.2 动作归一化 ✅
- Min-Max 归一化到 `[-1, 1]`：`2 * (action - min) / (max - min) - 1`

---

### 6. 网络架构（`ibc/networks/mlp_ebm.py`）

#### 6.1 ResNetPreActivation ✅
- [x] `width = 256` ✅
- [x] `depth = 2` (1 个残差块) ✅
- [x] `dropout = 0.0` ✅
- [x] `normalizer = None` ✅
- [x] `activation = 'relu'` ✅

---

## 关键差异总结

### ✅ 已修复的问题

1. **学习率衰减步数错误** ✅
   - 已修复：`decay_steps = 100`（匹配 IBC 默认值）

2. **缺失梯度惩罚损失** ✅
   - 已实现：`compute_gradient_penalty` 函数
   - 匹配 IBC 的梯度惩罚公式：`mean((max(0, ||dE/da|| - grad_margin))^2)`
   - 使用 L-infinity 范数（`grad_norm_type='inf'`）
   - 在训练循环中集成，总损失 = InfoNCE 损失 + 梯度惩罚损失

3. **梯度计算逻辑** ✅
   - 使用 `torch.autograd.grad` 计算能量对动作的梯度
   - 通过 `create_graph=True` 确保梯度可以反向传播
   - 对正样本和负样本都计算梯度惩罚（匹配 IBC 的 `combined_true_counter_actions`）

### ⚠️ 需要验证的问题

1. **学习率调度公式**
   - IBC: `lr = initial_lr * (0.99 ^ (step / 100))`
   - 当前: `lr = initial_lr * (0.99 ^ (step / 10000))`
   - 需要确认 PyTorch 的 `ExponentialLR` 是否与 TensorFlow 的 `ExponentialDecay` 行为一致

2. **MSE 计算**
   - IBC 配置中 `ImplicitBCAgent.compute_mse = True`
   - 当前实现未计算 MSE（仅用于监控，不影响训练）

---

## 已完成的修复

1. ✅ **第一步**：修复学习率衰减步数（已完成）
2. ✅ **第二步**：实现梯度惩罚损失（已完成）
3. ⚠️ **第三步**：验证学习率调度公式的一致性（需要测试验证）
4. ⚠️ **第四步**：添加 MSE 监控（可选，不影响训练）

---

## 参考文件

- 配置文件: `ibc/ibc/configs/particle/mlp_ebm_langevin.gin`
- 训练入口: `ibc/ibc/train_eval.py`
- Agent 实现: `ibc/ibc/agents/ibc_agent.py`
- 梯度惩罚: `ibc/ibc/losses/gradient_loss.py`
- MCMC 采样: `ibc/ibc/agents/mcmc.py`
- 学习率调度: `ibc/ibc/train/get_agent.py`

