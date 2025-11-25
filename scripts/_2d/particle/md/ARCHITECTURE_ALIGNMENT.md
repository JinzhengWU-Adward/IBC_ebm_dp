# 架构对齐验证：SequenceEBM vs IBC 官方 MLPEBM

## 1. Pre-Activation 顺序验证

### IBC 官方 ResNetPreActivationLayer (resnet.py:135-153)
```python
def call(self, x, training):
    x = self._projection_layer(x)
    
    for l in range(len(self._weight_layers)):
        x_start_block = tf.identity(x)
        
        # 第一层：Pre-Activation 顺序
        if self.normalizer is not None:
            x = self._norm_layers[l](x, training=training)
        x = self._activation_layers[l](x, training=training)
        x = self._dropouts[l](x, training=training)
        x = self._weight_layers[l](x, training=training)
        
        # 第二层：Pre-Activation 顺序
        if self.normalizer is not None:
            x = self._norm_layers_2[l](x, training=training)
        x = self._activation_layers_2[l](x, training=training)
        x = self._dropouts_2[l](x, training=training)
        x = self._weight_layers_2[l](x, training=training)
        
        x = x_start_block + x
    return x
```

### PyTorch MLPResidualBlock (models.py:396-421)
```python
def forward(self, x):
    identity = x
    
    # 第一层：Pre-Activation 顺序
    if self.norm1 is not None:
        out = self.norm1(x)
    else:
        out = x
    out = self.activation(out)
    out = self.dropout1(out)
    out = self.fc1(out)
    
    # 第二层：Pre-Activation 顺序
    if self.norm2 is not None:
        out = self.norm2(out)
    out = self.activation(out)
    out = self.dropout2(out)
    out = self.fc2(out)
    
    return out + identity
```

**✅ 验证通过**：顺序完全一致

---

## 2. norm=None 时行为验证

### IBC 官方 (resnet.py:76-79, 94-95, 141-142, 147-148)
```python
# 初始化时
elif self.normalizer is None:
    pass  # 不创建 norm 层

# forward 时
if self.normalizer is not None:
    x = self._norm_layers[l](x, training=training)
# 如果 normalizer=None，这个 if 块不执行，直接跳过
```

### PyTorch (models.py:385-388, 409-410, 415-416)
```python
# 初始化时
elif norm_type is None or norm_type.lower() == 'none':
    self.norm1 = None  # 不创建 norm 层
    self.norm2 = None

# forward 时
if self.norm1 is not None:
    out = self.norm1(x)
else:
    out = x  # 直接跳过
```

**✅ 验证通过**：当 norm=None 时，都不创建 norm 层，在 forward 中直接跳过

---

## 3. Projection Placement 验证

### IBC 官方 (mlp_ebm.py:72-88, resnet.py:136)
```python
# MLPEBM.call()
x = tf.concat([obs, act], -1)  # 拼接
x = self._mlp(x, training=training)  # 传入 ResNetPreActivationLayer

# ResNetPreActivationLayer.call()
x = self._projection_layer(x)  # 投影层在 ResNet 内部
# ... 残差块 ...
```

### PyTorch (models.py:502-510)
```python
# SequenceEBM.forward()
x = torch.cat([obs_expanded, action], dim=-1)  # 拼接
x = x.reshape(B * N, -1)
x = self.projection(x)  # 投影层在 SequenceEBM 中（等价于在 ResNet 内部）
# ... 残差块 ...
```

**✅ 验证通过**：投影层位置一致（都在进入残差块之前）

---

## 4. Block 内层数量验证

### IBC 官方 (resnet.py:69-104)
```python
# depth=2 时
hidden_sizes = [width, width]  # [256, 256]

# 每个残差块包含 2 个 Dense 层
for l in range(0, len(hidden_sizes), 2):  # l=0
    self._weight_layers.append(make_weight_fn(hidden_sizes[0]))  # Dense 1
    self._weight_layers_2.append(make_weight_fn(hidden_sizes[1]))  # Dense 2
    # 每个残差块：2 个 Dense 层
```

### PyTorch (models.py:391-392)
```python
# num_residual_blocks=1 时
self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # Dense 1
self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Dense 2
# 每个残差块：2 个 Dense 层
```

**✅ 验证通过**：每个残差块都包含 2 个 Dense 层

---

## 5. 完整架构流程对比

### IBC 官方完整流程
```
1. MLPEBM.call():
   - obs: [B, T, obs_dim] -> flatten -> [B, T*obs_dim]
   - concat([obs, act]) -> [B, T*obs_dim + action_dim]
   - x = self._mlp(x)  # ResNetPreActivationLayer

2. ResNetPreActivationLayer.call():
   - x = self._projection_layer(x)  # [B, T*obs_dim + action_dim] -> [B, hidden_dim]
   - for each residual block:
     - x_start = x
     - x = norm(x) if norm exists else x
     - x = relu(x)
     - x = dropout(x)
     - x = dense1(x)
     - x = norm(x) if norm exists else x
     - x = relu(x)
     - x = dropout(x)
     - x = dense2(x)
     - x = x_start + x
   - return x

3. MLPEBM.call() (继续):
   - x = self._project_energy(x)  # [B, hidden_dim] -> [B, 1]
   - x = tf.squeeze(x, axis=-1)  # [B, 1] -> [B]
   - return x
```

### PyTorch 完整流程
```
1. SequenceEBM.forward():
   - obs_seq: [B, T, obs_dim] -> reshape -> [B, T*obs_dim]
   - obs_expanded: [B, N, T*obs_dim]
   - concat([obs_expanded, action]) -> [B, N, T*obs_dim + action_dim]
   - reshape -> [B*N, T*obs_dim + action_dim]
   - x = self.projection(x)  # [B*N, T*obs_dim + action_dim] -> [B*N, hidden_dim]

2. MLPResidualBlock.forward() (for each block):
   - identity = x
   - x = norm(x) if norm exists else x
   - x = relu(x)
   - x = dropout(x)
   - x = fc1(x)
   - x = norm(x) if norm exists else x
   - x = relu(x)
   - x = dropout(x)
   - x = fc2(x)
   - x = identity + x
   - return x

3. SequenceEBM.forward() (继续):
   - x = self.energy_projection(x)  # [B*N, hidden_dim] -> [B*N, 1]
   - return x.view(B, N)  # [B*N, 1] -> [B, N]
```

**✅ 验证通过**：完整流程一致

---

## 6. 配置参数对应关系

| IBC 官方配置 | PyTorch 实现 | 说明 |
|------------|------------|------|
| `MLPEBM.width = 256` | `hidden_dim=256` | 隐藏层维度 |
| `MLPEBM.depth = 2` | `num_residual_blocks=1` | depth=2 → 1 个残差块 |
| `MLPEBM.rate = 0.0` | `dropout=0.0` | Dropout 率 |
| `ResNetLayer.normalizer = None` | `norm_type=None` | 不使用归一化 |
| `MLPEBM.activation = 'relu'` | `nn.ReLU()` | 激活函数 |

**✅ 验证通过**：参数对应关系正确

---

## 总结

所有关键点都已对齐：
1. ✅ Pre-activation 顺序完全一致
2. ✅ norm=None 时行为一致（不创建层，直接跳过）
3. ✅ Projection placement 一致（都在进入残差块之前）
4. ✅ Block 内层数量一致（每个块 2 个 Dense 层）

