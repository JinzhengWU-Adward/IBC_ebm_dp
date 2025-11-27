"""
使用 PyTorch 复刻 IBC 的 Particle 环境训练
基于 run_mlp_ebm_langevin.sh 的配置
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import sys

# 添加项目路径（与 A2B_train.py 保持一致）
# particle_train.py 位于: IBC_ebm_dp/scripts/_2d/particle/particle_train.py
# 需要找到 IBC_ebm_dp 目录
# 路径层级: particle_train.py -> particle -> _2d -> scripts -> IBC_ebm_dp
IBC_ROOT = Path(__file__).parent.parent.parent.parent  # IBC_ebm_dp
sys.path.insert(0, str(IBC_ROOT))

from core.models import SequenceEBM
from core.optimizers import ULASampler, TFExponentialDecay


class ParticleDataset(Dataset):
    """
    Particle 环境数据集
    从 JSON 文件加载轨迹数据
    """
    
    def __init__(
        self,
        data_dir: str,
        obs_seq_len: int = 2,
        normalize_actions: bool = True
    ):
        """
        Args:
            data_dir: 数据目录，包含 traj/ 和 pic/ 子目录
            obs_seq_len: 观测序列长度
            normalize_actions: 是否归一化动作到 [-1, 1]
        """
        self.data_dir = Path(data_dir)
        self.obs_seq_len = obs_seq_len
        self.normalize_actions = normalize_actions
        
        # 加载所有 JSON 文件
        traj_dir = self.data_dir / 'traj'
        self.json_files = sorted(traj_dir.glob('traj_*.json'))
        
        print(f"找到 {len(self.json_files)} 个轨迹文件")
        
        # 加载所有数据并计算归一化参数
        self.episodes = []
        all_actions = []
        all_obs = []
        
        for json_file in tqdm(self.json_files, desc="加载数据"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            positions = np.array(data['trajectory']['positions'], dtype=np.float32)
            actions = np.array(data['actions'], dtype=np.float32)
            first_goal = np.array(data['first_goal_position'], dtype=np.float32)
            second_goal = np.array(data['second_goal_position'], dtype=np.float32)
            # 第三个目标：优先从 JSON 中读取 third_goal_position，
            # 若不存在则回退为 second_goal（兼容旧数据）
            third_goal_key = 'third_goal_position'
            if third_goal_key in data:
                third_goal = np.array(data[third_goal_key], dtype=np.float32)
            else:
                third_goal = second_goal.copy()
            
            # 构建观测：对于 3-goals Particle 环境，观测包含
            #   pos_agent, vel_agent, pos_first_goal, pos_second_goal, pos_third_goal
            # 展平后顺序为:
            #   [pos_agent(2), vel_agent(2), first_goal(2), second_goal(2), third_goal(2)]
            # 维度：2 + 2 + 2 + 2 + 2 = 10 (hide_velocity=False)
            
            # 直接从 JSON 读取速度（如果可用），否则通过差分计算
            if 'velocities' in data['trajectory']:
                # ✅ 使用保存的真实速度（匹配 IBC 环境）
                velocities = np.array(data['trajectory']['velocities'], dtype=np.float32)
            else:
                # ⚠️ 降级方案：通过位置差分计算速度（仅用于旧数据）
                print(f"警告: {json_file.name} 没有速度信息，使用差分近似")
                velocities = np.zeros_like(positions)
                if len(positions) > 1:
                    velocities[1:] = positions[1:] - positions[:-1]
                    # 第一个速度设为 0（匹配 IBC 环境重置行为）
                    velocities[0] = np.zeros_like(positions[0])
            
            # ===== 先添加首帧重复的样本：[x1, x1, ..., x1] -> a1 =====
            if len(positions) >= self.obs_seq_len:
                first_obs = np.concatenate([
                    positions[0],   # pos_agent (2D)
                    velocities[0],  # vel_agent (2D)
                    first_goal,     # pos_first_goal (2D)
                    second_goal,    # pos_second_goal (2D)
                    third_goal      # pos_third_goal (2D)
                ])
                first_obs_seq = np.stack(
                    [first_obs for _ in range(self.obs_seq_len)],
                    axis=0
                )  # (obs_seq_len, obs_dim)
                first_action = actions[0]

                self.episodes.append({
                    'obs_seq': first_obs_seq.astype(np.float32),
                    'action': first_action.astype(np.float32)
                })

                all_actions.append(first_action)
                all_obs.extend(first_obs_seq)

            # ===== 继续原来的滑动窗口样本构造 =====
            for i in range(len(positions) - 1):
                if i + self.obs_seq_len <= len(positions):
                    # 构建观测序列
                    obs_seq = []
                    for j in range(self.obs_seq_len):
                        idx = i + j
                        if idx < len(positions):
                            # 观测：pos_agent, vel_agent, pos_first_goal, pos_second_goal, pos_third_goal
                            obs = np.concatenate([
                                positions[idx],  # pos_agent (2D)
                                velocities[idx], # vel_agent (2D)
                                first_goal,      # pos_first_goal (2D)
                                second_goal,     # pos_second_goal (2D)
                                third_goal       # pos_third_goal (2D)
                            ])
                            obs_seq.append(obs)
                    
                    if len(obs_seq) == self.obs_seq_len:
                        # 标签使用窗口最后一步的动作，匹配 IBC 中 action[:, -1]
                        action_idx = i + self.obs_seq_len - 1
                        action = actions[action_idx]  # 窗口末尾对应的动作
                        
                        self.episodes.append({
                            'obs_seq': np.array(obs_seq, dtype=np.float32),
                            'action': action.astype(np.float32)
                        })
                        
                        all_actions.append(action)
                        all_obs.extend(obs_seq)
        
        print(f"共加载 {len(self.episodes)} 个样本")
        
        # 计算归一化参数
        all_actions = np.array(all_actions)
        all_obs = np.array(all_obs)
        
        if self.normalize_actions:
            # Min-Max 归一化到 [-1, 1]
            # IBC 使用 min_max_actions=True
            self.action_min = all_actions.min(axis=0)
            self.action_max = all_actions.max(axis=0)
            self.action_range = self.action_max - self.action_min
            self.action_range[self.action_range < 1e-6] = 1.0  # 避免除零
            
            print(f"动作范围: min={self.action_min}, max={self.action_max}")
        else:
            self.action_min = None
            self.action_max = None
            self.action_range = None
        
        # 观测归一化（使用 Z-score 归一化，匹配 IBC）
        self.obs_mean = all_obs.mean(axis=0)
        self.obs_std = all_obs.std(axis=0)
        self.obs_std[self.obs_std < 1e-6] = 1.0  # 避免除零
        
        print(f"观测均值: {self.obs_mean}")
        print(f"观测标准差: {self.obs_std}")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        obs_seq = episode['obs_seq'].copy()
        action = episode['action'].copy()
        
        # 归一化观测（Z-score）
        obs_seq = (obs_seq - self.obs_mean) / self.obs_std
        
        # 归一化动作（Min-Max 到 [-1, 1]）
        if self.normalize_actions:
            action = 2.0 * (action - self.action_min) / self.action_range - 1.0
        
        return {
            'obs_seq': torch.FloatTensor(obs_seq),
            'action': torch.FloatTensor(action)
        }


def compute_info_nce_loss(
    energy_positives: torch.Tensor,  # (B, 1) 正样本能量
    energy_negatives: torch.Tensor,  # (B, num_negatives) 负样本能量
    temperature: float = 1.0
) -> torch.Tensor:
    """
    计算 InfoNCE 损失
    
    Args:
        energy_positives: 正样本的能量值，形状 (B, 1)
        energy_negatives: 负样本的能量值，形状 (B, num_negatives)
        temperature: softmax 温度参数
    
    Returns:
        loss: 标量损失值
    """
    # 拼接正负样本能量: (B, num_negatives + 1)
    # 正样本在最后一列（匹配 IBC 的实现）
    energies = torch.cat([energy_negatives, energy_positives], dim=1)  # (B, num_negatives + 1)
    
    # Softmax（能量越低，概率越高）
    probs = F.softmax(-energies / temperature, dim=-1)  # (B, num_negatives + 1)
    
    # 标签：正样本在最后一列（索引 num_negatives）
    num_negatives = energy_negatives.size(1)
    labels = torch.zeros(energies.size(0), dtype=torch.long, device=energies.device)
    labels.fill_(num_negatives)  # 正样本索引
    
    # 交叉熵损失（等价于 KL 散度）
    loss = F.cross_entropy(-energies / temperature, labels, reduction='mean')
    
    return loss


def compute_gradient_penalty(
    model: nn.Module,
    obs_seq: torch.Tensor,  # (B, obs_seq_len, obs_dim)
    actions: torch.Tensor,  # (B, num_samples, action_dim) 或 (B*num_samples, action_dim)
    grad_margin: float = 1.0,
    grad_norm_type: str = 'inf',
    square_grad_penalty: bool = True,
    grad_loss_weight: float = 1.0
) -> torch.Tensor:
    """
    计算梯度惩罚损失（匹配 IBC 的实现）
    
    Args:
        model: EBM 模型
        obs_seq: 观测序列 (B, obs_seq_len, obs_dim)
        actions: 动作样本，形状可以是 (B, num_samples, action_dim) 或 (B*num_samples, action_dim)
        grad_margin: 梯度 margin（默认 1.0）
        grad_norm_type: 梯度范数类型 ('inf', '1', '2')
        square_grad_penalty: 是否对梯度惩罚进行平方
        grad_loss_weight: 梯度损失权重
    
    Returns:
        grad_loss: 标量梯度惩罚损失
    """
    # 确保 actions 需要梯度
    if actions.dim() == 3:
        # (B, num_samples, action_dim) -> (B*num_samples, action_dim)
        B, num_samples, action_dim = actions.shape
        actions_flat = actions.view(-1, action_dim)
        # 扩展 obs_seq 以匹配 actions
        obs_seq_expanded = obs_seq.unsqueeze(1).expand(B, num_samples, -1, -1)  # (B, num_samples, obs_seq_len, obs_dim)
        obs_seq_flat = obs_seq_expanded.reshape(-1, obs_seq.size(1), obs_seq.size(2))  # (B*num_samples, obs_seq_len, obs_dim)
    else:
        # (B*num_samples, action_dim)
        actions_flat = actions
        B = obs_seq.size(0)
        num_samples = actions_flat.size(0) // B
        obs_seq_expanded = obs_seq.unsqueeze(1).expand(B, num_samples, -1, -1)
        obs_seq_flat = obs_seq_expanded.reshape(-1, obs_seq.size(1), obs_seq.size(2))
    
    # 确保 actions 需要梯度
    actions_flat = actions_flat.detach().requires_grad_(True)
    
    # 计算能量对动作的梯度
    # 注意：IBC 使用 -1.0 乘以梯度（见 mcmc.py 第 190 行）
    energies = model(obs_seq_flat, actions_flat.unsqueeze(1))  # (B*num_samples, 1)
    energies = energies.squeeze(-1)  # (B*num_samples,)
    
    # 计算梯度：dE/da
    # IBC 使用 -1.0 乘以梯度，但这里我们直接计算，然后在范数计算时处理
    grad_outputs = torch.ones_like(energies)
    de_dact = torch.autograd.grad(
        outputs=energies,
        inputs=actions_flat,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (B*num_samples, action_dim)
    
    # 计算梯度范数
    if grad_norm_type == 'inf':
        # L-infinity 范数：每行的最大绝对值
        grad_norms = torch.norm(de_dact, p=float('inf'), dim=1)  # (B*num_samples,)
    elif grad_norm_type == '1':
        grad_norms = torch.norm(de_dact, p=1, dim=1)  # (B*num_samples,)
    elif grad_norm_type == '2':
        grad_norms = torch.norm(de_dact, p=2, dim=1)  # (B*num_samples,)
    else:
        raise ValueError(f"Unsupported grad_norm_type: {grad_norm_type}")
    
    # 重塑为 (B, num_samples)
    grad_norms = grad_norms.view(B, -1)  # (B, num_samples)
    
    # 应用 margin：max(0, ||dE/da|| - grad_margin)
    if grad_margin is not None:
        grad_norms = grad_norms - grad_margin
        grad_norms = torch.clamp(grad_norms, min=0.0, max=1e10)
    
    # 平方（如果启用）
    if square_grad_penalty:
        grad_norms = grad_norms ** 2
    
    # 平均：对每个样本的所有负样本求平均，然后对所有批次样本求平均
    grad_loss = grad_norms.mean()  # 标量
    
    return grad_loss * grad_loss_weight


def train_particle_ebm(
    data_dir: str,
    output_dir: str,
    obs_seq_len: int = 2,
    hidden_dim: int = 256,
    num_residual_blocks: int = 1,  # depth=2 意味着 1 个残差块
    dropout: float = 0.0,
    norm_type: str = None,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    num_iterations: int = 100000,
    num_counter_examples: int = 8,
    ula_step_size: float = 0.1,
    ula_num_steps: int = 100,
    ula_noise_scale: float = 1.0,
    ula_step_size_final: float = 1e-5,
    ula_step_size_power: float = 2.0,
    temperature: float = 1.0,
    lr_decay_steps: int = 100,  # 匹配 IBC 的默认值（虽然 gin 文件中未显式设置，但 train_eval.py 默认值是 100）
    lr_decay_rate: float = 0.99,
    add_grad_penalty: bool = True,  # 是否添加梯度惩罚
    grad_margin: float = 1.0,  # 梯度 margin
    grad_norm_type: str = 'inf',  # 梯度范数类型
    square_grad_penalty: bool = True,  # 是否对梯度惩罚进行平方
    grad_loss_weight: float = 1.0,  # 梯度损失权重
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_interval: int = 5000,
    eval_interval: int = 10000,
    eval_episodes: int = 20
):
    """
    训练 Particle EBM 模型
    
    参数与 IBC 的 mlp_ebm_langevin.gin 配置对齐
    """
    print("=" * 60)
    print("训练 Particle EBM 模型")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")
    print()
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 加载数据集
    print("加载数据集...")
    dataset = ParticleDataset(data_dir, obs_seq_len=obs_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # 计算观测维度（展平后的观测序列）
    obs_dim = dataset.episodes[0]['obs_seq'].shape[-1]  # 单个观测的维度（8 = pos_agent(2) + vel_agent(2) + pos_first_goal(2) + pos_second_goal(2)）
    action_dim = dataset.episodes[0]['action'].shape[-1]  # 动作维度（2）
    
    print(f"观测维度: {obs_dim}")
    print(f"动作维度: {action_dim}")
    print(f"观测序列长度: {obs_seq_len}")
    print()
    
    # 创建模型
    print("创建模型...")
    model = SequenceEBM(
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_seq_len=obs_seq_len,
        hidden_dim=hidden_dim,
        num_residual_blocks=num_residual_blocks,
        dropout=dropout,
        norm_type=norm_type
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 创建优化器（使用指数衰减学习率，匹配 IBC）
    # 注意：TensorFlow Adam 的默认 epsilon=1e-7，而 PyTorch 是 1e-8
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    
    # 学习率调度器（使用 TFExponentialDecay，完全匹配 TensorFlow 的 ExponentialDecay）
    # IBC 使用: ExponentialDecay(learning_rate, decay_steps=100, decay_rate=0.99)
    # 公式: lr = initial_lr * (decay_rate ^ (step / decay_steps))
    # 注意：IBC 的默认 decay_steps=100，不是 10000
    scheduler = TFExponentialDecay(
        optimizer,
        decay_rate=lr_decay_rate,
        decay_steps=lr_decay_steps,
        staircase=False  # 连续衰减，不是阶梯式
    )
    
    # 创建 ULA 采样器（用于生成负样本）
    # 动作范围：在归一化空间 [-1, 1] 的基础上做 uniform_boundary_buffer 外扩
    # 对齐 gin 中的: train_eval.uniform_boundary_buffer = 0.05
    uniform_boundary_buffer = 0.05
    # 归一化动作的理论范围为 [-1, 1]，外扩后的范围为 [-1 - 0.1, 1 + 0.1] = [-1.1, 1.1]
    norm_min, norm_max = -1.0, 1.0
    expanded_min = norm_min - uniform_boundary_buffer * (norm_max - norm_min) * 2.0
    expanded_max = norm_max + uniform_boundary_buffer * (norm_max - norm_min) * 2.0
    action_bounds = np.array([[expanded_min, expanded_min],
                              [expanded_max, expanded_max]], dtype=np.float32)  # (2, action_dim)
    ula_sampler = ULASampler(
        bounds=action_bounds,
        step_size=ula_step_size,
        num_steps=ula_num_steps,
        noise_scale=ula_noise_scale,
        step_size_final=ula_step_size_final,
        step_size_power=ula_step_size_power,
        delta_action_clip=0.1,  # 匹配 IBC 的 delta_action_clip（关键修复！）
        device=device
    )
    
    # 训练循环
    print("开始训练...")
    print()
    
    model.train()
    global_step = 0
    epoch = 0
    
    # 保存归一化参数
    norm_params = {
        'obs_mean': dataset.obs_mean.tolist(),
        'obs_std': dataset.obs_std.tolist(),
        'action_min': dataset.action_min.tolist() if dataset.action_min is not None else None,
        'action_max': dataset.action_max.tolist() if dataset.action_max is not None else None,
        'action_range': dataset.action_range.tolist() if dataset.action_range is not None else None,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'obs_seq_len': obs_seq_len
    }
    
    with open(output_path / 'norm_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)
    
    pbar = tqdm(total=num_iterations, desc="训练进度")
    
    while global_step < num_iterations:
        epoch += 1
        
        for batch in dataloader:
            if global_step >= num_iterations:
                break
            
            obs_seq = batch['obs_seq'].to(device)  # (B, obs_seq_len, obs_dim)
            actions = batch['action'].to(device)  # (B, action_dim)
            
            B = obs_seq.size(0)
            
            # 计算正样本能量
            actions_expanded = actions.unsqueeze(1)  # (B, 1, action_dim)
            energy_positives = model(obs_seq, actions_expanded)  # (B, 1)
            
            # 生成负样本（使用 ULA）
            # 初始化：从均匀分布采样（使用带 uniform_boundary_buffer 的外扩范围）
            neg_min = float(action_bounds[0, 0])
            neg_max = float(action_bounds[1, 0])
            init_negatives = torch.rand(
                B, num_counter_examples, action_dim,
                device=device
            ) * (neg_max - neg_min) + neg_min  # 范围 [expanded_min, expanded_max]
            
            # ULA 采样
            negatives, _ = ula_sampler.sample(
                x=obs_seq,
                ebm=model,
                num_samples=num_counter_examples,
                init_samples=init_negatives,
                return_trajectory=False
            )
            
            # 断开负样本的梯度（匹配 IBC 的 stop_chain_grad=True）
            negatives = negatives.detach()
            
            # 计算负样本能量
            energy_negatives = model(obs_seq, negatives)  # (B, num_counter_examples)
            
            # 计算 InfoNCE 损失
            info_nce_loss = compute_info_nce_loss(
                energy_positives,
                energy_negatives,
                temperature=temperature
            )
            
            # 计算梯度惩罚损失（如果启用）
            if add_grad_penalty:
                # 合并正样本和负样本用于梯度惩罚
                # IBC 使用 combined_true_counter_actions，包括正样本和负样本
                combined_actions = torch.cat([
                    negatives,  # (B, num_counter_examples, action_dim)
                    actions.unsqueeze(1)  # (B, 1, action_dim)
                ], dim=1)  # (B, num_counter_examples + 1, action_dim)
                
                grad_loss = compute_gradient_penalty(
                    model=model,
                    obs_seq=obs_seq,
                    actions=combined_actions,
                    grad_margin=grad_margin,
                    grad_norm_type=grad_norm_type,
                    square_grad_penalty=square_grad_penalty,
                    grad_loss_weight=grad_loss_weight
                )
                
                # 总损失 = InfoNCE 损失 + 梯度惩罚损失
                loss = info_nce_loss + grad_loss
            else:
                loss = info_nce_loss
                grad_loss = None
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            # 更新进度条
            pbar.update(1)
            postfix_dict = {
                'loss': f'{loss.item():.4f}',
                'info_nce': f'{info_nce_loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            }
            if add_grad_penalty and grad_loss is not None:
                postfix_dict['grad_penalty'] = f'{grad_loss.item():.4f}'
            pbar.set_postfix(postfix_dict)
            
            # 保存检查点
            if global_step % save_interval == 0:
                checkpoint = {
                    'global_step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                    'obs_dim': obs_dim,
                    'action_dim': action_dim,
                    'obs_seq_len': obs_seq_len,
                    'hidden_dim': hidden_dim,
                    'num_residual_blocks': num_residual_blocks,
                    'dropout': dropout,
                    'norm_type': norm_type,
                    'norm_params': norm_params
                }
                checkpoint_path = checkpoint_dir / f'checkpoint_{global_step:06d}.pth'
                torch.save(checkpoint, checkpoint_path)
                print(f"\n保存检查点: {checkpoint_path}")
            
            # 评估（简化版，仅打印损失）
            if global_step % eval_interval == 0:
                model.eval()
                eval_losses = []
                for i, eval_batch in enumerate(dataloader):
                    if i >= eval_episodes:
                        break
                    eval_obs_seq = eval_batch['obs_seq'].to(device)
                    eval_actions = eval_batch['action'].to(device)
                    eval_B = eval_obs_seq.size(0)
                    
                    # 计算正样本能量（不需要梯度）
                    with torch.no_grad():
                        eval_actions_expanded = eval_actions.unsqueeze(1)
                        eval_energy_positives = model(eval_obs_seq, eval_actions_expanded)
                    
                    # ULA 采样需要梯度，所以不能放在 no_grad 中
                    eval_init_negatives = torch.rand(
                        eval_B, num_counter_examples, action_dim,
                        device=device
                    ) * (neg_max - neg_min) + neg_min
                    
                    # ULA 采样（需要梯度）
                    with torch.enable_grad():
                        eval_negatives, _ = ula_sampler.sample(
                            x=eval_obs_seq,
                            ebm=model,
                            num_samples=num_counter_examples,
                            init_samples=eval_init_negatives,
                            return_trajectory=False
                        )
                    
                    # 计算负样本能量（不需要梯度）
                    with torch.no_grad():
                        eval_negatives = eval_negatives.detach()
                        eval_energy_negatives = model(eval_obs_seq, eval_negatives)
                        
                        eval_loss = compute_info_nce_loss(
                            eval_energy_positives,
                            eval_energy_negatives,
                            temperature=temperature
                        )
                        eval_losses.append(eval_loss.item())
                
                avg_eval_loss = np.mean(eval_losses)
                print(f"\n[评估] Step {global_step}, 平均损失: {avg_eval_loss:.4f}")
                model.train()
    
    pbar.close()
    
    # 保存最终模型
    final_checkpoint = {
        'global_step': global_step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'obs_seq_len': obs_seq_len,
        'hidden_dim': hidden_dim,
        'num_residual_blocks': num_residual_blocks,
        'dropout': dropout,
        'norm_type': norm_type,
        'norm_params': norm_params
    }
    final_path = checkpoint_dir / 'final_model.pth'
    torch.save(final_checkpoint, final_path)
    print(f"\n保存最终模型: {final_path}")
    print("训练完成！")


if __name__ == '__main__':
    import argparse
    
    # 获取 IBC_ebm_dp 根目录（用于默认路径）
    IBC_ROOT = Path(__file__).parent.parent.parent.parent  # IBC_ebm_dp
    
    parser = argparse.ArgumentParser(description='训练 Particle EBM 模型')
    parser.add_argument('--data_dir', type=str, 
                       default=str(IBC_ROOT / 'data' / '_2d' / 'particle3goals'),
                       help='数据目录')
    parser.add_argument('--output_dir', type=str,
                       default=str(IBC_ROOT / 'models' / '_2d' / 'particle3goals'),
                       help='输出目录')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='批次大小')
    parser.add_argument('--num_iterations', type=int, default=10000,
                       help='训练迭代次数')
    
    args = parser.parse_args()
    
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_particle_ebm(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        obs_seq_len=2,
        hidden_dim=256,
        num_residual_blocks=1,  # depth=2 意味着 1 个残差块
        dropout=0.0,
        norm_type=None,
        batch_size=args.batch_size,
        learning_rate=1e-3,
        num_iterations=args.num_iterations,
        num_counter_examples=8,
        ula_step_size=0.1,
        ula_num_steps=100,
        ula_noise_scale=1.0,
        ula_step_size_final=1e-5,
        ula_step_size_power=2.0,
        temperature=1.0,
        lr_decay_steps=100,  # 匹配 IBC 的默认值（虽然 gin 文件中未显式设置，但 train_eval.py 默认值是 100）
        lr_decay_rate=0.99,
        add_grad_penalty=True,  # 匹配 IBC 配置（grad_penalty.grad_margin = 1.0）
        grad_margin=1.0,  # 匹配 IBC 配置
        grad_norm_type='inf',  # 匹配 IBC 配置（ImplicitBCAgent 默认使用 'inf'）
        square_grad_penalty=True,  # 匹配 IBC 配置
        grad_loss_weight=1.0,  # 梯度损失权重
        device=device,
        save_interval=5000,
        eval_interval=10000,
        eval_episodes=20
    )

