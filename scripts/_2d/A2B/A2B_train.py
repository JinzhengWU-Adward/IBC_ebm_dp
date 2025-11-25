"""
A2B 轨迹序列训练脚本
使用 IBC 风格的 EBM 模型学习从起点到终点的轨迹序列

任务：
- 输入: 图像 + 起点坐标
- 输出: 轨迹序列 (T, 2) - 从起点到终点的完整路径
- 模型: IBC 风格的 EBM
- 训练: InfoNCE 损失，使用 Derivative-Free Optimizer 生成负样本
"""
import numpy as np
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入核心模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.models import SequenceEBM
from core.optimizers import ULASampler

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = Path(__file__).parent.parent.parent / 'plots' / 'A2B_training'
output_dir.mkdir(parents=True, exist_ok=True)

# 创建模型保存目录
models_dir = Path(__file__).parent.parent.parent / 'models' / '_2d'
models_dir.mkdir(parents=True, exist_ok=True)


class A2BDataset(Dataset):
    """
    A2B 数据集：加载图像和轨迹序列
    """
    def __init__(self, data_dir, image_size=64, normalize_coords=True):
        """
        Args:
            data_dir: 数据目录路径（包含 pic/ 和 traj/ 文件夹）
            image_size: 图像尺寸（将图像resize到此大小）
            normalize_coords: 是否将坐标归一化到 [-1, 1]
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.normalize_coords = normalize_coords
        
        # 获取所有轨迹文件
        traj_dir = self.data_dir / 'traj'
        self.traj_files = sorted(list(traj_dir.glob('*.json')))
        
        # 加载所有数据
        self.data = []
        self.coord_bounds = None  # 用于归一化
        
        # 第一步：加载原始数据并计算统计信息
        raw_data_list = []
        all_coords_list = []
        
        print(f"加载数据从 {data_dir}...")
        for traj_file in tqdm(self.traj_files, desc="读取轨迹文件"):
            # 加载JSON
            with open(traj_file, 'r') as f:
                traj_data = json.load(f)
            
            # 获取对应的图片文件名
            timestamp = traj_file.stem.replace('traj_', '')
            img_file = self.data_dir / 'pic' / f'target_{timestamp}.png'
            
            if not img_file.exists():
                print(f"警告: 图片文件不存在: {img_file}")
                continue
            
            # 提取数据
            start_pos = np.array(traj_data['start_position'], dtype=np.float32)
            target_pos = np.array(traj_data['target_position'], dtype=np.float32)
            positions = np.array(traj_data['trajectory']['positions'], dtype=np.float32)
            
            raw_data_list.append({
                'image_path': str(img_file),
                'start_pos': start_pos,
                'target_pos': target_pos,
                'trajectory': positions,
                'trajectory_length': len(positions)
            })
            
            # 收集坐标用于统计
            all_coords_list.append(start_pos.reshape(1, -1))
            all_coords_list.append(target_pos.reshape(1, -1))
            all_coords_list.append(positions)

        if not raw_data_list:
            print("未找到有效数据")
            return
            
        # 计算坐标范围（从数据中）
        all_coords = np.concatenate(all_coords_list, axis=0)
        min_coords = all_coords.min(axis=0)
        max_coords = all_coords.max(axis=0)
        
        # 添加一点余量，防止边界值问题
        margin = 1e-5
        self.coord_bounds = np.array([
            min_coords - margin,
            max_coords + margin
        ], dtype=np.float32)
        
        print(f"数据统计范围:")
        print(f"  Min: {min_coords}")
        print(f"  Max: {max_coords}")
        print(f"  归一化边界: {self.coord_bounds.tolist()}")

        # 第二步：归一化并存储
        for item in raw_data_list:
            start_pos = item['start_pos']
            target_pos = item['target_pos']
            positions = item['trajectory']
            
            # 归一化坐标（如果需要）
            if self.normalize_coords:
                # 归一化到 [-1, 1]
                start_pos = self._normalize_coords(start_pos)
                target_pos = self._normalize_coords(target_pos)
                positions = np.array([self._normalize_coords(p) for p in positions])
            
            self.data.append({
                'image_path': item['image_path'],
                'start_pos': start_pos,
                'target_pos': target_pos,
                'trajectory': positions,  # (T, 2)
                'trajectory_length': item['trajectory_length']
            })
        
        print(f"成功加载 {len(self.data)} 条轨迹")
        if len(self.data) > 0:
            traj_lengths = [d['trajectory_length'] for d in self.data]
            print(f"轨迹长度范围: {min(traj_lengths)} - {max(traj_lengths)} 步")
    
    def _normalize_coords(self, coords):
        """将坐标归一化到 [-1, 1]"""
        if self.coord_bounds is None:
            return coords
        coords_norm = 2.0 * (coords - self.coord_bounds[0]) / (self.coord_bounds[1] - self.coord_bounds[0]) - 1.0
        return coords_norm
    
    def _denormalize_coords(self, coords_norm):
        """将归一化坐标还原"""
        if self.coord_bounds is None:
            return coords_norm
        coords = (coords_norm + 1.0) / 2.0 * (self.coord_bounds[1] - self.coord_bounds[0]) + self.coord_bounds[0]
        return coords
    
    def get_trajectory_bounds(self):
        """获取轨迹序列的边界（用于优化器）"""
        if len(self.data) == 0:
            return np.array([[-1.0, -1.0], [1.0, 1.0]])
        
        # 获取所有轨迹的最小/最大值
        all_trajectories = np.concatenate([d['trajectory'] for d in self.data], axis=0)
        min_coords = all_trajectories.min(axis=0)
        max_coords = all_trajectories.max(axis=0)
        
        # 添加一些边界
        margin = 0.1
        min_coords = np.maximum(min_coords - margin, -1.0)
        max_coords = np.minimum(max_coords + margin, 1.0)
        
        return np.array([min_coords, max_coords])
    
    def get_step_length_stats(self):
        """计算轨迹步长的统计信息（用于负样本采样）"""
        if len(self.data) == 0:
            return {'mean': 0.01, 'std': 0.005, 'median': 0.01}
        
        step_lengths = []
        for item in self.data:
            trajectory = item['trajectory']
            for i in range(len(trajectory) - 1):
                step_len = np.linalg.norm(trajectory[i+1] - trajectory[i])
                step_lengths.append(step_len)
        
        step_lengths = np.array(step_lengths)
        return {
            'mean': float(np.mean(step_lengths)),
            'std': float(np.std(step_lengths)),
            'median': float(np.median(step_lengths)),
            'min': float(np.min(step_lengths)),
            'max': float(np.max(step_lengths))
        }
    
    def get_trajectory_length(self):
        """获取轨迹长度（假设所有轨迹长度相同）"""
        if len(self.data) == 0:
            return 99  # 默认值
        lengths = [d['trajectory_length'] for d in self.data]
        # 返回最常见的长度
        from collections import Counter
        most_common_length = Counter(lengths).most_common(1)[0][0]
        return most_common_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像（保留作为context，但不再用于模型输入）
        img = Image.open(item['image_path']).convert('L')  # 转为灰度图
        img = img.resize((self.image_size, self.image_size))
        img_array = np.array(img, dtype=np.float32) / 255.0  # 归一化到 [0, 1]
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
        
        # 起点坐标
        start_pos = torch.from_numpy(item['start_pos']).float()
        
        # 轨迹序列
        trajectory = torch.from_numpy(item['trajectory']).float()  # (T, 2)
        
        return {
            'image': img_tensor,
            'start_pos': start_pos,
            'trajectory_2d': trajectory,  # (T, 2)
            'target_pos': torch.from_numpy(item['target_pos']).float()
        }


# 注意：这里不再需要SingleStepEBM，使用SequenceEBM替代


def train_a2b_ebm(
    data_dir: str,
    num_epochs: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_negatives: int = 8,  # ULA生成的负样本数量
    image_size: int = 64,
    trajectory_length: int = None,  # 如果为None，从数据集推断
    temperature: float = 1.0,  # 温度参数 (IBC 默认 1.0)
    ula_step_size: float = 0.1,  # ULA初始步长 (IBC 默认 0.1)
    ula_num_steps: int = 100,  # ULA迭代步数 (IBC 默认 100)
    hidden_dim: int = 256,  # 隐藏层维度
    num_residual_blocks: int = 1,  # 残差块数量 (MLPEBM.depth=2 -> 1 block)
    dropout: float = 0.0,  # Dropout概率
    norm_type: str = None,  # 归一化类型 (MLPEBM use None)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    训练 A2B 序列轨迹预测 EBM（使用观测序列和ULA采样）
    
    训练策略：
    - 输入：observation序列 [obs_{t-1}, obs_t]（即连续两个轨迹点）
    - 正样本：action_t（下一个轨迹点）
    - 负样本：8个通过ULA采样生成的action
    
    Args:
        data_dir: 数据目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        num_negatives: ULA生成的负样本数量
        image_size: 图像尺寸（保留用于可视化）
        trajectory_length: 轨迹长度（用于验证）
        temperature: 温度参数
        ula_step_size: ULA步长
        ula_num_steps: ULA迭代步数
        hidden_dim: 隐藏层维度
        num_residual_blocks: 残差块数量
        dropout: Dropout概率
        norm_type: 归一化类型
        device: 计算设备
    """
    print("=" * 60)
    print("A2B 序列轨迹预测 EBM 训练 (使用观测序列 + ULA)")
    print("=" * 60)
    
    # 1. 加载数据集
    print("\n1. 加载数据集...")
    dataset = A2BDataset(data_dir, image_size=image_size, normalize_coords=True)
    
    # 获取轨迹长度（如果未指定，从数据集推断）
    if trajectory_length is None:
        trajectory_length = dataset.get_trajectory_length()
    print(f"轨迹长度: {trajectory_length}")
    
    # 获取步长统计信息
    step_stats = dataset.get_step_length_stats()
    print(f"轨迹步长统计 (归一化空间):")
    print(f"  平均: {step_stats['mean']:.6f}")
    print(f"  标准差: {step_stats['std']:.6f}")
    print(f"  中位数: {step_stats['median']:.6f}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 获取轨迹边界（用于ULA采样）
    traj_bounds = dataset.get_trajectory_bounds()
    print(f"轨迹边界: {traj_bounds}")
    
    # 2. 创建模型
    print("\n2. 创建模型...")
    # 使用SequenceEBM（观测序列 + action）
    obs_dim = 2  # 每个observation是一个2D坐标
    action_dim = 2  # action也是2D坐标（下一个轨迹点）
    obs_seq_len = 2  # 使用连续两个observation [obs_{t-1}, obs_t]
    
    model = SequenceEBM(
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_seq_len=obs_seq_len,
        hidden_dim=hidden_dim,
        num_residual_blocks=num_residual_blocks,
        dropout=dropout,
        norm_type=norm_type
    )
    model = model.to(device)
    
    print(f"模型架构:")
    print(f"  输入维度: obs_seq_len={obs_seq_len}, obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  隐藏层维度: {hidden_dim}")
    print(f"  残差块数量: {num_residual_blocks}")
    print(f"  Dropout: {dropout}")
    
    # 3. 创建ULA采样器（用于生成负样本）
    print("\n3. 创建ULA采样器...")
    ula_sampler = ULASampler(
        bounds=traj_bounds,
        step_size=ula_step_size,
        num_steps=ula_num_steps,
        step_size_final=1e-5,  # IBC 默认值
        step_size_power=2.0,   # IBC 默认值
        noise_scale=1.0,       # IBC 默认值
        device=device
    )
    print(f"ULA参数: step_size={ula_step_size}, num_steps={ula_num_steps}")
    
    # 4. 创建优化器
    print("\n4. 创建优化器...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    print("\n5. 开始训练...")
    print(f"训练策略: 观测序列 [obs_{{t-1}}, obs_t] -> action_t")
    print(f"温度参数: {temperature} (越小能量越陡峭)")
    print(f"负样本数量: {num_negatives} (通过ULA采样生成)")
    
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            trajectories_2d = batch['trajectory_2d'].to(device)  # (B, T, 2)
            
            B, T, _ = trajectories_2d.shape
            
            # 构建训练样本：对于每个时间步 t (t >= 1)
            # 输入：observation序列 [obs_{t-1}, obs_t]
            # 正样本：action_t = obs_{t+1}
            
            # 收集所有训练样本
            all_obs_seq = []  # observation序列
            all_actions = []  # 正样本action
            
            # 遍历每个时间步（从t=1开始，因为需要obs_{t-1}，到T-2结束，因为需要obs_{t+1}）
            for t in range(1, T - 1):
                obs_seq = trajectories_2d[:, t-1:t+1, :]  # (B, 2, 2) - [obs_{t-1}, obs_t]
                action = trajectories_2d[:, t+1, :]  # (B, 2) - action_t = obs_{t+1}
                
                all_obs_seq.append(obs_seq)
                all_actions.append(action)
            
            # 合并所有时间步
            num_steps = T - 2  # 有效的训练步数
            if num_steps <= 0:
                continue  # 跳过太短的轨迹
            
            batch_obs_seq = torch.cat(all_obs_seq, dim=0)  # (B*num_steps, 2, 2)
            batch_actions = torch.cat(all_actions, dim=0)  # (B*num_steps, 2)
            
            # 使用ULA采样生成负样本
            # 为了使用ULA，我们需要创建一个包装函数
            # ULA采样器期望输入是 (x, ebm)，这里x是observation序列
            
            # 为每个observation序列生成负样本
            B_total = batch_obs_seq.size(0)
            
            # 创建包装模型，固定observation序列
            class EBMWrapper(nn.Module):
                def __init__(self, model, obs_seq):
                    super().__init__()
                    self.model = model
                    self.obs_seq = obs_seq
                
                def forward(self, x, y):
                    # x: 占位符（未使用）
                    # y: action候选 (B, N, action_dim)
                    return self.model(self.obs_seq, y)
            
            # 生成负样本
            # 关键：根据 IBC 的逻辑，即使 run_full_chain_under_gradient=True，
            # Langevin 内部仍然使用 stop_chain_grad=True（默认），
            # 这意味着负样本生成过程不会反向传播梯度到能量函数
            # 我们通过 detach 来实现相同的效果
            
            # 方法1：在梯度下运行（匹配 run_full_chain_under_gradient=True）
            # 但 ULA 内部会 detach，所以不会影响梯度
            model.eval()  # 暂时切换到eval模式用于ULA采样
            ebm_wrapper = EBMWrapper(model, batch_obs_seq)
            placeholder_x = torch.zeros(B_total, 1, device=device)
            
            # 关键修复：确保负样本生成不会影响能量函数的梯度
            # IBC 使用 stop_chain_grad=True，我们通过 detach 实现
            with torch.enable_grad():  # 允许计算能量梯度（用于 ULA 更新）
                negatives, _ = ula_sampler.sample(
                    placeholder_x,
                    ebm_wrapper,
                    num_samples=num_negatives,
                    init_samples=None,
                    return_trajectory=False
                )  # (B_total, num_negatives, 2)
            
            # 关键：detach 负样本，确保它们不会影响能量函数的梯度
            # 这匹配 IBC 的 stop_chain_grad=True 行为
            negatives = negatives.detach()
            
            model.train()  # 切换回训练模式
            
            # 合并正负样本
            positive_samples = batch_actions.unsqueeze(1)  # (B_total, 1, 2)
            all_samples = torch.cat([positive_samples, negatives], dim=1)  # (B_total, 1+num_negatives, 2)
            
            # 随机打乱
            permutation = torch.rand(B_total, all_samples.size(1), device=device).argsort(dim=1)
            all_samples = all_samples[
                torch.arange(B_total, device=device).unsqueeze(-1),
                permutation
            ]
            
            # 找到正样本的位置
            ground_truth = (permutation == 0).nonzero()[:, 1].to(device)
            
            # 计算能量
            energies = model(batch_obs_seq, all_samples)  # (B_total, 1+num_negatives)
            
            # InfoNCE 损失（带温度参数）
            logits = -energies / temperature
            loss = F.cross_entropy(logits, ground_truth)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("\n训练完成！")
    
    # 6. 保存模型
    model_path = models_dir / 'a2b_ebm_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'obs_seq_len': obs_seq_len,
        'hidden_dim': hidden_dim,
        'num_residual_blocks': num_residual_blocks,
        'dropout': dropout,
        'trajectory_length': trajectory_length,
        'image_size': image_size,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'num_negatives': num_negatives,
        'temperature': temperature,
        'ula_step_size': ula_step_size,
        'ula_num_steps': ula_num_steps,
        'traj_bounds': traj_bounds,
    }, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 7. 可视化训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve (Sequence-based Prediction with ULA)')
    plt.grid(True)
    plt.savefig(output_dir / 'training_loss.png')
    print(f"训练曲线已保存到: {output_dir / 'training_loss.png'}")
    
    return model, dataset, train_losses


def infer_trajectory(
    model: SequenceEBM,
    start_pos: np.ndarray,
    ula_sampler: ULASampler,
    max_steps: int = 100,
    num_action_samples: int = 512,  # IBC 默认值
    temperature: float = 1.0,  # IBC 默认值
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    使用ULA逐步推理轨迹序列
    
    关键改进：使用概率分布采样而不是直接选择能量最低的，避免陷入局部最优
    
    Args:
        model: 训练好的SequenceEBM模型
        start_pos: 起始位置，形状为 (2,) - 归一化坐标
        ula_sampler: ULASampler实例
        max_steps: 最大预测步数
        num_action_samples: 采样候选数量（IBC 默认 512）
        temperature: 温度参数（IBC 默认 1.0）
        device: 计算设备
    
    Returns:
        预测的轨迹序列，形状为 (T, 2)，包含起点 - 归一化坐标
    """
    model.eval()
    
    # 初始化轨迹，前两个点都是起点（因为需要observation序列）
    # 确保start_pos是1D数组 (2,)
    start_pos_flat = start_pos.flatten()
    trajectory = [start_pos_flat.copy(), start_pos_flat.copy()]
    
    with torch.no_grad():
        for step in range(max_steps):
            # 构建observation序列 [obs_{t-1}, obs_t]
            obs_seq = np.array([trajectory[-2], trajectory[-1]])  # (2, 2)
            obs_seq_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)  # (1, 2, 2)
            
            # 创建包装模型
            class EBMWrapper(nn.Module):
                def __init__(self, model, obs_seq):
                    super().__init__()
                    self.model = model
                    self.obs_seq = obs_seq
                
                def forward(self, x, y):
                    # x: 占位符
                    # y: action候选 (B, N, 2)
                    return self.model(self.obs_seq, y)
            
            # 使用ULA采样预测下一个位置
            with torch.enable_grad():
                ebm_wrapper = EBMWrapper(model, obs_seq_tensor)
                placeholder_x = torch.zeros(1, 1, device=device)
                
                # 采样多个候选（IBC 使用 512 个）
                candidates, _ = ula_sampler.sample(
                    placeholder_x,
                    ebm_wrapper,
                    num_samples=num_action_samples,
                    init_samples=None,
                    return_trajectory=False
                )  # (1, num_action_samples, 2)
                
                # 计算所有候选的能量
                energies = model(obs_seq_tensor, candidates)  # (1, num_action_samples)
                
                # 关键改进：使用概率分布采样而不是直接选最低的
                # 匹配 IBC 的 get_probabilities 逻辑
                logits = -energies / temperature  # (1, num_action_samples)
                probs = F.softmax(logits, dim=1)  # (1, num_action_samples)
                
                # 从概率分布中采样（而不是直接选最低的）
                dist = torch.distributions.Categorical(probs)
                sampled_idx = dist.sample().item()
                next_pos = candidates[0, sampled_idx].cpu().numpy().flatten()  # (2,)
            
            trajectory.append(next_pos)
            
            # 终止条件：移动很小
            if step > 10 and np.linalg.norm(next_pos - trajectory[-2]) < 0.01:
                break
    
    trajectory = np.array(trajectory)  # (T, 2)
    return trajectory

                
if __name__ == '__main__':
    # 数据目录
    data_dir = Path(__file__).parent.parent.parent / 'data' / '_2d' / 'A2B_data'
    
    # 训练参数
    model, dataset, losses = train_a2b_ebm(
        data_dir=str(data_dir),
        num_epochs=1000,
        batch_size=256,  # IBC 默认为 512，这里根据显存调整
        lr=1e-3,
        num_negatives=8,  # ULA生成的负样本数量
        image_size=64,
        temperature=1.0,  # 温度参数
        ula_step_size=0.1,  # ULA初始步长
        ula_num_steps=100,  # ULA迭代步数
        hidden_dim=256,  # 隐藏层维度
        num_residual_blocks=1,  # 残差块数量
        dropout=0.0,  # Dropout概率
        norm_type=None,  # 归一化类型
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n训练完成！")

