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
from core.models import IBC_EBM
from core.optimizers import DerivativeFreeOptimizer, SGLDSampler

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
    def __init__(self, data_dir, image_size=64, normalize_coords=True, use_first_trajectory_only=False):
        """
        Args:
            data_dir: 数据目录路径（包含 pic/ 和 traj/ 文件夹）
            image_size: 图像尺寸（将图像resize到此大小）
            normalize_coords: 是否将坐标归一化到 [-1, 1]
            use_first_trajectory_only: 如果为True，只加载第一条轨迹（用于验证拟合能力）
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.normalize_coords = normalize_coords
        self.use_first_trajectory_only = use_first_trajectory_only
        
        # 获取所有轨迹文件
        traj_dir = self.data_dir / 'traj'
        self.traj_files = sorted(list(traj_dir.glob('*.json')))
        
        # 如果只使用第一条轨迹，限制文件列表
        if self.use_first_trajectory_only:
            if len(self.traj_files) > 0:
                self.traj_files = [self.traj_files[0]]
                print(f"⚠️  仅使用第一条轨迹进行训练: {self.traj_files[0].name}")
            else:
                raise ValueError("没有找到轨迹文件！")
        
        # 加载所有数据
        self.data = []
        self.coord_bounds = None  # 用于归一化
        
        print(f"加载数据从 {data_dir}...")
        for traj_file in tqdm(self.traj_files, desc="加载轨迹数据"):
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
            
            # 获取坐标范围（用于归一化）
            if self.coord_bounds is None:
                config = traj_data.get('config', {})
                x_range = config.get('x_range', [-5.0, 5.0])
                y_range = config.get('y_range', [-5.0, 5.0])
                self.coord_bounds = np.array([
                    [x_range[0], y_range[0]],  # 下界
                    [x_range[1], y_range[1]]   # 上界
                ], dtype=np.float32)
            
            # 归一化坐标（如果需要）
            if self.normalize_coords:
                # 归一化到 [-1, 1]
                start_pos = self._normalize_coords(start_pos)
                target_pos = self._normalize_coords(target_pos)
                positions = np.array([self._normalize_coords(p) for p in positions])
            
            self.data.append({
                'image_path': str(img_file),
                'start_pos': start_pos,
                'target_pos': target_pos,
                'trajectory': positions,  # (T, 2)
                'trajectory_length': len(positions)
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
        
        # 加载图像
        img = Image.open(item['image_path']).convert('L')  # 转为灰度图
        img = img.resize((self.image_size, self.image_size))
        img_array = np.array(img, dtype=np.float32) / 255.0  # 归一化到 [0, 1]
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
        
        # 起点坐标
        start_pos = torch.from_numpy(item['start_pos']).float()
        
        # 轨迹序列（展平为 (T*2,)）
        trajectory = torch.from_numpy(item['trajectory']).float()  # (T, 2)
        trajectory_flat = trajectory.flatten()  # (T*2,)
        
        return {
            'image': img_tensor,
            'start_pos': start_pos,
            'trajectory': trajectory_flat,
            'trajectory_2d': trajectory,  # 保留2D形状用于可视化
            'target_pos': torch.from_numpy(item['target_pos']).float()
        }


class SingleStepEBM(nn.Module):
    """
    单步轨迹预测 EBM：预测下一个轨迹点
    输入: 图像 + 当前位置
    输出: 候选下一位置的能量值
    
    架构：
    1. 使用基础EBM提取图像特征
    2. 将当前位置和候选下一位置与图像特征融合
    3. 使用MLP计算下一位置的能量
    """
    def __init__(
        self,
        base_ebm: IBC_EBM,
        hidden_dim: int = 512
    ):
        """
        Args:
            base_ebm: 基础的 IBC_EBM 模型（用于提取图像特征）
            hidden_dim: MLP隐藏层维度
        """
        super().__init__()
        self.base_ebm = base_ebm
        
        # 获取图像特征维度（从基础EBM的MLP输入维度推断）
        # base_ebm.mlp[0].in_features = feature_channels * 2 + 2
        # 所以图像特征维度是 feature_channels * 2
        image_feature_dim = base_ebm.mlp[0].in_features - 2  # 减去坐标维度
        
        # 创建新的MLP来处理单步预测
        # 输入：图像特征 + 当前位置(2) + 候选下一位置(2)
        mlp_input_dim = image_feature_dim + 2 + 2
        self.step_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, current_pos: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 图像，形状为 (B, C, H, W)
            current_pos: 当前位置，形状为 (B, 2)
            y: 候选下一位置，形状为 (B, N, 2)
               其中 N 是候选数量
        
        Returns:
            能量值，形状为 (B, N)
        """
        B, N, _ = y.shape
        
        # 使用基础EBM提取图像特征
        # CoordConv
        if self.base_ebm.use_coord_conv:
            x_processed = self.base_ebm.coord_conv(x)
        else:
            x_processed = x
        
        # CNN特征提取
        cnn_out = self.base_ebm.cnn(x_processed, activate=True)  # (B, cnn_blocks[-1], H, W)
        
        # 1x1卷积降维
        conv_out = F.relu(self.base_ebm.conv_1x1(cnn_out))  # (B, feature_channels, H, W)
        
        # SpatialSoftArgmax：提取图像特征
        image_features = self.base_ebm.spatial_softargmax(conv_out)  # (B, feature_channels * 2)
        
        # 扩展图像特征和当前位置以匹配候选数量
        image_features_expanded = image_features.unsqueeze(1).expand(B, N, -1)  # (B, N, feature_dim)
        current_pos_expanded = current_pos.unsqueeze(1).expand(B, N, 2)  # (B, N, 2)
        
        # 拼接：图像特征 + 当前位置 + 候选下一位置
        fused = torch.cat([image_features_expanded, current_pos_expanded, y], dim=-1)  # (B, N, feature_dim + 4)
        fused = fused.reshape(B * N, -1)
        
        # 计算能量
        energy = self.step_mlp(fused)  # (B*N, 1)
        return energy.view(B, N)


def generate_counter_examples_langevin(
    model: SingleStepEBM,
    images: torch.Tensor,
    current_pos: torch.Tensor,
    num_counter_examples: int,
    bounds: np.ndarray,
    num_langevin_steps: int = 100,
    step_size: float = 0.01,
    noise_scale: float = 1.0,
    device: str = 'cuda',
    training: bool = False
):
    """
    使用 Langevin MCMC 生成负样本（硬负样本）
    
    参考 particle 环境中的实现：
    - 从随机均匀分布开始
    - 使用 Langevin 动力学优化，生成接近低能量区域的负样本
    - 这些负样本是"硬负样本"，更难区分，提高训练质量
    
    Args:
        model: 单步预测 EBM 模型
        images: 输入图像 (B, 1, H, W)
        current_pos: 当前位置 (B, 2)
        num_counter_examples: 负样本数量
        bounds: 动作边界 [[min_x, min_y], [max_x, max_y]]
        num_langevin_steps: Langevin 迭代步数
        step_size: Langevin 步长
        noise_scale: Langevin 噪声尺度
        device: 计算设备
        training: 是否在训练模式（影响梯度计算）
        
    Returns:
        负样本 (B, num_counter_examples, 2)
    """
    B = images.size(0)
    bounds_tensor = torch.as_tensor(bounds, dtype=torch.float32, device=device)
    
    # 1. 从均匀分布初始化负样本
    action_dim = 2
    samples = torch.rand(
        B, num_counter_examples, action_dim,
        device=device
    ) * (bounds_tensor[1, :] - bounds_tensor[0, :]) + bounds_tensor[0, :]
    
    # 2. 使用 Langevin MCMC 优化负样本
    samples.requires_grad_(True)
    
    for step in range(num_langevin_steps):
        # 计算能量（需要梯度）
        energies = model(images, current_pos, samples)  # (B, num_counter_examples)
        
        # 计算能量对动作的梯度
        grad = torch.autograd.grad(
            energies.sum(),
            samples,
            create_graph=training,  # 训练时保留计算图，推理时不需要
            retain_graph=True
        )[0]  # (B, num_counter_examples, 2)
        
        # Langevin 更新：y_{t+1} = y_t - step_size * grad + noise
        with torch.no_grad():
            # 梯度缩放（参考 particle 实现）
            gradient_scale = 0.5
            noise = torch.randn_like(samples) * noise_scale
            delta = step_size * (gradient_scale * grad + noise)
            
            # 限制更新幅度（参考 particle 的 delta_action_clip）
            delta_action_clip = 0.1 * 0.5 * (bounds_tensor[1, :] - bounds_tensor[0, :])
            delta = torch.clamp(delta, -delta_action_clip, delta_action_clip)
            
            samples = samples - delta
            
            # 限制在边界内
            samples = samples.clamp(
                min=bounds_tensor[0, :],
                max=bounds_tensor[1, :]
            )
        
        # 重新设置 requires_grad 以便下次迭代
        if step < num_langevin_steps - 1:
            samples = samples.detach().requires_grad_(True)
    
    return samples.detach()


def train_a2b_ebm(
    data_dir: str,
    num_epochs: int = 2000,
    batch_size: int = 8,
    lr: float = 1e-3,
    num_counter_examples: int = 8,  # 负样本数量（参考 particle 配置）
    image_size: int = 64,
    trajectory_length: int = None,  # 如果为None，从数据集推断
    temperature: float = 1.0,  # InfoNCE 温度参数（参考 particle 实现）
    num_langevin_steps: int = 100,  # Langevin MCMC 迭代步数（参考 particle 配置）
    langevin_step_size: float = 0.01,  # Langevin 步长
    langevin_noise_scale: float = 1.0,  # Langevin 噪声尺度
    run_full_chain_under_gradient: bool = True,  # 是否在梯度下运行完整链（参考 particle）
    use_first_trajectory_only: bool = False,  # 是否只使用第一条轨迹进行训练
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    训练 A2B 单步轨迹预测 EBM（参考 particle 环境的 IBC 实现）
    
    训练策略：
    - 输入：图像 + 当前位置 trajectory[t]
    - 正样本：下一位置 trajectory[t+1]（1个）
    - 负样本：使用 Langevin MCMC 从当前能量函数采样（num_counter_examples个）
    - 损失：InfoNCE（softmax + KL 散度），正样本在最后一列
    
    Args:
        data_dir: 数据目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        num_counter_examples: 负样本数量（参考 particle: 8）
        image_size: 图像尺寸
        trajectory_length: 轨迹长度（用于验证）
        temperature: InfoNCE 温度参数（参考 particle: 1.0）
        num_langevin_steps: Langevin MCMC 迭代步数（参考 particle: 100）
        langevin_step_size: Langevin 步长
        langevin_noise_scale: Langevin 噪声尺度
        run_full_chain_under_gradient: 是否在梯度下运行完整链
        use_first_trajectory_only: 是否只使用第一条轨迹进行训练
        device: 计算设备
    """
    print("=" * 60)
    print("A2B 单步轨迹预测 EBM 训练")
    if use_first_trajectory_only:
        print("⚠️  模式：仅使用第一条轨迹进行训练（验证拟合能力）")
    print("=" * 60)
    
    # 1. 加载数据集
    print("\n1. 加载数据集...")
    dataset = A2BDataset(data_dir, image_size=image_size, normalize_coords=True, 
                        use_first_trajectory_only=use_first_trajectory_only)
    
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
    
    # 如果只使用第一条轨迹，不使用shuffle，batch_size设为1
    if use_first_trajectory_only:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"⚠️  仅使用第一条轨迹，batch_size设为1，shuffle=False")
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 2. 创建模型
    print("\n2. 创建模型...")
    # 创建基础EBM
    base_ebm = IBC_EBM(
        in_channels=1,
        cnn_blocks=[32, 64, 64, 64],
        feature_channels=32,
        hidden_dim=1024,
        use_coord_conv=True
    )
    
    # 创建单步预测EBM
    model = SingleStepEBM(base_ebm)
    model = model.to(device)
    
    # 3. 创建优化器
    print("\n3. 创建优化器和损失函数...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    # 获取动作边界（用于 Langevin 采样）
    bounds = dataset.get_trajectory_bounds()
    print(f"动作边界: {bounds}")
    
    print("\n4. 开始训练...")
    print(f"训练策略: 逐步预测，每条轨迹有 {trajectory_length-1} 个训练步")
    print(f"负样本生成: Langevin MCMC ({num_langevin_steps} 步)")
    print(f"负样本数量: {num_counter_examples}")
    print(f"InfoNCE 温度: {temperature}")
    print(f"梯度下运行完整链: {run_full_chain_under_gradient}")
    
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = batch['image'].to(device)  # (B, 1, H, W)
            trajectories_2d = batch['trajectory_2d'].to(device)  # (B, T, 2)
            
            B, T, _ = trajectories_2d.shape
            
            # 对于每条轨迹的每个时间步，生成训练样本
            # 为了并行处理，我们将所有时间步展开成一个大batch
            all_images = []
            all_current_pos = []
            all_next_pos = []
            
            # 遍历每个时间步 (除了最后一个，因为它没有下一个点)
            for t in range(T - 1):
                all_images.append(images)  # (B, 1, H, W)
                all_current_pos.append(trajectories_2d[:, t, :])  # (B, 2)
                all_next_pos.append(trajectories_2d[:, t+1, :])  # (B, 2) - 正样本
            
            # 现在我们有 (T-1) 个时间步的数据，每个包含 B 个样本
            # 总共有 B * (T-1) 个训练样本
            num_steps = T - 1
            
            # 将所有时间步合并成一个大batch进行并行处理
            batch_images = torch.cat(all_images, dim=0)  # (B*num_steps, 1, H, W)
            batch_current_pos = torch.cat(all_current_pos, dim=0)  # (B*num_steps, 2)
            batch_next_pos = torch.cat(all_next_pos, dim=0)  # (B*num_steps, 2) - 正样本
            
            batch_size_total = batch_current_pos.size(0)
            
            # 生成负样本：使用 Langevin MCMC（参考 particle 实现）
            # 根据 run_full_chain_under_gradient 决定是否在梯度下运行
            if run_full_chain_under_gradient:
                # 在梯度下运行完整链（训练时）
                counter_examples = generate_counter_examples_langevin(
                    model,
                    batch_images,
                    batch_current_pos,
                    num_counter_examples,
                    bounds,
                    num_langevin_steps=num_langevin_steps,
                    step_size=langevin_step_size,
                    noise_scale=langevin_noise_scale,
                    device=device,
                    training=True
                )  # (B*num_steps, num_counter_examples, 2)
            else:
                # 在梯度外运行链（推理时，使用 stop_gradient）
                with torch.no_grad():
                    counter_examples = generate_counter_examples_langevin(
                        model,
                        batch_images,
                        batch_current_pos,
                        num_counter_examples,
                        bounds,
                        num_langevin_steps=num_langevin_steps,
                        step_size=langevin_step_size,
                        noise_scale=langevin_noise_scale,
                        device=device,
                        training=False
                    )  # (B*num_steps, num_counter_examples, 2)
            
            # 组织样本：负样本在前，正样本在后（参考 particle 实现）
            # 最终形状: (B*num_steps, num_counter_examples + 1, 2)
            positive_samples = batch_next_pos.unsqueeze(1)  # (B*num_steps, 1, 2)
            all_samples = torch.cat([
                counter_examples,  # (B*num_steps, num_counter_examples, 2) - 负样本
                positive_samples   # (B*num_steps, 1, 2) - 正样本（在最后）
            ], dim=1)  # (B*num_steps, num_counter_examples + 1, 2)
            
            # 计算能量
            predictions = model(batch_images, batch_current_pos, all_samples)  # (B*num_steps, num_counter_examples + 1)
            
            # InfoNCE 损失（参考 particle 实现）
            # predictions: [B*num_steps, num_counter_examples + 1]
            # 正样本在最后一列（索引 num_counter_examples）
            
            # 应用 softmax（带温度）
            softmaxed_predictions = F.softmax(-predictions / temperature, dim=-1)  # (B*num_steps, num_counter_examples + 1)
            
            # 创建标签：正样本位置为 1，其余为 0
            labels = torch.zeros(batch_size_total, dtype=torch.long, device=device)
            labels.fill_(num_counter_examples)  # 正样本在最后一列
            
            # 使用交叉熵损失（等价于 KL 散度）
            loss = F.cross_entropy(
                -predictions / temperature,  # logits
                labels
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("\n训练完成！")
    
    # 5. 保存模型
    model_path = models_dir / 'a2b_ebm_model_single.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'base_ebm_state_dict': model.base_ebm.state_dict(),
        'trajectory_length': trajectory_length,
        'image_size': image_size,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'num_counter_examples': num_counter_examples,
        'temperature': temperature,
        'num_langevin_steps': num_langevin_steps,
        'langevin_step_size': langevin_step_size,
        'langevin_noise_scale': langevin_noise_scale,
    }, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 6. 可视化训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve (Single-Step Prediction)')
    plt.grid(True)
    plt.savefig(output_dir / 'training_loss.png')
    print(f"训练曲线已保存到: {output_dir / 'training_loss.png'}")
    
    return model, dataset, train_losses


def infer_trajectory(
    model: SingleStepEBM,
    image: torch.Tensor,
    start_pos: torch.Tensor,
    optimizer: DerivativeFreeOptimizer,
    max_steps: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    使用优化器逐步推理轨迹序列
    
    Args:
        model: 训练好的单步预测EBM模型
        image: 输入图像，形状为 (1, 1, H, W)
        start_pos: 起始位置，形状为 (2,) 或 (1, 2)
        optimizer: DerivativeFreeOptimizer实例
        max_steps: 最大预测步数
        device: 计算设备
    
    Returns:
        预测的轨迹序列，形状为 (T, 2)，包含起点
    """
    model.eval()
    image = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
    
    if start_pos.dim() == 1:
        start_pos = start_pos.unsqueeze(0)
    start_pos = start_pos.to(device)
    
    trajectory = [start_pos[0].cpu().numpy()]
    current_pos = start_pos
    
    with torch.no_grad():
        for step in range(max_steps):
            # 创建一个包装函数，让优化器能够调用
            def energy_func(x, y):
                # x: 图像 (B, C, H, W)
                # y: 候选位置 (B, N, 2)
                # current_pos 需要复制B次
                B = y.size(0)
                curr_pos_batch = current_pos.expand(B, -1)
                return model(x, curr_pos_batch, y)
            
            # 使用优化器预测下一个位置
            prediction, _ = optimizer.infer(image, energy_func, return_history=False)
            
            # prediction 形状: (1, 2)
            next_pos = prediction[0:1, :]  # 保持 (1, 2) 形状
            trajectory.append(next_pos[0].cpu().numpy())
            
            # 更新当前位置
            current_pos = next_pos
            
            # 可以添加终止条件，例如：
            # - 达到目标位置
            # - 移动距离很小
            # - 到达边界
            if step > 0:
                # 如果移动很小，可能已经到达稳定状态
                prev_pos = torch.from_numpy(trajectory[-2]).to(device).unsqueeze(0)
                if step > 10 and torch.norm(next_pos - prev_pos).item() < 0.01:
                    break
    
    trajectory = np.array(trajectory)  # (T, 2)
    return trajectory

                
if __name__ == '__main__':
    # 数据目录
    data_dir = Path(__file__).parent.parent.parent / 'data' / '_2d' / 'A2B_data'
    
    # 训练参数（参考 particle 配置）
    model, dataset, losses = train_a2b_ebm(
        data_dir=str(data_dir),
        num_epochs=400,
        batch_size=16,
        lr=1e-3,
        num_counter_examples=8,  # 参考 particle: ImplicitBCAgent.num_counter_examples = 8
        image_size=64,
        temperature=1.0,  # 参考 particle: 默认 softmax_temperature=1.0
        num_langevin_steps=100,  # 参考 particle: langevin_actions_given_obs.num_iterations = 100
        langevin_step_size=0.01,
        langevin_noise_scale=1.0,  # 参考 particle: noise_scale=1.0
        run_full_chain_under_gradient=True,  # 参考 particle: ImplicitBCAgent.run_full_chain_under_gradient = True
        use_first_trajectory_only=True,  # 只使用第一条轨迹进行训练（验证拟合能力）
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n训练完成！")

