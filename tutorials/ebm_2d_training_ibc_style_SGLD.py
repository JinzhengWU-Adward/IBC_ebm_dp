"""
2D EBM 训练 + 优化推理（IBC 风格 + SGLD 采样）
整合 IBC 的关键特性：CoordConv、SpatialSoftArgmax、残差块等
使用 SGLD (Stochastic Gradient Langevin Dynamics) 进行负样本采样

任务：
- 输入: 简单的 2D 图像（如带有一个点的图像）
- 输出: 2D 坐标 (y1, y2) ∈ [-1,1]^2
- 模型: IBC 风格的 EBM（使用 CoordConv + SpatialSoftArgmax + 残差块）
- 训练: 使用 SGLD 采样负样本（替代随机采样）
- 推理: 使用 Derivative-Free Optimizer 进行迭代优化
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
script_name = Path(__file__).stem
output_dir = Path(__file__).parent / 'plots' / script_name
output_dir.mkdir(parents=True, exist_ok=True)


# ==================== IBC 关键模块 ====================

class CoordConv(nn.Module):
    """坐标卷积：在输入特征图上添加归一化的坐标通道"""
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, image_height, image_width = x.size()
        y_coords = (
            2.0
            * torch.arange(image_height, device=x.device).unsqueeze(1).expand(image_height, image_width)
            / (image_height - 1.0)
            - 1.0
        )
        x_coords = (
            2.0
            * torch.arange(image_width, device=x.device).unsqueeze(0).expand(image_height, image_width)
            / (image_width - 1.0)
            - 1.0
        )
        coords = torch.stack((y_coords, x_coords), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)
        x = torch.cat((coords, x), dim=1)
        return x


class SpatialSoftArgmax(nn.Module):
    """空间软最大值：保留空间位置信息"""
    
    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    def _coord_grid(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                    indexing="ij",
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
                indexing="ij",
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."
        
        _, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)
        
        xc, yc = self._coord_grid(h, w, x.device)
        
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)
        
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, depth: int, activation_fn=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.activation = activation_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(x)  # 注意：这里用的是 x 而不是 out
        out = self.conv2(out)
        return out + x


class IBCStyleCNN(nn.Module):
    """IBC 风格的 CNN：使用残差块"""
    
    def __init__(self, in_channels=1, blocks=[16, 32, 32], activation_fn=nn.ReLU):
        super().__init__()
        depth_in = in_channels
        layers = []
        for depth_out in blocks:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                ResidualBlock(depth_out, activation_fn),
            ])
            depth_in = depth_out
        self.net = nn.Sequential(*layers)
        self.activation = activation_fn()
    
    def forward(self, x: torch.Tensor, activate: bool = False) -> torch.Tensor:
        out = self.net(x)
        if activate:
            return self.activation(out)
        return out


class IBCStyleEBM(nn.Module):
    """IBC 风格的 EBM：整合 CoordConv + SpatialSoftArgmax"""
    
    def __init__(self, in_channels=1, cnn_blocks=[16, 32, 32], 
                 feature_channels=16, hidden_dim=256, use_coord_conv=True):
        super().__init__()
        self.use_coord_conv = use_coord_conv
        
        if use_coord_conv:
            self.coord_conv = CoordConv()
            # CoordConv 添加了 2 个通道
            cnn_in_channels = in_channels + 2
        else:
            cnn_in_channels = in_channels
        
        self.cnn = IBCStyleCNN(cnn_in_channels, cnn_blocks)
        
        # 1x1 卷积降维
        self.conv_1x1 = nn.Conv2d(cnn_blocks[-1], feature_channels, 1)
        
        # SpatialSoftArgmax：输出 feature_channels * 2 维（每个通道的 x, y 坐标）
        self.spatial_softargmax = SpatialSoftArgmax(normalize=True)
        
        # MLP：输入是 (feature_channels * 2 + 2)，输出是能量值
        # feature_channels * 2 来自 SpatialSoftArgmax，+2 是候选坐标 y
        mlp_input_dim = feature_channels * 2 + 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, y):
        """
        x: (B, C, H, W) - 图像
        y: (B, N, 2) - N 个候选 2D 坐标
        返回: (B, N) - 每个坐标的能量值
        """
        # CoordConv
        if self.use_coord_conv:
            x = self.coord_conv(x)
        
        # CNN 特征提取
        out = self.cnn(x, activate=True)  # (B, cnn_blocks[-1], H, W)
        
        # 1x1 卷积降维
        out = F.relu(self.conv_1x1(out))  # (B, feature_channels, H, W)
        
        # SpatialSoftArgmax：保留空间位置信息
        out = self.spatial_softargmax(out)  # (B, feature_channels * 2)
        
        # 特征融合
        B, feature_dim = out.shape
        N = y.shape[1]
        
        features_expanded = out.unsqueeze(1).expand(B, N, feature_dim)
        fused = torch.cat([features_expanded, y], dim=-1)  # (B, N, feature_dim+2)
        fused = fused.reshape(B * N, feature_dim + 2)
        
        # 计算能量
        energy = self.mlp(fused)  # (B*N, 1)
        return energy.view(B, N)


class Simple2DDataset(Dataset):
    """简单的 2D 数据集"""
    
    def __init__(self, num_samples=200, image_size=32):
        self.num_samples = num_samples
        self.image_size = image_size
        self.data = []
        self.targets = []
        
        # 生成数据
        for _ in range(num_samples):
            # 随机生成目标位置
            target_pos = (np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8))
            
            # 生成图像（在目标位置有一个亮点）
            image = self._generate_image(target_pos)
            
            self.data.append(image)
            self.targets.append(target_pos)
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
    
    def _generate_image(self, target_pos, sigma=0.1):
        """生成在 target_pos 位置有亮点的图像"""
        x = np.linspace(-1, 1, self.image_size)
        y = np.linspace(-1, 1, self.image_size)
        X, Y = np.meshgrid(x, y)
        dist_sq = (X - target_pos[0])**2 + (Y - target_pos[1])**2
        image = np.exp(-dist_sq / (2 * sigma**2))
        return image
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        # 转换为 tensor 并添加通道维度
        image_tensor = torch.FloatTensor(image).unsqueeze(0)  # (1, H, W)
        target_tensor = torch.FloatTensor(target)  # (2,)
        return image_tensor, target_tensor


class SGLDSampler:
    """SGLD (Stochastic Gradient Langevin Dynamics) 采样器
    
    用于从能量分布中采样负样本，生成"困难"的负样本以提高训练效果
    """
    
    def __init__(self, 
                 num_steps: int = 10,
                 step_size: float = 0.1,
                 step_size_decay: float = 1.0,
                 temperature: float = 1.0,
                 bounds: list = None,
                 device: str = 'cpu'):
        """
        num_steps: SGLD 迭代步数
        step_size: 初始步长
        step_size_decay: 步长衰减系数（每步衰减）
        temperature: 温度参数，控制噪声尺度
        bounds: 边界约束 [[min_x, min_y], [max_x, max_y]]
        device: 设备
        """
        self.num_steps = num_steps
        self.step_size = step_size
        self.step_size_decay = step_size_decay
        self.temperature = temperature
        self.bounds = np.array(bounds) if bounds is not None else None
        self.device = torch.device(device)
    
    def sample(self, 
               model: nn.Module,
               x: torch.Tensor,
               num_samples: int,
               init_samples: torch.Tensor = None) -> torch.Tensor:
        """
        使用 SGLD 从能量分布中采样负样本
        
        Args:
            model: EBM 模型
            x: 输入图像 (B, C, H, W)
            num_samples: 每个图像采样的负样本数
            init_samples: 初始样本 (B, num_samples, 2)，如果为 None 则随机初始化
        
        Returns:
            samples: (B, num_samples, 2) 采样得到的负样本
        """
        model.eval()  # 采样时不需要梯度更新模型参数
        B = x.size(0)
        
        # 边界张量（提前转换为 tensor）
        if self.bounds is not None:
            bounds_tensor = torch.as_tensor(self.bounds, dtype=torch.float32, device=self.device)
        else:
            bounds_tensor = None
        
        # 初始化样本
        if init_samples is None:
            if bounds_tensor is not None:
                samples = torch.rand(B, num_samples, 2, device=self.device)
                # 使用 tensor 操作进行缩放和平移
                samples = samples * (bounds_tensor[1] - bounds_tensor[0]) + bounds_tensor[0]
            else:
                samples = torch.rand(B, num_samples, 2, device=self.device) * 2 - 1
        else:
            samples = init_samples.clone().to(self.device)
        
        # SGLD 迭代
        current_step_size = self.step_size
        
        # 确保模型参数不需要梯度（只对 samples 求导）
        for param in model.parameters():
            param.requires_grad = False
        
        for step in range(self.num_steps):
            # 需要计算梯度，所以设置 requires_grad
            # 注意：只对 samples 求导，模型参数保持冻结
            samples.requires_grad_(True)
            
            # 计算能量（模型参数不参与梯度计算）
            energies = model(x, samples)  # (B, num_samples)
            
            # 计算能量对坐标的梯度
            # 只对 samples 求导，不对模型参数求导
            energy_sum = energies.sum()  # 标量，用于反向传播
            grad_y = torch.autograd.grad(
                energy_sum, 
                samples, 
                create_graph=False,  # 不需要二阶梯度
                retain_graph=False
            )[0]  # (B, num_samples, 2)
            
            # 关闭梯度计算以提高效率
            samples = samples.detach()
            
            # SGLD 更新：y = y - ε * ∇E + √(2ε*T) * η
            # 其中：ε 是步长，T 是温度，η ~ N(0, I) 是高斯噪声
            noise_scale = np.sqrt(2 * current_step_size * self.temperature)
            noise = torch.randn_like(samples) * noise_scale
            
            samples = samples - current_step_size * grad_y + noise
            
            # 应用边界约束
            if bounds_tensor is not None:
                samples = samples.clamp(min=bounds_tensor[0], max=bounds_tensor[1])
            else:
                samples = samples.clamp(min=-1.0, max=1.0)
            
            # 步长衰减（可选）
            current_step_size *= self.step_size_decay
        
        # 恢复模型参数的梯度设置（用于后续训练）
        for param in model.parameters():
            param.requires_grad = True
        
        return samples.detach()


class DerivativeFreeOptimizer:
    """无导数优化器（基于 IBC 的实现）"""
    
    def __init__(self, bounds, num_samples=16384, num_iters=3, 
                 noise_scale=0.33, noise_shrink=0.5, device='cpu'):
        """
        bounds: (2, 2) 数组，[[min_x, min_y], [max_x, max_y]]
        num_samples: 每次迭代的候选样本数
        num_iters: 迭代次数
        noise_scale: 初始噪声尺度
        noise_shrink: 噪声衰减系数
        """
        self.bounds = np.array(bounds)
        self.num_samples = num_samples
        self.num_iters = num_iters
        self.noise_scale = noise_scale
        self.noise_shrink = noise_shrink
        self.device = torch.device(device)
    
    def _sample(self, num_samples):
        """从均匀分布采样"""
        size = (num_samples, self.bounds.shape[1])
        samples = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=size)
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)
    
    @torch.no_grad()
    def infer(self, x, ebm, return_history=False):
        """
        优化推理
        x: (B, 1, H, W) - 输入图像
        ebm: EBM 模型
        return_history: 是否返回优化历史
        返回: (B, 2) - 预测的坐标
        """
        ebm.eval()
        B = x.size(0)
        noise_scale = self.noise_scale
        bounds_tensor = torch.as_tensor(self.bounds, dtype=torch.float32).to(self.device)
        
        # 初始化样本
        samples = self._sample(B * self.num_samples)
        samples = samples.reshape(B, self.num_samples, -1)
        
        history = [] if return_history else None
        
        for i in range(self.num_iters):
            # 计算能量
            energies = ebm(x, samples)  # (B, num_samples)
            
            # 转换为概率
            probs = F.softmax(-1.0 * energies, dim=-1)  # (B, num_samples)
            
            # 重要性采样：按概率分布重采样
            idxs = torch.multinomial(probs, self.num_samples, replacement=True)  # (B, num_samples)
            samples = samples[torch.arange(B).unsqueeze(-1), idxs]  # (B, num_samples, 2)
            
            # 添加噪声
            samples = samples + torch.randn_like(samples) * noise_scale
            
            # 限制在边界内
            samples = samples.clamp(min=bounds_tensor[0, :], max=bounds_tensor[1, :])
            
            # 记录历史（在添加噪声和裁剪之后）
            if return_history:
                # 重新计算能量以获取添加噪声后的能量
                energies_after_noise = ebm(x, samples)
                best_indices = energies_after_noise.argmin(dim=1)
                best_samples = samples[torch.arange(B), best_indices]
                history.append({
                    'samples': samples.clone().cpu().numpy(),
                    'energies': energies_after_noise.cpu().numpy(),
                    'best_samples': best_samples.clone().cpu().numpy(),
                    'noise_scale': noise_scale
                })
            
            # 噪声衰减
            noise_scale *= self.noise_shrink
        
        # 返回能量最低的样本
        energies = ebm(x, samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        predictions = samples[torch.arange(B), best_idxs, :]
        
        if return_history:
            return predictions, history
        return predictions


def train_2d_ebm():
    """训练 2D EBM（IBC 风格，使用 SGLD 采样）"""
    # 设置参数
    cnn_blocks = [16, 32, 32]
    feature_channels = 16
    hidden_dim = 256
    num_negatives = 128
    batch_size = 16
    num_epochs = 200
    lr = 0.001
    use_coord_conv = True
    
    # SGLD 采样器参数
    use_sgld = True  # 是否使用 SGLD 采样
    sgld_num_steps = 10  # SGLD 迭代步数
    sgld_step_size = 0.1  # 初始步长
    sgld_step_size_decay = 0.99  # 每步衰减系数
    sgld_temperature = 1.0  # 温度参数
    use_mixed_sampling = False  # 是否混合随机采样和 SGLD 采样
    sgld_ratio = 0.5  # SGLD 采样比例（如果使用混合采样）
    
    # 创建数据集
    train_dataset = Simple2DDataset(num_samples=1000, image_size=32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = IBCStyleEBM(
        in_channels=1,
        cnn_blocks=cnn_blocks,
        feature_channels=feature_channels,
        hidden_dim=hidden_dim,
        use_coord_conv=use_coord_conv
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 创建 SGLD 采样器
    if use_sgld:
        sgld_sampler = SGLDSampler(
            num_steps=sgld_num_steps,
            step_size=sgld_step_size,
            step_size_decay=sgld_step_size_decay,
            temperature=sgld_temperature,
            bounds=[[-1.0, -1.0], [1.0, 1.0]],
            device=device
        )
    
    # 训练历史
    train_losses = []
    
    # 用于可视化的测试样本
    test_image, test_target = train_dataset[0]
    test_image = test_image.unsqueeze(0).to(device)  # (1, 1, H, W)
    # test_target 可能是 tensor 或 numpy，统一转换为 numpy
    if isinstance(test_target, torch.Tensor):
        test_target = test_target.cpu().numpy() if test_target.is_cuda else test_target.numpy()
    else:
        test_target = np.array(test_target)
    
    print("开始训练...")
    print(f"模型配置:")
    print(f"  - CoordConv: {use_coord_conv}")
    print(f"  - CNN blocks: {cnn_blocks}")
    print(f"  - Feature channels: {feature_channels}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - 训练样本数: {len(train_dataset)}")
    print(f"  - 使用 SGLD 采样: {use_sgld}")
    if use_sgld:
        print(f"    - SGLD 步数: {sgld_num_steps}")
        print(f"    - 初始步长: {sgld_step_size}")
        print(f"    - 步长衰减: {sgld_step_size_decay}")
        print(f"    - 温度: {sgld_temperature}")
        print(f"    - 混合采样: {use_mixed_sampling}")
        if use_mixed_sampling:
            print(f"    - SGLD 比例: {sgld_ratio}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            B = images.shape[0]
            
            # 正样本：真实位置
            positive_coords = targets.unsqueeze(1)  # (B, 1, 2)
            
            # 负样本采样
            if use_sgld:
                if use_mixed_sampling:
                    # 混合采样：部分随机，部分 SGLD
                    num_random = int(num_negatives * (1 - sgld_ratio))
                    num_sgld = num_negatives - num_random
                    
                    # 随机采样
                    random_negatives = torch.rand(B, num_random, 2, device=device) * 2 - 1
                    
                    # SGLD 采样
                    sgld_negatives = sgld_sampler.sample(
                        model=model,
                        x=images,
                        num_samples=num_sgld
                    )
                    
                    negative_coords = torch.cat([random_negatives, sgld_negatives], dim=1)
                else:
                    # 完全使用 SGLD 采样
                    negative_coords = sgld_sampler.sample(
                        model=model,
                        x=images,
                        num_samples=num_negatives
                    )
            else:
                # 随机采样（原始方法）
                negative_coords = torch.rand(B, num_negatives, 2, device=device) * 2 - 1
            
            # 合并正负样本
            all_coords = torch.cat([positive_coords, negative_coords], dim=1)  # (B, 1+num_negatives, 2)
            
            # 随机打乱正负样本的位置
            permutation = torch.rand(B, all_coords.size(1), device=device).argsort(dim=1)
            all_coords = all_coords[torch.arange(B, device=device).unsqueeze(-1), permutation]
            
            # 找到打乱后正样本的位置
            ground_truth = (permutation == 0).nonzero()[:, 1].to(device)
            
            # 计算能量
            energies = model(images, all_coords)  # (B, 1+num_negatives)
            
            # InfoNCE 损失
            logits = -energies
            loss = F.cross_entropy(logits, ground_truth)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model, train_losses, test_image, test_target, train_dataset


def visualize_training_and_optimization(model, train_losses, test_image, test_target, 
                                       train_dataset, optimizer):
    """可视化训练结果和优化过程"""
    model.eval()
    
    # 使用优化器进行推理
    print("\n使用 Derivative-Free Optimizer 进行推理...")
    test_image_batch = test_image.to(optimizer.device)
    predictions, history = optimizer.infer(test_image_batch, model, return_history=True)
    pred_pos = predictions[0].cpu().numpy()
    
    # 计算能量景观（用于背景可视化）
    resolution = 50
    x_coords = np.linspace(-1, 1, resolution)
    y_coords = np.linspace(-1, 1, resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    coords_list = []
    for i in range(resolution):
        for j in range(resolution):
            coords_list.append([X_grid[i, j], Y_grid[i, j]])
    
    coords_tensor = torch.FloatTensor(np.array(coords_list)).unsqueeze(0).to(optimizer.device)
    
    with torch.no_grad():
        energies = model(test_image_batch, coords_tensor)
        energies = energies.squeeze(0).cpu().numpy()
    
    energy_grid = energies.reshape(resolution, resolution)
    
    # 创建可视化
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 训练损失曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, alpha=0.3)
    
    # 2. 测试图像
    ax2 = fig.add_subplot(gs[0, 1])
    test_image_np = test_image.squeeze(0).squeeze(0).cpu().numpy()
    im2 = ax2.imshow(test_image_np, extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
    ax2.plot(test_target[0], test_target[1], 'g*', markersize=15, 
             label=f'True ({test_target[0]:.3f}, {test_target[1]:.3f})')
    ax2.plot(pred_pos[0], pred_pos[1], 'ro', markersize=12, 
             label=f'Pred ({pred_pos[0]:.3f}, {pred_pos[1]:.3f})')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('Test Image')
    ax2.legend()
    plt.colorbar(im2, ax=ax2)
    
    # 3. 能量景观
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.contourf(X_grid, Y_grid, energy_grid, levels=20, cmap='viridis')
    ax3.plot(test_target[0], test_target[1], 'g*', markersize=15, label='True Position')
    ax3.plot(pred_pos[0], pred_pos[1], 'ro', markersize=12, label='Predicted Position')
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    ax3.set_title('Energy Landscape')
    ax3.legend()
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3)
    
    # 4. 优化轨迹（显示几个关键迭代）
    ax4 = fig.add_subplot(gs[1, :])
    ax4.contourf(X_grid, Y_grid, energy_grid, levels=15, cmap='viridis', alpha=0.3)
    ax4.contour(X_grid, Y_grid, energy_grid, levels=15, colors='gray', alpha=0.3, linewidths=0.5)
    
    # 显示优化轨迹
    iterations_to_show = [0, 1, 2] if len(history) >= 3 else list(range(len(history)))
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(iterations_to_show)))
    
    for idx, iter_idx in enumerate(iterations_to_show):
        if iter_idx < len(history):
            samples = history[iter_idx]['samples'][0]  # (num_samples, 2)
            # 只显示部分样本以避免过于密集
            sample_indices = np.random.choice(len(samples), min(500, len(samples)), replace=False)
            ax4.scatter(samples[sample_indices, 0], samples[sample_indices, 1], 
                       s=10, alpha=0.3, color=colors[idx], label=f'Iter {iter_idx}')
    
    # 显示最佳样本轨迹
    best_trajectory = np.array([h['best_samples'][0] for h in history])
    ax4.plot(best_trajectory[:, 0], best_trajectory[:, 1], 'r-o', 
            linewidth=2, markersize=8, alpha=0.8, label='Best Sample Trajectory')
    ax4.plot(test_target[0], test_target[1], 'b*', markersize=20, label='True Position')
    ax4.plot(best_trajectory[0, 0], best_trajectory[0, 1], 'go', 
            markersize=12, label='Start')
    ax4.plot(best_trajectory[-1, 0], best_trajectory[-1, 1], 'ro', 
            markersize=12, label='End')
    ax4.set_xlabel('X Coordinate')
    ax4.set_ylabel('Y Coordinate')
    ax4.set_title('Optimization Process: Sample Distribution Evolution')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    # 5. 能量值随迭代的变化
    ax5 = fig.add_subplot(gs[2, 0])
    best_energies = [h['energies'][0][h['energies'][0].argmin()] for h in history]
    mean_energies = [h['energies'][0].mean() for h in history]
    min_energies = [h['energies'][0].min() for h in history]
    
    ax5.plot(range(len(best_energies)), best_energies, 'r-', 
            linewidth=2, label='Best Sample Energy')
    ax5.plot(range(len(mean_energies)), mean_energies, 'b--', 
            linewidth=2, alpha=0.7, label='Mean Energy')
    ax5.plot(range(len(min_energies)), min_energies, 'g-', 
            linewidth=2, alpha=0.7, label='Min Energy')
    ax5.set_xlabel('Iteration Step')
    ax5.set_ylabel('Energy Value')
    ax5.set_title('Energy Value Change During Optimization')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 样本分布的标准差（收敛性指标）
    ax6 = fig.add_subplot(gs[2, 1])
    std_x = [np.std(h['samples'][0][:, 0]) for h in history]
    std_y = [np.std(h['samples'][0][:, 1]) for h in history]
    
    ax6.plot(range(len(std_x)), std_x, 'b-', linewidth=2, label='X Direction Std')
    ax6.plot(range(len(std_y)), std_y, 'r-', linewidth=2, label='Y Direction Std')
    ax6.set_xlabel('Iteration Step')
    ax6.set_ylabel('Standard Deviation')
    ax6.set_title('Sample Distribution Std (Convergence)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 测试集误差分布
    ax7 = fig.add_subplot(gs[2, 2])
    test_dataset = Simple2DDataset(num_samples=50, image_size=32)
    errors = []
    
    print("正在评估测试集...")
    with torch.no_grad():
        for idx, (image, target) in enumerate(test_dataset):
            image_batch = image.unsqueeze(0).to(optimizer.device)
            # target 可能是 tensor 或 numpy，统一转换为 numpy
            if isinstance(target, torch.Tensor):
                target_np = target.cpu().numpy() if target.is_cuda else target.numpy()
            else:
                target_np = np.array(target)
            
            pred = optimizer.infer(image_batch, model, return_history=False)
            pred_np = pred[0].cpu().numpy()
            
            error = np.sqrt((pred_np[0] - target_np[0])**2 + (pred_np[1] - target_np[1])**2)
            errors.append(error)
            
            if (idx + 1) % 10 == 0:
                print(f"  已评估 {idx + 1}/{len(test_dataset)} 个样本...")
    
    ax7.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax7.set_xlabel('Prediction Error (Euclidean Distance)')
    ax7.set_ylabel('Number of Samples')
    ax7.set_title(f'Test Set Error Distribution\nMean: {np.mean(errors):.4f}')
    ax7.grid(True, alpha=0.3)
    
    # 使用 subplots_adjust 替代 tight_layout 以避免警告
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.3)
    plt.savefig(output_dir / 'ebm_2d_training_ibc_style.png', dpi=150, bbox_inches='tight')
    
    print(f"\n测试结果:")
    print(f"真实位置: ({test_target[0]:.3f}, {test_target[1]:.3f})")
    print(f"预测位置: ({pred_pos[0]:.3f}, {pred_pos[1]:.3f})")
    print(f"预测误差: {np.sqrt((pred_pos[0] - test_target[0])**2 + (pred_pos[1] - test_target[1])**2):.4f}")
    print(f"测试集平均误差: {np.mean(errors):.4f}")
    print(f"测试集误差标准差: {np.std(errors):.4f}")
    print(f"测试集最小误差: {np.min(errors):.4f}")
    print(f"测试集最大误差: {np.max(errors):.4f}")
    
    # 打印优化过程信息
    if history:
        print(f"\n优化过程信息:")
        print(f"  迭代次数: {len(history)}")
        print(f"  每次迭代样本数: {optimizer.num_samples}")
        for i, h in enumerate(history):
            print(f"  迭代 {i}: 最佳能量={h['energies'][0].min():.4f}, "
                  f"平均能量={h['energies'][0].mean():.4f}, "
                  f"噪声尺度={h['noise_scale']:.4f}")
    
    plt.show()


def visualize_multiple_targets_inference(model, train_dataset, optimizer, num_targets=9):
    """可视化多个不同目标的推理结果，每个目标在新窗口中显示"""
    model.eval()
    
    # 选择多个不同的测试样本
    test_indices = np.linspace(0, len(train_dataset) - 1, num_targets, dtype=int)
    test_samples = [train_dataset[i] for i in test_indices]  # train_dataset[i] 返回 (image, target)
    
    print(f"\n对 {num_targets} 个不同目标进行推理...")
    
    all_errors = []
    figures = []  # 保存所有figure对象
    
    with torch.no_grad():
        for idx, (image, target) in enumerate(test_samples):
            if idx >= num_targets:
                break
            
            # target 可能是 tensor 或 numpy，统一转换为 numpy
            if isinstance(target, torch.Tensor):
                target_np = target.cpu().numpy() if target.is_cuda else target.numpy()
            else:
                target_np = np.array(target)
            
            # 进行推理
            image_batch = image.unsqueeze(0).to(optimizer.device)
            pred = optimizer.infer(image_batch, model, return_history=False)
            pred_np = pred[0].cpu().numpy()
            
            # 计算误差
            error = np.sqrt((pred_np[0] - target_np[0])**2 + (pred_np[1] - target_np[1])**2)
            all_errors.append(error)
            
            # 计算能量景观（用于背景）
            resolution = 50  # 提高分辨率以获得更好的可视化效果
            x_coords = np.linspace(-1, 1, resolution)
            y_coords = np.linspace(-1, 1, resolution)
            X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
            
            coords_list = []
            for i in range(resolution):
                for j in range(resolution):
                    coords_list.append([X_grid[i, j], Y_grid[i, j]])
            
            coords_tensor = torch.FloatTensor(np.array(coords_list)).unsqueeze(0).to(optimizer.device)
            energies = model(image_batch, coords_tensor)
            energies = energies.squeeze(0).cpu().numpy()
            energy_grid = energies.reshape(resolution, resolution)
            
            # 创建新窗口
            fig, ax = plt.subplots(figsize=(10, 10))
            figures.append(fig)
            
            # 绘制能量景观（背景）
            im1 = ax.contourf(X_grid, Y_grid, energy_grid, levels=20, cmap='viridis', alpha=0.7)
            ax.contour(X_grid, Y_grid, energy_grid, levels=20, colors='gray', alpha=0.3, linewidths=0.5)
            plt.colorbar(im1, ax=ax, label='Energy')
            
            # 绘制图像（叠加显示）
            # image 来自 dataset，通常是 CPU tensor，但为了安全还是检查一下
            if isinstance(image, torch.Tensor):
                image_np = image.squeeze(0).cpu().numpy() if image.is_cuda else image.squeeze(0).numpy()
            else:
                image_np = np.array(image).squeeze(0)
            im2 = ax.imshow(image_np, extent=[-1, 1, -1, 1], origin='lower', 
                           cmap='hot', alpha=0.5, aspect='auto')
            
            # 绘制真实位置和预测位置
            ax.plot(target_np[0], target_np[1], 'g*', markersize=25, 
                   label=f'True Position ({target_np[0]:.3f}, {target_np[1]:.3f})', 
                   zorder=5, markeredgecolor='white', markeredgewidth=2)
            ax.plot(pred_np[0], pred_np[1], 'ro', markersize=20, 
                   label=f'Predicted Position ({pred_np[0]:.3f}, {pred_np[1]:.3f})', 
                   zorder=5, markeredgecolor='white', markeredgewidth=2)
            
            # 绘制连接线
            ax.plot([target_np[0], pred_np[0]], [target_np[1], pred_np[1]], 
                   'r--', linewidth=3, alpha=0.8, zorder=4, label=f'Error: {error:.4f}')
            
            # 设置标题和标签
            ax.set_title(f'Sample {idx+1} - Inference Result (IBC Style)\n'
                        f'True: ({target_np[0]:.3f}, {target_np[1]:.3f}) | '
                        f'Pred: ({pred_np[0]:.3f}, {pred_np[1]:.3f}) | '
                        f'Error: {error:.4f}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('X Coordinate', fontsize=11)
            ax.set_ylabel('Y Coordinate', fontsize=11)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
            
            # 保存单独的图片
            plt.tight_layout()
            plt.savefig(output_dir / f'multiple_targets_inference_sample_{idx+1}.png', 
                       dpi=150, bbox_inches='tight')
            
            print(f"  样本 {idx+1}: 真实=({target_np[0]:.3f}, {target_np[1]:.3f}), "
                  f"预测=({pred_np[0]:.3f}, {pred_np[1]:.3f}), 误差={error:.4f}")
            
            # 显示当前窗口（非阻塞模式）
            plt.show(block=False)
    
    print(f"\n多目标推理结果:")
    print(f"  平均误差: {np.mean(all_errors):.4f}")
    print(f"  误差标准差: {np.std(all_errors):.4f}")
    print(f"  最小误差: {np.min(all_errors):.4f}")
    print(f"  最大误差: {np.max(all_errors):.4f}")
    print(f"\n已生成 {num_targets} 个独立窗口，每个窗口显示一个目标的推理结果")
    print(f"图片已保存到: {output_dir}")
    
    # 最后显示所有窗口（阻塞模式，等待用户关闭）
    plt.show()
    
    return all_errors


if __name__ == "__main__":
    print("=" * 60)
    print("2D EBM 训练 + 优化推理（IBC 风格 + SGLD 采样）")
    print("整合 CoordConv + SpatialSoftArgmax + 残差块")
    print("使用 SGLD 进行负样本采样")
    print("=" * 60)
    
    # 1. 训练模型
    print("\n1. 训练模型...")
    model, train_losses, test_image, test_target, train_dataset = train_2d_ebm()
    
    # 2. 创建优化器
    print("\n2. 创建优化器...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = DerivativeFreeOptimizer(
        bounds=[[-1.0, -1.0], [1.0, 1.0]],
        num_samples=16384,  # 与 IBC 的 inference_samples 一致 (2^14)
        num_iters=3,  # 与 IBC 的 iters 一致
        noise_scale=0.33,  # 与 IBC 的 noise_scale 一致
        noise_shrink=0.5,  # 与 IBC 的 noise_shrink 一致
        device=device
    )
    
    # 3. 可视化训练结果和优化过程
    print("\n3. 可视化训练结果和优化过程...")
    visualize_training_and_optimization(
        model, train_losses, test_image, test_target, train_dataset, optimizer
    )
    
    # 4. 可视化多个不同目标的推理结果
    print("\n4. 可视化多个不同目标的推理结果...")
    visualize_multiple_targets_inference(model, train_dataset, optimizer, num_targets=9)
    
    print("\n完成！")

