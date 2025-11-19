"""
2D 坐标回归简化版
在 2D 任务上演示完整流程

任务：
- 输入: 简单的 2D 图像（如带有一个点的图像）
- 输出: 2D 坐标 (y1, y2) ∈ [-1,1]^2
- 模型: 简化的 EBM（使用简单的 CNN）
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


class SimpleCNN(nn.Module):
    """简单的 CNN 用于提取图像特征"""
    
    def __init__(self, in_channels=1, feature_dim=32):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        )
        self.fc = nn.Linear(32, feature_dim)
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        返回: (B, feature_dim)
        """
        features = self.conv_layers(x)  # (B, 32, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 32)
        features = self.fc(features)  # (B, feature_dim)
        return features


class Simple2DEBM(nn.Module):
    """简化的 2D EBM"""
    
    def __init__(self, feature_dim=32, hidden_dim=64):
        super().__init__()
        self.cnn = SimpleCNN(in_channels=1, feature_dim=feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, y):
        """
        x: (B, 1, H, W) - 图像
        y: (B, N, 2) - N 个候选 2D 坐标
        返回: (B, N) - 每个坐标的能量值
        """
        # 提取图像特征
        features = self.cnn(x)  # (B, feature_dim)
        
        B, feature_dim = features.shape
        N = y.shape[1]
        
        # 特征融合
        features_expanded = features.unsqueeze(1).expand(B, N, feature_dim)
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


def train_2d_ebm():
    """训练 2D EBM"""
    # 设置参数 - 与 run_implicit.sh 保持一致
    feature_dim = 32
    hidden_dim = 64
    num_negatives = 128  # 从 30 增加到 128，与 run_implicit.sh 中的 --stochastic-optimizer-train-samples 128 一致
    batch_size = 16
    num_epochs = 2000  # 从 100 增加到 2000，与 run_implicit.sh 中的 --max-epochs 2000 一致
    lr = 0.001
    
    # 创建数据集
    train_dataset = Simple2DDataset(num_samples=200, image_size=32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = Simple2DEBM(feature_dim=feature_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练历史
    train_losses = []
    
    # 用于可视化的测试样本
    test_image, test_target = train_dataset[0]
    test_image = test_image.unsqueeze(0)  # (1, 1, H, W)
    test_target = test_target.numpy()
    
    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            B = images.shape[0]
            
            # 正样本：真实位置
            positive_coords = targets.unsqueeze(1)  # (B, 1, 2)
            
            # 负样本：随机采样（从均匀分布采样，与 IBC 的 DerivativeFreeOptimizer 一致）
            negative_coords = torch.rand(B, num_negatives, 2) * 2 - 1  # (B, num_negatives, 2)
            
            # 合并正负样本
            all_coords = torch.cat([positive_coords, negative_coords], dim=1)  # (B, 1+num_negatives, 2)
            
            # 关键修复：随机打乱正负样本的位置，避免模型学习到位置偏差
            # 这与 IBC 的实现一致（trainer.py 第 193-199 行）
            permutation = torch.rand(B, all_coords.size(1)).argsort(dim=1)
            all_coords = all_coords[torch.arange(B).unsqueeze(-1), permutation]
            
            # 找到打乱后正样本的位置（原来在索引 0）
            ground_truth = (permutation == 0).nonzero()[:, 1]
            
            # 计算能量
            energies = model(images, all_coords)  # (B, 1+num_negatives)
            
            # InfoNCE 损失：使用打乱后的正样本位置作为 ground truth
            logits = -energies  # 能量转 logits
            loss = F.cross_entropy(logits, ground_truth)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 200 == 0:  # 每 200 个 epoch 打印一次（因为总 epoch 数增加了）
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model, train_losses, test_image, test_target


def visualize_training_results(model, train_losses, test_image, test_target):
    """可视化训练结果"""
    model.eval()
    
    # 计算测试样本的能量景观
    resolution = 40
    x_coords = np.linspace(-1, 1, resolution)
    y_coords = np.linspace(-1, 1, resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    coords_list = []
    for i in range(resolution):
        for j in range(resolution):
            coords_list.append([X_grid[i, j], Y_grid[i, j]])
    
    coords_tensor = torch.FloatTensor(np.array(coords_list)).unsqueeze(0)  # (1, resolution*resolution, 2)
    
    with torch.no_grad():
        energies = model(test_image, coords_tensor)
        energies = energies.squeeze(0).numpy()
    
    energy_grid = energies.reshape(resolution, resolution)
    
    # 找到能量最低点
    min_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
    pred_pos = (X_grid[min_idx], Y_grid[min_idx])
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 左上：训练损失
    ax1 = axes[0, 0]
    ax1.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, alpha=0.3)
    
    # 右上：测试图像
    ax2 = axes[0, 1]
    test_image_np = test_image.squeeze(0).squeeze(0).numpy()
    im2 = ax2.imshow(test_image_np, extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
    ax2.plot(test_target[0], test_target[1], 'g*', markersize=15, label=f'True Position ({test_target[0]:.3f}, {test_target[1]:.3f})')
    ax2.plot(pred_pos[0], pred_pos[1], 'ro', markersize=12, label=f'Predicted Position ({pred_pos[0]:.3f}, {pred_pos[1]:.3f})')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('Test Image')
    ax2.legend()
    plt.colorbar(im2, ax=ax2)
    
    # 左下：能量景观
    ax3 = axes[1, 0]
    im3 = ax3.contourf(X_grid, Y_grid, energy_grid, levels=20, cmap='viridis')
    ax3.plot(test_target[0], test_target[1], 'g*', markersize=15, label='True Position')
    ax3.plot(pred_pos[0], pred_pos[1], 'ro', markersize=12, label='Predicted Position')
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    ax3.set_title('Energy Landscape')
    ax3.legend()
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3)
    
    # 右下：预测误差分布（在测试集上）
    ax4 = axes[1, 1]
    test_dataset = Simple2DDataset(num_samples=50, image_size=32)
    errors = []
    
    with torch.no_grad():
        for image, target in test_dataset:
            image = image.unsqueeze(0)
            target = target.numpy()
            
            energies = model(image, coords_tensor)
            energies = energies.squeeze(0).numpy()
            energy_grid_test = energies.reshape(resolution, resolution)
            min_idx_test = np.unravel_index(np.argmin(energy_grid_test), energy_grid_test.shape)
            pred_pos_test = (X_grid[min_idx_test], Y_grid[min_idx_test])
            
            error = np.sqrt((pred_pos_test[0] - target[0])**2 + (pred_pos_test[1] - target[1])**2)
            errors.append(error)
    
    ax4.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Prediction Error (Euclidean Distance)')
    ax4.set_ylabel('Number of Samples')
    ax4.set_title(f'Test Set Prediction Error Distribution\nMean Error: {np.mean(errors):.4f}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ebm_2d_training_results.png', dpi=150, bbox_inches='tight')
    
    print(f"\n测试结果:")
    print(f"真实位置: ({test_target[0]:.3f}, {test_target[1]:.3f})")
    print(f"预测位置: ({pred_pos[0]:.3f}, {pred_pos[1]:.3f})")
    print(f"预测误差: {np.sqrt((pred_pos[0] - test_target[0])**2 + (pred_pos[1] - test_target[1])**2):.4f}")
    print(f"测试集平均误差: {np.mean(errors):.4f}")
    
    plt.show()


def visualize_training_evolution(model, train_losses, test_image, test_target):
    """可视化训练过程中能量景观的演化"""
    resolution = 30
    x_coords = np.linspace(-1, 1, resolution)
    y_coords = np.linspace(-1, 1, resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    coords_list = []
    for i in range(resolution):
        for j in range(resolution):
            coords_list.append([X_grid[i, j], Y_grid[i, j]])
    
    coords_tensor = torch.FloatTensor(np.array(coords_list)).unsqueeze(0)
    
    # 重新训练并保存中间状态
    feature_dim = 32
    hidden_dim = 64
    num_negatives = 128  # 与训练函数保持一致
    batch_size = 16
    num_epochs = 2000  # 与训练函数保持一致
    lr = 0.001
    
    train_dataset = Simple2DDataset(num_samples=200, image_size=32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = Simple2DEBM(feature_dim=feature_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    epochs_to_plot = [0, 400, 800, 1200, 1600, 1999]  # 调整到新的 epoch 范围
    energy_landscapes = []
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            B = images.shape[0]
            positive_coords = targets.unsqueeze(1)
            negative_coords = torch.rand(B, num_negatives, 2) * 2 - 1
            all_coords = torch.cat([positive_coords, negative_coords], dim=1)
            
            # 关键修复：随机打乱正负样本的位置
            permutation = torch.rand(B, all_coords.size(1)).argsort(dim=1)
            all_coords = all_coords[torch.arange(B).unsqueeze(-1), permutation]
            ground_truth = (permutation == 0).nonzero()[:, 1]
            
            energies = model(images, all_coords)
            logits = -energies
            loss = F.cross_entropy(logits, ground_truth)  # 使用打乱后的 ground truth
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 记录能量景观
        if epoch in epochs_to_plot:
            model.eval()
            with torch.no_grad():
                energies = model(test_image, coords_tensor)
                energies = energies.squeeze(0).numpy()
                energy_grid = energies.reshape(resolution, resolution)
                energy_landscapes.append((epoch, energy_grid))
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (epoch, energy_grid) in enumerate(energy_landscapes):
        ax = axes[idx]
        im = ax.contourf(X_grid, Y_grid, energy_grid, levels=15, cmap='viridis')
        ax.plot(test_target[0], test_target[1], 'r*', markersize=12, label='True Position')
        min_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
        pred_pos = (X_grid[min_idx], Y_grid[min_idx])
        ax.plot(pred_pos[0], pred_pos[1], 'yo', markersize=10, label='Predicted Position')
        ax.set_title(f'Epoch {epoch+1}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ebm_2d_training_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("2D 坐标回归简化版")
    print("=" * 60)
    
    print("\n1. 训练模型...")
    model, train_losses, test_image, test_target = train_2d_ebm()
    
    print("\n2. 可视化训练结果...")
    visualize_training_results(model, train_losses, test_image, test_target)
    
    print("\n3. 可视化训练演化过程...")
    visualize_training_evolution(model, train_losses, test_image, test_target)
    
    print("\n完成！")

