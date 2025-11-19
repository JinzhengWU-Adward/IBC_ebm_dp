"""
2D 能量景观可视化
可视化 2D 坐标空间中的能量分布

任务：给定一个固定的图像 x，在 2D 坐标空间 [-1,1] × [-1,1] 中计算能量值
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
script_name = Path(__file__).stem
output_dir = Path(__file__).parent / 'plots' / script_name
output_dir.mkdir(parents=True, exist_ok=True)


class Simple2DEBM(nn.Module):
    """简化的 2D EBM：输入是图像特征，输出是 2D 坐标的能量值"""
    
    def __init__(self, feature_dim=32, hidden_dim=64):
        super().__init__()
        # 输入: [image_features, x_coord, y_coord] -> 能量值
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features, coords):
        """
        features: (B, feature_dim) - 图像特征
        coords: (B, N, 2) - N 个候选 2D 坐标
        返回: (B, N) - 每个坐标的能量值
        """
        B, feature_dim = features.shape
        N = coords.shape[1]
        
        # 扩展 features 以匹配 coords 的数量
        features_expanded = features.unsqueeze(1).expand(B, N, feature_dim)  # (B, N, feature_dim)
        
        # 拼接特征
        fused = torch.cat([features_expanded, coords], dim=-1)  # (B, N, feature_dim+2)
        fused = fused.reshape(B * N, feature_dim + 2)
        
        # 计算能量
        energy = self.mlp(fused)  # (B*N, 1)
        return energy.view(B, N)


def generate_simple_image(size=32, target_pos=(0.3, -0.2)):
    """生成一个简单的图像，在 target_pos 位置有一个亮点"""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # 在目标位置创建一个高斯峰值
    dist_sq = (X - target_pos[0])**2 + (Y - target_pos[1])**2
    image = np.exp(-dist_sq / (2 * 0.1**2))
    
    return image, target_pos


def extract_simple_features(image):
    """从图像中提取简单特征（这里用全连接层模拟）"""
    # 在实际 EBM 中，这里会是 CNN 提取的特征
    # 这里我们用一个简单的 MLP 来模拟
    feature_extractor = nn.Sequential(
        nn.Linear(image.size, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
    )
    
    image_flat = torch.FloatTensor(image.flatten()).unsqueeze(0)  # (1, size*size)
    with torch.no_grad():
        features = feature_extractor(image_flat)
    return features, feature_extractor


def visualize_2d_energy_landscape():
    """可视化 2D 能量景观"""
    # 生成测试图像
    image, target_pos = generate_simple_image(size=32, target_pos=(0.3, -0.2))
    
    # 提取特征
    features, feature_extractor = extract_simple_features(image)
    
    # 创建 EBM 模型
    model = Simple2DEBM(feature_dim=32, hidden_dim=64)
    
    # 创建 2D 坐标网格
    resolution = 50
    x_coords = np.linspace(-1, 1, resolution)
    y_coords = np.linspace(-1, 1, resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    # 计算每个坐标的能量值
    coords_list = []
    for i in range(resolution):
        for j in range(resolution):
            coords_list.append([X_grid[i, j], Y_grid[i, j]])
    
    coords_tensor = torch.FloatTensor(np.array(coords_list)).unsqueeze(0)  # (1, resolution*resolution, 2)
    
    model.eval()
    with torch.no_grad():
        energies = model(features, coords_tensor)  # (1, resolution*resolution)
        energies = energies.squeeze(0).numpy()
    
    # 重塑为网格形状
    energy_grid = energies.reshape(resolution, resolution)
    
    # 可视化
    fig = plt.figure(figsize=(15, 5))
    
    # 左图：输入图像
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(image, extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
    ax1.plot(target_pos[0], target_pos[1], 'b*', markersize=15, label='Target Position')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Input Image (Target Position Marked with Blue Star)')
    ax1.legend()
    plt.colorbar(im1, ax=ax1)
    
    # 中图：能量景观热力图
    ax2 = fig.add_subplot(132)
    im2 = ax2.contourf(X_grid, Y_grid, energy_grid, levels=20, cmap='viridis')
    ax2.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target Position')
    # 标记能量最低点
    min_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
    min_pos = (X_grid[min_idx], Y_grid[min_idx])
    ax2.plot(min_pos[0], min_pos[1], 'yo', markersize=10, label='Min Energy Point')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('2D Energy Landscape (Heatmap)')
    ax2.legend()
    plt.colorbar(im2, ax=ax2)
    
    # 右图：3D 能量表面
    ax3 = fig.add_subplot(133, projection='3d')
    surf = ax3.plot_surface(X_grid, Y_grid, energy_grid, cmap='viridis', 
                           alpha=0.8, linewidth=0, antialiased=True)
    ax3.scatter([target_pos[0]], [target_pos[1]], 
               [energy_grid[min_idx]], color='r', s=100, marker='*', label='Target Position')
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    ax3.set_zlabel('Energy Value')
    ax3.set_title('3D Energy Surface')
    ax3.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ebm_2d_energy_landscape.png', dpi=150, bbox_inches='tight')
    print(f"目标位置: {target_pos}")
    print(f"能量最低点位置: ({min_pos[0]:.3f}, {min_pos[1]:.3f})")
    print(f"能量最低值: {energy_grid[min_idx]:.3f}")
    plt.show()


def visualize_energy_contours_with_trajectory():
    """可视化能量等高线图，并显示优化轨迹"""
    # 生成测试图像
    image, target_pos = generate_simple_image(size=32, target_pos=(0.2, 0.4))
    
    # 提取特征
    features, _ = extract_simple_features(image)
    
    # 创建 EBM 模型
    model = Simple2DEBM(feature_dim=32, hidden_dim=64)
    
    # 创建坐标网格
    resolution = 50
    x_coords = np.linspace(-1, 1, resolution)
    y_coords = np.linspace(-1, 1, resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    coords_list = []
    for i in range(resolution):
        for j in range(resolution):
            coords_list.append([X_grid[i, j], Y_grid[i, j]])
    
    coords_tensor = torch.FloatTensor(np.array(coords_list)).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        energies = model(features, coords_tensor)
        energies = energies.squeeze(0).numpy()
    
    energy_grid = energies.reshape(resolution, resolution)
    
    # 模拟优化过程（梯度下降）
    num_steps = 30
    lr = 0.05
    trajectory = []
    
    # 随机初始化
    current_pos = torch.tensor([[-0.8, -0.8]], requires_grad=True)
    
    for step in range(num_steps):
        trajectory.append(current_pos.detach().clone().numpy()[0])
        
        # 计算当前点的能量
        energy = model(features, current_pos.unsqueeze(0))
        loss = energy.sum()
        
        # 反向传播
        loss.backward()
        
        # 梯度下降更新
        with torch.no_grad():
            current_pos -= lr * current_pos.grad
            current_pos.grad.zero_()
            # 限制在 [-1, 1] 范围内
            current_pos.clamp_(-1, 1)
    
    trajectory = np.array(trajectory)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：能量等高线 + 优化轨迹
    ax1 = axes[0]
    contour = ax1.contour(X_grid, Y_grid, energy_grid, levels=15, cmap='viridis', alpha=0.6)
    ax1.contourf(X_grid, Y_grid, energy_grid, levels=15, cmap='viridis', alpha=0.3)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', linewidth=2, 
            markersize=4, label='Optimization Trajectory', alpha=0.7)
    ax1.plot(target_pos[0], target_pos[1], 'b*', markersize=15, label='Target Position')
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start Point')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End Point')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Energy Contour + Optimization Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    plt.colorbar(contour, ax=ax1)
    
    # 右图：能量值随迭代的变化
    ax2 = axes[1]
    trajectory_energies = []
    for pos in trajectory:
        pos_tensor = torch.FloatTensor(pos).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            energy = model(features, pos_tensor)
            trajectory_energies.append(energy.item())
    
    ax2.plot(range(num_steps), trajectory_energies, 'b-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Iteration Step')
    ax2.set_ylabel('Energy Value')
    ax2.set_title('Energy Value Change During Optimization')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ebm_2d_optimization_trajectory.png', dpi=150, bbox_inches='tight')
    print(f"起始位置: ({trajectory[0, 0]:.3f}, {trajectory[0, 1]:.3f})")
    print(f"终点位置: ({trajectory[-1, 0]:.3f}, {trajectory[-1, 1]:.3f})")
    print(f"目标位置: {target_pos}")
    print(f"最终能量: {trajectory_energies[-1]:.3f}")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("2D 能量景观可视化")
    print("=" * 60)
    
    print("\n1. 可视化 2D 能量景观...")
    visualize_2d_energy_landscape()
    
    print("\n2. 可视化优化轨迹...")
    visualize_energy_contours_with_trajectory()
    
    print("\n完成！")

