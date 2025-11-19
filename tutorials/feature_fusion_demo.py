"""
特征融合效果可视化
理解图像特征和坐标如何影响能量

任务：固定图像特征，改变坐标，观察能量变化
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
    """简化的 2D EBM"""
    
    def __init__(self, feature_dim=32, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features, coords):
        B, feature_dim = features.shape
        N = coords.shape[1]
        features_expanded = features.unsqueeze(1).expand(B, N, feature_dim)
        fused = torch.cat([features_expanded, coords], dim=-1)
        fused = fused.reshape(B * N, feature_dim + 2)
        energy = self.mlp(fused)
        return energy.view(B, N)


def generate_simple_image(size=32, target_pos=(0.3, -0.2)):
    """生成测试图像"""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    dist_sq = (X - target_pos[0])**2 + (Y - target_pos[1])**2
    image = np.exp(-dist_sq / (2 * 0.1**2))
    return image, target_pos


def extract_simple_features(image):
    """提取特征"""
    feature_extractor = nn.Sequential(
        nn.Linear(image.size, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
    )
    image_flat = torch.FloatTensor(image.flatten()).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(image_flat)
    return features, feature_extractor


def visualize_feature_fusion_effect():
    """可视化特征融合的效果"""
    # 生成测试图像
    image, target_pos = generate_simple_image(size=32, target_pos=(0.3, -0.2))
    features, _ = extract_simple_features(image)
    
    # 创建 EBM
    ebm = Simple2DEBM(feature_dim=32, hidden_dim=64)
    
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
    
    # 计算能量
    ebm.eval()
    with torch.no_grad():
        energies = ebm(features, coords_tensor)
        energies = energies.squeeze(0).numpy()
    
    energy_grid = energies.reshape(resolution, resolution)
    
    # 可视化特征融合的过程
    fig = plt.figure(figsize=(16, 4))
    
    # 左图：输入图像
    ax1 = fig.add_subplot(141)
    im1 = ax1.imshow(image, extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
    ax1.plot(target_pos[0], target_pos[1], 'b*', markersize=15)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Input Image\n(Extract Features φ(x))')
    plt.colorbar(im1, ax=ax1)
    
    # 中左图：特征向量可视化（降维到2D显示）
    ax2 = fig.add_subplot(142)
    feature_vec = features.squeeze(0).numpy()
    # 使用 PCA 或简单的前两个维度
    ax2.bar(range(len(feature_vec)), feature_vec, alpha=0.7)
    ax2.set_xlabel('Feature Dimension')
    ax2.set_ylabel('Feature Value')
    ax2.set_title(f'Image Features φ(x)\n(Dimension: {len(feature_vec)})')
    ax2.grid(True, alpha=0.3)
    
    # 中右图：特征融合示意图
    ax3 = fig.add_subplot(143)
    # 绘制融合过程示意图
    ax3.text(0.5, 0.8, 'feature fusion', ha='center', fontsize=14, weight='bold', transform=ax3.transAxes)
    ax3.text(0.5, 0.6, 'φ(x) [32D]', ha='center', fontsize=10, transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax3.text(0.5, 0.5, '    +    ', ha='center', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.5, 0.4, 'y [2D]', ha='center', fontsize=10, transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax3.text(0.5, 0.3, '    =    ', ha='center', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.5, 0.2, '[φ(x), y] [34D]', ha='center', fontsize=10, transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Feature Fusion Process')
    
    # 右图：能量景观
    ax4 = fig.add_subplot(144)
    im4 = ax4.contourf(X_grid, Y_grid, energy_grid, levels=20, cmap='viridis')
    ax4.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target Position')
    min_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
    min_pos = (X_grid[min_idx], Y_grid[min_idx])
    ax4.plot(min_pos[0], min_pos[1], 'yo', markersize=10, label='Min Energy Point')
    ax4.set_xlabel('X Coordinate')
    ax4.set_ylabel('Y Coordinate')
    ax4.set_title('Energy Landscape E(φ(x), y)')
    ax4.legend()
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_fusion_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_energy_surface_3d():
    """3D 可视化能量表面"""
    image, target_pos = generate_simple_image(size=32, target_pos=(0.2, 0.3))
    features, _ = extract_simple_features(image)
    ebm = Simple2DEBM(feature_dim=32, hidden_dim=64)
    
    resolution = 40
    x_coords = np.linspace(-1, 1, resolution)
    y_coords = np.linspace(-1, 1, resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    coords_list = []
    for i in range(resolution):
        for j in range(resolution):
            coords_list.append([X_grid[i, j], Y_grid[i, j]])
    
    coords_tensor = torch.FloatTensor(np.array(coords_list)).unsqueeze(0)
    
    ebm.eval()
    with torch.no_grad():
        energies = ebm(features, coords_tensor)
        energies = energies.squeeze(0).numpy()
    
    energy_grid = energies.reshape(resolution, resolution)
    
    # 3D 可视化
    fig = plt.figure(figsize=(14, 6))
    
    # 左图：3D 表面
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X_grid, Y_grid, energy_grid, cmap='viridis', 
                           alpha=0.8, linewidth=0, antialiased=True)
    ax1.scatter([target_pos[0]], [target_pos[1]], 
               [energy_grid.min()], color='r', s=100, marker='*', label='Target Position')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Energy Value')
    ax1.set_title('3D Energy Surface')
    ax1.view_init(elev=30, azim=45)
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 右图：等高线图
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X_grid, Y_grid, energy_grid, levels=20, cmap='viridis')
    ax2.contourf(X_grid, Y_grid, energy_grid, levels=20, cmap='viridis', alpha=0.6)
    ax2.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target Position')
    min_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
    min_pos = (X_grid[min_idx], Y_grid[min_idx])
    ax2.plot(min_pos[0], min_pos[1], 'yo', markersize=10, label='Min Energy Point')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('Energy Contour Map')
    ax2.legend()
    ax2.set_aspect('equal')
    plt.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_fusion_3d.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_different_features():
    """可视化不同图像特征对应的能量景观"""
    # 生成多个不同的图像
    images = []
    target_positions = [
        (0.3, -0.2),
        (-0.4, 0.5),
        (0.6, 0.3),
        (-0.2, -0.6)
    ]
    
    for target_pos in target_positions:
        image, _ = generate_simple_image(size=32, target_pos=target_pos)
        images.append(image)
    
    # 提取特征
    feature_extractor = nn.Sequential(
        nn.Linear(32*32, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
    )
    
    features_list = []
    for image in images:
        image_flat = torch.FloatTensor(image.flatten()).unsqueeze(0)
        with torch.no_grad():
            features = feature_extractor(image_flat)
        features_list.append(features)
    
    # 创建 EBM
    ebm = Simple2DEBM(feature_dim=32, hidden_dim=64)
    
    # 计算每个图像的能量景观
    resolution = 30
    x_coords = np.linspace(-1, 1, resolution)
    y_coords = np.linspace(-1, 1, resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    coords_list = []
    for i in range(resolution):
        for j in range(resolution):
            coords_list.append([X_grid[i, j], Y_grid[i, j]])
    
    coords_tensor = torch.FloatTensor(np.array(coords_list)).unsqueeze(0)
    
    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    ebm.eval()
    with torch.no_grad():
        for idx, (image, features, target_pos) in enumerate(zip(images, features_list, target_positions)):
            # 计算能量
            energies = ebm(features, coords_tensor)
            energies = energies.squeeze(0).numpy()
            energy_grid = energies.reshape(resolution, resolution)
            
            # 上图：输入图像
            ax1 = axes[0, idx]
            im1 = ax1.imshow(image, extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
            ax1.plot(target_pos[0], target_pos[1], 'b*', markersize=10)
            ax1.set_title(f'Image {idx+1}')
            plt.colorbar(im1, ax=ax1)
            
            # 下图：能量景观
            ax2 = axes[1, idx]
            im2 = ax2.contourf(X_grid, Y_grid, energy_grid, levels=15, cmap='viridis')
            ax2.plot(target_pos[0], target_pos[1], 'r*', markersize=10, label='Target Position')
            min_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
            min_pos = (X_grid[min_idx], Y_grid[min_idx])
            ax2.plot(min_pos[0], min_pos[1], 'yo', markersize=8, label='Min Energy Point')
            ax2.set_title(f'Energy Landscape {idx+1}')
            ax2.set_aspect('equal')
            plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_fusion_multiple.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("特征融合效果可视化")
    print("=" * 60)
    
    print("\n1. 可视化特征融合过程...")
    visualize_feature_fusion_effect()
    
    print("\n2. 3D 可视化能量表面...")
    visualize_energy_surface_3d()
    
    print("\n3. 可视化不同图像特征对应的能量景观...")
    visualize_different_features()
    
    print("\n完成！")

