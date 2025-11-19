"""
SpatialSoftArgmax 可视化
理解空间软最大值如何定位特征

任务：可视化特征图及其对应的加权平均位置
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
script_name = Path(__file__).stem
output_dir = Path(__file__).parent / 'plots' / script_name
output_dir.mkdir(parents=True, exist_ok=True)


class SpatialSoftArgmax:
    """SpatialSoftArgmax 的简化实现"""
    
    def __init__(self, normalize=True):
        self.normalize = normalize
    
    def _coord_grid(self, h, w, device):
        """生成归一化的坐标网格（与 numpy.meshgrid 默认行为一致）"""
        if self.normalize:
            x_coords = torch.linspace(-1, 1, w, device=device)
            y_coords = torch.linspace(-1, 1, h, device=device)
        else:
            x_coords = torch.arange(0, w, device=device, dtype=torch.float32)
            y_coords = torch.arange(0, h, device=device, dtype=torch.float32)
        
        # 关键：使用 'xy' 索引模式以匹配 numpy.meshgrid 的默认行为
        try:
            # indexing='xy': X[i,j]=x_coords[j], Y[i,j]=y_coords[i]
            X, Y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        except TypeError:
            # 旧版本 PyTorch 默认是 'ij' 模式，需要手动调整
            # 交换参数顺序并转置
            Y_ij, X_ij = torch.meshgrid(y_coords, x_coords)
            X = X_ij  # X_ij 已经是 X[i,j] = x_coords[j]
            Y = Y_ij  # Y_ij 已经是 Y[i,j] = y_coords[i]
        
        return X, Y

    def forward(self, x):
        """
        x: (B, C, H, W) - 特征图
        返回: (B, C*2) - 每个特征通道的 (x_mean, y_mean)
        """
        B, C, H, W = x.shape
        
        # 计算空间 softmax
        softmax = F.softmax(x.view(B * C, H * W), dim=-1)  # (B*C, H*W)
        
        # 创建坐标网格
        X, Y = self._coord_grid(H, W, x.device)
        
        # 计算加权平均位置
        x_mean = (softmax * X.flatten()).sum(dim=1, keepdims=True)  # (B*C, 1)
        y_mean = (softmax * Y.flatten()).sum(dim=1, keepdims=True)  # (B*C, 1)
        
        # 拼接并重塑
        result = torch.cat([x_mean, y_mean], dim=1).view(B, C * 2)  # (B, C*2)
        return result, softmax.view(B, C, H, W)


def generate_feature_map_with_peak(H=32, W=32, peak_pos=(0.3, -0.2), sigma=0.1, temperature=10.0):
    """生成一个在指定位置有峰值的特征图"""
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(x, y)
    
    # 创建高斯峰值
    dist_sq = (X - peak_pos[0])**2 + (Y - peak_pos[1])**2
    feature_map = np.exp(-dist_sq / (2 * sigma**2))
    
    # 放大特征图的对比度，使 softmax 后权重更集中
    feature_map = feature_map * temperature
    
    return feature_map


def visualize_spatial_softargmax():
    """可视化 SpatialSoftArgmax 的工作原理"""
    # 创建 SpatialSoftArgmax
    spatial_softargmax = SpatialSoftArgmax(normalize=True)
    
    # 生成测试特征图
    H, W = 32, 32
    peak_pos = (0.3, -0.2)
    feature_map = generate_feature_map_with_peak(H, W, peak_pos, sigma=0.15)  # 减小 sigma 使峰值更尖锐
    
    # 转换为 tensor
    feature_tensor = torch.FloatTensor(feature_map).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # 计算 SpatialSoftArgmax
    coords, softmax_weights = spatial_softargmax.forward(feature_tensor)
    x_mean, y_mean = coords[0, 0].item(), coords[0, 1].item()
    
    # 调试信息
    print(f"特征图形状: {feature_map.shape}")
    print(f"特征图最大值: {feature_map.max():.4f}")
    print(f"特征图最大值位置: {np.unravel_index(feature_map.argmax(), feature_map.shape)}")
    print(f"Softmax 权重最大值: {softmax_weights[0, 0].numpy().max():.6f}")
    print(f"Softmax 权重最大值位置: {np.unravel_index(softmax_weights[0, 0].numpy().argmax(), softmax_weights[0, 0].shape)}")
    print(f"Softmax 权重总和: {softmax_weights[0, 0].numpy().sum():.6f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 左图：原始特征图
    ax1 = axes[0]
    im1 = ax1.imshow(feature_map, extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
    ax1.plot(peak_pos[0], peak_pos[1], 'b*', markersize=15, label='True Peak Position')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Original Feature Map')
    ax1.legend()
    plt.colorbar(im1, ax=ax1)
    
    # 中图：Softmax 权重分布
    ax2 = axes[1]
    softmax_map = softmax_weights[0, 0].numpy()
    im2 = ax2.imshow(softmax_map, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    ax2.plot(peak_pos[0], peak_pos[1], 'b*', markersize=15, label='True Peak Position')
    ax2.plot(x_mean, y_mean, 'ro', markersize=12, label=f'Computed Position ({x_mean:.3f}, {y_mean:.3f})')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('Softmax Weight Distribution (for Weighted Average)')
    ax2.legend()
    plt.colorbar(im2, ax=ax2)
    
    # 右图：对比图
    ax3 = axes[2]
    im3 = ax3.imshow(feature_map, extent=[-1, 1, -1, 1], origin='lower', cmap='hot', alpha=0.5)
    # 叠加 softmax 等高线
    X = np.linspace(-1, 1, W)
    Y = np.linspace(-1, 1, H)
    X_grid, Y_grid = np.meshgrid(X, Y)
    ax3.contour(X_grid, Y_grid, softmax_map, levels=10, colors='cyan', alpha=0.6, linewidths=1)
    ax3.plot(peak_pos[0], peak_pos[1], 'b*', markersize=15, label='True Peak Position')
    ax3.plot(x_mean, y_mean, 'ro', markersize=12, label=f'Computed Position ({x_mean:.3f}, {y_mean:.3f})')
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    ax3.set_title('Feature Map + Softmax Contour + Computed Position')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_softargmax_demo.png', dpi=150, bbox_inches='tight')
    print(f"真实峰值位置: {peak_pos}")
    print(f"计算出的位置: ({x_mean:.3f}, {y_mean:.3f})")
    print(f"误差: {np.sqrt((peak_pos[0] - x_mean)**2 + (peak_pos[1] - y_mean)**2):.4f}")
    plt.show()


def visualize_multiple_features():
    """可视化多个特征通道的 SpatialSoftArgmax"""
    # 创建 SpatialSoftArgmax
    spatial_softargmax = SpatialSoftArgmax(normalize=True)
    
    # 生成多个特征图（模拟 CNN 的多个通道）
    H, W = 32, 32
    num_channels = 4
    
    feature_maps = []
    peak_positions = [
        (0.3, -0.2),
        (-0.4, 0.5),
        (0.6, 0.3),
        (-0.2, -0.6)
    ]
    
    for i in range(num_channels):
        peak_pos = peak_positions[i]
        feature_map = generate_feature_map_with_peak(H, W, peak_pos, sigma=0.15, temperature=10.0)  # 减小 sigma 并增加 temperature
        feature_maps.append(feature_map)
    
    # 组合成多通道特征图
    feature_tensor = torch.FloatTensor(np.array(feature_maps)).unsqueeze(0)  # (1, C, H, W)
    
    # 计算 SpatialSoftArgmax
    coords, softmax_weights = spatial_softargmax.forward(feature_tensor)
    
    # 可视化
    fig, axes = plt.subplots(2, num_channels, figsize=(4*num_channels, 8))
    
    for i in range(num_channels):
        # 上图：特征图
        ax1 = axes[0, i]
        im1 = ax1.imshow(feature_maps[i], extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
        peak_pos = peak_positions[i]
        ax1.plot(peak_pos[0], peak_pos[1], 'b*', markersize=10, label='True Position')
        x_mean = coords[0, i*2].item()
        y_mean = coords[0, i*2+1].item()
        ax1.plot(x_mean, y_mean, 'ro', markersize=8, label='Computed Position')
        ax1.set_title(f'Channel {i+1}')
        ax1.legend(fontsize=8)
        plt.colorbar(im1, ax=ax1)
        
        # 下图：Softmax 权重
        ax2 = axes[1, i]
        softmax_map = softmax_weights[0, i].numpy()
        im2 = ax2.imshow(softmax_map, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
        ax2.plot(peak_pos[0], peak_pos[1], 'b*', markersize=10)
        ax2.plot(x_mean, y_mean, 'ro', markersize=8)
        ax2.set_title(f'Softmax Weights {i+1}')
        plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_softargmax_multi_channel.png', dpi=150, bbox_inches='tight')
    
    # 打印结果
    print("\n多通道 SpatialSoftArgmax 结果:")
    for i in range(num_channels):
        x_mean = coords[0, i*2].item()
        y_mean = coords[0, i*2+1].item()
        peak_pos = peak_positions[i]
        error = np.sqrt((peak_pos[0] - x_mean)**2 + (peak_pos[1] - y_mean)**2)
        print(f"通道 {i+1}: 真实位置 {peak_pos}, 计算位置 ({x_mean:.3f}, {y_mean:.3f}), 误差 {error:.4f}")
    
    plt.show()


def visualize_coord_conv_effect():
    """可视化 CoordConv 的效果"""
    H, W = 32, 32
    
    # 创建坐标通道
    y_coords = 2.0 * np.arange(H) / (H - 1.0) - 1.0
    x_coords = 2.0 * np.arange(W) / (W - 1.0) - 1.0
    # 兼容 Python 3.8: 旧版本 NumPy 可能不支持 indexing 参数
    try:
        Y_coord, X_coord = np.meshgrid(y_coords, x_coords, indexing='ij')
    except TypeError:
        # 旧版本 NumPy，需要手动转置
        X_coord, Y_coord = np.meshgrid(x_coords, y_coords)
        Y_coord = Y_coord.T
        X_coord = X_coord.T
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # X 坐标通道
    ax1 = axes[0]
    im1 = ax1.imshow(X_coord, extent=[-1, 1, -1, 1], origin='lower', cmap='RdBu')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('CoordConv: X Coordinate Channel')
    plt.colorbar(im1, ax=ax1)
    
    # Y 坐标通道
    ax2 = axes[1]
    im2 = ax2.imshow(Y_coord, extent=[-1, 1, -1, 1], origin='lower', cmap='RdBu')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('CoordConv: Y Coordinate Channel')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'coord_conv_demo.png', dpi=150, bbox_inches='tight')
    print("\nCoordConv 坐标通道可视化完成")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("SpatialSoftArgmax 可视化")
    print("=" * 60)
    
    print("\n1. 可视化单个特征图的 SpatialSoftArgmax...")
    visualize_spatial_softargmax()
    
    print("\n2. 可视化多个特征通道...")
    visualize_multiple_features()
    
    print("\n3. 可视化 CoordConv 坐标通道...")
    visualize_coord_conv_effect()
    
    print("\n完成！")

