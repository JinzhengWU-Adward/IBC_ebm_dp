"""
EBM 优化过程可视化
可视化推理时的优化轨迹（Derivative-Free Optimizer）

任务：在 2D 坐标空间中绘制优化迭代过程
"""
import numpy as np
import matplotlib.pyplot as plt
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


class DerivativeFreeOptimizer:
    """无导数优化器（重要性采样 + 噪声）"""
    
    def __init__(self, num_samples=100, num_iters=50, 
                 initial_noise_scale=0.5, noise_shrink=0.95):
        self.num_samples = num_samples
        self.num_iters = num_iters
        self.initial_noise_scale = initial_noise_scale
        self.noise_shrink = noise_shrink
    
    def optimize(self, ebm, features, initial_samples=None):
        """
        优化过程：
        1. 计算概率: p_i = exp(-E(x, y_i)) / sum(exp(-E(x, y_j)))
        2. 重采样: 按概率分布采样新候选
        3. 添加噪声: y_new = y_sampled + ε * N(0, 1)
        4. 噪声衰减: ε_{t+1} = ε_t * α
        """
        device = features.device
        B = features.shape[0]
        
        # 初始化候选样本
        if initial_samples is None:
            samples = torch.rand(B, self.num_samples, 2, device=device) * 2 - 1  # [-1, 1]
        else:
            samples = initial_samples.clone()
        
        noise_scale = self.initial_noise_scale
        
        # 存储优化历史
        history = {
            'samples': [],
            'energies': [],
            'best_samples': [],
            'noise_scales': []
        }
        
        ebm.eval()
        with torch.no_grad():
            for iter in range(self.num_iters):
                # 计算能量
                energies = ebm(features, samples)  # (B, num_samples)
                
                # 转换为概率（能量越低，概率越高）
                probs = F.softmax(-energies, dim=-1)  # (B, num_samples)
                
                # 重要性采样：按概率分布重采样
                indices = torch.multinomial(probs, self.num_samples, replacement=True)  # (B, num_samples)
                samples = samples[torch.arange(B).unsqueeze(1), indices]  # (B, num_samples, 2)
                
                # 添加噪声
                noise = torch.randn_like(samples) * noise_scale
                samples = samples + noise
                
                # 限制在 [-1, 1] 范围内
                samples = torch.clamp(samples, -1, 1)
                
                # 记录历史
                history['samples'].append(samples.clone().cpu().numpy())
                history['energies'].append(energies.cpu().numpy())
                # 记录能量最低的样本
                best_indices = energies.argmin(dim=1)
                best_samples = samples[torch.arange(B), best_indices]
                history['best_samples'].append(best_samples.clone().cpu().numpy())
                history['noise_scales'].append(noise_scale)
                
                # 噪声衰减
                noise_scale *= self.noise_shrink
        
        return samples, history


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


def visualize_optimization_process():
    """可视化优化过程"""
    # 生成测试图像
    image, target_pos = generate_simple_image(size=32, target_pos=(0.2, 0.4))
    features, _ = extract_simple_features(image)
    
    # 创建 EBM（使用预训练的特征提取器，但随机初始化 EBM）
    ebm = Simple2DEBM(feature_dim=32, hidden_dim=64)
    
    # 创建优化器
    optimizer = DerivativeFreeOptimizer(
        num_samples=100,
        num_iters=50,
        initial_noise_scale=0.5,
        noise_shrink=0.95
    )
    
    # 执行优化
    final_samples, history = optimizer.optimize(ebm, features)
    
    # 计算能量景观（用于背景）
    resolution = 50
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
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # 左上：优化轨迹（显示几个关键迭代）
    ax1 = axes[0, 0]
    ax1.contourf(X_grid, Y_grid, energy_grid, levels=15, cmap='viridis', alpha=0.3)
    ax1.contour(X_grid, Y_grid, energy_grid, levels=15, colors='gray', alpha=0.3, linewidths=0.5)
    
    # 显示几个关键迭代的样本分布
    iterations_to_show = [0, 10, 20, 30, 40, 49]
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(iterations_to_show)))
    
    for idx, iter in enumerate(iterations_to_show):
        samples = history['samples'][iter][0]  # (num_samples, 2)
        ax1.scatter(samples[:, 0], samples[:, 1], s=20, alpha=0.5, 
                   color=colors[idx], label=f'Iter {iter}')
    
    # 显示最佳样本轨迹
    best_trajectory = np.array([h[0] for h in history['best_samples']])
    ax1.plot(best_trajectory[:, 0], best_trajectory[:, 1], 'r-', 
            linewidth=2, alpha=0.7, label='Best Sample Trajectory')
    ax1.plot(target_pos[0], target_pos[1], 'b*', markersize=15, label='Target Position')
    ax1.plot(best_trajectory[0, 0], best_trajectory[0, 1], 'go', 
            markersize=10, label='Start Point')
    ax1.plot(best_trajectory[-1, 0], best_trajectory[-1, 1], 'ro', 
            markersize=10, label='End Point')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Optimization Process: Sample Distribution Evolution')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 右上：能量值随迭代的变化
    ax2 = axes[0, 1]
    best_energies = [h[0][h[0].argmin()] for h in history['energies']]
    mean_energies = [h[0].mean() for h in history['energies']]
    min_energies = [h[0].min() for h in history['energies']]
    
    ax2.plot(range(len(best_energies)), best_energies, 'r-', 
            linewidth=2, label='Best Sample Energy')
    ax2.plot(range(len(mean_energies)), mean_energies, 'b--', 
            linewidth=2, alpha=0.7, label='Mean Energy')
    ax2.plot(range(len(min_energies)), min_energies, 'g-', 
            linewidth=2, alpha=0.7, label='Min Energy')
    ax2.set_xlabel('Iteration Step')
    ax2.set_ylabel('Energy Value')
    ax2.set_title('Energy Value Change During Iterations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 左下：样本分布的标准差（收敛性指标）
    ax3 = axes[1, 0]
    std_x = [np.std(h[0][:, 0]) for h in history['samples']]
    std_y = [np.std(h[0][:, 1]) for h in history['samples']]
    
    ax3.plot(range(len(std_x)), std_x, 'b-', linewidth=2, label='X Direction Std')
    ax3.plot(range(len(std_y)), std_y, 'r-', linewidth=2, label='Y Direction Std')
    ax3.set_xlabel('Iteration Step')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Sample Distribution Standard Deviation (Convergence)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 右下：噪声尺度变化
    ax4 = axes[1, 1]
    ax4.plot(range(len(history['noise_scales'])), history['noise_scales'], 
            'g-', linewidth=2)
    ax4.set_xlabel('Iteration Step')
    ax4.set_ylabel('Noise Scale')
    ax4.set_title('Noise Scale Decay')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ebm_optimization_demo.png', dpi=150, bbox_inches='tight')
    
    print(f"目标位置: {target_pos}")
    print(f"最终最佳位置: ({best_trajectory[-1, 0]:.3f}, {best_trajectory[-1, 1]:.3f})")
    print(f"最终能量: {best_energies[-1]:.3f}")
    print(f"距离误差: {np.sqrt(np.sum((best_trajectory[-1] - np.array(target_pos))**2)):.4f}")
    
    plt.show()


def visualize_optimization_comparison():
    """比较不同优化器参数的效果"""
    image, target_pos = generate_simple_image(size=32, target_pos=(0.0, 0.0))
    features, _ = extract_simple_features(image)
    ebm = Simple2DEBM(feature_dim=32, hidden_dim=64)
    
    # 不同的优化器配置
    configs = [
        {'initial_noise_scale': 0.3, 'noise_shrink': 0.98, 'label': 'small noise, slow decay'},
        {'initial_noise_scale': 0.5, 'noise_shrink': 0.95, 'label': 'medium noise, medium decay'},
        {'initial_noise_scale': 0.8, 'noise_shrink': 0.90, 'label': 'large noise, fast decay'},
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, config in enumerate(configs):
        optimizer = DerivativeFreeOptimizer(
            num_samples=100,
            num_iters=50,
            initial_noise_scale=config['initial_noise_scale'],
            noise_shrink=config['noise_shrink']
        )
        
        _, history = optimizer.optimize(ebm, features)
        best_trajectory = np.array([h[0] for h in history['best_samples']])
        best_energies = [h[0][h[0].argmin()] for h in history['energies']]
        
        ax = axes[idx]
        ax.plot(best_trajectory[:, 0], best_trajectory[:, 1], 'r-o', 
               linewidth=2, markersize=4, alpha=0.7, label='Optimization Trajectory')
        ax.plot(target_pos[0], target_pos[1], 'b*', markersize=15, label='Target Position')
        ax.plot(best_trajectory[0, 0], best_trajectory[0, 1], 'go', 
               markersize=10, label='Start Point')
        ax.plot(best_trajectory[-1, 0], best_trajectory[-1, 1], 'ro', 
               markersize=10, label='End Point')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(config['label'])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ebm_optimization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("EBM 优化过程可视化")
    print("=" * 60)
    
    print("\n1. 可视化完整优化过程...")
    visualize_optimization_process()
    
    print("\n2. 比较不同优化器参数...")
    visualize_optimization_comparison()
    
    print("\n完成！")

