"""
1D-1D 能量函数可视化
理解能量函数 E(x, y) 的形状和特性

任务：给定一个固定的 1D 输入 x，可视化能量函数 E(x, y) 关于 y 的变化
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
script_name = Path(__file__).stem
output_dir = Path(__file__).parent / 'plots' / script_name
output_dir.mkdir(parents=True, exist_ok=True)


class Simple1DEBM(nn.Module):
    """简化的 1D EBM：输入是 1D 信号，输出是坐标的能量值"""
    
    def __init__(self, signal_dim=100, hidden_dim=64):
        super().__init__()
        # 输入: [flattened_signal, coordinate] -> 能量值
        self.mlp = nn.Sequential(
            nn.Linear(signal_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, y):
        """
        x: (B, signal_dim) - 1D 信号
        y: (B, N) - N 个候选坐标
        返回: (B, N) - 每个坐标的能量值
        """
        B, signal_dim = x.shape
        N = y.shape[1] if y.ndim > 1 else 1
        
        # 扩展 x 以匹配 y 的数量
        x_expanded = x.unsqueeze(1).expand(B, N, signal_dim)  # (B, N, signal_dim)
        # 确保 y 是 3 维的 (B, N, 1)
        if y.ndim == 1:
            y_expanded = y.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        elif y.ndim == 2:
            y_expanded = y.unsqueeze(-1)  # (B, N, 1)
        else:
            y_expanded = y  # 已经是 (B, N, 1)
        
        # 拼接特征
        fused = torch.cat([x_expanded, y_expanded], dim=-1)  # (B, N, signal_dim+1)
        fused = fused.reshape(B * N, signal_dim + 1)
        
        # 计算能量
        energy = self.mlp(fused)  # (B*N, 1)
        return energy.view(B, N)


def generate_1d_signal(signal_dim=100, peak_pos=0.3):
    """生成一个简单的 1D 信号，在 peak_pos 位置有一个峰值"""
    x = np.linspace(-1, 1, signal_dim)
    signal = np.exp(-((x - peak_pos) ** 2) / (2 * 0.1 ** 2))
    return signal


def visualize_energy_function():
    """可视化能量函数"""
    # 创建模型
    model = Simple1DEBM(signal_dim=100, hidden_dim=64)
    
    # 生成一个固定的 1D 输入信号（在 0.3 位置有峰值）
    signal = generate_1d_signal(signal_dim=100, peak_pos=0.3)
    signal_tensor = torch.FloatTensor(signal).unsqueeze(0)  # (1, 100)
    
    # 生成候选坐标范围
    y_candidates = torch.linspace(-1, 1, 200).unsqueeze(0)  # (1, 200)
    
    # 计算能量值
    model.eval()
    with torch.no_grad():
        energies = model(signal_tensor, y_candidates)  # (1, 200)
        energies = energies.squeeze(0).numpy()
    
    y_candidates_np = y_candidates.squeeze(0).numpy()
    
    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 上图：1D 输入信号
    axes[0].plot(np.linspace(-1, 1, 100), signal, 'b-', linewidth=2, label='Input Signal')
    axes[0].axvline(0.3, color='r', linestyle='--', linewidth=2, label='True Peak Position')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Signal Intensity')
    axes[0].set_title('1D Input Signal (Peak at x=0.3)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 下图：能量函数
    axes[1].plot(y_candidates_np, energies, 'g-', linewidth=2, label='Energy Function E(x, y)')
    axes[1].axvline(0.3, color='r', linestyle='--', linewidth=2, label='True Position')
    # 标记能量最低点
    min_idx = np.argmin(energies)
    min_y = y_candidates_np[min_idx]
    min_energy = energies[min_idx]
    axes[1].plot(min_y, min_energy, 'ro', markersize=10, label=f'Min Energy Point ({min_y:.3f})')
    axes[1].set_xlabel('Candidate Coordinate y')
    axes[1].set_ylabel('Energy Value E(x, y)')
    axes[1].set_title('Energy Function: Lower Energy = More Likely Correct Answer')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()  # 反转 y 轴，使低能量在顶部
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ebm_1d_energy.png', dpi=150, bbox_inches='tight')
    print(f"能量最低点位置: {min_y:.3f} (真实位置: 0.3)")
    print(f"能量最低值: {min_energy:.3f}")
    plt.show()


def visualize_training_process():
    """可视化训练过程中能量函数的变化"""
    # 创建模型
    model = Simple1DEBM(signal_dim=100, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 生成训练数据
    num_samples = 50
    signals = []
    targets = []
    for _ in range(num_samples):
        peak_pos = np.random.uniform(-0.8, 0.8)
        signal = generate_1d_signal(peak_pos=peak_pos)
        signals.append(signal)
        targets.append(peak_pos)
    
    signals = torch.FloatTensor(np.array(signals))  # (num_samples, 100)
    targets = torch.FloatTensor(np.array(targets))  # (num_samples,)
    
    # 训练
    num_epochs = 100
    y_candidates = torch.linspace(-1, 1, 200).unsqueeze(0)
    test_signal = generate_1d_signal(peak_pos=0.3)
    test_signal_tensor = torch.FloatTensor(test_signal).unsqueeze(0)
    
    energy_curves = []
    epochs_to_plot = [0, 10, 30, 50, 80, 99]
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 为每个样本生成正负样本
        batch_size = signals.shape[0]
        num_negatives = 10
        
        # 正样本：真实位置
        positive_coords = targets.unsqueeze(1)  # (batch_size, 1)
        
        # 负样本：随机采样
        negative_coords = torch.rand(batch_size, num_negatives) * 2 - 1  # (batch_size, num_negatives)
        
        # 合并正负样本
        all_coords = torch.cat([positive_coords, negative_coords], dim=1)  # (batch_size, 1+num_negatives)
        
        # 计算能量
        energies = model(signals, all_coords)  # (batch_size, 1+num_negatives)
        
        # InfoNCE 损失：正样本索引为 0
        logits = -energies  # 能量转 logits
        loss = F.cross_entropy(logits, torch.zeros(batch_size, dtype=torch.long))
        
        loss.backward()
        optimizer.step()
        
        # 记录能量曲线
        if epoch in epochs_to_plot:
            model.eval()
            with torch.no_grad():
                test_energies = model(test_signal_tensor, y_candidates)
                energy_curves.append((epoch, test_energies.squeeze(0).numpy()))
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # 可视化训练过程
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 上图：输入信号
    axes[0].plot(np.linspace(-1, 1, 100), test_signal, 'b-', linewidth=2)
    axes[0].axvline(0.3, color='r', linestyle='--', linewidth=2, label='True Position')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Signal Intensity')
    axes[0].set_title('Test Signal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 下图：不同训练阶段的能量函数
    y_candidates_np = y_candidates.squeeze(0).numpy()
    colors = plt.cm.viridis(np.linspace(0, 1, len(energy_curves)))
    
    for (epoch, energies), color in zip(energy_curves, colors):
        axes[1].plot(y_candidates_np, energies, linewidth=2, 
                    label=f'Epoch {epoch}', color=color)
    
    axes[1].axvline(0.3, color='r', linestyle='--', linewidth=2, label='True Position')
    axes[1].set_xlabel('Candidate Coordinate y')
    axes[1].set_ylabel('Energy Value E(x, y)')
    axes[1].set_title('Energy Function Evolution During Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ebm_1d_training.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("1D-1D 能量函数可视化")
    print("=" * 60)
    
    print("\n1. 可视化能量函数形状...")
    visualize_energy_function()
    
    print("\n2. 可视化训练过程...")
    visualize_training_process()
    
    print("\n完成！")

