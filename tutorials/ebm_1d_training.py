"""
简化 1D EBM 训练演示
在 1D 任务上完整演示 EBM 训练过程

任务：
- 输入: 1D 信号 x ∈ R^100
- 输出: 1D 坐标 y ∈ [-1, 1]
- 模型: 简化的 EBM（去掉 CNN，只用 MLP）
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


class Simple1DEBM(nn.Module):
    """简化的 1D EBM"""
    
    def __init__(self, signal_dim=100, hidden_dim=64):
        super().__init__()
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


class Simple1DDataset(Dataset):
    """简单的 1D 数据集"""
    
    def __init__(self, num_samples=200, signal_dim=100):
        self.num_samples = num_samples
        self.signal_dim = signal_dim
        self.data = []
        self.targets = []
        
        # 生成数据
        for _ in range(num_samples):
            # 随机生成峰值位置
            peak_pos = np.random.uniform(-0.8, 0.8)
            
            # 生成 1D 信号
            x = np.linspace(-1, 1, signal_dim)
            signal = np.exp(-((x - peak_pos) ** 2) / (2 * 0.1 ** 2))
            
            self.data.append(signal)
            self.targets.append(peak_pos)
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor([self.targets[idx]])


def train_1d_ebm():
    """训练 1D EBM"""
    # 设置参数
    signal_dim = 100
    hidden_dim = 64
    num_negatives = 20
    batch_size = 32
    num_epochs = 100
    lr = 0.001
    
    # 创建数据集
    train_dataset = Simple1DDataset(num_samples=200, signal_dim=signal_dim)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = Simple1DEBM(signal_dim=signal_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练历史
    train_losses = []
    
    # 用于可视化的测试样本
    test_signal, test_target = train_dataset[0]
    test_signal = test_signal.unsqueeze(0)
    test_target = test_target.item()
    
    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, (signals, targets) in enumerate(train_loader):
            B = signals.shape[0]
            
            # 正样本：真实位置
            positive_coords = targets  # (B, 1)
            
            # 负样本：随机采样
            negative_coords = torch.rand(B, num_negatives) * 2 - 1  # (B, num_negatives)
            
            # 合并正负样本
            all_coords = torch.cat([positive_coords, negative_coords], dim=1)  # (B, 1+num_negatives)
            
            # 计算能量
            energies = model(signals, all_coords)  # (B, 1+num_negatives)
            
            # InfoNCE 损失：正样本索引为 0
            logits = -energies  # 能量转 logits
            loss = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model, train_losses, test_signal, test_target


def visualize_training_results(model, train_losses, test_signal, test_target):
    """可视化训练结果"""
    model.eval()
    
    # 计算测试样本的能量函数
    y_candidates = torch.linspace(-1, 1, 200).unsqueeze(0)
    
    with torch.no_grad():
        energies = model(test_signal, y_candidates)
        energies = energies.squeeze(0).numpy()
    
    y_candidates_np = y_candidates.squeeze(0).numpy()
    test_signal_np = test_signal.squeeze(0).numpy()
    
    # 找到能量最低点
    min_idx = np.argmin(energies)
    pred_pos = y_candidates_np[min_idx]
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上：训练损失
    ax1 = axes[0, 0]
    ax1.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, alpha=0.3)
    
    # 右上：测试信号
    ax2 = axes[0, 1]
    x_coords = np.linspace(-1, 1, len(test_signal_np))
    ax2.plot(x_coords, test_signal_np, 'b-', linewidth=2, label='Input Signal')
    ax2.axvline(test_target, color='g', linestyle='--', linewidth=2, label=f'True Position ({test_target:.3f})')
    ax2.axvline(pred_pos, color='r', linestyle='--', linewidth=2, label=f'Predicted Position ({pred_pos:.3f})')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Signal Intensity')
    ax2.set_title('Test Signal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 左下：能量函数
    ax3 = axes[1, 0]
    ax3.plot(y_candidates_np, energies, 'g-', linewidth=2, label='Energy Function')
    ax3.axvline(test_target, color='g', linestyle='--', linewidth=2, label='True Position')
    ax3.axvline(pred_pos, color='r', linestyle='--', linewidth=2, label='Predicted Position')
    ax3.plot(pred_pos, energies[min_idx], 'ro', markersize=10, label='Min Energy Point')
    ax3.set_xlabel('Candidate Coordinate y')
    ax3.set_ylabel('Energy Value E(x, y)')
    ax3.set_title('Energy Function')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    
    # 右下：预测误差分布（在测试集上）
    ax4 = axes[1, 1]
    test_dataset = Simple1DDataset(num_samples=50, signal_dim=100)
    errors = []
    
    with torch.no_grad():
        for signal, target in test_dataset:
            signal = signal.unsqueeze(0)
            target = target.item()
            
            energies = model(signal, y_candidates)
            energies = energies.squeeze(0).numpy()
            pred_pos = y_candidates_np[np.argmin(energies)]
            
            error = abs(pred_pos - target)
            errors.append(error)
    
    ax4.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Prediction Error')
    ax4.set_ylabel('Number of Samples')
    ax4.set_title(f'Test Set Prediction Error Distribution\nMean Error: {np.mean(errors):.4f}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ebm_1d_training_results.png', dpi=150, bbox_inches='tight')
    
    print(f"\n测试结果:")
    print(f"真实位置: {test_target:.3f}")
    print(f"预测位置: {pred_pos:.3f}")
    print(f"预测误差: {abs(pred_pos - test_target):.4f}")
    print(f"测试集平均误差: {np.mean(errors):.4f}")
    
    plt.show()


def visualize_training_evolution(model, train_losses, test_signal, test_target):
    """可视化训练过程中能量函数的演化"""
    y_candidates = torch.linspace(-1, 1, 200).unsqueeze(0)
    
    # 保存不同训练阶段的模型状态
    epochs_to_plot = [0, 20, 40, 60, 80, 99]
    energy_curves = []
    
    # 重新训练并保存中间状态
    signal_dim = 100
    hidden_dim = 64
    num_negatives = 20
    batch_size = 32
    num_epochs = 100
    lr = 0.001
    
    train_dataset = Simple1DDataset(num_samples=200, signal_dim=signal_dim)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = Simple1DEBM(signal_dim=signal_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (signals, targets) in enumerate(train_loader):
            B = signals.shape[0]
            positive_coords = targets
            negative_coords = torch.rand(B, num_negatives) * 2 - 1
            all_coords = torch.cat([positive_coords, negative_coords], dim=1)
            energies = model(signals, all_coords)
            logits = -energies
            loss = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 记录能量曲线
        if epoch in epochs_to_plot:
            model.eval()
            with torch.no_grad():
                energies = model(test_signal, y_candidates)
                energy_curves.append((epoch, energies.squeeze(0).numpy()))
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    y_candidates_np = y_candidates.squeeze(0).numpy()
    colors = plt.cm.viridis(np.linspace(0, 1, len(energy_curves)))
    
    # 左图：能量函数演化
    ax1 = axes[0]
    for (epoch, energies), color in zip(energy_curves, colors):
        ax1.plot(y_candidates_np, energies, linewidth=2, 
                label=f'Epoch {epoch+1}', color=color)
    ax1.axvline(test_target, color='r', linestyle='--', linewidth=2, label='True Position')
    ax1.set_xlabel('Candidate Coordinate y')
    ax1.set_ylabel('Energy Value E(x, y)')
    ax1.set_title('Energy Function Evolution During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # 右图：预测位置随训练的变化
    ax2 = axes[1]
    pred_positions = []
    for epoch, energies in energy_curves:
        pred_pos = y_candidates_np[np.argmin(energies)]
        pred_positions.append(pred_pos)
    
    ax2.plot([e+1 for e, _ in energy_curves], pred_positions, 'b-o', 
            linewidth=2, markersize=8, label='Predicted Position')
    ax2.axhline(test_target, color='r', linestyle='--', linewidth=2, label='True Position')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Predicted Position')
    ax2.set_title('Predicted Position Change During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ebm_1d_training_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("简化 1D EBM 训练演示")
    print("=" * 60)
    
    print("\n1. 训练模型...")
    model, train_losses, test_signal, test_target = train_1d_ebm()
    
    print("\n2. 可视化训练结果...")
    visualize_training_results(model, train_losses, test_signal, test_target)
    
    print("\n3. 可视化训练演化过程...")
    visualize_training_evolution(model, train_losses, test_signal, test_target)
    
    print("\n完成！")

