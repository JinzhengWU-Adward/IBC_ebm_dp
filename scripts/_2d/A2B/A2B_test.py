"""
A2B 单步轨迹预测测试脚本
加载训练好的模型并进行逐步轨迹推理和动态可视化
"""
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from tqdm import tqdm
import time

# 导入核心模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.models import SequenceEBM
from core.optimizers import ULASampler

# 导入训练脚本中的类和函数
# 将当前目录添加到sys.path以便导入
sys.path.insert(0, str(Path(__file__).parent))
from A2B_train import A2BDataset, infer_trajectory

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = Path(__file__).parent.parent.parent / 'plots' / 'A2B_test'
output_dir.mkdir(parents=True, exist_ok=True)

# 模型目录
models_dir = Path(__file__).parent.parent.parent / 'models' / '_2d'


def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载训练好的SequenceEBM模型
    
    Args:
        model_path: 模型文件路径
        device: 计算设备
    
    Returns:
        model: 加载的模型
        checkpoint: 模型checkpoint信息
    """
    print(f"加载模型从: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型参数
    obs_dim = checkpoint.get('obs_dim', 2)
    action_dim = checkpoint.get('action_dim', 2)
    obs_seq_len = checkpoint.get('obs_seq_len', 2)
    hidden_dim = checkpoint.get('hidden_dim', 256)
    num_residual_blocks = checkpoint.get('num_residual_blocks', 1)  # Default updated to 1
    dropout = checkpoint.get('dropout', 0.0)  # Default updated to 0.0
    norm_type = checkpoint.get('norm_type', None)  # Default updated to None
    trajectory_length = checkpoint.get('trajectory_length', 99)
    image_size = checkpoint.get('image_size', 64)
    
    print(f"模型参数:")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, obs_seq_len={obs_seq_len}")
    print(f"  hidden_dim={hidden_dim}, num_residual_blocks={num_residual_blocks}")
    print(f"  dropout={dropout}, norm_type={norm_type}")
    print(f"  trajectory_length={trajectory_length}, image_size={image_size}")
    
    # 创建SequenceEBM
    model = SequenceEBM(
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_seq_len=obs_seq_len,
        hidden_dim=hidden_dim,
        num_residual_blocks=num_residual_blocks,
        dropout=dropout,
        norm_type=norm_type
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("模型加载完成！")
    return model, checkpoint


# 注意：infer_trajectory 函数已在 A2B_train.py 中定义并导入

def infer_single_step(
    model: SequenceEBM,
    obs_seq: torch.Tensor,
    ula_sampler: ULASampler,
    num_action_samples: int = 512,  # IBC 默认值
    temperature: float = 1.0,  # IBC 默认值
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    使用ULA推理下一个位置（在归一化空间）
    
    关键改进：使用概率分布采样而不是直接选择能量最低的，避免陷入局部最优
    
    Args:
        model: 训练好的SequenceEBM模型
        obs_seq: observation序列（归一化），形状为 (1, 2, 2) - [obs_{t-1}, obs_t]
        ula_sampler: ULASampler实例
        num_action_samples: 采样候选数量（IBC 默认 512）
        temperature: 温度参数（IBC 默认 1.0）
        device: 计算设备
    
    Returns:
        下一个预测位置（归一化），形状为 (2,)
    """
    model.eval()
    obs_seq = obs_seq.to(device)
    
    with torch.no_grad():
        # 创建包装模型
        class EBMWrapper(nn.Module):
            def __init__(self, model, obs_seq):
                super().__init__()
                self.model = model
                self.obs_seq = obs_seq
            
            def forward(self, x, y):
                return self.model(self.obs_seq, y)
        
        # 使用ULA采样预测
        with torch.enable_grad():
            ebm_wrapper = EBMWrapper(model, obs_seq)
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
            energies = model(obs_seq, candidates)  # (1, num_action_samples)
            
            # 关键改进：使用概率分布采样而不是直接选最低的
            # 匹配 IBC 的 get_probabilities 逻辑
            # probs = softmax(-energies / temperature)
            logits = -energies / temperature  # (1, num_action_samples)
            probs = torch.nn.functional.softmax(logits, dim=1)  # (1, num_action_samples)
            
            # 从概率分布中采样（而不是直接选最低的）
            # 这样可以避免陷入局部能量低谷
            dist = torch.distributions.Categorical(probs)
            sampled_idx = dist.sample().item()
            next_pos = candidates[0, sampled_idx].cpu().numpy()  # (2,)
    
    # 确保返回的是1D数组 (2,)
    return next_pos.flatten()


def denormalize_coords(coords_norm, coord_bounds):
    """
    将归一化坐标还原到原始坐标范围
    
    Args:
        coords_norm: 归一化坐标，形状可以是 (2,) 或 (N, 2)
        coord_bounds: 坐标边界，形状为 (2, 2) - [[x_min, y_min], [x_max, y_max]]
    
    Returns:
        原始坐标，形状与输入相同
    """
    if coord_bounds is None:
        return coords_norm
    
    coords_norm = np.asarray(coords_norm)
    coords = (coords_norm + 1.0) / 2.0 * (coord_bounds[1] - coord_bounds[0]) + coord_bounds[0]
    return coords


def compute_energy_landscape(
    model: SequenceEBM,
    obs_seq: np.ndarray,
    resolution: int = 50,
    coord_bounds=None,
    use_original_space: bool = False,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    计算能量地形图（基于observation序列）
    
    对于每个网格点，计算给定observation序列时该点作为下一步action的能量
    
    Args:
        model: SequenceEBM模型
        obs_seq: observation序列 (2, 2) - [obs_{t-1}, obs_t]（归一化坐标）
        resolution: 网格分辨率
        coord_bounds: 坐标边界（用于反归一化显示）
        use_original_space: 是否在原始坐标空间显示（5x5）
        device: 计算设备
    
    Returns:
        X_grid, Y_grid, energy_grid: 网格坐标和能量值
    """
    model.eval()
    
    # 创建网格
    if use_original_space and coord_bounds is not None:
        x_coords = np.linspace(coord_bounds[0][0], coord_bounds[1][0], resolution)
        y_coords = np.linspace(coord_bounds[0][1], coord_bounds[1][1], resolution)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
        # 将网格点归一化
        grid_points_orig = np.stack([X_grid.flatten(), Y_grid.flatten()], axis=-1)
        grid_points_norm = 2.0 * (grid_points_orig - coord_bounds[0]) / (coord_bounds[1] - coord_bounds[0]) - 1.0
    else:
        x_coords = np.linspace(-1, 1, resolution)
        y_coords = np.linspace(-1, 1, resolution)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
        grid_points_norm = np.stack([X_grid.flatten(), Y_grid.flatten()], axis=-1)
    
    # 批量计算能量
    with torch.no_grad():
        obs_seq_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)  # (1, 2, 2)
        grid_tensor = torch.from_numpy(grid_points_norm).float().to(device).unsqueeze(0)  # (1, resolution^2, 2)
        
        # 计算所有网格点的能量
        energies = model(obs_seq_tensor, grid_tensor)  # (1, resolution^2)
        energy_grid = energies[0].cpu().numpy().reshape(resolution, resolution)
    
    return X_grid, Y_grid, energy_grid


def visualize_trajectory_dynamic(
    model: SequenceEBM,
    image_np: np.ndarray,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    true_trajectory: np.ndarray,
    ula_sampler: ULASampler,
    save_path: Path,
    coord_bounds=None,
    max_steps: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    动态可视化轨迹生成过程（使用observation序列）
    
    单个窗口并排显示：
    左图：显示起点、终点、真实轨迹和逐步生成的预测轨迹
    右图：显示当前observation序列的能量地形图
    
    Args:
        model: SequenceEBM模型
        image_np: 输入图像numpy (H, W)（用于背景显示）
        start_pos: 起始位置（归一化） (2,)
        target_pos: 目标位置（归一化） (2,)
        true_trajectory: 真实轨迹（归一化） (T, 2)
        ula_sampler: ULASampler实例
        coord_bounds: 坐标边界（用于反归一化到5x5空间）
        save_path: 保存路径
        max_steps: 最大预测步数
        device: 计算设备
    """
    print(f"  生成动态可视化...")
    
    # 反归一化到原始坐标空间用于显示
    if coord_bounds is not None:
        start_pos_orig = denormalize_coords(start_pos.copy(), coord_bounds)
        target_pos_orig = denormalize_coords(target_pos.copy(), coord_bounds)
        true_trajectory_orig = denormalize_coords(true_trajectory.copy(), coord_bounds)
        xlim = [coord_bounds[0][0], coord_bounds[1][0]]
        ylim = [coord_bounds[0][1], coord_bounds[1][1]]
    else:
        start_pos_orig = start_pos
        target_pos_orig = target_pos
        true_trajectory_orig = true_trajectory
        xlim = [-1.1, 1.1]
        ylim = [-1.1, 1.1]
    
    # 创建单个窗口，并排两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.ion()  # 开启交互模式
    
    # 初始化轨迹（归一化空间，用于推理）
    # 前两个点都是起点（因为需要observation序列）
    # 确保start_pos是1D数组 (2,)
    start_pos_flat = start_pos.flatten()
    pred_trajectory_norm = [start_pos_flat.copy(), start_pos_flat.copy()]
    
    # 初始化colorbar（在循环外创建，避免重复添加）
    cbar = None
    
    # 逐步生成轨迹
    for step in range(max_steps):
        # 构建observation序列
        obs_seq = np.array([pred_trajectory_norm[-2], pred_trajectory_norm[-1]])  # (2, 2)
        obs_seq_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)  # (1, 2, 2)
        
        # 推理下一个位置（归一化空间）
        # 使用 IBC 的默认参数：512 个候选，temperature=1.0
        next_pos_norm = infer_single_step(
            model, obs_seq_tensor, ula_sampler, 
            num_action_samples=512, 
            temperature=1.0,
            device=device
        )
        # 确保next_pos_norm是1D数组 (2,)
        pred_trajectory_norm.append(next_pos_norm.flatten())
        
        # 反归一化预测轨迹用于显示
        if coord_bounds is not None:
            pred_traj_array_orig = denormalize_coords(np.array(pred_trajectory_norm), coord_bounds)
            next_pos_orig = denormalize_coords(next_pos_norm.reshape(1, -1), coord_bounds)[0]
        else:
            pred_traj_array_orig = np.array(pred_trajectory_norm)
            next_pos_orig = next_pos_norm
        
        # 清空左图并重新绘制（原始坐标空间）
        ax1.clear()
        # 图像显示范围需要根据coord_bounds调整
        if coord_bounds is not None:
            extent = [coord_bounds[0][0], coord_bounds[1][0], coord_bounds[0][1], coord_bounds[1][1]]
        else:
            extent = [-1, 1, -1, 1]
        ax1.imshow(image_np, cmap='gray', extent=extent, origin='lower', alpha=0.6)
        ax1.plot(start_pos_orig[0], start_pos_orig[1], 'bs', markersize=15, 
                 markeredgecolor='cyan', markeredgewidth=2, label='Start', zorder=10)
        ax1.plot(target_pos_orig[0], target_pos_orig[1], 'y*', markersize=25, 
                 markeredgecolor='red', markeredgewidth=2, label='Goal', zorder=10)
        ax1.plot(true_trajectory_orig[:, 0], true_trajectory_orig[:, 1], 'g-', 
                 linewidth=2, label='True Trajectory', alpha=0.7, zorder=3)
        # 绘制预测轨迹（从起点到当前位置）
        if len(pred_traj_array_orig) > 1:
            ax1.plot(pred_traj_array_orig[:, 0], pred_traj_array_orig[:, 1], 'r-', 
                     linewidth=2.5, label='Predicted Trajectory', zorder=5)
        # 当前位置标记
        ax1.plot(next_pos_orig[0], next_pos_orig[1], 'ro', markersize=12, 
                 markeredgecolor='white', markeredgewidth=2, label='Current Position', zorder=11)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'Trajectory Generation (Step {step+1})')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        
        # 清空右图并重新绘制能量地形图（归一化空间）
        ax2.clear()
        X_grid, Y_grid, energy_grid = compute_energy_landscape(
            model, obs_seq, resolution=40, 
            coord_bounds=coord_bounds, use_original_space=False, device=device
        )
        
        # 绘制能量等高线
        contour = ax2.contourf(X_grid, Y_grid, energy_grid, levels=20, cmap='viridis', alpha=0.8)
        ax2.contour(X_grid, Y_grid, energy_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        # 叠加轨迹和关键点（归一化空间）
        ax2.plot(start_pos[0], start_pos[1], 'bs', markersize=15, 
                markeredgecolor='cyan', markeredgewidth=2, label='Start', zorder=10)
        ax2.plot(target_pos[0], target_pos[1], 'y*', markersize=25, 
                markeredgecolor='red', markeredgewidth=2, label='Goal', zorder=10)
        if len(pred_trajectory_norm) > 1:
            pred_traj_norm_array = np.array(pred_trajectory_norm)
            ax2.plot(pred_traj_norm_array[:, 0], pred_traj_norm_array[:, 1], 'r-', 
                    linewidth=2, label='Predicted Trajectory', alpha=0.9, zorder=5)
        ax2.plot(next_pos_norm[0], next_pos_norm[1], 'ro', markersize=12, 
                markeredgecolor='white', markeredgewidth=2, label='Current Position', zorder=11)
        
        ax2.set_xlabel('X (Normalized)')
        ax2.set_ylabel('Y (Normalized)')
        ax2.set_title(f'Energy Landscape (Step {step+1})')
        ax2.legend(loc='upper right')
        ax2.set_aspect('equal')
        ax2.set_xlim([-1.1, 1.1])
        ax2.set_ylim([-1.1, 1.1])
        
        # 更新或创建colorbar
        if cbar is not None:
            # 删除旧的colorbar
            cbar.remove()
        # 创建新的colorbar
        cbar = plt.colorbar(contour, ax=ax2, label='Energy')
        
        # 更新显示
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        # 终止条件（归一化空间）
        if step > 10 and np.linalg.norm(next_pos_norm - pred_trajectory_norm[-2]) < 0.01:
            print(f"    收敛于步骤 {step+1}")
            break
    
    # 保存最终结果
    plt.ioff()  # 关闭交互模式
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  可视化已保存到: {save_path}")
    
    # 返回反归一化后的轨迹（原始坐标空间）
    if coord_bounds is not None:
        return denormalize_coords(np.array(pred_trajectory_norm), coord_bounds)
    else:
        return np.array(pred_trajectory_norm)


def visualize_trajectory(
    image: np.ndarray,
    true_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    save_path: Path,
    model: SequenceEBM = None,
    trajectory_length: int = None,
    coord_bounds=None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    可视化轨迹预测结果，包括输入图像、起点、能量地形图
    
    Args:
        image: 输入图像 (H, W)（用于背景显示）
        true_trajectory: true traj (T, 2)（归一化）
        pred_trajectory: pred traj (T, 2)（原始坐标空间）
        start_pos: start坐标 (2,)（归一化）
        target_pos: goal坐标 (2,)（归一化）
        save_path: 保存路径
        model: SequenceEBM模型（用于计算能量地形图）
        trajectory_length: 轨迹长度
        coord_bounds: 坐标边界（用于反归一化）
        device: 计算设备
    """
    # 反归一化坐标（如果需要）
    if coord_bounds is not None:
        true_trajectory_orig = denormalize_coords(true_trajectory.copy(), coord_bounds)
        pred_trajectory_orig = denormalize_coords(pred_trajectory.copy(), coord_bounds)
        start_pos_orig = denormalize_coords(start_pos.copy(), coord_bounds)
        target_pos_orig = denormalize_coords(target_pos.copy(), coord_bounds)
    else:
        true_trajectory_orig = true_trajectory
        pred_trajectory_orig = pred_trajectory
        start_pos_orig = start_pos
        target_pos_orig = target_pos
    
    # 创建3个子图：输入图像、轨迹对比、能量地形图
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    
    # 确定显示范围（原始坐标空间）
    if coord_bounds is not None:
        xlim = [coord_bounds[0][0], coord_bounds[1][0]]
        ylim = [coord_bounds[0][1], coord_bounds[1][1]]
        extent = [coord_bounds[0][0], coord_bounds[1][0], coord_bounds[0][1], coord_bounds[1][1]]
    else:
        xlim = [-1.1, 1.1]
        ylim = [-1.1, 1.1]
        extent = [-1, 1, -1, 1]
    
    # 左图：输入图像（目标点高亮）+ 起点
    ax1 = fig.add_subplot(gs[0, 0])
    # 显示图像（原始坐标空间）
    im1 = ax1.imshow(image, cmap='gray', extent=extent, origin='lower', alpha=0.8)
    # 高亮目标点（用大圆圈标记）
    ax1.plot(target_pos_orig[0], target_pos_orig[1], 'y*', markersize=25, 
             markeredgecolor='red', markeredgewidth=2, label='Goal (Highlighted)', zorder=10)
    # 起点
    ax1.plot(start_pos_orig[0], start_pos_orig[1], 'bs', markersize=15, 
             markeredgecolor='cyan', markeredgewidth=2, label='Start', zorder=10)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Input Image\n(Goal Highlighted + Start Position)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    plt.colorbar(im1, ax=ax1, label='Intensity')
    
    # 中图：轨迹对比（原始坐标空间）
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(true_trajectory_orig[:, 0], true_trajectory_orig[:, 1], 'g-', linewidth=2.5, 
             label='True Traj', alpha=0.8)
    ax2.plot(pred_trajectory_orig[:, 0], pred_trajectory_orig[:, 1], 'r--', linewidth=2.5, 
             label='Pred Traj', alpha=0.8)
    ax2.plot(start_pos_orig[0], start_pos_orig[1], 'bs', markersize=15, 
             markeredgecolor='cyan', markeredgewidth=2, label='Start', zorder=10)
    ax2.plot(target_pos_orig[0], target_pos_orig[1], 'y*', markersize=20, 
             markeredgecolor='red', markeredgewidth=2, label='Goal', zorder=10)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Trajectory Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    
    # 右图：能量地形图（归一化空间）
    ax3 = fig.add_subplot(gs[0, 2])
    if model is not None:
        # 计算能量地形图（归一化空间）
        print(f"  计算能量地形图...")
        # 使用预测轨迹的最后两个点作为observation序列
        if len(pred_trajectory) >= 2:
            # pred_trajectory 是原始坐标，需要归一化
            if coord_bounds is not None:
                pred_traj_norm = 2.0 * (pred_trajectory - coord_bounds[0]) / (coord_bounds[1] - coord_bounds[0]) - 1.0
            else:
                pred_traj_norm = pred_trajectory
            obs_seq_final = pred_traj_norm[-2:]  # (2, 2)
        else:
            # 如果轨迹太短，使用起点
            obs_seq_final = np.array([start_pos, start_pos])
        
        X_grid, Y_grid, energy_grid = compute_energy_landscape(
            model, obs_seq_final, resolution=40, 
            coord_bounds=coord_bounds, use_original_space=False, device=device
        )
        
        # 绘制能量等高线（归一化空间）
        contour = ax3.contourf(X_grid, Y_grid, energy_grid, levels=20, cmap='viridis', alpha=0.8)
        ax3.contour(X_grid, Y_grid, energy_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        # 叠加轨迹和关键点（归一化空间）
        # 需要将轨迹归一化
        if coord_bounds is not None:
            true_traj_norm = 2.0 * (true_trajectory_orig - coord_bounds[0]) / (coord_bounds[1] - coord_bounds[0]) - 1.0
            pred_traj_norm = 2.0 * (pred_trajectory_orig - coord_bounds[0]) / (coord_bounds[1] - coord_bounds[0]) - 1.0
            start_pos_norm = 2.0 * (start_pos_orig - coord_bounds[0]) / (coord_bounds[1] - coord_bounds[0]) - 1.0
            target_pos_norm = 2.0 * (target_pos_orig - coord_bounds[0]) / (coord_bounds[1] - coord_bounds[0]) - 1.0
        else:
            true_traj_norm = true_trajectory
            pred_traj_norm = pred_trajectory
            start_pos_norm = start_pos
            target_pos_norm = target_pos
        
        ax3.plot(true_traj_norm[:, 0], true_traj_norm[:, 1], 'g-', linewidth=2, 
                 label='True Traj', alpha=0.9, zorder=5)
        ax3.plot(pred_traj_norm[:, 0], pred_traj_norm[:, 1], 'r--', linewidth=2, 
                 label='Pred Traj', alpha=0.9, zorder=5)
        ax3.plot(start_pos_norm[0], start_pos_norm[1], 'bs', markersize=15, 
                 markeredgecolor='cyan', markeredgewidth=2, label='Start', zorder=10)
        ax3.plot(target_pos_norm[0], target_pos_norm[1], 'y*', markersize=20, 
                 markeredgecolor='red', markeredgewidth=2, label='Goal', zorder=10)
        
        ax3.set_xlabel('X (Normalized)')
        ax3.set_ylabel('Y (Normalized)')
        ax3.set_title('Energy Landscape (Final Step)')
        ax3.legend()
        ax3.set_aspect('equal')
        ax3.set_xlim([-1.1, 1.1])
        ax3.set_ylim([-1.1, 1.1])
        plt.colorbar(contour, ax=ax3, label='Energy')
    else:
        ax3.text(0.5, 0.5, 'Energy landscape\nnot available', 
                ha='center', va='center', fontsize=14)
        ax3.set_title('Energy Landscape')
    
    # 计算误差（原始坐标空间）
    if len(true_trajectory_orig) == len(pred_trajectory_orig):
        errors = np.sqrt(np.sum((true_trajectory_orig - pred_trajectory_orig)**2, axis=1))
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        final_error = np.sqrt(np.sum((true_trajectory_orig[-1] - pred_trajectory_orig[-1])**2))
        
        # 添加文本信息
        info_text = f'Mean Error: {mean_error:.4f}  |  Max Error: {max_error:.4f}  |  Goal Error: {final_error:.4f}'
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化已保存到: {save_path}")


def test_model(
    model_path: str,
    data_dir: str,
    num_test_samples: int = 10,
    use_dynamic_viz: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    测试SequenceEBM模型
    
    Args:
        model_path: 模型文件路径
        data_dir: 测试数据目录
        num_test_samples: 测试样本数量
        use_dynamic_viz: 是否使用动态可视化
        device: 计算设备
    """
    print("=" * 60)
    print("A2B 序列轨迹预测模型测试 (Sequence + ULA)")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    model, checkpoint = load_model(model_path, device)
    trajectory_length = checkpoint.get('trajectory_length', 99)
    image_size = checkpoint.get('image_size', 64)
    ula_step_size = checkpoint.get('ula_step_size', 0.1)  # Default updated to 0.1
    ula_num_steps = checkpoint.get('ula_num_steps', 100)  # Default updated to 100
    traj_bounds = checkpoint.get('traj_bounds', np.array([[-1.0, -1.0], [1.0, 1.0]]))
    
    # 2. 加载测试数据
    print("\n2. 加载测试数据...")
    dataset = A2BDataset(data_dir, image_size=image_size, normalize_coords=True)
    print(f"数据集大小: {len(dataset)}")
    
    # 3. 创建ULA采样器
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
    
    # 4. 进行测试
    print(f"\n4. 测试 {num_test_samples} 个样本...")
    test_indices = np.linspace(0, len(dataset) - 1, num_test_samples, dtype=int)
    
    all_errors = []
    coord_bounds = dataset.coord_bounds
    
    for idx, sample_idx in enumerate(tqdm(test_indices, desc="测试中")):
        sample = dataset[sample_idx]
        
        image = sample['image']  # (1, H, W)
        true_trajectory = sample['trajectory_2d'].numpy()  # (T, 2)（归一化）
        start_pos = sample['start_pos'].numpy()  # (2,)（归一化）
        target_pos = sample['target_pos'].numpy()  # (2,)（归一化）
        
        image_np = image.squeeze().numpy()  # (H, W)
        save_path = output_dir / f'test_sample_{idx:03d}.png'
        
        # 使用动态可视化
        if use_dynamic_viz:
            print(f"\n样本 {idx+1}/{num_test_samples}:")
            pred_trajectory = visualize_trajectory_dynamic(
                model, image_np,
                start_pos, target_pos, true_trajectory,
                ula_sampler, save_path,
                coord_bounds=coord_bounds,
                max_steps=trajectory_length,
                device=device
            )
        else:
            # 静态推理（返回归一化坐标空间）
            # 确保start_pos是1D数组
            start_pos_flat = start_pos.flatten()
            pred_trajectory_norm = infer_trajectory(
                model, start_pos_flat, ula_sampler, 
                max_steps=trajectory_length,
                num_action_samples=512,  # IBC 默认值
                temperature=1.0,  # IBC 默认值
                device=device
            )
            # 反归一化到原始坐标空间
            if coord_bounds is not None:
                pred_trajectory = denormalize_coords(pred_trajectory_norm, coord_bounds)
            else:
                pred_trajectory = pred_trajectory_norm
            
            # 静态可视化
            visualize_trajectory(
                image_np, true_trajectory, pred_trajectory,
                start_pos, target_pos, save_path,
                model=model,
                trajectory_length=trajectory_length,
                coord_bounds=coord_bounds,
                device=device
            )
        
        # 计算误差（使用相同长度的轨迹进行比较，原始坐标空间）
        # 反归一化真实轨迹用于误差计算
        if coord_bounds is not None:
            true_traj_orig = denormalize_coords(true_trajectory.copy(), coord_bounds)
        else:
            true_traj_orig = true_trajectory
        
        min_len = min(len(true_traj_orig), len(pred_trajectory))
        errors = np.sqrt(np.sum((true_traj_orig[:min_len] - pred_trajectory[:min_len])**2, axis=1))
        mean_error = np.mean(errors)
        final_error = np.sqrt(np.sum((true_traj_orig[-1] - pred_trajectory[-1])**2))
        all_errors.append({
            'mean_error': mean_error,
            'final_error': final_error,
            'max_error': np.max(errors),
            'pred_length': len(pred_trajectory),
            'true_length': len(true_trajectory)
        })
    
    # 5. 打印统计信息
    print("\n" + "=" * 60)
    print("5. 测试结果统计:")
    print("=" * 60)
    if all_errors:
        mean_errors = [e['mean_error'] for e in all_errors]
        final_errors = [e['final_error'] for e in all_errors]
        max_errors = [e['max_error'] for e in all_errors]
        pred_lengths = [e['pred_length'] for e in all_errors]
        true_lengths = [e['true_length'] for e in all_errors]
        
        print(f"平均轨迹误差: {np.mean(mean_errors):.4f} ± {np.std(mean_errors):.4f}")
        print(f"终点误差: {np.mean(final_errors):.4f} ± {np.std(final_errors):.4f}")
        print(f"最大误差: {np.mean(max_errors):.4f} ± {np.std(max_errors):.4f}")
        print(f"预测轨迹长度: {np.mean(pred_lengths):.1f} ± {np.std(pred_lengths):.1f}")
        print(f"真实轨迹长度: {np.mean(true_lengths):.1f}")
    
    print(f"\n所有可视化结果已保存到: {output_dir}")
    print("测试完成！")


if __name__ == '__main__':
    # 模型路径
    model_path = models_dir / 'a2b_ebm_model.pth'
    
    # 数据目录
    data_dir = Path(__file__).parent.parent.parent / 'data' / '_2d' / 'A2B_data'
    
    # 测试参数
    test_model(
        model_path=str(model_path),
        data_dir=str(data_dir),
        num_test_samples=10,  # 测试3个样本（动态可视化较慢）
        use_dynamic_viz=True,  # 使用动态可视化
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

