"""
Particle 环境 EBM 模型实时交互测试脚本
通过键盘控制主动物体，模型实时推理运动
"""
import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm
import sys
import time

# 添加项目路径
IBC_ROOT = Path(__file__).parent.parent.parent.parent  # IBC_ebm_dp
sys.path.insert(0, str(IBC_ROOT))

from core.models import SequenceEBM
from core.optimizers import ULASampler

# 导入训练脚本中的数据集类
sys.path.insert(0, str(Path(__file__).parent))
from particle_train import ParticleDataset

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = Path(__file__).parent.parent.parent / 'plots' / 'particle'
output_dir.mkdir(parents=True, exist_ok=True)


def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载训练好的 SequenceEBM 模型
    
    Args:
        model_path: 模型文件路径
        device: 计算设备
    
    Returns:
        model: 加载的模型
        checkpoint: 模型 checkpoint 信息
        norm_params: 归一化参数
    """
    print(f"加载模型从: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型参数
    obs_dim = checkpoint.get('obs_dim', 8)
    action_dim = checkpoint.get('action_dim', 2)
    obs_seq_len = checkpoint.get('obs_seq_len', 2)
    hidden_dim = checkpoint.get('hidden_dim', 256)
    num_residual_blocks = checkpoint.get('num_residual_blocks', 1)
    dropout = checkpoint.get('dropout', 0.0)
    norm_type = checkpoint.get('norm_type', None)
    norm_params = checkpoint.get('norm_params', None)
    
    print(f"模型参数:")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, obs_seq_len={obs_seq_len}")
    print(f"  hidden_dim={hidden_dim}, num_residual_blocks={num_residual_blocks}")
    print(f"  dropout={dropout}, norm_type={norm_type}")
    
    # 创建 SequenceEBM
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
    return model, checkpoint, norm_params


def denormalize_action(action_norm, norm_params):
    """
    将归一化的动作反归一化到原始空间 [0, 1]
    
    Args:
        action_norm: 归一化动作 (2,) 或 (N, 2)，范围 [-1, 1]
        norm_params: 归一化参数字典
    
    Returns:
        action_orig: 原始动作 (2,) 或 (N, 2)，范围 [0, 1]
    """
    if norm_params is None or norm_params.get('action_min') is None:
        return action_norm
    
    action_min = np.array(norm_params['action_min'])
    action_max = np.array(norm_params['action_max'])
    action_range = action_max - action_min
    
    # 从 [-1, 1] 反归一化到 [0, 1]
    action_orig = (action_norm + 1.0) / 2.0 * action_range + action_min
    return action_orig


def denormalize_obs(obs_norm, norm_params):
    """
    将归一化的观测反归一化到原始空间
    
    Args:
        obs_norm: 归一化观测 (8,) 或 (N, 8)
        norm_params: 归一化参数字典
    
    Returns:
        obs_orig: 原始观测
    """
    if norm_params is None or norm_params.get('obs_mean') is None:
        return obs_norm
    
    obs_mean = np.array(norm_params['obs_mean'])
    obs_std = np.array(norm_params['obs_std'])
    
    obs_orig = obs_norm * obs_std + obs_mean
    return obs_orig


def infer_single_step(
    model: SequenceEBM,
    obs_seq: torch.Tensor,
    ula_sampler: ULASampler,
    num_action_samples: int = 512,
    temperature: float = 1.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    return_debug_info: bool = False
):
    """
    使用 ULA 推理下一个动作（在归一化空间）
    
    Args:
        model: 训练好的 SequenceEBM 模型
        obs_seq: observation 序列（归一化），形状为 (1, obs_seq_len, obs_dim)
        ula_sampler: ULASampler 实例
        num_action_samples: 采样候选数量
        temperature: 温度参数
        device: 计算设备
        return_debug_info: 是否返回调试信息（候选动作、能量值等）
    
    Returns:
        如果 return_debug_info=False:
            下一个预测动作（归一化），形状为 (2,)
        如果 return_debug_info=True:
            (next_action, debug_info) 元组，其中 debug_info 包含：
            - candidates: 候选动作 (num_action_samples, 2)
            - energies: 候选动作的能量值 (num_action_samples,)
            - selected_energy: 最终选择动作的能量值（标量）
            - action_range: 动作范围 [min_x, max_x, min_y, max_y]
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
        
        # 使用 ULA 采样预测
        with torch.enable_grad():
            ebm_wrapper = EBMWrapper(model, obs_seq)
            placeholder_x = torch.zeros(1, 1, device=device)
            
            # 采样多个候选
            candidates, _ = ula_sampler.sample(
                placeholder_x,
                ebm_wrapper,
                num_samples=num_action_samples,
                init_samples=None,
                return_trajectory=False
            )  # (1, num_action_samples, 2)
            
            # 计算所有候选的能量
            energies = model(obs_seq, candidates)  # (1, num_action_samples)
            
            # 使用概率分布选择动作（匹配 IBC 的 GreedyPolicy.mode()）
            # IBC 使用 GreedyPolicy，它会调用 distribution.mode()，等价于 argmax
            logits = -energies / temperature  # (1, num_action_samples)
            probs = F.softmax(logits, dim=1)  # (1, num_action_samples)
            
            # 选择概率最高的动作（匹配 GreedyPolicy 的行为）
            sampled_idx = probs.argmax(dim=1).item()
            next_action = candidates[0, sampled_idx].detach().cpu().numpy()  # (2,)
            selected_energy = energies[0, sampled_idx].detach().item()  # 标量
    
    if return_debug_info:
        candidates_np = candidates[0].detach().cpu().numpy()  # (num_action_samples, 2)
        energies_np = energies[0].detach().cpu().numpy()  # (num_action_samples,)
        
        # 计算能量区间（所有候选动作的能量范围）
        energy_min = float(energies_np.min())
        energy_max = float(energies_np.max())
        energy_mean = float(energies_np.mean())
        energy_std = float(energies_np.std())
        
        debug_info = {
            'candidates': candidates_np,
            'energies': energies_np,
            'selected_energy': selected_energy,
            'energy_range': [energy_min, energy_max],  # 能量区间 [min, max]
            'energy_stats': {
                'min': energy_min,
                'max': energy_max,
                'mean': energy_mean,
                'std': energy_std
            }
        }
        return next_action.flatten(), debug_info
    
    return next_action.flatten()


def infer_trajectory(
    model: SequenceEBM,
    initial_obs_seq: torch.Tensor,
    ula_sampler: ULASampler,
    max_steps: int = 50,
    num_action_samples: int = 512,
    temperature: float = 1.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    return_intermediate: bool = False,
    norm_params = None
):
    """
    推理完整轨迹（修复版：在原始空间执行PD控制器）
    
    Args:
        model: SequenceEBM 模型
        initial_obs_seq: 初始观测序列 (1, obs_seq_len, obs_dim)（归一化）
        ula_sampler: ULASampler 实例
        max_steps: 最大步数
        num_action_samples: 采样候选数量
        temperature: 温度参数
        device: 计算设备
        return_intermediate: 是否返回中间状态（用于视频生成）
        norm_params: 归一化参数（用于反归一化）
    
    Returns:
        trajectory: 预测轨迹（归一化空间），形状为 (T, 2)
        如果 return_intermediate=True，还返回 intermediate_states 列表
    """
    model.eval()
    trajectory = []
    intermediate_states = [] if return_intermediate else None
    
    # 从初始观测序列中提取位置信息（归一化空间）
    current_obs_seq = initial_obs_seq.clone()
    current_obs_norm = current_obs_seq[0, -1].cpu().numpy()  # (8,) 归一化空间
    
    # ★ 关键修复：反归一化到原始空间执行PD控制器
    current_obs_orig = denormalize_obs(current_obs_norm, norm_params)  # (8,) 原始空间 [0, 1]
    current_pos_orig = current_obs_orig[:2]  # pos_agent 原始空间
    current_vel_orig = current_obs_orig[2:4]  # vel_agent 原始空间
    pos_first_goal_orig = current_obs_orig[4:6]  # 第一个目标 原始空间
    pos_second_goal_orig = current_obs_orig[6:8]  # 第二个目标 原始空间
    
    if return_intermediate:
        intermediate_states.append({
            'pos': current_pos_orig.copy(),
            'obs_seq': current_obs_seq.clone()
        })
    
    # PD 控制器参数（匹配 IBC Particle 环境）
    k_p = 10.0
    k_v = 5.0
    dt = 0.005
    repeat_actions = 10  # 每个动作重复 10 次
    
    for step in range(max_steps):
        # 推理下一个动作（在归一化空间）
        if return_intermediate:
            next_action_norm, debug_info = infer_single_step(
                model, current_obs_seq, ula_sampler,
                num_action_samples=num_action_samples,
                temperature=temperature,
                device=device,
                return_debug_info=True
            )
        else:
            next_action_norm = infer_single_step(
                model, current_obs_seq, ula_sampler,
                num_action_samples=num_action_samples,
                temperature=temperature,
                device=device,
                return_debug_info=False
            )
            debug_info = None
        
        trajectory.append(next_action_norm)  # 保存归一化的action用于返回
        
        # ★ 关键修复：反归一化action到原始空间
        action_orig = denormalize_action(next_action_norm, norm_params)  # [0, 1]
        
        if step == 0:
            print(f"第一步动作预测:")
            print(f"  当前位置 (原始空间): {current_pos_orig}")
            print(f"  第一个目标 (原始空间): {pos_first_goal_orig}")
            print(f"  第二个目标 (原始空间): {pos_second_goal_orig}")
            print(f"  预测动作 (原始空间): {action_orig}")
            print(f"  到第一个目标的距离: {np.linalg.norm(current_pos_orig - pos_first_goal_orig):.4f}")
            print(f"  到第二个目标的距离: {np.linalg.norm(current_pos_orig - pos_second_goal_orig):.4f}")
        
        # ★ 在原始空间执行PD控制器
        new_pos_orig = current_pos_orig.copy()
        new_vel_orig = current_vel_orig.copy()
        
        for _ in range(repeat_actions):
            # PD 控制器：u = k_p * (action - pos) - k_v * vel （原始空间）
            u_agent = k_p * (action_orig - new_pos_orig) - k_v * new_vel_orig
            
            # 更新位置和速度（原始空间）
            new_pos_orig = new_pos_orig + new_vel_orig * dt
            new_vel_orig = new_vel_orig + u_agent * dt
        
        # 确保位置在 [0, 1] 范围内
        new_pos_orig = np.clip(new_pos_orig, 0.0, 1.0)
        
        # 构建新的观测（原始空间）
        new_obs_orig = np.concatenate([
            new_pos_orig,          # pos_agent (2) 原始空间
            new_vel_orig,          # vel_agent (2) 原始空间
            pos_first_goal_orig,   # pos_first_goal (2) 原始空间
            pos_second_goal_orig   # pos_second_goal (2) 原始空间
        ])
        
        # ★ 归一化新观测
        obs_mean = np.array(norm_params['obs_mean'])
        obs_std = np.array(norm_params['obs_std'])
        new_obs_norm = (new_obs_orig - obs_mean) / obs_std
        
        # 更新观测序列（滑动窗口）
        new_obs_tensor = torch.from_numpy(new_obs_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        current_obs_seq = torch.cat([current_obs_seq[:, 1:], new_obs_tensor], dim=1)
        
        # 更新当前状态（原始空间）
        current_pos_orig = new_pos_orig.copy()
        current_vel_orig = new_vel_orig.copy()
        
        if return_intermediate:
            intermediate_states.append({
                'pos': new_pos_orig.copy(),
                'obs_seq': current_obs_seq.clone(),
                'debug_info': debug_info  # 包含候选动作、能量值等信息
            })
    
    if return_intermediate:
        return np.array(trajectory), intermediate_states
    return np.array(trajectory)


def compute_energy_landscape(
    model: SequenceEBM,
    obs_seq: np.ndarray,
    resolution: int = 50,
    norm_params=None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    计算能量地形图
    
    Args:
        model: SequenceEBM 模型
        obs_seq: observation 序列 (obs_seq_len, obs_dim)（归一化）
        resolution: 网格分辨率
        norm_params: 归一化参数（用于反归一化显示）
        device: 计算设备
    
    Returns:
        X_grid, Y_grid, energy_grid: 网格坐标和能量值
    """
    model.eval()
    
    # 创建网格（归一化空间 [-1, 1]）
    x_coords = np.linspace(-1, 1, resolution)
    y_coords = np.linspace(-1, 1, resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    grid_points_norm = np.stack([X_grid.flatten(), Y_grid.flatten()], axis=-1)
    
    # 批量计算能量
    with torch.no_grad():
        obs_seq_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)  # (1, obs_seq_len, obs_dim)
        grid_tensor = torch.from_numpy(grid_points_norm).float().to(device).unsqueeze(0)  # (1, resolution^2, 2)
        
        # 计算所有网格点的能量
        energies = model(obs_seq_tensor, grid_tensor)  # (1, resolution^2)
        energy_grid = energies[0].cpu().numpy().reshape(resolution, resolution)
    
    return X_grid, Y_grid, energy_grid


def visualize_trajectory(
    true_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    first_goal: np.ndarray,
    second_goal: np.ndarray,
    start_pos: np.ndarray,
    save_path: Path,
    model: SequenceEBM = None,
    obs_seq_final: np.ndarray = None,
    norm_params=None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    debug_info_final: dict = None
):
    """
    可视化轨迹预测结果
    
    Args:
        true_trajectory: 真实轨迹 (T, 2)，原始空间 [0, 1]
        pred_trajectory: 预测轨迹 (T, 2)，原始空间 [0, 1]
        first_goal: 第一个目标位置 (2,)，原始空间
        second_goal: 第二个目标位置 (2,)，原始空间
        start_pos: 起始位置 (2,)，原始空间
        save_path: 保存路径
        model: SequenceEBM 模型（用于计算能量地形图）
        obs_seq_final: 最终观测序列（用于能量地形图）
        norm_params: 归一化参数
        device: 计算设备
    """
    # 创建 2 个子图：轨迹对比和能量地形图
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    
    # 左图：轨迹对比
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'g-', linewidth=2.5, 
             label='True Trajectory', alpha=0.8)
    ax1.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', linewidth=2.5, 
             label='Predicted Trajectory', alpha=0.8)
    ax1.plot(start_pos[0], start_pos[1], 'bs', markersize=15, 
             markeredgecolor='cyan', markeredgewidth=2, label='Start', zorder=10)
    ax1.plot(first_goal[0], first_goal[1], 'go', markersize=12, 
             markeredgecolor='darkgreen', markeredgewidth=2, label='First Goal', zorder=10)
    ax1.plot(second_goal[0], second_goal[1], 'b*', markersize=20, 
             markeredgecolor='darkblue', markeredgewidth=2, label='Second Goal', zorder=10)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim([-0.1, 1.1])
    ax1.set_ylim([-0.1, 1.1])
    
    # 右图：能量地形图
    ax2 = fig.add_subplot(gs[0, 1])
    if model is not None and obs_seq_final is not None:
        print(f"  计算能量地形图...")
        X_grid, Y_grid, energy_grid = compute_energy_landscape(
            model, obs_seq_final, resolution=40, 
            norm_params=norm_params, device=device
        )
        
        # 绘制能量等高线
        contour = ax2.contourf(X_grid, Y_grid, energy_grid, levels=20, cmap='viridis', alpha=0.8)
        ax2.contour(X_grid, Y_grid, energy_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        # 叠加轨迹和关键点（归一化空间）
        # 将轨迹归一化到 [-1, 1]
        pred_traj_norm = 2.0 * (pred_trajectory - 0.5)  # [0, 1] -> [-1, 1]
        true_traj_norm = 2.0 * (true_trajectory - 0.5)
        start_norm = 2.0 * (start_pos - 0.5)
        first_goal_norm = 2.0 * (first_goal - 0.5)
        second_goal_norm = 2.0 * (second_goal - 0.5)
        
        ax2.plot(true_traj_norm[:, 0], true_traj_norm[:, 1], 'g-', linewidth=2, 
                 label='True Traj', alpha=0.9, zorder=5)
        ax2.plot(pred_traj_norm[:, 0], pred_traj_norm[:, 1], 'r--', linewidth=2, 
                 label='Pred Traj', alpha=0.9, zorder=5)
        ax2.plot(start_norm[0], start_norm[1], 'bs', markersize=15, 
                 markeredgecolor='cyan', markeredgewidth=2, label='Start', zorder=10)
        ax2.plot(first_goal_norm[0], first_goal_norm[1], 'go', markersize=12, 
                 markeredgecolor='darkgreen', markeredgewidth=2, label='First Goal', zorder=10)
        ax2.plot(second_goal_norm[0], second_goal_norm[1], 'b*', markersize=20, 
                 markeredgecolor='darkblue', markeredgewidth=2, label='Second Goal', zorder=10)
        
        ax2.set_xlabel('X (Normalized)')
        ax2.set_ylabel('Y (Normalized)')
        ax2.set_title('Energy Landscape (Final Step)')
        ax2.legend()
        ax2.set_aspect('equal')
        ax2.set_xlim([-1.1, 1.1])
        ax2.set_ylim([-1.1, 1.1])
        plt.colorbar(contour, ax=ax2, label='Energy')
    else:
        ax2.text(0.5, 0.5, 'Energy landscape\nnot available', 
                ha='center', va='center', fontsize=14)
        ax2.set_title('Energy Landscape')
    
    # 计算误差
    min_len = min(len(true_trajectory), len(pred_trajectory))
    errors = np.sqrt(np.sum((true_trajectory[:min_len] - pred_trajectory[:min_len])**2, axis=1))
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    final_error = np.sqrt(np.sum((true_trajectory[-1] - pred_trajectory[-1])**2))
    
    # 添加文本信息
    info_text = f'Mean Error: {mean_error:.4f}  |  Max Error: {max_error:.4f}  |  Final Error: {final_error:.4f}'
    
    # 添加能量区间和能量信息（如果可用）
    if debug_info_final is not None:
        energy_range = debug_info_final.get('energy_range', None)
        selected_energy = debug_info_final.get('selected_energy', None)
        
        if energy_range is not None:
            energy_text = f'Energy Range: [{energy_range[0]:.4f}, {energy_range[1]:.4f}]'
            info_text += f'\n{energy_text}'
        
        if selected_energy is not None:
            selected_text = f'Selected Action Energy: {selected_energy:.4f}'
            info_text += f'\n{selected_text}'
    
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), 
            verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 为底部文本留出空间
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化已保存到: {save_path}")
    
    return {
        'mean_error': mean_error,
        'max_error': max_error,
        'final_error': final_error
    }


def create_trajectory_video(
    true_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    intermediate_states: list,
    first_goal: np.ndarray,
    second_goal: np.ndarray,
    start_pos: np.ndarray,
    save_path: Path,
    model: SequenceEBM = None,
    norm_params=None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    fps: int = 10
):
    """
    创建轨迹生成的动态视频
    
    Args:
        true_trajectory: 真实轨迹 (T, 2)，原始空间 [0, 1]
        pred_trajectory: 预测轨迹 (T, 2)，原始空间 [0, 1]
        intermediate_states: 中间状态列表（从 infer_trajectory 返回）
        first_goal: 第一个目标位置 (2,)，原始空间
        second_goal: 第二个目标位置 (2,)，原始空间
        start_pos: 起始位置 (2,)，原始空间
        save_path: 保存路径（.mp4）
        model: SequenceEBM 模型（用于计算能量地形图）
        norm_params: 归一化参数
        device: 计算设备
        fps: 视频帧率
    """
    print(f"  生成视频: {save_path}")
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ★ 修复：从intermediate_states直接获取位置（已经是原始空间）
    pred_positions_list = [state['pos'] for state in intermediate_states]
    
    # 初始化 colorbar 和 imshow 对象（在动画函数外部创建，避免重复绘制）
    cbar = None
    im = None
    contour_lines = []
    
    def animate(frame):
        nonlocal cbar, im, contour_lines
        
        ax1.clear()
        
        # 对于 ax2，手动清除需要更新的元素，但保留 colorbar
        # 清除旧的等高线
        for coll in contour_lines:
            coll.remove()
        contour_lines.clear()
        
        # 清除旧的轨迹和标记（但保留 imshow 和 colorbar）
        # 获取所有需要清除的 lines
        lines_to_remove = []
        for line in ax2.lines:
            lines_to_remove.append(line)
        for line in lines_to_remove:
            line.remove()
        
        # 获取到当前帧为止的轨迹
        pred_positions_so_far = pred_positions_list[:frame+1]
        
        # 左图：轨迹对比
        ax1.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'g-', linewidth=2.5, 
                 label='True Trajectory', alpha=0.6)
        if len(pred_positions_so_far) > 1:
            pred_array = np.array(pred_positions_so_far)
            ax1.plot(pred_array[:, 0], pred_array[:, 1], 'r-', linewidth=2.5, 
                     label='Predicted Trajectory', alpha=0.8)
        if len(pred_positions_so_far) > 0:
            current = pred_positions_so_far[-1]
            ax1.plot(current[0], current[1], 'ro', markersize=12, 
                     markeredgecolor='white', markeredgewidth=2, label='Current Position', zorder=11)
        
        ax1.plot(start_pos[0], start_pos[1], 'bs', markersize=15, 
                 markeredgecolor='cyan', markeredgewidth=2, label='Start', zorder=10)
        ax1.plot(first_goal[0], first_goal[1], 'go', markersize=12, 
                 markeredgecolor='darkgreen', markeredgewidth=2, label='First Goal', zorder=10)
        ax1.plot(second_goal[0], second_goal[1], 'b*', markersize=20, 
                 markeredgecolor='darkblue', markeredgewidth=2, label='Second Goal', zorder=10)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.set_xlim([-0.1, 1.1])
        ax1.set_ylim([-0.1, 1.1])
        
        # 在左图上方添加能量区间和能量信息
        if frame < len(intermediate_states):
            state = intermediate_states[frame]
            debug_info = state.get('debug_info', None)
            if debug_info is not None:
                energy_range = debug_info.get('energy_range', None)
                selected_energy = debug_info.get('selected_energy', None)
                
                info_lines = []
                if energy_range is not None:
                    info_lines.append(f'Energy Range: [{energy_range[0]:.4f}, {energy_range[1]:.4f}]')
                if selected_energy is not None:
                    info_lines.append(f'Selected Energy: {selected_energy:.4f}')
                
                if info_lines:
                    info_text = '\n'.join(info_lines)
                    ax1.text(0.5, 1.02, info_text, transform=ax1.transAxes, 
                            ha='center', va='bottom', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # 右图：能量地形图
        if model is not None and frame < len(intermediate_states):
            state = intermediate_states[frame]
            obs_seq_norm = state['obs_seq'][0].cpu().numpy()  # (obs_seq_len, obs_dim)
            
            X_grid, Y_grid, energy_grid = compute_energy_landscape(
                model, obs_seq_norm, resolution=40, 
                norm_params=norm_params, device=device
            )
            
            # 使用 imshow 代替 contourf，便于更新数据
            if im is None:
                # 第一帧：创建 imshow 和 colorbar
                im = ax2.imshow(energy_grid, extent=[-1, 1, -1, 1], origin='lower', 
                               cmap='viridis', alpha=0.8, aspect='auto')
                # 添加等高线
                contour_coll = ax2.contour(X_grid, Y_grid, energy_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
                contour_lines.extend(contour_coll.collections)
                # 创建 colorbar（只创建一次）
                cbar = plt.colorbar(im, ax=ax2, label='Energy')
                # 设置坐标轴
                ax2.set_xlabel('X (Normalized)')
                ax2.set_ylabel('Y (Normalized)')
                ax2.set_aspect('equal')
                ax2.set_xlim([-1.1, 1.1])
                ax2.set_ylim([-1.1, 1.1])
            else:
                # 后续帧：只更新数据
                im.set_array(energy_grid)
                im.set_clim(vmin=energy_grid.min(), vmax=energy_grid.max())
                # 重新绘制等高线
                contour_coll = ax2.contour(X_grid, Y_grid, energy_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
                contour_lines.extend(contour_coll.collections)
            
            # 叠加轨迹（归一化空间）
            if len(pred_positions_so_far) > 1:
                pred_array = np.array(pred_positions_so_far)
                pred_norm = 2.0 * (pred_array - 0.5)  # [0, 1] -> [-1, 1]
                ax2.plot(pred_norm[:, 0], pred_norm[:, 1], 'r-', linewidth=2, 
                         label='Pred Traj', alpha=0.9, zorder=5)
            
            if len(pred_positions_so_far) > 0:
                current = pred_positions_so_far[-1]
                current_norm = 2.0 * (current - 0.5)
                ax2.plot(current_norm[0], current_norm[1], 'ro', markersize=12, 
                         markeredgecolor='white', markeredgewidth=2, label='Current', zorder=11)
            
            start_norm = 2.0 * (start_pos - 0.5)
            first_goal_norm = 2.0 * (first_goal - 0.5)
            second_goal_norm = 2.0 * (second_goal - 0.5)
            
            ax2.plot(start_norm[0], start_norm[1], 'bs', markersize=15, 
                     markeredgecolor='cyan', markeredgewidth=2, label='Start', zorder=10)
            ax2.plot(first_goal_norm[0], first_goal_norm[1], 'go', markersize=12, 
                     markeredgecolor='darkgreen', markeredgewidth=2, label='First Goal', zorder=10)
            ax2.plot(second_goal_norm[0], second_goal_norm[1], 'b*', markersize=20, 
                     markeredgecolor='darkblue', markeredgewidth=2, label='Second Goal', zorder=10)
            
            # 更新标题和标签（每次都需要更新）
            ax2.set_xlabel('X (Normalized)')
            ax2.set_ylabel('Y (Normalized)')
            ax2.set_title(f'Energy Landscape (Step {frame})')
            ax2.legend(loc='upper right')
            ax2.set_aspect('equal')
            ax2.set_xlim([-1.1, 1.1])
            ax2.set_ylim([-1.1, 1.1])
        else:
            ax2.text(0.5, 0.5, 'Energy landscape\nnot available', 
                    ha='center', va='center', fontsize=14)
            ax2.set_title('Energy Landscape')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # 为底部文本留出空间
    
    # 创建动画
    n_frames = len(intermediate_states) if intermediate_states else len(pred_trajectory)
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps, repeat=False)
    
    # 保存视频
    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Particle EBM Test'), bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"  视频已保存到: {save_path}")
    except Exception as e:
        print(f"  警告: 无法保存视频 ({e})，尝试保存为 GIF...")
        try:
            anim.save(save_path.with_suffix('.gif'), writer='pillow', fps=fps)
            print(f"  GIF 已保存到: {save_path.with_suffix('.gif')}")
        except Exception as e2:
            print(f"  错误: 无法保存 GIF ({e2})")
    
    plt.close(fig)


def load_episode_data(data_dir, episode_idx):
    """
    从 JSON 文件加载完整的 episode 数据
    
    Args:
        data_dir: 数据目录
        episode_idx: episode 索引
    
    Returns:
        episode_data: 包含完整轨迹信息的字典
    """
    data_path = Path(data_dir)
    traj_dir = data_path / 'traj'
    json_files = sorted(traj_dir.glob('traj_*.json'))
    
    if episode_idx >= len(json_files):
        return None
    
    with open(json_files[episode_idx], 'r') as f:
        data = json.load(f)
    
    return {
        'positions': np.array(data['trajectory']['positions'], dtype=np.float32),
        'actions': np.array(data['actions'], dtype=np.float32),
        'start_position': np.array(data['start_position'], dtype=np.float32),
        'first_goal': np.array(data['first_goal_position'], dtype=np.float32),
        'second_goal': np.array(data['second_goal_position'], dtype=np.float32),
        'target_position': np.array(data['target_position'], dtype=np.float32)
    }


class InteractiveParticleEnv:
    """
    交互式 Particle 环境
    """
    def __init__(self, model, norm_params, ula_sampler, device='cuda'):
        self.model = model
        self.norm_params = norm_params
        self.ula_sampler = ula_sampler
        self.device = device
        
        # 环境状态
        self.agent_pos = np.array([0.2, 0.2], dtype=np.float32)  # 原始空间 [0, 1]
        self.agent_vel = np.zeros(2, dtype=np.float32)  # 原始空间
        self.first_goal = np.array([0.3, 0.7], dtype=np.float32)  # 原始空间
        self.second_goal = np.array([0.7, 0.3], dtype=np.float32)  # 原始空间
        
        # 观测序列（归一化空间）
        self.obs_seq_len = 2
        self.obs_seq = self._build_obs_seq()
        
        # 轨迹历史
        self.trajectory = [self.agent_pos.copy()]
        
        # PD 控制器参数
        self.k_p = 10.0
        self.k_v = 5.0
        self.dt = 0.005
        self.repeat_actions = 10
        
        # 控制模式
        self.auto_mode = False  # False: 手动控制, True: 自动推理
        self.keyboard_move_step = 0.05  # 键盘移动步长
        
        print("交互式环境初始化完成！")
        print("控制说明:")
        print("  - 方向键: 移动主动物体")
        print("  - Space: 切换自动/手动模式")
        print("  - R: 重置环境")
        print("  - Q/Escape: 退出")
        print("  - 1/2: 移动第一个/第二个目标")
    
    def _build_obs_seq(self):
        """构建观测序列（归一化）"""
        obs_orig = np.concatenate([
            self.agent_pos,
            self.agent_vel,
            self.first_goal,
            self.second_goal
        ])
        
        obs_mean = np.array(self.norm_params['obs_mean'])
        obs_std = np.array(self.norm_params['obs_std'])
        obs_norm = (obs_orig - obs_mean) / obs_std
        
        # 重复两次（序列长度=2）
        obs_seq_norm = np.stack([obs_norm, obs_norm])  # (2, 8)
        return torch.from_numpy(obs_seq_norm).float().unsqueeze(0).to(self.device)  # (1, 2, 8)
    
    def reset(self):
        """重置环境"""
        self.agent_pos = np.array([0.2, 0.2], dtype=np.float32)
        self.agent_vel = np.zeros(2, dtype=np.float32)
        self.first_goal = np.array([0.3, 0.7], dtype=np.float32)
        self.second_goal = np.array([0.7, 0.3], dtype=np.float32)
        self.obs_seq = self._build_obs_seq()
        self.trajectory = [self.agent_pos.copy()]
        print("环境已重置！")
    
    def move_agent(self, dx, dy):
        """手动移动主动物体"""
        self.agent_pos[0] = np.clip(self.agent_pos[0] + dx, 0.0, 1.0)
        self.agent_pos[1] = np.clip(self.agent_pos[1] + dy, 0.0, 1.0)
        self.agent_vel = np.array([dx, dy], dtype=np.float32) / self.dt  # 估算速度
        self.obs_seq = self._build_obs_seq()
        self.trajectory.append(self.agent_pos.copy())
    
    def step_auto(self):
        """自动推理一步"""
        # 推理下一个动作（归一化空间）
        next_action_norm = infer_single_step(
            self.model, self.obs_seq, self.ula_sampler,
            num_action_samples=512,
            temperature=1.0,
            device=self.device,
            return_debug_info=False
        )
        
        # 反归一化到原始空间
        action_orig = denormalize_action(next_action_norm, self.norm_params)
        
        # 在原始空间执行PD控制器
        new_pos = self.agent_pos.copy()
        new_vel = self.agent_vel.copy()
        
        for _ in range(self.repeat_actions):
            u_agent = self.k_p * (action_orig - new_pos) - self.k_v * new_vel
            new_pos = new_pos + new_vel * self.dt
            new_vel = new_vel + u_agent * self.dt
        
        new_pos = np.clip(new_pos, 0.0, 1.0)
        
        # 更新状态
        self.agent_pos = new_pos
        self.agent_vel = new_vel
        self.obs_seq = self._build_obs_seq()
        self.trajectory.append(self.agent_pos.copy())
        
        return action_orig


def test_model(
    model_path: str,
    data_dir: str,
    num_test_samples: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    交互式测试 Particle EBM 模型
    
    Args:
        model_path: 模型文件路径
        data_dir: 测试数据目录（用于加载归一化参数）
        num_test_samples: （此参数在交互模式下不使用）
        device: 计算设备
    """
    print("=" * 60)
    print("Particle EBM 模型 - 实时交互测试")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    model, checkpoint, norm_params = load_model(model_path, device)
    
    # 2. 加载归一化参数
    print("\n2. 加载归一化参数...")
    dataset = ParticleDataset(data_dir, obs_seq_len=checkpoint.get('obs_seq_len', 2))
    norm_params['obs_mean'] = dataset.obs_mean
    norm_params['obs_std'] = dataset.obs_std
    
    # 3. 创建 ULA 采样器
    print("\n3. 创建 ULA 采样器...")
    action_bounds = np.array([[-1.0, -1.0], [1.0, 1.0]])  # 归一化空间
    ula_sampler = ULASampler(
        bounds=action_bounds,
        step_size=0.1,
        num_steps=100,
        noise_scale=1.0,
        step_size_final=1e-5,
        step_size_power=2.0,
        delta_action_clip=0.1,
        device=device
    )
    print("ULA 采样器创建完成")
    
    # 4. 创建交互式环境
    print("\n4. 创建交互式环境...")
    env = InteractiveParticleEnv(model, norm_params, ula_sampler, device)
    
    # 5. 启动交互式可视化
    print("\n5. 启动交互式可视化...")
    run_interactive_visualization(env, device)
    
    print("\n测试完成！")


def run_interactive_visualization(env: InteractiveParticleEnv, device='cuda'):
    """
    运行交互式可视化
    """
    # 创建图形
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # 轨迹图
    ax2 = fig.add_subplot(gs[0, 1])  # 能量景观
    
    # 状态变量
    state = {
        'running': True,
        'goal_edit_mode': 0,  # 0: 不编辑, 1: 编辑第一个目标, 2: 编辑第二个目标
        'energy_resolution': 40,
        'update_energy': True,
        'last_update_time': time.time(),
        'energy_im': None,  # imshow对象（用于更新能量景观）
        'energy_cbar': None,  # colorbar对象（只创建一次）
        'contour_lines': []  # 等高线集合
    }
    
    def update_plot():
        """更新绘图"""
        # 清除左图
        ax1.clear()
        
        # 左图：轨迹
        trajectory = np.array(env.trajectory)
        if len(trajectory) > 1:
            ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, 
                     label='Trajectory', alpha=0.7)
        
        # 当前位置
        ax1.plot(env.agent_pos[0], env.agent_pos[1], 'ro', markersize=15, 
                 markeredgecolor='white', markeredgewidth=2, label='Agent', zorder=10)
        
        # 目标点
        ax1.plot(env.first_goal[0], env.first_goal[1], 'go', markersize=12, 
                 markeredgecolor='darkgreen', markeredgewidth=2, label='First Goal', zorder=10)
        ax1.plot(env.second_goal[0], env.second_goal[1], 'b*', markersize=20, 
                 markeredgecolor='darkblue', markeredgewidth=2, label='Second Goal', zorder=10)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'Particle Trajectory (Mode: {"AUTO" if env.auto_mode else "MANUAL"})')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.set_xlim([-0.1, 1.1])
        ax1.set_ylim([-0.1, 1.1])
        
        # 显示距离信息
        dist_first = np.linalg.norm(env.agent_pos - env.first_goal)
        dist_second = np.linalg.norm(env.agent_pos - env.second_goal)
        info_text = f'Dist to Goal 1: {dist_first:.4f}\nDist to Goal 2: {dist_second:.4f}'
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # 右图：能量景观
        if state['update_energy']:
            obs_seq_norm = env.obs_seq[0].cpu().numpy()  # (obs_seq_len, obs_dim)
            X_grid, Y_grid, energy_grid = compute_energy_landscape(
                env.model, obs_seq_norm, resolution=state['energy_resolution'], 
                norm_params=env.norm_params, device=device
            )
            
            # 清除旧的等高线
            for coll in state['contour_lines']:
                coll.remove()
            state['contour_lines'].clear()
            
            # 清除旧的轨迹和标记（但保留 imshow 和 colorbar）
            lines_to_remove = [line for line in ax2.lines]
            for line in lines_to_remove:
                line.remove()
            
            # 使用 imshow 显示能量景观（便于更新）
            if state['energy_im'] is None:
                # 第一次：创建 imshow 和 colorbar
                state['energy_im'] = ax2.imshow(energy_grid, extent=[-1, 1, -1, 1], origin='lower', 
                                                cmap='viridis', alpha=0.8, aspect='auto')
                # 添加等高线
                contour_coll = ax2.contour(X_grid, Y_grid, energy_grid, levels=20, 
                                          colors='black', alpha=0.3, linewidths=0.5)
                state['contour_lines'].extend(contour_coll.collections)
                # 创建 colorbar（只创建一次）
                state['energy_cbar'] = plt.colorbar(state['energy_im'], ax=ax2, label='Energy')
                # 设置坐标轴
                ax2.set_xlabel('X (Normalized)')
                ax2.set_ylabel('Y (Normalized)')
                ax2.set_aspect('equal')
                ax2.set_xlim([-1.1, 1.1])
                ax2.set_ylim([-1.1, 1.1])
            else:
                # 后续：只更新数据
                state['energy_im'].set_array(energy_grid)
                state['energy_im'].set_clim(vmin=energy_grid.min(), vmax=energy_grid.max())
                # 重新绘制等高线
                contour_coll = ax2.contour(X_grid, Y_grid, energy_grid, levels=20, 
                                          colors='black', alpha=0.3, linewidths=0.5)
                state['contour_lines'].extend(contour_coll.collections)
            
            # 叠加轨迹（归一化空间）
            if len(trajectory) > 1:
                traj_norm = 2.0 * (trajectory - 0.5)
                ax2.plot(traj_norm[:, 0], traj_norm[:, 1], 'b-', linewidth=2, 
                         label='Traj', alpha=0.7, zorder=5)
            
            # 当前位置（归一化空间）
            agent_norm = 2.0 * (env.agent_pos - 0.5)
            ax2.plot(agent_norm[0], agent_norm[1], 'ro', markersize=15, 
                     markeredgecolor='white', markeredgewidth=2, label='Agent', zorder=10)
            
            # 目标点（归一化空间）
            first_goal_norm = 2.0 * (env.first_goal - 0.5)
            second_goal_norm = 2.0 * (env.second_goal - 0.5)
            ax2.plot(first_goal_norm[0], first_goal_norm[1], 'go', markersize=12, 
                     markeredgecolor='darkgreen', markeredgewidth=2, label='Goal 1', zorder=10)
            ax2.plot(second_goal_norm[0], second_goal_norm[1], 'b*', markersize=20, 
                     markeredgecolor='darkblue', markeredgewidth=2, label='Goal 2', zorder=10)
            
            # 更新标题和标签（每次都需要更新）
            ax2.set_xlabel('X (Normalized)')
            ax2.set_ylabel('Y (Normalized)')
            ax2.set_title('Energy Landscape')
            ax2.legend(loc='upper right')
            ax2.set_aspect('equal')
            ax2.set_xlim([-1.1, 1.1])
            ax2.set_ylim([-1.1, 1.1])
        else:
            # 如果不更新能量景观，显示提示文本
            if state['energy_im'] is None:
                ax2.text(0.5, 0.5, 'Energy landscape\n(press U to update)', 
                        ha='center', va='center', fontsize=14)
                ax2.set_title('Energy Landscape')
        
        plt.tight_layout()
        fig.canvas.draw()
    
    def on_key_press(event):
        """键盘事件处理"""
        if not state['running']:
            return
        
        key = event.key.lower() if event.key else ''
        
        # 退出
        if key in ['q', 'escape']:
            state['running'] = False
            plt.close(fig)
            print("退出交互模式")
            return
        
        # 重置
        elif key == 'r':
            env.reset()
            state['update_energy'] = True
            print("环境已重置")
        
        # 切换模式
        elif key == ' ':
            env.auto_mode = not env.auto_mode
            print(f"切换到 {'自动' if env.auto_mode else '手动'} 模式")
        
        # 更新能量景观
        elif key == 'u':
            state['update_energy'] = True
            print("更新能量景观")
        
        # 切换目标编辑模式
        elif key == '1':
            state['goal_edit_mode'] = 1 if state['goal_edit_mode'] != 1 else 0
            print(f"{'进入' if state['goal_edit_mode'] == 1 else '退出'} 第一个目标编辑模式")
        
        elif key == '2':
            state['goal_edit_mode'] = 2 if state['goal_edit_mode'] != 2 else 0
            print(f"{'进入' if state['goal_edit_mode'] == 2 else '退出'} 第二个目标编辑模式")
        
        # 方向键控制
        elif key == 'up':
            if state['goal_edit_mode'] == 1:
                env.first_goal[1] = np.clip(env.first_goal[1] + env.keyboard_move_step, 0.0, 1.0)
                env.obs_seq = env._build_obs_seq()
                state['update_energy'] = True
            elif state['goal_edit_mode'] == 2:
                env.second_goal[1] = np.clip(env.second_goal[1] + env.keyboard_move_step, 0.0, 1.0)
                env.obs_seq = env._build_obs_seq()
                state['update_energy'] = True
            elif not env.auto_mode:
                env.move_agent(0, env.keyboard_move_step)
                state['update_energy'] = True
        
        elif key == 'down':
            if state['goal_edit_mode'] == 1:
                env.first_goal[1] = np.clip(env.first_goal[1] - env.keyboard_move_step, 0.0, 1.0)
                env.obs_seq = env._build_obs_seq()
                state['update_energy'] = True
            elif state['goal_edit_mode'] == 2:
                env.second_goal[1] = np.clip(env.second_goal[1] - env.keyboard_move_step, 0.0, 1.0)
                env.obs_seq = env._build_obs_seq()
                state['update_energy'] = True
            elif not env.auto_mode:
                env.move_agent(0, -env.keyboard_move_step)
                state['update_energy'] = True
        
        elif key == 'left':
            if state['goal_edit_mode'] == 1:
                env.first_goal[0] = np.clip(env.first_goal[0] - env.keyboard_move_step, 0.0, 1.0)
                env.obs_seq = env._build_obs_seq()
                state['update_energy'] = True
            elif state['goal_edit_mode'] == 2:
                env.second_goal[0] = np.clip(env.second_goal[0] - env.keyboard_move_step, 0.0, 1.0)
                env.obs_seq = env._build_obs_seq()
                state['update_energy'] = True
            elif not env.auto_mode:
                env.move_agent(-env.keyboard_move_step, 0)
                state['update_energy'] = True
        
        elif key == 'right':
            if state['goal_edit_mode'] == 1:
                env.first_goal[0] = np.clip(env.first_goal[0] + env.keyboard_move_step, 0.0, 1.0)
                env.obs_seq = env._build_obs_seq()
                state['update_energy'] = True
            elif state['goal_edit_mode'] == 2:
                env.second_goal[0] = np.clip(env.second_goal[0] + env.keyboard_move_step, 0.0, 1.0)
                env.obs_seq = env._build_obs_seq()
                state['update_energy'] = True
            elif not env.auto_mode:
                env.move_agent(env.keyboard_move_step, 0)
                state['update_energy'] = True
        
        # 刷新显示
        update_plot()
    
    def on_timer(frame):
        """定时器：自动模式下推理"""
        if not state['running']:
            return
        
        if env.auto_mode:
            # 限制更新频率
            current_time = time.time()
            if current_time - state['last_update_time'] > 0.1:  # 每 0.1 秒推理一次
                env.step_auto()
                state['update_energy'] = True
                state['last_update_time'] = current_time
                update_plot()
    
    # 绑定事件
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # 初始绘制
    update_plot()
    
    # 启动定时器（用于自动模式）
    timer = fig.canvas.new_timer(interval=100)  # 每 100ms 检查一次
    timer.add_callback(on_timer, None)
    timer.start()
    
    # 显示窗口
    plt.show()
    
    print("交互式可视化结束")


if __name__ == '__main__':
    import argparse
    
    # 默认模型路径（最新的 checkpoint）
    default_model_dir = IBC_ROOT / 'models' / '_2d' / 'particle' / 'checkpoints'
    default_model_path = None
    if default_model_dir.exists():
        checkpoint_files = sorted(default_model_dir.glob('checkpoint_*.pth'))
        if checkpoint_files:
            default_model_path = str(checkpoint_files[-1])  # 使用最新的 checkpoint
        elif (default_model_dir.parent / 'final_model.pth').exists():
            default_model_path = str(default_model_dir.parent / 'final_model.pth')
    
    parser = argparse.ArgumentParser(description='实时交互测试 Particle EBM 模型')
    parser.add_argument('--model_path', type=str, default=default_model_path,
                       help=f'模型文件路径 (默认: {default_model_path})')
    parser.add_argument('--data_dir', type=str,
                       default=str(IBC_ROOT / 'data' / '_2d' / 'particle'),
                       help='数据目录（用于加载归一化参数）')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.model_path is None:
        print("错误: 未找到模型文件，请使用 --model_path 指定模型路径")
        print(f"或者确保模型目录存在: {default_model_dir}")
        exit(1)
    
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        num_test_samples=None,  # 交互模式不需要此参数
        device=device
    )

