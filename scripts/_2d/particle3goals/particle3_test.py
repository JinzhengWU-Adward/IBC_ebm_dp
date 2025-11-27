"""
Particle 环境 EBM 模型测试脚本
加载训练好的模型并进行推理和可视化
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

# 添加项目路径
IBC_ROOT = Path(__file__).parent.parent.parent.parent  # IBC_ebm_dp
sys.path.insert(0, str(IBC_ROOT))

from core.models import SequenceEBM
from core.optimizers import ULASampler

# 导入训练脚本中的数据集类
sys.path.insert(0, str(Path(__file__).parent))
from particle3_train import ParticleDataset

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = Path(__file__).parent.parent.parent / 'plots' / 'particle3goals'
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
    hidden_dim = checkpoint.get('hidden_dim', 128)
    num_residual_blocks = checkpoint.get('num_residual_blocks', 8)
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
    current_obs_orig = denormalize_obs(current_obs_norm, norm_params)  # (obs_dim,) 原始空间 [0, 1]
    # 对应 Particle3Goals 观测顺序:
    # [pos_agent(2), vel_agent(2), first_goal(2), second_goal(2), third_goal(2)]
    current_pos_orig = current_obs_orig[:2]          # pos_agent 原始空间
    current_vel_orig = current_obs_orig[2:4]         # vel_agent 原始空间
    pos_first_goal_orig = current_obs_orig[4:6]      # 第一个目标 原始空间
    pos_second_goal_orig = current_obs_orig[6:8]     # 第二个目标 原始空间
    pos_third_goal_orig = current_obs_orig[8:10]     # 第三个目标 原始空间
    
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
            print(f"  第三个目标 (原始空间): {pos_third_goal_orig}")
            print(f"  预测动作 (原始空间): {action_orig}")
            print(f"  到第一个目标的距离: {np.linalg.norm(current_pos_orig - pos_first_goal_orig):.4f}")
            print(f"  到第二个目标的距离: {np.linalg.norm(current_pos_orig - pos_second_goal_orig):.4f}")
            print(f"  到第三个目标的距离: {np.linalg.norm(current_pos_orig - pos_third_goal_orig):.4f}")
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
            pos_second_goal_orig,  # pos_second_goal (2) 原始空间
            pos_third_goal_orig    # pos_third_goal (2) 原始空间
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
    third_goal: np.ndarray,
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
    ax1.plot(second_goal[0], second_goal[1], 'b*', markersize=16, 
             markeredgecolor='darkblue', markeredgewidth=2, label='Second Goal', zorder=10)
    ax1.plot(third_goal[0], third_goal[1], 'r^', markersize=18,
             markeredgecolor='darkred', markeredgewidth=2, label='Third Goal', zorder=10)
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
        third_goal_norm = 2.0 * (third_goal - 0.5)
        
        ax2.plot(true_traj_norm[:, 0], true_traj_norm[:, 1], 'g-', linewidth=2, 
                 label='True Traj', alpha=0.9, zorder=5)
        ax2.plot(pred_traj_norm[:, 0], pred_traj_norm[:, 1], 'r--', linewidth=2, 
                 label='Pred Traj', alpha=0.9, zorder=5)
        ax2.plot(start_norm[0], start_norm[1], 'bs', markersize=15, 
                 markeredgecolor='cyan', markeredgewidth=2, label='Start', zorder=10)
        ax2.plot(first_goal_norm[0], first_goal_norm[1], 'go', markersize=12, 
                 markeredgecolor='darkgreen', markeredgewidth=2, label='First Goal', zorder=10)
        ax2.plot(second_goal_norm[0], second_goal_norm[1], 'b*', markersize=16, 
                 markeredgecolor='darkblue', markeredgewidth=2, label='Second Goal', zorder=10)
        ax2.plot(third_goal_norm[0], third_goal_norm[1], 'r^', markersize=18, 
                 markeredgecolor='darkred', markeredgewidth=2, label='Third Goal', zorder=10)
        
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
    third_goal: np.ndarray,
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
            third_goal_norm = 2.0 * (third_goal - 0.5)
            
            ax2.plot(start_norm[0], start_norm[1], 'bs', markersize=15, 
                     markeredgecolor='cyan', markeredgewidth=2, label='Start', zorder=10)
            ax2.plot(first_goal_norm[0], first_goal_norm[1], 'go', markersize=12, 
                     markeredgecolor='darkgreen', markeredgewidth=2, label='First Goal', zorder=10)
            ax2.plot(second_goal_norm[0], second_goal_norm[1], 'b*', markersize=16, 
                     markeredgecolor='darkblue', markeredgewidth=2, label='Second Goal', zorder=10)
            ax2.plot(third_goal_norm[0], third_goal_norm[1], 'r^', markersize=18, 
                     markeredgecolor='darkred', markeredgewidth=2, label='Third Goal', zorder=10)
            
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
    
    # 兼容无 third_goal_position 的旧数据：若不存在则将第三个目标设为第二个目标
    first_goal = np.array(data['first_goal_position'], dtype=np.float32)
    second_goal = np.array(data['second_goal_position'], dtype=np.float32)
    third_key = 'third_goal_position'
    if third_key in data:
        third_goal = np.array(data[third_key], dtype=np.float32)
    else:
        third_goal = second_goal.copy()

    return {
        'positions': np.array(data['trajectory']['positions'], dtype=np.float32),
        'actions': np.array(data['actions'], dtype=np.float32),
        'start_position': np.array(data['start_position'], dtype=np.float32),
        'first_goal': first_goal,
        'second_goal': second_goal,
        'third_goal': third_goal,
        'target_position': np.array(data['target_position'], dtype=np.float32)
    }


def test_model(
    model_path: str,
    data_dir: str,
    num_test_samples: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    测试 Particle EBM 模型
    
    Args:
        model_path: 模型文件路径
        data_dir: 测试数据目录
        num_test_samples: 测试样本数量
        device: 计算设备
    """
    print("=" * 60)
    print("Particle EBM 模型测试")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    model, checkpoint, norm_params = load_model(model_path, device)
    
    # 2. 加载测试数据
    print("\n2. 加载测试数据...")
    dataset = ParticleDataset(data_dir, obs_seq_len=checkpoint.get('obs_seq_len', 2))
    print(f"数据集大小: {len(dataset)}")
    
    # 获取所有 JSON 文件
    data_path = Path(data_dir)
    traj_dir = data_path / 'traj'
    json_files = sorted(traj_dir.glob('traj_*.json'))
    print(f"找到 {len(json_files)} 个 episode 文件")
    
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
        delta_action_clip=0.1,  # 匹配 IBC 的 delta_action_clip（关键修复！）
        device=device
    )
    print("ULA 采样器创建完成")
    
    # 4. 进行测试
    print(f"\n4. 测试 {num_test_samples} 个样本...")
    # 选择不同的 episode 进行测试
    episode_indices = np.linspace(0, len(json_files) - 1, num_test_samples, dtype=int)
    
    all_errors = []
    
    for idx, episode_idx in enumerate(tqdm(episode_indices, desc="测试中")):
        # 加载完整 episode 数据
        episode_data = load_episode_data(data_dir, episode_idx)
        if episode_data is None:
            continue
        
        # 获取真实轨迹
        true_positions = episode_data['positions']  # (T, 2) 原始空间 [0, 1]
        true_actions = episode_data['actions']  # (T-1, 2) 原始空间 [0, 1]
        start_pos = episode_data['start_position']  # (2,)
        pos_first_goal = episode_data['first_goal']   # (2,)
        pos_second_goal = episode_data['second_goal'] # (2,)
        pos_third_goal = episode_data['third_goal']   # (2,)
        
        # 从 episode 数据直接构建初始观测序列，确保初始速度为 0
        initial_pos = episode_data['start_position']  # 原始空间 [0, 1]
        initial_vel = np.zeros(2, dtype=np.float32)  # 初始速度为 0（匹配 IBC）
        first_goal = episode_data['first_goal']   # 原始空间
        second_goal = episode_data['second_goal'] # 原始空间
        third_goal = episode_data['third_goal']   # 原始空间

        # 构建初始观测（原始空间）
        initial_obs = np.concatenate([
            initial_pos,      # pos_agent (2)
            initial_vel,      # vel_agent (2) = 0
            first_goal,       # pos_first_goal (2)
            second_goal,      # pos_second_goal (2)
            third_goal        # pos_third_goal (2)
        ])

        # 归一化（使用数据集的归一化参数）
        obs_mean = np.array(dataset.obs_mean)
        obs_std = np.array(dataset.obs_std)
        initial_obs_norm = (initial_obs - obs_mean) / obs_std  # (8,)

        # 构建序列（重复两次，因为序列长度=2）
        # 匹配 IBC 的 HistoryWrapper：tile_first_step_obs=True
        obs_seq_norm = np.stack([initial_obs_norm, initial_obs_norm])  # (2, 8)
        
        # 推理轨迹（返回中间状态用于视频生成）
        obs_seq_tensor = torch.from_numpy(obs_seq_norm).float().unsqueeze(0).to(device)
        pred_trajectory_norm, intermediate_states = infer_trajectory(
            model, obs_seq_tensor, ula_sampler,
            max_steps=max(80, len(true_positions)),
            num_action_samples=512,
            temperature=1.0,
            device=device,
            return_intermediate=True,
            norm_params=norm_params
        )
        
        # ★ 修复：从intermediate_states直接提取位置（已经是原始空间）
        pred_positions = [state['pos'] for state in intermediate_states]
        pred_positions = np.array(pred_positions)  # (T+1, 2) 原始空间 [0, 1]
        
        # 对齐长度用于误差计算
        min_len = min(len(true_positions), len(pred_positions))
        true_traj_for_error = true_positions[:min_len]
        pred_traj_for_error = pred_positions[:min_len]
        
        # 可视化使用完整轨迹（不截断真实轨迹）
        true_traj_for_vis = true_positions  # 使用完整的真实轨迹
        pred_traj_for_vis = pred_positions  # 预测轨迹保持原样
        
        # 可视化（静态图片）
        save_path = output_dir / f'test_sample_{idx:03d}.png'
        
        # 获取最终观测序列用于能量地形图（使用归一化的观测）
        if intermediate_states and len(intermediate_states) > 0:
            obs_seq_final = intermediate_states[-1]['obs_seq'][0].cpu().numpy()  # (obs_seq_len, obs_dim) 归一化空间
        else:
            obs_seq_final = obs_seq_norm
        
        # 获取最终步骤的调试信息
        debug_info_final = None
        if intermediate_states and len(intermediate_states) > 0:
            debug_info_final = intermediate_states[-1].get('debug_info', None)
        
        # 使用完整轨迹进行可视化
        errors = visualize_trajectory(
            true_traj_for_vis, pred_traj_for_vis,
            pos_first_goal, pos_second_goal, pos_third_goal, start_pos,
            save_path,
            model=model,
            obs_seq_final=obs_seq_final,
            norm_params=norm_params,
            device=device,
            debug_info_final=debug_info_final
        )
        
        # 误差计算基于对齐的轨迹
        errors_aligned = {
            'mean_error': errors['mean_error'],  # visualize_trajectory 内部会计算对齐后的误差
            'max_error': errors['max_error'],
            'final_error': errors['final_error']
        }
        
        all_errors.append(errors_aligned)
        
        # 生成视频（使用完整轨迹）
        video_path = output_dir / f'test_sample_{idx:03d}.mp4'
        create_trajectory_video(
            true_traj_for_vis, pred_traj_for_vis,
            intermediate_states,
            pos_first_goal, pos_second_goal, pos_third_goal, start_pos,
            video_path,
            model=model,
            norm_params=norm_params,
            device=device,
            fps=10
        )
    
    # 5. 打印统计信息
    print("\n" + "=" * 60)
    print("5. 测试结果统计:")
    print("=" * 60)
    if all_errors:
        mean_errors = [e['mean_error'] for e in all_errors]
        final_errors = [e['final_error'] for e in all_errors]
        max_errors = [e['max_error'] for e in all_errors]
        
        print(f"平均轨迹误差: {np.mean(mean_errors):.4f} ± {np.std(mean_errors):.4f}")
        print(f"终点误差: {np.mean(final_errors):.4f} ± {np.std(final_errors):.4f}")
        print(f"最大误差: {np.mean(max_errors):.4f} ± {np.std(max_errors):.4f}")
    
    print(f"\n所有可视化结果已保存到: {output_dir}")
    print("测试完成！")


if __name__ == '__main__':
    import argparse
    
    # 默认模型路径（最新的 checkpoint）
    default_model_dir = IBC_ROOT / 'models' / '_2d' / 'particle3goals' / 'checkpoints'
    default_model_path = None
    if default_model_dir.exists():
        checkpoint_files = sorted(default_model_dir.glob('checkpoint_*.pth'))
        if checkpoint_files:
            default_model_path = str(checkpoint_files[-1])  # 使用最新的 checkpoint
        elif (default_model_dir.parent / 'final_model.pth').exists():
            default_model_path = str(default_model_dir.parent / 'final_model.pth')
    
    parser = argparse.ArgumentParser(description='测试 Particle EBM 模型')
    parser.add_argument('--model_path', type=str, default=default_model_path,
                       help=f'模型文件路径 (默认: {default_model_path})')
    parser.add_argument('--data_dir', type=str,
                       default=str(IBC_ROOT / 'data' / '_2d' / 'particle3goals'),
                       help='测试数据目录')
    parser.add_argument('--num_test_samples', type=int, default=10,
                       help='测试样本数量')
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
        num_test_samples=args.num_test_samples,
        device=device
    )

