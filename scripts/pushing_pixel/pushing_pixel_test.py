"""
Block Pushing (RGB 图像) 环境 EBM 模型测试脚本
在 pybullet 环境中加载训练好的 PixelEBM 模型并进行评估
复现 IBC 的 run_pixel_ebm_langevin.sh 评估场景
"""
import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from tqdm import tqdm
import copy
import os
from datetime import datetime
import time
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("警告: 未安装 opencv-python，无法保存视频。请运行: pip install opencv-python")
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: 未安装 matplotlib，无法绘制能量图景。请运行: pip install matplotlib")

# 添加项目路径
IBC_ROOT = Path(__file__).parent.parent.parent  # IBC_ebm_dp
sys.path.insert(0, str(IBC_ROOT))

from core.models import PixelEBM
from core.optimizers import ULASampler

# 添加 IBC 路径以便导入环境
IBC_PARENT_DIR = Path(__file__).parent.parent.parent.parent
if str(IBC_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(IBC_PARENT_DIR))

# 兼容 gym / tf-agents 的 patch（需在 tf-agents 之前 import）
from ibc.ibc.utils import gym_compat  # noqa: F401

from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

from ibc.environments.block_pushing import block_pushing


def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载训练好的 PixelEBM 模型
    
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
    action_dim = checkpoint.get('action_dim', 2)
    obs_seq_len = checkpoint.get('obs_seq_len', 2)
    target_height = checkpoint.get('target_height', 180)
    target_width = checkpoint.get('target_width', 240)
    image_channels = checkpoint.get('image_channels', 6)  # 3 * sequence_length
    value_width = checkpoint.get('value_width', 1024)
    value_num_blocks = checkpoint.get('value_num_blocks', 1)
    norm_params = checkpoint.get('norm_params', None)
    
    print(f"模型参数:")
    print(f"  action_dim={action_dim}, obs_seq_len={obs_seq_len}")
    print(f"  target_height={target_height}, target_width={target_width}")
    print(f"  image_channels={image_channels}, value_width={value_width}, value_num_blocks={value_num_blocks}")
    
    # 创建 PixelEBM
    model = PixelEBM(
        image_channels=image_channels,
        action_dim=action_dim,
        target_height=target_height,
        target_width=target_width,
        value_width=value_width,
        value_num_blocks=value_num_blocks
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("模型加载完成！")
    return model, checkpoint, norm_params


def denormalize_action(action_norm, norm_params):
    """
    将归一化的动作反归一化到原始空间
    
    Args:
        action_norm: 归一化动作 (2,)，范围 [-1, 1]
        norm_params: 归一化参数字典
    
    Returns:
        action_orig: 原始动作 (2,)
    """
    if norm_params is None or norm_params.get('action_min') is None:
        return action_norm
    
    action_min = np.array(norm_params['action_min'])
    action_max = np.array(norm_params['action_max'])
    action_range = action_max - action_min
    
    # 从 [-1, 1] 反归一化到原始范围
    action_orig = (action_norm + 1.0) / 2.0 * action_range + action_min
    return action_orig


# 注意：normalize_image 函数已不再使用，因为 stack_image_sequence 现在直接处理
# 保留此函数以防其他地方需要，但推荐使用 stack_image_sequence
def normalize_image(image, target_height=180, target_width=240):
    """
    归一化图像到 [0, 1] 并 resize 到目标尺寸（已废弃，推荐使用 stack_image_sequence）
    
    Args:
        image: 输入图像，形状为 (H, W, 3)，值范围 [0, 255]
        target_height: 目标高度
        target_width: 目标宽度
    
    Returns:
        image_norm: 归一化后的图像，形状为 (target_height, target_width, 3)，值范围 [0, 1]
    """
    # 转换为 float32 并归一化到 [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Resize（如果需要）
    if image.shape[0] != target_height or image.shape[1] != target_width:
        import cv2
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    return image


def stack_image_sequence(images_list, target_height=180, target_width=240):
    """
    准备图像序列用于模型输入（匹配 IBC 官方流程）
    
    ⚠️ 关键修复：匹配训练代码的修改
    - 只归一化，不进行堆叠和 resize
    - 堆叠和 resize 在模型内部的 encode 方法中执行（GPU 上）
    
    Args:
        images_list: 图像列表，每个图像形状为 (H, W, 3)，值范围 [0, 255]
        target_height: 目标高度（用于文档，实际 resize 在模型内部执行）
        target_width: 目标宽度（用于文档，实际 resize 在模型内部执行）
    
    Returns:
        image_sequence: 图像序列，形状为 (seq_len, H_orig, W_orig, 3)
                       值范围 [0, 1]，保持原始尺寸
    """
    # 只归一化，不进行堆叠和 resize（匹配训练代码）
    normalized_images = []
    for img in images_list:
        img_norm = img.astype(np.float32) / 255.0  # (H, W, 3)
        normalized_images.append(img_norm)
    
    # 转换为序列格式 (seq_len, H, W, 3)，保持原始尺寸
    image_sequence = np.stack(normalized_images, axis=0)  # (seq_len, H_orig, W_orig, 3)
    
    return image_sequence


def plot_energy_landscape(
    model: PixelEBM,
    images: torch.Tensor,  # (1, seq_len, H_orig, W_orig, 3) 或 (seq_len, H_orig, W_orig, 3)
    norm_params: dict,
    action_range: tuple = None,  # ((min_x, max_x), (min_y, max_y))，如果为None则从norm_params推断
    grid_size: int = 50,
    current_action: np.ndarray = None,  # 当前选择的动作（反归一化后的原始空间），形状为 (2,)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> np.ndarray:
    """
    绘制能量图景（反归一化后的动作空间）
    
    Args:
        model: PixelEBM 模型
        images: 图像序列，形状为 (1, seq_len, H_orig, W_orig, 3) 或 (seq_len, H_orig, W_orig, 3)
               注意：保持原始尺寸，堆叠和 resize 在模型内部执行
        norm_params: 归一化参数
        action_range: 动作范围 ((min_x, max_x), (min_y, max_y))，如果为None则从norm_params推断
        grid_size: 网格大小（每边的点数）
        device: 计算设备
    
    Returns:
        energy_frame: 能量图景图像，形状为 (H, W, 3)，值范围 [0, 255]
    """
    if not HAS_MATPLOTLIB:
        return None
    
    model.eval()
    
    # 获取动作范围（反归一化后的原始空间）
    if action_range is None:
        if norm_params is not None and norm_params.get('action_min') is not None:
            action_min = np.array(norm_params['action_min'])
            action_max = np.array(norm_params['action_max'])
            # 稍微扩展范围以便可视化
            action_range_x = (action_min[0] - 0.01, action_max[0] + 0.01)
            action_range_y = (action_min[1] - 0.01, action_max[1] + 0.01)
        else:
            # 默认范围
            action_range_x = (-0.05, 0.05)
            action_range_y = (-0.05, 0.05)
    else:
        action_range_x = action_range[0]
        action_range_y = action_range[1]
    
    # 创建动作网格（反归一化空间）
    x_vals = np.linspace(action_range_x[0], action_range_x[1], grid_size)
    y_vals = np.linspace(action_range_y[0], action_range_y[1], grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # 将网格转换为归一化空间
    if norm_params is not None and norm_params.get('action_min') is not None:
        action_min = np.array(norm_params['action_min'])
        action_max = np.array(norm_params['action_max'])
        action_range_np = action_max - action_min
        action_range_np[action_range_np < 1e-6] = 1.0
        
        # 反归一化公式的逆：从原始空间到归一化空间
        X_norm = 2.0 * (X - action_min[0]) / action_range_np[0] - 1.0
        Y_norm = 2.0 * (Y - action_min[1]) / action_range_np[1] - 1.0
    else:
        X_norm = X
        Y_norm = Y
    
    # 构建动作网格 (grid_size * grid_size, 2)
    actions_norm = np.stack([
        X_norm.flatten(),
        Y_norm.flatten()
    ], axis=1)  # (grid_size^2, 2)
    
    # 转换为 Tensor
    actions_norm_tensor = torch.from_numpy(actions_norm).float().to(device)  # (grid_size^2, 2)
    actions_norm_tensor = actions_norm_tensor.unsqueeze(0)  # (1, grid_size^2, 2)
    
    # 计算能量
    with torch.no_grad():
        # Late Fusion: 先编码图像
        obs_encoding = model.encode(images)  # (1, 256)
        
        # 计算能量
        energies = model(
            images=None,
            actions=actions_norm_tensor,
            obs_encoding=obs_encoding
        )  # (1, grid_size^2)
        
        energies_np = energies.squeeze(0).cpu().numpy()  # (grid_size^2,)
    
    # 重塑为网格形状
    energy_grid = energies_np.reshape(grid_size, grid_size)
    
    # 绘制能量图景
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 使用 pcolormesh 绘制热力图
    im = ax.pcolormesh(X, Y, energy_grid, cmap='viridis', shading='auto')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Energy', rotation=270, labelpad=20)
    
    # 设置坐标轴标签
    ax.set_xlabel('Delta X (m)', fontsize=12)
    ax.set_ylabel('Delta Y (m)', fontsize=12)
    ax.set_title('Energy Landscape', fontsize=14)
    
    # 设置坐标轴范围
    ax.set_xlim(action_range_x[0], action_range_x[1])
    ax.set_ylim(action_range_y[0], action_range_y[1])
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 标记当前选择的动作（如果提供）
    if current_action is not None:
        ax.plot(current_action[0], current_action[1], 'r*', 
                markersize=15, markeredgewidth=2, markeredgecolor='white',
                label='Selected Action')
        ax.legend(loc='upper right')
    
    # 转换为图像数组
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return buf


def infer_action(
    model: PixelEBM,
    images: torch.Tensor,  # (1, seq_len, H_orig, W_orig, 3) 或 (seq_len, H_orig, W_orig, 3)
    ula_sampler: ULASampler,
    num_action_samples: int = 2048,  # 匹配 gin: IbcPolicy.num_action_samples = 2048
    optimize_again: bool = True,  # 匹配 gin: IbcPolicy.optimize_again = True
    again_noise_scale: float = 1.0,  # 匹配 gin: IbcPolicy.inference_langevin_noise_scale = 1.0 (默认值)
    again_step_size_init: float = 1e-1,  # 匹配 gin: IbcPolicy.again_stepsize_init = 1e-1
    again_step_size_final: float = 1e-5,  # 匹配 gin: IbcPolicy.again_stepsize_final = 1e-5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    使用 ULA 推理下一个动作（在归一化空间，支持 Late Fusion）
    
    匹配 IBC 官方的 optimize_again 功能：
    1. 第一次 ULA 采样（带噪声）
    2. 如果 optimize_again=True，进行第二次 ULA 采样（无噪声或小噪声）
    3. 选择能量最低的动作
    
    Args:
        model: 训练好的 PixelEBM 模型
        images: 图像序列（归一化），形状为 (1, seq_len, H_orig, W_orig, 3) 或 (seq_len, H_orig, W_orig, 3)
               注意：保持原始尺寸，堆叠和 resize 在模型内部执行
        ula_sampler: ULASampler 实例（用于第一次采样）
        num_action_samples: 采样候选数量
        optimize_again: 是否进行二次优化（匹配 gin: IbcPolicy.optimize_again = True）
        again_noise_scale: 二次优化时的噪声尺度（通常为 0.0，无噪声）
        again_step_size_init: 二次优化的初始步长
        again_step_size_final: 二次优化的最终步长
        device: 计算设备
    
    Returns:
        next_action: 预测的动作（归一化空间），形状为 (2,)
    """
    model.eval()
    
    # Late Fusion: 先编码图像（在模型内部进行堆叠和 resize）
    with torch.no_grad():
        obs_encoding = model.encode(images)  # (1, 256)
    
    # 初始化候选动作（归一化空间 [-1, 1]）
    # 修复：匹配训练代码的边界扩展公式
    uniform_boundary_buffer = 0.05
    norm_min, norm_max = -1.0, 1.0
    expanded_min = norm_min - uniform_boundary_buffer * (norm_max - norm_min)
    expanded_max = norm_max + uniform_boundary_buffer * (norm_max - norm_min)
    
    init_candidates = torch.rand(
        1, num_action_samples, 2,
        device=device
    ) * (expanded_max - expanded_min) + expanded_min
    
    # 第一次 ULA 采样（使用 Late Fusion，带噪声）
    with torch.enable_grad():
        candidates, _ = ula_sampler.sample(
            x=images,  # 传入 images（虽然会被忽略，因为使用 obs_encoding）
            ebm=model,
            num_samples=num_action_samples,
            init_samples=init_candidates,
            return_trajectory=False,
            obs_encoding=obs_encoding.detach()  # Late Fusion
        )
    
    # 二次优化（optimize_again）：无噪声的 ULA 采样
    # 匹配 IBC 官方的实现：使用相同的 langevin_actions_given_obs，但 noise_scale=0.0
    if optimize_again:
        # 创建二次优化的 ULA 采样器（使用更小的步长和零噪声）
        from core.optimizers import ULASampler
        uniform_boundary_buffer = 0.05
        norm_min, norm_max = -1.0, 1.0
        expanded_min = norm_min - uniform_boundary_buffer * (norm_max - norm_min) * 2.0
        expanded_max = norm_max + uniform_boundary_buffer * (norm_max - norm_min) * 2.0
        action_bounds = np.array([[expanded_min, expanded_min],
                                  [expanded_max, expanded_max]], dtype=np.float32)
        
        again_ula_sampler = ULASampler(
            bounds=action_bounds,
            step_size=again_step_size_init,
            num_steps=ula_sampler.num_steps,  # 使用相同的步数
            noise_scale=again_noise_scale,  # 无噪声或小噪声
            step_size_final=again_step_size_final,
            step_size_power=ula_sampler.step_size_power,
            delta_action_clip=ula_sampler.delta_action_clip,
            device=device
        )
        
        # 第二次 ULA 采样（无噪声）
        with torch.enable_grad():
            candidates, _ = again_ula_sampler.sample(
                x=images,
                ebm=model,
                num_samples=num_action_samples,
                init_samples=candidates,  # 使用第一次采样的结果作为初始值
                return_trajectory=False,
                obs_encoding=obs_encoding.detach()  # Late Fusion
            )
    
    # 计算能量并选择最优动作
    with torch.no_grad():
        energies = model(
            images=None,  # 不使用 images，使用 obs_encoding
            actions=candidates,
            obs_encoding=obs_encoding
        )  # (1, num_action_samples)
        best_idx = energies.argmin(dim=1)  # (1,)
        next_action = candidates[0, best_idx[0]]  # (2,)
    
    return next_action.cpu().numpy()


def evaluate_policy(
    model: PixelEBM,
    norm_params: dict,
    num_episodes: int = 20,
    max_steps: int = 100,
    num_action_samples: int = 2048,  # 匹配 gin: IbcPolicy.num_action_samples = 2048
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    shared_memory: bool = False,
    seed: int = None,
    save_video: bool = False,
    video_output_dir: str = None,
    video_episodes: int = 1,  # 保存前 N 个 episodes 的视频
    save_energy_landscape: bool = True,  # 是否保存能量图景视频
    energy_episodes: int = 1  # 保存前 N 个 episodes 的能量图景
):
    """
    在 pybullet BlockPush 环境中评估策略（使用 RGB 图像观测）
    
    Args:
        model: 训练好的 PixelEBM 模型
        norm_params: 归一化参数
        num_episodes: 评估的 episode 数量
        max_steps: 每个 episode 的最大步数
        num_action_samples: ULA 采样候选数量
        device: 计算设备
        shared_memory: 是否使用共享内存（用于可视化）
        seed: 随机种子
        save_video: 是否保存视频
        video_output_dir: 视频输出目录
        video_episodes: 保存前 N 个 episodes 的视频
    
    Returns:
        results: 评估结果字典
    """
    print("=" * 60)
    print("在 pybullet BlockPush 环境中评估策略（RGB 图像观测）")
    print("=" * 60)
    print(f"Episode 数量: {num_episodes}")
    print(f"最大步数: {max_steps}")
    print(f"设备: {device}")
    print()
    
    # 获取模型参数
    obs_seq_len = norm_params.get('obs_seq_len', 2)
    target_height = norm_params.get('target_height', 180)
    target_width = norm_params.get('target_width', 240)
    
    # 创建环境（使用图像观测）
    env_name = block_pushing.build_env_name(
        task="PUSH", 
        shared_memory=shared_memory, 
        use_image_obs=True  # 使用图像观测
    )
    env = suite_gym.load(env_name)
    
    # 添加 HistoryWrapper（匹配训练时的配置）
    env = wrappers.HistoryWrapper(
        env, 
        history_length=obs_seq_len, 
        tile_first_step_obs=True
    )
    
    # 创建 ULA 采样器
    uniform_boundary_buffer = 0.05
    norm_min, norm_max = -1.0, 1.0
    expanded_min = norm_min - uniform_boundary_buffer * (norm_max - norm_min) * 2.0
    expanded_max = norm_max + uniform_boundary_buffer * (norm_max - norm_min) * 2.0
    action_bounds = np.array([[expanded_min, expanded_min],
                              [expanded_max, expanded_max]], dtype=np.float32)
    
    ula_sampler = ULASampler(
        bounds=action_bounds,
        step_size=0.1,  # 匹配训练时的配置
        num_steps=3,  # 匹配 gin: langevin_actions_given_obs.num_iterations = 100
        noise_scale=1.0,
        step_size_final=1e-5,
        step_size_power=2.0,
        delta_action_clip=0.1,
        device=device
    )
    
    # 评估指标
    successes = []
    final_goal_distances = []
    episode_lengths = []
    
    # 频率统计
    inference_times = []  # 推理时间列表
    execution_times = []  # 执行（IK + step）时间列表
    total_inference_time = 0.0
    total_execution_time = 0.0
    total_steps = 0
    
    # 视频保存相关
    video_writers = []
    if save_video and HAS_CV2:
        video_output_dir = Path(video_output_dir)
        video_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"视频将保存到: {video_output_dir}")
    
    print("开始评估...")
    for ep_idx in tqdm(range(num_episodes), desc="评估进度"):
        # 重置环境
        time_step = env.reset()
        
        # 初始化视频写入器（如果需要）
        video_writer = None
        video_frames = []
        video_path = None
        if save_video and HAS_CV2 and ep_idx < video_episodes:
            # 视频文件保存在 video_output_dir 目录下
            video_path = video_output_dir / f'episode_{ep_idx:03d}.mp4'
            # 先渲染一帧以获取实际尺寸
            try:
                if hasattr(env, 'pyenv'):
                    base_env = env.pyenv.envs[0]
                    if hasattr(base_env, 'render'):
                        test_frame = base_env.render(mode='rgb_array')
                        if test_frame is not None:
                            height, width = test_frame.shape[:2]
                        else:
                            height, width = 240, 320
                    else:
                        height, width = 240, 320
                else:
                    height, width = 240, 320
                
                # 确保尺寸是偶数（避免 codec 问题）
                if height % 2 != 0:
                    height -= 1
                if width % 2 != 0:
                    width -= 1
                
                # 尝试多种编码格式（按兼容性排序）
                fourcc_options = [
                    cv2.VideoWriter_fourcc(*'XVID'),  # 最兼容
                    cv2.VideoWriter_fourcc(*'mp4v'),  # 备选
                    cv2.VideoWriter_fourcc(*'H264'),  # 如果支持
                ]
                
                video_writer = None
                for fourcc in fourcc_options:
                    try:
                        video_writer = cv2.VideoWriter(
                            str(video_path), fourcc, 10.0, (width, height)
                        )
                        if video_writer.isOpened():
                            break
                        else:
                            video_writer.release()
                            video_writer = None
                    except:
                        if video_writer is not None:
                            video_writer.release()
                            video_writer = None
                        continue
                
                if video_writer is None or not video_writer.isOpened():
                    print(f"警告: 无法创建视频文件 {video_path}")
                    video_writer = None
                else:
                    print(f"  创建视频文件: {video_path} (尺寸: {width}x{height})")
            except Exception as e:
                print(f"警告: 创建视频写入器失败: {e}")
                import traceback
                traceback.print_exc()
                video_writer = None
        
        # 初始化图像序列（用于 HistoryWrapper）
        image_seq_list = []
        step_count = 0
        success = False
        final_goal_distance = None
        
        # 能量图景视频相关
        energy_writer = None
        energy_frames = []
        energy_path = None
        if save_energy_landscape and HAS_MATPLOTLIB and HAS_CV2 and ep_idx < energy_episodes:
            energy_path = video_output_dir / f'energy_landscape_episode_{ep_idx:03d}.mp4'
            # 先绘制一帧以获取尺寸
            try:
                # 创建一个临时图像来获取尺寸（匹配新的数据格式）
                # 使用原始尺寸的占位图像
                temp_h, temp_w = 240, 320  # 典型的原始图像尺寸
                temp_images_tensor = torch.zeros(1, obs_seq_len, temp_h, temp_w, 3).to(device)
                temp_energy_frame = plot_energy_landscape(
                    model, temp_images_tensor, norm_params, device=device
                )
                if temp_energy_frame is not None:
                    h, w = temp_energy_frame.shape[:2]
                    # 确保尺寸是偶数
                    if h % 2 != 0:
                        h -= 1
                    if w % 2 != 0:
                        w -= 1
                    
                    # 创建视频写入器
                    fourcc_options = [
                        cv2.VideoWriter_fourcc(*'XVID'),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        cv2.VideoWriter_fourcc(*'H264'),
                    ]
                    
                    for fourcc in fourcc_options:
                        try:
                            energy_writer = cv2.VideoWriter(
                                str(energy_path), fourcc, 2.0, (w, h)  # 2 FPS
                            )
                            if energy_writer.isOpened():
                                break
                            else:
                                energy_writer.release()
                                energy_writer = None
                        except:
                            if energy_writer is not None:
                                energy_writer.release()
                                energy_writer = None
                            continue
                    
                    if energy_writer is not None and energy_writer.isOpened():
                        print(f"  创建能量图景视频文件: {energy_path} (尺寸: {w}x{h})")
                    else:
                        print(f"  警告: 无法创建能量图景视频文件 {energy_path}")
                        energy_writer = None
            except Exception as e:
                print(f"  警告: 创建能量图景视频写入器失败: {e}")
                energy_writer = None
        
        # 保存第一帧（reset 后的初始状态）
        if video_writer is not None:
            try:
                # 获取底层环境（HistoryWrapper -> GymWrapper -> TFPyEnvironment -> BlockPush）
                base_env = env
                while hasattr(base_env, '_env'):
                    base_env = base_env._env
                
                # 如果是 TFPyEnvironment，获取实际的 gym 环境
                if hasattr(base_env, 'pyenv'):
                    base_env = base_env.pyenv.envs[0]
                
                # 调用 render 方法
                if hasattr(base_env, 'render'):
                    frame = base_env.render(mode='rgb_array')
                    if frame is not None and frame.size > 0:
                        h, w = frame.shape[:2]
                        if h % 2 != 0:
                            frame = frame[:h-1, :, :]
                        if w % 2 != 0:
                            frame = frame[:, :w-1, :]
                        if frame.shape[2] == 3:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            video_writer.write(frame_bgr)
                            video_frames.append(frame_bgr.copy())
                            print(f"    第一帧已保存 (尺寸: {w}x{h})")
            except Exception as e:
                print(f"    警告: 保存第一帧时出错: {e}")
        
        while not time_step.is_last() and step_count < max_steps:
            # 获取当前观测（RGB 图像）
            obs_dict = time_step.observation
            
            # HistoryWrapper 已经堆叠了观测，我们需要提取每个时间步的图像
            # obs_dict['rgb'] 的形状是 (history_length, H, W, 3)
            # 我们需要构建图像序列用于模型输入
            
            # 提取每个时间步的图像
            rgb_obs = obs_dict.get('rgb', None)
            if rgb_obs is None:
                print("警告: 观测中没有 'rgb' 字段")
                break
            
            rgb_obs = np.array(rgb_obs)
            
            # 处理 HistoryWrapper 的堆叠
            if rgb_obs.ndim == 4:
                # 形状为 (history_length, H, W, 3)
                image_seq_list = [rgb_obs[t] for t in range(rgb_obs.shape[0])]
            elif rgb_obs.ndim == 3:
                # 单个图像 (H, W, 3)，可能是第一步
                image_seq_list = [rgb_obs]
                # 如果序列长度不足，重复第一帧
                while len(image_seq_list) < obs_seq_len:
                    image_seq_list.insert(0, image_seq_list[0])
            else:
                print(f"警告: 意外的 rgb 观测形状: {rgb_obs.shape}")
                break
            
            # 确保序列长度为 obs_seq_len
            if len(image_seq_list) > obs_seq_len:
                image_seq_list = image_seq_list[-obs_seq_len:]
            elif len(image_seq_list) < obs_seq_len:
                # 如果不足，重复第一帧
                while len(image_seq_list) < obs_seq_len:
                    image_seq_list.insert(0, image_seq_list[0])
            
            # 准备图像序列（只归一化，不堆叠和 resize）
            image_sequence = stack_image_sequence(
                image_seq_list,
                target_height=target_height,
                target_width=target_width
            )  # (seq_len, H_orig, W_orig, 3)
            
            # 转换为 PyTorch 张量（添加 batch 维度）
            images_tensor = torch.from_numpy(image_sequence).float().unsqueeze(0).to(device)  # (1, seq_len, H_orig, W_orig, 3)
            
            # 推理动作（归一化空间）- 测量时间
            inference_start = time.perf_counter()
            action_norm = infer_action(
                model, images_tensor, ula_sampler,
                num_action_samples=num_action_samples,
                optimize_again=True,  # 匹配 gin: IbcPolicy.optimize_again = True
                again_noise_scale=1.0,  # 匹配 gin: IbcPolicy.inference_langevin_noise_scale = 1.0 (默认值)
                again_step_size_init=1e-1,  # 匹配 gin: IbcPolicy.again_stepsize_init = 1e-1
                again_step_size_final=1e-5,  # 匹配 gin: IbcPolicy.again_stepsize_final = 1e-5
                device=device
            )
            inference_end = time.perf_counter()
            inference_time = inference_end - inference_start
            inference_times.append(inference_time)
            total_inference_time += inference_time
            
            # 绘制并保存能量图景
            if energy_writer is not None and HAS_MATPLOTLIB:
                try:
                    # 反归一化动作到原始空间（用于标记）
                    action_orig_for_plot = denormalize_action(action_norm, norm_params)
                    energy_frame = plot_energy_landscape(
                        model, images_tensor, norm_params, 
                        current_action=action_orig_for_plot,
                        device=device
                    )
                    if energy_frame is not None:
                        # 转换为 BGR（OpenCV 格式）
                        energy_frame_bgr = cv2.cvtColor(energy_frame, cv2.COLOR_RGB2BGR)
                        # 确保尺寸是偶数
                        h, w = energy_frame_bgr.shape[:2]
                        if h % 2 != 0:
                            energy_frame_bgr = energy_frame_bgr[:h-1, :, :]
                            h -= 1
                        if w % 2 != 0:
                            energy_frame_bgr = energy_frame_bgr[:, :w-1, :]
                            w -= 1
                        energy_writer.write(energy_frame_bgr)
                        energy_frames.append(energy_frame_bgr.copy())
                except Exception as e:
                    print(f"    警告: 绘制能量图景失败: {e}")
            
            # 反归一化动作到原始空间
            action_orig = denormalize_action(action_norm, norm_params)
            
            # 执行动作（包括 IK 求解）- 测量时间
            execution_start = time.perf_counter()
            time_step = env.step(action_orig)
            execution_end = time.perf_counter()
            execution_time = execution_end - execution_start
            execution_times.append(execution_time)
            total_execution_time += execution_time
            total_steps += 1
            
            # 保存视频帧（在执行动作之后，显示新的状态）
            if video_writer is not None:
                try:
                    # 获取底层环境（HistoryWrapper -> GymWrapper -> TFPyEnvironment -> BlockPush）
                    base_env = env
                    while hasattr(base_env, '_env'):
                        base_env = base_env._env
                    
                    # 如果是 TFPyEnvironment，获取实际的 gym 环境
                    if hasattr(base_env, 'pyenv'):
                        base_env = base_env.pyenv.envs[0]
                    
                    # 调用 render 方法
                    if hasattr(base_env, 'render'):
                        frame = base_env.render(mode='rgb_array')
                        if frame is not None and frame.size > 0:
                            # 确保尺寸是偶数（避免 codec 问题）
                            h, w = frame.shape[:2]
                            if h % 2 != 0:
                                frame = frame[:h-1, :, :]
                                h -= 1
                            if w % 2 != 0:
                                frame = frame[:, :w-1, :]
                                w -= 1
                            
                            # 转换 RGB 到 BGR（opencv 使用 BGR）
                            if frame.shape[2] == 3:
                                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                video_writer.write(frame_bgr)
                                video_frames.append(frame_bgr.copy())
                except Exception as e:
                    # 如果渲染失败，跳过这一帧（不打印错误，避免输出过多）
                    pass
            
            step_count += 1
        
        # 获取最终状态
        # 需要获取底层 BlockPush 环境来判断是否成功（box 是否在目标区域内）
        try:
            # 获取底层环境（HistoryWrapper -> GymWrapper -> TFPyEnvironment -> BlockPush）
            base_env = env
            while hasattr(base_env, '_env'):
                base_env = base_env._env
            
            # 如果是 TFPyEnvironment，获取实际的 gym 环境
            if hasattr(base_env, 'pyenv'):
                base_env = base_env.pyenv.envs[0]
            
            # 检查是否成功（box 是否在目标区域内）
            # BlockPush 环境的 succeeded 属性会检查 goal_distance < goal_dist_tolerance
            if hasattr(base_env, 'succeeded'):
                success = bool(base_env.succeeded)
            else:
                # 如果没有 succeeded 属性，通过 goal_distance 判断
                if hasattr(base_env, 'goal_distance'):
                    goal_dist = float(base_env.goal_distance)
                    # 获取目标容差（默认 0.01，即 1cm）
                    goal_tolerance = getattr(base_env, 'goal_dist_tolerance', 0.01)
                    success = goal_dist < goal_tolerance
                else:
                    success = False
            
            # 获取最终目标距离
            if hasattr(base_env, 'goal_distance'):
                final_goal_distance = float(base_env.goal_distance)
        except Exception as e:
            # 如果获取失败，使用默认值
            success = False
            final_goal_distance = None
        
        # 关闭视频写入器
        if video_writer is not None:
            try:
                # 确保写入最后一帧（如果需要）
                if len(video_frames) > 0:
                    # 写入最后一帧
                    video_writer.write(video_frames[-1])
                
                video_writer.release()
                video_writer = None
                
                if ep_idx < video_episodes and video_path is not None:
                    # 检查文件是否存在且大小合理
                    if video_path.exists() and video_path.stat().st_size > 0:
                        print(f"  视频已保存: {video_path} ({len(video_frames)} 帧)")
                    else:
                        print(f"  警告: 视频文件可能未正确保存: {video_path}")
            except Exception as e:
                print(f"  警告: 关闭视频写入器时出错: {e}")
        
        # 关闭能量图景视频写入器
        if energy_writer is not None:
            try:
                # 确保写入最后一帧（如果需要）
                if len(energy_frames) > 0:
                    # 写入最后一帧
                    energy_writer.write(energy_frames[-1])
                
                energy_writer.release()
                energy_writer = None
                
                if ep_idx < energy_episodes and energy_path is not None:
                    # 检查文件是否存在且大小合理
                    if energy_path.exists() and energy_path.stat().st_size > 0:
                        print(f"  能量图景视频已保存: {energy_path} ({len(energy_frames)} 帧)")
                    else:
                        print(f"  警告: 能量图景视频文件可能未正确保存: {energy_path}")
            except Exception as e:
                print(f"  警告: 关闭能量图景视频写入器时出错: {e}")
        
        successes.append(success)
        if final_goal_distance is not None:
            final_goal_distances.append(final_goal_distance)
        episode_lengths.append(step_count)
    
    # 关闭环境
    env.close()
    
    # 计算统计信息
    success_rate = np.mean(successes) if successes else 0.0
    avg_final_goal_distance = np.mean(final_goal_distances) if final_goal_distances else None
    avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
    
    # 计算频率统计
    avg_inference_time = np.mean(inference_times) if inference_times else 0.0
    avg_execution_time = np.mean(execution_times) if execution_times else 0.0
    min_inference_time = np.min(inference_times) if inference_times else 0.0
    max_inference_time = np.max(inference_times) if inference_times else 0.0
    min_execution_time = np.min(execution_times) if execution_times else 0.0
    max_execution_time = np.max(execution_times) if execution_times else 0.0
    
    # 计算频率（Hz）
    inference_freq = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
    execution_freq = 1.0 / avg_execution_time if avg_execution_time > 0 else 0.0
    
    # 总时间
    total_time = total_inference_time + total_execution_time
    avg_total_time = total_time / total_steps if total_steps > 0 else 0.0
    overall_freq = 1.0 / avg_total_time if avg_total_time > 0 else 0.0
    
    results = {
        'success_rate': success_rate,
        'avg_final_goal_distance': avg_final_goal_distance,
        'avg_episode_length': avg_episode_length,
        'num_episodes': num_episodes,
        'num_successes': sum(successes),
        'successes': successes,
        'final_goal_distances': final_goal_distances,
        'episode_lengths': episode_lengths,
        'inference_stats': {
            'avg_time_ms': avg_inference_time * 1000,
            'min_time_ms': min_inference_time * 1000,
            'max_time_ms': max_inference_time * 1000,
            'frequency_hz': inference_freq,
            'total_time_s': total_inference_time
        },
        'execution_stats': {
            'avg_time_ms': avg_execution_time * 1000,
            'min_time_ms': min_execution_time * 1000,
            'max_time_ms': max_execution_time * 1000,
            'frequency_hz': execution_freq,
            'total_time_s': total_execution_time
        },
        'overall_stats': {
            'avg_total_time_ms': avg_total_time * 1000,
            'overall_frequency_hz': overall_freq,
            'total_steps': total_steps
        }
    }
    
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"成功率: {success_rate * 100:.2f}% ({sum(successes)}/{num_episodes})")
    if avg_final_goal_distance is not None:
        print(f"平均最终目标距离: {avg_final_goal_distance:.4f}")
    print(f"平均 Episode 长度: {avg_episode_length:.2f} 步")
    print()
    print("-" * 60)
    print("推理频率统计（EBM 推理）")
    print("-" * 60)
    print(f"平均推理时间: {avg_inference_time * 1000:.2f} ms")
    print(f"最小推理时间: {min_inference_time * 1000:.2f} ms")
    print(f"最大推理时间: {max_inference_time * 1000:.2f} ms")
    print(f"推理频率: {inference_freq:.2f} Hz")
    print(f"总推理时间: {total_inference_time:.2f} s")
    print()
    print("-" * 60)
    print("执行频率统计（IK 求解 + 环境步进）")
    print("-" * 60)
    print(f"平均执行时间: {avg_execution_time * 1000:.2f} ms")
    print(f"最小执行时间: {min_execution_time * 1000:.2f} ms")
    print(f"最大执行时间: {max_execution_time * 1000:.2f} ms")
    print(f"执行频率: {execution_freq:.2f} Hz")
    print(f"总执行时间: {total_execution_time:.2f} s")
    print()
    print("-" * 60)
    print("整体频率统计（推理 + 执行）")
    print("-" * 60)
    print(f"平均总时间: {avg_total_time * 1000:.2f} ms")
    print(f"整体频率: {overall_freq:.2f} Hz")
    print(f"总步数: {total_steps}")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    # 获取 IBC_ebm_dp 根目录
    IBC_ROOT = Path(__file__).parent.parent.parent
    
    # ============================================
    # 配置参数（硬编码在代码中，直接修改下面的变量即可）
    # ============================================
    
    # ===== 模型选择配置 =====
    # 方式1: 通过训练步数指定模型（推荐，例如: 15000 会查找 checkpoint_015000.pth）
    #       设置为 None 则使用其他方式
    CHECKPOINT_STEP = 15000  # 例如: 15000, 20000, 50000, None
    
    # 方式2: 直接指定模型文件路径（相对于 IBC_ebm_dp 根目录或绝对路径）
    #       设置为 None 则使用其他方式
    MODEL_PATH = None  # 例如: 'models/pushing_pixel/checkpoints/checkpoint_015000.pth'
    
    # 方式3: 如果上面两个都是 None，则自动查找最新的 checkpoint 或 final_model.pth
    
    # ===== 辅助函数 =====
    def find_model_by_step(step: int):
        """通过步数查找模型文件"""
        checkpoints_dir = IBC_ROOT / 'models' / 'pushing_pixel' / 'checkpoints'
        checkpoint_path = checkpoints_dir / f'checkpoint_{step:06d}.pth'
        if checkpoint_path.exists():
            return checkpoint_path
        return None
    
    def find_default_model():
        """查找默认模型文件（优先 final_model.pth，否则选择最新的 checkpoint）"""
        checkpoints_dir = IBC_ROOT / 'models' / 'pushing_pixel' / 'checkpoints'
        
        # 优先查找 final_model.pth
        final_model = checkpoints_dir / 'final_model.pth'
        if final_model.exists():
            return final_model
        
        # 否则查找最新的 checkpoint
        if checkpoints_dir.exists():
            checkpoint_files = sorted(checkpoints_dir.glob('checkpoint_*.pth'))
            if checkpoint_files:
                return checkpoint_files[-1]  # 返回最新的
        
        return None
    
    # ===== 确定模型路径 =====
    if CHECKPOINT_STEP is not None:
        # 方式1: 通过步数指定
        model_path = find_model_by_step(CHECKPOINT_STEP)
        if model_path is None:
            print(f"错误: 未找到步数为 {CHECKPOINT_STEP} 的模型文件")
            checkpoints_dir = IBC_ROOT / 'models' / 'pushing_pixel' / 'checkpoints'
            if checkpoints_dir.exists():
                checkpoint_files = sorted(checkpoints_dir.glob('checkpoint_*.pth'))
                if checkpoint_files:
                    print(f"  可用的 checkpoint 文件:")
                    for f in checkpoint_files:
                        # 尝试从文件名提取步数
                        try:
                            step_str = f.stem.split('_')[1]
                            step = int(step_str)
                            print(f"    {f.name} (步数: {step})")
                        except:
                            print(f"    {f.name}")
            sys.exit(1)
    elif MODEL_PATH is not None:
        # 方式2: 直接指定路径
        model_path = Path(MODEL_PATH)
        if not model_path.is_absolute():
            # 相对路径，尝试相对于 IBC_ROOT
            model_path = IBC_ROOT / model_path
        if not model_path.exists():
            print(f"错误: 模型文件不存在: {model_path}")
            sys.exit(1)
    else:
        # 方式3: 使用默认查找逻辑
        model_path = find_default_model()
        if model_path is None:
            print("错误: 未找到模型文件")
            checkpoints_dir = IBC_ROOT / 'models' / 'pushing_pixel' / 'checkpoints'
            if checkpoints_dir.exists():
                model_files = list(checkpoints_dir.glob('*.pth'))
                if model_files:
                    print(f"  找到以下模型文件:")
                    for f in sorted(model_files):
                        print(f"    {f}")
            sys.exit(1)
    
    # 评估参数（匹配 gin 配置）
    num_episodes = 100  # 匹配 gin: train_eval.eval_episodes = 20
    max_steps = 100
    num_action_samples = 2048  # 匹配 gin: IbcPolicy.num_action_samples = 2048
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    shared_memory = False
    seed = 0
    
    # 视频保存参数
    save_video = False
    video_output_dir = IBC_ROOT / 'plots' / 'pushing_pixel'  # 视频输出目录
    video_episodes = num_episodes  # 保存所有 episodes 的视频（设置为 num_episodes 以保存全部）
    
    # 能量图景视频保存参数
    save_energy_landscape = False  # 是否保存能量图景视频
    energy_episodes = num_episodes  # 保存所有 episodes 的能量图景（设置为 num_episodes 以保存全部）
    
    # ============================================
    # 执行评估
    # ============================================
    
    print(f"使用模型: {model_path}")
    print(f"视频输出目录: {video_output_dir}")
    print(f"保存能量图景: {save_energy_landscape}")
    print()
    
    # 加载模型
    model, checkpoint, norm_params = load_model(str(model_path), device=device)
    
    # 评估策略
    results = evaluate_policy(
        model=model,
        norm_params=norm_params,
        num_episodes=num_episodes,
        max_steps=max_steps,
        num_action_samples=num_action_samples,
        device=device,
        shared_memory=shared_memory,
        seed=seed,
        save_video=save_video,
        video_output_dir=str(video_output_dir),
        video_episodes=video_episodes,
        save_energy_landscape=save_energy_landscape,
        energy_episodes=energy_episodes
    )
    
    # 保存结果
    output_path = model_path.parent / 'eval_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n评估结果已保存到: {output_path}")

