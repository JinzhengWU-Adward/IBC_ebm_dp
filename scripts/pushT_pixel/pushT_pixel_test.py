"""
使用训练好的 PushT EBM 模型进行测试和可视化
基于 diffusion_policy 的 PushT 环境
参考 pushing_pixel_test.py 的测试框架
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import sys
import zarr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2


# 添加项目路径
IBC_ROOT = Path(__file__).parent.parent.parent  # IBC_ebm_dp
sys.path.insert(0, str(IBC_ROOT))

from core.models import PixelEBM
from core.optimizers import ULASampler

# 导入 PushT 环境（使用本地实现）
try:
    from env._2d.pushT import PushTImageEnv
    PUSHT_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入 PushTImageEnv: {e}")
    print("将跳过环境测试")
    PUSHT_AVAILABLE = False


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """
    加载训练好的模型
    """
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取模型参数
    action_dim = checkpoint['action_dim']
    obs_seq_len = checkpoint['obs_seq_len']
    target_height = checkpoint['target_height']
    target_width = checkpoint['target_width']
    image_channels = checkpoint['image_channels']
    value_width = checkpoint.get('value_width', 1024)
    value_num_blocks = checkpoint.get('value_num_blocks', 1)
    norm_params = checkpoint['norm_params']
    
    print(f"模型参数:")
    print(f"  动作维度: {action_dim}")
    print(f"  观测序列长度: {obs_seq_len}")
    print(f"  图像大小: {target_height}x{target_width}")
    print(f"  图像通道数: {image_channels}")
    print()
    
    # 创建模型
    model = PixelEBM(
        image_channels=image_channels,
        action_dim=action_dim,
        target_height=target_height,
        target_width=target_width,
        value_width=value_width,
        value_num_blocks=value_num_blocks
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载成功")
    print()
    
    return model, norm_params, obs_seq_len, action_dim


def predict_action(
    model: PixelEBM,
    images: torch.Tensor,  # (seq_len, H, W, 3)
    ula_sampler: ULASampler,
    num_samples: int = 512,
    device: str = 'cuda'
) -> np.ndarray:
    """
    使用 EBM 预测动作
    """
    model.eval()
    
    with torch.no_grad():
        # 添加 batch 维度
        images_batch = images.unsqueeze(0).to(device)  # (1, seq_len, H, W, 3)
        
        # 编码图像
        obs_encoding = model.encode(images_batch)  # (1, 256)
        
        # 初始化动作样本
        action_bounds = ula_sampler.bounds
        neg_min = float(action_bounds[0, 0])
        neg_max = float(action_bounds[1, 0])
        action_dim = model.action_dim
        
        init_samples = torch.rand(
            1, num_samples, action_dim,
            device=device
        ) * (neg_max - neg_min) + neg_min
    
    # ULA 采样（需要梯度）
    with torch.enable_grad():
        sampled_actions, _ = ula_sampler.sample(
            x=images_batch,
            ebm=model,
            num_samples=num_samples,
            init_samples=init_samples,
            return_trajectory=False,
            obs_encoding=obs_encoding.detach()
        )  # (1, num_samples, action_dim)
    
    # 计算能量并选择最低能量的动作
    with torch.no_grad():
        energies = model(
            images=None,
            actions=sampled_actions,
            obs_encoding=obs_encoding
        )  # (1, num_samples)
        
        # 选择最低能量的动作
        best_idx = torch.argmin(energies, dim=1)
        best_action = sampled_actions[0, best_idx].cpu().numpy()  # (action_dim,)
    
    return best_action


def denormalize_action(action: np.ndarray, norm_params: dict) -> np.ndarray:
    """
    反归一化动作（从 [-1, 1] 到原始范围）
    """
    action_min = np.array(norm_params['action_min'])
    action_max = np.array(norm_params['action_max'])
    action_range = np.array(norm_params['action_range'])
    
    # 从 [-1, 1] 反归一化到原始范围
    denorm_action = (action + 1.0) / 2.0 * action_range + action_min
    return denorm_action


def test_on_dataset(
    model: PixelEBM,
    zarr_path: str,
    norm_params: dict,
    obs_seq_len: int,
    ula_sampler: ULASampler,
    num_test_samples: int = 100,
    use_delta_action: bool = True,  # 是否使用 action_delta（True）或 action（False）
    seed: int = None,  # 随机种子（None 表示不设置）
    device: str = 'cuda'
):
    """
    在数据集上测试模型
    """
    print("=" * 60)
    print("在数据集上测试模型")
    print("=" * 60)
    
    # 设置随机种子（如果提供）
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"设置随机种子: {seed}")
    
    # 加载数据
    root = zarr.open(str(zarr_path), mode='r')
    images = root['data']['img'][:]  # (N, 96, 96, 3) uint8
    
    # 根据 use_delta_action 选择使用 action_delta 或 action
    if use_delta_action:
        try:
            actions = root['data']['action_delta'][:]  # (N, 2) float32
            print("使用 action_delta 进行测试")
        except KeyError:
            print("警告: 未找到 action_delta，回退到 action")
            actions = root['data']['action'][:]  # (N, 2) float32
            use_delta_action = False
    else:
        actions = root['data']['action'][:]  # (N, 2) float32
        print("使用 action（绝对位置）进行测试")
    
    episode_ends = root['meta']['episode_ends'][:]
    
    # 随机选择测试样本
    num_samples = min(num_test_samples, len(images) - obs_seq_len)
    test_indices = np.random.choice(len(images) - obs_seq_len, num_samples, replace=False)
    
    mse_errors = []
    mae_errors = []
    
    print(f"测试 {num_samples} 个样本...")
    for idx in tqdm(test_indices):
        # 构建图像序列
        img_seq = []
        for i in range(obs_seq_len):
            img = images[idx + i].astype(np.float32) / 255.0
            img_seq.append(img)
        img_seq = np.stack(img_seq, axis=0)  # (seq_len, H, W, 3)
        img_seq_tensor = torch.FloatTensor(img_seq)
        
        # 预测动作（归一化的）
        pred_action_norm = predict_action(
            model, img_seq_tensor, ula_sampler, num_samples=512, device=device
        )
        
        # 反归一化
        pred_action = denormalize_action(pred_action_norm, norm_params)
        
        # 真实动作
        gt_action = actions[idx + obs_seq_len - 1]
        
        # 计算误差
        mse = np.mean((pred_action - gt_action) ** 2)
        mae = np.mean(np.abs(pred_action - gt_action))
        
        mse_errors.append(mse)
        mae_errors.append(mae)
    
    # 统计结果
    avg_mse = np.mean(mse_errors)
    avg_mae = np.mean(mae_errors)
    std_mse = np.std(mse_errors)
    std_mae = np.std(mae_errors)
    
    print()
    print("测试结果:")
    print(f"  平均 MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    print(f"  平均 MAE: {avg_mae:.6f} ± {std_mae:.6f}")
    print()


def test_on_environment(
    model: PixelEBM,
    norm_params: dict,
    obs_seq_len: int,
    ula_sampler: ULASampler,
    num_episodes: int = 10,
    max_steps: int = 200,
    render: bool = True,
    save_video: bool = True,
    output_dir: str = None,
    use_delta_action: bool = True,  # 是否使用 action_delta（True）或 action（False）
    seed: int = None,  # 随机种子（None 表示不设置）
    device: str = 'cuda'
):
    """
    在 PushT 环境中测试模型
    """
    if not PUSHT_AVAILABLE:
        print("跳过环境测试（PushTImageEnv 不可用）")
        return
    
    print("=" * 60)
    print("在 PushT 环境中测试模型")
    print("=" * 60)
    
    # 设置随机种子（如果提供）
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"设置随机种子: {seed}")
    
    # 创建环境
    env = PushTImageEnv(render_size=96)
    
    # 设置环境的随机种子（如果提供）
    if seed is not None:
        env.seed(seed)
    
    success_count = 0
    
    for episode in range(num_episodes):
        print(f"\n回合 {episode + 1}/{num_episodes}")
        
        # 如果设置了 seed，为每个 episode 设置不同的 seed（基于基础 seed + episode 索引）
        # 这样可以确保每个 episode 的初始状态是可复现的，但不同 episode 之间是不同的
        if seed is not None:
            episode_seed = seed + episode
            env.seed(episode_seed)
            # 重新设置 numpy 和 torch 的随机种子（确保每个 episode 的随机性一致）
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(episode_seed)
                torch.cuda.manual_seed_all(episode_seed)
        
        # 重置环境
        obs = env.reset()
        # PushTImageEnv 返回的 image 形状是 (3, H, W)，需要转换为 (H, W, 3)
        img_obs = np.moveaxis(obs['image'], 0, -1)  # (3, H, W) -> (H, W, 3)
        obs_history = [img_obs]  # 保存观测历史
        
        frames = []
        episode_success = False  # 记录回合是否成功
        
        # ⚠️ 根据 use_delta_action 选择不同的处理逻辑
        if use_delta_action:
            # 使用 action_delta（相对位移）模式：
            # - 所有动作的 delta 都是相对位移（包括第一个动作）
            # - 第一个动作的 delta = action[1] - action[0]（相对位移）
            # - 中间动作的 delta = action[i] - action[i-1]（相对位移）
            # 
            # 在测试时，我们需要：
            # 1. 先执行一个动作来获取初始智能体位置（从 info 中获取）
            # 2. 重置环境以重新开始
            # 3. 使用首帧图像堆叠来预测第一个动作 delta
            # 4. 将第一个 delta 累积到初始位置上
            # 5. 后续动作继续累积 delta
            
            # 先执行一个中心位置的动作来获取初始智能体位置
            initial_guess = np.array([256.0, 256.0])  # 窗口中心作为初始猜测
            obs_temp, _, _, info_temp = env.step(initial_guess)
            initial_agent_pos = info_temp.get('pos_agent', initial_guess.copy())
            
            # 重置环境以重新开始（因为刚才执行了一步来获取位置）
            obs = env.reset()
            img_obs = np.moveaxis(obs['image'], 0, -1)  # (3, H, W) -> (H, W, 3)
            obs_history = [img_obs]  # 保存观测历史
            
            # 当前目标位置从初始智能体位置开始
            # 第一个动作的 delta 会累积到这个位置上
            current_target_pos = initial_agent_pos.copy()
        else:
            # 使用 action（绝对位置）模式：
            # - 动作是绝对位置，直接使用
            # - 不需要累积，直接使用预测的绝对位置
            obs = env.reset()
            img_obs = np.moveaxis(obs['image'], 0, -1)  # (3, H, W) -> (H, W, 3)
            obs_history = [img_obs]  # 保存观测历史
            
            # 不需要初始化 current_target_pos，因为每个动作都是绝对位置
            current_target_pos = None
        
        for step in range(max_steps):
            # 构建观测序列
            if len(obs_history) < obs_seq_len:
                # 不足时重复第一帧（首帧堆叠）
                img_seq = [obs_history[0]] * (obs_seq_len - len(obs_history)) + obs_history
            else:
                img_seq = obs_history[-obs_seq_len:]
            
            # 转换为张量 (seq_len, H, W, 3)
            # 注意：图像已经是 [0, 1] 范围（从 env 返回时已归一化）
            img_seq_np = np.stack(img_seq, axis=0).astype(np.float32)
            img_seq_tensor = torch.FloatTensor(img_seq_np)
            
            # 预测动作（模型输出的是归一化后的动作）
            pred_action_norm = predict_action(
                model, img_seq_tensor, ula_sampler, num_samples=512, device=device
            )
            # 反归一化到原始范围
            pred_action = denormalize_action(pred_action_norm, norm_params)
            
            # ⚠️ 根据 use_delta_action 选择不同的处理逻辑
            if use_delta_action:
                # 使用 action_delta（相对位移）模式：
                # 所有动作的 delta 都是相对位移，需要从当前目标位置累积
                # 第一步：current_target_pos = initial_agent_pos + first_delta
                # 后续步骤：current_target_pos = current_target_pos + delta
                current_target_pos = current_target_pos + pred_action
                # 确保在有效范围内 [0, 512]
                current_target_pos = np.clip(current_target_pos, 0.0, 512.0)
            else:
                # 使用 action（绝对位置）模式：
                # 动作是绝对位置，直接使用
                current_target_pos = pred_action.copy()
                # 确保在有效范围内 [0, 512]
                current_target_pos = np.clip(current_target_pos, 0.0, 512.0)
            
            # 执行动作（使用绝对位置）
            obs, reward, done, info = env.step(current_target_pos)
            
            # 保存观测（转换格式）
            img_obs = np.moveaxis(obs['image'], 0, -1)  # (3, H, W) -> (H, W, 3)
            obs_history.append(img_obs)
            
            # 渲染
            if render:
                frame = env.render(mode='rgb_array')
                if save_video:
                    frames.append(frame)
            
            # 如果任务完成（done=True），说明方块覆盖率达到了 0.95 以上
            if done:
                episode_success = True
                # 打印覆盖率信息
                coverage = info.get('coverage', 0.0)
                intersection_area = info.get('intersection_area', 0.0)
                goal_area = info.get('goal_area', 0.0)
                block_area = info.get('block_area', 0.0)
                block_pos = info.get('block_position', np.array([0, 0]))
                goal_pos = info.get('goal_position', np.array([0, 0]))
                block_angle = info.get('block_angle', 0.0)
                goal_angle = info.get('goal_angle', 0.0)
                print(f"  任务完成！步数: {step + 1}")
                print(f"  覆盖率: {coverage:.4f} ({coverage*100:.2f}%)")
                print(f"  交集面积: {intersection_area:.2f}, 目标面积: {goal_area:.2f}, 方块面积: {block_area:.2f}")
                print(f"  方块位置: ({block_pos[0]:.1f}, {block_pos[1]:.1f}), 角度: {block_angle:.3f}")
                print(f"  目标位置: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f}), 角度: {goal_angle:.3f}")
                break
        
        # 如果任务未完成，在回合结束时也打印覆盖率信息
        if not episode_success:
            # 获取最后一步的信息
            coverage = info.get('coverage', 0.0)
            intersection_area = info.get('intersection_area', 0.0)
            goal_area = info.get('goal_area', 0.0)
            block_area = info.get('block_area', 0.0)
            block_pos = info.get('block_position', np.array([0, 0]))
            goal_pos = info.get('goal_position', np.array([0, 0]))
            block_angle = info.get('block_angle', 0.0)
            goal_angle = info.get('goal_angle', 0.0)
            print(f"  任务未完成，步数: {step + 1}")
            print(f"  最终覆盖率: {coverage:.4f} ({coverage*100:.2f}%)")
            print(f"  交集面积: {intersection_area:.2f}, 目标面积: {goal_area:.2f}, 方块面积: {block_area:.2f}")
            print(f"  方块位置: ({block_pos[0]:.1f}, {block_pos[1]:.1f}), 角度: {block_angle:.3f}")
            print(f"  目标位置: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f}), 角度: {goal_angle:.3f}")
        
        # 判断是否成功：使用环境的 done 标志（更准确）
        # PushT 环境中，done=True 表示 coverage > 0.95（方块与目标区域重叠超过 95%）
        if episode_success:
            success_count += 1
        
        # 保存视频
        if save_video and output_dir is not None and len(frames) > 0:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            video_path = output_path / f'episode_{episode + 1}.mp4'
            
            # 使用 opencv 保存视频
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (width, height))
            
            for frame in frames:
                # OpenCV 使用 BGR 格式
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"  视频已保存: {video_path}")
    
    env.close()
    
    # 统计结果
    success_rate = success_count / num_episodes
    
    print()
    print("=" * 60)
    print("环境测试结果:")
    print(f"  成功率: {success_rate:.2%} ({success_count}/{num_episodes})")
    print("=" * 60)


def visualize_energy_landscape(
    model: PixelEBM,
    zarr_path: str,
    norm_params: dict,
    obs_seq_len: int,
    num_samples: int = 5,
    output_dir: str = None,
    device: str = 'cuda'
):
    """
    可视化能量地形
    """
    print("=" * 60)
    print("可视化能量地形")
    print("=" * 60)
    
    # 加载数据
    root = zarr.open(str(zarr_path), mode='r')
    images = root['data']['img'][:]
    # ⚠️ 修复：使用 action_delta 而不是 action
    try:
        actions = root['data']['action_delta'][:]
    except KeyError:
        actions = root['data']['action'][:]
    
    # 随机选择样本
    sample_indices = np.random.choice(len(images) - obs_seq_len, num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        print(f"\n样本 {i + 1}/{num_samples}")
        
        # 构建图像序列
        img_seq = []
        for j in range(obs_seq_len):
            img = images[idx + j].astype(np.float32) / 255.0
            img_seq.append(img)
        img_seq = np.stack(img_seq, axis=0)
        img_seq_tensor = torch.FloatTensor(img_seq).unsqueeze(0).to(device)
        
        # 真实动作
        gt_action = actions[idx + obs_seq_len - 1]
        
        # 归一化真实动作
        action_min = np.array(norm_params['action_min'])
        action_max = np.array(norm_params['action_max'])
        action_range = np.array(norm_params['action_range'])
        gt_action_norm = 2.0 * (gt_action - action_min) / action_range - 1.0
        
        # 创建动作网格
        grid_size = 50
        action_0 = np.linspace(-1.1, 1.1, grid_size)
        action_1 = np.linspace(-1.1, 1.1, grid_size)
        A0, A1 = np.meshgrid(action_0, action_1)
        
        # 计算能量
        energies = np.zeros_like(A0)
        
        with torch.no_grad():
            obs_encoding = model.encode(img_seq_tensor)
            
            for row in tqdm(range(grid_size), desc="计算能量"):
                for col in range(grid_size):
                    action = torch.FloatTensor([[A0[row, col], A1[row, col]]]).to(device)
                    action_expanded = action.unsqueeze(1)  # (1, 1, 2)
                    energy = model(
                        images=None,
                        actions=action_expanded,
                        obs_encoding=obs_encoding
                    )
                    energies[row, col] = energy.item()
        
        # 绘制能量地形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制等高线
        contour = ax.contourf(A0, A1, energies, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='Energy')
        
        # 标记真实动作
        ax.plot(gt_action_norm[0], gt_action_norm[1], 'r*', markersize=20, label='Ground Truth')
        
        ax.set_xlabel('Action Dimension 0 (normalized)')
        ax.set_ylabel('Action Dimension 1 (normalized)')
        ax.set_title(f'Energy Landscape (Sample {i + 1})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存图像
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fig_path = output_path / f'energy_landscape_{i + 1}.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"保存图像: {fig_path}")
        
        plt.close(fig)
    
    print()
    print("能量地形可视化完成")


if __name__ == '__main__':
    # 获取 IBC_ebm_dp 根目录
    IBC_ROOT = Path(__file__).parent.parent.parent
    
    # ============================================
    # 配置参数（硬编码在代码中，直接修改下面的变量即可）
    # ============================================
    
    # ===== 动作类型选择（必须在确定模型路径之前设置）=====
    # True: 使用 action_delta（相对位移），适用于使用 delta 动作训练的模型
    #       模型目录: models/pusht_pixel
    # False: 使用 action（绝对位置），适用于使用原始动作训练的模型
    #        模型目录: models/pusht_pixel_abs
    use_delta_action = True  # 根据训练的模型选择 True 或 False
    
    # ===== 模型选择配置 =====
    # 方式1: 通过训练步数指定模型（推荐，例如: 15000 会查找 checkpoint_015000.pth）
    #       设置为 None 则使用其他方式
    CHECKPOINT_STEP = 15000  # 例如: 15000, 20000, 50000, None
    
    # 方式2: 直接指定模型文件路径（相对于 IBC_ebm_dp 根目录或绝对路径）
    #       设置为 None 则使用其他方式
    MODEL_PATH = None  # 例如: 'models/pusht_pixel/checkpoints/checkpoint_015000.pth'
    
    # 方式3: 如果上面两个都是 None，则自动查找最新的 checkpoint 或 final_model.pth
    
    # ===== 辅助函数 =====
    def find_model_by_step(step: int, use_delta_action: bool = True):
        """通过步数查找模型文件"""
        # 根据 use_delta_action 选择不同的模型目录
        if use_delta_action:
            checkpoints_dir = IBC_ROOT / 'models' / 'pusht_pixel' / 'checkpoints'
        else:
            checkpoints_dir = IBC_ROOT / 'models' / 'pusht_pixel_abs' / 'checkpoints'
        
        checkpoint_path = checkpoints_dir / f'checkpoint_{step:06d}.pth'
        if checkpoint_path.exists():
            return checkpoint_path
        return None
    
    def find_default_model(use_delta_action: bool = True):
        """查找默认模型文件（优先 final_model.pth，否则选择最新的 checkpoint）"""
        # 根据 use_delta_action 选择不同的模型目录
        if use_delta_action:
            checkpoints_dir = IBC_ROOT / 'models' / 'pusht_pixel' / 'checkpoints'
        else:
            checkpoints_dir = IBC_ROOT / 'models' / 'pusht_pixel_abs' / 'checkpoints'
        
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
        model_path = find_model_by_step(CHECKPOINT_STEP, use_delta_action)
        if model_path is None:
            print(f"错误: 未找到步数为 {CHECKPOINT_STEP} 的模型文件")
            # 根据 use_delta_action 选择目录
            if use_delta_action:
                checkpoints_dir = IBC_ROOT / 'models' / 'pusht_pixel' / 'checkpoints'
            else:
                checkpoints_dir = IBC_ROOT / 'models' / 'pusht_pixel_abs' / 'checkpoints'
            
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
        model_path = find_default_model(use_delta_action)
        if model_path is None:
            print("错误: 未找到模型文件")
            # 根据 use_delta_action 选择目录
            if use_delta_action:
                checkpoints_dir = IBC_ROOT / 'models' / 'pusht_pixel' / 'checkpoints'
            else:
                checkpoints_dir = IBC_ROOT / 'models' / 'pusht_pixel_abs' / 'checkpoints'
            
            if checkpoints_dir.exists():
                model_files = list(checkpoints_dir.glob('*.pth'))
                if model_files:
                    print(f"  找到以下模型文件:")
                    for f in sorted(model_files):
                        print(f"    {f}")
            sys.exit(1)
    
    # ===== 测试配置 =====
    # 数据路径
    zarr_path = IBC_ROOT / 'data' / 'pusht_cchi_v7_replay.zarr'
    
    # 输出目录（根据动作类型选择不同的输出目录）
    if use_delta_action:
        output_dir = IBC_ROOT / 'plots' / 'pusht_pixel'
    else:
        output_dir = IBC_ROOT / 'plots' / 'pusht_pixel_abs'
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试选项
    test_dataset = True  # 是否在数据集上测试
    test_env = True  # 是否在环境中测试
    visualize_energy = False  # 是否可视化能量地形
    
    # ⚠️ 随机种子设置
    # 设置 seed 可以确保每次测试的随机性一致（可复现）
    # None 表示不设置种子（每次运行结果不同）
    # 设置为整数（如 42）可以确保每次运行结果相同
    seed = 1  # 例如: 42, 123, None
    
    # 数据集测试参数
    num_test_samples = 10  # 数据集测试样本数
    
    # 环境测试参数
    num_episodes = 10  # 环境测试回合数
    max_steps = 800  # 每回合最大步数
    save_video = True  # 是否保存视频
    
    # 能量地形可视化参数
    energy_num_samples = 5  # 能量地形可视化样本数
    
    # ============================================
    # 执行测试
    # ============================================
    
    print(f"使用模型: {model_path}")
    print(f"数据路径: {zarr_path}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")
    print()
    
    # 加载模型
    model, norm_params, obs_seq_len, action_dim = load_model(str(model_path), device=device)
    
    # 创建 ULA 采样器
    uniform_boundary_buffer = 0.05
    norm_min, norm_max = -1.0, 1.0
    expanded_min = norm_min - uniform_boundary_buffer * (norm_max - norm_min)
    expanded_max = norm_max + uniform_boundary_buffer * (norm_max - norm_min)
    action_bounds = np.array([[expanded_min, expanded_min],
                              [expanded_max, expanded_max]], dtype=np.float32)
    
    ula_sampler = ULASampler(
        bounds=action_bounds,
        step_size=0.1,
        num_steps=20,  # 匹配训练时的配置
        noise_scale=1.0,
        step_size_final=1e-5,
        step_size_power=2.0,
        delta_action_clip=0.1,
        device=device
    )
    
    # 在数据集上测试
    if test_dataset:
        test_on_dataset(
            model, str(zarr_path), norm_params, obs_seq_len,
            ula_sampler, num_test_samples, use_delta_action, seed, device
        )
    
    # 在环境中测试
    if test_env:
        test_on_environment(
            model, norm_params, obs_seq_len, ula_sampler,
            num_episodes, max_steps,
            render=True, save_video=save_video,
            output_dir=str(output_dir), use_delta_action=use_delta_action, seed=seed, device=device
        )
    
    # 可视化能量地形
    if visualize_energy:
        visualize_energy_landscape(
            model, str(zarr_path), norm_params, obs_seq_len,
            num_samples=energy_num_samples, output_dir=str(output_dir), device=device
        )
    
    print()
    print("测试完成！")

