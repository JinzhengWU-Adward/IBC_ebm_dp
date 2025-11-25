"""
直接使用 IBC Particle 环境和 Oracle 策略生成数据
参考 IBC 的数据生成方式，适配 PyTorch 使用
"""
import numpy as np
import json
import os
import copy
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import gin

# 添加 IBC 路径以便导入
# IBC 的目录结构：generative_models/ibc/
# 需要将包含 ibc 的目录添加到 PYTHONPATH
ibc_parent_dir = Path(__file__).parent.parent.parent.parent.parent
if str(ibc_parent_dir) not in sys.path:
    sys.path.insert(0, str(ibc_parent_dir))

# 导入 IBC 模块
from ibc.environments.particle import particle
from ibc.environments.particle import particle_oracles
from tf_agents.environments import suite_gym

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_particle_episode(env, oracle, max_steps=50):
    """
    生成一个 particle episode
    
    Args:
        env: 包装后的 PyEnvironment 实例（通过 suite_gym.wrap_env）
        oracle: ParticleOracle 策略
        max_steps: 最大步数
    
    Returns:
        episode_data: 包含轨迹信息的字典
    """
    # 重置环境（返回 TimeStep 对象）
    time_step = env.reset()
    oracle.reset()
    
    positions = []
    velocities = []  # ← 添加速度列表
    actions = []
    observations = []
    
    # 记录初始目标
    first_goal = None
    second_goal = None
    
    step_count = 0
    
    while not time_step.is_last() and step_count < max_steps:
        # 记录观测
        obs = time_step.observation
        observations.append(copy.deepcopy(obs))
        
        # 记录位置和速度
        pos_agent = obs['pos_agent']
        vel_agent = obs['vel_agent']  # ← 记录速度
        positions.append(pos_agent.copy())
        velocities.append(vel_agent.copy())  # ← 保存速度
        
        # 记录目标（只在第一步）
        if step_count == 0:
            first_goal = obs['pos_first_goal'].copy()
            second_goal = obs['pos_second_goal'].copy()
        
        # Oracle 策略选择动作
        policy_step = oracle.action(time_step)
        action = policy_step.action
        actions.append(action.copy())
        
        # 执行动作（包装后的环境返回 TimeStep）
        time_step = env.step(action)
        
        step_count += 1
    
    # 添加最后一步的位置和速度
    if not time_step.is_last() and len(positions) > 0:
        final_obs = time_step.observation
        positions.append(final_obs['pos_agent'].copy())
        velocities.append(final_obs['vel_agent'].copy())  # ← 保存最后一步的速度
    
    return {
        'positions': np.array(positions, dtype=np.float32),
        'velocities': np.array(velocities, dtype=np.float32),  # ← 返回速度
        'actions': np.array(actions, dtype=np.float32),
        'observations': observations,
        'first_goal': first_goal,
        'second_goal': second_goal,
        'num_steps': len(positions)
    }


def create_trajectory_image(episode_data, image_size=64):
    """
    创建轨迹可视化图像
    
    Args:
        episode_data: 包含 positions, first_goal, second_goal 等的字典
        image_size: 图像尺寸
    
    Returns:
        PIL Image 对象
    """
    positions = episode_data['positions']
    first_goal = episode_data['first_goal']
    second_goal = episode_data['second_goal']
    start_pos = positions[0] if len(positions) > 0 else np.array([0.5, 0.5])
    
    # Particle 环境使用 [0, 1] 范围
    # 计算坐标范围（添加边距）
    all_points = np.vstack([positions, start_pos.reshape(1, -1), 
                           first_goal.reshape(1, -1),
                           second_goal.reshape(1, -1)])
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    
    # 添加边距
    margin = 0.1
    range_coords = max_coords - min_coords
    if range_coords[0] < 0.01:
        range_coords[0] = 0.1
    if range_coords[1] < 0.01:
        range_coords[1] = 0.1
    min_coords = min_coords - margin * range_coords
    max_coords = max_coords + margin * range_coords
    
    # 限制在 [0, 1] 范围内
    min_coords = np.clip(min_coords, 0.0, 1.0)
    max_coords = np.clip(max_coords, 0.0, 1.0)
    
    # 创建图像
    img = Image.new('L', (image_size, image_size), 255)  # 白色背景
    draw = ImageDraw.Draw(img)
    
    def coord_to_pixel(coord):
        """将坐标转换为像素位置"""
        x = int((coord[0] - min_coords[0]) / (max_coords[0] - min_coords[0] + 1e-6) * image_size)
        y = int((coord[1] - min_coords[1]) / (max_coords[1] - min_coords[1] + 1e-6) * image_size)
        x = np.clip(x, 0, image_size - 1)
        y = np.clip(y, 0, image_size - 1)
        return x, y
    
    # 绘制轨迹
    if len(positions) > 1:
        pixel_positions = [coord_to_pixel(pos) for pos in positions]
        for i in range(len(pixel_positions) - 1):
            draw.line([pixel_positions[i], pixel_positions[i+1]], 
                     fill=128, width=1)  # 灰色轨迹
    
    # 绘制起点（蓝色方块）
    start_pixel = coord_to_pixel(start_pos)
    draw.rectangle([start_pixel[0]-2, start_pixel[1]-2, 
                   start_pixel[0]+2, start_pixel[1]+2], 
                  fill=0, outline=0)
    
    # 绘制第一个目标（绿色圆圈）
    first_goal_pixel = coord_to_pixel(first_goal)
    draw.ellipse([first_goal_pixel[0]-3, first_goal_pixel[1]-3,
                 first_goal_pixel[0]+3, first_goal_pixel[1]+3],
                fill=64, outline=0)
    
    # 绘制第二个目标（红色星形）- 这是最终目标
    target_pixel = coord_to_pixel(second_goal)
    # 绘制星形（简化版）
    size = 4
    points = []
    for i in range(5):
        angle = i * 2 * np.pi / 5 - np.pi / 2
        x = target_pixel[0] + size * np.cos(angle)
        y = target_pixel[1] + size * np.sin(angle)
        points.append((int(x), int(y)))
    draw.polygon(points, fill=0, outline=0)
    
    return img


def generate_particle_dataset(
    output_dir,
    num_episodes=1000,
    n_dim=2,
    n_steps=50,
    image_size=64,
    seed=None
):
    """
    生成 Particle 数据集
    
    Args:
        output_dir: 输出目录
        num_episodes: 生成的 episode 数量
        n_dim: 维度（2D）
        n_steps: 每个 episode 的最大步数
        image_size: 图像尺寸
        seed: 随机种子
    """
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'traj').mkdir(exist_ok=True)
    (output_path / 'pic').mkdir(exist_ok=True)
    
    print("=" * 60)
    print("生成 IBC Particle 数据集")
    print("=" * 60)
    print(f"输出目录: {output_path}")
    print(f"Episode 数量: {num_episodes}")
    print(f"维度: {n_dim}D")
    print(f"最大步数: {n_steps}")
    print()
    
    # 创建环境（使用 IBC 的配置）
    gym_env = particle.ParticleEnv(
        n_steps=n_steps,
        n_dim=n_dim,
        hide_velocity=False,
        seed=seed,
        dt=0.005,
        repeat_actions=10,
        k_p=10.0,
        k_v=5.0,
        goal_distance=0.05
    )
    
    # 使用 suite_gym 包装环境，使其具有 time_step_spec() 方法
    env = suite_gym.wrap_env(gym_env)
    
    # 创建 Oracle 策略（需要包装后的环境）
    oracle = particle_oracles.ParticleOracle(
        env,
        wait_at_first_goal=1,
        multimodal=False,
        goal_threshold=0.01
    )
    
    # 生成数据
    total_samples = 0
    
    for ep_idx in tqdm(range(num_episodes), desc="生成 episodes"):
        try:
            # 生成 episode
            episode_data = generate_particle_episode(env, oracle, max_steps=n_steps)
            
            if episode_data['num_steps'] < 2:
                continue
            
            # 生成时间戳
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S_%f", time.localtime())
            timestamp = f"{timestamp}_{total_samples:06d}"
            
            # 创建图像
            img = create_trajectory_image(episode_data, image_size)
            img_path = output_path / 'pic' / f'target_{timestamp}.png'
            img.save(img_path)
            
            # 创建 JSON 文件
            json_data = {
                'sample_id': total_samples,
                'timestamp': timestamp,
                'start_position': episode_data['positions'][0].tolist(),
                'target_position': episode_data['second_goal'].tolist(),
                'first_goal_position': episode_data['first_goal'].tolist(),
                'second_goal_position': episode_data['second_goal'].tolist(),
                'trajectory': {
                    'num_steps': episode_data['num_steps'],
                    'positions': episode_data['positions'].tolist(),
                    'velocities': episode_data['velocities'].tolist()  # ← 保存速度
                },
                'actions': episode_data['actions'].tolist(),
                'config': {
                    'ndof': n_dim,
                    'x_range': [0.0, 1.0],  # Particle 环境使用 [0, 1]
                    'y_range': [0.0, 1.0],
                    'n_steps': n_steps
                }
            }
            
            json_path = output_path / 'traj' / f'traj_{timestamp}.json'
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            total_samples += 1
            
        except Exception as e:
            print(f"警告: 生成 episode {ep_idx} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n生成完成！共生成 {total_samples} 个样本")
    print(f"输出目录: {output_path}")


if __name__ == '__main__':
    # 输出路径
    output_dir = Path(__file__).parent.parent.parent.parent / 'data' / '_2d' / 'particle'
    
    # 生成参数（匹配 IBC 配置）
    # 注意：IBC 官方使用 seed=None（随机种子）以确保数据多样性
    # 如果使用固定种子（如 seed=0），每次生成的数据会完全相同
    generate_particle_dataset(
        output_dir=str(output_dir),
        num_episodes=2000,  # 匹配 IBC 官方：200 episodes × 10 replicas = 2000
        n_dim=2,
        n_steps=50,  # IBC 默认 50 步
        image_size=64,
        seed=None  # 使用随机种子，与 IBC 官方一致（确保数据多样性）
    )
