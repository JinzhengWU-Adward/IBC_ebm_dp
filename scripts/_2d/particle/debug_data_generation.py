"""
调试数据生成 - 检查轨迹是否包含完整路径
"""
import numpy as np
import json
from pathlib import Path
import sys

# 添加 IBC 路径
ibc_parent_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(ibc_parent_dir))

from ibc.environments.particle import particle
from ibc.environments.particle import particle_oracles
from tf_agents.environments import suite_gym

def test_single_episode():
    """测试单个 episode 的数据生成"""
    print("=" * 60)
    print("测试 Particle 数据生成")
    print("=" * 60)
    
    # 创建环境
    gym_env = particle.ParticleEnv(
        n_steps=50,
        n_dim=2,
        hide_velocity=False,
        seed=42,  # 使用固定种子便于调试
        dt=0.005,
        repeat_actions=10,
        k_p=10.0,
        k_v=5.0,
        goal_distance=0.05
    )
    
    # 包装环境
    env = suite_gym.wrap_env(gym_env)
    
    # 创建 Oracle 策略
    oracle = particle_oracles.ParticleOracle(
        env,
        wait_at_first_goal=1,
        multimodal=False,
        goal_threshold=0.01
    )
    
    # 生成一个 episode
    time_step = env.reset()
    oracle.reset()
    
    print(f"\n初始观测:")
    print(f"  pos_agent: {time_step.observation['pos_agent']}")
    print(f"  vel_agent: {time_step.observation['vel_agent']}")
    print(f"  pos_first_goal: {time_step.observation['pos_first_goal']}")
    print(f"  pos_second_goal: {time_step.observation['pos_second_goal']}")
    
    # 检查 Oracle 是否能访问 env 的内部状态
    print(f"\n检查 Oracle 访问环境状态:")
    try:
        # Oracle 需要访问 env.obs_log[0]
        if hasattr(oracle._env, 'obs_log'):
            print(f"  ✓ oracle._env.obs_log 存在")
            print(f"  obs_log[0] keys: {oracle._env.obs_log[0].keys()}")
        elif hasattr(oracle._env, '_env') and hasattr(oracle._env._env, 'obs_log'):
            print(f"  ✓ oracle._env._env.obs_log 存在 (双层包装)")
            print(f"  obs_log[0] keys: {oracle._env._env.obs_log[0].keys()}")
        else:
            print(f"  ✗ 无法访问 obs_log!")
            # 尝试查找环境
            env_to_check = oracle._env
            depth = 0
            while hasattr(env_to_check, '_env') and depth < 5:
                env_to_check = env_to_check._env
                depth += 1
                if hasattr(env_to_check, 'obs_log'):
                    print(f"  ✓ 在第 {depth} 层找到 obs_log")
                    break
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    positions = []
    actions = []
    step_count = 0
    
    # 记录到达第一个目标的时间
    reached_first_goal = False
    first_goal_step = -1
    
    # 记录到达第二个目标的时间
    reached_second_goal = False
    second_goal_step = -1
    
    pos_first_goal = time_step.observation['pos_first_goal']
    pos_second_goal = time_step.observation['pos_second_goal']
    
    print(f"\n开始执行 episode:")
    while not time_step.is_last() and step_count < 50:
        # 记录位置
        pos_agent = time_step.observation['pos_agent']
        positions.append(pos_agent.copy())
        
        # 检查距离
        dist_first = np.linalg.norm(pos_agent - pos_first_goal)
        dist_second = np.linalg.norm(pos_agent - pos_second_goal)
        
        if dist_first < 0.05 and not reached_first_goal:
            reached_first_goal = True
            first_goal_step = step_count
            print(f"  步骤 {step_count}: 到达第一个目标 (距离={dist_first:.4f})")
        
        if dist_second < 0.05 and not reached_second_goal:
            reached_second_goal = True
            second_goal_step = step_count
            print(f"  步骤 {step_count}: 到达第二个目标 (距离={dist_second:.4f})")
        
        # Oracle 选择动作
        policy_step = oracle.action(time_step)
        action = policy_step.action
        actions.append(action.copy())
        
        # 检查动作目标
        if step_count < 5 or step_count == first_goal_step or step_count == second_goal_step:
            dist_action_first = np.linalg.norm(action - pos_first_goal)
            dist_action_second = np.linalg.norm(action - pos_second_goal)
            if dist_action_first < dist_action_second:
                target = "第一个目标"
            else:
                target = "第二个目标"
            print(f"  步骤 {step_count}: 动作指向 {target} (action={action})")
        
        # 执行动作
        time_step = env.step(action)
        step_count += 1
        
        # 打印 episode 状态
        if step_count % 10 == 0:
            print(f"  步骤 {step_count}: pos={pos_agent}, dist_first={dist_first:.4f}, dist_second={dist_second:.4f}")
    
    # 添加最后一步
    if not time_step.is_last():
        final_pos = time_step.observation['pos_agent']
        positions.append(final_pos.copy())
    
    print(f"\nEpisode 完成:")
    print(f"  总步数: {step_count}")
    print(f"  记录的位置数: {len(positions)}")
    print(f"  记录的动作数: {len(actions)}")
    print(f"  到达第一个目标: {'是' if reached_first_goal else '否'} (步骤 {first_goal_step})")
    print(f"  到达第二个目标: {'是' if reached_second_goal else '否'} (步骤 {second_goal_step})")
    print(f"  终止原因: {'到达最大步数' if step_count >= 50 else '其他原因'}")
    
    # 分析轨迹
    positions = np.array(positions)
    print(f"\n轨迹分析:")
    print(f"  起点: {positions[0]}")
    print(f"  终点: {positions[-1]}")
    print(f"  第一个目标: {pos_first_goal}")
    print(f"  第二个目标: {pos_second_goal}")
    
    # 计算最终距离
    final_dist_first = np.linalg.norm(positions[-1] - pos_first_goal)
    final_dist_second = np.linalg.norm(positions[-1] - pos_second_goal)
    print(f"  最终到第一个目标距离: {final_dist_first:.4f}")
    print(f"  最终到第二个目标距离: {final_dist_second:.4f}")
    
    # 绘制轨迹
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(positions[:, 0], positions[:, 1], 'k-', linewidth=2, label='Trajectory', alpha=0.6)
    ax.plot(positions[0, 0], positions[0, 1], 'bs', markersize=15, label='Start', zorder=10)
    ax.plot(pos_first_goal[0], pos_first_goal[1], 'go', markersize=12, label='First Goal', zorder=10)
    ax.plot(pos_second_goal[0], pos_second_goal[1], 'r*', markersize=20, label='Second Goal', zorder=10)
    ax.plot(positions[-1, 0], positions[-1, 1], 'mx', markersize=15, label='End', zorder=10)
    
    # 标注关键点
    if reached_first_goal and first_goal_step < len(positions):
        ax.plot(positions[first_goal_step, 0], positions[first_goal_step, 1], 
               'go', markersize=8, markerfacecolor='none', markeredgewidth=2)
        ax.text(positions[first_goal_step, 0], positions[first_goal_step, 1] + 0.05, 
               f'Step {first_goal_step}', ha='center', fontsize=8)
    
    if reached_second_goal and second_goal_step < len(positions):
        ax.plot(positions[second_goal_step, 0], positions[second_goal_step, 1], 
               'ro', markersize=8, markerfacecolor='none', markeredgewidth=2)
        ax.text(positions[second_goal_step, 0], positions[second_goal_step, 1] + 0.05, 
               f'Step {second_goal_step}', ha='center', fontsize=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Particle Episode Trajectory (Debug)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    
    output_dir = Path(__file__).parent.parent.parent / 'plots' / 'particle'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'debug_trajectory.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存到: {output_dir / 'debug_trajectory.png'}")
    plt.close()
    
    env.close()

if __name__ == '__main__':
    test_single_episode()

