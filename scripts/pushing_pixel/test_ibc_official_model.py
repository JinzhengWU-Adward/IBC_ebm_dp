#!/usr/bin/env python3
"""
测试 IBC 官方训练好的 Pixel EBM 模型

所有参数都在脚本中硬编码，直接运行即可：
    python test_ibc_official_model.py
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

# 添加 IBC 路径
IBC_ROOT = Path(__file__).parent.parent.parent.parent / 'ibc'
sys.path.insert(0, str(IBC_ROOT.parent))  # 添加 generative_models 目录

# 设置 TensorFlow 日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warnings, 3=errors

import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers
from tf_agents.policies import py_tf_eager_policy

# 导入 IBC 模块
from ibc.environments.block_pushing import block_pushing

# ============================================================================
# 配置参数（在这里修改所有参数）
# ============================================================================

# 模型路径配置
MODEL_BASE_DIR = '/tmp/ibc_logs/mlp_ebm'  # IBC 训练输出的基础目录
CHECKPOINT_STEP = None  # 检查点步数（如果为 None，则使用最新的检查点）

# 评估参数（匹配 gin 配置）
NUM_EPISODES = 20  # 评估回合数（匹配 gin: train_eval.eval_episodes = 20）
MAX_STEPS = 150  # 每回合最大步数
SEQUENCE_LENGTH = 2  # 观测序列长度（匹配 gin: train_eval.sequence_length = 2）
GOAL_TOLERANCE = 0.02  # 目标容差（匹配 gin: train_eval.goal_tolerance = 0.02）

# 其他参数
SHARED_MEMORY = False  # 是否使用共享内存
SAVE_VIDEO = False  # 是否保存视频
SEED = 0  # 随机种子（匹配 gin: train_eval.seed = 0）

# ============================================================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(model_base_dir: str, checkpoint_step: int = None):
    """
    查找最新的模型目录和检查点
    
    IBC 官方的目录结构：
    - model_base_dir/pixel_ibc_langevin/YYYYMMDD-HHMMSS/policies/policy
    - model_base_dir/pixel_ibc_langevin/YYYYMMDD-HHMMSS/policies/checkpoints/policy_checkpoint_*
    
    Args:
        model_base_dir: 模型基础目录（例如 /tmp/ibc_logs/mlp_ebm）
        checkpoint_step: 检查点步数（如果为 None，则查找最新的）
    
    Returns:
        (saved_model_path, checkpoint_path) 元组
    """
    model_base_dir = Path(model_base_dir)
    
    # 查找 tag 目录（pixel_ibc_langevin）
    tag_dir = model_base_dir / 'pixel_ibc_langevin'
    if not tag_dir.exists():
        raise FileNotFoundError(f"未找到 tag 目录: {tag_dir}")
    
    # 查找所有时间戳目录（YYYYMMDD-HHMMSS 格式）
    timestamp_dirs = sorted([d for d in tag_dir.iterdir() if d.is_dir() and len(d.name) == 15], reverse=True)
    
    if not timestamp_dirs:
        raise FileNotFoundError(f"未找到时间戳目录: {tag_dir}/*")
    
    latest_timestamp_dir = timestamp_dirs[0]
    logger.info(f"使用时间戳目录: {latest_timestamp_dir.name}")
    
    # 查找 SavedModel
    saved_model_path = latest_timestamp_dir / 'policies' / 'policy'
    if not saved_model_path.exists():
        raise FileNotFoundError(f"SavedModel 不存在: {saved_model_path}")
    
    # 查找检查点
    checkpoints_dir = latest_timestamp_dir / 'policies' / 'checkpoints'
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"检查点目录不存在: {checkpoints_dir}")
    
    if checkpoint_step is None:
        # 查找最新的检查点
        checkpoint_dirs = sorted([d for d in checkpoints_dir.iterdir() 
                                 if d.is_dir() and d.name.startswith('policy_checkpoint_')], 
                                reverse=True)
        if not checkpoint_dirs:
            raise FileNotFoundError(f"未找到检查点目录: {checkpoints_dir}")
        checkpoint_path = checkpoint_dirs[0]
    else:
        # 使用指定的检查点步数
        checkpoint_path = checkpoints_dir / f'policy_checkpoint_{checkpoint_step:010d}'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
    
    logger.info(f"使用检查点: {checkpoint_path.name}")
    
    return str(saved_model_path), str(checkpoint_path)


def main():
    """主函数：运行评估"""
    logger.info("=" * 60)
    logger.info("评估 IBC 官方 Pixel EBM 模型")
    logger.info("=" * 60)
    logger.info(f"模型基础目录: {MODEL_BASE_DIR}")
    logger.info(f"检查点步数: {CHECKPOINT_STEP}")
    logger.info(f"评估回合数: {NUM_EPISODES}")
    logger.info(f"最大步数: {MAX_STEPS}")
    logger.info(f"序列长度: {SEQUENCE_LENGTH}")
    logger.info(f"目标容差: {GOAL_TOLERANCE}")
    logger.info("")
    
    # 查找模型路径
    try:
        saved_model_path, checkpoint_path = find_latest_checkpoint(
            MODEL_BASE_DIR,
            CHECKPOINT_STEP
        )
    except FileNotFoundError as e:
        logger.error(f"模型路径错误: {e}")
        logger.error("请检查 MODEL_BASE_DIR 和 CHECKPOINT_STEP 配置")
        return 1
    
    logger.info(f"SavedModel 路径: {saved_model_path}")
    logger.info(f"Checkpoint 路径: {checkpoint_path}")
    logger.info("")
    
    # 检查路径是否存在
    if not os.path.exists(saved_model_path):
        logger.error(f"SavedModel 路径不存在: {saved_model_path}")
        return 1
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint 路径不存在: {checkpoint_path}")
        return 1
    
    # 创建环境
    logger.info("创建环境...")
    try:
        env_name = block_pushing.build_env_name('PUSH', SHARED_MEMORY, use_image_obs=True)
        env = suite_gym.load(env_name)
        
        # 设置目标容差
        if hasattr(env, 'set_goal_dist_tolerance'):
            env.set_goal_dist_tolerance(GOAL_TOLERANCE)
        
        # 添加 HistoryWrapper（匹配训练时的观测格式）
        env = wrappers.HistoryWrapper(
            env, 
            history_length=SEQUENCE_LENGTH, 
            tile_first_step_obs=True
        )
        logger.info(f"环境已创建: {env_name}")
        logger.info("")
    except Exception as e:
        logger.error(f"创建环境失败: {e}")
        return 1
    
    # 加载策略
    logger.info("加载策略...")
    try:
        policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            saved_model_path, 
            load_specs_from_pbtxt=True
        )
        policy.update_from_checkpoint(checkpoint_path)
        logger.info("策略加载成功")
        logger.info("")
    except Exception as e:
        logger.error(f"策略加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 设置随机种子
    if SEED is not None:
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        if hasattr(env, 'seed'):
            env.seed(SEED)
    
    # 评估循环
    logger.info("开始评估...")
    logger.info("")
    
    success_count = 0
    total_rewards = []
    episode_lengths = []
    final_distances = []
    
    pbar = tqdm(total=NUM_EPISODES, desc="评估进度")
    
    for episode in range(NUM_EPISODES):
        # 重置环境
        time_step = env.reset()
        policy_state = policy.get_initial_state(1)
        
        episode_reward = 0.0
        step = 0
        
        # 运行回合
        while not time_step.is_last() and step < MAX_STEPS:
            # 获取动作
            policy_step = policy.action(time_step, policy_state)
            action = policy_step.action
            policy_state = policy_step.state
            
            # 执行动作
            next_time_step = env.step(action)
            
            # 更新统计
            if hasattr(next_time_step.reward, 'numpy'):
                episode_reward += next_time_step.reward.numpy()[0]
            else:
                episode_reward += float(next_time_step.reward)
            
            time_step = next_time_step
            step += 1
        
        # 检查是否成功
        success = False
        final_distance = float('inf')
        
        # 尝试多种方式获取成功状态
        if hasattr(env, 'succeeded'):
            success = bool(env.succeeded)
        elif hasattr(env, '_env') and hasattr(env._env, 'succeeded'):
            success = bool(env._env.succeeded)
        
        # 尝试获取最终距离
        if hasattr(env, 'goal_distance'):
            final_distance = float(env.goal_distance)
        elif hasattr(env, '_env') and hasattr(env._env, 'goal_distance'):
            final_distance = float(env._env.goal_distance)
        
        # 如果无法直接获取成功状态，通过距离判断
        if not success and final_distance != float('inf'):
            success = final_distance <= GOAL_TOLERANCE
        
        if success:
            success_count += 1
        
        if final_distance != float('inf'):
            final_distances.append(final_distance)
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        # 更新进度条
        pbar.update(1)
        pbar.set_postfix({
            'success': f'{success_count}/{episode+1}',
            'reward': f'{episode_reward:.2f}',
            'steps': step
        })
    
    pbar.close()
    
    # 计算统计信息
    success_rate = success_count / NUM_EPISODES
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    avg_final_distance = np.mean(final_distances) if final_distances else None
    
    # 打印结果
    logger.info("")
    logger.info("=" * 60)
    logger.info("评估结果")
    logger.info("=" * 60)
    logger.info(f"成功率: {success_count}/{NUM_EPISODES} ({success_rate*100:.1f}%)")
    logger.info(f"平均奖励: {avg_reward:.4f}")
    logger.info(f"平均步数: {avg_length:.1f}")
    if avg_final_distance is not None:
        logger.info(f"平均最终距离: {avg_final_distance:.4f}")
    logger.info("")
    
    # 保存结果到文件
    results_file = Path(saved_model_path).parent.parent / 'eval_results.txt'
    try:
        with open(results_file, 'w') as f:
            f.write("IBC 官方 Pixel EBM 模型评估结果\n")
            f.write("=" * 60 + "\n")
            f.write(f"SavedModel 路径: {saved_model_path}\n")
            f.write(f"Checkpoint 路径: {checkpoint_path}\n")
            f.write(f"评估回合数: {NUM_EPISODES}\n")
            f.write(f"最大步数: {MAX_STEPS}\n")
            f.write(f"序列长度: {SEQUENCE_LENGTH}\n")
            f.write(f"目标容差: {GOAL_TOLERANCE}\n")
            f.write("\n")
            f.write(f"成功率: {success_count}/{NUM_EPISODES} ({success_rate*100:.1f}%)\n")
            f.write(f"平均奖励: {avg_reward:.4f}\n")
            f.write(f"平均步数: {avg_length:.1f}\n")
            if avg_final_distance is not None:
                f.write(f"平均最终距离: {avg_final_distance:.4f}\n")
            f.write("\n")
            f.write("每回合详情:\n")
            for i in range(NUM_EPISODES):
                reward = total_rewards[i]
                length = episode_lengths[i]
                is_success = (i < success_count) if success_count > 0 else False
                # 更准确的判断：根据最终距离
                if i < len(final_distances):
                    is_success = final_distances[i] <= GOAL_TOLERANCE
                f.write(f"  Episode {i+1}: reward={reward:.4f}, steps={length}, success={is_success}\n")
                if i < len(final_distances):
                    f.write(f"    final_distance={final_distances[i]:.4f}\n")
        
        logger.info(f"结果已保存到: {results_file}")
    except Exception as e:
        logger.warning(f"保存结果文件失败: {e}")
    
    logger.info("评估完成！")
    return 0


if __name__ == '__main__':
    sys.exit(main())
