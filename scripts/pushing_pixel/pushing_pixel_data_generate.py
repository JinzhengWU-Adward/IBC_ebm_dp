"""
使用 IBC 的 Block Pushing 环境和 Oracle 策略生成数据（RGB 图像观测版）。

参考：
- `ibc/ibc/configs/pushing_pixels/collect_data.sh`
- `ibc/data/policy_eval.py`
- `pushing_states/pushing_states_data_generate.py`

区别：
- 使用 RGB 图像观测（use_image_obs=True）
- 保存每个时间步的 RGB 图像
- 保存 action 数据
- 保存为 JSON + PNG 格式，便于 PyTorch 训练

性能优化：
- 移除不必要的图像resize（环境已返回正确尺寸）
- 使用浅拷贝替代deepcopy
- 直接使用numpy数组引用，避免不必要的copy
- 支持并行处理（multiprocessing）
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import multiprocessing as mp

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 追加 IBC 路径，方便从当前仓库直接运行
# 目录结构：generative_models/ibc/ 和 generative_models/IBC_ebm_dp/
# 本脚本位于：IBC_ebm_dp/scripts/pushing_pixel/
# 需要把包含 ibc 的上级目录加入 sys.path
# ---------------------------------------------------------------------------

IBC_PARENT_DIR = Path(__file__).parent.parent.parent.parent
if str(IBC_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(IBC_PARENT_DIR))

# 兼容 gym / tf-agents 的 patch（需在 tf-agents 之前 import）
from ibc.ibc.utils import gym_compat  # noqa: F401

from tf_agents.environments import suite_gym

from ibc.environments.block_pushing import block_pushing
from ibc.environments.collect.utils import get_oracle as get_oracle_module


def generate_pushing_episode(
    env,
    oracle,
    max_steps: int = 100,
    image_size=(240, 320),  # (H, W) - 默认图像尺寸
) -> Dict[str, Any]:
    """
    使用 IBC 的 Oracle 策略在 BlockPush 环境中生成一个 episode（RGB 图像版）。

    Args:
        env: 通过 suite_gym.load 得到的 PyEnvironment（使用 use_image_obs=True）。
        oracle: 由 ibc.environments.collect.utils.get_oracle 获得的 oracle 策略。
        max_steps: 每个 episode 的最大步数。
        image_size: RGB 图像尺寸 (height, width) - 仅用于验证，环境应已返回正确尺寸。

    Returns:
        episode_data: 包含轨迹、动作和 RGB 图像的字典。
    """
    time_step = env.reset()
    oracle.reset()

    rgb_images: List[np.ndarray] = []  # 存储每个时间步的 RGB 图像
    actions: List[np.ndarray] = []
    rewards: List[float] = []

    step_count = 0

    # 保存初始状态的图像（环境应已返回正确尺寸，直接使用）
    raw_obs = time_step.observation
    if 'rgb' in raw_obs:
        # 直接使用numpy数组，避免不必要的转换和copy
        rgb_img = np.asarray(raw_obs['rgb'], dtype=np.uint8)
        # 只在尺寸不匹配时才resize（通常不会发生）
        if rgb_img.shape[:2] != image_size:
            rgb_img = np.array(Image.fromarray(rgb_img).resize(
                (image_size[1], image_size[0]), Image.BILINEAR), dtype=np.uint8)
        rgb_images.append(rgb_img)  # 直接append，不需要copy（后续会保存为文件）
    else:
        rgb_img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        rgb_images.append(rgb_img)

    while not time_step.is_last() and step_count < max_steps:
        policy_step = oracle.action(time_step)
        action = policy_step.action
        # 直接转换为numpy数组，避免不必要的copy
        actions.append(np.asarray(action, dtype=np.float32))

        time_step = env.step(action)
        rewards.append(float(time_step.reward))

        # 保存执行动作后的图像
        raw_obs = time_step.observation
        if 'rgb' in raw_obs:
            rgb_img = np.asarray(raw_obs['rgb'], dtype=np.uint8)
            # 只在尺寸不匹配时才resize（通常不会发生）
            if rgb_img.shape[:2] != image_size:
                rgb_img = np.array(Image.fromarray(rgb_img).resize(
                    (image_size[1], image_size[0]), Image.BILINEAR), dtype=np.uint8)
            rgb_images.append(rgb_img)  # 直接append，不需要copy
        else:
            rgb_img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
            rgb_images.append(rgb_img)

        step_count += 1
    
    # 为最后一个观测添加"停留"动作（保持在当前位置）
    # 这样每个观测都有对应的动作：前 N 个是实际执行的动作，最后一个是停留动作
    if len(rgb_images) > len(actions):
        # 最后一个动作是"停留"（通常是 [0, 0] 或保持当前位置）
        # 使用零动作表示停留
        stay_action = np.zeros_like(actions[0]) if len(actions) > 0 else np.array([0.0, 0.0], dtype=np.float32)
        actions.append(stay_action)
    
    # 确保 images 和 actions 数量一致
    assert len(rgb_images) == len(actions), \
        f"图像数量 ({len(rgb_images)}) 和动作数量 ({len(actions)}) 不匹配"

    # 检查是否成功
    succeeded = False
    goal_distance = None
    if hasattr(env, "pyenv"):
        try:
            base_env = env.pyenv.envs[0]
            if hasattr(base_env, "succeeded"):
                succeeded = bool(base_env.succeeded)
            if hasattr(base_env, "goal_distance"):
                goal_distance = float(base_env.goal_distance)
        except Exception:
            pass

    return {
        "rgb_images": rgb_images,  # 所有时间步的 RGB 图像
        "actions": np.array(actions, dtype=np.float32),
        "rewards": rewards,
        "num_steps": len(actions) - 1,  # 实际执行的步数（不包括停留动作）
        "num_images": len(rgb_images),
        "succeeded": succeeded,
        "goal_distance": goal_distance,
    }


def _worker_generate_episodes(
    worker_id: int,
    num_episodes: int,
    output_dir: str,
    max_steps: int,
    image_size: tuple,
    seed_offset: int,
    queue: mp.Queue,
):
    """工作进程：生成指定数量的episodes并保存到磁盘。"""
    # 每个进程需要重新导入（multiprocessing需要）
    from tf_agents.environments import suite_gym
    from ibc.environments.block_pushing import block_pushing
    from ibc.environments.collect.utils import get_oracle as get_oracle_module
    
    # 每个进程需要创建自己的环境和oracle
    env_name = block_pushing.build_env_name(
        task="PUSH", shared_memory=False, use_image_obs=True
    )
    env = suite_gym.load(env_name)
    oracle = get_oracle_module.get_oracle(env, task="PUSH")

    output_path = Path(output_dir)
    local_samples = 0

    for ep_idx in range(num_episodes):
        try:
            episode_data = generate_pushing_episode(
                env, oracle, max_steps=max_steps, image_size=image_size
            )

            if episode_data["num_steps"] < 2:
                continue

            # 生成唯一的时间戳（包含worker_id避免冲突）
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            sample_id = worker_id * num_episodes + ep_idx
            timestamp = f"{timestamp}_{sample_id:06d}"

            # 为每条轨迹创建独立的文件夹
            episode_dir = output_path / f"traj_{timestamp}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            # 保存每个时间步的 RGB 图像到轨迹文件夹
            rgb_images = episode_data["rgb_images"]
            image_paths = []
            for step_idx, rgb_img in enumerate(rgb_images):
                img_path = episode_dir / f"frame_{step_idx:04d}.png"
                # 直接保存，避免不必要的转换
                Image.fromarray(rgb_img).save(img_path, optimize=False)
                image_paths.append(str(img_path.relative_to(output_path)))

            # 构造 JSON 数据
            json_data = {
                "sample_id": sample_id,
                "worker_id": worker_id,
                "timestamp": timestamp,
                "task": "PUSH",
                "use_image_obs": True,
                "success": bool(episode_data["succeeded"]),
                "final_goal_distance": episode_data["goal_distance"],
                "trajectory": {
                    "num_steps": int(episode_data["num_steps"]),
                    "num_images": int(episode_data["num_images"]),
                    "actions": episode_data["actions"].tolist(),
                    "rewards": episode_data["rewards"],
                    "image_paths": image_paths,
                },
                "config": {
                    "max_steps": max_steps,
                    "env_name": env_name,
                    "image_size": image_size,
                },
            }

            # 保存 JSON 文件
            json_path = episode_dir / "traj.json"
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)

            local_samples += 1
            queue.put(1)  # 通知主进程进度

        except Exception as e:
            print(f"Worker {worker_id}: 生成 episode {ep_idx} 时出错: {e}")
            continue

    env.close()
    queue.put(("done", worker_id, local_samples))


def generate_pushing_dataset(
    output_dir: str,
    num_episodes: int = 2000,
    max_steps: int = 100,
    image_size=(240, 320),  # (H, W) - 默认 240x320，匹配 IBC 默认
    seed: int = 0,
    num_workers: int = 10,  # 并行工作进程数（默认10，与IBC官方一致）
) -> None:
    """
    生成 Block Pushing（RGB 图像）数据集。

    Args:
        output_dir: 输出目录。
        num_episodes: 生成的 episode 总数量（默认 2000，与原始 IBC 近似）。
        max_steps: 每个 episode 最大步数。
        image_size: RGB 图像尺寸 (height, width)。
        seed: 环境随机种子（未使用，每个worker使用不同的seed_offset）。
        num_workers: 并行工作进程数（默认10，与IBC官方 collect_data.sh 一致）。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("生成 IBC Block Pushing (RGB 图像) 数据集")
    print("=" * 60)
    print(f"输出目录: {output_path}")
    print(f"Episode 总数量: {num_episodes}")
    print(f"并行工作进程数: {num_workers}")
    print(f"每个进程生成: {num_episodes // num_workers} episodes")
    print(f"最大步数: {max_steps}")
    print(f"图像尺寸: {image_size[0]}x{image_size[1]}")
    print()

    # 计算每个worker需要生成的episodes数
    episodes_per_worker = num_episodes // num_workers
    remaining_episodes = num_episodes % num_workers

    # 创建进度队列
    queue = mp.Queue()
    processes = []

    # 启动工作进程
    for worker_id in range(num_workers):
        worker_episodes = episodes_per_worker + (1 if worker_id < remaining_episodes else 0)
        if worker_episodes == 0:
            continue

        p = mp.Process(
            target=_worker_generate_episodes,
            args=(
                worker_id,
                worker_episodes,
                str(output_path),
                max_steps,
                image_size,
                seed + worker_id * 1000,  # 每个worker使用不同的seed offset
                queue,
            ),
        )
        p.start()
        processes.append(p)

    # 监控进度
    total_samples = 0
    done_workers = 0
    worker_samples = {}

    with tqdm(total=num_episodes, desc="生成 pushing episodes") as pbar:
        while done_workers < len(processes):
            try:
                result = queue.get(timeout=1.0)
                if isinstance(result, tuple) and result[0] == "done":
                    _, worker_id, samples = result
                    worker_samples[worker_id] = samples
                    done_workers += 1
                else:
                    total_samples += 1
                    pbar.update(1)
            except:
                # 超时，继续等待
                continue

    # 等待所有进程完成
    for p in processes:
        p.join()

    total_samples = sum(worker_samples.values())
    print(f"\n生成完成！共生成 {total_samples} 个样本")
    print(f"输出目录: {output_path}")
    print(f"  - 每条轨迹保存在独立的文件夹中: traj_YYYYMMDD_HHMMSS_XXXXXX/")
    print(f"  - 每个轨迹文件夹包含: traj.json 和 frame_*.png 图像文件")


if __name__ == "__main__":
    # 默认输出到 IBC_ebm_dp/data/pushing_pixel/
    default_output_dir = (
        Path(__file__).parent.parent.parent / "data" / "pushing_pixel"
    )
    os.makedirs(default_output_dir, exist_ok=True)

    # 与 IBC 原始脚本 collect_data.sh 规模一致：
    # 200 episodes × 10 replicas = 2000 episodes
    # 使用 RGB 图像观测，图像尺寸 240x320（IBC 默认）
    # 使用 10 个并行工作进程加速数据生成
    generate_pushing_dataset(
        output_dir=str(default_output_dir),
        num_episodes=2000,
        max_steps=100,
        image_size=(240, 320),  # (H, W) - 匹配 IBC 的 IMAGE_HEIGHT x IMAGE_WIDTH
        seed=0,
        num_workers=30,  # 并行工作进程数，与 IBC 官方 collect_data.sh 的 --replicas=10 一致
    )


