"""
使用 IBC 的 Block Pushing 环境和 Oracle 策略生成数据（状态观察版）。

参考：
- `ibc/ibc/configs/pushing_states/collect_data.sh`
- `ibc/data/policy_eval.py`
- `_2d/particle/particle_data_generate.py`

区别：
- 不再保存 TFRecord，而是和 particle 数据类似，保存为 JSON + 可视化 PNG。
- 直接在 PyTorch 项目内使用，避免 TF & tf-agents 的训练依赖。
"""

import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 追加 IBC 路径，方便从当前仓库直接运行
# 目录结构：generative_models/ibc/ 和 generative_models/IBC_ebm_dp/
# 本脚本位于：IBC_ebm_dp/scripts/pushing_states/
# 需要把包含 ibc 的上级目录加入 sys.path
# ---------------------------------------------------------------------------

IBC_PARENT_DIR = Path(__file__).parent.parent.parent.parent
if str(IBC_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(IBC_PARENT_DIR))

# 兼容 gym / tf-agents 的 patch（需在 tf-agents 之前 import）
from ibc.utils import gym_compat  # noqa: F401

from tf_agents.environments import suite_gym

from ibc.environments.block_pushing import block_pushing
from ibc.environments.collect.utils import get_oracle as get_oracle_module


def _extract_state_from_observation(obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """从 IBC BlockPush 环境的 observation 中提取关键的状态信息."""

    def _to_list(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32).tolist()
        return x

    state = {
        "block_translation": _to_list(obs.get("block_translation")),
        "block_orientation": _to_list(obs.get("block_orientation")),
        "effector_translation": _to_list(obs.get("effector_translation")),
        "effector_target_translation": _to_list(
            obs.get("effector_target_translation")
        ),
        "target_translation": _to_list(obs.get("target_translation")),
        "target_orientation": _to_list(obs.get("target_orientation")),
    }
    # 可能还包含 rgb（如果用图像观测），这里只保存 state 相关
    return state


def generate_pushing_episode(
    env,
    oracle,
    max_steps: int = 100,
) -> Dict[str, Any]:
    """
    使用 IBC 的 Oracle 策略在 BlockPush 环境中生成一个 episode。

    Args:
        env: 通过 suite_gym.load 或 wrap_env 得到的 PyEnvironment。
        oracle: 由 ibc.environments.collect.utils.get_oracle 获得的 oracle 策略。
        max_steps: 每个 episode 的最大步数。

    Returns:
        episode_data: 包含轨迹和动作的字典。
    """
    from tf_agents.trajectories import time_step as ts  # 局部导入，避免全局依赖

    time_step = env.reset()
    oracle.reset()

    observations: List[Dict[str, Any]] = []
    states: List[Dict[str, Any]] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []

    step_count = 0

    while not time_step.is_last() and step_count < max_steps:
        raw_obs = time_step.observation
        # raw_obs 通常是 dict[str, np.ndarray]
        observations.append(copy.deepcopy(raw_obs))
        states.append(_extract_state_from_observation(raw_obs))

        policy_step = oracle.action(time_step)
        action = policy_step.action
        actions.append(np.array(action, dtype=np.float32).copy())

        time_step = env.step(action)
        rewards.append(float(time_step.reward))

        step_count += 1

    # 记录最终状态
    if not time_step.is_last():
        raw_obs = time_step.observation
        observations.append(copy.deepcopy(raw_obs))
        states.append(_extract_state_from_observation(raw_obs))

    succeeded = False
    goal_distance = None
    if hasattr(env, "pyenv"):
        # suite_gym.load 返回的是 PyEnvironment 包装，可以通过 .pyenv.envs[0] 拿到底层 gym env
        try:
            base_env = env.pyenv.envs[0]
            if hasattr(base_env, "succeeded"):
                succeeded = bool(base_env.succeeded)
            if hasattr(base_env, "goal_distance"):
                goal_distance = float(base_env.goal_distance)
        except Exception:
            pass

    return {
        "observations": states,
        "raw_observations_len": len(observations),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": rewards,
        "num_steps": len(states),
        "succeeded": succeeded,
        "goal_distance": goal_distance,
    }


def _render_topdown_image(env, image_size=(180, 320)) -> np.ndarray:
    """
    从 BlockPush 环境渲染一个 RGB 图像。

    对于 state-only 环境，使用 env.render()，对于 tf-agents 包装的环境，
    尝试从底层 gym env 渲染。
    """
    # 尝试从底层 gym Env 渲染
    img = None

    # suite_gym.load 返回的 env 支持 .render()，但有些 wrapper 需要访问 pyenv
    try:
        img = env.render()
    except Exception:
        pass

    if img is None and hasattr(env, "pyenv"):
        try:
            base_env = env.pyenv.envs[0]
            img = base_env.render(mode="rgb_array")
        except Exception:
            img = None

    if img is None:
        # 回退：生成空白图，避免脚本崩溃
        img = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255

    # resize 到统一大小
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    return np.array(pil_img)


def generate_pushing_dataset(
    output_dir: str,
    num_episodes: int = 2000,
    max_steps: int = 100,
    image_size=(180, 320),
    seed: int = 0,
) -> None:
    """
    生成 Block Pushing（states）数据集。

    Args:
        output_dir: 输出目录。
        num_episodes: 生成的 episode 数量（默认 2000，与原始 IBC 近似）。
        max_steps: 每个 episode 最大步数。
        image_size: 渲染图像大小 (H, W)。
        seed: 环境随机种子。
    """
    output_path = Path(output_dir)
    (output_path / "traj").mkdir(parents=True, exist_ok=True)
    (output_path / "pic").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("生成 IBC Block Pushing (states) 数据集")
    print("=" * 60)
    print(f"输出目录: {output_path}")
    print(f"Episode 数量: {num_episodes}")
    print(f"最大步数: {max_steps}")
    print()

    # 构造与 ibc/data/policy_eval.py 一致的环境名称
    env_name = block_pushing.build_env_name(
        task="PUSH", shared_memory=False, use_image_obs=False
    )
    env = suite_gym.load(env_name)

    # 创建 Oracle 策略（和 policy_eval 中逻辑一致）
    oracle = get_oracle_module.get_oracle(env, task="PUSH")

    total_samples = 0

    for ep_idx in tqdm(range(num_episodes), desc="生成 pushing episodes"):
        try:
            episode_data = generate_pushing_episode(env, oracle, max_steps=max_steps)

            if episode_data["num_steps"] < 2:
                continue

            # 生成时间戳
            import time

            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            timestamp = f"{timestamp}_{total_samples:06d}"

            # 渲染图像
            img = _render_topdown_image(env, image_size=image_size)
            img_path = output_path / "pic" / f"pushing_{timestamp}.png"
            Image.fromarray(img).save(img_path)

            # 构造 JSON 数据
            json_data = {
                "sample_id": total_samples,
                "timestamp": timestamp,
                "task": "PUSH",
                "success": bool(episode_data["succeeded"]),
                "final_goal_distance": episode_data["goal_distance"],
                "trajectory": {
                    "num_steps": int(episode_data["num_steps"]),
                    "observations": episode_data["observations"],
                    "actions": episode_data["actions"].tolist(),
                    "rewards": episode_data["rewards"],
                },
                "config": {
                    "max_steps": max_steps,
                    "env_name": env_name,
                },
            }

            json_path = output_path / "traj" / f"traj_{timestamp}.json"
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)

            total_samples += 1

        except Exception as e:
            print(f"警告: 生成 episode {ep_idx} 时出错: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n生成完成！共生成 {total_samples} 个样本")
    print(f"输出目录: {output_path}")


if __name__ == "__main__":
    # 默认输出到 IBC_ebm_dp/data/pushing_states/
    default_output_dir = (
        Path(__file__).parent.parent.parent.parent / "data" / "pushing_states"
    )
    os.makedirs(default_output_dir, exist_ok=True)

    # 与 IBC 原始脚本 collect_data.sh 规模接近：
    # 200 episodes × 10 replicas = 2000 episodes
    generate_pushing_dataset(
        output_dir=str(default_output_dir),
        num_episodes=2000,
        max_steps=100,
        image_size=(180, 320),
        seed=0,
    )


