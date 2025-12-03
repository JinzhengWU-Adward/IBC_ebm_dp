"""
将 PushT zarr 数据中的动作从绝对位置转换为 delta 形式
- 所有动作的 delta 都是相对位移（包括第一个动作）
- 第一个动作的 delta = action[1] - action[0]（如果存在第二个动作）
- 中间动作的 delta = action[i] - action[i-1]
- 最后一个动作的 delta = (0, 0)
- 如果 episode 只有 1 个动作，delta = (0, 0)
"""
import numpy as np
import zarr
from pathlib import Path
import sys


def convert_actions_to_delta(zarr_path: str, output_key: str = 'action_delta'):
    """
    将 zarr 数据中的动作转换为 delta 形式
    
    Args:
        zarr_path: zarr 数据路径
        output_key: 输出数据的键名（默认 'action_delta'）
    """
    print(f"加载 zarr 数据: {zarr_path}")
    root = zarr.open(str(zarr_path), mode='r+')  # 使用 'r+' 以支持写入
    
    # 读取数据
    actions = root['data']['action'][:]  # (N, 2) float32
    episode_ends = root['meta']['episode_ends'][:]  # (num_episodes,) int64
    
    print(f"动作形状: {actions.shape}")
    print(f"回合数: {len(episode_ends)}")
    
    # 创建 delta 动作数组
    action_delta = np.zeros_like(actions)  # (N, 2) float32
    
    episode_start = 0
    for episode_idx, episode_end in enumerate(episode_ends):
        episode_length = episode_end - episode_start
        
        if episode_length == 0:
            episode_start = episode_end
            continue
        
        # 对于每个 episode：
        if episode_length == 1:
            # 如果只有一个动作，delta = (0, 0)
            action_delta[episode_start] = np.array([0.0, 0.0], dtype=np.float32)
        elif episode_length == 2:
            # 如果只有两个动作：
            # 1. 第一个动作的 delta = action[1] - action[0]（相对位移）
            action_delta[episode_start] = actions[episode_start + 1] - actions[episode_start]
            # 2. 最后一个动作的 delta = (0, 0)
            action_delta[episode_end - 1] = np.array([0.0, 0.0], dtype=np.float32)
        else:
            # 如果有多个动作：
            # 1. 第一个动作的 delta = action[1] - action[0]（相对位移，而不是绝对位置）
            action_delta[episode_start] = actions[episode_start + 1] - actions[episode_start]
            
            # 2. 中间动作的 delta = action[i] - action[i-1]（不包括第一个和最后一个）
            for i in range(episode_start + 1, episode_end - 1):
                action_delta[i] = actions[i] - actions[i-1]
            
            # 3. 最后一个动作的 delta = (0, 0)
            action_delta[episode_end - 1] = np.array([0.0, 0.0], dtype=np.float32)
        
        episode_start = episode_end
        
        if (episode_idx + 1) % 50 == 0:
            print(f"  处理了 {episode_idx + 1}/{len(episode_ends)} 个回合")
    
    print(f"共处理 {len(episode_ends)} 个回合")
    
    # 保存到 zarr
    print(f"\n保存 delta 动作到 'data/{output_key}'...")
    
    # 检查是否已存在，如果存在则删除
    if output_key in root['data']:
        print(f"  警告: 'data/{output_key}' 已存在，将被覆盖")
        del root['data'][output_key]
    
    # 创建新的 zarr 数组
    root['data'].create_dataset(
        output_key,
        data=action_delta,
        dtype=np.float32,
        chunks=(1000, 2)  # 设置合适的 chunk 大小
    )
    
    print(f"✓ Delta 动作已保存到 'data/{output_key}'")
    print(f"  Delta 动作形状: {action_delta.shape}")
    print(f"  Delta 动作范围: min={action_delta.min(axis=0)}, max={action_delta.max(axis=0)}")
    
    # 验证：检查每个 episode 的最后一个动作是否为 (0, 0)
    print("\n验证最后一个动作的 delta...")
    episode_start = 0
    all_last_deltas_zero = True
    for episode_idx, episode_end in enumerate(episode_ends):
        if episode_end > episode_start:
            last_delta = action_delta[episode_end - 1]
            if not np.allclose(last_delta, [0.0, 0.0], atol=1e-6):
                print(f"  警告: Episode {episode_idx} 的最后一个 delta 不是 (0, 0): {last_delta}")
                all_last_deltas_zero = False
        episode_start = episode_end
    
    if all_last_deltas_zero:
        print("✓ 所有 episode 的最后一个动作的 delta 都是 (0, 0)")
    else:
        print("⚠ 部分 episode 的最后一个动作的 delta 不是 (0, 0)")
    
    return action_delta


def main():
    # 硬编码路径
    IBC_ROOT = Path(__file__).parent.parent.parent  # IBC_ebm_dp
    zarr_path = IBC_ROOT / 'data' / 'pusht_cchi_v7_replay.zarr'
    output_key = 'action_delta'
    
    if not zarr_path.exists():
        print(f"错误: 数据路径不存在: {zarr_path}")
        sys.exit(1)
    
    try:
        convert_actions_to_delta(str(zarr_path), output_key)
        print("\n转换完成！")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

