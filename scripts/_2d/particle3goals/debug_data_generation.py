import json, numpy as np
from pathlib import Path

# 脚本所在目录：.../IBC_ebm_dp/scripts/_2d/particle3goals
ROOT = Path(__file__).parent.parent.parent.parent  # IBC_ebm_dp 根目录
traj_dir = ROOT / "data" / "_2d" / "particle3goals" / "traj"

files = sorted(traj_dir.glob("traj_*.json"))[:3]

for f in files:
    print("====", f.name)
    data = json.loads(f.read_text())
    pos = np.array(data["trajectory"]["positions"], dtype=np.float32)   # (T,2)
    acts = np.array(data["actions"], dtype=np.float32)                  # (T-1,2)
    T = min(len(pos)-1, len(acts))
    for t in range(0, T, max(1, T//5)):
        print(f"t={t} pos={pos[t]}  action={acts[t]}")
    print()