#!/usr/bin/env python3
"""Time-stretch so101_red_pick_clean by 2x and 3x (one decode pass, two writers).

Joints (state + action) are linearly interpolated between source frames; images
are nearest-neighbor (a genuinely slower demo seen by a 30 fps camera repeats
near-identical frames). fps stays 30, so per-tick action deltas shrink by the
stretch factor — the policy learns intrinsically slower motion.
"""
import shutil
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SRC_ROOT = Path("/mnt/nvme/lerobot/yosubshin/so101_red_pick_clean")
FACTORS = {1: Path("/mnt/nvme/lerobot/yosubshin/so101_red_pick_trim1x"),
           2: Path("/mnt/nvme/lerobot/yosubshin/so101_red_pick_trim2x"),
           3: Path("/mnt/nvme/lerobot/yosubshin/so101_red_pick_trim3x")}
CAMS = ["observation.images.forward", "observation.images.wrist"]

src = LeRobotDataset("yosubshin/so101_red_pick_clean", root=SRC_ROOT)
feat = {}
for key in ["action", "observation.state"] + CAMS:
    f = dict(src.meta.features[key])
    f.pop("info", None)
    feat[key] = f

writers = {}
for k, root in FACTORS.items():
    if root.exists():
        raise SystemExit(f"{root} already exists")
    writers[k] = LeRobotDataset.create(
        repo_id=f"yosubshin/{root.name}", fps=src.fps, features=feat,
        root=root, robot_type=src.meta.robot_type, use_videos=True,
        image_writer_threads=4,
    )

ep_meta = src.meta.episodes
for ep in range(src.meta.total_episodes):
    lo, hi = int(ep_meta[ep]["dataset_from_index"]), int(ep_meta[ep]["dataset_to_index"])
    states, actions, imgs, task = [], [], {c: [] for c in CAMS}, None
    for i in range(lo, hi):
        item = src[i]
        states.append(item["observation.state"].numpy())
        actions.append(item["action"].numpy())
        task = item["task"]
        for c in CAMS:
            im = item[c]
            imgs[c].append((im.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    states, actions = np.stack(states), np.stack(actions)
    # Trim leading/trailing idle (median ~28/26 ticks of dwell per episode —
    # 32% of frames are near-still, teaching a "stay home" mode that makes the
    # policy dither at episode start). Keep a 5-frame margin on each side.
    speed = np.abs(np.diff(actions, axis=0)).max(1)
    moving = speed > 0.5
    if moving.any():
        first = max(0, int(np.argmax(moving)) - 5)
        last = min(len(actions), len(actions) - int(np.argmax(moving[::-1])) + 5)
        states, actions = states[first:last], actions[first:last]
        for c in CAMS:
            imgs[c] = imgs[c][first:last]
    n = len(states)
    for k, ds in writers.items():
        total = k * (n - 1) + 1
        for j in range(total):
            t = j / k
            i0 = int(np.floor(t)); i1 = min(i0 + 1, n - 1); frac = t - i0
            frame = {
                "observation.state": (1 - frac) * states[i0] + frac * states[i1],
                "action": (1 - frac) * actions[i0] + frac * actions[i1],
                "task": task,
            }
            near = int(round(t))
            for c in CAMS:
                frame[c] = imgs[c][near]
            ds.add_frame(frame)
        ds.save_episode()
    if ep % 10 == 0:
        print(f"episode {ep}/{src.meta.total_episodes} done", flush=True)

for k, ds in writers.items():
    ds.finalize()
    shutil.copy2(SRC_ROOT / "meta/val_episodes.json", FACTORS[k] / "meta/val_episodes.json")
    print(f"{k}x: {ds.meta.total_episodes} eps, {ds.meta.total_frames} frames")
print("ALL DONE")
