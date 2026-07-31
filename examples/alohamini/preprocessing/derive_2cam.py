#!/usr/bin/env python3
"""Derive a 2-camera variant of the gripper-only dataset: drop wrist_left.

Keeps forward + wrist_right. State (2-dim gripper) and action (16-dim joints)
unchanged; data parquets copied verbatim (v3 stores no image columns there).
"""
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

SRC = Path("/mnt/nvme/lerobot/yosubshin/am1_red_bin_gripperonly")
DST = Path("/mnt/nvme/lerobot/yosubshin/am1_red_bin_gripperonly_2cam")
DROP = "observation.images.wrist_left"

if DST.exists():
    raise SystemExit(f"{DST} already exists, refusing to overwrite")

# meta/info.json
info = json.loads((SRC / "meta/info.json").read_text())
assert DROP in info["features"]
del info["features"][DROP]
(DST / "meta").mkdir(parents=True)
(DST / "meta/info.json").write_text(json.dumps(info, indent=4))

# meta/stats.json
stats = json.loads((SRC / "meta/stats.json").read_text())
stats.pop(DROP, None)
(DST / "meta/stats.json").write_text(json.dumps(stats, indent=4))

shutil.copy2(SRC / "meta/tasks.parquet", DST / "meta/tasks.parquet")
(DST / "meta/val_episodes.json").write_text(json.dumps({"val_episodes": [30, 31, 32, 33]}))

# meta/episodes: drop the wrist_left video-pointer and stats columns
for src_file in sorted((SRC / "meta/episodes").rglob("*.parquet")):
    rel = src_file.relative_to(SRC)
    t = pq.read_table(src_file)
    keep = [n for n in t.schema.names if f"/{DROP}/" not in n]
    dropped = len(t.schema.names) - len(keep)
    assert dropped > 0, "expected wrist_left columns in episodes metadata"
    out = t.select(keep)
    dst_file = DST / rel
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst_file)

# data parquets: unchanged
for src_file in sorted((SRC / "data").rglob("*.parquet")):
    dst_file = DST / src_file.relative_to(SRC)
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)

# videos: only the two kept cameras
for cam in ["observation.images.forward", "observation.images.wrist_right"]:
    shutil.copytree(SRC / "videos" / cam, DST / "videos" / cam)

print("done")
