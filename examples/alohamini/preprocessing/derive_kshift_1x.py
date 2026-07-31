#!/usr/bin/env python3
"""Derive k-frame action-shift variants of trim2x: action(t) = state(t+k).

Lead compensation for the kinesthetic no-plant-lead problem. Shift applied on
the stretched/execution timeline (servo lag is a real-time constant; execution
is 30fps real for any stretch, so lead-in-ticks is the invariant).
Variants: k1, k2, k3, kavg (mean of states t+1..t+3 — smoothed ~2-tick lead).
Videos hardlinked (identical content). Stats + per-episode stats recomputed
for the action channel.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path("/mnt/nvme/lerobot/yosubshin/so101_red_pick_trim1x")
VARIANTS = {"k3": 3, "k4": 4}

for name, k in VARIANTS.items():
    DST = Path(f"/mnt/nvme/lerobot/yosubshin/so101_red_pick_trim1x_{name}")
    if DST.exists():
        print(f"skip {name} (exists)"); continue
    (DST / "meta").mkdir(parents=True)
    shutil.copy2(SRC / "meta/info.json", DST / "meta/info.json")
    shutil.copy2(SRC / "meta/tasks.parquet", DST / "meta/tasks.parquet")
    shutil.copy2(SRC / "meta/val_episodes.json", DST / "meta/val_episodes.json")

    all_actions = []
    for src_file in sorted((SRC / "data").rglob("*.parquet")):
        t = pq.read_table(src_file)
        st = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False))
        eps = np.array(t.column("episode_index").to_pylist())
        new_a = np.empty_like(st)
        for ep in np.unique(eps):
            m = eps == ep
            s = st[m]
            n = len(s)
            if k == "avg":
                targ = np.stack([s[np.minimum(np.arange(n) + d, n - 1)] for d in (1, 2, 3)]).mean(0)
            else:
                targ = s[np.minimum(np.arange(n) + k, n - 1)]
            new_a[m] = targ
        all_actions.append(new_a.astype(np.float32))
        col = pa.FixedSizeListArray.from_arrays(pa.array(new_a.astype(np.float32).reshape(-1)), st.shape[1])
        t = t.set_column(t.schema.get_field_index("action"), "action", col)
        t = t.replace_schema_metadata(None)
        dst_file = DST / src_file.relative_to(SRC)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, dst_file)
    A = np.concatenate(all_actions)

    stats = json.loads((SRC / "meta/stats.json").read_text())
    q = lambda p: np.percentile(A, p, axis=0).tolist()
    stats["action"] = {"min": A.min(0).tolist(), "max": A.max(0).tolist(),
                       "mean": A.mean(0).tolist(), "std": A.std(0).tolist(),
                       "count": [int(len(A))],
                       "q01": q(1), "q10": q(10), "q50": q(50), "q90": q(90), "q99": q(99)}
    (DST / "meta/stats.json").write_text(json.dumps(stats, indent=4))

    # per-episode action stats
    for src_file in sorted((SRC / "meta/episodes").rglob("*.parquet")):
        t = pq.read_table(src_file)
        cols = {n_: t.column(n_) for n_ in t.schema.names}
        froms = t.column("dataset_from_index").to_pylist()
        tos = t.column("dataset_to_index").to_pylist()
        per = {kk: [] for kk in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")}
        counts = []
        for f_, t_ in zip(froms, tos):
            seg = A[f_:t_]
            per["min"].append(seg.min(0)); per["max"].append(seg.max(0))
            per["mean"].append(seg.mean(0)); per["std"].append(seg.std(0))
            for pp, kk in [(1, "q01"), (10, "q10"), (50, "q50"), (90, "q90"), (99, "q99")]:
                per[kk].append(np.percentile(seg, pp, axis=0))
            counts.append([len(seg)])
        for sk, vals in per.items():
            cols[f"stats/action/{sk}"] = pa.array([v.astype(np.float32).tolist() for v in vals],
                                                  type=pa.list_(pa.float32()))
        cols["stats/action/count"] = pa.array(counts, type=pa.list_(pa.int64()))
        dst_file = DST / src_file.relative_to(SRC)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols), dst_file)

    subprocess.run(["cp", "-al", str(SRC / "videos"), str(DST / "videos")], check=True)
    print(f"{name} done")
print("ALL DONE")
