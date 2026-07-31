#!/usr/bin/env python3
"""Map recorded episodes to the 1080p fisheye archive segments that cover them.

Reads <dataset_root>/meta/episode_wallclock.jsonl (written by record_so101.py)
and an archive directory of fisheye_*.mkv files, and prints which segment(s)
cover each episode and at what offset.

Archive naming schemes:
  fisheye_%Y%m%dT%H%M%SZ_%04d.mkv   — UTC (current host)
  fisheye_%Y%m%d_%H%M%S[_%04d].mkv  — legacy, Pi-LOCAL time: pass
                                       --legacy_utc_offset (e.g. 1 for BST)

Usage:
  python map_episodes_to_archive.py \
      --dataset_root ~/.cache/huggingface/lerobot/local/<name> \
      --archive_dir /mnt/nvme/lerobot/fisheye_archive [--legacy_utc_offset 1]
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
import subprocess
import time
from pathlib import Path

SEGMENT_S = 300.0

_PAT_UTC = re.compile(r"fisheye_(\d{8}T\d{6})Z_(\d{4})\.mkv$")
_PAT_LEGACY = re.compile(r"fisheye_(\d{8})_(\d{6})(?:_(\d{4}))?\.mkv$")


def _duration_s(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=30).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def _archive_windows(archive_dir: Path, legacy_utc_offset: float) -> list[tuple[float, float, Path]]:
    """Return (start_unix, end_unix, path) per file, resolving segment offsets."""
    windows = []
    for p in sorted(archive_dir.glob("fisheye_*.mkv")):
        m = _PAT_UTC.search(p.name)
        if m:
            base = calendar.timegm(time.strptime(m.group(1), "%Y%m%dT%H%M%S"))
            seg = int(m.group(2))
        else:
            m = _PAT_LEGACY.search(p.name)
            if not m:
                continue
            base = calendar.timegm(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))
            base -= legacy_utc_offset * 3600.0
            seg = int(m.group(3)) if m.group(3) else 0
        start = base + seg * SEGMENT_S
        # A segment truncated by ffmpeg being killed has no trailer and probes
        # as 0 — assume a full segment window rather than dropping it.
        dur = _duration_s(p) or SEGMENT_S
        windows.append((start, start + dur, p))
    return windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--archive_dir", action="append", required=True,
                    help="Repeatable — e.g. workstation dir and a Pi-synced dir")
    ap.add_argument("--legacy_utc_offset", type=float, default=0.0,
                    help="Hours the LEGACY (no 'Z') filenames are ahead of UTC (BST=1)")
    ap.add_argument("--extract", action="store_true",
                    help="Cut per-episode 1080p clips (stream copy, MJPG is "
                         "all-keyframe) into <dataset_root>/videos_1080p/")
    args = ap.parse_args()

    jsonl = Path(args.dataset_root).expanduser() / "meta" / "episode_wallclock.jsonl"
    if not jsonl.is_file():
        raise SystemExit(f"No {jsonl} — dataset predates the wallclock sidecar.")
    windows = []
    for d in args.archive_dir:
        windows += _archive_windows(Path(d).expanduser(), args.legacy_utc_offset)
    windows.sort()

    for line in jsonl.read_text().splitlines():
        ep = json.loads(line)
        s, e = ep["start_unix"], ep["end_unix"]
        note = f" [{ep['source']}]" if ep.get("source") else ""
        hits = [(w, x, p) for w, x, p in windows if w < e and x > s]
        print(f"ep{ep['episode_index']}: {ep['start_utc']} .. {ep['end_utc']}{note}")
        if not hits:
            print("    !! NO ARCHIVE COVERAGE")
        for n, (w, x, p) in enumerate(hits):
            off, upto = max(0.0, s - w), min(e, x) - w
            print(f"    {p.name}  offset {off:7.1f}s .. {upto:7.1f}s")
            if args.extract:
                out_dir = Path(args.dataset_root).expanduser() / "videos_1080p"
                out_dir.mkdir(exist_ok=True)
                part = f"_part{n}" if len(hits) > 1 else ""
                out = out_dir / f"episode_{ep['episode_index']:06d}{part}.mkv"
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-ss", f"{off:.3f}", "-i", str(p),
                     "-t", f"{upto - off:.3f}", "-c", "copy", str(out)], check=True)
                print(f"      -> {out}")


if __name__ == "__main__":
    main()
