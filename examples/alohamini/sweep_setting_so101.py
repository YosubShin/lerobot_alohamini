#!/usr/bin/env python3
"""Run one gain-setting's worth of tracking measurements. MOVES THE ARM.

For each requested episode: ramp gently to the episode's start pose, then run
replay_measure_so101.py (which streams the episode as commands and logs
cmd/obs). Finishes with a held-pose hunting check: hold the home pose
quietly and record observation jitter — catches integral-windup oscillation
that the moving benchmark cannot see.

  python sweep_setting_so101.py --log_dir /mnt/nvme/lerobot/outputs/so101_gain_sweep/i4_d32 \
      --episodes 0 1 2 3 4 5 --remote_ip 192.168.0.50
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_zmq_client import SO101ZmqClient, SO101ZmqClientConfig  # noqa: E402

HOLD_SECONDS = 10.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="so101_red_pick_trim1x")
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--remote_ip", default="192.168.0.50")
    p.add_argument("--log_dir", required=True)
    p.add_argument("--ramp_seconds", type=float, default=2.5)
    args = p.parse_args()

    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    # start poses for all episodes + a hold pose (episode 0 start)
    starts = {}
    for ep in args.episodes:
        ds = LeRobotDataset(args.dataset, episodes=[ep])
        names = list(ds.features[ACTION]["names"])
        acts = ds.hf_dataset.select_columns(ACTION)
        starts[ep] = {n: float(acts[0][ACTION][i]) for i, n in enumerate(names)}
    hold_pose = starts[args.episodes[0]]

    # The host publishes observations on a PUSH socket, which round-robins
    # between connected PULL clients — so this orchestrator must never stay
    # connected while replay_measure runs, or both see half the stream.
    def make_robot():
        r = SO101ZmqClient(SO101ZmqClientConfig(remote_ip=args.remote_ip, cameras={}))
        r.connect()
        r.enable_torque()
        return r

    def ramp_to(robot, target: dict, seconds: float) -> None:
        # ensure a FRESH observation (never ramp from the zero-initialized cache)
        deadline = time.perf_counter() + 5.0
        while robot.msgs_received == 0 and time.perf_counter() < deadline:
            robot.get_observation(); time.sleep(0.01)
        if robot.msgs_received == 0:
            raise RuntimeError("no fresh observation from host — refusing to ramp")
        cur = {n: float(robot.get_observation()[n]) for n in target}
        n_steps = max(1, int(seconds * 30))
        for i in range(1, n_steps + 1):
            w = i / n_steps
            robot.send_action({k: cur[k] + w * (target[k] - cur[k]) for k in target})
            precise_sleep(1.0 / 30)

    script = Path(__file__).resolve().parent / "replay_measure_so101.py"
    for ep in args.episodes:
        print(f"\n##### episode {ep}: ramping to start pose #####", flush=True)
        robot = make_robot()
        ramp_to(robot, starts[ep], args.ramp_seconds)
        robot.disconnect()  # free the obs stream for replay_measure
        time.sleep(0.5)
        r = subprocess.run([sys.executable, str(script),
                            "--dataset", args.dataset, "--episode", str(ep),
                            "--remote_ip", args.remote_ip,
                            "--log_dir", str(log_dir)],
                           capture_output=True, text=True)
        tail = "\n".join(r.stdout.splitlines()[-14:])
        print(tail, flush=True)
        if r.returncode != 0:
            print(f"episode {ep} FAILED:\n{r.stderr[-1200:]}", flush=True)

    # held-pose hunting check
    print("\n##### held-pose check #####", flush=True)
    robot = make_robot()
    ramp_to(robot, hold_pose, args.ramp_seconds)
    time.sleep(1.0)
    names = list(hold_pose.keys())
    rows = []
    m0 = robot.msgs_received
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < HOLD_SECONDS:
        o = robot.get_observation()
        rows.append([time.perf_counter() - t0] + [float(o[n]) for n in names])
        precise_sleep(1.0 / 30)
    fresh = robot.msgs_received - m0
    print(f"fresh observations during hold: {fresh} ({fresh / HOLD_SECONDS:.0f} Hz)"
          + ("  << STALE STREAM, hold check invalid" if fresh < 100 else ""), flush=True)
    arr = np.array(rows)
    obs = arr[:, 1:]
    drift = obs - obs[:1]
    pkpk = (obs.max(0) - obs.min(0))
    print(f"hold {HOLD_SECONDS:.0f}s: per-joint pk-pk " +
          " ".join(f"{n.split('.')[0]}={v:.2f}" for n, v in zip(names, pkpk)), flush=True)
    print(f"worst pk-pk {pkpk.max():.2f} units | worst drift-from-start "
          f"{np.abs(drift).max():.2f} units", flush=True)
    np.savez_compressed(log_dir / "hold_check.npz", rows=arr.astype(np.float32),
                        columns=np.array(["t"] + names))
    robot.disconnect()
    print("SETTING DONE", flush=True)


if __name__ == "__main__":
    main()
