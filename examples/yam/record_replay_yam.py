#!/usr/bin/env python3
"""Record a teleoperated YAM trajectory, replay it N times, and measure
tracking fidelity + replay-to-replay repeatability.

This is the first hardware characterization for the QDD chapter: on the
SO-101, backlash put millimeters of slack between identical commands and
the resulting pose, which became the 27% same-state grasp flip floor. The
replay-to-replay spread this tool measures is the same question asked in
joint space, camera-free.

Modes:
  record   — host in teleop mode (leader hand-driven, bilateral); samples
             measured joints + commanded targets at --rate; ENTER starts,
             ENTER stops; saves an .npz
  replay   — host in policy mode; ramps to the trajectory start, streams
             the recorded TARGETS at the recorded rate, N times; reports
             per-joint tracking error, command->measured lag, and
             replay-to-replay repeatability
  selftest — synthesizes a small sine trajectory around the current pose
             and runs the replay machinery (works against --sim host)

Usage (host first, in the i2rt venv):
  python yam_zmq_host.py --leader_channel can1 --follower_channel can0
Then:
  python record_replay_yam.py --mode record --file traj1.npz
  python record_replay_yam.py --mode replay --file traj1.npz --replays 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alohamini"))
from record_so101 import FocusKeyboard  # noqa: E402
from yam_zmq_client import YamZmqClient, YamZmqClientConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YAM record/replay characterization")
    p.add_argument("--mode", choices=["record", "replay", "selftest"], required=True)
    p.add_argument("--file", type=str, default="yam_traj.npz")
    p.add_argument("--remote_ip", type=str, default="127.0.0.1")
    p.add_argument("--rate", type=float, default=30.0, help="sample/replay Hz")
    p.add_argument("--replays", type=int, default=3)
    p.add_argument("--home_wait", type=float, default=3.0,
                   help="seconds to wait after ramping to the start pose")
    return p.parse_args()


def make_client(ip: str) -> YamZmqClient:
    cfg = YamZmqClientConfig(remote_ip=ip, cameras={})  # camera-free
    client = YamZmqClient(cfg)
    client.connect()
    return client


def obs_vecs(client: YamZmqClient, obs: dict) -> tuple[np.ndarray, np.ndarray]:
    names = client.config.joint_names
    meas = np.array([obs[n] for n in names], dtype=np.float64)
    tgt = np.array([obs.get(f"target.{n}", obs[n]) for n in names], dtype=np.float64)
    return meas, tgt


def record(args: argparse.Namespace) -> None:
    client = make_client(args.remote_ip)
    client.set_mode("teleop")
    kb = FocusKeyboard(); kb.start()
    print("Teleop live (bilateral). ENTER to START recording…", flush=True)
    try:
        while "\r" not in kb.get_pressed() and "\n" not in kb.get_pressed():
            client.get_observation()
            time.sleep(0.05)
        kb.clear()
        print("RECORDING — ENTER to stop.", flush=True)
        meas_log, tgt_log, t_log = [], [], []
        dt = 1.0 / args.rate
        t0 = time.perf_counter()
        while True:
            tick = time.perf_counter()
            pressed = kb.get_pressed()
            if "\r" in pressed or "\n" in pressed:
                break
            obs = client.get_observation()
            meas, tgt = obs_vecs(client, obs)
            meas_log.append(meas); tgt_log.append(tgt)
            t_log.append(tick - t0)
            time.sleep(max(0.0, dt - (time.perf_counter() - tick)))
    finally:
        kb.stop()
        client.disconnect()
    np.savez(args.file, measured=np.array(meas_log), target=np.array(tgt_log),
             t=np.array(t_log), rate=args.rate,
             joint_names=np.array(client.config.joint_names))
    dur = t_log[-1] if t_log else 0.0
    print(f"saved {len(meas_log)} frames ({dur:.1f}s @ {args.rate}Hz) -> {args.file}")


def run_one_replay(client: YamZmqClient, targets: np.ndarray, rate: float,
                   home_wait: float) -> np.ndarray:
    names = client.config.joint_names
    client.go_home({n: float(targets[0][i]) for i, n in enumerate(names)})
    t_end = time.time() + home_wait
    while time.time() < t_end:
        client.get_observation(); time.sleep(0.03)
    meas_log = []
    dt = 1.0 / rate
    for row in targets:
        tick = time.perf_counter()
        client.send_action({n: float(row[i]) for i, n in enumerate(names)})
        obs = client.get_observation()
        meas, _ = obs_vecs(client, obs)
        meas_log.append(meas)
        time.sleep(max(0.0, dt - (time.perf_counter() - tick)))
    return np.array(meas_log)


def lag_ticks(cmd: np.ndarray, meas: np.ndarray, max_lag: int = 20) -> int:
    """Cross-correlation lag on the most-traveled joint."""
    j = int(np.argmax(np.abs(np.diff(cmd, axis=0)).sum(axis=0)))
    c = cmd[:, j] - cmd[:, j].mean()
    m = meas[:, j] - meas[:, j].mean()
    best, best_lag = -np.inf, 0
    for lag in range(0, max_lag + 1):
        if lag >= len(c) - 2:
            break
        score = float(np.dot(c[: len(c) - lag], m[lag:]))
        if score > best:
            best, best_lag = score, lag
    return best_lag


def report(tag: str, err: np.ndarray, names: tuple[str, ...]) -> None:
    print(f"  {tag}: RMSE per joint (rad):")
    rmse = np.sqrt((err ** 2).mean(axis=0))
    for n, v in zip(names, rmse):
        print(f"    {n:12s} {v:.4f}")
    print(f"    max abs anywhere: {np.abs(err).max():.4f} rad")


def replay(args: argparse.Namespace, targets: np.ndarray | None = None,
           reference: np.ndarray | None = None, rate: float | None = None) -> None:
    client = make_client(args.remote_ip)
    names = client.config.joint_names
    if targets is None:
        data = np.load(args.file, allow_pickle=True)
        targets, reference, rate = data["target"], data["measured"], float(data["rate"])
    runs = []
    try:
        for i in range(args.replays):
            print(f"replay {i + 1}/{args.replays} ({len(targets)} frames)…", flush=True)
            runs.append(run_one_replay(client, targets, rate, args.home_wait))
    finally:
        client.disconnect()

    print("\n===== RESULTS =====")
    lag = lag_ticks(targets, runs[0])
    print(f"command->measured lag: {lag} ticks ({1000.0 * lag / rate:.0f} ms @ {rate:.0f}Hz)")
    L = min(len(r) for r in runs)
    for i, run in enumerate(runs):
        report(f"replay {i + 1} vs commanded targets", run[:L] - targets[:L], names)
    if reference is not None:
        report("replay 1 vs original recording (measured)",
               runs[0][:L] - reference[:L], names)
    if len(runs) > 1:
        print("\n  REPLAY-TO-REPLAY repeatability (the flip-floor-relevant number):")
        for i in range(1, len(runs)):
            err = runs[i][:L] - runs[0][:L]
            rmse = np.sqrt((err ** 2).mean())
            print(f"    replay {i + 1} vs replay 1: overall RMSE {rmse:.5f} rad, "
                  f"max {np.abs(err).max():.5f} rad")
    stem = Path(args.file).stem
    out = f"{stem}_replays.npz"
    np.savez(out, targets=targets, runs=np.array([r[:L] for r in runs]),
             reference=reference[:L] if reference is not None else np.zeros(0))
    print(f"\nsaved raw runs -> {out}")


def selftest(args: argparse.Namespace) -> None:
    client = make_client(args.remote_ip)
    names = client.config.joint_names
    obs = client.get_observation()
    start, _ = obs_vecs(client, obs)
    client.disconnect()
    T = int(args.rate * 6)
    t = np.linspace(0, 2 * np.pi, T)
    targets = np.tile(start, (T, 1))
    targets[:, 1] += 0.25 * np.sin(t)          # joint2 sine
    targets[:, 3] += 0.15 * np.sin(2 * t)      # joint4 faster sine
    args.replays = max(2, args.replays)
    replay(args, targets=targets, reference=None, rate=args.rate)


def main() -> None:
    args = parse_args()
    if args.mode == "record":
        record(args)
    elif args.mode == "replay":
        replay(args)
    else:
        selftest(args)


if __name__ == "__main__":
    main()
