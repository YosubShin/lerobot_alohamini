#!/usr/bin/env python3
"""Replay a kinesthetic episode and measure how well the servo tracks it.

MOVES THE ARM. Ctrl-C e-stops (freeze; twice = torque off).

Why this exists
---------------
In kinesthetic data action[t] == observation.state[t] exactly: the recorded
"command" is just the pose a human physically put the arm in, so tracking error
is zero by construction. Replaying that same trajectory as *commands* isolates
the servo's own following error, with no policy, no action chunking, and no
splices in the loop. Whatever error shows up here is the floor that any policy
inherits.

It also separates two very different causes:
  * pure DELAY      -- obs[t+k] ~= cmd[t]. Fixable with lead/feedforward.
  * residual ERROR  -- error remains even at the best k. That is gain/stiction,
                       i.e. an actual PID problem.

Run the same episode before and after a P_Coefficient change to A/B it.

  python replay_measure_so101.py --dataset so101_red_pick_trim1x --episode 0 \
      --remote_ip 127.0.0.1 --log_dir ~/yosub/outputs/so101_replay_logs
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_zmq_client import SO101ZmqClient, SO101ZmqClientConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a recorded episode and measure tracking")
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--remote_ip", default="127.0.0.1")
    p.add_argument("--zmq_cmd_port", type=int, default=5601)
    p.add_argument("--zmq_obs_port", type=int, default=5602)
    p.add_argument("--fps", type=float, default=0.0, help="0 = use the dataset's fps")
    p.add_argument("--settle_s", type=float, default=1.5,
                   help="Hold the first pose this long before streaming")
    p.add_argument("--max_start_delta", type=float, default=15.0,
                   help="Refuse to start if any joint is farther than this from the "
                        "episode's first pose (0 disables). The move to the start pose "
                        "is a STEP at servo speed.")
    p.add_argument("--min_obs_hz", type=float, default=25.0,
                   help="Refuse to run below this fresh-observation rate (0 disables)")
    p.add_argument("--max_lag_ticks", type=int, default=12)
    p.add_argument("--log_dir", default="")
    p.add_argument("--dry_run", action="store_true",
                   help="Load, check the start pose, report — command no motion")
    return p.parse_args()


class EStop:
    """First Ctrl-C freezes at the observed pose; second disables torque."""

    def __init__(self, robot, names):
        self.robot, self.names, self.last_obs, self.tripped = robot, names, None, False

    def install(self):
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):  # noqa: ARG002
        if not self.tripped:
            self.tripped = True
            if self.last_obs:
                try:
                    for _ in range(5):
                        self.robot.send_action(dict(self.last_obs))
                        time.sleep(0.01)
                    print("\n*** E-STOP: frozen at observed pose (still torqued). "
                          "Ctrl-C again to go limp. ***", flush=True)
                except Exception as e:
                    print(f"\n*** E-STOP freeze failed: {e} ***", flush=True)
            else:
                print("\n*** E-STOP: no pose cached; Ctrl-C again for torque off ***",
                      flush=True)
            raise KeyboardInterrupt
        try:
            self.robot.disable_torque()
            print("\n*** TORQUE DISABLED — arm is LIMP. Support it. ***", flush=True)
        except Exception as e:
            print(f"\n*** disable_torque failed: {e} ***", flush=True)
        raise KeyboardInterrupt


def measure_obs_hz(robot, seconds=2.0) -> float:
    n0, t0 = robot.msgs_received, time.perf_counter()
    while (time.perf_counter() - t0) < seconds:
        robot.get_observation()
        time.sleep(0.002)
    return (robot.msgs_received - n0) / max(time.perf_counter() - t0, 1e-9)


def main() -> None:
    args = parse_args()

    ds = LeRobotDataset(args.dataset, episodes=[args.episode])
    names = list(ds.features[ACTION]["names"])
    acts = ds.hf_dataset.select_columns(ACTION)
    traj = np.array([[float(acts[i][ACTION][j]) for j in range(len(names))]
                     for i in range(ds.num_frames)], dtype=np.float64)
    fps = args.fps or float(ds.fps)
    print(f"{args.dataset} ep{args.episode}: {len(traj)} frames @ {fps:g} fps "
          f"= {len(traj) / fps:.1f}s")
    d = np.abs(np.diff(traj, axis=0)).max(axis=1)
    print(f"  commanded |d action|: median {np.median(d):.2f}  p95 "
          f"{np.percentile(d, 95):.2f}  max {d.max():.2f} units/tick")

    robot = SO101ZmqClient(SO101ZmqClientConfig(
        remote_ip=args.remote_ip, zmq_cmd_port=args.zmq_cmd_port,
        zmq_obs_port=args.zmq_obs_port, cameras={}))
    robot.connect()

    hz = measure_obs_hz(robot)
    print(f"  observation rate: {hz:.1f} Hz")
    if args.min_obs_hz > 0 and hz < args.min_obs_hz:
        robot.disconnect()
        raise SystemExit(
            f"Observation rate {hz:.1f} Hz < {args.min_obs_hz:.0f} Hz. Another ZMQ "
            f"consumer may be attached (check `ss -tnp | grep 5602` on the Pi).")

    cur = {k: float(v) for k, v in robot.get_observation().items() if k.endswith(".pos")}
    start = {n: traj[0][i] for i, n in enumerate(names)}
    delta = {n: start[n] - cur.get(n, np.nan) for n in names}
    worst = max(abs(v) for v in delta.values())
    print(f"\n  {'joint':<18}{'current':>9}{'ep start':>10}{'move':>9}")
    for n in names:
        print(f"  {n.split('.')[0]:<18}{cur.get(n, float('nan')):>9.1f}"
              f"{start[n]:>10.1f}{delta[n]:>+9.1f}")
    print(f"  max move to start: {worst:.1f} units (sent as a STEP)")

    if args.max_start_delta > 0 and worst > args.max_start_delta:
        robot.disconnect()
        raise SystemExit(
            f"Refusing: start pose is {worst:.1f} units away (> "
            f"--max_start_delta {args.max_start_delta:.0f}). Park the arm near the "
            f"episode start first, or raise the threshold deliberately.")

    if args.dry_run:
        robot.disconnect()
        print("\nDry run — no motion commanded.")
        return

    estop = EStop(robot, names)
    estop.last_obs = cur
    estop.install()
    print("\nE-STOP armed: Ctrl-C freezes, twice disables torque.")

    rows = []
    try:
        robot.enable_torque()
        robot.send_action(start)
        time.sleep(max(args.settle_s, 0.0))

        interval = 1.0 / fps
        t_start = time.perf_counter()
        for i in range(len(traj)):
            t0 = time.perf_counter()
            cmd = {n: traj[i][j] for j, n in enumerate(names)}
            robot.send_action(cmd)
            obs = {k: float(v) for k, v in robot.get_observation().items()
                   if k.endswith(".pos")}
            estop.last_obs = obs
            rows.append([time.perf_counter() - t_start]
                        + [cmd[n] for n in names]
                        + [obs.get(n, np.nan) for n in names])
            precise_sleep(max(interval - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("Replay aborted by e-stop.")
    finally:
        robot.disconnect()

    if len(rows) < 20:
        print("Too few samples to analyze.")
        return

    r = np.array(rows)
    nj = len(names)
    cmd_a, obs_a = r[:, 1:1 + nj], r[:, 1 + nj:1 + 2 * nj]
    elapsed = r[-1, 0] - r[0, 0]
    print(f"\n{'=' * 66}\nreplayed {len(r)} ticks in {elapsed:.1f}s "
          f"({(len(r) - 1) / elapsed:.1f} Hz)")

    inst = np.abs(obs_a - cmd_a).max(axis=1)
    print(f"\n  instantaneous |obs - cmd|: median {np.median(inst):.2f}  "
          f"p95 {np.percentile(inst, 95):.2f}  max {inst.max():.2f} units")

    # Pure delay vs residual: shift observations back by k and re-measure.
    print(f"\n  {'lag k':>6}{'ticks':>8}{'ms':>7}{'median |obs[t+k]-cmd[t]|':>28}")
    best_k, best_v = 0, None
    for k in range(args.max_lag_ticks + 1):
        if len(r) - k < 20:
            break
        v = float(np.median(np.abs(obs_a[k:] - cmd_a[:len(cmd_a) - k]).max(axis=1)))
        if best_v is None or v < best_v:
            best_k, best_v = k, v
        print(f"  {k:>6}{k:>8}{k / fps * 1000:>7.0f}{v:>28.3f}")

    print(f"\n  best alignment at k={best_k} ({best_k / fps * 1000:.0f} ms): "
          f"{best_v:.2f} units residual")
    explained = (1 - best_v / max(np.median(inst), 1e-9)) * 100
    print(f"  pure delay explains {explained:.0f}% of the tracking error")
    if explained > 70:
        print("  => mostly DELAY. Lead/feedforward compensation is the lever; "
              "raising P may add shakiness for little gain.")
    elif explained < 40:
        print("  => mostly RESIDUAL. A real gain/stiction problem — this is where "
              "P_Coefficient would help.")
    else:
        print("  => mixed delay and residual.")

    print(f"\n  {'joint':<18}{'median err':>12}{'max err':>10}")
    for j, n in enumerate(names):
        e = np.abs(obs_a[:, j] - cmd_a[:, j])
        print(f"  {n.split('.')[0]:<18}{np.median(e):>12.2f}{e.max():>10.2f}")

    if args.log_dir:
        out_dir = Path(args.log_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"replay_{time.strftime('%Y%m%d_%H%M%S')}_ep{args.episode}.npz"
        np.savez_compressed(
            out, rows=r.astype(np.float32),
            columns=np.array(["t"] + [f"cmd.{n}" for n in names]
                             + [f"obs.{n}" for n in names]),
            best_lag_ticks=np.array([best_k]), fps=np.array([fps]),
            config_json=np.array([json.dumps(vars(args), default=str)]))
        print(f"\n  log: {out}")

    print("\n  NOTE: arm is left at the episode's final pose — park it before "
          "the next run.")


if __name__ == "__main__":
    main()
