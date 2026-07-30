#!/usr/bin/env python
"""Single SO-101 follower: kinesthetic trajectory record + faithful replay.

Unlike leader-arm teleop, you move the *follower* by hand (torque disabled while
recording). Positions are sampled at --fps and saved to a JSON trajectory file.
Replay re-enables torque and streams the same Goal_Position sequence back.

Cameras are intentionally unused here — use record_so101.py / replay_so101.py
for LeRobot episode datasets (wrist + base cams).

Examples:
  # Record (torque off — guide the arm by hand; press 'q' to stop & save)
  uv run python examples/alohamini/record_replay_so101.py record \\
    --port /dev/ttyACM0 --robot_id my_follower \\
    --out outputs/traj_so101.json

  # Replay the saved trajectory
  uv run python examples/alohamini/record_replay_so101.py replay \\
    --port /dev/ttyACM0 --robot_id my_follower \\
    --traj outputs/traj_so101.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import termios
import time
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep


# ---------------- focus-bound keyboard (stdin) ----------------
class FocusKeyboard:
    """Reads keys from this terminal's stdin: only active while the terminal is focused."""

    def __init__(self, active_timeout: float = 0.35):
        self.active_timeout = active_timeout
        self.last_seen: dict[str, float] = {}
        self.fd = sys.stdin.fileno()
        self._saved = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            # Headless / piped runs (e.g. ssh without a real TTY): no key control.
            self._saved = None
            return
        self._saved = termios.tcgetattr(self.fd)
        new = termios.tcgetattr(self.fd)
        new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
        new[6][termios.VMIN] = 0
        new[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, new)
        os.set_blocking(self.fd, False)

    def stop(self) -> None:
        if self._saved is not None:
            os.set_blocking(self.fd, True)
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def get_pressed(self) -> set[str]:
        if self._saved is None and not sys.stdin.isatty():
            return set()
        now = time.monotonic()
        try:
            data = os.read(self.fd, 4096)
        except (BlockingIOError, OSError):
            data = b""
        for ch in data.decode("utf-8", "ignore"):
            self.last_seen[ch] = now
        return {ch for ch, t in self.last_seen.items() if now - t <= self.active_timeout}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kinesthetic record / replay of a single SO-101 follower trajectory"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--port", type=str, default="/dev/ttyACM0", help="Follower USB serial port")
    common.add_argument("--robot_id", type=str, default="my_so101_follower", help="Calibration id")
    common.add_argument(
        "--arm_profile",
        type=str,
        default="so-arm-5dof",
        choices=["so-arm-5dof", "am-follower-6dof", "am-follower-6dof-hd"],
    )
    common.add_argument("--fps", type=int, default=30)
    common.add_argument(
        "--key_timeout",
        type=float,
        default=0.35,
        help="Seconds a key stays 'active' after its last stdin event",
    )

    rec = sub.add_parser("record", parents=[common], help="Disable torque, move arm by hand, save JSON")
    rec.add_argument(
        "--out",
        type=str,
        default="outputs/traj_so101.json",
        help="Output trajectory JSON path",
    )
    rec.add_argument(
        "--max_seconds",
        type=float,
        default=0.0,
        help="Optional auto-stop after N seconds (0 = until 'q')",
    )

    rep = sub.add_parser("replay", parents=[common], help="Enable torque and replay a saved JSON")
    rep.add_argument(
        "--traj",
        type=str,
        default="outputs/traj_so101.json",
        help="Trajectory JSON to replay",
    )
    rep.add_argument(
        "--loop",
        action="store_true",
        help="Replay in a loop until 'q' or Ctrl+C",
    )
    rep.add_argument(
        "--settle_s",
        type=float,
        default=1.0,
        help="Seconds to hold the first pose before streaming the rest",
    )

    return parser.parse_args()


def _make_robot(args: argparse.Namespace) -> SO101Follower:
    # No cameras: this script is joints-only.
    cfg = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        arm_profile=args.arm_profile,
        cameras={},
    )
    return SO101Follower(cfg)


def _read_joint_action(robot: SO101Follower) -> dict[str, float]:
    """Read Present_Position without going through get_observation (skips current spam / cams)."""
    pos = robot.bus.sync_read("Present_Position")
    return {f"{motor}.pos": float(val) for motor, val in pos.items()}


def record(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    robot = _make_robot(args)
    kb = FocusKeyboard(active_timeout=args.key_timeout)

    robot.connect()
    # configure() ends with torque enabled — turn it off so the arm is free to move.
    robot.bus.disable_torque()
    print("\nTorque DISABLED — guide the follower by hand.")
    if args.max_seconds > 0:
        print(f"Recording for up to {args.max_seconds:g}s (or press 'q' if this is a TTY).\n")
    else:
        print("Recording joint positions. Press 'q' to stop and save.")
        if not sys.stdin.isatty():
            raise RuntimeError(
                "stdin is not a TTY and --max_seconds was not set — "
                "cannot stop recording. Pass --max_seconds N for headless runs."
            )
        print()

    frames: list[dict[str, float]] = []
    kb.start()
    t_start = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            pressed = kb.get_pressed()
            if "q" in pressed:
                print("\n'q' pressed — stopping record.")
                break
            if args.max_seconds > 0 and (t0 - t_start) >= args.max_seconds:
                print(f"\nReached --max_seconds={args.max_seconds:g} — stopping record.")
                break

            action = _read_joint_action(robot)
            frames.append(action)

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
            loop_dt = time.perf_counter() - t0
            loop_fps = 1.0 / loop_dt if loop_dt > 0 else float("inf")
            if len(frames) % 5 == 0:
                warn = "  ⚠️LOW-FPS" if loop_fps < args.fps * 0.7 else ""
                sys.stdout.write(f"\rframes={len(frames):5d}  fps={loop_fps:5.1f}{warn}   ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nCtrl+C — stopping record.")
    finally:
        kb.stop()
        try:
            robot.disconnect()  # leaves torque disabled (config default)
        except Exception as e:
            print(f"(cleanup warning: {e})")

    if not frames:
        print("No frames recorded — nothing saved.")
        return

    payload = {
        "fps": args.fps,
        "robot_id": args.robot_id,
        "arm_profile": args.arm_profile,
        "port": args.port,
        "joint_names": list(frames[0].keys()),
        "num_frames": len(frames),
        "duration_s": len(frames) / float(args.fps),
        "frames": frames,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"\nSaved {len(frames)} frames ({payload['duration_s']:.2f}s @ {args.fps} fps) → {out_path.resolve()}"
    )
    print(f"Replay with:\n  uv run python examples/alohamini/record_replay_so101.py replay --traj {out_path}")


def replay(args: argparse.Namespace) -> None:
    traj_path = Path(args.traj)
    if not traj_path.is_file():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    data = json.loads(traj_path.read_text())
    frames: list[dict[str, float]] = data["frames"]
    fps = int(data.get("fps", args.fps))
    if not frames:
        raise ValueError(f"Trajectory has no frames: {traj_path}")

    robot = _make_robot(args)
    kb = FocusKeyboard(active_timeout=args.key_timeout)

    robot.connect()
    # Torque is on after configure(); keep it on for position tracking.
    print(f"\nReplaying {len(frames)} frames @ {fps} fps from {traj_path}")
    print("Press 'q' to abort (TTY only). Arm will move — clear the workspace.\n")
    if sys.stdin.isatty():
        input("Press ENTER to move to the start pose and begin replay…")
    else:
        print("Non-interactive stdin — starting replay immediately.")

    # Snap / settle on the first pose before streaming.
    robot.send_action(frames[0])
    time.sleep(max(args.settle_s, 0.0))

    kb.start()
    pass_idx = 0
    try:
        while True:
            pass_idx += 1
            print(f"— pass {pass_idx} —")
            for i, action in enumerate(frames):
                t0 = time.perf_counter()
                if "q" in kb.get_pressed():
                    print("\n'q' pressed — aborting replay.")
                    return
                robot.send_action(action)
                precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
                if (i + 1) % 5 == 0 or i + 1 == len(frames):
                    sys.stdout.write(f"\rframe {i + 1}/{len(frames)}   ")
                    sys.stdout.flush()
            print()
            if not args.loop:
                break
            if not sys.stdin.isatty():
                print("Non-interactive stdin — not looping.")
                break
            print("Looping… (press 'q' to stop)")
    except KeyboardInterrupt:
        print("\nCtrl+C — aborting replay.")
    finally:
        kb.stop()
        try:
            robot.disconnect()
        except Exception as e:
            print(f"(cleanup warning: {e})")
        print("Replay stopped.")


def main() -> None:
    args = _parse_args()
    if args.mode == "record":
        record(args)
    elif args.mode == "replay":
        replay(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
