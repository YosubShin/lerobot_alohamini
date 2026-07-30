#!/usr/bin/env python3
"""Replay a LeRobot dataset episode on one SO-101 follower.

Counterpart to replay_bi.py. Prefer remote mode so the dataset stays on the
workstation and the Pi only runs so101_zmq_host.py.

Example:
  # Pi
  sudo systemctl stop alohamini-host.service
  python examples/alohamini/so101_zmq_host.py

  # Workstation
  python examples/alohamini/replay_so101.py \\
    --remote_ip 192.168.0.50 \\
    --dataset local/so101_kinesthetic_pick --episode 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_zmq_client import SO101ZmqClient, SO101ZmqClientConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a LeRobot episode on one SO-101 follower")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset repo_id")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument(
        "--remote_ip",
        type=str,
        default=None,
        help="If set, replay through so101_zmq_host.py on the Pi",
    )
    parser.add_argument("--zmq_cmd_port", type=int, default=5601)
    parser.add_argument("--zmq_obs_port", type=int, default=5602)
    parser.add_argument("--port", type=str, default="/dev/am_arm_follower_right")
    parser.add_argument("--robot_id", type=str, default="am_follower_right")
    parser.add_argument(
        "--arm_profile",
        type=str,
        default="so-arm-5dof",
        choices=["so-arm-5dof", "am-follower-6dof", "am-follower-6dof-hd"],
    )
    parser.add_argument(
        "--settle_s",
        type=float,
        default=1.0,
        help="Seconds to hold the first pose before streaming",
    )
    return parser.parse_args()


def _make_robot(args: argparse.Namespace):
    if args.remote_ip:
        return SO101ZmqClient(
            SO101ZmqClientConfig(
                remote_ip=args.remote_ip,
                zmq_cmd_port=args.zmq_cmd_port,
                zmq_obs_port=args.zmq_obs_port,
                cameras={},  # replay does not need images
            )
        )
    return SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            arm_profile=args.arm_profile,
            cameras={},
        )
    )


def main() -> None:
    args = parse_args()
    robot = _make_robot(args)
    dataset = LeRobotDataset(args.dataset, episodes=[args.episode])
    actions = dataset.hf_dataset.select_columns(ACTION)

    robot.connect()
    if hasattr(robot, "enable_torque"):
        robot.enable_torque()

    print(
        f"Replaying episode {args.episode} from {args.dataset} "
        f"({dataset.num_frames} frames @ {dataset.fps} fps)"
        + (f" via {args.remote_ip}" if args.remote_ip else " (direct USB)")
    )
    log_say(f"Replaying episode {args.episode}")
    if sys.stdin.isatty():
        input("Press ENTER to move to the start pose and begin replay…")
    else:
        print("Non-interactive stdin — starting replay immediately.")

    try:
        first = {
            name: float(actions[0][ACTION][i])
            for i, name in enumerate(dataset.features[ACTION]["names"])
        }
        robot.send_action(first)
        time.sleep(max(args.settle_s, 0.0))

        for idx in range(dataset.num_frames):
            t0 = time.perf_counter()
            action = {
                name: float(actions[idx][ACTION][i])
                for i, name in enumerate(dataset.features[ACTION]["names"])
            }
            robot.send_action(action)
            precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()
        print("Replay done.")


if __name__ == "__main__":
    main()
