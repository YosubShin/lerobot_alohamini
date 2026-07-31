#!/usr/bin/env python3
"""Record LeRobot training episodes on one SO-101 follower via LEADER-ARM TELEOP.

Teleop counterpart of record_so101.py (kinesthetic), sharing its guards and
plumbing: pre-flight + in-episode obs-rate and frozen-camera watchdogs with
glitch-retry, Enter/r/q terminal controls, live obs-rate display, wall-clock
sidecar (meta/episode_wallclock.jsonl), session-gated 1080p fisheye archive,
and automatic post-session per-episode 1080p extraction.

The key difference from kinesthetic recording: **action = leader pose**, not
the follower's observed state — the follower lags the leader by the tracking
error, so the recorded (obs, action) pairs carry a natural plant lead and
need no k-frame shift at training time.

Setup: leader arm plugs into THIS machine (workstation); the follower is
driven through so101_zmq_host.py on the Pi (low command latency over
ethernet). The follower ramps to the leader's pose at start (no snap), then
follows continuously — including between episodes, so you reposition
naturally; only in-episode ticks are recorded.

Controls (same as kinesthetic):
    Enter  finish + save episode  |  between episodes: start the next
    r      discard current episode and redo
    q      quit session
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from lerobot.common.control_utils import init_keyboard_listener
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.teleoperators.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_so101 import (  # noqa: E402
    _MIN_OBS_RATE_FRACTION,
    FocusKeyboard,
    _apply_stdin_controls,
    _extract_session_1080p,
    _frozen_cam_banner,
    _log_episode_wallclock,
    _measure_obs_rate,
    _obs_rate_banner,
    _ramp_to_pose,
    parse_bool,
)
from so101_zmq_client import SO101ZmqClient, SO101ZmqClientConfig  # noqa: E402

_DEFAULT_REMOTE_IP = "192.168.0.17"  # Pi ethernet (wlan fallback: .50)
_DEFAULT_LEADER_PORT = "/dev/am_arm_leader_right"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teleop-record episodes on one SO-101 follower")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--num_episodes", type=int, default=1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episode_time", type=int, default=60)
    p.add_argument("--task_description", type=str, default="SO-101 teleop demonstration")
    p.add_argument("--remote_ip", type=str, default=_DEFAULT_REMOTE_IP)
    p.add_argument("--zmq_cmd_port", type=int, default=5601)
    p.add_argument("--zmq_obs_port", type=int, default=5602)
    p.add_argument("--leader_port", type=str, default=_DEFAULT_LEADER_PORT)
    p.add_argument("--leader_id", type=str, default="am_leader_right")
    p.add_argument(
        "--leader_profile",
        type=str,
        default="so-arm-5dof",
        choices=["so-arm-5dof", "am-leader-6dof"],
    )
    p.add_argument("--cam_width", type=int, default=320)
    p.add_argument("--cam_height", type=int, default=240)
    p.add_argument("--no_cameras", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--push_to_hub", type=parse_bool, nargs="?", const=True, default=False)
    p.add_argument("--archive_dir", type=str, default="/mnt/nvme/lerobot/fisheye_archive")
    p.add_argument("--remote_user", type=str, default="yosub")
    p.add_argument("--no_extract_1080p", action="store_true")
    p.add_argument(
        "--follow_ramp_s",
        type=float,
        default=3.0,
        help="Seconds to ramp the follower to the leader's pose at startup",
    )
    return p.parse_args()


def _wait_for_next_episode_following(kb, events, robot, leader, fps, prompt=None) -> bool:
    """Between episodes: keep the follower tracking the leader while waiting
    for Enter. Returns False on quit."""
    kb.clear()
    time.sleep(0.2)
    kb.clear()
    if prompt is None:
        prompt = "RESET THE SCENE (follower keeps following), then Enter for the next episode"
    print("\n" + "▒" * 74)
    print(f"▒▒  ⏸  {prompt}")
    print("▒▒       q = quit session")
    print("▒" * 74 + "\n", flush=True)
    log_say("Press Enter when ready")
    interval = 1.0 / fps
    while not events["stop_recording"]:
        t0 = time.perf_counter()
        try:
            robot.send_action(leader.get_action())
            robot.get_observation()
        except Exception:
            pass
        pressed = kb.get_pressed()
        if "q" in pressed:
            events["stop_recording"] = True
            kb.clear()
            print("\n[q] quit.")
            return False
        if "\r" in pressed or "\n" in pressed:
            kb.clear()
            print("\n[Enter] starting episode…", flush=True)
            time.sleep(0.3)
            kb.clear()
            return True
        precise_sleep(max(interval - (time.perf_counter() - t0), 0.0))
    return False


def teleop_record_loop(*, robot, leader, events, fps, control_time_s, dataset,
                       single_task, kb) -> None:
    import numpy as np

    interval = 1.0 / fps
    start_t = time.perf_counter()
    frame_i = 0
    rate_check_every = max(int(fps * 3), 1)
    msgs0 = getattr(robot, "msgs_received", None)
    frozen_limit = int(fps * 1.0)
    prev_frames: dict[str, np.ndarray] = {}
    frozen_run: dict[str, int] = {}

    while (time.perf_counter() - start_t) < control_time_s:
        if kb is not None:
            _apply_stdin_controls(kb, events)
        if events.get("exit_early"):
            events["exit_early"] = False
            break
        if events.get("stop_recording") or events.get("rerecord_episode"):
            break

        t0 = time.perf_counter()
        action = leader.get_action()          # THE teleop difference:
        robot.send_action(action)             # action = leader pose, sent as goal
        obs = robot.get_observation()

        for cam, frame in obs.items():
            if not (isinstance(frame, np.ndarray) and frame.ndim == 3):
                continue
            if cam in prev_frames and np.array_equal(frame, prev_frames[cam]):
                frozen_run[cam] = frozen_run.get(cam, 0) + 1
            else:
                frozen_run[cam] = 0
                prev_frames[cam] = frame.copy()
        stuck = [c for c, n in frozen_run.items() if n >= frozen_limit]
        if stuck:
            print(_frozen_cam_banner(stuck), flush=True)
            print("Episode discarded — will wait for recovery and let you redo it.", flush=True)
            events["rerecord_episode"] = True
            events["glitch_abort"] = True
            break

        if dataset is not None:
            oframe = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            aframe = build_dataset_frame(dataset.features, action, prefix=ACTION)
            dataset.add_frame({**oframe, **aframe, "task": single_task})

        frame_i += 1
        if frame_i % rate_check_every == 0:
            elapsed = time.perf_counter() - start_t
            loop_hz = frame_i / elapsed
            obs_hz = loop_hz if msgs0 is None else (robot.msgs_received - msgs0) / elapsed
            if min(loop_hz, obs_hz) < fps * _MIN_OBS_RATE_FRACTION:
                print(_obs_rate_banner(min(loop_hz, obs_hz), fps * _MIN_OBS_RATE_FRACTION), flush=True)
                print("Episode discarded — will wait for recovery and let you redo it.", flush=True)
                events["rerecord_episode"] = True
                events["glitch_abort"] = True
                break
        if frame_i % 10 == 0:
            elapsed = time.perf_counter() - start_t
            obs_hz = "" if msgs0 is None else f"  obs={(robot.msgs_received - msgs0) / elapsed:4.1f}Hz"
            print(f"\r  recorded frames={frame_i}  t={elapsed:5.1f}s/{control_time_s:.0f}s{obs_hz}   ",
                  end="", flush=True)

        precise_sleep(max(interval - (time.perf_counter() - t0), 0.0))
    if frame_i:
        print()


def main() -> None:
    args = parse_args()

    cam_shapes = {}
    if not args.no_cameras:
        cam_shapes = {
            "forward": (args.cam_height, args.cam_width, 3),
            "wrist": (args.cam_height, args.cam_width, 3),
        }
    robot = SO101ZmqClient(SO101ZmqClientConfig(
        remote_ip=args.remote_ip, zmq_cmd_port=args.zmq_cmd_port,
        zmq_obs_port=args.zmq_obs_port, cameras=cam_shapes))
    leader = SOLeader(SOLeaderTeleopConfig(
        port=args.leader_port, id=args.leader_id, arm_profile=args.leader_profile))

    use_videos = bool(cam_shapes)
    action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=use_videos)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=use_videos)
    dataset_features = {**action_features, **obs_features}

    if args.resume:
        print("Resuming existing dataset:", args.dataset)
        dataset = LeRobotDataset.resume(
            repo_id=args.dataset, root=HF_LEROBOT_HOME / args.dataset, image_writer_threads=4)
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.dataset, fps=args.fps, features=dataset_features,
            robot_type=robot.name, use_videos=use_videos, image_writer_threads=4)
        print(f"Dataset created with id: {dataset.repo_id}")
    print(f"Local dataset path: {dataset.root.resolve()}")

    robot.connect()
    robot.archive_start()
    time.sleep(3.0)

    print("Pre-flight: measuring host observation rate (3 s)…", flush=True)
    obs_hz, frozen = _measure_obs_rate(robot)
    needed_hz = args.fps * _MIN_OBS_RATE_FRACTION
    if (obs_hz is not None and obs_hz < needed_hz) or frozen:
        if obs_hz is not None and obs_hz < needed_hz:
            print(_obs_rate_banner(obs_hz, needed_hz), flush=True)
        if frozen:
            print(_frozen_cam_banner(frozen), flush=True)
        try:
            robot.archive_stop()
            robot.disconnect()
        except Exception:
            pass
        raise SystemExit(1)
    print(f"Observation rate OK: {obs_hz:.1f} Hz (need >= {needed_hz:.1f}); cameras live: all", flush=True)

    leader.connect()
    print("\nHold the leader arm roughly where the follower should start.")
    input("Press ENTER to ramp the follower to the leader's pose…")
    _ramp_to_pose(robot, leader.get_action(), args.follow_ramp_s)
    print("Follower is now tracking the leader.", flush=True)

    listener, events = init_keyboard_listener()
    kb = FocusKeyboard()
    kb.start()

    print("\nTeleop record ready — the follower mirrors the leader continuously.")
    print("  CONTROLS: Enter=save/start   r=discard+redo   q=quit")
    print("  Flow: Enter(start) → record → Enter(save) → reset scene → Enter(next)\n", flush=True)

    recorded_episodes = 0
    try:
        if not _wait_for_next_episode_following(
                kb, events, robot, leader, args.fps,
                prompt="Position leader at the start pose, then Enter for the first episode"):
            return
        while recorded_episodes < args.num_episodes and not events["stop_recording"]:
            print("\n" + "█" * 74)
            print(f"██  ▶  RECORDING episode {recorded_episodes + 1}/{args.num_episodes}"
                  f"      Enter=save   r=discard   q=quit")
            print("█" * 74 + "\n", flush=True)
            log_say(f"Recording episode {recorded_episodes + 1} of {args.num_episodes}")
            events["exit_early"] = False
            events["rerecord_episode"] = False
            kb.clear()
            time.sleep(0.2)
            kb.clear()

            ep_start_unix = time.time()
            teleop_record_loop(robot=robot, leader=leader, events=events, fps=args.fps,
                               control_time_s=args.episode_time, dataset=dataset,
                               single_task=args.task_description, kb=kb)
            ep_end_unix = time.time()

            if events["stop_recording"]:
                dataset.clear_episode_buffer()
                print("Quit — in-progress episode discarded (not saved).")
                break

            if events["rerecord_episode"]:
                log_say("Discarded episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                print("Episode discarded (not saved).")
                if events.pop("glitch_abort", False):
                    print("Waiting for the stream to recover (follower keeps following)…", flush=True)
                    while True:
                        try:
                            robot.send_action(leader.get_action())
                        except Exception:
                            pass
                        hz, frz = _measure_obs_rate(robot, seconds=2.0)
                        if (hz is None or hz >= needed_hz) and not frz:
                            print(f"Recovered: {hz:.1f} Hz, cameras live.", flush=True)
                            break
                        print(f"  not yet (rate={hz and f'{hz:.1f}'} Hz, frozen={frz or 'none'})"
                              f" — retrying in 3 s", flush=True)
                        time.sleep(3)
                    # A protective trip (e.g. overcurrent) reconnects the robot
                    # with torque OFF — re-latch by ramping to the leader pose.
                    print("Re-engaging follower (ramp to leader pose)…", flush=True)
                    _ramp_to_pose(robot, leader.get_action(), 2.0)
                if not _wait_for_next_episode_following(kb, events, robot, leader, args.fps):
                    break
                continue

            if not dataset.has_pending_frames():
                print("No frames captured — not saving.")
                dataset.clear_episode_buffer()
                if not _wait_for_next_episode_following(kb, events, robot, leader, args.fps):
                    break
                continue

            print("\n… saving + encoding episode …", flush=True)
            dataset.save_episode()
            _log_episode_wallclock(dataset, ep_start_unix, ep_end_unix)
            recorded_episodes += 1
            print(f"\n✔✔✔  SAVED  —  {recorded_episodes}/{args.num_episodes} episodes\n", flush=True)

            if recorded_episodes < args.num_episodes and not events["stop_recording"]:
                if not _wait_for_next_episode_following(kb, events, robot, leader, args.fps):
                    break
    finally:
        log_say("Stop recording")
        kb.stop()
        try:
            robot.archive_stop()
            time.sleep(0.3)
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception as e:
            print(f"(robot disconnect warning: {e})")
        try:
            leader.disconnect()
        except Exception:
            pass
        if listener is not None:
            listener.stop()
        dataset.finalize()

    print(f"Dataset saved locally at: {dataset.root.resolve()}")
    if recorded_episodes and not args.no_extract_1080p:
        _extract_session_1080p(args, dataset.root)
    if args.push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
