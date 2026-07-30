#!/usr/bin/env python3
"""Record LeRobot training episodes on one SO-101 follower (kinesthetic).

Mirror of record_bi.py for the single-arm embodiment:
  - No leader arm — torque is disabled; you guide the follower by hand.
  - Cameras: forehead/base (`forward`) + wrist.
  - action = observed joint positions (demonstration = pose you moved to).

Preferred setup (data lands on THIS machine):
  1) On the Pi — stop alohamini-host, start the ZMQ host:
       conda activate lerobot_alohamini
       sudo systemctl stop alohamini-host.service
       python examples/alohamini/so101_zmq_host.py

  2) On this workstation:
       conda activate lerobot_alohamini
       python examples/alohamini/record_so101.py \\
         --remote_ip 192.168.0.50 \\
         --dataset local/so101_kinesthetic_pick \\
         --num_episodes 20 \\
         --task_description "Pick the block and place it in the bin" \\
         --push_to_hub false

Direct USB mode (runs on the Pi, dataset stays on the Pi) still works if you
omit --remote_ip.

Controls:
  During an episode:
    Enter      finish + save episode
    r          discard current episode (do not save) and redo
    q          quit session (discards the in-progress episode)
  Between episodes (after home):
    Enter      start the next episode (after you reset the scene)
    q          quit session
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.control_utils import init_keyboard_listener
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Make `examples/alohamini/` importable when launched as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_zmq_client import SO101ZmqClient, SO101ZmqClientConfig  # noqa: E402

_DEFAULT_PORT = "/dev/am_arm_follower_right"
_DEFAULT_ROBOT_ID = "am_follower_right"
# Match so101_zmq_host.py: physical USB wiring is swapped vs udev names.
_DEFAULT_FORWARD_CAM = "/dev/am_camera_wrist_right"
_DEFAULT_WRIST_CAM = "/dev/am_camera_forward"
_DEFAULT_WRIST_FALLBACK = "/dev/video0"
_DEFAULT_REMOTE_IP = "192.168.0.50"
_HOME_POSE_PATH = Path(__file__).resolve().parent / "so101_home_pose.json"


class FocusKeyboard:
    """Terminal-focus stdin keys — works over SSH where pynput arrow keys often don't."""

    def __init__(self, active_timeout: float = 0.35):
        self.active_timeout = active_timeout
        self.last_seen: dict[str, float] = {}
        self.fd = None
        self._saved = None

    def start(self) -> None:
        import os
        import termios

        if not sys.stdin.isatty():
            return
        self.fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self.fd)
        new = termios.tcgetattr(self.fd)
        new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
        new[6][termios.VMIN] = 0
        new[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, new)
        os.set_blocking(self.fd, False)

    def stop(self) -> None:
        import os
        import termios

        if self._saved is not None and self.fd is not None:
            os.set_blocking(self.fd, True)
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def get_pressed(self) -> set[str]:
        import os

        if self._saved is None:
            return set()
        now = time.monotonic()
        try:
            data = os.read(self.fd, 4096)
        except (BlockingIOError, OSError):
            data = b""
        for ch in data.decode("utf-8", "ignore"):
            self.last_seen[ch] = now
        return {ch for ch, t in self.last_seen.items() if now - t <= self.active_timeout}

    def clear(self) -> None:
        """Drain stdin and forget active keys (prevents Enter from double-firing across phases)."""
        import os

        self.last_seen.clear()
        if self._saved is None or self.fd is None:
            return
        try:
            while True:
                data = os.read(self.fd, 4096)
                if not data:
                    break
        except (BlockingIOError, OSError):
            pass


def _apply_stdin_controls(kb: FocusKeyboard, events: dict) -> None:
    """Episode-loop keys: Enter=save, r=discard+redo, q=quit."""
    pressed = kb.get_pressed()
    if "\r" in pressed or "\n" in pressed:
        if not events["exit_early"]:
            print("\n[Enter] finish episode — saving…")
        events["exit_early"] = True
        kb.clear()
    if "r" in pressed:
        if not events["rerecord_episode"]:
            print("\n[r] discard current episode (not saved)…")
        events["rerecord_episode"] = True
        events["exit_early"] = True
        kb.clear()
    if "q" in pressed:
        if not events["stop_recording"]:
            print("\n[q] quit — discarding in-progress episode…")
        events["stop_recording"] = True
        events["exit_early"] = True
        kb.clear()


def _wait_for_next_episode(
    kb: FocusKeyboard,
    events: dict,
    robot,
    *,
    prompt: str | None = None,
) -> bool:
    """After home: wait for Enter before starting an episode.

    Returns True if we should start the next episode, False if quitting.
    Keeps polling observations so the remote stream stays warm.
    """
    kb.clear()
    time.sleep(0.2)
    kb.clear()

    if prompt is None:
        prompt = "RESET THE SCENE, then press  Enter  to start the next episode"

    print("\n" + "▒" * 74)
    print(f"▒▒  ⏸  {prompt}")
    print("▒▒       q = quit session")
    print("▒" * 74 + "\n", flush=True)
    log_say("Press Enter when ready")
    _disable_torque(robot)

    while not events["stop_recording"]:
        try:
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
            # Extra settle so the same Enter cannot immediately end the new episode.
            time.sleep(0.3)
            kb.clear()
            return True
        if events.get("exit_early"):
            events["exit_early"] = False
            kb.clear()
            print("\n[→] starting episode…", flush=True)
            time.sleep(0.3)
            kb.clear()
            return True
        precise_sleep(0.05)

    return False


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def _cam_index(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _resolve_cam_path(preferred: str, fallback: str | None = None) -> str:
    if Path(preferred).exists():
        return preferred
    if fallback and Path(fallback).exists():
        print(f"Camera path {preferred} missing — using fallback {fallback}")
        return fallback
    return preferred


def _disable_torque(robot) -> None:
    if hasattr(robot, "disable_torque"):
        robot.disable_torque()
    else:
        robot.bus.disable_torque()


def _enable_torque(robot) -> None:
    if hasattr(robot, "enable_torque"):
        robot.enable_torque()
    else:
        robot.bus.enable_torque()


def _read_joint_pose(robot) -> dict[str, float]:
    obs = robot.get_observation()
    return {k: float(v) for k, v in obs.items() if isinstance(k, str) and k.endswith(".pos")}


def _go_home(robot, home: dict[str, float], settle_s: float = 1.5) -> None:
    """Enable torque, drive to home, wait, then disable torque for the next kinesthetic episode."""
    print(f"Returning to home pose (settle {settle_s:g}s)…", flush=True)
    print("  target:", {k: round(v, 2) for k, v in home.items()}, flush=True)

    if hasattr(robot, "go_home"):
        robot.go_home(home)
    else:
        _enable_torque(robot)
        time.sleep(0.2)
        robot.send_action(home)

    # Keep refreshing the goal while settling; also useful for direct-USB mode.
    t_end = time.perf_counter() + max(settle_s, 0.5)
    while time.perf_counter() < t_end:
        if hasattr(robot, "go_home"):
            # Host already holds the goal; just wait / watch.
            pass
        else:
            robot.send_action(home)
        try:
            pose = _read_joint_pose(robot)
            if pose:
                err = sum(abs(pose.get(k, 0.0) - v) for k, v in home.items())
                print(f"\r  home err sum={err:6.2f} deg   ", end="", flush=True)
        except Exception:
            pass
        time.sleep(0.05)
    print()

    _disable_torque(robot)
    # Small delay so disable_torque is processed before the next kinesthetic move.
    time.sleep(0.15)
    print("At home — torque OFF for next episode.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kinesthetic record episodes on one SO-101 follower (record_bi counterpart)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset repo_id, e.g. local/so101_kinesthetic_pick or <hf_user>/so101_pick",
    )
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode_time", type=int, default=60)
    parser.add_argument("--reset_time", type=int, default=10)
    parser.add_argument("--task_description", type=str, default="SO-101 kinesthetic demonstration")
    parser.add_argument(
        "--remote_ip",
        type=str,
        default=None,
        help=(
            "If set, connect to so101_zmq_host.py on the Pi and write the dataset HERE. "
            f"Typical value: {_DEFAULT_REMOTE_IP}"
        ),
    )
    parser.add_argument("--zmq_cmd_port", type=int, default=5601)
    parser.add_argument("--zmq_obs_port", type=int, default=5602)
    parser.add_argument("--port", type=str, default=_DEFAULT_PORT, help="Local USB port (direct mode only)")
    parser.add_argument("--robot_id", type=str, default=_DEFAULT_ROBOT_ID)
    parser.add_argument(
        "--arm_profile",
        type=str,
        default="so-arm-5dof",
        choices=["so-arm-5dof", "am-follower-6dof", "am-follower-6dof-hd"],
    )
    parser.add_argument("--cam_width", type=int, default=320)
    parser.add_argument("--cam_height", type=int, default=240)
    parser.add_argument("--forward_cam", type=str, default=_DEFAULT_FORWARD_CAM)
    parser.add_argument("--wrist_cam", type=str, default=_DEFAULT_WRIST_CAM)
    parser.add_argument("--no_cameras", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--push_to_hub",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--home_settle_s",
        type=float,
        default=1.5,
        help="Seconds to hold home pose after each episode before the next record",
    )
    parser.add_argument(
        "--home_pose",
        type=str,
        default=str(_HOME_POSE_PATH),
        help="JSON file to load/save the home joint pose (default: so101_home_pose.json)",
    )
    parser.add_argument(
        "--relatch_home",
        action="store_true",
        help="Ignore saved home JSON and re-capture home from the arm's pose at connect time",
    )
    return parser.parse_args()


def _build_local_cameras(args: argparse.Namespace) -> dict[str, OpenCVCameraConfig]:
    if args.no_cameras:
        return {}
    common = dict(
        fps=args.fps,
        width=args.cam_width,
        height=args.cam_height,
        rotation=Cv2Rotation.NO_ROTATION,
        fourcc="MJPG",
    )
    return {
        "forward": OpenCVCameraConfig(
            index_or_path=_cam_index(str(_resolve_cam_path(args.forward_cam))),
            **common,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=_cam_index(str(_resolve_cam_path(args.wrist_cam, _DEFAULT_WRIST_FALLBACK))),
            **common,
        ),
    }


def _make_robot(args: argparse.Namespace):
    """Return (robot, use_videos). Remote mode streams cams from the Pi host."""
    if args.remote_ip:
        cam_shapes = {}
        if not args.no_cameras:
            cam_shapes = {
                "forward": (args.cam_height, args.cam_width, 3),
                "wrist": (args.cam_height, args.cam_width, 3),
            }
        robot = SO101ZmqClient(
            SO101ZmqClientConfig(
                remote_ip=args.remote_ip,
                zmq_cmd_port=args.zmq_cmd_port,
                zmq_obs_port=args.zmq_obs_port,
                cameras=cam_shapes,
            )
        )
        return robot, bool(cam_shapes)

    cameras = _build_local_cameras(args)
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            arm_profile=args.arm_profile,
            cameras=cameras,
        )
    )
    return robot, bool(cameras)


def _joint_action_from_obs(obs: dict) -> dict[str, float]:
    return {k: float(v) for k, v in obs.items() if isinstance(k, str) and k.endswith(".pos")}


def kinesthetic_record_loop(
    *,
    robot,
    events: dict,
    fps: int,
    control_time_s: float,
    dataset: LeRobotDataset | None,
    single_task: str,
    display_data: bool,
    kb: FocusKeyboard | None = None,
) -> None:
    control_interval = 1.0 / fps
    start_episode_t = time.perf_counter()
    frame_i = 0

    while (time.perf_counter() - start_episode_t) < control_time_s:
        if kb is not None:
            _apply_stdin_controls(kb, events)

        if events.get("exit_early"):
            events["exit_early"] = False
            break
        if events.get("stop_recording"):
            break

        t0 = time.perf_counter()
        obs = robot.get_observation()
        action = _joint_action_from_obs(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            dataset.add_frame({**observation_frame, **action_frame, "task": single_task})

        if display_data:
            log_rerun_data(observation=obs, action=action)

        frame_i += 1
        if frame_i % 10 == 0:
            elapsed = time.perf_counter() - start_episode_t
            print(
                f"\r  recorded frames={frame_i}  t={elapsed:5.1f}s/{control_time_s:.0f}s   ",
                end="",
                flush=True,
            )

        precise_sleep(max(control_interval - (time.perf_counter() - t0), 0.0))

    if frame_i:
        print()


def main() -> None:
    args = parse_args()
    robot, use_videos = _make_robot(args)

    action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=use_videos)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=use_videos)
    dataset_features = {**action_features, **obs_features}

    if args.resume:
        print("Resuming existing dataset:", args.dataset)
        dataset = LeRobotDataset.resume(
            repo_id=args.dataset,
            root=HF_LEROBOT_HOME / args.dataset,
            image_writer_threads=4,
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.dataset,
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=use_videos,
            image_writer_threads=4,
        )
        print(f"Dataset created with id: {dataset.repo_id}")

    print(f"Local dataset path: {dataset.root.resolve()}")
    if args.remote_ip:
        print(f"Mode: REMOTE via {args.remote_ip} (dataset writes on this machine)")
    else:
        print("Mode: DIRECT USB (dataset writes on this machine — usually the Pi)")

    robot.connect()

    # Latch home: prefer saved JSON unless --relatch_home, else capture current pose.
    home_path = Path(args.home_pose)
    if not args.relatch_home and home_path.is_file():
        home = {k: float(v) for k, v in json.loads(home_path.read_text()).items() if k.endswith(".pos")}
        print(f"Loaded home pose from {home_path}")
    else:
        # Warm one observation (remote may need a poll).
        for _ in range(5):
            home = _read_joint_pose(robot)
            if home:
                break
            time.sleep(0.1)
        if not home:
            raise RuntimeError("Could not read joint pose to latch as home.")
        home_path.write_text(json.dumps(home, indent=2) + "\n")
        print(f"Latched current arm pose as home → {home_path}")
    print("  home:", {k: round(v, 2) for k, v in home.items()})

    print("\nMoving to home pose at startup…")
    _go_home(robot, home, settle_s=args.home_settle_s)

    listener, events = init_keyboard_listener()
    kb = FocusKeyboard()
    kb.start()
    if args.rerun:
        init_rerun(session_name="so101_kinesthetic_record")

    print("\nKinesthetic record ready — torque OFF, move the follower by hand.")
    print("  CONTROLS (this terminal):")
    print("    Enter        = SAVE episode  |  after home, START next episode")
    print("    r            = DISCARD current episode (not saved) and redo")
    print("    q            = QUIT session (discards in-progress episode)")
    print("  Flow: home → Enter(start) → record → Enter(save) → home → reset → Enter(next)\n", flush=True)

    # Same gate as between episodes: user confirms before the first recording starts.
    if not _wait_for_next_episode(
        kb,
        events,
        robot,
        prompt="At home. Press  Enter  when ready to start the first episode",
    ):
        print("Quit before first episode.")
        kb.stop()
        try:
            robot.disconnect()
        except Exception as e:
            print(f"(robot disconnect warning: {e})")
        if listener is not None:
            listener.stop()
        dataset.finalize()
        return

    recorded_episodes = 0
    try:
        while recorded_episodes < args.num_episodes and not events["stop_recording"]:
            print("\n" + "█" * 74)
            print(
                f"██  ▶  RECORDING episode {recorded_episodes + 1}/{args.num_episodes}"
                f"      Enter=save   r=discard   q=quit"
            )
            print("█" * 74 + "\n", flush=True)
            log_say(f"Recording episode {recorded_episodes + 1} of {args.num_episodes}")

            _disable_torque(robot)
            events["exit_early"] = False
            events["rerecord_episode"] = False
            kb.clear()
            time.sleep(0.2)
            kb.clear()

            kinesthetic_record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                control_time_s=args.episode_time,
                dataset=dataset,
                single_task=args.task_description,
                display_data=args.rerun,
                kb=kb,
            )

            # Quit mid-episode: drop the buffer, do not save.
            if events["stop_recording"]:
                dataset.clear_episode_buffer()
                print("Quit — in-progress episode discarded (not saved).")
                break

            # Discard + redo: clear buffer, home, wait for Enter, then loop.
            if events["rerecord_episode"]:
                log_say("Discarded episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                print("Episode discarded (not saved). Returning home…")
                _go_home(robot, home, settle_s=args.home_settle_s)
                if not _wait_for_next_episode(kb, events, robot):
                    break
                continue

            if not dataset.has_pending_frames():
                print("No frames captured (empty episode) — not saving. Press Enter when ready to retry.")
                dataset.clear_episode_buffer()
                if not _wait_for_next_episode(kb, events, robot):
                    break
                continue

            print("\n… saving + encoding episode (video-encoder logs below are normal) …", flush=True)
            dataset.save_episode()
            recorded_episodes += 1
            print(
                f"\n✔✔✔  SAVED  —  {recorded_episodes}/{args.num_episodes} episodes now in the dataset\n",
                flush=True,
            )

            more_to_go = recorded_episodes < args.num_episodes and not events["stop_recording"]
            if more_to_go:
                _go_home(robot, home, settle_s=args.home_settle_s)
                if not _wait_for_next_episode(kb, events, robot):
                    break
    finally:
        log_say("Stop recording")
        kb.stop()
        try:
            robot.disconnect()
        except Exception as e:
            print(f"(robot disconnect warning: {e})")
        if listener is not None:
            listener.stop()
        dataset.finalize()

    print(f"Dataset saved locally at: {dataset.root.resolve()}")
    if args.push_to_hub:
        print(f"Uploading dataset to Hugging Face Hub: {dataset.repo_id}")
        dataset.push_to_hub()
        print(f"Dataset uploaded to: https://huggingface.co/datasets/{dataset.repo_id}")
    else:
        print("Skipping Hugging Face upload (--push_to_hub false).")
        replay_cmd = (
            f"  python examples/alohamini/replay_so101.py --dataset {args.dataset}"
        )
        if args.remote_ip:
            replay_cmd += f" --remote_ip {args.remote_ip}"
        else:
            replay_cmd += f" --port {args.port} --robot_id {args.robot_id}"
        print(f"Replay with:\n{replay_cmd}")


if __name__ == "__main__":
    main()
