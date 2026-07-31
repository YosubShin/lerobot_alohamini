#!/usr/bin/env python3
"""SO-101 ZMQ host — run on the Raspberry Pi.

Exposes the local follower arm + cameras to a workstation client over ZMQ
(same pattern as alohamini lekiwi_host). Stop alohamini-host.service first so
the serial port / cameras are free.

  conda activate lerobot_alohamini
  sudo systemctl stop alohamini-host.service
  python examples/alohamini/so101_zmq_host.py

Workstation then records with:
  python examples/alohamini/record_so101.py --remote_ip 192.168.0.50 ...
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from pathlib import Path

import cv2
import zmq

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

logging.basicConfig(force=True, level=logging.INFO, format="[%(asctime)s] %(message)s")

_DEFAULT_PORT = "/dev/am_arm_follower_right"
_DEFAULT_ROBOT_ID = "am_follower_right"
# Physical USB wiring on this Pi is swapped vs the udev names: the forehead
# (forward) image comes from the node labeled wrist_right, and the wrist image
# comes from video0 (the missing am_camera_forward symlink).
_DEFAULT_FORWARD = "/dev/am_camera_wrist_right"
_DEFAULT_WRIST = "/dev/am_camera_forward"
_DEFAULT_WRIST_FALLBACK = "/dev/video0"


def _resolve(path: str, fallback: str | None = None) -> str:
    if Path(path).exists():
        return path
    if fallback and Path(fallback).exists():
        logging.warning("Camera %s missing — using %s", path, fallback)
        return fallback
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO-101 ZMQ host (run on the Pi)")
    p.add_argument("--port", default=_DEFAULT_PORT)
    p.add_argument("--robot_id", default=_DEFAULT_ROBOT_ID)
    p.add_argument(
        "--arm_profile",
        default="so-arm-5dof",
        choices=["so-arm-5dof", "am-follower-6dof", "am-follower-6dof-hd"],
    )
    p.add_argument("--forward_cam", default=_DEFAULT_FORWARD, help="Forehead / base camera device")
    p.add_argument("--wrist_cam", default=_DEFAULT_WRIST, help="Wrist camera device")
    p.add_argument("--cam_width", type=int, default=320)
    p.add_argument("--cam_height", type=int, default=240)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--zmq_cmd_port", type=int, default=5601)
    p.add_argument("--zmq_obs_port", type=int, default=5602)
    p.add_argument("--no_cameras", action="store_true")
    p.add_argument(
        "--duration_s",
        type=int,
        default=0,
        help="Exit after N seconds (0 = run until Ctrl+C)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cameras = {}
    if not args.no_cameras:
        common = dict(
            fps=args.fps,
            width=args.cam_width,
            height=args.cam_height,
            rotation=Cv2Rotation.NO_ROTATION,
            fourcc="MJPG",
        )
        cameras = {
            "forward": OpenCVCameraConfig(index_or_path=_resolve(args.forward_cam), **common),
            "wrist": OpenCVCameraConfig(
                index_or_path=_resolve(args.wrist_cam, _DEFAULT_WRIST_FALLBACK), **common
            ),
        }

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            arm_profile=args.arm_profile,
            cameras=cameras,
        )
    )
    robot.connect()
    logging.info("Robot connected. Cameras: %s", list(cameras) or "none")

    ctx = zmq.Context()
    cmd_sock = ctx.socket(zmq.PULL)
    # Do NOT conflate commands — enable_torque then go_home must both arrive.
    cmd_sock.setsockopt(zmq.RCVHWM, 50)
    cmd_sock.bind(f"tcp://*:{args.zmq_cmd_port}")

    obs_sock = ctx.socket(zmq.PUSH)
    obs_sock.setsockopt(zmq.CONFLATE, 1)
    obs_sock.bind(f"tcp://*:{args.zmq_obs_port}")

    logging.info(
        "SO-101 host listening  cmd=:%d  obs=:%d  (Ctrl+C to stop)",
        args.zmq_cmd_port,
        args.zmq_obs_port,
    )

    hold_goal: dict[str, float] | None = None
    t0 = time.perf_counter()
    loop_dt = 1.0 / args.fps
    try:
        while True:
            loop_start = time.perf_counter()
            if args.duration_s > 0 and (loop_start - t0) >= args.duration_s:
                break

            # Drain all pending commands this tick (order preserved, no conflate).
            while True:
                try:
                    msg = cmd_sock.recv_string(zmq.NOBLOCK)
                except zmq.Again:
                    break
                try:
                    data = json.loads(msg)
                    cmd = data.get("_cmd")
                    if cmd == "disable_torque":
                        hold_goal = None
                        robot.bus.disable_torque()
                        logging.info("Torque DISABLED (kinesthetic)")
                    elif cmd == "enable_torque":
                        robot.bus.enable_torque()
                        logging.info("Torque ENABLED")
                    elif cmd in ("go_home", "go_to"):
                        action = {k: float(v) for k, v in data.items() if k.endswith(".pos")}
                        if not action:
                            logging.warning("%s with no joint targets", cmd)
                            continue
                        robot.bus.enable_torque()
                        robot.send_action(action)
                        hold_goal = action
                        logging.info("%s → holding %d joints", cmd, len(action))
                    elif cmd == "ping":
                        pass
                    else:
                        action = {k: float(v) for k, v in data.items() if k.endswith(".pos")}
                        if action:
                            robot.send_action(action)
                            hold_goal = action
                except Exception:
                    logging.exception("Command handling failed")

            # Keep streaming the last goal so the arm actually tracks home.
            if hold_goal is not None:
                try:
                    robot.send_action(hold_goal)
                except Exception:
                    logging.exception("hold_goal send_action failed")

            # --- observation ---
            try:
                obs = robot.get_observation()
            except Exception:
                logging.exception("get_observation failed")
                time.sleep(loop_dt)
                continue

            payload = {k: float(v) for k, v in obs.items() if isinstance(k, str) and k.endswith(".pos")}
            for cam_key in cameras:
                frame = obs.get(cam_key)
                if frame is None:
                    payload[cam_key] = ""
                    continue
                try:
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    payload[cam_key] = base64.b64encode(buf).decode("utf-8") if ok else ""
                except Exception:
                    payload[cam_key] = ""

            try:
                obs_sock.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            precise = loop_dt - (time.perf_counter() - loop_start)
            if precise > 0:
                time.sleep(precise)
    except KeyboardInterrupt:
        logging.info("Ctrl+C — shutting down")
    finally:
        try:
            robot.disconnect()
        except Exception as e:
            logging.warning("disconnect: %s", e)
        cmd_sock.close()
        obs_sock.close()
        ctx.term()
        logging.info("Host stopped.")


if __name__ == "__main__":
    main()
