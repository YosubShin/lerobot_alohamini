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
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
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


class H264FisheyeCamera:
    """Wrist fisheye (Arducam Low Light) — H264-only, max 1080p30.

    One ffmpeg process owns the device and tees the stream:
      (a) the camera's own H264, container-remuxed (`-c copy`, zero transcode)
          into a timestamped .mkv archive — full-res record for SLAM;
      (b) software-decoded + downscaled rgb24 frames on a pipe — fed into the
          existing per-frame JPEG/ZMQ path, so client/recorder are unchanged.
    rgb24 matches OpenCVCamera's channel order, preserving the pipeline's
    encode/decode color semantics.
    """

    SEGMENT_S = 300           # archive in 5-min chunks so completed ones can be moved off the Pi
    MIN_FREE_GB = 3.0         # below this, run decode-only: live feed survives, archive pauses
    RESTART_BACKOFF_S = 3.0

    def __init__(self, device: str, out_w: int, out_h: int, archive_dir: str,
                 ffmpeg: str = "ffmpeg", fps: int = 30, codec: str = "mjpeg"):
        # codec: "mjpeg" (video1 node, ~43 Mbps, 4x sharper — better SLAM source)
        #        or "h264" (video4 node, fixed ~9.4 Mbps, smaller archives)
        self.device, self.out_w, self.out_h, self.fps = device, out_w, out_h, fps
        self.codec = codec
        self.archive_dir = Path(archive_dir).expanduser()
        self.ffmpeg = ffmpeg
        self.proc: subprocess.Popen | None = None
        self.latest: np.ndarray | None = None
        self.lock = threading.Lock()
        self.frames_read = 0
        self.archive_path: Path | None = None
        self._stopping = False

    def start(self) -> None:
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._supervise, daemon=True).start()

    def _free_gb(self) -> float:
        return shutil.disk_usage(self.archive_dir).free / 1e9

    def _spawn(self) -> None:
        cmd = [self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
               "-f", "v4l2", "-input_format", self.codec, "-framerate", str(self.fps),
               "-video_size", "1920x1080", "-i", self.device]
        free = self._free_gb()
        if free >= self.MIN_FREE_GB:
            base = time.strftime("fisheye_%Y%m%d_%H%M%S")
            self.archive_path = self.archive_dir / f"{base}_%04d.mkv"
            cmd += ["-map", "0:v", "-c", "copy", "-f", "segment",
                    "-segment_time", str(self.SEGMENT_S), "-segment_format", "matroska",
                    "-reset_timestamps", "1", str(self.archive_path)]
        else:
            self.archive_path = None
            logging.critical(
                "FISHEYE ARCHIVE DISABLED — only %.1f GB free in %s (need %.0f). "
                "Live wrist feed continues but 1080p is NOT being saved. Free disk "
                "space (move old fisheye_*.mkv off the Pi) and restart the host.",
                free, self.archive_dir, self.MIN_FREE_GB)
        cmd += ["-map", "0:v", "-vf", f"scale={self.out_w}:{self.out_h}",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0)
        logging.info("Fisheye H264: %s -> %s (+%dx%d live, %.1f GB free)",
                     self.device, self.archive_path or "NO ARCHIVE", self.out_w, self.out_h, free)

    def _supervise(self) -> None:
        """Keep ffmpeg alive: respawn on death (disk-full kills the writer) with
        backoff. During downtime `latest` goes stale — the recorder's frozen-frame
        guard aborts rather than saving duplicated wrist images."""
        while not self._stopping:
            try:
                self._spawn()
            except Exception:
                logging.exception("Fisheye spawn failed — retrying in %.0f s", self.RESTART_BACKOFF_S)
                time.sleep(self.RESTART_BACKOFF_S)
                continue
            self._read_frames()
            if self._stopping:
                return
            logging.critical(
                "FISHEYE FFMPEG DIED (rc=%s, %.1f GB free) — restarting in %.0f s; "
                "wrist frames are STALE until it recovers.",
                self.proc.poll(), self._free_gb(), self.RESTART_BACKOFF_S)
            time.sleep(self.RESTART_BACKOFF_S)

    def _read_frames(self) -> None:
        n = self.out_w * self.out_h * 3
        stream = self.proc.stdout
        while True:
            # raw pipe: read(n) may return short — accumulate exactly one frame
            chunks, got = [], 0
            while got < n:
                piece = stream.read(n - got)
                if not piece:
                    return
                chunks.append(piece)
                got += len(piece)
            frame = np.frombuffer(b"".join(chunks), dtype=np.uint8).reshape(self.out_h, self.out_w, 3)
            with self.lock:
                self.latest = frame
                self.frames_read += 1

    def read(self) -> np.ndarray | None:
        with self.lock:
            return self.latest

    def stop(self) -> None:
        self._stopping = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        logging.info("Fisheye archive closed: %s (%d frames served)",
                     self.archive_path, self.frames_read)


def _find_fisheye_node(codec: str) -> str | None:
    """The Low Light fisheye exposes MJPG and H264 on different /dev/video
    nodes with identical udev attributes — pick the node by probing formats."""
    want = "MJPG" if codec == "mjpeg" else "H264"
    for name_file in sorted(Path("/sys/class/video4linux").glob("video*/name")):
        if not name_file.read_text().startswith("Arducam 1080P Low Light"):
            continue
        dev = f"/dev/{name_file.parent.name}"
        try:
            out = subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats"],
                                 capture_output=True, text=True, timeout=5).stdout
        except Exception:
            continue
        if want in out:
            return dev
    return None


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
    p.add_argument(
        "--p_coefficient",
        type=int,
        default=16,
        help="Feetech position-loop P gain (servo default 32; this stack has used 16 to "
             "damp shakiness). Open-loop replay of a kinesthetic episode at P=16 showed "
             "167 ms of following error. Raise toward 32 to trade lag for stiffness.",
    )
    p.add_argument("--i_coefficient", type=int, default=0)
    p.add_argument("--d_coefficient", type=int, default=32)
    p.add_argument("--cam_width", type=int, default=320)
    p.add_argument("--cam_height", type=int, default=240)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--zmq_cmd_port", type=int, default=5601)
    p.add_argument("--zmq_obs_port", type=int, default=5602)
    p.add_argument("--no_cameras", action="store_true")
    p.add_argument("--wrist_h264", action="store_true",
                   help="Use the H264-only fisheye as the wrist camera: ffmpeg tees "
                        "full-res 1080p30 H264 into --wrist_archive_dir (SLAM record) "
                        "and feeds downscaled live frames into the normal stream")
    p.add_argument("--wrist_h264_dev", default="/dev/am_camera_fisheye")
    p.add_argument("--wrist_codec", choices=["mjpeg", "h264"], default="mjpeg",
                   help="Fisheye interface/archive codec. mjpeg (~43 Mbps) is ~4x sharper "
                        "on this camera and the better SLAM source; h264 (~9.4 Mbps) for "
                        "small archives. The two live on different /dev nodes.")
    p.add_argument("--wrist_archive_dir", default="~/fisheye_archive")
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
        }
        if not args.wrist_h264:
            cameras["wrist"] = OpenCVCameraConfig(
                index_or_path=_resolve(args.wrist_cam, _DEFAULT_WRIST_FALLBACK), **common
            )

    fisheye: H264FisheyeCamera | None = None
    if args.wrist_h264 and not args.no_cameras:
        ff = shutil.which("ffmpeg") or str(Path.home() / "miniforge3/envs/lerobot_alohamini/bin/ffmpeg")
        dev = _find_fisheye_node(args.wrist_codec) or _resolve(args.wrist_h264_dev, "/dev/video4")
        fisheye = H264FisheyeCamera(dev, args.cam_width, args.cam_height,
                                    args.wrist_archive_dir, ffmpeg=ff, fps=args.fps,
                                    codec=args.wrist_codec)

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            arm_profile=args.arm_profile,
            cameras=cameras,
            p_coefficient=args.p_coefficient,
            i_coefficient=args.i_coefficient,
            d_coefficient=args.d_coefficient,
        )
    )
    robot.connect()
    if fisheye is not None:
        fisheye.start()
    logging.info(
        "Robot connected. Cameras: %s  gains: P=%d I=%d D=%d",
        (list(cameras) + (["wrist(h264-fisheye)"] if fisheye else [])) or "none",
        args.p_coefficient,
        args.i_coefficient,
        args.d_coefficient,
    )

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
            frames = {cam_key: obs.get(cam_key) for cam_key in cameras}
            if fisheye is not None:
                frames["wrist"] = fisheye.read()
            for cam_key, frame in frames.items():
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
        if fisheye is not None:
            fisheye.stop()
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
