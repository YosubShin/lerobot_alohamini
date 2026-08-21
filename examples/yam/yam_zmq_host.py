#!/usr/bin/env python3
"""ZMQ host for one or two I2RT YAM arms — the single owner of the CAN buses.

Speaks the same wire protocol as so101_zmq_host.py (flat JSON joints +
base64-JPEG cameras on a PULL/PUSH pair), so SO101ZmqClient-family clients —
and therefore every recorder / DAgger / grasp-eval tool in this repo —
connect unchanged.

Runs the bilateral follower-as-leader teleop loop ported from
i2rt/scripts/teleop.py: the leader arm sits in gravity compensation and is
moved by hand; the follower tracks a rate-limited target; while syncing, the
leader receives a weak PD spring toward the follower's MEASURED position
(deadband + stretch cap), so the operator feels contact. Modes:

  idle    — follower holds pose (keeps the motor timeout fed)
  teleop  — bilateral leader->follower; client actions ignored
  policy  — client actions command the follower; leader (if any) idles

Observations publish at --publish_hz with measured joints, the currently
commanded target (``target.<name>`` — in teleop mode this is the
teleop-semantics ACTION a recorder should store), camera JPEGs, and the mode.

Joint naming: i2rt URDF order, our ``.pos`` suffix convention —
joint1..joint6 + gripper (gripper slot resolved from robot info at runtime).
Units are i2rt-native: radians for arm joints, normalized 0..1 gripper.

Run with --sim for a MuJoCo-backed dry run (no CAN hardware, synthetic
camera frames): the full client/recorder stack can be exercised locally.
"""
from __future__ import annotations

import argparse
import base64
import json
import threading
import time

import cv2
import numpy as np
import zmq

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZMQ host for YAM arm(s)")
    p.add_argument("--follower_channel", type=str, default="can0")
    p.add_argument("--leader_channel", type=str, default="none",
                   help="CAN channel of the leader arm, or 'none' for policy-only")
    p.add_argument("--gripper", type=str, default="linear_4310")
    p.add_argument("--sim", action="store_true",
                   help="MuJoCo sim follower, synthetic cameras, no CAN")
    p.add_argument("--cmd_port", type=int, default=5601)
    p.add_argument("--obs_port", type=int, default=5602)
    p.add_argument("--control_hz", type=float, default=100.0)
    p.add_argument("--publish_hz", type=float, default=30.0)
    p.add_argument("--cameras", type=str, default="",
                   help="name:device list, e.g. 'wrist:0,front:2'. Empty = none "
                        "(sim always publishes synthetic frames)")
    p.add_argument("--cam_width", type=int, default=320)
    p.add_argument("--cam_height", type=int, default=240)
    p.add_argument("--jpeg_quality", type=int, default=80)
    # follower tracking (ported from i2rt/scripts/teleop.py)
    p.add_argument("--approach_step", type=float, default=0.004,
                   help="rad/tick while closing a pose gap (~0.4 rad/s @100Hz)")
    p.add_argument("--track_step", type=float, default=0.03,
                   help="rad/tick once matched (~3 rad/s)")
    p.add_argument("--match_tol", type=float, default=0.05)
    p.add_argument("--max_err", type=float, default=0.15,
                   help="rad; caps commanded-target runaway from measured pose")
    # bilateral feedback (teleop mode)
    p.add_argument("--feedback_kp_scale", type=float, default=0.3)
    p.add_argument("--arm_deadband", type=float, default=0.03)
    p.add_argument("--gripper_deadband", type=float, default=0.04)
    p.add_argument("--arm_err_cap", type=float, default=0.25)
    p.add_argument("--gripper_err_cap", type=float, default=0.35)
    return p.parse_args()


class Cameras:
    """Background camera reader; synthetic frames in sim / when none attached."""

    def __init__(self, spec: str, w: int, h: int, synthetic: bool):
        self.w, self.h = w, h
        self.frames: dict[str, np.ndarray] = {}
        self.caps: dict[str, cv2.VideoCapture] = {}
        self.synthetic = synthetic
        self._tick = 0
        if spec:
            for item in spec.split(","):
                name, dev = item.split(":")
                cap = cv2.VideoCapture(int(dev))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                self.caps[name.strip()] = cap
        if synthetic and not self.caps:
            self.frames = {"wrist": self._synth(), "front": self._synth()}
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _synth(self) -> np.ndarray:
        # moving gradient so freshness/frozen-frame checks see change
        self._tick += 1
        x = np.linspace(0, 255, self.w, dtype=np.uint8)
        img = np.tile(x, (self.h, 1))
        img = np.roll(img, self._tick * 3, axis=1)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def _loop(self) -> None:
        while True:
            if self.caps:
                for name, cap in self.caps.items():
                    ok, frame = cap.read()
                    if ok:
                        if frame.shape[1] != self.w or frame.shape[0] != self.h:
                            frame = cv2.resize(frame, (self.w, self.h))
                        self.frames[name] = frame
            elif self.synthetic:
                self.frames = {"wrist": self._synth(), "front": self._synth()}
            time.sleep(1.0 / 30)

    def encoded(self, quality: int) -> dict[str, str]:
        out = {}
        for name, frame in self.frames.items():
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                out[name] = base64.b64encode(buf.tobytes()).decode("ascii")
        return out


def gravity_idle(robot) -> None:
    if hasattr(robot, "enter_gravity_comp_idle"):
        robot.enter_gravity_comp_idle()
    elif hasattr(robot, "enable_gravity_comp"):
        robot.enable_gravity_comp()


def main() -> None:
    args = parse_args()
    gripper = GripperType[args.gripper.upper()]

    follower = get_yam_robot(channel=args.follower_channel, gripper_type=gripper,
                             zero_gravity_mode=False, sim=args.sim)
    leader = None
    if args.leader_channel.lower() != "none" and not args.sim:
        leader = get_yam_robot(channel=args.leader_channel, gripper_type=gripper)

    n = len(follower.get_joint_pos())
    info = follower.get_robot_info() if hasattr(follower, "get_robot_info") else {}
    gripper_idx = info.get("gripper_index", n - 1)
    if gripper_idx is None:
        gripper_idx = n - 1
    joint_names = [f"joint{i + 1}.pos" for i in range(n)]
    joint_names[gripper_idx] = "gripper.pos"
    print(f"{n} joints: {joint_names} (follower={args.follower_channel}"
          f"{', leader=' + args.leader_channel if leader else ''}"
          f"{', SIM' if args.sim else ''})")

    # bilateral gains (leader side)
    fb_kp = fb_kd = None
    deadband = np.full(n, args.arm_deadband)
    err_cap = np.full(n, args.arm_err_cap)
    deadband[gripper_idx] = args.gripper_deadband
    err_cap[gripper_idx] = args.gripper_err_cap
    if leader is not None:
        linfo = leader.get_robot_info()
        fb_kp = np.asarray(linfo["kp"], dtype=float) * args.feedback_kp_scale
        fb_kd = np.asarray(linfo["grav_comp_kd"], dtype=float)

    cams = Cameras(args.cameras, args.cam_width, args.cam_height, synthetic=args.sim)

    ctx = zmq.Context()
    cmd_sock = ctx.socket(zmq.PULL)
    cmd_sock.bind(f"tcp://*:{args.cmd_port}")
    obs_sock = ctx.socket(zmq.PUSH)
    obs_sock.setsockopt(zmq.SNDHWM, 5)
    obs_sock.bind(f"tcp://*:{args.obs_port}")

    mode = "teleop" if leader is not None else "idle"
    target = follower.get_joint_pos().copy()
    policy_target = target.copy()
    home_ramp: list[np.ndarray] = []
    dt = 1.0 / args.control_hz
    pub_every = max(1, int(round(args.control_hz / args.publish_hz)))
    tick = 0
    fb_stretch = np.zeros(n)

    def vec_from_msg(msg: dict) -> np.ndarray | None:
        vals = target.copy()
        seen = False
        for i, name in enumerate(joint_names):
            if name in msg:
                vals[i] = float(msg[name])
                seen = True
        return vals if seen else None

    print(f"mode={mode}; listening cmd:{args.cmd_port} obs:{args.obs_port}")
    while True:
        loop_start = time.time()

        # ---- drain commands ----
        while True:
            try:
                msg = json.loads(cmd_sock.recv_string(zmq.NOBLOCK))
            except zmq.Again:
                break
            except json.JSONDecodeError:
                continue
            cmd = msg.get("_cmd")
            if cmd == "set_mode" and msg.get("mode") in ("idle", "teleop", "policy"):
                want = msg["mode"]
                if want == "teleop" and leader is None:
                    print("\n[host] teleop requested but no leader — ignored")
                else:
                    mode = want
                    target = follower.get_joint_pos().copy()
                    policy_target = target.copy()
                    print(f"\n[host] mode -> {mode}")
            elif cmd == "disable_torque":
                gravity_idle(follower)
                mode = "idle"
            elif cmd == "enable_torque":
                target = follower.get_joint_pos().copy()
                policy_target = target.copy()
            elif cmd == "go_home":
                vals = vec_from_msg(msg)
                if vals is not None:
                    cur = follower.get_joint_pos().copy()
                    steps = int(2.0 * args.control_hz)
                    home_ramp = [cur + (vals - cur) * (i + 1) / steps for i in range(steps)]
                    mode = "policy"
            elif cmd in ("archive_start", "archive_stop"):
                pass  # SO-101 compat no-op
            elif cmd is None:
                vals = vec_from_msg(msg)
                if vals is not None and mode != "teleop":
                    mode = "policy"
                    policy_target = vals

        q_foll = follower.get_joint_pos()

        # ---- control ----
        if home_ramp:
            target = home_ramp.pop(0)
            follower.command_joint_pos(target)
        elif mode == "teleop" and leader is not None:
            q_lead = leader.get_joint_pos()
            err = q_lead - target
            step = args.track_step if np.max(np.abs(err)) < args.match_tol else args.approach_step
            target = target + np.clip(err, -step, step)
            target = q_foll + np.clip(target - q_foll, -args.max_err, args.max_err)
            follower.command_joint_pos(target)
            e = q_foll - q_lead
            fb_stretch = np.clip(np.sign(e) * np.maximum(np.abs(e) - deadband, 0.0),
                                 -err_cap, err_cap)
            leader.command_joint_state({"pos": q_lead + fb_stretch,
                                        "vel": np.zeros(n), "kp": fb_kp, "kd": fb_kd})
        elif mode == "policy":
            err = policy_target - target
            step = args.track_step if np.max(np.abs(err)) < args.match_tol else args.approach_step
            target = target + np.clip(err, -step, step)
            target = q_foll + np.clip(target - q_foll, -args.max_err, args.max_err)
            follower.command_joint_pos(target)
        else:  # idle: keep feeding the motor timeout
            follower.command_joint_pos(target)

        # ---- publish ----
        tick += 1
        if tick % pub_every == 0:
            obs = {name: float(q_foll[i]) for i, name in enumerate(joint_names)}
            obs.update({f"target.{name}": float(target[i]) for i, name in enumerate(joint_names)})
            obs["mode"] = mode
            obs["t"] = time.time()
            obs.update(cams.encoded(args.jpeg_quality))
            try:
                obs_sock.send_string(json.dumps(obs), zmq.NOBLOCK)
            except zmq.Again:
                pass

        time.sleep(max(0.0, dt - (time.time() - loop_start)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
