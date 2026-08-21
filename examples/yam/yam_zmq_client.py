#!/usr/bin/env python3
"""Workstation-side client for yam_zmq_host.py.

Reuses SO101ZmqClient verbatim (the wire protocol is identical) with YAM
joint names and cameras, plus YAM-specific helpers: mode switching and
reading the host's currently commanded target (the teleop-semantics action
a recorder should store).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alohamini"))
from so101_zmq_client import SO101ZmqClient, SO101ZmqClientConfig  # noqa: E402

YAM_JOINT_NAMES = (
    "joint1.pos", "joint2.pos", "joint3.pos",
    "joint4.pos", "joint5.pos", "joint6.pos",
    "gripper.pos",
)


@dataclass
class YamZmqClientConfig(SO101ZmqClientConfig):
    zmq_cmd_port: int = 5601
    zmq_obs_port: int = 5602
    cameras: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "wrist": (240, 320, 3),
            "front": (240, 320, 3),
        }
    )
    joint_names: tuple[str, ...] = YAM_JOINT_NAMES


class YamZmqClient(SO101ZmqClient):
    name = "yam_zmq_client"

    def __init__(self, config: YamZmqClientConfig):
        super().__init__(config)
        self.last_targets: dict[str, float] = {}
        self.last_mode: str = "unknown"

    def set_mode(self, mode: str) -> None:
        """mode: 'idle' | 'teleop' | 'policy'."""
        self._send_json({"_cmd": "set_mode", "mode": mode})

    def get_observation(self):
        # parent drops keys outside joint_names/cameras, so capture the raw
        # message once and harvest target.* / mode before delegating caching
        msg = self._recv_latest()
        if msg is not None:
            for name in self.config.joint_names:
                if name in msg:
                    self.last_joints[name] = float(msg[name])
                tkey = f"target.{name}"
                if tkey in msg:
                    self.last_targets[tkey] = float(msg[tkey])
            if "mode" in msg:
                self.last_mode = msg["mode"]
            for cam in self.config.cameras:
                frame = self._decode_image(msg.get(cam, ""))
                if frame is not None:
                    self.last_frames[cam] = frame
        import numpy as np
        obs = dict(self.last_joints)
        obs.update(self.last_targets)
        obs["mode"] = self.last_mode
        for cam, shape in self.config.cameras.items():
            frame = self.last_frames.get(cam)
            obs[cam] = frame if frame is not None else np.zeros(shape, dtype=np.uint8)
        return obs

    def get_target_action(self, obs: dict) -> dict[str, float]:
        """Extract the host's commanded target from an observation dict —
        in teleop mode this is the action a dataset should record."""
        return {name: float(obs.get(f"target.{name}", obs.get(name, 0.0)))
                for name in self.config.joint_names}


if __name__ == "__main__":
    # smoke: connect, read a few observations, command a small joint move
    import time

    import numpy as np

    client = YamZmqClient(YamZmqClientConfig(remote_ip="127.0.0.1"))
    client.connect()
    obs = client.get_observation()
    joints = {n: obs[n] for n in client.config.joint_names}
    print("joints:", {k: round(v, 3) for k, v in joints.items()})
    print("cams:", {c: obs[c].shape for c in client.config.cameras if hasattr(obs[c], "shape")})
    tgt = dict(joints)
    tgt["joint2.pos"] = joints["joint2.pos"] + 0.2
    client.send_action(tgt)
    time.sleep(1.5)
    obs2 = client.get_observation()
    moved = obs2["joint2.pos"] - joints["joint2.pos"]
    frames_differ = not np.array_equal(obs["wrist"], obs2["wrist"])
    print(f"joint2 moved {moved:+.3f} rad (want ~+0.2) | frames fresh: {frames_differ}")
    client.disconnect()
    ok = abs(moved - 0.2) < 0.05 and frames_differ
    print("SMOKE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
