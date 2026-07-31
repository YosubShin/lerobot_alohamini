#!/usr/bin/env python3
"""Workstation-side ZMQ client for a remote SO-101 follower (see so101_zmq_host.py).

Mirrors the AlohaMini LeKiwiClient pattern at a smaller scale: joints + cameras
stream from the Pi; torque / goal-position commands go the other way. Dataset
recording runs here so files land on the local machine.
"""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from functools import cached_property

import cv2
import numpy as np

from lerobot.types import RobotAction, RobotObservation


@dataclass
class SO101ZmqClientConfig:
    remote_ip: str
    zmq_cmd_port: int = 5601
    zmq_obs_port: int = 5602
    connect_timeout_s: float = 10.0
    polling_timeout_ms: int = 200
    # Must match host camera names + resolution used for dataset feature shapes.
    cameras: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "forward": (240, 320, 3),  # H, W, C
            "wrist": (240, 320, 3),
        }
    )
    joint_names: tuple[str, ...] = (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )


class SO101ZmqClient:
    """Duck-typed robot client: connect / get_observation / send_action / torque."""

    name = "so101_zmq_client"

    def __init__(self, config: SO101ZmqClientConfig):
        import zmq

        self._zmq = zmq
        self.config = config
        self.id = f"so101@{config.remote_ip}"
        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None
        self._is_connected = False
        self.last_frames: dict[str, np.ndarray] = {}
        self.last_joints: dict[str, float] = dict.fromkeys(config.joint_names, 0.0)
        # Count of genuinely fresh messages off the wire. get_observation()
        # silently serves the cache on a miss, so this is the only way to tell a
        # healthy 30 Hz stream from one whose frames are being taken by another
        # consumer — and unlike a second socket, counting does not itself steal.
        self.msgs_received = 0

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        joints = {name: float for name in self.config.joint_names}
        return {**joints, **self.config.cameras}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {name: float for name in self.config.joint_names}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        zmq = self._zmq
        self.zmq_context = zmq.Context()

        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        # Do not conflate commands — sequenced enable/go_home/disable must all arrive.
        self.zmq_cmd_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_cmd_socket.connect(f"tcp://{self.config.remote_ip}:{self.config.zmq_cmd_port}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_observation_socket.connect(
            f"tcp://{self.config.remote_ip}:{self.config.zmq_obs_port}"
        )

        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(int(self.config.connect_timeout_s * 1000)))
        if self.zmq_observation_socket not in socks:
            raise TimeoutError(
                f"No observation from SO-101 host at {self.config.remote_ip}:"
                f"{self.config.zmq_obs_port} within {self.config.connect_timeout_s}s. "
                "Is so101_zmq_host.py running on the Pi?"
            )
        self._is_connected = True
        logging.info("Connected to SO-101 host at %s", self.config.remote_ip)

    def disconnect(self) -> None:
        if self.zmq_cmd_socket is not None:
            self.zmq_cmd_socket.close()
        if self.zmq_observation_socket is not None:
            self.zmq_observation_socket.close()
        if self.zmq_context is not None:
            self.zmq_context.term()
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None
        self.zmq_context = None
        self._is_connected = False

    def _send_json(self, payload: dict) -> None:
        if not self._is_connected:
            raise RuntimeError("SO101ZmqClient is not connected")
        self.zmq_cmd_socket.send_string(json.dumps(payload))

    def disable_torque(self) -> None:
        self._send_json({"_cmd": "disable_torque"})

    def enable_torque(self) -> None:
        self._send_json({"_cmd": "enable_torque"})

    def send_action(self, action: RobotAction) -> RobotAction:
        payload = {k: float(v) for k, v in action.items() if k.endswith(".pos")}
        self._send_json(payload)
        return payload

    def go_home(self, home: dict[str, float]) -> None:
        """Atomically enable torque and move to ``home`` on the Pi host."""
        payload = {"_cmd": "go_home"}
        payload.update({k: float(v) for k, v in home.items() if k.endswith(".pos")})
        self._send_json(payload)

    def _decode_image(self, image_b64: str) -> np.ndarray | None:
        if not image_b64:
            return None
        try:
            arr = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.warning("image decode failed: %s", e)
            return None

    def _recv_latest(self) -> dict | None:
        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.config.polling_timeout_ms))
        if self.zmq_observation_socket not in socks:
            return None
        last = None
        while True:
            try:
                last = self.zmq_observation_socket.recv_string(zmq.NOBLOCK)
                self.msgs_received += 1
            except zmq.Again:
                break
        if last is None:
            return None
        try:
            return json.loads(last)
        except json.JSONDecodeError:
            return None

    def get_observation(self) -> RobotObservation:
        msg = self._recv_latest()
        if msg is not None:
            for name in self.config.joint_names:
                if name in msg:
                    self.last_joints[name] = float(msg[name])
            for cam in self.config.cameras:
                frame = self._decode_image(msg.get(cam, ""))
                if frame is not None:
                    self.last_frames[cam] = frame

        obs: RobotObservation = dict(self.last_joints)
        for cam, shape in self.config.cameras.items():
            frame = self.last_frames.get(cam)
            if frame is None:
                frame = np.zeros(shape, dtype=np.uint8)
            obs[cam] = frame
        return obs
