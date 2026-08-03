#!/usr/bin/env python3
"""Drive the remote SO-101 follower with a trained policy (DP/ACT checkpoint).

Built on so101_zmq_client.py (same transport as record_so101.py / replay_so101.py):
the Pi runs so101_zmq_host.py; observations (joints + jpeg cameras) stream here,
goal positions stream back.

Matches the training pipeline of so101_red_pick_* datasets:
  - normalization stats come from --train_dataset_root (NOT recomputed)
  - policy sees exactly the cameras + state dims the training dataset has
  - diffusion runs DDIM-10 by default (DDPM-100 is ~10x too slow for 30 fps)
  - per-tick slew limit defaults to 7 units (kinesthetic demo p99.9 ~ 6.2;
    teleop-era datasets should pass 4)
  - before each episode the arm is sent to the median first-frame pose of the
    training set (in-distribution start), via the host's atomic go_home.

Example:
  # Pi:  python examples/alohamini/so101_zmq_host.py
  python examples/alohamini/evaluate_so101.py \
    --hf_model_id /mnt/nvme/lerobot/outputs/dp_redpick_joint_final_20260729/checkpoints/002000/pretrained_model \
    --train_dataset_root /mnt/nvme/lerobot/yosubshin/so101_red_pick_clean \
    --task_description "Put the red block into the bin" \
    --remote_ip 192.168.0.50 --num_episodes 5 --episode_time 20
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

# Pin inference to the RTX 6000 by default (129 ms/chunk vs 208 ms on the 3090 —
# more headroom for the async planner). Override with CUDA_VISIBLE_DEVICES if a
# training job owns it. UUIDs because CUDA/nvidia-smi enumerate GPUs in opposite
# order on this machine. Must precede any torch import.
RTX_6000_UUID = "GPU-e93eba70-d129-000d-8077-63d41338f759"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", RTX_6000_UUID)

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import populate_queues, prepare_observation_for_inference
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_zmq_client import SO101ZmqClient, SO101ZmqClientConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Policy-driven rollout on the remote SO-101")
    p.add_argument("--hf_model_id", type=str, required=True)
    p.add_argument("--train_dataset_root", type=str, required=True,
                   help="Dataset the policy was trained on (stats + feature layout)")
    p.add_argument("--task_description", type=str, default="pick the red block")
    p.add_argument("--remote_ip", type=str, required=True)
    p.add_argument("--zmq_cmd_port", type=int, default=5601)
    p.add_argument("--zmq_obs_port", type=int, default=5602)
    p.add_argument("--num_episodes", type=int, default=3)
    p.add_argument("--episode_time", type=float, default=60.0,
                   help="Safety cap on rollout duration; normally you stop with ENTER")
    p.add_argument("--fps", type=int, default=30)
    # diffusion inference knobs
    p.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--num_inference_steps", type=int, default=10,
                   help="0 = checkpoint default")
    p.add_argument("--n_action_steps", type=int, default=16,
                   help="Actions executed per chunk before replanning (0 = checkpoint default)")
    # safety / start pose
    p.add_argument("--max_delta_per_tick", type=float, default=7.0,
                   help="Per-tick clamp on commanded joint change (0 = off)")
    p.add_argument("--gripper_max_delta", type=float, default=20.0,
                   help="Separate (looser) per-tick clamp for the gripper — the arm "
                        "limiter was silently halving close speed at fps15, giving a "
                        "slippery block time to squirt out (0 = no gripper clamp)")
    p.add_argument("--gripper_binarize", action="store_true",
                   help="Snap gripper commands to fully-open (100) or committed-close "
                        "(--gripper_closed_cmd) with hysteresis (<60 closes, >75 opens). "
                        "Kills the mode-averaged 60-70 hover — the policy outputs the "
                        "average of open and close modes, which is simultaneously too "
                        "closed to descend cleanly and proprioceptively identical to "
                        "'holding the block' (drives descend-lift oscillation).")
    p.add_argument("--gripper_closed_cmd", type=float, default=30.0,
                   help="Close-state command when --gripper_binarize is on (teleop data "
                        "median close is ~30)")
    p.add_argument("--gripper_close_bias", type=float, default=0.0,
                   help="Extra closure subtracted from gripper commands below 45 (committed closes only — "
                        "biasing mid-range hover commands parks the gripper in the holding-ambiguity zone "
                        "and causes descend-lift oscillation; kinesthetic models need ~10-15)")
    p.add_argument("--action_ema", type=float, default=0.0,
                   help="Low-pass the executed action stream: cmd = a*new + (1-a)*prev. "
                        "DP's chunks carry a ~0.6 unit/tick noise floor regardless of "
                        "denoise steps; on 2x-stretched models real motion is smaller than "
                        "that, so rollouts wander. Try 0.3 (≈2 Hz cutoff at 30 fps). 0 = off.")
    p.add_argument("--chunk_trigger", type=int, default=6,
                   help="Regenerate the next chunk when this many actions remain buffered. "
                        "Raise to 8-10 when running inference on the slower 3090 at 30 fps "
                        "(chunk generation there is ~208 ms vs a 6-tick/200 ms deadline).")
    p.add_argument("--no_rtc", action="store_true",
                   help="Disable RTC prefix-inpainting (each chunk becomes an independent "
                        "sample again, as before 2026-07-30)")
    p.add_argument("--blend_splices", action="store_true",
                   help="Cross-fade between consecutive action chunks (2-chunk temporal "
                        "ensemble). Smooths splice jumps at the cost of averaging modes; "
                        "felt worse at fps 10, untested at fps 30.")
    p.add_argument("--obs_history_hz", type=int, default=0,
                   help="Feed the policy's observation-history queue at this rate "
                        "even when --fps is lower, by sampling extra observations "
                        "at interp substep boundaries (host streams 30 Hz "
                        "regardless). Restores the training-time 33 ms obs-pair "
                        "spacing at fps15/substeps2: use --obs_history_hz 30. "
                        "0 = off (history spaced at the control tick).")
    p.add_argument("--interp_substeps", type=int, default=1,
                   help="Stream N interpolated micro-commands per policy tick. At low --fps "
                        "the raw action staircase makes the servo dash between targets; e.g. "
                        "--fps 10 --interp_substeps 3 sends smooth 30 Hz ramps instead.")
    p.add_argument("--no_reset", action="store_true",
                   help="Skip the go-home between episodes")
    p.add_argument("--home_seconds", type=float, default=3.0,
                   help="Duration of the ramped move to the start pose (higher = gentler)")
    p.add_argument("--log_dir", type=str, default="/mnt/nvme/lerobot/outputs/so101_rollout_logs",
                   help="Per-tick trajectory logs (commanded/observed joints) land here")
    return p.parse_args()


class AsyncChunkPlanner:
    """Zero-stall action streaming for a diffusion policy.

    The synchronous pattern (policy.select_action per tick) freezes the arm for
    a full chunk-generation pass (~130 ms on the RTX 6000) every n_action_steps
    ticks — visible stop-and-go at 30 fps. Here the control loop instead:
      - feeds the policy's observation history every tick (preserving the 33 ms
        obs spacing the model was trained on),
      - triggers generation of the NEXT chunk in a worker thread while `trigger`
        actions still remain in the buffer,
      - splices the fresh chunk in when the buffer drains, dropping the actions
        whose timesteps already elapsed while the worker ran.
    The policy's queues are only touched under a short lock (append / stack);
    the expensive denoising runs lock-free on a queue snapshot.
    """

    def __init__(self, policy, postprocessor, device: str, trigger: int = 6,
                 blend: bool = False, rtc: bool = True):
        self.policy = policy
        self.post = postprocessor
        self.device = device
        self.trigger = trigger
        self.blend = blend
        # RTC-style prefix inpainting (RePaint/Diffuser/Real-Time-Chunking):
        # each new chunk is generated with the horizon steps that overlap
        # already-committed actions clamped (forward-noised per denoise step),
        # so it is a continuation of the executing mode rather than an
        # independent sample. Kills chunk-boundary mode flip-flop.
        self.rtc = rtc
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.buffer: deque[np.ndarray] = deque()
        self.buffer_norm: deque[torch.Tensor] = deque()  # normalized twins of buffer
        self.last_exec_norm: torch.Tensor | None = None
        self.future = None
        self.ticks_since_trigger = 0

    def reset(self) -> None:
        if self.future is not None:
            self.future.result()  # let a stale job finish; discard it
        self.future = None
        self.buffer.clear()
        self.buffer_norm.clear()
        self.last_exec_norm = None
        self.policy.reset()

    def _feed_obs(self, batch: dict) -> None:
        batch = dict(batch)
        if self.policy.config.image_features:
            batch[OBS_IMAGES] = torch.stack(
                [batch[k] for k in self.policy.config.image_features], dim=-4)
        with self.lock:
            self.policy._queues = populate_queues(self.policy._queues, batch)

    def _generate(self, prefix_norm: torch.Tensor | None) -> tuple[list[np.ndarray], torch.Tensor]:
        """Generate one chunk; if prefix_norm (P, adim) is given, inpaint it.

        prefix_norm rows are the normalized actions for horizon indices 0..P-1:
        index 0 is the last EXECUTED action (chunk j = -1), the rest are the
        still-buffered committed actions (chunk j = 0..). generate_actions'
        chunk slice starts at horizon index n_obs_steps-1 = 1, so this prefix
        occupies exactly the steps the new chunk must agree with.
        """
        with self.lock:  # snapshot obs history (cheap)
            stacked = {
                k: torch.stack(list(q), dim=1)
                for k, q in self.policy._queues.items()
                if k != "action" and len(q) > 0
            }
        diff = self.policy.diffusion
        with torch.inference_mode():  # expensive denoising — no lock held
            if prefix_norm is None or not self.rtc or len(prefix_norm) == 0:
                chunk = diff.generate_actions(stacked)  # (1, n, dim)
                chunk_norm = chunk
            else:
                global_cond = diff._prepare_global_conditioning(stacked)
                device, dtype = global_cond.device, global_cond.dtype
                horizon = diff.config.horizon
                adim = diff.config.action_feature.shape[0]
                known = prefix_norm.unsqueeze(0).to(device=device, dtype=dtype)  # (1,P,adim)
                P = known.shape[1]
                sample = torch.randn(1, horizon, adim, device=device, dtype=dtype)
                diff.noise_scheduler.set_timesteps(diff.num_inference_steps)
                for t in diff.noise_scheduler.timesteps:
                    ts = torch.as_tensor([t], device=device)
                    noised = diff.noise_scheduler.add_noise(known, torch.randn_like(known), ts)
                    sample[:, :P] = noised
                    model_output = diff.unet(
                        sample,
                        torch.full(sample.shape[:1], t, dtype=torch.long, device=device),
                        global_cond=global_cond,
                    )
                    sample = diff.noise_scheduler.step(model_output, t, sample).prev_sample
                sample[:, :P] = known  # exact clamp on the committed steps
                # Bridge the clamp boundary: with few DDIM steps the free region
                # harmonizes imperfectly with the clamped prefix, leaving a
                # 5-10 deg step exactly where the clamp ends (measured on-robot:
                # jumps cluster at +3 ticks after splices). Linearly bridge a
                # 4-step window from the last clamped action into the generated
                # tail — one mode, so this is smoothing, not mode-averaging.
                B = 4
                if P + B < horizon:
                    a0 = sample[:, P - 1]
                    a1 = sample[:, P + B]
                    for i in range(B):
                        w = (i + 1.0) / (B + 1.0)
                        sample[:, P + i] = (1 - w) * a0 + w * a1
                start = diff.config.n_obs_steps - 1
                chunk_norm = sample[:, start:start + diff.config.n_action_steps]
                chunk = chunk_norm
            out = []
            for i in range(chunk.shape[1]):
                out.append(self.post(chunk[:, i]).squeeze(0).cpu().numpy())
        return out, chunk_norm.squeeze(0).detach().cpu()

    def step(self, batch: dict) -> np.ndarray | None:
        """Feed one observation, return the action for this tick (None = warm-up)."""
        self._feed_obs(batch)
        self.ticks_since_trigger += 1

        if self.future is None and len(self.buffer) <= self.trigger:
            self.ticks_since_trigger = 0
            prefix = None
            if self.rtc and self.last_exec_norm is not None:
                prefix = torch.stack([self.last_exec_norm] + list(self.buffer_norm))
            self.future = self.pool.submit(self._generate, prefix)

        if (not self.buffer) or ((self.blend or self.rtc)
                                 and self.future is not None and self.future.done()):
            # Splice in the new chunk. Default: execute the old chunk to its
            # end, then hard-switch (time-aligned by dropping elapsed actions).
            # With RTC inpainting the new chunk is already a continuation of
            # the old one, so the hard switch is seamless by construction.
            # blend=True cross-fades instead (pre-RTC option, kept for A/B).
            chunk, chunk_norm = self.future.result()  # blocks only if generation outran the buffer
            self.future = None
            skip = min(self.ticks_since_trigger, len(chunk) - 1)
            new = chunk[skip:]
            new_norm = [chunk_norm[i] for i in range(skip, len(chunk))]
            if self.blend and not self.rtc:
                old = list(self.buffer)
                n_blend = min(len(old), len(new) - 1)
                self.buffer.clear()
                self.buffer_norm.clear()
                for i in range(n_blend):
                    w = (i + 1.0) / (n_blend + 1.0)  # ramp old -> new
                    self.buffer.append((1.0 - w) * old[i] + w * new[i])
                    self.buffer_norm.append(new_norm[i])
                self.buffer.extend(new[n_blend:])
                self.buffer_norm.extend(new_norm[n_blend:])
            else:
                self.buffer.clear()
                self.buffer_norm.clear()
                self.buffer.extend(new)
                self.buffer_norm.extend(new_norm)

        self.last_exec_norm = self.buffer_norm.popleft()
        return self.buffer.popleft()


def enter_pressed() -> bool:
    """Non-blocking check for a completed ENTER on stdin."""
    import select

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False


def median_first_frame_state(dataset_root: str, state_names: list[str],
                             val_episodes: set[int]) -> dict[str, float]:
    """Median first-frame observation.state over TRAIN episodes -> start pose."""
    import pandas as pd

    files = glob.glob(f"{dataset_root}/data/**/*.parquet", recursive=True)
    df = pd.concat([pd.read_parquet(f) for f in files])
    firsts = df[(df["frame_index"] == 0) & (~df["episode_index"].isin(val_episodes))]
    states = np.stack(firsts["observation.state"].to_numpy()).astype(np.float64)
    med = np.median(states, axis=0)
    return {name: float(med[i]) for i, name in enumerate(state_names)}


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- training dataset metadata: stats, state layout, cameras ---
    root = Path(args.train_dataset_root)
    meta = LeRobotDatasetMetadata(repo_id=root.name, root=root)
    state_names = meta.features["observation.state"]["names"]
    cam_features = sorted(meta.camera_keys)  # e.g. observation.images.forward / .wrist
    client_cams = [c.removeprefix("observation.images.") for c in cam_features]
    print(f"Policy inputs: state={state_names}  cameras={client_cams}")

    import json
    val_file = root / "meta" / "val_episodes.json"
    val_eps = set(json.loads(val_file.read_text())["val_episodes"]) if val_file.exists() else set()

    # --- policy ---
    cfg = PreTrainedConfig.from_pretrained(args.hf_model_id)
    cfg.pretrained_path = args.hf_model_id
    if cfg.type == "diffusion":
        cfg.noise_scheduler_type = args.scheduler.upper()
        if args.num_inference_steps > 0:
            cfg.num_inference_steps = args.num_inference_steps
        if args.n_action_steps > 0:
            cfg.n_action_steps = args.n_action_steps
        print(f"Diffusion sampler: {cfg.noise_scheduler_type}-{cfg.num_inference_steps}, "
              f"n_action_steps={cfg.n_action_steps}")
    policy = get_policy_class(cfg.type).from_pretrained(args.hf_model_id, config=cfg)
    policy = policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.hf_model_id, dataset_stats=meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    planner = (AsyncChunkPlanner(policy, postprocessor, device,
                                 trigger=args.chunk_trigger, blend=args.blend_splices,
                                 rtc=not args.no_rtc)
               if cfg.type == "diffusion" else None)
    if planner is not None:
        print(f"Chunk planner: RTC inpainting {'ON' if planner.rtc else 'off'}")

    # --- robot ---
    hw_shapes = {c: tuple(meta.features[f"observation.images.{c}"]["shape"]) for c in client_cams}
    robot = SO101ZmqClient(SO101ZmqClientConfig(
        remote_ip=args.remote_ip, zmq_cmd_port=args.zmq_cmd_port,
        zmq_obs_port=args.zmq_obs_port,
        cameras={c: hw_shapes[c] for c in client_cams},
    ))
    robot.connect()
    robot.enable_torque()

    reset_pose = None
    if not args.no_reset:
        reset_pose = median_first_frame_state(str(root), state_names, val_eps)
        print("Reset pose (median train first-frame):",
              {k: round(v, 1) for k, v in reset_pose.items()})

    def policy_obs() -> tuple[dict, dict[str, float]]:
        raw = robot.get_observation()
        obs = {"observation.state": np.array([raw[n] for n in state_names], dtype=np.float32)}
        for c in client_cams:
            obs[f"observation.images.{c}"] = np.ascontiguousarray(raw[c])
        joints = {n: float(raw[n]) for n in state_names}
        return obs, joints

    def ramp_to_pose(target: dict[str, float], seconds: float) -> None:
        """Move to `target` along a smooth interpolated path instead of letting
        the servos snap there at full speed (the host's go_home is a single
        goal-position jump)."""
        _, cur = policy_obs()
        n = max(1, int(seconds * 30))
        for i in range(1, n + 1):
            w = i / n
            robot.send_action({k: cur[k] + w * (target[k] - cur[k]) for k in state_names})
            precise_sleep(1.0 / 30)

    interval = 1.0 / args.fps
    try:
        for ep in range(args.num_episodes):
            if reset_pose is not None:
                log_say("Going to start pose")
                ramp_to_pose(reset_pose, args.home_seconds)
            input(f"\nReset the scene, then press ENTER to start episode {ep + 1} "
                  f"of {args.num_episodes}…")
            log_say(f"Episode {ep + 1}")
            print("Rolling out — press ENTER to stop this episode.")
            if planner is not None:
                planner.reset()
            else:
                policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            _, last_sent = policy_obs()  # seed limiter from the actual pose
            gr_closed = False  # binarize hysteresis state, reset per episode

            rows = []
            video_writers: dict = {}
            ep_stamp = time.strftime("%Y%m%d_%H%M%S")
            log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
            ep_start = time.perf_counter()
            t_end = ep_start + args.episode_time
            while time.perf_counter() < t_end:
                if enter_pressed():
                    print("Stopped by user.")
                    break
                t0 = time.perf_counter()
                obs, joints = policy_obs()
                # prepare_observation_for_inference mutates obs in place
                # (numpy -> CUDA tensors) — keep numpy refs for the video log.
                frames = {c: obs[f"observation.images.{c}"] for c in client_cams}
                with torch.inference_mode():
                    o = prepare_observation_for_inference(
                        obs, torch.device(device), args.task_description, robot.name)
                    o = preprocessor(o)
                if planner is not None:
                    raw = planner.step(o)
                    buf_len = len(planner.buffer)
                else:
                    with torch.inference_mode():
                        raw = postprocessor(policy.select_action(o)).squeeze(0).cpu().numpy()
                    buf_len = -1
                prev = dict(last_sent)
                cmd = {n: float(raw[i]) for i, n in enumerate(state_names)}
                if args.action_ema > 0:
                    a = args.action_ema
                    cmd = {n: a * cmd[n] + (1 - a) * prev[n] for n in state_names}
                if args.gripper_binarize:
                    g = cmd["gripper.pos"]
                    if gr_closed and g > 75:
                        gr_closed = False
                    elif not gr_closed and g < 60:
                        gr_closed = True
                    cmd["gripper.pos"] = args.gripper_closed_cmd if gr_closed else 100.0
                elif args.gripper_close_bias > 0 and cmd["gripper.pos"] < 45:
                    cmd["gripper.pos"] = max(cmd["gripper.pos"] - args.gripper_close_bias, 0.0)
                if args.max_delta_per_tick > 0:
                    for n in state_names:
                        d = (args.gripper_max_delta or float("inf")) if n == "gripper.pos" \
                            else args.max_delta_per_tick
                        cmd[n] = min(max(cmd[n], last_sent[n] - d), last_sent[n] + d)
                        last_sent[n] = cmd[n]
                if args.interp_substeps > 1:
                    S = args.interp_substeps
                    feed_mid = planner is not None and args.obs_history_hz > args.fps
                    for s in range(1, S + 1):
                        w = s / S
                        robot.send_action({n: prev[n] + w * (cmd[n] - prev[n])
                                           for n in state_names})
                        if s < S:
                            precise_sleep(interval / S)
                            if feed_mid:
                                # fresh 30 Hz frame exists on the stream; feed the
                                # history queue only (no policy step) so replans
                                # see training-native 33 ms obs pairs
                                obs_m, _ = policy_obs()
                                with torch.inference_mode():
                                    o_m = prepare_observation_for_inference(
                                        obs_m, torch.device(device),
                                        args.task_description, robot.name)
                                    o_m = preprocessor(o_m)
                                planner._feed_obs(o_m)
                    # outer precise_sleep covers the final sub-interval
                else:
                    robot.send_action(cmd)
                rows.append([time.perf_counter() - ep_start, buf_len]
                            + [joints[n] for n in state_names]          # observed
                            + [float(raw[i]) for i in range(len(state_names))]  # policy raw
                            + [cmd[n] for n in state_names])            # sent (post-limiter)
                for c in client_cams:
                    frame = frames[c]
                    if c not in video_writers:
                        h, w = frame.shape[:2]
                        video_writers[c] = cv2.VideoWriter(
                            str(log_dir / f"rollout_{ep_stamp}_ep{ep}_{c}.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                    video_writers[c].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                precise_sleep(max(interval - (time.perf_counter() - t0), 0.0))
            else:
                print(f"Episode hit the {args.episode_time:.0f}s safety cap.")
            for vw in video_writers.values():
                vw.release()
            if video_writers:
                print(f"Videos: {log_dir}/rollout_{ep_stamp}_ep{ep}_*.mp4")
            out = log_dir / f"rollout_{ep_stamp}_ep{ep}.npz"
            np.savez_compressed(out, rows=np.array(rows, dtype=np.float32),
                                columns=np.array(["t", "buffer_len"]
                                                 + [f"obs.{n}" for n in state_names]
                                                 + [f"raw.{n}" for n in state_names]
                                                 + [f"cmd.{n}" for n in state_names]),
                                config_json=json.dumps(vars(args)))
            print(f"Trajectory log: {out} ({len(rows)} ticks)")
            log_say("Episode done")
        if reset_pose is not None:
            ramp_to_pose(reset_pose, args.home_seconds)  # leave the arm parked at the start pose
    finally:
        robot.disconnect()
        print("Rollout session finished.")


if __name__ == "__main__":
    main()
