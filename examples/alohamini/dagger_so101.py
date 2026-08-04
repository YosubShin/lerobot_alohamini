#!/usr/bin/env python3
"""Shadowing DAgger: policy drives the follower; the LEADER servo-tracks it
with soft gains under your resting hands. Deviate the leader past a threshold
and control flips to you instantly (leader torque drops the same tick) —
the intervention is recorded from the exact in-motion failure state, teleop
semantics (action = leader pose), into a growing `_dagger` dataset.

Lineage: HG-DAgger (Kelly et al. 2019) / intervention-based learning
(Sirius, Liu et al. 2022), shadowing variant — takeover latency ~0 because
the leader is always aligned.

Per-round flow:
  home -> place block -> ENTER -> policy rolls out, leader shadows
    - SPACE = take over (explicit trigger; the leader-follower offset at
      that instant is bled off over ~1 s so the follower never snaps) ->
      demonstrate the recovery -> ENTER saves the intervention (r discards)
    - ENTER = rollout SUCCEEDED: save the policy's own trajectory to the
      --success_dataset (Sirius/RFT-style self-imitation, human-labeled)
    - s = end round, record nothing
  q quits the session (post-session 1080p extraction runs as usual).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.teleoperators.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_so101 import AsyncChunkPlanner, median_first_frame_state  # noqa: E402
from record_so101 import (  # noqa: E402
    _MIN_OBS_RATE_FRACTION,
    FocusKeyboard,
    _extract_session_1080p,
    _frozen_cam_banner,
    _log_episode_wallclock,
    _measure_obs_rate,
    _obs_rate_banner,
)
from so101_zmq_client import SO101ZmqClient, SO101ZmqClientConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shadowing-DAgger intervention collection")
    p.add_argument("--hf_model_id", type=str, required=True)
    p.add_argument("--train_dataset_root", type=str, required=True)
    p.add_argument("--dagger_dataset", type=str, required=True,
                   help="e.g. local/so101_dagger_v1 — intervention episodes land here")
    p.add_argument("--task_description", type=str, default="SO-101 teleop demonstration")
    p.add_argument("--remote_ip", type=str, default="192.168.0.17")
    p.add_argument("--zmq_cmd_port", type=int, default=5601)
    p.add_argument("--zmq_obs_port", type=int, default=5602)
    p.add_argument("--leader_port", type=str, default="/dev/ttyACM0")
    p.add_argument("--leader_id", type=str, default="my_leader_arm")
    p.add_argument("--leader_profile", type=str, default="so-arm-5dof")
    # policy/inference (current best defaults)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--interp_substeps", type=int, default=2)
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--chunk_trigger", type=int, default=6)
    p.add_argument("--max_delta_per_tick", type=float, default=7.0)
    p.add_argument("--gripper_max_delta", type=float, default=20.0)
    # shadowing
    p.add_argument("--success_dataset", type=str, default="",
                   help="Where human-labeled successful policy rollouts go "
                        "(default: <dagger_dataset>_success)")
    p.add_argument("--handover_decay", type=float, default=0.85,
                   help="Per-tick decay of the leader-follower offset captured at "
                        "takeover (0.85 ~ 1 s bleed-off; prevents the snap)")
    p.add_argument("--leader_p", type=int, default=20,
                   help="Leader tracking P gain. P sets responsiveness, not max "
                        "torque — a hand overpowers the servo at any P, so track "
                        "firmly (20) rather than softly (8, which lags motion and "
                        "sags under gravity)")
    p.add_argument("--leader_d", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--archive_dir", type=str, default="/mnt/nvme/lerobot/fisheye_archive")
    p.add_argument("--remote_user", type=str, default="yosub")
    p.add_argument("--no_extract_1080p", action="store_true")
    p.add_argument("--home_seconds", type=float, default=3.0)
    p.add_argument("--rollout_time", type=float, default=60.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(args.train_dataset_root)
    meta = LeRobotDatasetMetadata(repo_id=root.name, root=root)
    state_names = meta.features["observation.state"]["names"]
    cam_features = sorted(meta.camera_keys)
    client_cams = [c.removeprefix("observation.images.") for c in cam_features]
    motor_names = [n.removesuffix(".pos") for n in state_names]
    print(f"Policy inputs: state={state_names} cams={client_cams}")

    cfg = PreTrainedConfig.from_pretrained(args.hf_model_id)
    cfg.pretrained_path = args.hf_model_id
    if cfg.type == "diffusion":
        cfg.noise_scheduler_type = args.scheduler.upper()
        cfg.num_inference_steps = args.num_inference_steps
        cfg.n_action_steps = args.n_action_steps
    policy = get_policy_class(cfg.type).from_pretrained(args.hf_model_id, config=cfg).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.hf_model_id, dataset_stats=meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}})
    planner = AsyncChunkPlanner(policy, postprocessor, device,
                                trigger=args.chunk_trigger, blend=False, rtc=True)

    hw_shapes = {c: tuple(meta.features[f"observation.images.{c}"]["shape"]) for c in client_cams}
    robot = SO101ZmqClient(SO101ZmqClientConfig(
        remote_ip=args.remote_ip, zmq_cmd_port=args.zmq_cmd_port,
        zmq_obs_port=args.zmq_obs_port, cameras={c: hw_shapes[c] for c in client_cams}))
    robot.connect()
    robot.archive_start()
    time.sleep(3.0)

    print("Pre-flight: measuring host observation rate (3 s)…", flush=True)
    hz, frozen = _measure_obs_rate(robot)
    needed = args.fps * _MIN_OBS_RATE_FRACTION
    if (hz is not None and hz < needed) or frozen:
        if hz is not None and hz < needed:
            print(_obs_rate_banner(hz, needed), flush=True)
        if frozen:
            print(_frozen_cam_banner(frozen), flush=True)
        robot.archive_stop(); robot.disconnect()
        raise SystemExit(1)
    print(f"Observation rate OK: {hz:.1f} Hz", flush=True)

    leader = SOLeader(SOLeaderTeleopConfig(
        port=args.leader_port, id=args.leader_id, arm_profile=args.leader_profile))
    leader.connect()

    def leader_track_gains() -> None:
        for m in motor_names:
            leader.bus.write("P_Coefficient", m, args.leader_p, normalize=False)
            leader.bus.write("I_Coefficient", m, 0, normalize=False)
            leader.bus.write("D_Coefficient", m, args.leader_d, normalize=False)

    leader_track_gains()

    def leader_shadow(joints: dict[str, float]) -> None:
        leader.bus.sync_write("Goal_Position", {m: joints[f"{m}.pos"] for m in motor_names})

    ARM = [n for n in state_names if n != "gripper.pos"]  # gripper excluded:
    # the trigger servo carries the operator's resting finger

    def leader_offsets(joints: dict[str, float]) -> dict[str, float]:
        lp = leader.get_action()
        return {n: lp[n] - joints[n] for n in ARM}

    # dagger dataset (teleop semantics)
    action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True)
    if args.resume:
        dataset = LeRobotDataset.resume(repo_id=args.dagger_dataset,
                                        root=HF_LEROBOT_HOME / args.dagger_dataset,
                                        image_writer_threads=4)
        print("Resuming dagger dataset:", args.dagger_dataset)
    else:
        dataset = LeRobotDataset.create(repo_id=args.dagger_dataset, fps=args.fps,
                                        features={**action_features, **obs_features},
                                        robot_type=robot.name, use_videos=True,
                                        image_writer_threads=4)
        print("Created dagger dataset:", args.dagger_dataset,
              f"(NOTE: recorded at {args.fps} fps — the deploy tick)")

    success_repo = args.success_dataset or (args.dagger_dataset + "_success")
    sroot = HF_LEROBOT_HOME / success_repo
    if sroot.exists():
        sdataset = LeRobotDataset.resume(repo_id=success_repo, root=sroot,
                                         image_writer_threads=4)
        print("Resuming success dataset:", success_repo)
    else:
        sdataset = LeRobotDataset.create(repo_id=success_repo, fps=args.fps,
                                         features={**action_features, **obs_features},
                                         robot_type=robot.name, use_videos=True,
                                         image_writer_threads=4)
        print("Created success dataset:", success_repo)

    reset_pose = median_first_frame_state(str(root), state_names, set())

    def policy_obs():
        raw = robot.get_observation()
        obs = {"observation.state": np.array([raw[n] for n in state_names], dtype=np.float32)}
        for c in client_cams:
            obs[f"observation.images.{c}"] = np.ascontiguousarray(raw[c])
        joints = {n: float(raw[n]) for n in state_names}
        return raw, obs, joints

    def ramp_to(target: dict[str, float], seconds: float) -> None:
        _, _, cur = policy_obs()
        n = max(1, int(seconds * 30))
        for i in range(1, n + 1):
            w = i / n
            robot.send_action({k: cur[k] + w * (target[k] - cur[k]) for k in state_names})
            precise_sleep(1.0 / 30)

    kb = FocusKeyboard()
    kb.start()
    interval = 1.0 / args.fps
    saved = 0
    n_success = 0
    print("\nSHADOWING DAGGER READY.")
    print("  ENTER=start rollout, ENTER again=save as SUCCESS | SPACE=take over | s=end round")
    print("  after takeover: ENTER=save intervention  r=discard | q=quit\n", flush=True)

    try:
        while True:
            robot.enable_torque()
            ramp_to(reset_pose, args.home_seconds)
            kb.clear()
            print(f"\n[{saved} interventions, {n_success} successes]  Place the block, ENTER to roll out (q=quit)…",
                  flush=True)
            while True:
                pressed = kb.get_pressed()
                if "q" in pressed:
                    return
                if "\r" in pressed or "\n" in pressed:
                    break
                # keep leader converged to follower while idle
                _, _, j = policy_obs()
                leader.bus.enable_torque()
                leader_shadow(j)
                precise_sleep(0.05)
            kb.clear()

            planner.reset()
            preprocessor.reset()
            postprocessor.reset()
            _, _, last_sent = policy_obs()
            mode = "policy"
            handover_offset = None
            rollout_frames = []   # policy trajectory buffer (kept only on ENTER)
            prebuf = []           # last 2 policy-mode frames; prepended at takeover
                                  # so intervention episodes carry the DYNAMIC
                                  # failure observation history (train with
                                  # LEROBOT_DROP_N_FIRST=2: obs context only,
                                  # stored policy actions never supervised)
            for _ in range(6):    # let the leader settle onto the follower
                _, _, j0 = policy_obs()
                leader_shadow(j0)
                precise_sleep(1.0 / args.fps)
            t_end = time.perf_counter() + args.rollout_time
            print("Policy rolling — SPACE to take over, ENTER when it succeeds…", flush=True)

            while time.perf_counter() < t_end:
                t0 = time.perf_counter()
                raw, obs, joints = policy_obs()
                pressed = kb.get_pressed()
                if "q" in pressed:
                    return
                if mode == "policy":
                    if "s" in pressed:
                        print("Round ended (nothing recorded).", flush=True)
                        break
                    if "\r" in pressed or "\n" in pressed or "g" in pressed:
                        for fr in rollout_frames:
                            sdataset.add_frame(fr)
                        if sdataset.has_pending_frames():
                            sdataset.save_episode()
                            _log_episode_wallclock(sdataset, time.time(), time.time())
                            n_success += 1
                            print(f"SUCCESS rollout saved ({n_success} total).", flush=True)
                        break
                    leader_shadow(joints)
                    if " " in pressed:
                        leader.bus.disable_torque()   # leader instantly passive
                        # capture the leader-follower mismatch at this instant and
                        # bleed it off — the follower must NOT snap to the leader
                        lp = leader.get_action()
                        handover_offset = {n: lp[n] - joints[n] for n in state_names}
                        mode = "takeover"
                        for fr in prebuf:   # dynamic pre-takeover context frames
                            dataset.add_frame(dict(fr))
                        print("\n>>> TAKEOVER — you have control (offset bleeding off). "
                              "Recover, then ENTER=save r=discard", flush=True)
                        continue
                    with torch.inference_mode():
                        o = prepare_observation_for_inference(
                            dict(obs), torch.device(device), args.task_description, robot.name)
                        o = preprocessor(o)
                    a = planner.step(o)
                    cmd = {n: float(a[i]) for i, n in enumerate(state_names)}
                    prev = dict(last_sent)
                    for n in state_names:
                        d = args.gripper_max_delta if n == "gripper.pos" else args.max_delta_per_tick
                        cmd[n] = min(max(cmd[n], last_sent[n] - d), last_sent[n] + d)
                        last_sent[n] = cmd[n]
                    if args.interp_substeps > 1:
                        S = args.interp_substeps
                        for s_ in range(1, S + 1):
                            w = s_ / S
                            robot.send_action({n: prev[n] + w * (cmd[n] - prev[n])
                                               for n in state_names})
                            if s_ < S:
                                precise_sleep(interval / S)
                    else:
                        robot.send_action(cmd)
                    oframe = build_dataset_frame(dataset.features, raw, prefix=OBS_STR)
                    aframe = build_dataset_frame(dataset.features, cmd, prefix=ACTION)
                    fr = {**oframe, **aframe, "task": args.task_description}
                    rollout_frames.append(fr)
                    prebuf.append(fr)
                    if len(prebuf) > 2:
                        prebuf.pop(0)
                else:  # takeover: teleop + record
                    if "\r" in pressed or "\n" in pressed:
                        if dataset.has_pending_frames():
                            dataset.save_episode()
                            _log_episode_wallclock(dataset, time.time(), time.time())
                            saved += 1
                            print(f"Intervention SAVED ({saved} total).", flush=True)
                        break
                    if "r" in pressed:
                        dataset.clear_episode_buffer()
                        print("Intervention discarded.", flush=True)
                        break
                    lp = leader.get_action()
                    act = {n: lp[n] - handover_offset[n] for n in state_names}
                    for n in state_names:
                        handover_offset[n] *= args.handover_decay
                    robot.send_action(act)
                    oframe = build_dataset_frame(dataset.features, raw, prefix=OBS_STR)
                    aframe = build_dataset_frame(dataset.features, act, prefix=ACTION)
                    dataset.add_frame({**oframe, **aframe, "task": args.task_description})
                el = time.perf_counter() - t0
                precise_sleep(max(interval - el, 0.0))
            else:
                print("Rollout time cap reached.", flush=True)
            if mode == "takeover" and dataset.has_pending_frames():
                dataset.clear_episode_buffer()  # cap hit mid-takeover: discard partial
    finally:
        kb.stop()
        try:
            leader.bus.disable_torque()
            leader.disconnect()
        except Exception:
            pass
        try:
            robot.archive_stop(); time.sleep(0.3)
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        dataset.finalize()
        try:
            sdataset.finalize()
        except Exception:
            pass
        print(f"\nSession done: {saved} interventions in {args.dagger_dataset}, "
              f"{n_success} successes in {success_repo}")
        if saved and not args.no_extract_1080p:
            _extract_session_1080p(args, dataset.root)


if __name__ == "__main__":
    main()
