#!/usr/bin/env python3
"""Fair, blinded, multi-checkpoint eval for AlohaMini.

For each block PLACEMENT you set up, this runs every checkpoint on that *identical* scene
(so eyeballing the block position is fine — the block doesn't move between checkpoints),
resetting the arm to a consistent in-distribution start pose before each rollout, in a
shuffled/blinded order so your scoring isn't biased toward the model you expect to win.

Workflow per placement:
  place block -> ENTER -> [reset arm -> run ckpt (blinded) -> you score y/n] x N -> repeat.

No marks go in the camera frame; the reset pose is the median first-frame state of the
training episodes (the true in-distribution start). Results -> CSV + a success matrix.

Example:
  python examples/alohamini/eval_compare.py \
    --checkpoints "full_10k=/mnt/.../pi05_am1_red_bin_20260717_171335/checkpoints/010000/pretrained_model,\
full_5k=/mnt/.../checkpoints/005000/pretrained_model,\
expert_10k=/mnt/.../pi05_am1_red_bin_expert_.../checkpoints/010000/pretrained_model" \
    --task_description "Put the red block into the bin with the right arm" \
    --remote_ip 192.168.0.50 --robot_model alohamini1
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

import lerobot.robots.alohamini  # noqa: F401 — registers alohamini_client
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.rollout.inference.factory import SyncInferenceConfig, create_inference_engine
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.device_utils import auto_select_torch_device
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


def parse_checkpoints(spec: str):
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            p = Path(path)
            label = p.parts[-2] if p.name == "pretrained_model" else p.name
        out.append((label.strip(), path.strip()))
    return out


def compute_reset_pose(train_root: str, ordered_action_keys):
    """Median of each training episode's FIRST-frame observation.state -> in-distribution start."""
    import glob
    import pandas as pd

    files = glob.glob(f"{train_root}/data/**/*.parquet", recursive=True)
    df = pd.concat([pd.read_parquet(f) for f in files])
    firsts = df[df["frame_index"] == 0]
    states = np.stack(firsts["observation.state"].to_numpy()).astype(np.float64)  # [n_ep, 16]
    med = np.median(states, axis=0)
    return {k: float(med[i]) for i, k in enumerate(ordered_action_keys)}


def drive_to_pose(robot, robot_action_processor, pose_dict, fps, seconds):
    """Command a fixed pose repeatedly so the arm settles there (base vel ~0 = stop)."""
    interval = 1.0 / fps
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        obs_raw = robot.get_observation()
        robot.send_action(robot_action_processor((dict(pose_dict), obs_raw)))
        time.sleep(interval)


def run_rollout(robot, engine, robot_observation_processor, robot_action_processor,
                dataset_features, ordered_action_keys, task, fps, seconds, abort):
    engine.reset()
    interval = 1.0 / fps
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < seconds and not abort["stop"]:
        loop_start = time.perf_counter()
        obs_raw = robot.get_observation()
        obs_processed = robot_observation_processor(obs_raw)
        obs_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
        action_tensor = engine.get_action(obs_frame)
        if action_tensor is not None:
            action_dict = {k: action_tensor[i].item() for i, k in enumerate(ordered_action_keys)}
            robot.send_action(robot_action_processor((action_dict, obs_raw)))
        dt = time.perf_counter() - loop_start
        if (sleep_t := interval - dt) > 0:
            precise_sleep(sleep_t)


def main():
    ap = argparse.ArgumentParser(description="Blinded multi-checkpoint eval for AlohaMini")
    ap.add_argument("--checkpoints", required=True, help="comma list of 'label=path' (or bare paths)")
    ap.add_argument("--task_description", required=True, help="MUST match the training prompt")
    ap.add_argument("--remote_ip", default="192.168.0.50")
    ap.add_argument("--robot_id", default="lekiwi")
    ap.add_argument("--robot_model", default="alohamini1")
    ap.add_argument("--train_dataset_root", default="/mnt/nvme/lerobot/yosubshin/am1_red_bin",
                    help="training dataset root — for the reset pose + normalization stats")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--episode_time", type=int, default=25, help="seconds per rollout")
    ap.add_argument("--reset_time", type=float, default=3.0, help="seconds to settle to the start pose")
    ap.add_argument("--results_csv", default="/mnt/nvme/lerobot/eval_compare_results.csv")
    ap.add_argument("--show_labels", action="store_true", help="disable blinding (show which ckpt is running)")
    args = ap.parse_args()

    ckpts = parse_checkpoints(args.checkpoints)
    device = str(auto_select_torch_device())
    print(f"Device: {device}   Checkpoints: {[l for l, _ in ckpts]}")

    # --- Robot (once) ---
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id,
                                            robot_model=args.robot_model))
    robot.connect()
    robot_wrapper = ThreadSafeRobot(robot)

    # --- Shared processors + feature schema (once) ---
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    obs_hw = robot.observation_features
    act_hw = robot.action_features
    action_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor, initial_features=create_initial_features(action=act_hw), use_videos=True)
    observation_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor, initial_features=create_initial_features(observation=obs_hw), use_videos=True)
    dataset_features = combine_feature_dicts(action_features, observation_features)
    hw_features = hw_to_dataset_features(obs_hw, "observation")
    ordered_action_keys = list(act_hw.keys())

    train_stats = LeRobotDatasetMetadata("yosubshin/am1_red_bin", root=Path(args.train_dataset_root)).stats
    reset_pose = compute_reset_pose(args.train_dataset_root, ordered_action_keys)
    print("Reset pose (start) computed from training first-frames.")

    # --- Load every checkpoint's policy + engine (kept resident) ---
    engines = []
    for label, path in ckpts:
        print(f"Loading {label} <- {path} ...")
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(device)
        policy.eval()
        pre, post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=path, dataset_stats=train_stats,
            preprocessor_overrides={"device_processor": {"device": device}})
        engine = create_inference_engine(
            SyncInferenceConfig(), policy=policy, preprocessor=pre, postprocessor=post,
            robot_wrapper=robot_wrapper, hw_features=hw_features, dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys, task=args.task_description, fps=float(args.fps), device=device)
        engine.start()
        engines.append((label, engine))
    print(f"All {len(engines)} checkpoints resident. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    init_rerun(session_name="alohamini_eval_compare")

    # --- Esc = abort the current rollout early ---
    abort = {"stop": False}
    try:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.esc:
                abort["stop"] = True
        keyboard.Listener(on_press=on_press).start()
        print("(Esc aborts the current rollout early.)")
    except Exception as e:
        print(f"(No Esc listener: {e}. Ctrl+C is the hard stop.)")

    results = []  # (placement, label, success)
    placement = 0
    print("\n" + "=" * 70)
    print("Place the block for placement 1. Nothing but the scene should be in frame.")
    while True:
        cmd = input(f"\n[Placement {placement + 1}] Block set? ENTER to run all {len(engines)} ckpts, or 'q' to finish: ").strip().lower()
        if cmd == "q":
            break
        placement += 1
        order = list(range(len(engines)))
        if not args.show_labels:
            random.shuffle(order)  # blind: randomize which model runs when

        for trial_i, idx in enumerate(order, 1):
            label, engine = engines[idx]
            shown = label if args.show_labels else f"trial {trial_i}/{len(engines)}"
            print(f"  -> {shown}: resetting arm to start pose...")
            drive_to_pose(robot, robot_action_processor, reset_pose, args.fps, args.reset_time)
            log_say(f"Placement {placement}, {shown}")
            abort["stop"] = False
            print(f"  -> {shown}: RUNNING ({args.episode_time}s). Esc = end early, Ctrl+C = emergency stop.")
            run_rollout(robot, engine, robot_observation_processor, robot_action_processor,
                        dataset_features, ordered_action_keys, args.task_description,
                        args.fps, args.episode_time, abort)
            while True:
                s = input(f"     [{shown}] success? y / n / r (redo): ").strip().lower()
                if s in ("y", "n"):
                    results.append((placement, label, 1 if s == "y" else 0))
                    break
                if s == "r":
                    drive_to_pose(robot, robot_action_processor, reset_pose, args.fps, args.reset_time)
                    abort["stop"] = False
                    print(f"  -> {shown}: RE-RUNNING ({args.episode_time}s).")
                    run_rollout(robot, engine, robot_observation_processor, robot_action_processor,
                                dataset_features, ordered_action_keys, args.task_description,
                                args.fps, args.episode_time, abort)

    # --- Reveal + summary ---
    drive_to_pose(robot, robot_action_processor, reset_pose, args.fps, args.reset_time)
    for _, engine in engines:
        engine.stop()
    robot.disconnect()

    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["placement", "checkpoint", "success"])
        w.writerows(results)

    print("\n" + "=" * 70 + "\nSUCCESS RATE BY CHECKPOINT")
    for label, _ in engines:
        rows = [s for p, lbl, s in results if lbl == label]
        n = len(rows)
        print(f"  {label:16s} {sum(rows)}/{n}  ({100*sum(rows)/n:.0f}%)" if n else f"  {label:16s} (no trials)")
    print(f"\nPer-trial results -> {args.results_csv}")


if __name__ == "__main__":
    main()
