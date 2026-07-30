#!/usr/bin/env python3
"""Evaluate a gripper-only policy on the AlohaMini.

Same rollout loop as evaluate_bi.py, but adapted for policies trained on the
gripper-only observation space (see am1_red_bin_gripperonly_2cam):
  - observation.state fed to the policy is sliced to the gripper dims only
    (the policy never sees arm/base/lift kinematics — it learned IK from vision)
  - wrist_left camera is dropped from the policy input (policy trained on
    forward + wrist_right)
  - normalization stats come from the TRAINING dataset, not the freshly
    created eval-recording dataset (whose stats are empty)
  - actions: only the right arm is driven. Left arm and lift are frozen at
    their observed pose and the base is held at zero velocity via the robot
    client's component mask (same mechanism used during recording with
    --enable_left_arm false --enable_base false --enable_lift false), so the
    policy's noise predictions for those constant dims never move hardware.
  - inference runs on the RTX 3090 by default (so it never contends with a
    training job on the RTX 6000). Override by exporting CUDA_VISIBLE_DEVICES
    before launch.

The eval recording dataset still stores the FULL robot observation (16-dim
state, all cameras) so recordings stay complete for later analysis.

Example:
    python examples/alohamini/evaluate_bi_gripperonly.py \
        --hf_model_id /mnt/nvme/lerobot/outputs/dp_am1_red_bin_go2cam_20260728_091858/checkpoints/005000/pretrained_model \
        --train_dataset_root /mnt/nvme/lerobot/yosubshin/am1_red_bin_gripperonly_2cam \
        --hf_dataset_id yosubshin/am1_red_bin_go2cam_eval_5k \
        --task_description "Put the red block into the bin with the right arm" \
        --remote_ip 192.168.0.50 --robot_model alohamini1 \
        --num_episodes 5 --episode_time 30 --fps 30
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

# Pin inference to the RTX 3090 unless the caller already chose a device.
# Must happen before torch is imported (directly or via lerobot). Note CUDA
# and nvidia-smi enumerate this machine's GPUs in opposite order — hence UUID.
RTX_3090_UUID = "GPU-0f2b19f7-8074-ec55-de85-6b09c4ce5ffa"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", RTX_3090_UUID)

import lerobot.robots.alohamini  # noqa: F401 — registers alohamini_client robot type

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.rollout.inference.factory import SyncInferenceConfig, create_inference_engine
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.device_utils import auto_select_torch_device
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

def compute_reset_pose(reset_root: str, ordered_action_keys, max_episode: int | None):
    """Median of training episodes' FIRST-frame 16-dim observation.state.

    Starting each rollout from an in-distribution pose matters a lot for this
    policy: it has no joint-state input, so an out-of-distribution arm pose
    means an out-of-distribution wrist-camera view, and its absolute joint
    predictions can command a violent jump from wherever the arm happens to be.
    Reads from the ORIGINAL (pre-derivation) dataset, whose state is 16-dim.
    """
    import glob

    import pandas as pd

    files = glob.glob(f"{reset_root}/data/**/*.parquet", recursive=True)
    df = pd.concat([pd.read_parquet(f) for f in files])
    firsts = df[df["frame_index"] == 0]
    if max_episode is not None:
        firsts = firsts[firsts["episode_index"] < max_episode]
    states = np.stack(firsts["observation.state"].to_numpy()).astype(np.float64)
    med = np.median(states, axis=0)
    return {k: float(med[i]) for i, k in enumerate(ordered_action_keys)}


def drive_to_pose(robot, robot_action_processor, pose_dict, fps, seconds):
    """Command a fixed pose repeatedly so the arm settles there (masked components stay frozen)."""
    interval = 1.0 / fps
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        obs_raw = robot.get_observation()
        robot.send_action(robot_action_processor((dict(pose_dict), obs_raw)))
        time.sleep(interval)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AlohaMini with a gripper-only policy")
    parser.add_argument("--num_episodes", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode_time", type=int, default=60)
    parser.add_argument("--task_description", type=str, default="robot task")
    parser.add_argument("--hf_model_id", type=str, required=True)
    parser.add_argument("--hf_dataset_id", type=str, required=True)
    parser.add_argument("--train_dataset_root", type=str,
                        default="/mnt/nvme/lerobot/yosubshin/am1_red_bin_gripperonly_2cam",
                        help="Root of the dataset the policy was trained on (for stats + state layout)")
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1")
    parser.add_argument("--robot_id", type=str, default="lekiwi")
    parser.add_argument("--robot_model", type=str, default="alohamini1",
                        choices=["alohamini1", "alohamini2", "alohamini2pro"],
                        help="Must match the robot_model on the Pi host side")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the eval recording dataset to the hub when done")
    # Component masks — defaults match how the training data was recorded
    # (right arm only). Disabled components are frozen by the robot client.
    parser.add_argument("--enable_left_arm", type=parse_bool, default=False)
    parser.add_argument("--enable_right_arm", type=parse_bool, default=True)
    parser.add_argument("--enable_base", type=parse_bool, default=False)
    parser.add_argument("--enable_lift", type=parse_bool, default=False)
    # Diffusion-policy inference sampler. DDPM-100 takes ~1.0s/chunk on the
    # 3090 — longer than the 32-tick chunk covers at 30fps, so the arm stalls
    # at every chunk boundary. DDIM-10 takes ~0.1s and fits the budget.
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddim",
                        help="Noise scheduler for diffusion inference (ignored by other policy types)")
    parser.add_argument("--num_inference_steps", type=int, default=10,
                        help="Denoising steps at inference (diffusion only; 0 = policy default)")
    # Start-pose reset (same idea as eval_compare.py): before each episode,
    # drive the arm to the median first-frame pose of the training episodes.
    parser.add_argument("--reset", type=parse_bool, default=True)
    parser.add_argument("--reset_dataset_root", type=str,
                        default="/mnt/nvme/lerobot/yosubshin/am1_red_bin",
                        help="Original dataset with 16-dim state (for the reset pose)")
    parser.add_argument("--reset_max_episode", type=int, default=30,
                        help="Use first-frames of episodes < this index (train split only)")
    parser.add_argument("--reset_seconds", type=float, default=4.0)
    # Smoothing. The policy's chunks are smooth internally but successive chunks
    # can disagree by ~10+ units on uncertain states, causing a snap at every
    # chunk boundary. The slew limiter clips per-tick position changes to demo
    # speed (training p99 across right-arm joints is ~2-4 units/tick), which
    # leaves in-distribution motion untouched. Shorter chunks (n_action_steps)
    # replan more often so each boundary's disagreement is smaller.
    parser.add_argument("--max_delta_per_tick", type=float, default=4.0,
                        help="Max change per control tick for .pos action keys (0 = no limit). "
                             "Tune to the training data's speed: teleop am1_red_bin p99 ≈ 4.0; "
                             "kinesthetic so101_red_pick p99.9 ≈ 6.2 (use ~7 there — 4.0 would "
                             "clip legitimate demo-speed motion)")
    parser.add_argument("--n_action_steps", type=int, default=16,
                        help="Actions executed per chunk before replanning (0 = checkpoint default)")
    args = parser.parse_args()

    device = str(auto_select_torch_device())

    # === Policy ===
    policy_cfg = PreTrainedConfig.from_pretrained(args.hf_model_id)
    policy_cfg.pretrained_path = args.hf_model_id
    if policy_cfg.type == "diffusion":
        policy_cfg.noise_scheduler_type = args.scheduler.upper()
        if args.num_inference_steps > 0:
            policy_cfg.num_inference_steps = args.num_inference_steps
        if args.n_action_steps > 0:
            policy_cfg.n_action_steps = args.n_action_steps
        print(f"Diffusion sampler: {policy_cfg.noise_scheduler_type}-{policy_cfg.num_inference_steps}, "
              f"n_action_steps: {policy_cfg.n_action_steps}")
    policy = get_policy_class(policy_cfg.type).from_pretrained(args.hf_model_id, config=policy_cfg)
    policy = policy.to(device)
    policy.eval()

    # === Training dataset metadata (stats + the state layout the policy expects) ===
    train_root = Path(args.train_dataset_root)
    train_meta = LeRobotDatasetMetadata(repo_id=train_root.name, root=train_root)
    policy_state_names = train_meta.features["observation.state"]["names"]
    policy_camera_keys = set(train_meta.camera_keys)
    print(f"Policy cameras: {sorted(policy_camera_keys)}")

    # === Robot ===
    robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id,
                                      robot_model=args.robot_model,
                                      enable_left_arm=args.enable_left_arm,
                                      enable_right_arm=args.enable_right_arm,
                                      enable_base=args.enable_base,
                                      enable_lift=args.enable_lift)
    robot = LeKiwiClient(robot_config)
    robot.connect()
    robot_wrapper = ThreadSafeRobot(robot)

    # === Processors ===
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # === Dataset features (full robot obs — used for the eval recording) ===
    observation_features_hw = robot.observation_features
    action_features_hw = robot.action_features

    action_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=action_features_hw),
        use_videos=True,
    )
    observation_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=observation_features_hw),
        use_videos=True,
    )
    dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)
    hw_features = hw_to_dataset_features(observation_features_hw, "observation")
    ordered_action_keys = list(action_features_hw.keys())

    # Map robot state names -> indices of the dims the policy was trained on.
    robot_state_names = dataset_features["observation.state"]["names"]
    try:
        policy_state_idx = [robot_state_names.index(n) for n in policy_state_names]
    except ValueError as e:
        raise SystemExit(
            f"Training state name not found in robot state {robot_state_names}: {e}"
        )
    print(f"Policy state dims: {policy_state_names} -> robot indices {policy_state_idx}")

    def to_policy_obs(obs_frame: dict) -> dict:
        """Slice the full robot obs frame down to what the policy trained on:
        keep only the cameras present in the training dataset, and the state
        dims it was trained with (by name)."""
        policy_frame = {k: v for k, v in obs_frame.items()
                        if not k.startswith("observation.images.") or k in policy_camera_keys}
        policy_frame["observation.state"] = obs_frame["observation.state"][policy_state_idx]
        return policy_frame

    # === Eval recording dataset (full obs) ===
    dataset = LeRobotDataset.create(
        repo_id=args.hf_dataset_id,
        fps=args.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # === Policy processors (stats from the TRAINING dataset) ===
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.hf_model_id,
        dataset_stats=train_meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # === Inference engine ===
    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot_wrapper,
        hw_features=hw_features,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task=args.task_description,
        fps=float(args.fps),
        device=device,
    )
    engine.start()

    init_rerun(session_name="alohamini_evaluate_gripperonly")
    log_say("Starting evaluation")

    control_interval = 1.0 / args.fps
    recorded = 0

    reset_pose = None
    if args.reset:
        reset_pose = compute_reset_pose(args.reset_dataset_root, ordered_action_keys,
                                        args.reset_max_episode)
        print("Reset pose (median training first-frame):")
        for k, v in reset_pose.items():
            print(f"  {k:35s} {v:8.2f}")

    # Slew limiter state: only position-type keys are limited; velocities
    # (base) pass through untouched (and are frozen by the component mask anyway).
    limited_keys = [k for k in ordered_action_keys if k.endswith(".pos")]
    state_names = dataset_features["observation.state"]["names"]

    while recorded < args.num_episodes:
        if reset_pose is not None:
            log_say("Resetting arm to start pose")
            drive_to_pose(robot, robot_action_processor, reset_pose, args.fps, args.reset_seconds)
        log_say(f"Eval episode {recorded + 1} of {args.num_episodes}")
        engine.reset()
        # Seed the limiter from the arm's actual observed pose so the very
        # first policy command cannot snap away from wherever the arm is.
        obs_now = robot_observation_processor(robot.get_observation())
        state_now = build_dataset_frame(dataset_features, obs_now, prefix=OBS_STR)["observation.state"]
        last_sent = {k: float(state_now[state_names.index(k)]) for k in limited_keys}
        start = time.perf_counter()

        while (time.perf_counter() - start) < args.episode_time:
            loop_start = time.perf_counter()

            obs_raw = robot.get_observation()
            obs_processed = robot_observation_processor(obs_raw)
            obs_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            action_tensor = engine.get_action(to_policy_obs(obs_frame))
            if action_tensor is not None:
                action_dict = {k: action_tensor[i].item() for i, k in enumerate(ordered_action_keys)}
                if args.max_delta_per_tick > 0:
                    d = args.max_delta_per_tick
                    for k in limited_keys:
                        action_dict[k] = min(max(action_dict[k], last_sent[k] - d), last_sent[k] + d)
                        last_sent[k] = action_dict[k]
                robot.send_action(robot_action_processor((action_dict, obs_raw)))
                action_frame = build_dataset_frame(dataset_features, action_dict, prefix=ACTION)
                dataset.add_frame({**obs_frame, **action_frame, "task": args.task_description})

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)

        dataset.save_episode()
        recorded += 1

    log_say("Evaluation complete")
    engine.stop()
    robot.disconnect()
    dataset.finalize()
    if args.push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
