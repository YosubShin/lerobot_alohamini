#!/usr/bin/env python3
"""Compute mean validation loss for a single ACT checkpoint.

Library-style: importable as `compute_checkpoint_val_loss()` for use in
the val-loss sidecar, or runnable as a CLI for one-off use.

The val set is the dataset's `meta/val_episodes.json` — written by
mcap_to_lerobot.py at conversion time when `--val-fraction > 0`. Loss
is the policy's training-time loss returned from `policy.forward()`
(L1 + KL for ACT).

Usage (CLI):
    python3 scripts/compute_val_loss.py <checkpoint_dir>

    where <checkpoint_dir> is the dir containing model.safetensors —
    e.g., outputs/train/<run>/checkpoints/030000/pretrained_model

    Reads dataset path from the checkpoint's train_config.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _detect_policy_type(checkpoint_dir: Path) -> str:
    """Return 'act' or 'diffusion' from the saved config.json's `type` field."""
    cfg = json.loads((checkpoint_dir / "config.json").read_text())
    return str(cfg.get("type", "act")).lower()


def _load_policy(checkpoint_dir: Path):
    """Lazy-load the policy class matching the checkpoint's config type."""
    ptype = _detect_policy_type(checkpoint_dir)
    if ptype == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        return DiffusionPolicy.from_pretrained(str(checkpoint_dir))
    # Default / fall-through: ACT
    from lerobot.policies.act.modeling_act import ACTPolicy
    return ACTPolicy.from_pretrained(str(checkpoint_dir))


def _decode_policy_config(checkpoint_dir: Path):
    """Decode the saved config.json into the matching policy-config dataclass.

    Used to read the `observation_delta_indices` / `action_delta_indices`
    properties, which tell the val DataLoader exactly how to stack the
    observation history + action chunk that the policy's forward() expects.
    (ACT: obs=None, action=[0..chunk_size-1]; Diffusion: obs=[-1,0],
    action=[-1,0,..,horizon-2].)
    """
    import draccus
    cfg_dict = json.loads((checkpoint_dir / "config.json").read_text())
    ptype = str(cfg_dict.pop("type", "act")).lower()  # draccus rejects unknown `type` field
    if ptype == "diffusion":
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        return draccus.decode(DiffusionConfig, cfg_dict)
    from lerobot.policies.act.configuration_act import ACTConfig
    return draccus.decode(ACTConfig, cfg_dict)


DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_WORKERS = 8


def load_val_episodes(dataset_root: Path) -> list[int]:
    """Read meta/val_episodes.json from the dataset.

    Returns empty list if the file doesn't exist (no val split was
    written at conversion time).
    """
    val_file = dataset_root / "meta" / "val_episodes.json"
    if not val_file.exists():
        return []
    payload = json.loads(val_file.read_text())
    return list(payload["val_episodes"])


def build_val_loader(
    dataset_root: Path,
    repo_id: str,
    val_episodes: list[int],
    policy_config,
    fps: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> DataLoader:
    """Construct a DataLoader over only the val-episode subset.

    `delta_timestamps` MUST reproduce what the policy was trained on so
    forward() receives correctly-shaped batches:
      - ACT: only the action chunk is stacked (obs is single-step).
      - Diffusion: BOTH the observation history (n_obs_steps) AND the
        action horizon are stacked. Diffusion's compute_loss asserts
        `n_obs_steps == config.n_obs_steps`, so the obs features must
        carry their delta_timestamps too.

    We read `observation_delta_indices` / `action_delta_indices` straight
    off the decoded policy config — these are the exact index lists
    lerobot-train uses internally. fps comes from the dataset meta.
    """
    delta_timestamps = {}
    action_idx = policy_config.action_delta_indices
    delta_timestamps["action"] = [i / fps for i in action_idx]
    # Diffusion stacks observation history; ACT leaves obs single-step (None).
    obs_idx = getattr(policy_config, "observation_delta_indices", None)
    if obs_idx is not None:
        for feat_key in policy_config.input_features:
            delta_timestamps[feat_key] = [i / fps for i in obs_idx]
    # video_backend="pyav" matches train.sh's --dataset.video_backend=pyav.
    # On Blackwell GPUs torchcodec fails to load (libtorchcodec /
    # libnppicc.so.12 incompatibility) so the default "torchcodec" backend
    # crashes the DataLoader workers with RuntimeError, leaving val_loss
    # uncomputed. PyAV + the prebuilt frame cache (when present) is the
    # working alternative on this hardware.
    ds = LeRobotDataset(
        repo_id=repo_id,
        root=str(dataset_root),
        episodes=val_episodes,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )


def compute_checkpoint_val_loss(
    checkpoint_dir: Path,
    val_loader: DataLoader,
    device: str = "cuda",
) -> float:
    """Load a checkpoint and compute mean val loss across the loader.

    Returns scalar (mean of per-batch losses, weighted by batch).
    Note: we keep the policy in train mode so forward() returns the
    chunked training loss. .eval() doesn't change ACT's forward path
    in current lerobot, but we don't rely on that — train-mode + no_grad
    is unambiguously the right setup. Supports both ACT and Diffusion
    policies (auto-detected from the checkpoint's config.json type field).
    """
    policy = _load_policy(checkpoint_dir)
    policy.to(device).train()  # train mode runs the VAE encoder (ACT); diffusion's forward is unaffected

    # lerobot >= 0.5 moved input normalization out of policy.forward() into a
    # separate preprocessor pipeline saved alongside the checkpoint. If one is
    # saved, we must apply it (as lerobot_train.py does) or forward() sees
    # unnormalized inputs and the val loss is garbage. Older checkpoints have
    # no policy_preprocessor.json and normalize inside forward() — skip then.
    preprocessor = None
    if (checkpoint_dir / "policy_preprocessor.json").exists():
        from lerobot.policies import make_pre_post_processors
        preprocessor, _ = make_pre_post_processors(policy.config, str(checkpoint_dir))

    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            if preprocessor is not None:
                batch = preprocessor(batch)
            out = policy.forward(batch)
            loss = out[0] if isinstance(out, tuple) else out["loss"]
            # Weight by batch size so partial last batch doesn't get
            # equal vote in the mean.
            n = next(iter(batch.values())).shape[0] if any(
                torch.is_tensor(v) for v in batch.values()
            ) else 1
            total_loss += float(loss) * n
            total_samples += n

    del policy
    torch.cuda.empty_cache()

    return total_loss / max(total_samples, 1)


def _read_train_config_dataset(checkpoint_pretrained_dir: Path) -> tuple[str, Path]:
    """Pull dataset repo_id + root from a checkpoint's train_config.json."""
    cfg = json.loads((checkpoint_pretrained_dir / "train_config.json").read_text())
    return cfg["dataset"]["repo_id"], Path(cfg["dataset"]["root"])


def _read_dataset_fps(dataset_root: Path) -> int:
    """fps from the dataset's meta/info.json."""
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    return int(info.get("fps", 20))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path,
                        help="Path to pretrained_model dir (contains model.safetensors)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if not (args.checkpoint_dir / "model.safetensors").exists():
        print(f"ERROR: no model.safetensors in {args.checkpoint_dir}", file=sys.stderr)
        return 2

    repo_id, dataset_root = _read_train_config_dataset(args.checkpoint_dir)
    val_episodes = load_val_episodes(dataset_root)
    if not val_episodes:
        print(f"ERROR: no meta/val_episodes.json in {dataset_root}", file=sys.stderr)
        print("       Re-run mcap_to_lerobot.py with --val-fraction > 0", file=sys.stderr)
        return 2

    policy_config = _decode_policy_config(args.checkpoint_dir)
    fps = _read_dataset_fps(dataset_root)

    print(f"dataset:        {dataset_root}", flush=True)
    print(f"val episodes:   {val_episodes[0]}..{val_episodes[-1]} ({len(val_episodes)})", flush=True)
    print(f"policy type:    {_detect_policy_type(args.checkpoint_dir)}", flush=True)
    print(f"action_delta:   {policy_config.action_delta_indices[0]}..{policy_config.action_delta_indices[-1]} "
          f"({len(policy_config.action_delta_indices)} steps)", flush=True)
    print(f"obs_delta:      {getattr(policy_config, 'observation_delta_indices', None)}", flush=True)
    print(f"fps:            {fps} (from dataset meta)", flush=True)

    val_loader = build_val_loader(
        dataset_root=dataset_root,
        repo_id=repo_id,
        val_episodes=val_episodes,
        policy_config=policy_config,
        fps=fps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    avg_loss = compute_checkpoint_val_loss(
        checkpoint_dir=args.checkpoint_dir,
        val_loader=val_loader,
        device=args.device,
    )
    print(f"val_loss = {avg_loss:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
