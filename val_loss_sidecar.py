#!/usr/bin/env python3
"""Validation-loss sidecar: polls a training run dir for new checkpoints,
computes mean val loss on each, logs to JSONL + (optionally) wandb.

Designed to run concurrently with `lerobot-train` on the same GPU.
Each val pass takes ~30 sec at batch=512, while training proceeds at
~4.5 step/s, so the GPU contention amortizes to ~1% net training
slowdown over a 60K-step run (~30 sec stall every ~18 min between
checkpoints).

Started by `scripts/train.sh` after `lerobot-train` is launched. Exits
when the training process is gone OR when the final checkpoint is
processed (whichever first).

Outputs:
    <run_dir>/val_loss.jsonl           # one line per checkpoint
    <run_dir>/val_loss_sidecar.log     # this script's stdout/stderr
    wandb run (resumed)                # train + val curves overlay

Usage:
    python3 scripts/val_loss_sidecar.py <run_dir> [--batch-size 512] [--no-wandb]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Defer heavy imports until we know we have work to do
DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_WORKERS = 8
DEFAULT_POLL_INTERVAL_S = 10.0
CHECKPOINT_STABLE_GRACE_S = 5.0   # wait this long after model.safetensors appears


def list_completed_checkpoints(checkpoints_dir: Path) -> list[int]:
    """Return integer step counts for fully-written checkpoints.

    A checkpoint is considered written when:
      <step>/pretrained_model/model.safetensors exists
      AND it hasn't been modified in the last CHECKPOINT_STABLE_GRACE_S sec
        (avoids racing on a partially-written file)
    """
    if not checkpoints_dir.is_dir():
        return []
    out = []
    now = time.time()
    for d in sorted(checkpoints_dir.iterdir()):
        if not d.is_dir() or not d.name.isdigit():
            continue
        model_file = d / "pretrained_model" / "model.safetensors"
        if not model_file.exists():
            continue
        mtime = model_file.stat().st_mtime
        if now - mtime < CHECKPOINT_STABLE_GRACE_S:
            continue
        out.append(int(d.name))
    return sorted(out)


def load_already_processed(jsonl_path: Path) -> set[int]:
    """Read existing val_loss.jsonl, return set of step values already done.

    Lets the sidecar resume cleanly if it crashes or the training run
    was paused/restarted.
    """
    if not jsonl_path.exists():
        return set()
    seen = set()
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            seen.add(int(json.loads(line)["step"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return seen


def append_jsonl(jsonl_path: Path, record: dict) -> None:
    with jsonl_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def init_wandb_resume(train_config: dict, explicit_run_id: str = "") -> Optional[object]:
    """Resume the training run's wandb session for overlaid logging.

    Returns an active wandb run handle on success, None on failure.
    Failure is non-fatal — JSONL logging continues regardless.

    The training process logs to wandb in real time. We attach to the
    SAME run via `wandb.init(id=<run_id>, resume="allow")` so the
    sidecar's `wandb.log({"val_loss": x})` calls land on the same run
    as training's `loss` calls. This lets you overlay train + val
    curves in the wandb UI.

    Resolution order for run_id:
      1. explicit_run_id (CLI flag / env var) — preferred path because
         lerobot-train does NOT write its run_id back to train_config.json.
      2. train_config["wandb"]["run_id"] — works only if a future lerobot
         release writes that field, which today's release does not.
    """
    wandb_cfg = train_config.get("wandb", {}) or {}
    if not wandb_cfg.get("enable"):
        return None
    run_id = explicit_run_id or wandb_cfg.get("run_id")
    if not run_id:
        return None
    try:
        import wandb
    except ImportError:
        return None
    project = wandb_cfg.get("project", "lerobot")
    entity = wandb_cfg.get("entity")  # may be None — wandb defaults to user's default entity
    try:
        run = wandb.init(
            id=run_id,
            project=project,
            entity=entity,
            resume="allow",
            # Don't sync code etc. — we're just appending metrics
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        )
        # Define val_loss against an independent step axis. Without this,
        # wandb rejects out-of-order logs because the parallel training
        # run advances its global step past the sidecar's checkpoint step
        # (e.g. training at step 48k while sidecar logs val for step 35k
        # → "Tried to log to step 35000 that is less than the current
        # step 48201" warning, log discarded).
        run.define_metric("val_step")
        run.define_metric("val_loss", step_metric="val_step")
        run.define_metric("val_loss_elapsed_s", step_metric="val_step")
        return run
    except Exception as e:
        print(f"[sidecar] wandb attach failed: {type(e).__name__}: {e}", flush=True)
        return None


def find_train_pid(run_dir: Path) -> Optional[int]:
    """Return PID of the lerobot-train process for this run, if alive.

    Heuristic: scan /proc for a python process whose cmdline contains
    the run_dir path. Used by the sidecar to detect when training has
    exited (so the sidecar can also exit).
    """
    needle = str(run_dir)
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
        except (OSError, FileNotFoundError):
            continue
        if "lerobot-train" in cmdline and needle in cmdline:
            return int(pid_dir.name)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path,
                        help="Training output dir (contains checkpoints/, train_config.json)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging (JSONL only)")
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID", ""),
                        help="wandb run id to attach to (overrides train_config.json's "
                             "wandb.run_id, which lerobot-train doesn't write). Read "
                             "from $WANDB_RUN_ID env var if not given. Without this, "
                             "sidecar logs only to JSONL.")
    parser.add_argument("--max-idle-s", type=float, default=300.0,
                        help="Exit if no new checkpoint for this many seconds AND train PID is gone")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoints_dir = run_dir / "checkpoints"
    jsonl_path = run_dir / "val_loss.jsonl"

    # Wait for the first checkpoint or train_config to appear. The sidecar
    # is spawned right after lerobot-train starts, but the first save_freq
    # boundary can be many minutes out (e.g. step 5000 at 4 step/s = ~21 min).
    # Bound the wait to the first checkpoint's expected arrival, with margin.
    BOOTSTRAP_WAIT_S = 60 * 60  # 1 h — covers slow first-checkpoint cases
    deadline = time.time() + BOOTSTRAP_WAIT_S
    train_config_path = None
    while time.time() < deadline:
        cand = checkpoints_dir / "last" / "pretrained_model" / "train_config.json"
        if cand.exists():
            train_config_path = cand
            break
        # Or check any other checkpoint dir
        for d in (checkpoints_dir.iterdir() if checkpoints_dir.exists() else []):
            tc = d / "pretrained_model" / "train_config.json"
            if tc.exists():
                train_config_path = tc
                break
        if train_config_path:
            break
        time.sleep(args.poll_interval)

    if not train_config_path:
        print(f"[sidecar] no train_config.json found after {BOOTSTRAP_WAIT_S}s — exiting", flush=True)
        return 1

    train_config = json.loads(train_config_path.read_text())
    repo_id = train_config["dataset"]["repo_id"]
    dataset_root = Path(train_config["dataset"]["root"])

    # Load val episodes; bail if no split
    from compute_val_loss import (
        load_val_episodes,
        build_val_loader,
        compute_checkpoint_val_loss,
        _decode_policy_config,
        _detect_policy_type,
        _read_dataset_fps,
    )
    val_episodes = load_val_episodes(dataset_root)
    if not val_episodes:
        print(f"[sidecar] no val_episodes.json in {dataset_root}/meta/ — exiting", flush=True)
        return 0

    # The val DataLoader's delta_timestamps must reproduce training exactly:
    # ACT stacks only the action chunk; Diffusion also stacks the observation
    # history (n_obs_steps). We read the index lists off the decoded policy
    # config so this works for both architectures.
    policy_config = _decode_policy_config(train_config_path.parent)
    fps = _read_dataset_fps(dataset_root)

    print(f"[sidecar] dataset:    {dataset_root}", flush=True)
    print(f"[sidecar] val eps:    {val_episodes[0]}..{val_episodes[-1]} ({len(val_episodes)})", flush=True)
    print(f"[sidecar] policy:     {_detect_policy_type(train_config_path.parent)}  fps: {fps}", flush=True)
    print(f"[sidecar] action_delta: {len(policy_config.action_delta_indices)} steps  "
          f"obs_delta: {getattr(policy_config, 'observation_delta_indices', None)}", flush=True)
    print(f"[sidecar] run_dir:    {run_dir}", flush=True)
    print(f"[sidecar] jsonl:      {jsonl_path}", flush=True)

    val_loader = build_val_loader(
        dataset_root=dataset_root,
        repo_id=repo_id,
        val_episodes=val_episodes,
        policy_config=policy_config,
        fps=fps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    wandb_run = None if args.no_wandb else init_wandb_resume(train_config, args.wandb_run_id)
    if wandb_run is None and not args.no_wandb:
        print(f"[sidecar] wandb attach skipped (no run_id provided — "
              f"pass --wandb-run-id or set $WANDB_RUN_ID). JSONL only.", flush=True)

    processed = load_already_processed(jsonl_path)
    print(f"[sidecar] already processed: {sorted(processed) if processed else '(none)'}", flush=True)

    last_progress_t = time.time()
    train_pid = find_train_pid(run_dir)
    if train_pid:
        print(f"[sidecar] tracking train PID {train_pid}", flush=True)

    while True:
        steps = list_completed_checkpoints(checkpoints_dir)
        new_steps = [s for s in steps if s not in processed]

        for step in new_steps:
            ckpt = checkpoints_dir / f"{step:06d}" / "pretrained_model"
            print(f"[sidecar] computing val_loss for step {step}...", flush=True)
            t0 = time.time()
            try:
                avg = compute_checkpoint_val_loss(
                    checkpoint_dir=ckpt,
                    val_loader=val_loader,
                    device=args.device,
                )
            except Exception as e:
                print(f"[sidecar]   FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            elapsed = time.time() - t0
            record = {"step": step, "val_loss": avg, "elapsed_s": round(elapsed, 2)}
            append_jsonl(jsonl_path, record)
            processed.add(step)
            print(f"[sidecar]   step {step}: val_loss = {avg:.6f}  ({elapsed:.1f}s)", flush=True)
            if wandb_run is not None:
                try:
                    # Use the custom val_step axis (defined in init_wandb_resume)
                    # so val points stay correctly ordered even when training's
                    # global step has already moved past `step`.
                    wandb_run.log({"val_loss": avg, "val_step": step,
                                   "val_loss_elapsed_s": elapsed})
                    # Also write into run.summary so wandb's workspace UI
                    # discovers val_loss as a chartable metric. Without this,
                    # metrics with a custom step_metric only appear in
                    # run.history (queryable) but NOT in the metric picker —
                    # users can't easily add a panel for them.
                    wandb_run.summary["val_loss"] = avg
                    wandb_run.summary["val_step"] = step
                except Exception as e:
                    print(f"[sidecar]   wandb log failed: {e}", flush=True)
                    wandb_run = None  # disable for the rest of the run
            last_progress_t = time.time()

        # Exit conditions
        if train_pid and not Path(f"/proc/{train_pid}").exists():
            # Training process gone — wait one more poll, then exit if
            # no new checkpoints (allow the final checkpoint to appear)
            time.sleep(args.poll_interval)
            final_steps = list_completed_checkpoints(checkpoints_dir)
            final_new = [s for s in final_steps if s not in processed]
            if not final_new:
                print(f"[sidecar] train PID {train_pid} is gone, no more checkpoints — done", flush=True)
                return 0
            # else continue, will pick up the final checkpoint next loop

        if (time.time() - last_progress_t) > args.max_idle_s:
            train_alive = train_pid and Path(f"/proc/{train_pid}").exists()
            if not train_alive:
                print(f"[sidecar] idle {args.max_idle_s}s + train not running — done", flush=True)
                return 0

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
