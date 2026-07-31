# SO-101 workflow — command reference

Working commands for the SO-101 single-arm setup (Pi host + workstation).
Context and why these flags exist: `docs/experiments/so101-kinesthetic-experiments-2026-07-30.md`.

Conventions: workstation paths below use `/mnt/nvme/lerobot/...`; the Pi is
`yosub@192.168.0.50`; all workstation commands run from the repo root with the
`lerobot_alohamini` conda env active.

## 1. Robot host (on the Pi)

```bash
# start (gains are NOT persisted in firmware defaults — always pass them)
cd ~/lerobot_alohamini
python examples/alohamini/so101_zmq_host.py \
    --p_coefficient 32 --i_coefficient 4 --d_coefficient 32
```

- Production gains (measured 2026-07-31): **P=32 I=4 D=32** — tracking residual
  −29% vs stock, command→observation round trip ≈ 100 ms (3 ticks @30 fps).
- The `gains: P=.. I=.. D=..` line in the log confirms what is live.
- The host publishes observations on a PUSH socket: **only one client at a
  time** gets the full stream — never leave a second client connected while
  another tool runs. Beware: a killed client's connection can linger on the
  Pi for a few seconds and still steal frames; check with
  `ss -tn state established '( sport = :5602 )'` on the Pi.

## 2. Kinesthetic data collection

```bash
# torque off, hand-guide the arm; dataset records on the workstation
python examples/alohamini/record_so101.py \
    --dataset local/<dataset_name> --num_episodes 100 \
    --remote_ip 192.168.0.50 --fps 30
```

The recorder **refuses to record on a slow observation stream**: a 3 s
pre-flight after connect plus a continuous in-episode check both require
≥ 93% of `--fps` true host messages (`msgs_received`, not loop iterations —
a conflated client at 15 Hz would otherwise silently duplicate frames). On
failure it prints a loud banner and aborts, discarding the in-progress
episode. Usual cause: a second client on the obs socket (see §1).

Protocol (see the experiments log, "Data collection protocol v2"): pilot 5
episodes first; start recording after gripping the arm and stop at release;
grid placements and alternate sides; keep motion ≥ ~1°/tick (final 5 cm in
1–2 s — sub-floor creeping trains wobble); ~20% recovery demos; always finish
the place.

## 3. Replay + tracking benchmark

```bash
# faithful replay of a recorded episode (MOVES THE ARM)
python examples/alohamini/replay_so101.py \
    --remote_ip 192.168.0.50 --dataset local/<dataset_name> --episode 0

# tracking benchmark: replay as commands, measure |obs - cmd|,
# decompose delay vs residual (Ctrl-C e-stops: freeze, twice = limp)
python examples/alohamini/replay_measure_so101.py \
    --dataset <dataset_name> --episode 0 \
    --remote_ip 192.168.0.50 --log_dir /mnt/nvme/lerobot/outputs/so101_replay_logs

# one full gain-setting measurement (6 episodes + held-pose hunting check)
python examples/alohamini/sweep_setting_so101.py \
    --log_dir /mnt/nvme/lerobot/outputs/so101_gain_sweep/<setting_name> \
    --remote_ip 192.168.0.50
```

## 4. Policy rollout (eval harness)

```bash
python examples/alohamini/evaluate_so101.py \
    --hf_model_id  <checkpoint>/pretrained_model \
    --train_dataset_root /mnt/nvme/lerobot/yosubshin/<matching_dataset> \
    --task_description "Put the red block into the bin" \
    --remote_ip 192.168.0.50 --num_episodes 5 \
    --fps 20 --interp_substeps 2 --n_action_steps 32 --max_delta_per_tick 7
```

**`--train_dataset_root` must be the dataset the checkpoint trained on**
(normalization stats + feature layout come from it).

### The speed ladder (train at 1×, pick speed at deploy)

Policy noise attaches to *actions*, playback rate sets *seconds per action* —
so speed is a deployment flag, and the k-lead must match the ~133 ms
command→obs round trip:

| goal | flags | model family |
|---|---|---|
| full speed | `--fps 30` | trim1x-**k4** |
| ⅔ speed | `--fps 20 --interp_substeps 2` | trim1x-**k3** |
| half speed | `--fps 15 --interp_substeps 2` | trim1x-**k2** |

### Flag glossary

| flag | what it does / when to change |
|---|---|
| `--fps` | playback rate; slows the robot without retraining (position-based policy is ~rate-invariant) |
| `--interp_substeps N` | stream N micro-commands per tick — removes the command staircase at low fps |
| `--n_action_steps` | actions executed per chunk; sets replan cadence (reaction latency ≈ this × tick − trigger) |
| `--chunk_trigger` | regenerate when this many actions remain (raise on a slower GPU) |
| `--max_delta_per_tick` | slew limit; match demo speed (1×: 7, 2×: 3.5, 3×: 2.5) |
| `--action_ema a` | low-pass executed actions (try 0.3 if motion wanders) |
| `--no_rtc` | disable prefix-inpainting (chunks become independent samples) |
| `--blend_splices` | cross-fade chunks instead (mode-averaging; usually worse) |
| `--no_reset` / `--home_seconds` | skip / slow the ramped move to the start pose |

Every episode logs per-tick cmd/obs/raw npz + per-camera mp4 to
`/mnt/nvme/lerobot/outputs/so101_rollout_logs/`. Episodes are ENTER-gated;
ENTER stops an episode; Ctrl-C freezes the arm in place (torque held).

## 5. Fisheye wrist camera

The Low Light fisheye is a multi-interface UVC camera: an H264 node
(`/dev/am_camera_fisheye`) and an MJPG/YUYV node — both max **1080p30**
(no 60 fps on any interface; verified). Integrated into the host via
`--wrist_h264`: one ffmpeg process tees the camera's stream into
`~/fisheye_archive/fisheye_<ts>_%04d.mkv` **5-minute segments** (full-res
SLAM source; MJPG default ≈ **19 GB/hour**) and pipes decoded 320x240 frames
into the normal live stream — client/recorder unchanged.

Hard-learned disk rules (a full SD card killed ffmpeg mid-session once —
frozen wrist frames in every episode after, plus write backpressure that
dropped the host loop to 26 Hz as it filled):

- The host **auto-restarts** a dead archive ffmpeg (3 s backoff) and drops to
  **decode-only** below 3 GB free (live feed survives; CRITICAL in the log).
- The recorder **aborts on frozen frames**: pre-flight and a per-tick
  watchdog (any camera byte-identical for 1 s) both refuse to record.
- For sessions longer than ~40 min, run the rolling mover on the
  workstation — it pulls completed segments off the Pi every minute:

```bash
./examples/alohamini/fisheye_archive_mover.sh   # user@pi and dest overridable
```

Start the host with:

```bash
python examples/alohamini/so101_zmq_host.py \
    --p_coefficient 32 --i_coefficient 4 --d_coefficient 32 \
    --forward_cam /dev/am_camera_forward --wrist_h264
```

Cameras are udev-named by model (`am_camera_forward` = HDR,
`am_camera_fisheye` = Low Light), robust to port changes. Standalone
inspection clip without the host:

```bash
# on the Pi: 30 s @ 1080p30 straight to file (hardware-encoded, ~9.4 Mbps)
ssh yosub@192.168.0.50 \
  'v4l2-ctl -d /dev/video4 --set-fmt-video=width=1920,height=1080,pixelformat=H264 \
     --set-parm=30 && v4l2-ctl -d /dev/video4 --stream-mmap --stream-count=900 \
     --stream-to=/tmp/fisheye_test.h264'

# pull + wrap into an mp4 for viewing
scp yosub@192.168.0.50:/tmp/fisheye_test.h264 /tmp/ && \
ffmpeg -y -framerate 30 -i /tmp/fisheye_test.h264 -c copy /tmp/fisheye_test.mp4
```

## 6. Training (on the training box)

```bash
lerobot-train \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
    --policy.drop_n_last_frames=31 \
    --dataset.repo_id=<repo_id> --dataset.root=<dataset_root> \
    --dataset.episodes="<train episode list>" \
    --dataset.image_transforms.enable=true \
    --batch_size=64 --steps=3000 --save_freq=500 --num_workers=6 \
    --output_dir=<out> --job_name=<name> \
    --wandb.enable=true --wandb.project=<project> --wandb.disable_artifact=true
```

Non-negotiables learned the hard way: `image_transforms.enable=true` (off by
default; ~12% better), `save_freq=500` (peak is ~2k steps / ~4 epochs — 5k
checkpoints miss it), `wandb.disable_artifact=true` (default silently copies
every checkpoint to disk), and a seeded **random** val split written to
`meta/val_episodes.json` (`{"val_episodes": [...]}`). Val loss selects
checkpoints *within* a run only — it cannot rank observation designs, action
spaces, or stretch factors. Rollouts are the arbiter.
