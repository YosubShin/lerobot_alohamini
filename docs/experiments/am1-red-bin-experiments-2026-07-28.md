# AM1 Red-Bin / UMI-Style Experiments — 2026-07-28

Task: "Put the red block into the bin with the right arm" — AlohaMini (dual SO-101),
right arm only (left arm / base / lift disabled during collection).
Source data: `am1_red_bin` — 34 episodes, 34,764 frames @ 30 fps, 3 cameras,
16-dim joint state/action. NAS: `/mnt/nas/lerobot/yosubshin/am1_red_bin`,
local: `/mnt/nvme/lerobot/yosubshin/am1_red_bin`.

Motivating question: can we train UMI-style — treat the data as coming from a
gripper-only device, mask out arm kinematics from the observations, and force
the policy to learn IK implicitly from vision?

## Protocol (shared by all diffusion-policy runs)

- LeRobot 0.5.2 fork, `--policy.type=diffusion`, pretrained-nothing (from scratch)
- Fork defaults: horizon 64, n_action_steps 32, n_obs_steps 2, separate
  ResNet18 per camera; `--policy.drop_n_last_frames=31` (fork default 7 is
  stale for horizon 64)
- batch 64, 20k steps (early runs 60k — wasteful, every variant peaks at 5k),
  checkpoints every 5k
- Split: episodes 0–29 train / **30–33 held-out val** (`meta/val_episodes.json`;
  chronological, not random). Note: the July pi05 runs trained on all 34, so
  their loss is not comparable.
- Val loss: `aic/scripts/val_loss_sidecar.py` + `compute_val_loss.py` running
  per-checkpoint on the idle GPU. **Patched** `compute_val_loss.py` to apply
  the checkpoint's saved preprocessor before `policy.forward()` — lerobot ≥0.5
  normalizes in a separate pipeline; without the patch val loss reads ~3.45
  garbage instead of ~0.02.
- wandb project `alohamini_dp`.

## Datasets derived (all under /mnt/nvme/lerobot/yosubshin/)

| dataset | state | action | cameras |
|---|---|---|---|
| am1_red_bin_gripperonly | 2d [L/R gripper] | 16d joints | 3 |
| am1_red_bin_gripperonly_2cam | 2d | 16d | forward + wrist_right |
| am1_red_bin_fullstate_2cam | 16d joints | 16d | forward + wrist_right |
| am1_red_bin_eepose_2cam | 10d [EE pos, rot6d, R gripper] | 16d | forward + wrist_right |
| am1_red_bin_eeaction_2cam | 10d EE | 10d EE | forward + wrist_right |
| am1_red_bin_fullstate_wristonly | 16d | 16d | wrist_right only |

EE pose computed by SO-101 FK (chain from `aic/scripts/so101_fk_explorer.py`)
over recorded joints; normalized-units → radians via the follower calibration
(`AlohaMiniRobot.json` from the Pi: per-joint tick ranges, homing centers the
mid pose at tick 2048, radians = (ticks−2048)·2π/4096). Derivation scripts in
the session scratchpad (`derive_*.py`); EE trajectories validated smooth
(p99 9.6 mm/tick) with a sane reach envelope.

## Results — val loss by checkpoint (held-out episodes 30–33)

| variant | 5k | 10k | 15k | 20k | deployable? |
|---|---|---|---|---|---|
| **fullstate wristonly** | **0.0124** | 0.0171 | 0.0302 | 0.0501 | yes |
| fullstate 2cam | 0.0165 | 0.0178 | 0.0238 | 0.0233 (25k: 0.0267, stopped) | yes |
| gripperonly 2cam | 0.0209 | 0.0361 | 0.117* | 0.0443 (→0.157 @60k) | yes |
| eepose 2cam | 0.0239 | 0.0373 | 0.0738 | 0.1024 | yes |
| eeaction 2cam† | 0.0218 | 0.0349 | 0.0655 | 0.1005 | needs 5-DOF IK |

\* noisy outlier. † own scale (10-dim EE MSE) — not comparable to joint-space rows.

**Every variant peaks at 5k steps and overfits after** — the signature of 30
training episodes, not of any particular observation design.

### Robot rollouts (5k checkpoints, DDIM-10, start-pose reset, slew limiter)

- **gripperonly**: erratic. Root causes were compounded: DDPM-100 inference
  (1109 ms/chunk vs the 1067 ms a 32-step chunk covers → freeze-lurch at every
  chunk boundary), no start-pose reset, chunk-to-chunk disagreement (~10–14
  units absolute prediction error on val frames — the policy can't localize
  the arm from pixels reliably at this data scale), plus severe camera/host
  hardware failures (below).
- **fullstate 2cam**: reaches the block, does not grasp. Clearly better.
  "Close but no grasp" = classic 30-episode BC signature.
- **fullstate wristonly**: reaches forward and hovers over the block with jerky
  movement in that area — worse than fullstate 2cam on the robot despite the
  best val loss of the day. No wild flailing (unlike gripperonly).

### Conclusions

1. **DP learns real signal from 30 episodes — but not implicit IK.** Joint-state
   obs (fullstate) beats gripper-only and, surprisingly, beats exact-FK EE-pose
   obs (eepose 0.0239 — worse than even the blind gripper-only 0.0209). At this
   data scale, every UMI-style abstraction costs performance.
2. **The base (forward) camera looks harmful offline but is load-bearing
   on-robot.** Wrist-only won val by ~25% (0.0124 vs 0.0165) yet rolled out
   worse: jerky hovering over the block vs fullstate-2cam's near-grasp. Val
   frames are demo states where the wrist view already contains the answer;
   on-policy, the forward cam anchors recovery from self-induced drift —
   exactly what offline action-MSE never measures.
3. **Corollary: val loss ranks checkpoints within a run, but cannot rank
   observation designs across runs.** Robot rollouts remain the arbiter.
4. **More demos is the highest-leverage next step** (~100 episodes, varied
   block/bin placements, deliberate grasp phases).

## Deployment harness

`examples/alohamini/evaluate_bi_gripperonly.py` (new; stock `evaluate_bi.py`
feeds the full 16-dim state + 3 cams and would break reduced-obs policies):

- Slices `observation.state` by name to the training dataset's dims; keeps only
  the training dataset's cameras (both read from `--train_dataset_root`).
- Normalization stats from the training dataset (not the empty eval-recording
  dataset).
- **DDIM-10 sampler by default** (100 ms/chunk vs DDPM-100's ~1 s on the 3090).
- `--n_action_steps 16` default (replan 2×/s), slew limiter
  `--max_delta_per_tick 4.0` (≈ demo p99) seeded from the observed pose —
  kills chunk-boundary snaps; safe for state-blind policies, near-neutral for
  state-aware ones (`0` to disable).
- Start-pose reset to the median training first-frame pose (from the original
  16-dim dataset), right-arm-only component mask (left arm/lift latched, base
  zeroed by the LeKiwi client).
- Golden-frame audit: inference-path normalized tensors are bit-identical to
  the training dataloader's; 45k ckpt reproduces training chunks to ~0.2 units.

## SO-101 IK study (toward an honest external-IK / UMI-retargeting experiment)

`scratchpad/so101_ik.py` — damped least squares on a numeric 6×5 Jacobian.

- Clean FK→IK round trip is **lossless** (median 2e-4 rad) — but only with
  ground-truth init. From a rest-pose init, the three near-parallel pitch
  joints settle on 2π-wrapped FK-identical branches: EE residual 0.0 mm,
  joints physically impossible. Non-uniqueness is the *first* failure mode.
- Hardened solver: limit clamping (projected DLS) + tiny rest tie-breaker
  (1e-4 — anything larger biases; a 5-DOF arm has no nullspace) + multi-start
  frame 0 + warm-start tracking. Control: 3 mm median, 0 violations.
- Registration sensitivity (random rigid offset of the EE trajectory):
  ≤2 cm absorbed (3.5 mm med), 5 cm degrades (14 mm), 10 cm broken (78 mm).
  Demos sit ~2–3 cm from the workspace boundary.
- Implication: literally round-tripping our own data through IK proves nothing
  (poses are on-manifold by construction). The honest experiment needs: random
  per-episode base registration → **feasibility-optimizing registration**
  (outer loop over the rigid transform — not yet built) → hardened IK →
  train → compare against the ground-truth-joints policy. Real-UMI degradation
  additionally includes SLAM noise and 6-DoF→5-DOF manifold projection.
- Context: original UMI trains/acts in relative-EE space (runtime IK in the
  controller); many labs instead IK-retarget UMI data to joints at training
  time and run joint-native stacks. With wrist-only obs the absolute placement
  is unobservable to the policy, so retargeting placement is a free parameter
  chosen for feasibility, not truth.

## Hardware findings (day-long saga)

- The Pi host (`alohamini-host.service`, `yosub@192.168.0.50`) **homes the
  lift (down-stall-up to 200 mm) on every service start** — a crash-looping
  host presents as "the lift moves by itself."
- Both wrist cameras (identical Arducams behind USB hub 1-1, udev-named by
  port) failed repeatedly: wrist_left USB-disconnect; wrist_right empty-frame
  imdecode bursts (uvcvideo EPROTO −71) that kill the camera thread → host
  crash → systemd restart → lift homing. wrist_left is now **physically
  unplugged and commented out** of `config_lekiwi.py` on BOTH the Pi and the
  workstation (re-enable both for bimanual work). wrist_right was moved to a
  direct Pi port (udev rule updated: `KERNELS=="1-1"`), then failed again →
  swapped in the old wrist_left camera body. Suspect list: camera cable
  flex/wear, the hub, Pi USB power. Unresolved risk: every wrist_right dropout
  feeds the policy stale frames on its only camera.

## Infrastructure

- Ripper (Brian's machine, `ssh btuan@ripper`, RTX PRO 6000): self-contained
  env under `~/yosub/` (uv + venv, torch 2.11 cu130, lerobot editable, all
  datasets, wandb key in `~/yosub/wandb.env`). Smoke-tested; trained the
  eeaction run at 7.2 step/s (vs 1.4 on the power-capped 3090).
- Local: RTX 6000 = training workhorse; 3090 = sidecars + robot inference.
  Address GPUs by UUID (CUDA/nvidia-smi order is inverted on this box).

## Next steps

1. Robot-test `dp_am1_fullstate_wristonly` 5k — current champion.
2. Collect ~100 episodes (varied placements, slow deliberate grasps); re-run
   the variant ladder to see which abstractions become affordable.
3. If UMI-fidelity is pursued: build the feasibility-registration outer loop
   and run the honest IK-degradation experiment.
4. Blind head-to-head via `eval_compare.py` (needs the same obs-slicing shim
   as the harness) including the July pi05 checkpoints.
