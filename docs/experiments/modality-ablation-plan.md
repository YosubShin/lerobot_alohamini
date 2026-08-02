# Data-modality ablation plan — kinesthetic vs teleop vs UMI

*Living plan doc. Started 2026-07-31; update as arms complete.*
Task: "Put the red block into the bin" (SO-101 follower, fisheye wrist cam).
Prior context: `so101-kinesthetic-experiments-2026-07-30.md`.

## Question

Which collection modality trains the best policy per hour of human effort,
and what failure quirks does each carry?

## Arms

| arm | episodes | action source | expected quirks to watch |
|---|---|---|---|
| kinesthetic | 100 | follower state (action ≡ state) | k-lag (needs k-shift), hands in wrist view, compliance-squeezed grip labels, dwell |
| teleop | 100 | leader pose (natural plant lead) | operator jerk/skill drift, slower collection; **cleanest test of the k-lag hypothesis — should need no k-shift** |
| UMI (handheld) | 200 (low-yield buffer) | SLAM EE pose + gripper width | embodiment gap (no arm in view), SLAM drift/scale, gripper mapping, yield rate (log it — it's a result) |

## Controls (what makes the comparison valid)

- **Same placement grid** for all modalities: alternating sides, right-heavy
  to counter the historical left bias, ~20% recovery demos, no dwell,
  always finish the place, gripper transitions ≥ 0.5 s full travel.
- **Primary comparison is wrist-only** (fisheye): UMI vs kinesthetic both
  trained wrist-only. Secondary arm: kinesthetic ± base cam (same episodes,
  two trainings) to price the base camera separately (known to matter from
  the 2026-07-29 ablation). Decided 2026-07-31.
- **Action-space caveat:** UMI differs in both modality and action space
  (EE vs joints). Decide before training: external IK at deploy, or
  gripper-state → joint-action (am1_red_bin recipe). Either way the UMI
  arm's conclusion is "modality + action-space package," not pure modality.
- **Interleave collection** in blocks of ~25 episodes, rotating modalities,
  so lighting/scene/operator drift doesn't correlate with modality.
- **Scaling curves, not endpoints:** train each modality at n=40 / 70 / 100
  (reuse the data-scaling workflow). Slope beats a single-point comparison.
- Same training recipe throughout: DP, image transforms on, save_freq 500,
  seeded random val split per dataset, rollouts as the arbiter (val loss
  cannot rank across action spaces or modalities).

## Eval protocol (shared, fixed before training)

- ~20 standard block placements, left/right balanced, including 3–4
  recovery-style starts.
- Deploy config per the speed ladder (fps30 primary; note k-shift variant
  used — kinesthetic needs k3/k4, teleop expected k0, UMI TBD by measured
  round trip).
- Metrics: success rate, time-to-success, intervention/assist count,
  left-vs-right success split.

## Session logistics

- Recorder guards are live (obs ≥ 93% of fps, frozen-camera watchdog).
- Run `fisheye_archive_mover.sh` for any session > 40 min (SD holds < 1 h
  of MJPG archive; a full disk killed a session on 2026-07-31).
- Watch the `obs=` readout; a Pi reboot fixes brcmfmac WiFi collapse.
- Rough collection budget: kinesthetic ~2 h, teleop ~2.5–3 h,
  UMI ~1.5–2 h + SLAM post-processing. Plan ≥ 3 sessions; interleave.

## Findings — modality verdict (2026-08-01)

**Headline: what matters in demonstration data is whether the demonstrator
experienced the plant the policy will control.**

Evidence chain (all rollouts on the real robot, fps15/interp2):

1. Both single-modality wrist-only models (76 eps each) reached, transported,
   and placed, but failed the GRASP (~1-2 successes per 100+ tries).
2. Trace forensics: gripper commands mode-averaged into a 60-70 hover — the
   mean of the open (100) and close (kinesthetic ~52 / teleop ~30) modes —
   which is also proprioceptively identical to "holding the block", driving
   descend-lift oscillation (reversals uniform across the replan grid, i.e.
   NOT chunk-boundary averaging).
3. `deepgrip` (kinesthetic close deepened in-data to teleop semantics):
   **no improvement** → gripper label semantics were not the constraint.
4. `teleop2x` (teleop episodes duplicated in the mix): **~20× grasp-rate
   jump** (several grasps in <10 tries).
5. Commitment crutches (`--gripper_binarize`, n_action_steps 20) helped only
   the conflicted models and HURT teleop2x — its careful descend-and-feel
   with micro-adjustments is the learned skill, not hesitation.

Interpretation: hand-guided (kinesthetic) demos bypass the arm's backlash —
the demonstrator's fingers never feel the slop, so the data contains no
corrective micro-behavior. Teleop demos are produced THROUGH the real plant
(leader quantization, follower backlash, visual feedback), so the operator's
compensations are recorded, and the policy needs exactly those at deploy.
Corollaries:
- No per-frame transform (k-shift, close-deepening, learned lag simulation)
  can synthesize undemononstrated corrective behavior — those fixes are dead ends.
- Kinesthetic data's remaining value: cheap non-contact coverage in a mix
  (pending the teleop-solo control), and the UMI comparison baseline.
- Hardware-quality argument (backlash vs Franka-class arms): real but
  bounded — humans teleop the task at ~100% through the same hardware; the
  slop narrows the tolerance funnel and raises the data requirement rather
  than blocking the approach.

UMI pre-registration (made before the rig exists): UMI shares kinesthetic's
no-plant-in-loop flaw for the ARM (expect casual positioning) but not the
GRIPPER (real fingers, real contact). Predicted mitigation: mix with teleop
or deliberately careful final approaches in UMI demos.

Next collection (~80-100 teleop eps, after teleop-solo + graspx2 evals lock
the recipe): left-heavy (fix 27L/49R), grasp-dense, ~20-30% explicit
recovery demos (miss → re-open → re-descend → succeed).

Retired without rollout (superseded by the above): mix-k3, kinesthetic
k0/k1 wrist-only (staged but unranked), kinesthetic-2cam+bias.

## Status

- [x] Kinesthetic pilot QA'd (frozen-wrist eps excluded — pilot dataset)
- [x] Kinesthetic 77 collected → 76 after QA (`so101_fisheye_red_pick`);
      trained k3/k2 two-cam (local) + k3 wrist-only (ripper), all peak @1500
- [x] Teleop pilot QA'd: natural 3–4-tick plant lead confirmed in-data
- [x] Teleop 78 collected → 76 after crash repair (`so101_teleop_red_pick`,
      27L/49R right-skew — watch for side bias); trained 2-cam no-k on
      ripper, peak @1500 (0.0223). Winners staged under
      /mnt/nvme/lerobot/outputs/: dp_fisheye_trim1x_{k3,k2} (run dirs),
      dp_fisheye_k3_wristonly_1500, dp_teleop_2cam_1500.
      Best deploy flags so far: fps15 interp2 n_action_steps15.
- [ ] UMI rig ready (SLAM pipeline validated on archive segments), pilot 10
- [ ] UMI 200
- [ ] First training pair (decided 2026-07-31): local RTX 6000 = best-known
      recipe (two-cam, trim + k-shift, transforms on, save_freq 500);
      ripper = **wrist-only fisheye ablation** (config-only; doubles as the
      UMI-comparison baseline — the old wrist-only result predates the
      fisheye). Queue weight-EMA after it; spline/basis parked (breaks RTC
      prefix-inpainting; smoothness currently mitigated by fps ladder).
- [ ] Trainings: 3 modalities × n∈{40,70,100} wrist-only (+2 two-cam arms)
- [ ] Shared eval grid runs
- [ ] Write-up in experiments doc
