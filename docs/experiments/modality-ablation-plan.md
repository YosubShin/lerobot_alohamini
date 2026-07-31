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
- **Primary comparison is wrist-only** (fisheye) for all three arms — UMI
  cannot have a base cam, so the others must match. Secondary arm:
  kinesthetic + teleop retrained with wrist+base to price the base camera
  separately (known to matter from the 2026-07-29 ablation).
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

## Status

- [ ] Kinesthetic pilot QA'd (eps 0–1 good; eps 2–4 frozen-wrist, excluded)
- [ ] Kinesthetic 100
- [ ] Teleop rig re-checked (leader arm), teleop pilot 5
- [ ] Teleop 100
- [ ] UMI rig ready (SLAM pipeline validated on archive segments), pilot 10
- [ ] UMI 200
- [ ] Trainings: 3 modalities × n∈{40,70,100} wrist-only (+2 two-cam arms)
- [ ] Shared eval grid runs
- [ ] Write-up in experiments doc
