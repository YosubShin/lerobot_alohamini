# SO-101 Kinesthetic Red-Pick — Experiments & Learnings (2026-07-29 → 30)

Task: "Put the red block into the bin" — single SO-101 follower on a Pi
(`so101_zmq_host.py`), forward + wrist cameras, 6-dim joint state/action.
Data: `so101_kinesthetic_red_pick` — 86 episodes hand-guided (kinesthetic
teaching, torque off), 33,104 frames @ 30 fps. Cleaned to 85 episodes
(`so101_red_pick_clean`, ep40 was a 1.8 s outlier). Val split: seeded random
episodes [3, 14, 31, 35, 81].

Prior context: `am1-red-bin-experiments-2026-07-28.md` (same task, AlohaMini
teleop, 34 episodes). Record→replay fidelity was validated *before* dataset
collection — the arm faithfully tracks recorded kinesthetic trajectories, so
all failures below are policy/deployment-side, not plant-side.

## Status (2026-07-30 midday)

**First successful rollout**: trim2× checkpoint (2.5k steps, val 0.0164) at
`--fps 30 --n_action_steps 32`. trim1×/trim3× checkpoints in final training.

Winning checkpoints (all local, `/mnt/nvme/lerobot/outputs/`):

| run | dataset | best ckpt | val | notes |
|---|---|---|---|---|
| dp_redpick_joint_final | clean (1×) | 2k | 0.0224 | first deployable |
| dp_redpick_ee_final | EE 10-dim | 3k | 0.0206 | benched (needs 5-DOF IK) |
| dp_redpick_slow2x | 2× stretch | 2.5k | 0.0141 | |
| dp_redpick_slow3x | 3× stretch | 3.5k | 0.0117 | dithers at home (see below) |
| dp_redpick_trim2x | trim + 2× | 2.5k | 0.0164 | **first success** |
| dp_redpick_trim1x | trim, 1× | 2k | 0.0242 | untested |
| dp_redpick_trim3x | trim + 3× | (training) | | |

Val numbers are comparable ONLY within a row's own run (see Learnings).

## Learnings

### Data & task design

- **Kinesthetic ≫ teleop for collection.** Same frame budget as the teleop
  round (33k vs 35k frames) but 2.5× more episodes, smoother and faster demos.
  Episode diversity-per-frame mattered more than raw frame count.
- **Idle dwell is poison.** 32% of frames were near-still (median ~1 s lead-in
  and tail-out per episode). At the home pose the data therefore said "stay
  put" a third of the time → the policy dithered at episode start (lift a
  little, settle back, repeat), sampling the "stay" mode. Trimming
  leading/trailing dwell (motion threshold 0.5 units/tick, 5-frame margin)
  removed the mode at its source. Rhymes with the 2026-04 ACT lesson (43%
  zero-action frames → "do nothing" default): **near-still frames are a
  recurring failure class — audit every new dataset for them.**
- **Time-stretching is a free post-hoc speed knob.** Lerp joints, repeat
  images nearest-neighbor, keep native fps: per-tick deltas shrink by the
  factor (verified ½ / ⅓), so the *weights* encode slow motion and no
  deploy-time tricks are needed.
- **Stretch interacts with the horizon.** Keeping horizon=64 while stretching
  3× meant a whole prediction horizon fit inside the (also-stretched) dwell —
  complete "do nothing" chunks became sampleable, and the 3× model couldn't
  escape the home pose. Scale horizon with stretch, or trim the dwell first.
- **A policy cannot sample behavior that isn't in the data.** No
  recovery/back-off demos existed, so when the gripper brushed the block the
  only in-distribution continuation was to keep going (plow-through). Next
  collection: include failed-grasp recoveries, wider placement variation,
  minimal dwell (start recording after gripping the arm).
- **The left-side success bias is data imbalance, verified end-to-end.**
  Rollouts only succeeded with the block on the left. FK analysis of grasp
  positions (Brian's pose cache, `~btuan/co/trajedy/`): **47 left vs 27 right**
  (+y = left, physically confirmed by replaying extreme-y episodes). Three
  compounding factors: (1) ~2:1 lateral imbalance overall; (2) the early
  session was 3:1 left, and right-side placements are entangled with the
  later/slower session style (regime change ~ep 35–44); (3) 4 of 5 val
  episodes come from the left-heavy first half, so checkpoint selection
  optimized left performance. Training shuffles frames fully, so this is a
  *proportion* problem, not an ordering one — 2:1 exposure means the loss
  weights left states ~2×. Fixes by leverage: right-heavy collection round >
  side/session-stratified val split > per-episode sample reweighting.
- **`action` ≡ `observation.state` in kinesthetic data** (byte-identical —
  Brian's finding, verified): there is no leader, so both columns are the
  measured position. Chunked policies therefore learn to predict future
  *measured states* — workable for position control (chunk ≈ trajectory to
  track) but the first chunk action is the current pose (a no-op step) and
  the whole signal lags one frame; shifting actions by one frame in the next
  derivation recovers a true target.
- **The deeper consequence — kinesthetic actions carry no plant lead**
  (Brian, 2026-07-30, quantified via original-vs-replayed trajectory error:
  large). Teleop data implicitly encodes the inverse plant: the leader's
  command leads the follower's achieved pose by exactly the tracking error,
  so a policy trained on teleop learns to command *ahead* of the trajectory.
  Kinesthetic data has command = achievement by construction, so the
  deployed policy issues setpoints *at* the desired path and the servo
  systematically sags (gravity) and lags (bandwidth) behind it —
  degree-level error, invisible to the eye, fatal at grasp tolerance.
  Partly explains reach-but-no-grasp, and why 2×-slow execution helped
  (velocity-proportional lag halves; gravity sag doesn't). Mitigations,
  ordered: (1) **servo gain tuning** (Brian, in progress) — makes
  setpoint ≈ achievement at the plant level; needs integral/feedforward for
  gravity sag, costs contact stiffness, has a bandwidth ceiling; acceptance
  test = replay error, measured also while holding the block. (2)
  **k-frame action shift** (`action(t) = state(t+k)`) — zero-model lead
  compensation; calibrate per-joint k from replay (command, achieved)
  pairs, which are the correct plant-ID data (regenerable in minutes after
  any gain change — old teleop data is from a different rig/gains and is
  stale the moment gains move). (3) **Full inverse-plant command
  correction** (fit + invert per-joint delay/lag/gravity model on replay
  pairs, rewrite dataset actions) — most precise, held in reserve:
  inversion amplifies noise and is load/configuration-dependent.
- Cross-check of our end-trimming vs Brian's grasp/release event detection:
  fully consistent — trim never cut into the task (0 episodes trimmed past
  grasp or before release); we keep ~2.4 s median post-release retreat that
  his release-based episode end would drop. 6 episodes never reopen the
  gripper (no place phase) — exclude in the next revision.

### Training

- **The true val peak is ~2k steps (~4 epochs), not 5k.** Dense checkpointing
  (save_freq 500) revealed every earlier "5k champion" was ~1.7× past peak.
  From-scratch DP on ~30k frames converges in minutes — checkpoint densely.
- **Image augmentation is off by default in lerobot and worth turning on**:
  ~12% lower val minimum, flatter overfit slope, peak shifts later (1.5k→2-3k).
- **Val loss is valid only for within-run checkpoint selection.** It cannot
  rank: observation designs (wrist-only beat 2-cam offline by 25%, then lost
  on the robot), action spaces (joint vs EE units), dim counts (constant dims
  dilute MSE — the 16-dim AlohaMini losses needed a ×16/6 correction), stretch
  factors (slower targets are mechanically easier), or trimmed-vs-untrimmed
  (the val set changes too). **Rollouts are the arbiter for everything else.**
- **Use a seeded random val split**, not last-N — late episodes carry
  operator-practice drift.
- Fork bug fixed in `lerobot_train.py`: non-contiguous `--dataset.episodes`
  crashed (sampler got global frame offsets; the filtered dataset reindexes
  contiguously).

### Inference & deployment

The multi-day debugging arc, in causal order — each fix exposed the next
failure:

1. **DDPM-100 sampling is ~10× too slow for real time** (~1 s/chunk vs the
   1.07 s a 32-action chunk covers). DDIM-10 (~130 ms on the RTX 6000) is
   drop-in at inference; action quality change was not noticeable.
2. **Synchronous `select_action` freezes the arm at every chunk boundary**
   (stall = full generation latency). Fix: async chunk planner — feed the
   observation queue every tick (so the 2-frame history stays 33 ms apart,
   training-consistent), generate the next chunk in a worker thread while
   ~6 buffered actions remain, splice time-aligned (drop actions whose ticks
   elapsed during generation). Steady-state p99 tick cost: 3.9 ms. Zero stalls.
3. **Consecutive chunks disagree by 10–47 units** (independent diffusion
   samples from a multimodal distribution) → lurch at every splice, smeared
   by the slew limiter into stop-and-go. Cross-fade blending was implemented
   and *rejected on the robot* — averaging two modes produces motion belonging
   to neither. Hard splice + limiter felt better. The principled fix, if ever
   needed: prefix-inpainting (clamp the overlap to committed actions during
   denoising) — selects a mode instead of averaging. Unbuilt.
4. **Low-fps execution is a legitimate speed hack with sharp limits.**
   Position-based DP (no velocities/timestamps) is ~playback-rate invariant,
   so running the 30 fps-trained policy at fps 5–15 slows the robot "for
   free". But: (a) the command staircase makes servos dash between sparse
   targets — fixed with `--interp_substeps` micro-ramps; (b) **reaction
   latency scales with tick duration** — at fps 5 a fresh observation reaches
   the motors only every ~2 s, so contact events (gripper brushing the block)
   couldn't trigger correction in time (plow-through); (c) obs-history timing
   goes off-distribution.
5. **Inference cadence ≠ tick rate.** Observations are ingested at 30 Hz but
   the denoiser runs once per chunk cycle: every ~(n_action_steps − 6) ticks
   (n16 → ~3 Hz, n32 → ~1.2 Hz). Between generations the policy is open-loop.
   Reaction latency ≈ replan interval, controlled by `n_action_steps` × tick
   duration.
6. **The commitment–reactivity dilemma, mapped empirically**: short chunks
   (n_action_steps 6–12) replan inside the ambiguous home region and
   mode-flip (lift, settle, lift…); n16+ commits and reaches, but at low fps
   the resulting multi-second blindness knocks the block over. Resolution
   that finally worked: slow motion **in the weights** (2× stretch) + dwell
   **removed from the data** (trim) + native 30 fps execution + n32 —
   simultaneously smooth, slow, committed, and ~1 s reactive.
7. Supporting cast: slew limiter tuned to demo p99 (teleop 4, kinesthetic 7
   units/tick — dataset-dependent); start-pose reset to the median training
   first-frame; ramped (3 s interpolated) homing instead of goal-position
   snap; ENTER-gated episodes; Ctrl-C freezes in place (torque held, limiter
   bounds the residual jump).

### Ops

- **Instrument rollouts from day one.** Per-tick npz logs (observed / raw /
  post-limiter commands + planner buffer depth) + per-camera mp4s turned
  "feels jerky" into measured, distinguishable causes (tracking lag vs splice
  jumps vs staircase). Every deployment diagnosis above came out of the logs.
- **wandb artifact upload is ON by default** and silently duplicates every
  checkpoint into `~/.local/share/wandb` — combined with dense checkpointing
  (~1.2 GB/save) it filled a collaborator's disk to 100% and killed a run
  mid-save. Use `--wandb.disable_artifact=true`; prune superseded checkpoints
  aggressively (keep val curves + winners).
- Two-GPU split works well: big GPU trains or serves inference, second GPU
  runs the val-loss sidecar; ripper (Brian's RTX 6000) as overflow trainer.
  Pin GPUs by UUID — CUDA and nvidia-smi enumerate them in opposite order.

## Data-scaling probe (2026-07-30 evening)

Same dataset (trim2×), same val episodes, same recipe — only the training
episode count varies (seeded subsets, lateral mix preserved). Directly
comparable val numbers, for once:

| train episodes | best val | Δ vs next |
|---|---|---|
| 40 | 0.0204 (@1.5k) | — |
| 60 | 0.0173 (@2.5k) | −15% |
| 80 | 0.0164 (@2.5k) | −5% |

The curve is **alive but decelerating** on this val set. Two readings, both
true: (a) more same-distribution data still helps, so collection is
justified; (b) the val set is drawn from the same left-biased distribution
(1–2 right-side episodes of 5), so it structurally *cannot* measure
right-side coverage gains — the region where rollouts actually fail. Volume
alone extrapolates to only ~5–10% further val gain per doubling; **targeted
right-side coverage is the high-leverage axis**, and its payoff will show in
rollout success maps, not this curve.

## RTC prefix-inpainting (2026-07-30, deployed)

Implemented in the async planner (default on, `--no_rtc` to disable):
RePaint-style — every denoise step, forward-noise the committed actions
(last executed + still-buffered, normalized) to the current noise level and
clamp them into horizon slots 0..P-1; the UNet's temporal receptive field
makes the free tail a continuation of the executing mode. Splices land
inside the clamp (seamless by construction). Lineage: RePaint (CVPR'22) /
Diffuser conditioning / Real-Time Chunking (Physical Intelligence '25).

Measured: splice deltas 7-10° → ~1° (indistinguishable from within-chunk).
Gotcha found on-robot: with 10 one-shot DDIM steps the clamp boundary
harmonizes imperfectly — a 5-10° step relocates to the clamp's end (jumps
clustered exactly +3 ticks post-splice). Fixed with a 4-slot linear bridge
from the last committed action into the generated tail (single conditioned
mode — smoothing, not mode-averaging). Residual: ≤5° rare steps, capped by a
speed-matched limiter (per-dataset: 7 @1×, 3.5 @2×, 2.5 @3×). Escalation
ladder if jerk returns: DDIM 20 + trigger 10 (flags only) → RePaint
resampling (code).

## Rollout verdicts (2026-07-30, full smoothing stack)

- trim1× (full speed): not working.
- **trim2×: successes** — but only left-side placements in particular
  configurations (see bias analysis). Smooth with slight residual
  chunk-boundary jerk.
- trim3×: struggled to lift pre-RTC (chunk-disagreement churn at ⅓-speed
  signal); RTC verdict pending.
- scale40 (half data): staged for behavioral A/B vs the 80-episode model.

## Servo I/D gain sweep (2026-07-31, completing Brian's P sweep)

Method: Brian's `replay_measure_so101.py` benchmark (open-loop kinesthetic
replay, |obs−cmd|, delay/residual decomposition), n=6 episodes (trim1x 0–5)
per setting via `sweep_setting_so101.py` (ramps between episodes; disconnects
its client around each measurement — the host's PUSH socket round-robins
observations between connected clients; ends with a 10 s held-pose hunting
check with a stream-freshness guard). P=24 fixed throughout.

| setting | median | p95 | max | lag | residual |
|---|---|---|---|---|---|
| I=0 D=32 | 5.96 | 20.06 | 27.62 | 133 ms | 1.36 |
| **I=4 D=32** | 5.83 | 19.05 | 24.89 | 133 ms | **0.97** |
| I=8 D=32 | 5.96 | 19.71 | 26.10 | 133 ms | 1.10 |
| I=4 D=16 | 5.85 | 19.27 | 24.59 | 133 ms | 0.95 |

- **I=4 is the winner**: residual −29% (the gravity-droop component integral
  action was predicted to remove), every episode improved, tails improved,
  and no hunting at rest (held-pose pk-pk ≤ 0.44 units, verified on a fresh
  30 Hz stream). I=8 gives part of it back (mild windup on transitions).
- **D=16 changed nothing measurable** — lag did not drop below 133 ms, max
  did not degrade. Kept D=32 (fewer deltas from stock).
- **Lag is exactly 4 ticks (133 ms) in all 24 runs across every setting** —
  a hard floor no gain touches. This is the full command→observation
  round-trip (transport + host loop + servo), which is precisely the latency
  a deployed policy experiences → it directly calibrates the k-frame action
  shift: **k≈4 at 30 fps**. Our k-shift sweep trained k ∈ {1,2,3}; k3
  (100 ms) is closest, and a k4 variant is the natural follow-up.
- Final gains live on the host: **P=24 I=4 D=32** (flags on
  `so101_zmq_host.py`; defaults in `SOFollowerConfig` remain 16/0/32).
- Also fixed Brian's flagged `logging.basicConfig` no-op on the Pi host
  (`force=True`) — the `gains:` line now prints, and every sweep setting was
  verified live before measuring.
- Open question (Brian's): whether tracking gains transfer to policy
  jerkiness/success — his P=24-vs-16 rollout comparison is the pending test,
  now extendable to I=4.

**P-sweep replication (2026-07-31, n=6 vs Brian's n=3):** P16
6.77/22.4/29.8/144ms/2.03 (his: 7.50/22.5/29.7/167/1.96) — replicates. P24
5.96/20.1/27.6/133/1.36 (his: 6.30/19.2/25.6/122/1.38) — replicates. **P32
diverges in our favor**: 5.36/18.4/25.4/**106ms**/1.22 (his n=3 called 24≈32) —
with n=6, P=32 is genuinely better and, notably, **lag drops to 3 ticks** in
most episodes, disproving the "hard 4-tick floor": ~1 tick of the round trip
is servo response and P-dependent; the other ~3 ticks are pipeline
(obs sample→transport→conflate + cmd transport→host loop→bus write).
**Combo run P=32 I=4 D=32**: median 5.46, p95 18.8, lag 106 ms (5/6 episodes
at 3 ticks), residual 1.01, held-pose clean (pk-pk ≤ 0.44). Recommended
production setting — and it pairs with the **k3** action-shift model
(3-tick lead), already trained. Gains and k trade against each other:
finalize gains before choosing k.

## The noise-floor discovery (2026-07-31) — why 2× dithers and denoising can't fix it

All k-shift 2× variants dithered identically on the robot (left-right
hesitation; pan churn 25:1 vs net progress). Log analysis: the oscillation is
**within-chunk** (period ~8 ticks, only mildly splice-correlated) — the
generated chunks themselves wiggle. Offline sampler sweep on a real
observation settled the cause:

| sampler | within-chunk Δ p50 | demo target |
|---|---|---|
| DDIM-10 | 0.64 | 0.04 |
| DDIM-25 | 0.60 | |
| DDIM-50 | 0.59 | |
| DDPM-100 | 0.55 | |

**DP reproduces trajectories to ~±0.6 units/tick regardless of denoising
effort** — a model-precision floor, not sampler noise. At 1× the real signal
(~1.7–4 units/tick) dominates it; **2× stretching pushed slow-phase signal
below the floor**, so rollouts became noise-dominated. More denoise
steps/RePaint passes cannot help (measured). Mitigations: `--action_ema 0.3`
low-pass on executed actions (halves churn on logged rollouts, ~0.5 units
added lag; legitimately separable because noise is high-frequency and 2×
signal is slow) — but the strategic conclusion is that **stretching is now a
net negative**: its original purpose (compensating unmodeled lag) is solved
properly by the P32/I4 gains + k-shift lead, and its SNR cost is fundamental.
Current flagship config: **trim1× + k3 + P32/I4/D32 + fps30 + RTC stack**
(1×-k3 winner 1.5k, 1×-k4 winner 1.5k as bracket, both staged).

## The speed ladder & the assisted-grasp factorization (2026-07-31)

Since noise attaches to *actions* (per generated step) while playback rate
sets *seconds per step*, the right design is: train at 1× (per-step signal
3–6× above the floor), choose speed at deployment via fps, and match the
k-lead to tick duration (round trip ≈ 133 ms wall-clock):

| deploy fps | speed | ideal lead | model |
|---|---|---|---|
| 30 | 1× | 4 ticks | trim1x-k4 (1.5k) |
| 20 | 0.67× | ~2.7 ticks | trim1x-k3 (1.5k) ← flagship |
| 15 | 0.5× | 2 ticks | trim1x-k2 (2k) |

Same-wall-clock configs are NOT equivalent: trim2×@30fps vs trim1×@15fps
differ in noise draws per centimeter (2×), replan coin-flips per centimeter
(2×), training-target SNR, and duplicated-frame ambiguity — all favoring 1×.
On-robot: 1×-k3@20fps moved decisively (transport churn 1.7× vs 25× for the
2× ditherer).

**Assisted-grasp episode** (rollout_20260730_220951_ep1): with the block
hand-placed in the gripper, the policy executed transport→place→release
flawlessly, zero hesitation — first full task completion. The task factorizes
cleanly: post-grasp execution is solved; the sole remaining weakness is the
final-approach grasp (tip pushes block ~1 cm off, no corrective behavior in
the data). Keep "success given grasp" and "grasp success" as separate metrics.

## Lowering the DP noise floor — levers (2026-07-31)

The ~0.6-unit floor decomposes into: (a) score-estimation error, (b)
conditional ambiguity in data, (c) representation freedom for high-frequency
content. Levers by component: (a) **weight EMA — the lerobot DP port dropped
it** (original diffusion_policy trains a shadow-weight average and evals with
it; conventional for diffusion; needs a trainer patch + retrain, cannot be
retrofitted to a checkpoint); more data (sub-linear); (b) consistent demo
style, funnels/recovery structure, no dwell, no duplicated-frame stretching;
(c) smoothed targets (kavg), frozen initial noise across chunk generations
(one-line; correlates samples), best-of-N chunk selection (BID-style,
selection not averaging), and structurally: **diffuse spline/basis
coefficients instead of per-tick actions** (high-frequency jitter becomes
unrepresentable; big change, held in reserve). Camera resolution is a grasp-
precision lever, not a floor lever (1080p Arducams currently downscaled to
240×320 in the host).

## Data collection protocol v2 (for the fisheye-wrist session)

The new wrist lens changes the visual distribution — old wrist views are
off-distribution for the new camera, so **round 2 must stand alone**: target
~150–200 episodes (~1–1.5 h at demo pace).

1. **Pilot first**: record 5 episodes → run record→replay fidelity + data QA
   (per-tick deltas, camera health) → only then commit to the session.
2. **Recording hygiene**: start after gripping the arm, stop at release
   (+a beat); always complete the place (6 episodes never reopened last
   time); no <5 s episodes.
3. **Placement coverage is the point**: grid the workspace; right-heavy
   (correcting 47L/27R); include center; vary block orientation;
   **alternate sides within the session** (last time side was confounded
   with session time).
4. **Speed floor**: keep motion ≥ ~1 unit/tick even when careful — final
   5 cm in 1–2 s, not slow motion. Precision through *consistency and
   convergence*, not slowness (sub-floor creeping trains wobble).
5. **Grasp-centric**: deliberate hug-and-close, wrist camera sighted on the
   block through the approach (fisheye's job), consistent approach style per
   region (reduces multimodality); brief stable hold after grasp.
6. **~20% recovery demos**: brush/miss deliberately → back off →
   re-approach → succeed; vary the error direction. This builds the
   corrective funnel that plow-through revealed as absent.
7. **Fisheye specifics**: mount rigidly (lens motion = distribution shift);
   fix exposure; consider capturing wrist at 640×480 — the wide FOV spreads
   pixels, and retraining is required anyway, so capture high and downsample
   later if needed (LAN bandwidth is fine).
8. **After collection, before training**: QA pass (sub-floor segments,
   lateral grasp balance from FK, length outliers, never-reopen check) and a
   side/time-stratified val split.

## Next steps (current plan, in order)

1. **Brian: servo gain tuning**; acceptance test = replay trajectory error
   (incl. while holding the block). Re-run replay calibration after every
   gain change.
2. **Replay-based plant ID**: log (command, achieved) in replay_so101,
   fit per-joint delay/lag → choose k for the action shift.
3. **Data collection round 2**: right-heavy + center placements (correct
   the 47L/27R imbalance; alternate sides within-session), ~15–20% recovery
   demos, no dwell, always finish the place, ~100 episodes.
4. **Re-derive + retrain**: merge → trim → k-frame action shift → 2× stretch
   → side/session-stratified val split → aug + dense saves. Placement-grid
   success protocol before/after to measure the gain.
5. Held in reserve: inverse-plant command correction; RePaint resampling /
   DDIM-20 for residual boundary jerk; EE-action deployment via the
   hardened 5-DOF IK (projected DLS + rest tie-break + multi-start).
