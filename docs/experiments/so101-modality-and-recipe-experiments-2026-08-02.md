# SO-101 experiments, 2026-08-01 → 08-02: modality, grasp forensics, recipe audit

Sequel to `so101-kinesthetic-experiments-2026-07-30.md`. Headline: from
~1% grasp rate to the **first repeatable end-to-end policy — 2/5 full task
successes with close-call failures** (`dp_teleop2x_EMA_3000`), on wrist-only
fisheye input.

## 1. The champion recipe

- **Data**: `so101_mix_k2teleop2x_wristonly` — kinesthetic-k2 (76 eps) +
  teleop ×2 (76 eps duplicated), wrist fisheye only, ~99k effective frames.
- **Training**: LeRobot DP + image transforms + `LEROBOT_EMA=1`
  (weight EMA, DP-codebase power-0.75 schedule), 5000 steps, save 500,
  `drop_n_last_frames=31`. EMA checkpoint at step 3000.
- **Deploy**: `--fps 15 --interp_substeps 2 --n_action_steps 15` — no
  binarize, no bias, no action-EMA. Clean.

## 2. The modality finding (the project's key scientific result)

**What matters in demonstration data is whether the demonstrator
experienced the plant the policy will control.**

Evidence chain:
1. Kinesthetic-only and teleop-only models (76 eps each) both failed the
   grasp (~1-2 per 100+) while reaching/transporting/placing fine.
2. Gripper trace forensics vs training distributions: policies mode-average
   the open (100) and close (kinesthetic ~52 / teleop ~30) modes into a
   useless 60-70 hover that is also proprioceptively identical to "holding" —
   driving descend-lift oscillation (reversals uniform across the replan
   grid → NOT chunk-boundary averaging).
3. `deepgrip` (kinesthetic closes deepened in-data to teleop semantics):
   **no effect** → the labels were not the problem.
4. `teleop2x` (teleop behavior amplified): **~20× grasp-rate jump** → the
   *behavior distribution* was the problem.
5. Kinesthetic's specific deficit: hand-guiding bypasses the arm's backlash,
   so demos contain no closed-loop micro-corrections; teleop is performed
   through the real plant (leader quantization, follower backlash, visual
   feedback), so the operator's compensations are IN the data — and the
   policy needs exactly those at deploy. Corollary: no per-frame transform
   (k-shift, close-deepening, learned lag models) can synthesize
   undemonstrated corrective behavior.

## 3. Data-composition experiments (all rollout-judged)

| model | composition | verdict |
|---|---|---|
| teleop2x | kin-k2 + teleop×2 | best approach; pre-EMA champion |
| teleop-solo (wrist-only) | teleop only | most decisive grasp; weaker approach |
| graspx2 | teleop + grasp-clips (40% of frames) | approach destroyed — clip share starved the approach distribution |
| phase-split | kin approach-only (back half discarded) + teleop + clips | erratic — lost 42% volume, 3-way fragmentation |
| grasppure2x | champion + kin grasp-window excised (2 s) | 2nd best; hesitation persists |
| grasppure2x-wide | excision widened to 2.5 s pre-close | erratic, can't find block |

**Conclusion: the data-surgery line is closed.** Whole-episode reweighting
works; mid-trajectory excision/clipping destabilizes (volume loss, segment
fragmentation, padded-window edges) and never removed the grasp
bimodality. Video-state alignment was verified pixel-exact in the merged
datasets — failures were compositional, not data bugs.

## 4. Grasp forensics + the crutch-flag story

- The eval slew limiter silently clipped every gripper transition to
  7 units/tick (≈ half demo close speed at fps15) → slippery block escapes.
  Fixed: gripper exempted (`--gripper_max_delta 20` default).
- Kinesthetic action≡state ⇒ commanded closure = block width = zero squeeze
  margin (no force signal in the modality). Teleop's trigger passes block
  width — structural advantage.
- `--gripper_binarize` / `--gripper_close_bias` / larger `n_action_steps`:
  helped ONLY conflicted-data models; on teleop-weighted models they HURT —
  the careful descend-and-feel behavior is the learned skill, not
  hesitation. **Good data needs no crutches.**

## 5. Recipe audit vs Chi et al.'s reference implementation

Two large silent divergences in LeRobot's DP port, both restored, both wins:

| item | reference | LeRobot | result of restoring |
|---|---|---|---|
| weight EMA | on by default (codebase, not paper text) | absent entirely | **+9% val at every checkpoint, > seed variance (~4%); 0→2/5 successes** |
| image augmentation | random crop primary | off by default | ~12% val (restored 07-29 via jitter/affine) |
| random crop 90% | primary aug | `crop_shape: None` | tested 08-02: **wash** (0.4%) — jitter/affine covers it at 240×320 |
| `drop_n_last_frames` | consistent w/ horizon | stale 7 after horizon 16→64 bump | we'd been correcting it unknowingly (31) |
| encoder | ResNet-18 from scratch + GroupNorm | ImageNet-pretrained + BatchNorm (deliberate, PR #3202) | **restored 08-02: −14% val (EMA 0.0140 vs 0.0163) — the paper's BatchNorm-vs-EMA warning was real, and from-scratch beats ImageNet pretraining even at 152 eps. Biggest single win after EMA itself.** |
| optimizer/scheduler/noise | AdamW 1e-4 (0.95,0.999) / cosine+500 / DDPM-100 squaredcos ε | identical | ✓ |

**Encoder-ablation confound (unresolved)**: the GroupNorm experiment
necessarily changed TWO factors at once — (a) BatchNorm → GroupNorm and
(b) ImageNet-pretrained → from-scratch initialization — because GroupNorm
cannot be swapped into pretrained weights (they are calibrated against
BN's statistics; the code refuses). The observed −14% val / near-always
easy-scenario success is the JOINT effect; we cannot attribute it between:
  1. GN removing the EMA-vs-BN-running-stats mismatch,
  2. GN removing batch-composition noise and train/eval mode split,
  3. from-scratch features being better suited than ImageNet's
     (classification invariances + rectilinear prior vs our
     fisheye/localization needs).
The confound IS separable in one direction: a **BatchNorm + from-scratch**
run (use_group_norm=false, pretrained_backbone_weights=null) is legal and
would isolate the initialization factor from the normalization factor.
Not yet run — recorded here so the claim "GroupNorm did it" is not
over-stated in any write-up.

**Port-divergence lesson**: ports translate model definitions faithfully but
rewrite training harnesses — recipe details (EMA, augmentation, schedule
constants) are the casualties. When adopting any ported policy, diff the
full training recipe against the reference before trusting results.

EMA mechanics note: the shadow copy is passive (never in the forward pass,
never receives gradients — training is bit-identical with or without it);
`decay` is the retention factor (Wikipedia's 1−α), schedule
`1−(1+t)^-0.75` capped 0.9999 ⇒ effective averaging window ~500 steps late
in training. Enable: `export LEROBOT_EMA=1`; eval the `<step>_ema` siblings.

## 6. Standing next steps

- **Collection (highest value)**: ~80-100 teleop eps on the frozen recipe —
  left-heavy (27L/49R skew), grasp-dense, ~25% explicit recovery demos
  (miss → re-open → re-descend → succeed).
- **Architecture lane**: GroupNorm CONFIRMED (now recipe-standard: --policy.use_group_norm=true --policy.pretrained_backbone_weights=null); 8000-step control confirms the peak plateaus at ~3500-5500 steps (0.0139, = 5k run within noise) — 5000-step budget stands; then
  spline/basis action head (parked — requires RTC-harness redesign).
- **UMI pre-registration** (made before rig exists): UMI shares
  kinesthetic's no-plant-in-loop flaw for the ARM (expect casual
  positioning) but not the GRIPPER (real fingers, real contact). Predicted
  mitigation: mix with teleop / deliberately careful final approaches.
