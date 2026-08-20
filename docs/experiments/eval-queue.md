# Eval queue — active candidates

*Maintained checklist. Finished/obsolete entries live in the findings log in
`modality-ablation-plan.md`. All winners staged under `/mnt/nvme/lerobot/outputs/`.*

**Current best**: `dp_mix_teleop2x_3000` — several grasps in <10 tries
(2026-08-01), vs 1-2 per 100+ historically. Deploy clean:
`--fps 15 --interp_substeps 2 --n_action_steps 15`, **no** `--gripper_binarize`,
no bias — the commitment crutches only helped conflicted models; careful
closed-loop descent is the learned skill.

Common flags:
`--task_description "Put the red block into the bin" --num_episodes 5
--fps 15 --interp_substeps 2 --n_action_steps 15`
(remote_ip defaults to ethernet .17 — pass nothing)

Template:
```bash
python examples/alohamini/evaluate_so101.py \
    --hf_model_id /mnt/nvme/lerobot/outputs/<MODEL>/pretrained_model \
    --train_dataset_root /mnt/nvme/lerobot/yosubshin/<DATASET> \
    <common flags>
```

## Pending

| model | dataset root (`yosubshin/…`) | tests | status |
|---|---|---|---|
| `dp_teleop_solo_wristonly` | `so101_teleop_trim1x_wristonly` | teleop-solo vs teleop2x at matched cameras — does kinesthetic in the mix add anything? solo ≈ 2x → go pure-teleop | **EVAL'D: most decisive grasp behavior — block between jaws several times. Best near-grasp model.** |
| `dp_teleop_graspx2_wristonly` | `so101_teleop_graspx2_wristonly` | **teleop-ONLY + grasp-segment 2× oversampling** — 65 clip-episodes of [close−3 s .. close+2 s] appended (they point into the same video files; no re-encode); no kinesthetic at all | **EVAL'D: FAILED — random wandering during approach. Grasp-clip share (40% of frames) starved/distorted the approach distribution.** |

Priority-0 hardware control (if still undone): replay a TELEOP episode with
the block placed per its `videos_1080p/` clip. Partially superseded —
teleop2x's grasps prove the tolerance is achievable — but replay still
bounds the open-loop share of remaining misses.

New candidate:

| model | dataset root (`yosubshin/…`) | tests | status |
|---|---|---|---|
| `dp_mix_phasesplit_wristonly` | `so101_mix_phasesplit_wristonly` | **phase-split composition**: kinesthetic approach-only (70 eps truncated pre-grasp, zero grasp contamination) + full teleop (76) + moderate grasp clips (65; 17% of frames vs graspx2's 40%). Approach gets volume, grasp gets purity. Trained with drop_n_last_frames=63 (no padded windows at truncation cuts). | **EVAL'D: FAILED — erratic arm, worse than champion. Post-mortem: lost 42% of champion's volume (kinesthetic back-half discarded), 3-way distribution fragmentation, k3 (champion=k2). Video-state alignment verified clean.** |

New candidate v2:

| model | dataset root (`yosubshin/…`) | tests | status |
|---|---|---|---|
| `dp_mix_grasppure2x_wristonly` | `so101_mix_grasppure2x_wristonly` | **champion minus ONE thing**: exact teleop2x recipe (kin k2 + teleop×2, ~90k frames) but kinesthetic episodes SPLIT into approach + post-grasp segments — only the close windows excised (kin gripper range in data: 45.6-100, zero table-grasp closes; 4 regrip segments excluded from training). Single-variable test of the grasp-conflict theory at full volume. | **EVAL'D: 2nd best — approach smooth, grasp hesitation persists despite excision.** |

New candidate v3:

| model | dataset root (`yosubshin/…`) | tests | status |
|---|---|---|---|
| `dp_mix_grasppure2x_wide_wristonly` | `so101_mix_grasppure2x_wide_wristonly` | v2 with the excision widened to 2.5 s before each kinesthetic close — near-block endgame is teleop-only (v2 eval: approach smooth, residual grasp bimodality in the uncovered 0.5 s zone) | **EVAL'D: FAILED — can't find block, erratic approach.** |

2026-08-01 eval verdicts (rollouts): teleop2x = best APPROACH (volume) but
bimodal near grasp (kinesthetic + teleop grasp distributions conflict, policy
alternates per chunk); teleop-solo = most DECISIVE grasp; graspx2 = approach
broken (over-weighted grasp clips). Phase-split is the synthesis.

## Probes on whichever model grasps best
- deeper squeeze (`--gripper_close_bias`) only if grasps hold then slip on lift
- `--n_action_steps 20/30` control runs (expect worse, per 2026-08-01 finding)
- side-balance: grasp success split L vs R (teleop data skews 27L/49R)
- grip tape on the block (still untried)

## 2026-08-02 verdict: data-surgery line CLOSED

Dose-response across cuts: phase-split (heavy cut) erratic; grasppure2x
(2 s excision) 2nd-best but grasp hesitation persists; grasppure2x-wide
(2.5 s pre-cut) erratic, can't find block. Artificial mid-trajectory cutting
does not remove the grasp bimodality and destabilizes rollouts (volume loss
+ segment fragmentation). CHAMPION REMAINS dp_mix_teleop2x_3000.

Next phase (data frozen at so101_mix_k2teleop2x_wristonly):
1. Model-side, one at a time: weight EMA (DP-paper trainer feature), then
   spline/basis action heads (requires eval-harness RTC rework).
2. Parallel: collect ~80-100 more teleop eps (left-heavy, grasp-dense,
   ~25% recovery demos) — the confirmed data lever.

## 2026-08-02 EMA result (data frozen at champion recipe)

Weight EMA (LEROBOT_EMA=1, DP-paper schedule) beats raw at EVERY checkpoint
on the shared val set. EMA@3000 = 0.0163, the project'''s best val loss
(champion raw was 0.0186; same-recipe seed-rerun raw peaked 0.0179 — so EMA'''s
gain exceeds seed variance). Staged: dp_teleop2x_EMA_3000 (primary),
dp_teleop2x_rerun_3500 (seed-control). Next model-side experiment after
rollout verdict: spline/basis action head (needs eval-harness RTC rework).

**2026-08-02 ROLLOUT VERDICT: dp_teleop2x_EMA_3000 is the NEW CHAMPION —
2/5 full task successes, failures are close calls.** First repeatable
end-to-end policy. EMA is now a standard recipe ingredient (LEROBOT_EMA=1).

**2026-08-02 crop-augmentation ablation: WASH.** EMA+crop(216x288) peak
0.01625 vs EMA-only 0.01632 — 0.4%, an order below measured seed variance
(~4%). Crop adds nothing at 240x320 alongside affine/jitter transforms;
NOT added to the recipe. (crop EMA@3500 kept on ripper only.) Remaining
architecture-lane candidates: GroupNorm+from-scratch encoder (tests the
paper'''s BatchNorm-EMA warning), spline/basis head (parked, needs RTC
harness rework). Recipe otherwise verified against Chi et al. reference:
optimizer/scheduler/noise-process identical; horizon choices deliberate.

**2026-08-02 GroupNorm ablation: −14% val, new best.** From-scratch
ResNet18+GroupNorm (reference-faithful) EMA peak 0.0140 @4500 vs
BatchNorm-pretrained incumbent 0.0163 — the audit's last divergence was the
largest. Staged for rollout: `dp_teleop2x_EMAgn_4500` (dataset root
`so101_mix_k2teleop2x_wristonly`, usual flags). 8000-step version training
(raw curve still descending at 5000).

**2026-08-03 ROLLOUT: dp_teleop2x_EMAgn_4500 is the NEW CHAMPION — easy
scenarios (block near home) now succeed almost always. Night-and-day vs
two days ago. GroupNorm+from-scratch is recipe-standard.**

## 2026-08-03 grid eval of champion (dp_teleop2x_EMAgn_4500)

right 3/5 (misses close or far-right) | center 2/5 (LYING-DOWN blocks
always fail) | left 0/5 (one grasp, dropped in transport).
Maps exactly to data: 27L/49R skew -> left dead; upright-only demos ->
lying blocks OOD; thin far-right coverage; marginal grip in transport.

## Targeted collection spec (~100 teleop eps, frozen recipe)

- ~45 left placements (incl. far-left), ~25 center, ~20 right (bias far-right)
- ~30% of episodes with the block LYING DOWN (all regions) — demonstrate the
  side-grasp/reorient behavior; this is a new sub-skill, not just coverage
- ~20-25% recovery demos: missed grasp -> re-open -> retry; PLUS mid-transport
  drop -> re-approach -> re-grasp (observed failure mode)
- firm trigger squeezes (transport-drop prevention), finish every place
- after collection: retrain frozen recipe (teleop_new x2 + old mix? decide by
  count — target >=230 effective episodes), EMA+GroupNorm+from-scratch,
  5000 steps, eval same grid for before/after comparison

**2026-08-03 mirror-augmentation experiment (user-driven, overriding the
analysis): val says YES — EMA 0.0126 @7500 vs champion 0.0140, on
original-side val episodes.** Staged: dp_teleop2x_EMAmirror_7500 (dataset
root so101_mix_k2teleop2x_mirror_wristonly). Rollout must test: (a) LEFT
placements (0/5 baseline), (b) transport direction after left grasps (the
phantom-bin failure mode predicted by the feasibility analysis). If left
improves without transport confusion, mirroring = free 2x on all future
data and the rejection analysis was too conservative.

**2026-08-03 ROLLOUT: MIRROR AUGMENTATION VALIDATED — left 0/5 -> 3/5.**
dp_teleop2x_EMAmirror_7500 is the new champion. Mirror step is now standard
in the derivation pipeline (applies to the incoming v2 collection too:
~100 new eps -> ~200 effective, x2 again with existing data).

**2026-08-03 full grid (mirror champion): left 3/5, center 3/5, right 2/5
(total 8/15 vs 5/15) — remaining failures are corner cases: L/R edges,
very far/near, lying-down blocks, slips.**

## Recovery-demo protocol (v2 collection)

STAGE the failure state; never perform the failure: set up post-failure
scene by hand (block displaced ~2cm / escaped after partial close /
dropped in transit zone), gripper empty nearby, THEN press Enter and
record only the recovery (re-approach, re-grasp, deliver). Zero
miss-frames enter the data — DAgger-style state coverage without teaching
the mistake. Keep naturally-occurring in-line misses (organic corrections,
rare, diluted). Do NOT perform deliberate fake-bad approaches from normal
states. Corner-case placements (edges, very far/near, lying) remain the
other collection priority.

**2026-08-03 GRAND V2 trained: dp_grand_v2_10500 — EMA 0.0111 @10500 (vs
0.0126 mirror champion, on a harder val incl. corner/recovery eps).**
Data: kin-k2 x1 + (teleop v1 + v2-corners + recoveries) x2, all mirrored =
740 entries / 344k frames. Full recipe. AWAITING GRID EVAL vs 8/15
baseline — watch corner cells (edges/far/near/lying) and first-ever
recovery behavior after failed grasps.

Eval:
python examples/alohamini/evaluate_so101.py \
  --hf_model_id /mnt/nvme/lerobot/outputs/dp_grand_v2_10500/pretrained_model \
  --train_dataset_root /mnt/nvme/lerobot/yosubshin/so101_grand_v2_mirror_wristonly \
  --task_description "Put the red block into the bin" \
  --num_episodes 5 --fps 15 --interp_substeps 2 --n_action_steps 15

## 2026-08-03 grand-v2 grid: 7/15 (R 3/5, C 2/5, L 2/5) — slight regression
with major qualitative wins and a diagnosed culprit.

WINS: lying-down grasp works (new sub-skill); far/near pickups work (right);
recovery behavior EXISTS (re-grasp attempts replace erratic flailing —
staging protocol validated for covered states).

REGRESSION: left approach shows left-right SHAKING (right smooth) ->
hypothesis: mirror-center offset. If c_pan is off by delta, mirrored data
is offset 2*delta (~5-10mm at reach); left = sparse-real (correct) +
mirrored-right (offset) = two attractors -> oscillation. Right = abundant
consistent real data -> smooth. Same delta explains far-edge failures
(angular offset x max radius). DECISIVE TEST: no-mirror control (training)
— if its left approach is smooth, theory confirmed.

NEW FAILURE: knocked-over block -> policy pushes with half-closed gripper
(state never staged). 

Next session checklist:
1. Eval no-mirror model, esp. left-approach smoothness.
2. MEASURE mirror centers (30 s): arm dead-straight -> read shoulder_pan;
   gripper level -> read wrist_roll. Rebuild mirrors about measured centers.
3. Mini-collection (~25 eps): knocked-over recoveries (open-first, lift,
   side re-approach) + far-edge anchors both sides.

**2026-08-03 mirror-only diagnostic staged: dp_grand_v2b_mirroronly_6500**
(EMA 0.0120 — identical to its all-original twin dp_grand_v2b_6000 at
0.0119: mirrored data equally learnable, no gross transform inconsistency).
Twin rollout protocol: same placements on both models —
(1) competence map should swap sides cleanly;
(2) consistent lateral grasp bias in mirror-only = center offset, miss
distance ~ 2*delta (ruler measurement -> exact mirror correction);
(3) any shakiness in mirror-only (zero data conflict by construction) =
transform-level inconsistency (camera plane / image geometry).

**2026-08-03 recovery lead-in dose ladder** (stationary failure-state frames
KEPT per episode): trim=5 -> recovery too weak (pushing); keeplead=all
(~41-88) -> freeze-mode regression on NORMAL cases (v2b, user-reported —
stationary near-block states alias mid-grasp states); lead20=20 -> v2c,
staged as dp_grand_v2c_6500 (EMA 0.0120). Eval v2c on: (1) normal cases
(regression gone?), (2) knocked-over recovery (open-first?). Note:
mirror-only diagnostic model carries keeplead — discount any near-block
hesitation in its rollouts; its lateral-bias readout is still valid.

**2026-08-03 MIRROR RETIRED** (mirror-only eval: outward eccentricity-growing
bias = fisheye flip about pixel centerline invalid off-axis; see experiments
doc). Path forward: v2c (no-mirror, lead20 recovery) as base; next
collection = real left/edge episodes. Mirror-only + mirrored datasets kept
for the record; mirror step removed from the standard pipeline.

**2026-08-04 staged: dp_grand_v2d_7000** (EMA 0.0119) — recovery x4
(80 entries, ~4% of frames), sole change vs v2c. Eval: provoke pushing
failures; open-lift fires -> dose theory holds, tune weight/collect more
staged (incl. fail-forward: brief push then recover). Still pushes ->
dose exonerated, velocity-signature gap confirmed -> build the DAgger
takeover tool (policy drives, t=freeze, align leader, t=engage+record).

**2026-08-04 ROLLOUT: v2d recovery WORKS — drops/slips followed by
successful re-approach (vs prior random/shaky flailing). Dose theory held:
recovery recipe = staged states + 20-frame lead-in + x4 weight.
dp_grand_v2d_7000 is the new working base.** DAgger tool deprioritized to
reserve (dose was the bottleneck, not state coverage). Remaining levers:
full-grid number for v2d; fail-forward staged demos + more recovery variety
at x4; real left/edge collection (mirror replacement).

**2026-08-04 v2d grid: 8/15 (R 2/5, C 4/5, L 2/5)** — same total as mirror
champion, different shape: center strong, failures now PARTIAL recoveries
(lift starts, truncates into pushing). ONE-CHUNK RECOVERY hypothesis: at
n15 the recovery chunk executes ~1s; the half-lifted state visually
matches a normal approach state (96% descend supervision) -> replan
re-samples descend -> truncation. Zero-cost test: provoked failures with
--n_action_steps 25 on same checkpoint. Data fix regardless: fail-forward
demos supervise the mid-recovery ambiguous states. Next session: ~12
fail-forward + ~40 left/edge/far-right/too-close eps -> retrain rec x4.

## 2026-08-04 n-steps sweep + recovery synthesis

- v2d grid at n10: R 3/5, C 2/5, L 3/5. **n_action_steps 10 is the new
  deploy default** (finer closed-loop correction at contact -> better grasp
  accuracy). n25 fails even easy scenes — the apparent recovery lift was
  the PRE-failure chunk plan executing open-loop, not recovery.
- Recovery diagnosis (user): recovery is a FAN (arbitrary block-gripper
  pose after failure), approach is a FUNNEL (converging from home) —
  20/200 recovery episodes is inverted vs state diversity. Universal
  failure: pushes or partial open/close beside block instead of full-open
  + re-angle + re-grasp. Also: hover on DIAGONAL blocks (orientation gap).

## Dynamic-failure collection protocol (next session, ~30 eps)

Teleop the failure FOR REAL (varied: off-center pinch, tip-over, edge
clip), **pause ~0.5 s after the block escapes**, then demonstrate full
recovery (full open -> re-angle -> grasp -> deliver). Post-hoc auto-cut
keeps only [failure-settle : end] via block-blob tracking (validated:
~100% blob lock; protocol pause makes the cut trivial) — real failure
dynamics enter the data, the causing actions do not. Fan coverage: vary
where the block ends up. Include diagonal-orientation placements in the
same session. Then retrain rec x4, eval grid at n10.

**2026-08-04 staged: dp_grand_v2e_9000** (EMA 0.0120 @9000) — first full
DAgger-flywheel model: v2d recipe + 30 interventions x4 (upsampled
15->30 fps, untrimmed) + 14 successes x1. Grid eval at n10; key question:
do policy-native corrections complete the open-lift-regrasp where staged
data plateaued?

Eval:
python examples/alohamini/evaluate_so101.py \
  --hf_model_id /mnt/nvme/lerobot/outputs/dp_grand_v2e_9000/pretrained_model \
  --train_dataset_root /mnt/nvme/lerobot/yosubshin/so101_grand_v2e_dagger_wristonly \
  --task_description "Put the red block into the bin" \
  --num_episodes 5 --fps 15 --interp_substeps 2 --n_action_steps 10

**v2e grid verdict (2026-08-04)**: right 4/5 (best cell ever; the miss was
a WORKING recovery cut off by the 60 s episode cap), center 2/5
(REGRESSION: +/-90deg wrist-roll twisting + hover on easy straight-approach
blocks), left 1/5 (hover above block, no commit; n15 also 1/5). Diagnosis:
intervention stream x4 overdose — untrimmed interventions are all
near-block/slow/cautious frames (~19% of training mass) aliasing normal
final-approach states => "hover, don't commit" prior (keeplead lesson,
third occurrence). Secondary: teleop time-on-task bias (slow twisted
approaches dominate frame counts vs quick aligned grabs => twist modes
oversampled); upsampling frame-dup weakens visual motion cues in dagger
data. Meta-rule now standing: EVERY new corrective stream gets its own
dose ladder, starting LOW.

**2026-08-04 staged: dp_grand_v2f_9000** (EMA 0.0120 @9000) — v2e with ONE
knob turned: interventions x4 -> x1 (dataset
`so101_grand_v2f_dagger1x_wristonly`, 454 eps / 213,016 frames / 423 train).
Prediction: center+left hesitation clears toward v2d approach quality;
right recovery partially survives. If recovery vanishes on right, next
rung is x2. Protocol change: **--episode_time 90** is now standard
(recovery legitimately needs the clock).

Eval:
python examples/alohamini/evaluate_so101.py \
  --hf_model_id /mnt/nvme/lerobot/outputs/dp_grand_v2f_9000/pretrained_model \
  --train_dataset_root /mnt/nvme/lerobot/yosubshin/so101_grand_v2f_dagger1x_wristonly \
  --task_description "Put the red block into the bin" \
  --num_episodes 5 --fps 15 --interp_substeps 2 --n_action_steps 10 \
  --episode_time 90

**v2f grid verdict (2026-08-04): 9/15 — NEW BEST** (v2d was 8/15).
right 4/5 (miss: block too close — likely wrist-cam occlusion/blind spot,
arm went to mid-air and stalled), center 2/5 (hesitation reduced; new
misses = shallow grab-and-lift without full descent, the old "quick grab"
mode), left 3/5 (up from 1/5; hesitation cleared; robust recovery not yet
observed). Intervention-overdose diagnosis CONFIRMED: x1 restored
committed approaches. Standing champion: dp_grand_v2f_9000.

Next: dagger round 2 at NATIVE 30 fps (tool patched: records at
fps*interp_substeps; policy mode = one frame per substep w/ fresh obs;
takeover loop at substep tick; handover decay rescaled; resume of 15 fps
datasets refused — use so101_dagger_v2). Train new dagger eps with
LEROBOT_DROP_N_FIRST=4. Session targets: (1) twisted-wrist stuck states
(take over when it twists wrong, demo untwist->re-approach), (2) center
shallow-grasp (take over at premature lift, demo full descent), (3)
too-close right placements (check block visibility in obs stream first).
RL deferred: covariate-shift gaps are DAgger's home turf; RL enters at
~80-90% plateau when it needs autonomous reward + resets.

**2026-08-05 staged: dp_grand_v2g_9000** (EMA 0.0119 @9000, tied best with
@6500) — DAgger round 2: v2f mix + 33 native-30fps interventions x1
(targets: twisted-wrist stuck states, center shallow-grasp, close-right)
+ 10 trimmed successes x1. ~63 interventions total, all x1.
Dataset so101_grand_v2g_dagger2_wristonly (497 eps / 230,844 frames / 463
train). First run trained with LEROBOT_DROP_N_FIRST=4 (round-2
interventions open with 4 policy-context frames, obs-only). Key questions:
does the twist-hesitation on center clear, and do twisted-state recoveries
appear?

Eval:
python examples/alohamini/evaluate_so101.py \
  --hf_model_id /mnt/nvme/lerobot/outputs/dp_grand_v2g_9000/pretrained_model \
  --train_dataset_root /mnt/nvme/lerobot/yosubshin/so101_grand_v2g_dagger2_wristonly \
  --task_description "Put the red block into the bin" \
  --num_episodes 5 --fps 15 --interp_substeps 2 --n_action_steps 10 \
  --episode_time 90

**v2g grid verdict (2026-08-05): 7/15** — right 3/5 (far-right recovery
still pushes block side instead of re-approaching), center 3/5 (twist
hesitation GONE, but now fails to twist when it should; recovery twists
appropriately), left 1/5 (won't twist on blocks needing it — prefers
straight approach; mode suppression). SEESAW DIAGNOSIS: round-2 early
takeovers were one-sided in condition (all "aligned block -> go
straight"), so the model learned the MARGINAL (straight preferred), not
the CONDITIONAL (straight iff aligned). v2f (9/15) remains deploy
champion. Also noted (user): no contact-force awareness — presses hard,
block flies on slip; position-error (cmd - measured) is a derivable
proxy signal, candidate future obs augmentation.

**2026-08-05 building: dp_grand_v2h** — v2g minus the 13 straight-only
round-2 interventions (wrist-roll excursion <= 25u for all of takeover +
intervention), keeping the 20 twist-involved ones. Tests: does the twist
mode return on left/diagonal while keeping recovery-twist? Dataset
so101_grand_v2h_twistkeep_wristonly (484 eps / 226,456 frames / 450
train), LEROBOT_DROP_N_FIRST=4, standard recipe.
Real fix beyond this ablation: BALANCED conditional coverage — matched
"diagonal/lying -> twist" demos to pair with the "aligned -> straight"
guidance, so intervention batches never shift the marginal.

**2026-08-05 staged: dp_grand_v2h_7500** (EMA 0.01171 @7500 — best raw val
of any run; val remains a weak grid predictor). v2g minus 13 straight-only
r2 interventions. Attribution test only (seesaw knob, not a conditional
fix — user called this correctly).

**OBSERVABILITY AUDIT of base teleop demos** (twist onset vs approach
progress, threshold 15u, onset <40% of descent-to-grasp = "early"):
tel_old: 76 eps = 29 straight / 33 EARLY pre-twist / 14 late-twist.
tel_v2: 51 eps = 20 straight / 25 EARLY / 6 late.
=> ~74% of all twist supervision is confounded (demonstrator pre-twisted
on privileged eye-view before the wrist cam could resolve orientation).
Only 20/127 demos twist on observable evidence — same reason recovery
twists work (near-block, orientation visible).

**2026-08-05 building: dp_grand_v3a (obs-filter)** — v2h minus the 58
early-pre-twist B/C episodes (both dup copies). 368 eps / 168,626 frames /
342 train. Pure quality-vs-quantity test, $0 collection.
PREDICTIONS (logged before eval): if left IMPROVES, the 20 clean twist
demos + recovery/dagger twists suffice — quality wins. If left collapses
further, twist-absence confirmed — the observability-disciplined
collection session (neutral wrist, orient-on-visibility, orientation-
balanced bins) is mandatory. NOTE: val set differs from v2h — val losses
not comparable across these runs.

**2026-08-05 staged: dp_grand_v3a_6500** (EMA 0.01239 @6500; val set
differs from v2h — not comparable). Obs-filter model: v2h minus 58
early-pre-twist base demos. Predictions logged above. Grid at n10,
episode_time 90.

Eval:
python examples/alohamini/evaluate_so101.py \
  --hf_model_id /mnt/nvme/lerobot/outputs/dp_grand_v3a_6500/pretrained_model \
  --train_dataset_root /mnt/nvme/lerobot/yosubshin/so101_grand_v3a_obsfilter_wristonly \
  --task_description "Put the red block into the bin" \
  --num_episodes 5 --fps 15 --interp_substeps 2 --n_action_steps 10 \
  --episode_time 90

**2026-08-05 AUDIT CORRECTION — v3a SHELVED (do not grid-eval).** User
spot-check overturned the joint-space proxy: "early pre-twist" eps B44/C16
are FINE (fisheye resolves orientation from home height), while "late"
B40 commits toward a gripper-OCCLUDED block. The criterion is VISIBILITY
AT COMMITMENT, not twist timing. Red-blob visibility audit (detector
visually validated; blue distractors ignored): commit-without-recent-
visibility (blob <150 px^2 within 20 frames pre-commitment ~ chunk
horizon): B 5 eps (21,25,48,54,63) + C 8 eps (2,4,16,18,27,28,30,50)
= 13/127 (10%), and C16 is a borderline false positive per user. B54 is
the memory archetype: startA 2516 -> occluded at commit. CONCLUSION:
observability contamination is ~10%, NOT 46% — quality-filtering alone
cannot move the grid much; the twist-conditional gap is coverage/absence,
not contamination. Strategy: proceed to observability-disciplined,
orientation-balanced collection with ACQUIRE-THEN-ACT protocol (center
block in frame before aligning/approaching — also fixes the occlusion-
memory class). Optional v3b (drop 13 flagged eps) is cheap but low-EV.

**2026-08-05 VISIBILITY AUDIT FINAL** (after two user-spot-check
corrections: (1) color detector was fine, commitment detector wasn't —
straight approaches with no pre-grasp pan/twist got their "commitment"
stamped in the TRANSPORT phase where the grasped block is occluded ->
false flag on the best episodes, e.g. B54; fixed by bounding commitment
search to pre-grasp; (2) visibility threshold calibrated by user's eye:
~100 px^2 is readable, use <50 as blind).
BLIND-COMMIT EPISODES (correct behavior unlearnable — demonstrator flew
by eye with block outside fisheye FOV): B: 21, 25, 48 | C: 2, 18, 28,
30, 50 (8/127 = 6%). Borderline kept: B63, C4, C16, C27 (98-141 px^2).
KEY INSIGHT (user): these blind demos are concentrated at FAR-LEFT,
FAR-RIGHT, and CLOSE placements — precisely the persistently failing
grid cells. Those cells' only supervision is unlearnable rote motion;
explains e.g. v2f close-right "went to mid-air and stalled".
CONSEQUENCES: (a) exclude the 8 from the next merge (v3 build);
(b) collection session for far/close/edge bins MUST demonstrate
SCAN-ACQUIRE (pan/tilt until block enters frame, center it, then
approach) — this is new skill data, not just rebalancing.

**2026-08-05 WIDE AUDIT (blind STARTS, user-directed)**: 21/127 base eps
start with no readable block in frame (B: 21,23,25,40,48,52,60 | C: 2,4,
5,7,17,18,21,22,27,28,30,38,42,50; C=27% — corner dataset). Blind leads
0.6-2.6 s; every episode eventually discovers the block. Only ~1.3% of
frames BUT maximally harmful placement: all 21 leads start from
near-identical observations (home pose, no red in frame) with DIVERGENT
action targets (pan left/right/forward per hidden placement) = identical
obs -> conflicting supervision at the start state of the hard cells.
Mid-air-and-stall is the policy sampling from that conflicted mixture.
REMEDY: repair not exclusion — trim blind eps to their discovery frame
(post-discovery supervision is clean; keeps hard-cell coverage).
Building so101_teleop_trim1x_visfix + so101_teleop_v2_trim1x_visfix for
the v3 merge. Supersedes the exclude-8 plan.
COLLECTION COROLLARY: the scan-acquire demos for out-of-view placements
must use ONE CONSISTENT sweep pattern (e.g., always rise-then-pan the
same direction until the block appears, then center) — a deterministic
search is unimodal and learnable; ad-hoc direct reaches are exactly the
conflict being removed.

**2026-08-06 NEW COLLECTION QA + v3 LAUNCH.**
so101_teleop_visibility: 20 eps / 14,155 frames / 30 fps / wrist-only,
episodes ~24 s (acquire-then-act takes time). QA: 19/20 blind starts
(placements deliberately out of initial view); ZERO twist-while-blind
(all 10 twist eps twist with block readable >= 100 px^2); the 7
"blind-commit" flags are FALSE POSITIVES — the commitment detector cannot
distinguish a scan sweep from a direct reach; sweep-shape check shows a
DETERMINISTIC left-first sweep (pan -> ~-90 -> return right) identical
across all long searches => unimodal, learnable. Protocol held.
Prep: so101_teleop_visibility_trim (13,491 frames, dwell-trimmed —
motion-based trim preserves scans; val [1,15]).
**v3 mix** (so101_grand_v3_visibility_wristonly, 524 eps / 251,768
frames / 486 train): v2h composition with B/C swapped for visfix repairs
(blind leads trimmed at discovery) + tel_vis x2. Dagger2 stays at the
20 twist-involved (v2h choice); the 13 straight-guidance interventions
stay out — aligned->straight conditioning is now supplied by the balanced
session itself. Training: 9000 steps, standard recipe, DROP_N_FIRST=4.
NEXT: stage best EMA as dp_grand_v3_<step>; FINAL EVAL = 30-placement
grid (2x the 15-grid for signal), v3 vs v2f champion, episode_time 90,
n10. Watch: left cell, out-of-view placements (scan behavior should
appear for the first time), twist-on-diagonal at center.

**2026-08-06 staged: dp_grand_v3_8500** (EMA 0.0111 @8500 — lowest val of
the project; caveat: val set includes tel_vis eps, not directly comparable
to earlier runs). THE endgame model: visfix bases + observability session.

Eval (30-placement final, vs dp_grand_v2f_9000 baseline):
python examples/alohamini/evaluate_so101.py \
  --hf_model_id /mnt/nvme/lerobot/outputs/dp_grand_v3_8500/pretrained_model \
  --train_dataset_root /mnt/nvme/lerobot/yosubshin/so101_grand_v3_visibility_wristonly \
  --task_description "Put the red block into the bin" \
  --num_episodes 10 --fps 15 --interp_substeps 2 --n_action_steps 10 \
  --episode_time 90

## 2026-08-06 FINAL 30-PLACEMENT SHOWDOWN — PROJECT CONCLUSION

dp_grand_v2f_9000:  R 4/10  C 4/10 (no recoveries seen)  L 5/10 = 13/30 (43%)
dp_grand_v3_8500:   R 4/10  C 4/10                        L 3/10 = 11/30 (37%)

STATISTICAL TIE (diff 2, SE of diff ~3.5). Two conclusions:
1. The observability arc (visfix repairs + disciplined balanced session)
   did NOT raise the grid ceiling. Pre-registered outcome (b): boundary
   result — data composition/curation on this rig is exhausted.
2. v2f's 9/15 (60%) was a high roll; true level ~43%. The 7-9/15
   oscillation across v2d..v2h was largely binomial noise around a
   plateau reached at v2d. n=15 evals cannot rank models this close;
   n=30 barely can.

Remaining failure modes (far-right/too-close, fine-orientation grasps)
now attributed to perception/hardware floor, not data: 320x240 fisheye
(distant block ~15 px — orientation marginal), gear backlash vs
mm-tolerance grasps, zero contact sensing (position-target control).
CHAPTER CLOSED. Champion for demos: dp_grand_v2f_9000 (or v3 — equal
within noise; v3 has scan data + repaired conditioning, prefer it for
out-of-view placements if scans confirmed qualitatively).
Queued next chapters: UMI rig (pre-registered modality prediction),
QDD arm (backlash/contact hypotheses), optional cmd-minus-measured
contact-signal obs experiment.

**2026-08-06 qualitative addendum (user):** v3 does NOT reproduce the
demonstrated scan sweep, but from occluded starts it "slowly finds its
way to the block" — vs v2f's mid-air stall on the same placements. Skill
transferred in EFFECT, not FORM: only ~6 episodes contained real sweeps
(short blind starts discovered in 2-3 s without sweeping), and 6 eps @ x2
is far below every behavior-emergence dose measured here (staged recovery
needed 20 @ x4). Diluted behavior = weak "move until red appears" prior +
wide fisheye + closed-loop replanning. Known cure if reopened: 10-15
deliberately-far placements @ x4.
CHAMPION CALL: dp_grand_v3_8500 for demos — equal total (tie), strictly
better occluded-start behavior + repaired conditioning. v2f retained as
the pre-observability reference model.

**2026-08-06 EPILOGUE EXPERIMENT — DP paper 5.4 backbone-LR prescription.**
Setup: pretrained BN ResNet18 (port defaults) but backbone fine-tuned at
10x LOWER LR (1e-5; U-Net/pool/projection at 1e-4) via
LEROBOT_BACKBONE_LR_SCALE=0.1 (optim/factory.py env patch, param-group
split; 60 backbone tensors verified in log). Same v3 dataset/val/recipe
as dp_grand_v3_8500 — directly comparable.
RESULT: EMA best 0.01164 @8500 vs from-scratch 0.01110 @8500.
Gap shrinks ~14% (full-LR fine-tune, teleop2x era) -> ~5% (paper
prescription), near the ~3% seed-variance floor. REINTERPRETATION: most
of the "pretraining hurts" effect was full-LR fine-tuning destroying
pretrained features; the paper's LR discipline recovers nearly all of
it. From-scratch still edges it and remains the recipe. Val-only
verdict (no rollout); checkpoint staged dp_grand_v3ft_8500 if a
behavioral check is ever wanted. Caveat: tests the LR prescription with
ResNet18, not the paper's ablation-best (CLIP ViT).

**2026-08-06 seed control (user-requested):** from-scratch v3 recipe
rerun with --seed=2000, everything else identical. EMA best 0.01104
@7500 vs seed-1000's 0.01110 @8500 => measured seed noise ~0.5% on this
dataset. The backbone-LR-prescription gap (0.01164, ~5%) is therefore
~10x the noise floor: FROM-SCRATCH'S EDGE IS REAL on this task, though
small. Final §4 statement: the paper's 10x-lower-LR prescription
recovers most of the pretraining penalty (14% -> 5%); a genuine ~5%
residual favors from-scratch here. (Historical caveat: teleop2x-era
seed variance was larger — smaller dataset; noise floors are
per-dataset.) Run pruned, not staged (control only).

**2026-08-06 v3ft grid verdict (user): 6/30** — R 3/10, C 2/10, L 1/10
with a NOVEL failure mode: 4 drops during transport (never observed in
any prior model). vs from-scratch v3 11/30 and v2f 13/30 on the same
protocol. Val gap was only ~5% but behavioral gap is large — another
instance of "val action-MSE under-predicts rollout differences" (cf.
wrist-only ablation).
TRANSPORT-DROP HYPOTHESES (pre-registered before the remaining evals):
(A) EMA/BatchNorm-buffer mismatch: our EMA checkpoints pair averaged
weights with NON-averaged BN running stats (buffers aren't EMA'd) —
exactly the interaction Chi et al.'s GroupNorm swap exists to avoid.
The paper's throwaway sentence strikes a THIRD time.
(B) Pretrained-feature distraction: ImageNet features are semantic;
during transport the background sweeps rapidly and the block is a
close-up occluded blob — feature noise -> slight gripper drift -> drop.
DISCRIMINATOR: the queued BN-scratch checkpoint has (A) but not (B).
If BN-scratch also drops during transport => BN(+EMA) is the culprit.
If BN-scratch transports cleanly => pretrained features are.
(Raw non-EMA v3ft ckpt would test (A) directly but was pruned; only
EMA staged.)

**2026-08-06 full-LR pretrained result:** EMA best 0.01264 @5000 (raw
0.01417; note EMA peak came EARLY, @4000-5000, then degraded — full-LR
fine-tune overfits fastest of all cells). Penalty vs from-scratch:
+14%, REPLICATING the teleop2x-era gap on v3 data. Ladder now: naive
fine-tune +14% -> paper 10x-lower-LR +5% -> from-scratch (seed spread
0.5%). Staged: dp_grand_v3ftfull_5000 (grid-eval candidate; predict
worst of the four, expect transport drops if either hypothesis holds
since it has BOTH BN+EMA mismatch AND degraded pretrained features).

**2026-08-06 ENCODER MATRIX COMPLETE** (v3 dataset, identical recipe,
EMA-view best val; seed noise 0.5%):
  from-scratch + GroupNorm  : 0.01110 / 0.01104 (two seeds)  <- recipe
  from-scratch + BatchNorm  : 0.01122  (@8000; staged dp_grand_v3bnscratch_8000)
  pretrained + BN, 0.1x LR  : 0.01164  (grid 6/30, transport drops)
  pretrained + BN, full LR  : 0.01264  (staged dp_grand_v3ftfull_5000)
DECOMPOSITION (normalization controlled): pretraining penalty ~3.7%
even with the paper's LR prescription (7x noise floor); GroupNorm worth
~1.4% over BatchNorm from scratch (~3x noise — small but likely real,
sharper than the teleop2x-era "~3% ~= noise" call); naive LR costs a
further ~9%. GRID DISCRIMINATOR PENDING: BN-scratch transport behavior
decides EMA/BN-buffer mismatch (A) vs pretrained-feature noise (B) for
the v3ft drops.

Eval:
python examples/alohamini/evaluate_so101.py \
  --hf_model_id /mnt/nvme/lerobot/outputs/dp_grand_v3bnscratch_8000/pretrained_model \
  --train_dataset_root /mnt/nvme/lerobot/yosubshin/so101_grand_v3_visibility_wristonly \
  --task_description "Put the red block into the bin" \
  --num_episodes 10 --fps 15 --interp_substeps 2 --n_action_steps 10 \
  --episode_time 90

**2026-08-06 VALIDATION ACTION ERROR probe** (user-proposed, per the
offline-metrics paper: full-inference action error vs noise-prediction
val loss). Scorer: scratchpad/action_error_eval.py — DDIM-10
deploy-matched, n_action_steps=10, identical fixed noise draws per
checkpoint, normalized MAE/RMSE over executed window, v3 val split.
RESULTS: scratch_GN 0.04488 / scratch_BN 0.04527 / pretrained_slowLR
0.04415 (BEST) / pretrained_fullLR 0.05333 (worst, +20%).
READING: action error agrees with val loss that fullLR is clearly
degraded, but RANKS THE BEHAVIORALLY-WORST MEASURED MODEL (slowLR,
6/30 + transport drops) BEST among the rest — top-3 spread ~2% is
likely within noise (no action-error seed floor available; seed-2 ckpt
was pruned). CONCLUSION for this project: on-distribution offline
metrics (either kind) score demo-state reproduction; the slowLR model
reproduces demos fine and fails off-distribution/closed-loop — exactly
the paper's own caveat ("cannot reliably predict task success rates or
reveal failure modes"). Offline metrics remain useful for catching
clear degradation (fullLR flagged by both); ranking near-peers still
requires the grid. Tiny n caveat: 4 models / 2 grids — this
illustrates the caveat, it does not refute the paper's correlation
claim (theirs was across many models).

**2026-08-06 PHASE-RESOLVED OFFLINE METRICS (user-proposed grasp-area
narrowing, extended to approach/grasp/transport + gripper-dim):**
scratchpad/phase_metrics_eval.py, results json alongside. OUTCOME: the
pre-registered hope was FALSIFIED — pretrained_slowLR (grid 6/30,
transport drops) is best or tied-best in EVERY phase and metric,
INCLUDING transport-phase gripper MAE (0.1115, lowest; the exact
dimension+phase of its real failure). fullLR worst everywhere
(consistent). The two scratch cells are offline-identical.
CONCLUSION: the failure is invisible on demonstration states at any
granularity we can slice — it lives off-distribution / closed-loop.
Demo-state offline metrics (noise loss OR action error, whole or
phase-restricted) can catch gross degradation but cannot rank
near-peers behaviorally; the grid (or sim) is irreplaceable.
FOOTNOTE (weak, multiple-comparisons caveat): grasp-phase NOISE loss is
the single metric whose ordering matches the behavioral ranking
(GN 0.0152 < BN 0.0154 < slowLR 0.0157 << fullLR 0.0168) — 1 of ~8
phase-metrics, no noise floor available; treat as coincidence unless it
repeats on the BN-scratch grid.

**2026-08-06 MODALITY-BUCKETED offline metrics** (val split composition:
24 eps = 6 kin / 11 teleop-base / 7 corrective — 25% KINESTHETIC,
inherited per-source since v2b; applies to every grand-mix val number).
Findings: (1) kin contamination is BENIGN FOR RANKINGS — per-metric
orderings stable across buckets — but it DILUTES signal: on teleop-base
noise loss the behavioral ordering (GN 0.00979 < BN 0.00992 < slowLR
0.01006 << fullLR 0.01095) is cleaner than on the mixed val, and the
kin bucket slightly inverts it (slowLR best there). (2) The consistent
cross-slicing pattern: NOISE LOSS tracks the behavioral ranking
(matches grid in teleop-base AND grasp-phase buckets); full-inference
ACTION ERROR crowns slowLR in every bucket — the OPPOSITE of the cited
paper's correlation claim, on our task. Margins 2.5-3.5% vs ~0.5-1%
floors — directionally consistent across independent buckets, so
plausibly real. RECOMMENDATIONS (next chapters): draw val from
deploy-relevant modality only (or report per-modality); keep plain
noise-loss val as the cheap gate; decisions on the grid.

**2026-08-06 BN-scratch grid verdict (user): 13/30** (R 6/10, C 3/10,
L 4/10) — BEST grid of the encoder matrix (ties v2f champion) — but
WITH 3 TRANSPORT FAILURES (left). Discriminator reading: drops are NOT
pretrained-feature-specific; both BN models drop (3-4x, left-heavy),
GN didn't (in its earlier session) => hypothesis (A) EMA/BN-buffer
mismatch leads. User's live alternates: (1) hardware degradation
between eval sessions, (2) left->bin long-carry weakness shared by all
models. Controls: quick left-placement recheck of dp_grand_v3_8500
(hardware-now test) + fresh-seed GN full grid.
Note: BN 13/30 vs GN 11/30 within noise, but GroupNorm's offline edge
has NOT translated behaviorally.
**Staged: dp_grand_v3seed2** (retrained seed-2000 GN, EMA 0.01106
@7500 — replicates pruned original's 0.01104). Grid discriminates:
drops => hardware/spatial (not BN); clean => BN+EMA mechanism confirmed
(paper's GroupNorm sentence hat trick).

## 2026-08-06 SAME-MODEL REPEAT — EVAL RELIABILITY RECKONING

dp_grand_v3_8500 grid, 2nd attempt: 8/30 (R 3/10 w/ 1 TRANSPORT DROP,
C 1/10, L 4/10) vs 11/30 first attempt. Same checkpoint.
STATISTICS: at p~0.32, n=30, per-run SD ~2.6, SD of a difference ~3.6.
11-vs-8 is within 1 SD — consistent with pure binomial noise, though
the cell structure (C 4->1) and the drop hint at environmental drift
too. CONSEQUENCES:
- The encoder-matrix BEHAVIORAL ranking is unresolvable at n=30/arm:
  GN 11&8 / BN 13 / slowLR 6 — only slowLR-vs-BN (~2 SD) is even
  suggestive. DOWNGRADE all behavioral encoder claims to val-only.
- Transport-drop mechanism: GN dropped too (1x) => BN+EMA hypothesis
  WEAKENED; drops now span all 3 recent sessions and no early session
  => session-level environmental factor (gripper pad wear? block wear?
  loose screw?) is the leading suspect. Inspect hardware.
- Power reality: detecting a 0.30-vs-0.45 gap at 80% power needs
  ~170 trials/arm — infeasible manually. 30-trial grids resolve only
  gaps >= ~10/30.
- Project-level conclusions SURVIVE (rested on large effects/existence
  proofs/val ladders): plateau ~30-45%, recipe restorations, modality
  finding, recovery existence. What dies: every <=3-success comparison
  ever interpreted, now including parts of the seesaw narrative.
IF FURTHER COMPARISONS EVER MATTER: paired interleaved A/B protocol —
placement template sheet (fixed positions+orientations), alternate
models on the SAME placement back-to-back within one session, McNemar
on pairs. Kills drift confound, roughly doubles effective power.

**2026-08-06 GRASP-EVAL HARNESS BUILT** —
examples/alohamini/grasp_eval_so101.py. Matched-block grasp-only A/B/N:
taught pre-grasp slot poses (~1 s autonomy), per-slot randomized
rotation over all checkpoints x reps, closure detection (gripper cmd <
45 for 3 ticks, 8 s budget), scripted policy-independent stress
(lift-to-hover w/ grip held 1.5 s + hold 2 s + pan jiggle +/-3 x2),
ordinal scoring 0/1/2 (0 auto), live tallies, JSONL log, optional
per-trial wrist(+forward) clips. Defaults to the 5 encoder-matrix
checkpoints (GN s1000/s2000, BN, PT-slowLR, PT-fullLR).
Usage: (1) --teach --slots 20 (guide via leader, ENTER per slot);
(2) run: python examples/alohamini/grasp_eval_so101.py [--save_clips
--cameras wrist,forward]. Read order: fullLR separates? (instrument
validation) -> GN seed pair tight? (noise floor) -> then BN/slowLR.

**2026-08-07 GRASP HARNESS REDESIGNED (user protocol — supersedes the
teach-mode version):** rollout-seeded start states instead of taught
poses. Per scene: place block anywhere (remember spot); seeding model
(round-robin) rolls out from home; SPACE at the moment the block enters
the gripper opening captures pose + last 2 policy-tick obs; the seeding
rollout continues uninterrupted = its own first trial. Remaining
(model, rep) randomized: re-place block, ramp to captured pose, obs
history primed with the captured pre-handoff frame (identical moving
context for all models), fresh chunk (deliberately NO RTC action-prefix
prefill — the seeder's frozen actions would inject its style into every
other model's first chunk). Closure/stress/score unchanged. Advantages
over teach mode: start states guaranteed on-distribution, no teach
session, no leader needed, seeding trials are free.

**2026-08-07 grasp session final config: 6 arms.** Added GN_raw
(dp_grand_v3_raw8500 — raw weights of the v3 rerun, same seed/recipe as
GN_s1000's EMA view) to behaviorally test the EMA restoration (§2's
headline; val +9%, behavioral evidence previously anecdotal). Isolated
from the EMA×BN-buffer question since GN has no running stats. Session:
6 models x 2 reps x N scenes; ~2-2.5 h at 15 scenes. Rig-drift caveat
added to blog §4.

**2026-08-18 BLOG SPLIT INTO TWO PARTS.** Part 1 (training):
so101-part1-training-DRAFT.md — original draft + expanded corrective-
stream subsection in §3 (dose ladders / condition balance / effects-
leak-before-forms) + §4 bridge to part 2 + closing pointer. Part 2
(evaluation): so101-part2-evaluation-DRAFT.md — new full draft: noise
hierarchy (grid oscillation -> 30-trial deflation -> same-model repeat
-> drift), offline-metrics probe incl. val-composition hygiene, grasp
benchmark design (all pilot-earned elements), 27% flip floor + funnel
reframing + pre-registered QDD flip-rate prediction, three-wrong-stories
time-lapse (n=12/24/46), surviving findings + day-one protocol
checklist. Old single-file draft replaced by a pointer stub. Figures
still wanted: three-snapshot tally evolution (part 2 centerpiece),
noise-hierarchy ladder, flip-floor schematic.

**2026-08-19 figure-audit cell (user-caught):** the recipe-ladder
figure's "port defaults" bar had augmentation ON (true default is OFF)
and the +12% aug claim was kinesthetic-era, never measured on the final
mix. Trained the missing cell (mix_k2teleop2x, pretrained BN, no EMA,
transforms OFF, same episodes/val as ripper_ema.sh): best 0.02025 @3000
then hard overfit to 0.0272 @5000. Aug on this dataset = -8% AND
flattened overfit tail. Ladder now 5 bars, all one dataset: 0.0203 ->
0.0186 (+aug) -> 0.0163 (+EMA) -> 0.0144 (+scratch) -> 0.0140 (+GN),
-31% total. Figure rebuilt: bars from zero, floating EMA dot removed
(was a different run's scalar). Blog §2 aug bullet softened to
measured ~8%. Run pruned from ripper.
