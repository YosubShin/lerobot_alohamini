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
