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
