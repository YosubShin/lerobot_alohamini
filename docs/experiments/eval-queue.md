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
| `dp_mix_phasesplit_wristonly` | `so101_mix_phasesplit_wristonly` | **phase-split composition**: kinesthetic approach-only (70 eps truncated pre-grasp, zero grasp contamination) + full teleop (76) + moderate grasp clips (65; 17% of frames vs graspx2's 40%). Approach gets volume, grasp gets purity. Trained with drop_n_last_frames=63 (no padded windows at truncation cuts). | **staged: `dp_mix_phasesplit_wristonly_1500`** (val peak 0.0209 @1500) |

2026-08-01 eval verdicts (rollouts): teleop2x = best APPROACH (volume) but
bimodal near grasp (kinesthetic + teleop grasp distributions conflict, policy
alternates per chunk); teleop-solo = most DECISIVE grasp; graspx2 = approach
broken (over-weighted grasp clips). Phase-split is the synthesis.

## Probes on whichever model grasps best
- deeper squeeze (`--gripper_close_bias`) only if grasps hold then slip on lift
- `--n_action_steps 20/30` control runs (expect worse, per 2026-08-01 finding)
- side-balance: grasp success split L vs R (teleop data skews 27L/49R)
- grip tape on the block (still untried)
