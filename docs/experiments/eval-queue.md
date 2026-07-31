# Eval queue — staged models awaiting rollouts

*Maintained checklist. Fill the result column after each session; keep rows
sorted by priority. All winners staged under `/mnt/nvme/lerobot/outputs/`.*

Common flags (current best config):
`--task_description "Put the red block into the bin" --num_episodes 5
--fps 15 --interp_substeps 2 --n_action_steps 20 --gripper_binarize`
(remote_ip defaults to ethernet .17 — pass nothing)

Template:
```bash
python examples/alohamini/evaluate_so101.py \
    --hf_model_id <MODEL>/pretrained_model \
    --train_dataset_root /mnt/nvme/lerobot/yosubshin/<DATASET> \
    <common flags>
```

| pri | model (`/mnt/nvme/lerobot/outputs/…`) | dataset root (`yosubshin/…`) | tests | result |
|---|---|---|---|---|
| 1 | `dp_mix_deepgrip_2500` | `so101_mix_k2deep_teleop_wristonly` | kinesthetic gripper deepened IN DATA — removes the "close=52" mode entirely (root-cause fix for the 60-70 mode-average hover) | |
| 2 | `dp_mix_teleop2x_3000` | `so101_mix_k2teleop2x_wristonly` | teleop episodes 2× weight — dilutes ambiguous kinesthetic modes (val peak 0.0186, best of any run) | |
| 3 | `dp_mix_k2teleop_wristonly_3000` | `so101_mix_k2teleop_wristonly` | baseline 50/50 mix rerun WITH `--gripper_binarize` (n20 already confirmed helping without it) | |
| 4 | `dp_fisheye_k0_wristonly_1500` | `so101_fisheye_trim1x_wristonly` | no lead compensation at all — does k hurt grasp commitment? (val 0.0204 is inflated: action≡state lets the model copy proprioception) | |
| 5 | `dp_fisheye_k1_wristonly_1500` | `so101_fisheye_trim1x_k1_wristonly` | minimal lead | |
| 6 | `dp_teleop_2cam_1500` | `so101_teleop_trim1x` | teleop solo + binarize (its natural deep closes + decisive trigger) | |
| 7 | `dp_mix_k3teleop_wristonly_3000` | `so101_mix_k3teleop_wristonly` | k3 flavor of the mix (only if k2 mix underwhelms) | |
| 8 | `dp_fisheye_trim1x_k2_20260731/checkpoints/001500` | `so101_fisheye_trim1x_k2` | kinesthetic solo two-cam + `--gripper_close_bias 12` (bias now fires only <45) | |

Also worth one probe each, on whichever model grasps best:
- `--n_action_steps 30` (even longer commitment; reaction latency tradeoff)
- `--gripper_closed_cmd 20` (deeper squeeze if grasps hold but slip on lift)
- no `--gripper_binarize` control run (isolate its contribution)

Notes / findings log:
- 2026-07-31: n_action_steps 15→20 reduced descend-lift oscillation
  (user-verified on mix_k2). Oscillation = state-image mode conflict,
  gripper mode-averaged into 60-70 "holding" ambiguity zone; reversals
  uniform across replan grid (not chunk-boundary averaging).
- Physical adjunct still untried: grip tape on the block.
