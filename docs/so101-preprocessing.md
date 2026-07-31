# SO-101 preprocessing checklist — dataset → training-ready

Every filtering/derivation step we validated across the 2026-07-30/31
experiments, in the order they must run. Reference scripts:
`examples/alohamini/preprocessing/`. Context: the experiments log.

## A. Episode-level filtering (QA gate — run first)

1. **Drop too-short episodes** (mis-recordings): duration < 5 s auto-drop,
   5–8 s review by hand. (Precedent: ep40 of the first kinesthetic set was a
   1.8 s outlier and measurably hurt training.) Normal episodes: 12–19 s.
2. **Drop sensor-fault episodes**: any camera byte-identical for ≥ 1 s
   (frozen feed — pilot eps 2–4), or recorded below ~28 Hz true obs rate.
   Data recorded after 2026-07-31 is protected by recorder guards; older
   data must be scanned.
3. **Keep intentional recovery demos** (~20% by protocol); drop only
   *accidental* failures (block dropped and episode ended without recovery).
4. **Sanity: action ≡ state exactly** (kinesthetic only): max|a−s| must be
   0.0 — anything else means the wrong recording path. (Teleop: action =
   leader pose, deliberately ≠ state.)
5. **Timestamps uniform**: dt = 1/fps ± nothing, no gaps.
6. **Gripper signature**: each episode should traverse open → closed (grasp)
   → open (release); flag episodes without it. Also flag gripper commands
   faster than ~8 units/tick — the servo saturates near 10 and the policy
   would learn untrackable snaps.
7. **Left/right balance**: compute grasp-side split (+y = left, physically
   verified); > 60/40 imbalance → collect more on the weak side (the 47L/27R
   imbalance produced a left-only policy). Don't fix by dropping episodes.
8. **Contiguous episode indices**: discarded episodes leak their index
   (`clear_episode_buffer` doesn't reset the counter), leaving gaps that make
   the loader's sufficiency check fail — it then silently falls back to the
   HF hub and 404s. Check `set(range(total_episodes)) == data episode set`;
   repair by renumbering data + meta + sidecar + 1080p clips (2026-07-31:
   teleop set had ghosts 13/14, and `info.json` counted their frames).

## B. Sample-level derivation (order matters)

1. **Trim leading/trailing dwell**: per-frame speed = max-per-joint
   |Δaction|; moving = speed > **0.5 units/tick**; cut before first and
   after last moving frame with a **5-frame margin**. Rationale: 32% of raw
   frames were near-still, teaching a "stay" mode (dither at episode start).
   Cross-checked against grasp/release event detection: the trim never cuts
   into the task.
2. **No time-stretching.** 1× only — stretching pushes real per-tick signal
   below DP's ~0.6 units/tick reproduction floor (dithering). Speed is a
   *deployment* knob (`--fps` ladder), not a data transform.
3. **k-frame action shift** (kinesthetic only — lead compensation for
   action ≡ state): `action(t) = state(t+k)`, applied AFTER trimming, on the
   execution timeline. k matches deploy fps: **k4 @ 30 fps, k3 @ 20 fps
   (flagship), k2 @ 15 fps** (~100–133 ms command→obs round trip; re-measure
   over ethernet — it was calibrated on WiFi and may now be smaller).
   Teleop data needs **no shift** (natural plant lead).
4. **Val split**: seeded RANDOM episode split (~6%), written to
   `meta/val_episodes.json` — never last-N (late episodes differ by
   operator drift). For the new datasets: stratify by grasp side and by
   session. Same val episodes propagate to every derived variant.

## C. Training-time flags (not data transforms — see commands doc §6)

- `--dataset.image_transforms.enable=true` (~12% better; off by default)
- `--policy.drop_n_last_frames=31`
- `--save_freq=500` — true val peak ≈ 2k steps (~4 epochs); 5k misses it
- `--wandb.disable_artifact=true` (else silent checkpoint copies fill disks)
- Non-contiguous `--dataset.episodes` lists require the EpisodeAwareSampler
  reindexing fix (committed 2026-07-30).
- Val loss selects checkpoints *within* a run only — rollouts arbitrate
  across recipes/action spaces/modalities.

## D. Fisheye-era additions (datasets recorded ≥ 2026-07-31)

- `meta/episode_wallclock.jsonl` + `videos_1080p/` ride along in the dataset
  root; they are provenance/SLAM assets — keep them out of training features
  and carry them through derivations (copy the jsonl like tasks.parquet).
- Known-bad episodes in `local/so101_fisheye_red_pick`: **eps 2–4**
  (frozen wrist) — exclude at derivation time.
