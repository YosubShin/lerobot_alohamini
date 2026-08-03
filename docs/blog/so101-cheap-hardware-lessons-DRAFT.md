# Field notes: teaching a $300 arm to pick up a block (the long way)

*DRAFT skeleton — target audience: people running robot-learning experiments
on cheap hardware (SO-101-class arms) who want to train a simple model
themselves rather than fine-tune a giant VLA. This is a journey post: the
dead ends are the content, not the final recipe.*

---

## 0. What we set out to do

- One SO-101 follower + Pi host + a $30 fisheye wrist camera. Task:
  "put the red block into the bin." Policy: vanilla Diffusion Policy
  (LeRobot port), wrist-camera-only, trained on a single consumer GPU.
- Where it ended up: from ~1% grasp rate to [TODO: final grid numbers]
  across a 15-placement eval grid, with recovery behavior, in about a week
  of evenings.
- What this post is NOT: a recipe dump. The recipe is three flags and a
  data mix; the transferable part is *how each piece was found*.

## 1. Pick a small model, not a large VLA — iteration velocity is the whole game

- Training runs are ~15–40 minutes on one GPU (RTX-6000-class; a 3090
  works). We ran up to [TODO: count] trainings in a single day.
- The working rhythm: launch a training → walk to the rig → eval the
  *previous* model while the next one bakes → notes → next hypothesis.
  Train time ≈ eval time means the GPU and the human pipeline at 100%.
- Every finding in this post is downstream of that loop being fast. With a
  multi-hour fine-tune, we'd have tested five hypotheses instead of fifty.
- [Anecdote: the day we ran 6 dataset-composition ablations before dinner.]

## 2. Don't blindly trust a port — audit the training recipe against the reference

- Our biggest single-day gains came from *restoring things the framework's
  port had silently dropped* from the original Diffusion Policy recipe:
  - image augmentation: off by default in the port (+~12% val when enabled)
  - **weight EMA: entirely absent** (+9% val at every checkpoint, and the
    difference between 0 and our first repeatable successes)
  - encoder: port defaults to ImageNet-pretrained ResNet + BatchNorm; the
    paper uses from-scratch + GroupNorm (−14% val when restored; see §4
    for the decomposition)
  - a stale `drop_n_last_frames` default left over after upstream changed
    the horizon (we'd been fixing it by accident)
- The kicker: the paper's ONLY mention of EMA is one throwaway sentence
  explaining why BatchNorm was swapped for GroupNorm. That sentence
  encoded two of our three biggest wins, coupled. Read the reference
  implementation's configs, not just the paper.
- Lesson: ports translate *model definitions* faithfully but rewrite
  *training harnesses*, and recipe details (EMA, augmentation, schedule
  constants) are what fall out. Diff the whole recipe before trusting
  results.

## 3. Ramp data gradually: collect → train → eval → repeat

- We never collected more than ~100 episodes without training and
  evaluating in between. Every batch's protocol was shaped by the previous
  batch's failures:
  - gen 1 (76 eps): learned the task structure, failed the grasp →
    diagnosed WHY before collecting more
  - gen 2 (+76 teleop): modality ablation → found the real fix wasn't
    volume at all (see the modality story below)
  - gen 3 (+51 corner cases + 20 staged recoveries): targeted at the exact
    grid cells that failed
- A structured eval grid (15 fixed placements, scored per region) turned
  "collect more data" into "collect THESE 50 episodes." Coverage-targeted
  collection beat uniform collection every time.
- Counterfactual: if we'd collected 300 episodes on day one, they'd have
  been hand-guided (kinesthetic) — and we'd have baked in a subtle flaw
  that no amount of volume fixes:
- **The modality sub-story** (worth its own post): hand-guided
  (kinesthetic) demos taught reach/transport but could not teach grasping.
  Hand-guiding bypasses the arm's backlash — your fingers never feel the
  slop, so the data contains no corrective micro-behavior, and no
  per-frame transform can synthesize corrections that were never
  demonstrated (we tried: action time-shifts, gripper relabeling, segment
  surgery — all dead ends, all documented). Teleop demos are made THROUGH
  the plant, so the operator's compensations are in the data. One
  sentence: *what matters is whether the demonstrator experienced the
  plant the policy will control.*

## 4. From-scratch beat ImageNet pretraining (and how we almost mis-attributed it)

- Swapping to the paper-faithful encoder (from-scratch + GroupNorm) was
  −14% val and a night-and-day rollout difference.
- We almost wrote "GroupNorm did it" — the swap necessarily changes TWO
  things (you can't put GroupNorm into pretrained BN weights). A reader
  pushed for the confound to be documented; the disambiguating run
  (BatchNorm + from-scratch, perfectly legal) decomposed it: **~12% was
  dropping ImageNet, ~3% (≈noise) was the normalization**.
- Why pretraining hurt: ImageNet features are classification-invariant
  (position is deliberately discarded) and rectilinear; the task needs
  millimeter spatial localization through a fisheye. At ~100k frames of a
  single scene, a task-specific encoder wins.
- Boundary of the claim: single scene, narrow distribution. Wide-scene
  setups (or future UMI-style data) may flip this back.

## 5. Humans are (still) far better than AI at reading rollouts

- The pattern that produced every breakthrough: the human watches five
  rollouts, names a *behavioral* oddity in one sentence; the AI assistant
  turns it into log forensics, a mechanism, and an experiment by the next
  training cycle.
- Actual examples of one-sentence human observations that cracked it open:
  - "it hesitates — goes down, comes back up, repeats" → gripper trace
    analysis → mode-averaged gripper in a proprioceptive ambiguity zone
  - "teleop feels smoother, and I have to move carefully because of
    backlash" → the modality/plant hypothesis
  - "the left approach shakes; the right doesn't" → mirrored-data offset
    conflict [TODO: resolution]
  - "it pushes the knocked-over block instead of reopening" → discovered
    our trim step had deleted exactly the staged failure-state supervision
- The AI's complementary strengths: overnight-scale log analysis, never
  losing a hypothesis, catching its own wrong attributions when pushed.
  But it graded VIDEO poorly compared to a human glancing at the rig.
  Current-gen multimodal models don't yet parse "subtly wrong robot
  motion" the way any human does instantly.
- Practical takeaway: budget human eval time as a first-class resource;
  write down the verbal observations verbatim — they are the dataset.

## 6. Fine manipulation on cheap hardware is legitimately hard (and that's the point)

- The $300 arm's sins, all of which we hit: gear backlash that is
  *invisible to the encoders* (they sit motor-side, before the gears);
  plastic links that flex; screws that loosen mid-week; servos that brick
  if a position command fights an obstacle (we cooked a gripper motor;
  overcurrent protection + auto-reconnect became load-bearing
  infrastructure).
- These aren't just annoyances — they shaped the *science*: the backlash
  is WHY kinesthetic demos failed (§3) and why the operator's teleop
  compensations were the missing data. On a Franka this whole story might
  never have happened.
- Half our wall-clock went to infrastructure hardening: stream-rate
  guards, frozen-camera watchdogs, disk-full handling, index-corruption
  repair. On cheap hardware the data pipeline IS the experiment.
- Open question we'll test next: how much of this vanishes on a
  QDD-actuator arm (backlash-free, torque-controllable)? [TODO once
  hardware arrives]

## 7. Things that didn't work (kept on purpose)

- k-frame action shifts to simulate plant lag (helped, then superseded)
- gripper relabeling / "deepgrip" (no effect — labels weren't the problem)
- grasp-segment oversampling at 40% share (destroyed the approach)
- mid-trajectory episode surgery, three escalating variants (all
  destabilized; whole-episode reweighting was always better)
- inference-time crutches (binarization, commitment stretching) — helped
  only models trained on conflicted data; hurt good models
- feeding observation history at training rate during slowed-down
  deployment (definitively worse — the slowdown must be a UNIFORM time
  reparameterization; we broke the symmetry and the policy misread its
  own velocity)
- mirror augmentation: rejected by analysis (off-axis bin, chiral
  fiducials), validated by a $0 experiment (left side 0/5 → 3/5), then
  [TODO: offset-calibration resolution] — a full arc worth telling
  honestly

## 8. Where it stands / what's next

- [TODO: final grid table, before/after per generation]
- Next: UMI-style handheld collection (pre-registered prediction: it will
  share kinesthetic's arm-side flaw but not its gripper-side flaw — we
  wrote this down before building the rig, come back to check us)
- Repo + full experiment logs: [TODO: link] — every claim above has a
  dated entry.

---
*Working notes for the authors:*
- tone: lab-notebook honesty over triumph; each section leads with the
  mistake
- pull quotes: the EMA one-sentence-in-the-paper anecdote; "what matters
  is whether the demonstrator experienced the plant"
- figures wanted: val-loss ladder across recipe restorations; grid
  heatmaps per generation; the gripper mode-averaging trace; timeline
  strip of all runs
