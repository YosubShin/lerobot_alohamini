# Field notes: teaching a $300 arm to pick up a block (part 1 — training)

*DRAFT — target audience: people running robot-learning experiments on
cheap hardware (SO-101-class arms) who want to train a simple model
themselves rather than fine-tune a giant VLA. This is a journey post: the
dead ends are the content, not the final recipe. Part 1 covers getting the
policy from 0% to ~43%. Part 2 covers what "~43%" even means — how we
ended up measuring the evaluation itself, down to the physics of why two
identical grasp attempts disagree 27% of the time.*

---

## 0. What we set out to do

- One SO-101 follower + one SO-101 leader arm (teleop) + a Raspberry Pi
host + a $60 fisheye wrist camera. Task: "put the red block into the
bin." Policy: vanilla Diffusion Policy (LeRobot port),
wrist-camera-only, trained on a single consumer GPU.
- Where it ended up: from 0% grasp rate to ~43% (13/30 on a 30-placement eval across left/center/right regions, with deliberately hard placements — far, close, lying-down, initially-out-of-view — mixed in), with working failure-recovery behavior, in about a week and a half— and a measured, mechanism-attributed account of the ceiling.
- What this post is NOT: a recipe dump. The whole recipe fits in one
sentence — enable image augmentation, restore weight EMA, train the
encoder from scratch with GroupNorm, and weight the data mix right —
the transferable part is *how each piece was found*.



## 1. Pick a small model, not a large VLA — iteration velocity is the whole game

- Training runs are ~15–40 minutes on one GPU (RTX-6000-class; a 3090
works). The tally over ~10 days: **50+ training runs** (42 distinct
named experiments in the logs, plus variant families — k-shift ladders,
data-scaling probes), peaking at **eight or nine trainings in a single
day**; on the eval side, 10 structured grid sessions (~175 scored
placements), two DAgger sessions (~90 policy rollouts with human
takeover), and several hundred informal rollouts during debugging. The
full collect→train→eval loop turned over roughly 20 times, across 7
distinct data-collection rounds.
- The working rhythm: launch a training → walk to the rig → eval the
*previous* model while the next one bakes → notes → next hypothesis.
Train time ≈ eval time means the GPU and the human pipeline at 100%.
- Every finding in this post is downstream of that loop being fast. With a
multi-hour fine-tune, we'd have tested five hypotheses instead of fifty.



## 2. Don't blindly trust a port — audit the training recipe against the reference

- Our biggest single-day gains came from *restoring pieces of the original Diffusion Policy training recipe that the lerobot's DP port's defaults had drifted away from*:
  - image augmentation: shipped off by default
  (`--dataset.image_transforms.enable=false`); turning on the port's
  color-jitter set was worth ~8% val on our final data mix (~12% on an
  earlier, smaller dataset) — and, more visibly, it flattened the
  overfit curve: without augmentation the val loss bottoms out early and
  climbs straight back up (see figure). Funny detail: the paper's own
  augmentation choice — random crop (Appendix A.3) — was a wash when we
  added it on top. The win was having *any* appearance augmentation on,
  not the specific one the paper used.
  - **weight EMA: genuinely absent** — the reference implementation
  enables it in every config and evaluates the EMA weights; LeRobot
  removed it early on ([PR #134, merged May
  2024](https://github.com/huggingface/lerobot/pull/134)) and never
  restored it ([issue
  #4259](https://github.com/huggingface/lerobot/issues/4259); an
  opt-in EMA PR is open as
  [#4323](https://github.com/huggingface/lerobot/pull/4323)). For us:
  +9% val at every checkpoint, and the difference between 0 and our
  first repeatable successes.
  - encoder: the port *defaults* to an ImageNet-pretrained ResNet18 with
  BatchNorm (`use_group_norm=false`,
  `pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1"`); the
  paper's main-experiment encoder — "a standard ResNet-18 (without
  pretraining)" with "BatchNorm [replaced] with GroupNorm" (§3.2, used
  for all main benchmark tables) — is one flag away but not the default.
  Switching was −14% val at first — though §4 tells the fuller story:
  most of that gap turned out to be the fine-tuning learning rate, not
  pretraining itself.
![Validation loss across recipe restorations — same data, same val split. Left: each restoration lowers the curve AND flattens the overfit tail; the true port default (augmentation off) bottoms early and climbs straight back up. Right: the cumulative best-val ladder, −31% from port defaults to full recipe.](figures/recipe-ladder-val-loss.png)

- The kicker: the paper's ONLY mention of EMA is one throwaway sentence
explaining why BatchNorm was swapped for GroupNorm. That sentence
encoded two of our three biggest wins, coupled. Read the reference
implementation's configs, not just the paper.
- Lesson: ports translate *model definitions* faithfully, but training
harnesses get rewritten and defaults drift — EMA, augmentation,
encoder init are exactly what falls out. Diff the full training recipe
against the reference before trusting results.



## 3. Ramp data gradually: collect → train → eval → repeat

- We never collected more than ~100 episodes without training and
evaluating in between. Every batch's protocol was shaped by the previous
batch's failures:
  - gen 1 (76 eps): learned the task structure, failed the grasp →
  diagnosed WHY before collecting more
  - gen 2 (+76 teleop): modality ablation → found the real fix wasn't
  volume at all (next bullet)
  - gen 3 (+51 corner cases + 20 staged recoveries): aimed at the
  regions and behaviors that were failing
- Our eval was a simple region-scored grid: five arbitrary placements in
each of three regions (left / center / right), scored per region. Even
that coarse structure turned "collect more data" into a targeted request
— "the left region and lying-down blocks are failing; collect those" —
which is what shaped every batch after the first.
- **The modality finding — hand-guided demos could not teach grasping.**
Our first dataset was kinesthetic: you grab the arm and move it through
the task by hand, and the joint trajectory is recorded. It's fast and
smooth, and it taught reaching and transport — but grasp success stayed
near zero, and no amount of label surgery on that data fixed it. Action
time-shifts, gripper relabeling, and segment surgery were all dead ends.
The reason is the arm's gear backlash. When your hand moves the arm
directly, the slop in the gears never stands between your intention and
the motion — so the recorded trajectories contain no correction for it.
In teleoperation you drive a leader arm, and the follower has to move
under its own imperfect gears; you can see it lag and slip, and you
compensate without thinking. Those compensations are recorded, and they
turned out to be exactly the missing skill: with teleop data the grasp
rate jumped ~20x. If we had collected 300 kinesthetic episodes on day
one, we'd have baked in a flaw that volume cannot fix. One sentence:
*demonstrations must be performed through the same imperfect machine the
policy will have to control.*
- **From corrective data to DAgger** (gen 4+): with the base skill
working, the remaining failures were specific — missed grasps with no
retry, weak coverage cells. Our fixes escalated in fidelity. First,
upweighting the grasp segments of existing demos (helped as whole-episode
reweighting; destabilized as segment surgery). Next, staged recoveries:
place the block as if a grasp had just failed and demonstrate the retry —
which produced the first real recovery behavior but only covers failure
states we could imagine and stage. Finally,
[DAgger](https://arxiv.org/abs/1011.0686)-style takeover (specifically
the human-gated variant, à la
[HG-DAgger](https://arxiv.org/abs/1810.02890)): the leader arm
servo-tracks the follower during a live policy rollout, and pressing
SPACE flips control to the operator mid-failure — so corrections
are collected in exactly the failure states the *policy* gets itself
into, and successful rollouts get kept as free extra demonstrations. Two
rounds of that produced genuine mid-task recovery (drop the block, re-open,
re-approach, re-grasp) — imperfect, but a behavior that no amount of
ordinary demonstration data had produced. Two lessons repeated at
every rung. First, dosage: corrective episodes are few, so they need
upweighting to matter — but overweight them and the corrective behavior
leaks into normal operation (our 4x-weighted interventions taught the
policy to hover cautiously over every block, not just failed ones).
Start at 1x and raise only if the behavior doesn't appear; we overdosed
three times before adopting that rule. Second, balance: if every
correction in a batch demonstrates the same response — ours all said
"stop twisting the wrist" — the model doesn't learn *when* to do it, it
just does less of it everywhere, including where twisting was required.
A correction batch has to show both sides of the condition.

## 4. The encoder ablation — or, how hard you have to work to trust even a validation-loss claim

- Swapping to the paper's main-recipe encoder (from-scratch + GroupNorm,
per §3.2 "Visual Encoder" — the configuration behind all their main
benchmark tables) was −14% val, and the models of that era felt visibly
better on the robot. Hold that "felt" — Part 2 is about what it was
worth. This section is the story of what it took to trust even the
validation-loss half of the claim — the half you can measure from logged
data alone, without ever running the robot.
- We almost wrote "GroupNorm did it" — the swap necessarily changes TWO
things (you can't put GroupNorm into pretrained BN weights). Before
letting the attribution stand, we made ourselves run the disambiguation
(BatchNorm + from-scratch, perfectly legal), and it decomposed the gain:
**~12% was dropping ImageNet, ~3% (≈noise) was the normalization**.
- Here's where it gets interesting. Our winning configuration agrees with
the paper's *main* recipe but seemed to contradict the paper's own
ablation — Chi et al. §5.4 / Table 5 report that *fine-tuning* a
pretrained encoder beats from-scratch, with one crucial detail: the
backbone gets a **10× lower learning rate** than the rest of the
network. Our original comparison had fine-tuned at full LR. So we ran
the paper's exact prescription (backbone at 1e-5, everything else at
1e-4, same data and val split): the gap collapsed from **~14% to ~5%**.
Then we asked whether even 5% was real, and reran the from-scratch
recipe with a different seed: seed-to-seed spread was ~0.5%, so the 5%
edge is about ten times the noise floor — small but genuine. The final
statement: most of the "pretraining hurts" effect was never about
pretraining — it was full-LR fine-tuning destroying the pretrained
features in the first few hundred steps — but a real ~5% residual
still favors from-scratch on this task (0.0110–0.0111 across seeds vs
0.0116). The dramatic version of the claim died on replication of the
paper's actual protocol; the modest version survived a seed control.
- The full matrix, for the record — one dataset, one val split, identical
recipe, seed noise measured at ~0.5%:

  | encoder | norm | fine-tune LR | best val (EMA) |
  |---|---|---|---|
  | from scratch | GroupNorm | uniform | 0.0110–0.0111 (2 seeds) |
  | from scratch | BatchNorm | uniform | 0.0112 |
  | ImageNet-pretrained | BatchNorm | backbone 10× lower (§5.4) | 0.0116 |
  | ImageNet-pretrained | BatchNorm | uniform (naive) | 0.0126 |

  Decomposed with normalization held fixed: pretraining costs ~4% even
  under the paper's LR prescription; GroupNorm is worth ~1.4% over
  BatchNorm from scratch; naive full-LR fine-tuning costs a further ~9%.
- Where this section lands, stated at the right level: the val ladder is
real, replicated, and decisively ordered against a measured seed floor —
and it is a result about numbers computed from recorded data, not about
behavior on the robot. We kept from-scratch + GroupNorm as the
recipe because it is val-best, simplest, cheapest (no pretrained weights
to download or protect with a special learning rate), and avoids a
BatchNorm/EMA interaction — not because we can prove it grips blocks
better. Whether any of these encoder differences survive contact with
the physical world is precisely the question Part 2 was built to answer.
  A caveat that became its own investigation: when we tried to confirm
  this val ladder *behaviorally*, the ranking dissolved — full-task grids
  contradicted each other, a same-model repeat spanned 8/30 to 11/30, and
  a purpose-built 276-trial grasp benchmark found all six encoder
  configurations statistically indistinguishable at the task's bottleneck
  (while the model with the *worst* validation loss co-led it). That story — how we ended up
  measuring our evaluation instead of our models — is Part 2. The val ladder above is the defensible result; the robot
  told us only that the naive fine-tune's degradation is real and that
  nothing else separates cleanly at this eval size.

- Two lessons, then, not one: don't trust the port's defaults — and
before declaring that a paper's ablation "doesn't transfer to your
setup," make sure you replicated its *exact* protocol. A single
sentence of the ablation ("10× smaller learning rate") carried the
whole effect.
- The literature is consistent with where we landed: Hansen et al.
(ICML 2023, ["On Pre-Training for Visuo-Motor Control: Revisiting a
Learning-from-Scratch Baseline"](https://arxiv.org/abs/2212.05749))
found a learning-from-scratch baseline with augmentation competitive
with frozen large-scale pretrained representations (PVR/MVP/R3M)
across sim and real robot tasks, attributing the remaining gap to
pre-training/deployment domain mismatch — and our fisheye-wrist
single-scene setup is about as far from ImageNet's distribution as
tabletop robotics gets. From-scratch competitive, pretraining not
harmful if fine-tuned carefully: both our result and theirs.



## 5. How I actually worked with an AI assistant on this

Everyone is figuring out their own AI workflow right now, so here is
what mine looked like after two weeks of daily use on a real research
project. Most of the code, scripts, and log analysis in this post was
written by the AI. Most of the decisions were not.

What the AI was good at:

- Converting a high-level description of an algorithm into working code.
  "Drive the leader arm to track the follower, and pressing SPACE hands
  control to me" became a working DAgger rig the same evening.
- Chaining long-running operational tasks. "Build a grid over parameter
  X, schedule the trainings on the remote box, score the checkpoints,
  and stage the winners so I can eval" — it would run the whole chain,
  including babysitting the GPU queue overnight. Evals happen in the
  real world, so those stayed mine.
- Build and environment issues. Driver mismatches, disk-full crashes
  mid-checkpoint, a servo throwing voltage errors — it debugged these
  faster than I would have, and hardened the scripts afterward.
- Knowing the literature. When I spelled out an idea, it could usually
  tell me who had done something similar and under what name (HG-DAgger,
  Sirius, RT-C-style inpainting). I double-checked the citations in case
  they were fake. They were mostly real.

What the AI was bad at:

- Watching rollout videos. It cannot spot a subtly wrong robot motion.
  Every one of the breakthroughs in this post started with me watching
  rollouts and saying something like "it hesitates — goes down, comes
  back up, repeats" or "the left approach shakes, the right doesn't."
  The AI could turn those one-liners into trace analysis, a mechanism,
  and a next experiment by the following training run — but the noticing
  was never its.
- Remembering what matters. Over 50 training runs, the AI would quietly
  lose decisions we had already settled — and settled for a reason.
  Concrete example: we ran inference at 15 Hz because at 30 Hz the
  success rate dropped and you could see the arm overshooting in the
  sensitive grasp region; at some point the AI reverted to an arbitrary
  20 Hz with no justification. Minor, but things like this happened a
  lot, and each one silently invalidates a comparison if you don't
  notice. Humans don't forget the details they personally fought for.

Practical tips:

- Don't believe the AI's hypotheses. It produces tidy, confident,
  well-argued conclusions at a rate that will lull you. Twice I
  spot-checked four videos behind a data-quality audit it had built and
  falsified its classifier both times. Push back as an independent
  thinker; when you're suspicious, check the raw thing yourself.
- Own the experiments. You need the bird's-eye view (what question is
  this week actually answering) and the low-level details (which flags,
  which dataset version, which checkpoint) in your own head. The AI can
  hold the middle — the execution — but if you give up either end, you
  are no longer doing the science; you're watching someone else's.



## 6. Things that didn't work (kept on purpose)

- k-frame action time-shifts, to fake the command-to-motion lag missing
from kinesthetic data (hand-guided recording stores action = observed
state, so the data has no lead between command and response; shifting
actions k frames earlier simulated one). Helped, then superseded — teleop
data contains the real thing.
- gripper command hacks, on both sides of training. Background: the
SO-101 has no force control — grip strength comes from commanding a
position *past* the block, so the stalled servo keeps pushing. Teleop
data contains that (the operator's trigger travels ~20 units beyond
block width); kinesthetic data cannot, because it records the *measured*
finger position, which physically stops at the block — zero squeeze
margin, every episode. So we tried rewriting the kinesthetic gripper
targets to command deeper closure. No effect: the models still failed
the grasp, because the deeper problem was the missing corrective motion
throughout the trajectory (section 3), not the closure depth label. We
also tried deploy-time overrides — snapping every gripper command to
fully-open/fully-closed, or biasing it a constant amount toward closed.
These helped our early models, which had been trained on a mix of
kinesthetic and teleop data and, fed those contradictory gripper
targets, produced averaged, indecisive closures that slipped off the
block. But once the data was fixed, the same overrides hurt: the newer
models had learned to close carefully and progressively, and the
overrides destroyed exactly those deliberate intermediate commands. A
deploy-time hack that helps is a symptom of a training-data problem —
and it turns into damage the moment the data problem is fixed.
- carving up episodes: oversampling grasp segments (up to a 40% share —
destroyed the approach behavior) and three escalating mid-trajectory
surgery variants on the early kinesthetic+teleop mixes, all of which
destabilized training. Reweighting whole teleop episodes did the same
job safely, every time.
- mixing clocks during slowed-down deployment. We run the policy at half
the training frame rate (a deliberate 2x slowdown that buys reaction
time), and the model conditions on a 2-frame observation history. The
question was how far apart those two frames should be. What we ended up
doing: one deployment tick apart (1/15 s) — at half speed, the motion
between frames then looks exactly like the motion between consecutive
training frames, so every clock slows down together. What we tried
instead: spacing the history at the training rate (1/30 s), on the
logic that "the model expects frames 1/30 s apart" — definitively
worse, because at half speed those frames show half the motion the
model saw in training, so the policy misread its own velocity. Lesson:
if you slow a policy down, slow *everything* down by the same factor.
- mirror augmentation: doubling the data by flipping frames horizontally
and negating the joints that move the arm laterally (base pan, wrist
roll). It helped while left-side data was scarce (left 0/5 → 3/5), hurt
once real left-side data existed, and the diagnosis turned out to be
optics: the fisheye's optical center is not the image center, so a naive
flip shifts the whole scene laterally. Dropped in favor of just
collecting more real episodes.

## 7. Fine manipulation on cheap hardware is legitimately hard — and where that points next

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
- Where this points next for us: a QDD-actuator arm (quasi-direct-drive —
hopefully far less backlash, torque-commandable). We actually started
this whole exploration intending to make UMI-style handheld data
collection work with the SO-101, and the project talked us out of it
three times over:
  1. **Kinematics**: UMI captures arbitrary hand orientations; the
     5-DOF SO-101 has no wrist yaw, so most freely-collected
     trajectories would simply be infeasible and thrown away (or
     distorted by projection exactly at the contact-critical moments).
  2. **The kinesthetic lesson generalizes**: hand-guided demos failed
     because the demonstrator never experienced the arm's backlash and
     so never demonstrated the compensations (§3). A handheld UMI
     gripper has the same structural flaw on the arm side — the
     trajectories are collected without the robot's gears in the loop — so we
     expect the same failure mode, and would rather test that prediction
     on hardware where backlash isn't the dominant error to compensate
     for in the first place.
  3. **Position control burns motors**: a position-commanded servo that
     fights an obstacle just keeps pushing — we cooked a gripper motor
     exactly this way, and hard contact during grasps stayed a
     failure mode all project (the policy can't feel how hard it's
     pressing). A compliant, torque-commanded arm is inherently more
     robust to both, and makes contact information available rather than
     invisible.



- Where it stands: ~43% overall success on a 30-placement eval spanning
easy to deliberately-hard cells, with recovery behavior that did not
exist in any early model. How much to trust that number — and what we
found when we tried — is [Part 2](so101-part2-evaluation-DRAFT.md). [TODO: replace with published URL]
- Repo: [github.com/YosubShin/lerobot_alohamini](https://github.com/YosubShin/lerobot_alohamini); full experiment logs:
[docs/experiments](https://github.com/YosubShin/lerobot_alohamini/tree/main/docs/experiments) — every claim above has a dated entry.

---

*Working notes for the authors:*

- tone: lab-notebook honesty over triumph; each section leads with the
mistake
- pull quotes: the EMA one-sentence-in-the-paper anecdote;
"demonstrations must be performed through the same imperfect machine the
policy will have to control"
- figures: recipe-ladder val-loss DONE (figures/recipe-ladder-val-loss.png);
still wanted: grid heatmaps per generation, gripper mode-averaging trace
- video: policy success rollout DONE (figures/success-rollout-v2f-ep8.mp4 —
wrist-fisheye 720p, 30 s, v2f policy, approach → grasp → drop into bin);
embed near §0 or §7

