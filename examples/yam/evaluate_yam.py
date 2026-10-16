#!/usr/bin/env python3
"""Policy rollout on the YAM for the pilot arms (B / Cf / Cf_noeef / A).

Reuses the SO-101 harness's AsyncChunkPlanner (RTC inpainting, zero-stall
chunk streaming). YAM-specific parts:
  - cameras are captured HERE (the host is camera-free): forward B0587 at
    half-res for B/Cf/Cf_noeef, wrist fisheye for A; ritual auto-applied
  - observations are FK-derived (no live fiducial decode): state and the
    EEF-crop center come from joints + hand-eye (calib/handeye.json),
    matching training's vision-derived quantities to ~5mm
  - pose-action arms (B/A) go through warm-started mink IK (i2rt model,
    loosened convergence: vision-noise targets sit mm off-manifold)
  - joint-space slew limiter for all arms

Deploy config defaults follow the SO-101 champion: fps 15, DDIM-10,
n_action_steps 15, interp_substeps 2.

Usage:
  python evaluate_yam.py --arm Cf                      # joint-action control
  python evaluate_yam.py --arm B                       # pose-action + IK
  (host first, in 'policy'-capable mode: yam_zmq_host.py ...)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alohamini"))
sys.path.insert(0, str(Path.home() / "co/pilot/scripts"))
sys.path.insert(0, str(Path.home() / "co/i2rt"))

from evaluate_so101 import (AsyncChunkPlanner, enter_pressed,  # noqa: E402
                            median_first_frame_state)
import torch  # noqa: E402  (after evaluate_so101 pins the GPU)
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies import (get_policy_class,  # noqa: E402
                              make_pre_post_processors)
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.utils import log_say  # noqa: E402
from yam_zmq_client import YamZmqClient, YamZmqClientConfig  # noqa: E402

PILOT = Path.home() / "co/pilot"
FORWARD_DEV = "/dev/v4l/by-path/pci-0000:44:00.0-usb-0:1:1.0-video-index0"
FISH_DEV = "/dev/v4l/by-path/pci-0000:06:00.0-usb-0:1:1.0-video-index0"
HEAD_DEV = "/dev/v4l/by-path/pci-0000:10:00.3-usb-0:2.1:1.0-video-index0"   # SO-101 arm cam (b0587b); tracker must run with --release_cam
TWIN_DEV = "/dev/v4l/by-path/pci-0000:10:00.3-usb-0:2.1:1.0-video-index0"
# Per arm: obs camera + label semantics, then one (dataset repo, checkpoint)
# per architecture. --model picks the architecture; ACT is the default recipe
# (round-7 val winners, 197-ep corpus). DP entries are the round-6 winners
# until the round-7 DP block lands. pose_actions=True -> runtime IK.
#
# val-min steps (round-7 ACT / 12 shared val eps): Cw .2312@48k, Aj .2469@44k,
# Cwf .2479@48k, Cf .2577@48k, Bjt .2937@38k, Bj .2948@48k.
ARMS = {
    "Cw": {"cam": "fish",
           "act": ("pilot_all7_Cw", "all7_Cw_act50/checkpoints/048000"),
           "dp": ("pilot_all6_Cw", "all6_Cw_s1000_dp/checkpoints/004000_ema")},
    "Cf": {"cam": "forward",
           "act": ("pilot_all7_Cf", "all7_Cf_act50/checkpoints/048000"),
           "dp": ("pilot_all6_Cf", "all6_Cf_s1000_dp/checkpoints/003500_ema")},
    "Aj": {"cam": "fish",
           "act": ("pilot_all7_Aj", "all7_Aj_act50/checkpoints/044000"),
           "dp": ("pilot_all6_Aj", "all6_Aj_s1000_dp/checkpoints/004500_ema")},
    "Bj": {"cam": "forward",
           "act": ("pilot_all7_Bj", "all7_Bj_act50/checkpoints/048000"),
           "dp": ("pilot_all6_Bj", "all6_Bj_s1000_dp/checkpoints/003000_ema")},
    "Bjt": {"cam": "forward",
            "act": ("pilot_all7_Bjt", "all7_Bjt_act50/checkpoints/038000"),
            "dp": ("pilot_all6_Bjt",
                   "all6_Bjt_s1000_dp/checkpoints/004500_ema")},
    # standard-rig reference (wrist + forward full-frame): 7.2% WORSE than
    # wrist alone at 197 eps — kept for the paper, not a deploy candidate
    "Cwf": {"cam": "fish+forward",
            "act": ("pilot_all7_Cwf", "all7_Cwf_act50/checkpoints/048000"),
            "dp": ("pilot_all6_Cwf",
                   "all6_Cwf_s1000_dp/checkpoints/004500_ema")},
    # (round-4 glove/mix arms, 69-episode corpus, archived to the NAS 2026-10-09:
    #  /mnt/nas/nvme_archive/pilot/train_runs/*_r4 — superseded by round 5 / _s)
    # round 5 (2026-10-08 evening): FULL glove corpus, 171 eps / 20.5k frames.
    # Deploy at the ROBOT-val minimum (the deployment distribution):
    # glove_Bj .867@32k, glove_Aj .847@34k, mix_Bj .273@46k (robot-only .295),
    # mix_Aj .237@40k (robot-only .247).
    "glove_Bj_r5": {"cam": "forward",
                    "act": ("pilot_glove_Bj", "glove_Bj_act50/checkpoints/032000")},
    "glove_Aj_r5": {"cam": "fish",
                    "act": ("pilot_glove_Aj", "glove_Aj_act50/checkpoints/034000")},
    "mix_Bj_r5": {"cam": "forward",
                  "act": ("pilot_mix_Bj", "mix_Bj_act50/checkpoints/046000")},
    "mix_Aj_r5": {"cam": "fish",
                  "act": ("pilot_mix_Aj", "mix_Aj_act50/checkpoints/040000")},
    # "_s" (2026-10-09): the SAME 171 glove eps rebuilt at teleop pace (--slow 2.5)
    # with the gripper mapped measured-width -> commanded (glove plateau .335 ->
    # teleop grasp .246) and median-smoothed. Robot val: glove_Bj .867 -> .593,
    # glove_Aj .847 -> .584; mix_Bj .260 (robot-only .295), mix_Aj .236.
    # Spot checks of the native-pace glove policies ran 2.6x too fast and
    # dropped the block mid-air (no squeeze) — use these for rollouts.
    "glove_Bj_s": {"cam": "forward",
                   "act": ("pilot_glove_Bj_s", "glove_Bj_s_act50/checkpoints/036000")},
    "glove_Aj_s": {"cam": "fish",
                   "act": ("pilot_glove_Aj_s", "glove_Aj_s_act50/checkpoints/044000")},
    "mix_Bj_s": {"cam": "forward",
                 "act": ("pilot_mix_Bj_s", "mix_Bj_s_act50/checkpoints/046000")},
    "mix_Aj_s": {"cam": "fish",
                 "act": ("pilot_mix_Aj_s", "mix_Aj_s_act50/checkpoints/040000")},
    # teleop-dilution at Aj's 185-episode budget, paced glove corpus
    # (calib/train_eps_dilute{92,46}_Aj_s.txt). Robot val .267 / .271 vs Aj .247.
    "Aj_glove50_s": {"cam": "fish",   # teleop 92 + glove 93
                     "act": ("pilot_mix_Aj_s", "dilute92_Aj_s_act50/checkpoints/044000")},
    "Aj_glove75_s": {"cam": "fish",   # teleop 46 + glove 139
                     "act": ("pilot_mix_Aj_s", "dilute46_Aj_s_act50/checkpoints/050000")},
    # --- shelved probes: DP only, round-6 corpus ---
    "Cft": {"cam": "forward+side",
            "dp": ("pilot_all6_Cft",
                   "all6_Cft_s1000_dp/checkpoints/004500_ema")},
    # slow2x (30Hz-sampled / fps-15-declared): deploy at --fps 15 for native
    # 2x slow, --fps 20 for 1.5x. Matched slow training bought nothing.
    "Cw2": {"cam": "fish",
            "dp": ("pilot_s2_Cw", "s2_Cw_s1000_dp/checkpoints/007000_ema")},
    "Cf2": {"cam": "forward",
            "dp": ("pilot_s2_Cf", "s2_Cf_s1000_dp/checkpoints/005000_ema")},
    # --- head-cam round (2026-10-15): SO-101 head cam parked/tracking via so101_cam_track.py,
    # key observation.images.b0587b, 640x360; "crop" = centred fraction before resize.
    # Tracker must be running (tracking for the head arms; --static for the parked ablation).
    # All teleop arms: the SAME 96 matched teleop eps (head12/13 minus 6 shared val eps);
    # mixes add the same 104 glove eps (glove4+5 head view). Deploy at the robot-val minimum.
    # robot val: head_Hc .365@32k, head_H .355@42k, head_A .335@40k, head_B .377@30k;
    #            mix_hHc .278@50k, mix_hH .282@48k, mix_hA .276@48k, mix_hB .301@48k,
    #            mix_hHc_half .291@50k; glove_Hc (glove-only) .527@48k.
    "head_Hc": {"cam": "head", "crop": 0.8,
                "act": ("pilot_head_Hcj", "head_Hcj_act50/checkpoints/032000")},
    "head_H": {"cam": "head",
               "act": ("pilot_head_Hj", "head_Hj_act50/checkpoints/042000")},
    "head_A": {"cam": "fish",      # wrist control, same 96 eps
               "act": ("pilot_head_Aj", "head_Aj_act50/checkpoints/040000")},
    "head_B": {"cam": "forward",   # forward control, same 96 eps
               "act": ("pilot_head_Bj", "head_Bj_act50/checkpoints/030000")},
    "mix_hHc": {"cam": "head", "crop": 0.8,
                "act": ("pilot_mix_hHc", "mix_hHc_act50/checkpoints/050000")},
    "mix_hH": {"cam": "head",
               "act": ("pilot_mix_hH", "mix_hH_act50/checkpoints/048000")},
    "mix_hA": {"cam": "fish",
               "act": ("pilot_mix_hA", "mix_hA_act50/checkpoints/048000")},
    "mix_hB": {"cam": "forward",
               "act": ("pilot_mix_hB", "mix_hB_act50/checkpoints/048000")},
    "mix_hHc_half": {"cam": "head", "crop": 0.8,
                     "act": ("pilot_mix_hHc", "mix_hHc_half_act50/checkpoints/050000")},
    "glove_Hc": {"cam": "head", "crop": 0.8,
                 "act": ("pilot_glove_Hcj_s45", "glove_Hcj_s45_act50/checkpoints/048000")},
    "glove_H": {"cam": "head",    # NAS run linked into train_runs; glove val .286@40k, no robot val
                "act": ("pilot_glove_Hj_s45", "Hj_s45_act50/checkpoints/040000")},
    # --- head-cam round 2 (2026-10-15 evening): head-cam-ONLY label chain. HH = dodeca + gripper decoded from
    # the head cam, placed via the passive tiles (base<->tile link calib/base_world_20261015.json); Hgt = encoders.
    # Teleop arms: ALL 172 matched teleop eps (head12/13/14 minus the same 6 val eps); "96" = round-1's 96-ep list.
    # Mixes add 186 glove eps (glove5-8, head chain with fixed-cam fallback on frames without a tile, 9-36%/session).
    # Glove-only arm starts from its own first-frame pose ("skeleton"). Deploy at the robot-val minimum.
    # robot val: head_HHc .287@44k, head_Hcgt .305@26k, head_HHc96 .331@42k, head_Hcgt96 .368@50k, head_HH .282@50k;
    #            mix_HHc .269@50k, mix_HHc_half .252@46k, mix_HH .251@44k; glove_HHc .503@44k (glove val .369@38k).
    "head_HHc": {"cam": "head", "crop": 0.8,
                 "act": ("pilot_head_HHc2", "head_HHc_act50/checkpoints/044000")},
    "head_Hcgt": {"cam": "head", "crop": 0.8,   # encoder labels, same 172 eps
                  "act": ("pilot_head_Hcgt", "head_Hcgt_act50/checkpoints/026000")},
    "head_HHc96": {"cam": "head", "crop": 0.8,  # round-1 96-ep budget
                   "act": ("pilot_head_HHc2", "head_HHc96_act50/checkpoints/042000")},
    "head_Hcgt96": {"cam": "head", "crop": 0.8,
                    "act": ("pilot_head_Hcgt", "head_Hcgt96_act50/checkpoints/050000")},
    "head_HH": {"cam": "head",
                "act": ("pilot_head_HH2", "head_HH_act50/checkpoints/050000")},
    "mix_HHc": {"cam": "head", "crop": 0.8,
                "act": ("pilot_mix_HHc2", "mix_HHc_act50/checkpoints/050000")},
    "mix_HHc_half": {"cam": "head", "crop": 0.8,
                     "act": ("pilot_mix_HHc2", "mix_HHc_half_act50/checkpoints/046000")},
    "mix_HH": {"cam": "head",
               "act": ("pilot_mix_HH2", "mix_HH_act50/checkpoints/044000")},
    "glove_HHc": {"cam": "head", "crop": 0.8, "skeleton": "pilot_glove_HHc2",
                  "act": ("pilot_glove_HHc2", "glove_HHc_act50/checkpoints/044000")},
    # strict head-only glove set (pilot_glove_HHsc): NO forward-cam fallback, chained single-tile solves,
    # blind stretches > 0.3 s cut out (198 eps -> 230 segments, 89% of frames kept), same 172 teleop eps in the mix.
    # robot val: mix_HHsc .260@36k (glove val .274), glove_HHsc .516@34k (glove val .317).
    "mix_HHsc": {"cam": "head", "crop": 0.8,
                 "act": ("pilot_mix_HHsc", "mix_HHsc_act50/checkpoints/036000")},
    "glove_HHsc": {"cam": "head", "crop": 0.8, "skeleton": "pilot_glove_HHsc",
                   "act": ("pilot_glove_HHsc", "glove_HHsc_act50/checkpoints/034000")},
    # same, with the tile-solve reprojection gate at 8 px (200 segments, 194 train): mix .262@38k (glove .270), glove .503@36k
    # wrist-fisheye control at the full budget: encoder labels, 174 of the 178 teleop eps (wrist recording gaps),
    # 168 of the 172 matched train eps, same 6 val eps. robot val .300@46k (vs head_Hcgt .305: same labels, same eps).
    "wrist_gt": {"cam": "fish",
                 "act": ("pilot_head_Agt", "wrist_gt_act50/checkpoints/046000")},
    "mix_HHs2c": {"cam": "head", "crop": 0.8,
                  "act": ("pilot_mix_HHs2c", "mix_HHs2c_act50/checkpoints/038000")},
    "glove_HHs2c": {"cam": "head", "crop": 0.8, "skeleton": "pilot_glove_HHs2c",
                    "act": ("pilot_glove_HHs2c", "glove_HHs2c_act50/checkpoints/036000")},
    # --- final head-chain glove set (2026-10-16 02:40): tile layout v3 (calib/workspace_tiles_pilot_v3.json, refit from
    # head14 single-tile poses; 3-tile solve rms 3.1 -> 0.7 px), base link calib/base_world_v3_20261016.json,
    # single-tile chaining, 8 px gate, blind stretches > 0.3 s cut (198 eps -> 200 segments, 194 train; 0.7-7.4%
    # of frames dropped per session). Teleop half unchanged (v2 layout, multi-tile averages absorb the offset).
    # robot val: mix_HHs3c .266@46k (glove val .287), glove_HHs3c .520@36k (glove val .330).
    "mix_HHs3c": {"cam": "head", "crop": 0.8,
                  "act": ("pilot_mix_HHs3c", "mix_HHs3c_act50/checkpoints/046000")},
    # same checkpoint with the head cam PARKED (tracker mode "static"): value of the head-cam tracking itself
    "mix_HHs3c_static": {"cam": "head", "crop": 0.8, "tracker": "static",
                         "act": ("pilot_mix_HHs3c", "mix_HHs3c_act50/checkpoints/046000")},
    "glove_HHs3c": {"cam": "head", "crop": 0.8, "skeleton": "pilot_glove_HHs3c",
                    "act": ("pilot_glove_HHs3c", "glove_HHs3c_act50/checkpoints/036000")},
    # wrist-fisheye co-training, conventional UMI labelling: glove poses from the FIXED forward cam only (frames the
    # body occludes are cut: 8-22% per session, 198 eps -> 259 segments, 253 train), gripper width from the head cam,
    # observation = glove wrist fisheye; robot half = pilot_head_Agt (168 train, encoder labels).
    # robot val: mix_FA .257@50k (glove val .183) vs wrist_gt .300.
    "mix_FA": {"cam": "fish",
               "act": ("pilot_mix_FA", "mix_FA_act50/checkpoints/050000")},
    # teleop-budget ("exchange rate") arms: nested 96 -> 48 -> 24 teleop subsets (seed 7 from the 96 list), each
    # with and without ALL 194 glove segments. robot val: head_HHc96 .331, head_HHc48 .427, head_HHc24 .574;
    # mix_HHs3c96 .277@36k, mix_HHs3c48 .298@32k, mix_HHs3c24 .341@42k (172 teleop: head_HHc .287, mix_HHs3c .266).
    "head_HHc48": {"cam": "head", "crop": 0.8,
                   "act": ("pilot_head_HHc2", "head_HHc48_act50/checkpoints/026000")},
    "head_HHc24": {"cam": "head", "crop": 0.8,
                   "act": ("pilot_head_HHc2", "head_HHc24_act50/checkpoints/034000")},
    "mix_HHs3c96": {"cam": "head", "crop": 0.8,
                    "act": ("pilot_mix_HHs3c", "mix_HHs3c96_act50/checkpoints/036000")},
    "mix_HHs3c48": {"cam": "head", "crop": 0.8,
                    "act": ("pilot_mix_HHs3c", "mix_HHs3c48_act50/checkpoints/032000")},
    "mix_HHs3c24": {"cam": "head", "crop": 0.8,
                    "act": ("pilot_mix_HHs3c", "mix_HHs3c24_act50/checkpoints/042000")},
    # --- fixed FORWARD cam as the observation, HEAD-cam labels ("HB", 2026-10-16 09:17): same label chains and the
    # same episodes/segments as head_HHc / mix_HHs3c, only the observation stream swapped (key observation.images.b0587,
    # no crop). Control for "does the policy need the egocentric view" with the labels held fixed.
    # robot val (comparable to the head arms: same labels, same val eps): head_HB .223@40k, mix_HB .232@48k (glove val .205).
    "head_HB": {"cam": "forward",
                "act": ("pilot_head_HB", "head_HB_act50/checkpoints/040000")},
    "mix_HB": {"cam": "forward",
               "act": ("pilot_mix_HB", "mix_HB_act50/checkpoints/048000")},
    # --- fixed FORWARD cam as observation AND label source ("FB"): what a fixed camera gives as a data-collection
    # device. Teleop pilot_head_Bgt (forward stream, encoder labels, 172/6 as Hgt); glove pilot_glove_FB = the FA
    # recipe (fixed-cam poses, body-occluded stretches cut: 8-22%/session) with the forward view (255 train segments).
    # Caveat: the forward cam sat LOW (the head-cam rig took the high vantage), so its occlusion rate is an upper bound.
    # robot val (encoder-label chain, comparable to head_Hcgt .305 / wrist_gt .300 / mix_FA .257): head_Bgt .318@50k, mix_FB .256@44k (glove val .247).
    "head_Bgt": {"cam": "forward",
                 "act": ("pilot_head_Bgt", "head_Bgt_act50/checkpoints/050000")},
    "mix_FB": {"cam": "forward",
               "act": ("pilot_mix_FB", "mix_FB_act50/checkpoints/044000")},
    # A/B (round-1 pose-action arms) RETIRED 2026-09-16: stale 22-ep
    # checkpoints, pre-wedge visuals; the pose-vs-joint ablation, if ever
    # wanted, should train on pilot_all6_B (built every round).
}
JOINTS = [f"joint{i}.pos" for i in range(1, 7)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", choices=list(ARMS), required=True)
    p.add_argument("--model", choices=["act", "dp"], default="act",
                   type=str.lower,
                   help="architecture to deploy (default act = the round-7 "
                        "recipe); picks that arm's val-winning checkpoint "
                        "and its matching dataset. --checkpoint overrides.")
    p.add_argument("--spot", action="store_true",
                   help="lightweight spot-check loop: endless repeated "
                        "rollouts, NO logging, auto-home after every "
                        "episode and on Ctrl-C (never drops mid-air)")
    p.add_argument("--checkpoint", help="override checkpoint dir "
                                        "(…/pretrained_model parent)")
    p.add_argument("--skeleton_root", default=None,
                   help="dataset whose median first-frame state is the reset "
                        "pose (default: the arm's \"skeleton\" repo if set, "
                        "else yam_pilot1)")
    p.add_argument("--num_episodes", type=int, default=5)
    p.add_argument("--episode_time", type=float, default=60.0)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--scheduler", default="ddim")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--n_action_steps", type=int, default=15)
    p.add_argument("--chunk_trigger", type=int, default=6)
    p.add_argument("--no_rtc", action="store_true")
    p.add_argument("--interp_substeps", type=int, default=2)
    p.add_argument("--action_ema", type=float, default=0.0)
    p.add_argument("--act_ensemble", type=float, default=None,
                   help="ACT only: temporal-ensemble coeff, forces "
                        "n_action_steps=1 (replan every tick). Default 0.01 "
                        "under --model act (removes the chunk-boundary jump); "
                        "pass 0 to disable.")
    p.add_argument("--max_delta_per_tick", type=float, default=None,
                   help="rad per tick per joint (0=off). Default holds a "
                        "CONSTANT 0.9 rad/s cap by scaling with --fps "
                        "(0.06 @ 15 fps, 0.09 @ 10 fps) — the 0.06 figure "
                        "was tuned at 15 fps and over-throttles slower ticks.")
    p.add_argument("--gripper_max_delta", type=float, default=0.2)
    p.add_argument("--gripper_close_bias", type=float, default=0.0,
                   help="deepen grasp-intent gripper commands (cmd<0.5) by "
                        "this amount, floor 0.2 — counters diffusion "
                        "mean-regression (policy carries ~0.03 shallower "
                        "than demos and can relax mid-transport; measured "
                        "2026-09-16, drops at ~0.33 vs demo carry 0.28)")
    p.add_argument("--home_seconds", type=float, default=3.0)
    p.add_argument("--no_reset", action="store_true")
    p.add_argument("--tracker_ctrl", default="tcp://127.0.0.1:5612", help="so101_cam_track.py control socket: the arm's \"tracker\" field (track|static) is sent once at start; 'none' disables")
    p.add_argument("--remote_ip", default="127.0.0.1")
    p.add_argument("--zmq_cmd_port", type=int, default=5601)
    p.add_argument("--zmq_obs_port", type=int, default=5602)
    p.add_argument("--task_description", default="pick red block into bin")
    p.add_argument("--log_dir", default=str(PILOT / "rollout_logs"))
    p.add_argument("--tag", default="",
                   help="free text folded into the log filename, e.g. the "
                        "eval-grid cell (A1..C3) — block positions are not "
                        "observed, so this is how a trial records them")
    return p.parse_args()


class CamReader:
    """Latest-frame camera capture (drop, never queue)."""

    def __init__(self, dev, w, h, fps):
        import threading
        self.cap = cv2.VideoCapture(dev)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {dev}")
        for _ in range(8):
            self.cap.read()
        from uvc_ritual import apply as apply_ritual
        apply_ritual(dev)
        self.frame = None
        self.dead = False

        def _run():
            while not self.dead:
                ok, f = self.cap.read()
                if ok:
                    self.frame = f
        threading.Thread(target=_run, daemon=True).start()
        while self.frame is None:
            time.sleep(0.02)

    def stop(self):
        self.dead = True


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = ARMS[args.arm]
    if args.model not in spec:
        have = ", ".join(m for m in ("act", "dp") if m in spec)
        raise SystemExit(f"arm {args.arm} has no {args.model.upper()} "
                         f"checkpoint (available: {have})")
    repo, ck_rel = spec[args.model]
    cam = spec["cam"]
    pose_state_f = spec.get("pose_state", False)
    pose_actions_f = spec.get("pose_actions", False)
    head_crop = float(spec.get("crop", 1.0))   # head arms: centred crop fraction
    ck = Path(args.checkpoint) if args.checkpoint else \
        PILOT / "train_runs" / ck_rel
    # knobs whose right value depends on the recipe / tick rate
    if args.act_ensemble is None:
        args.act_ensemble = 0.01 if args.model == "act" else 0.0
    if args.max_delta_per_tick is None:
        args.max_delta_per_tick = round(0.9 / args.fps, 4)
    print(f"model {args.model.upper()}  fps {args.fps:g}  "
          f"slew {args.max_delta_per_tick} rad/tick"
          + (f"  act_ensemble {args.act_ensemble}"
             if args.model == "act" and args.act_ensemble > 0 else ""))
    model_dir = str(ck / "pretrained_model")
    root = Path.home() / ".cache/huggingface/lerobot" / repo
    meta = LeRobotDatasetMetadata(repo_id=repo, root=str(root))
    state_names = meta.features["observation.state"]["names"]
    img_keys = sorted(k for k in meta.features if k.startswith(
        "observation.images"))
    print(f"arm {args.arm}: ckpt={ck}\n  state={state_names}\n  "
          f"images={img_keys}")

    pose_state = pose_state_f
    pose_actions = pose_actions_f
    # geometry for FK-derived obs + pose->joint actions
    he = json.load(open(PILOT / "calib/handeye.json"))
    A_he = np.array(he["T_cam_base"])
    B_he = np.array(he["T_flange_body"])
    Ainv, Binv = np.linalg.inv(A_he), np.linalg.inv(B_he)
    eef_tgt = np.array(json.load(open(PILOT / "calib/eef_target.json"))
                       ["p_body_mm"]).reshape(3) / 1000.0
    intr = json.load(open(PILOT / "calib/B0587_intrinsics.json"))
    K_half = np.array(intr["camera_matrix"]) / 2.0
    K_half[2, 2] = 1.0
    dist = np.array(intr["dist_coeffs"])
    A2_he = K2_half = dist2 = None
    if cam == "forward+side":
        he2 = json.load(open(PILOT / "calib/handeye_twin.json"))
        A2_he = np.array(he2["T_cam_base"])
        intr2 = json.load(open(PILOT / "calib/B0587b_intrinsics.json"))
        K2_half = np.array(intr2["camera_matrix"]) / 2.0
        K2_half[2, 2] = 1.0
        dist2 = np.array(intr2["dist_coeffs"])

    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import (ArmType, GripperType,
                                   combine_arm_and_gripper_xml)
    kin = Kinematics(combine_arm_and_gripper_xml(
        ArmType.YAM, GripperType.NO_GRIPPER), "grasp_site")

    # --- policy ---
    cfg = PreTrainedConfig.from_pretrained(model_dir)
    cfg.pretrained_path = model_dir
    if cfg.type == "diffusion":
        cfg.noise_scheduler_type = args.scheduler.upper()
        cfg.num_inference_steps = args.num_inference_steps
        cfg.n_action_steps = args.n_action_steps
    elif cfg.type == "act":
        # ACT: execute only the first n_action_steps of each predicted
        # chunk (default 100 = 6.7s open-loop at fps 15 — far too long)
        cfg.n_action_steps = min(args.n_action_steps, cfg.chunk_size)
        if args.act_ensemble > 0:
            # temporal ensembling: replan every tick, exponentially
            # average overlapping chunks — removes the chunk-boundary
            # jump (clips at exactly every n_action_steps ticks)
            cfg.temporal_ensemble_coeff = args.act_ensemble
            cfg.n_action_steps = 1
    policy = get_policy_class(cfg.type).from_pretrained(
        model_dir, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=model_dir, dataset_stats=meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}})
    planner = AsyncChunkPlanner(policy, post, device,
                                trigger=args.chunk_trigger,
                                rtc=not args.no_rtc) \
        if cfg.type == "diffusion" else None

    # --- cameras (captured locally; host is camera-free) ---
    rdr2 = None
    if cam.startswith("forward"):
        rdr = CamReader(FORWARD_DEV, 1920, 1080, 25)
        if cam == "forward+side":
            rdr2 = CamReader(TWIN_DEV, 1920, 1080, 25)
    elif cam == "head":
        # SO-101 head cam: so101_cam_track.py must be running with --release_cam
        # (tracking, or --static for the parked ablation)
        rdr = CamReader(HEAD_DEV, 1920, 1080, 25)
    else:
        rdr = CamReader(FISH_DEV, 1920, 1080, 30)
        if cam == "fish+forward":
            rdr2 = CamReader(FORWARD_DEV, 1920, 1080, 25)

    # --- robot ---
    robot = YamZmqClient(YamZmqClientConfig(
        remote_ip=args.remote_ip, zmq_cmd_port=args.zmq_cmd_port,
        zmq_obs_port=args.zmq_obs_port, cameras={}))
    robot.connect()
    robot.set_mode("policy")
    time.sleep(0.5)

    def raw_joints():
        o = robot.get_observation()
        q = np.array([float(o[n]) for n in JOINTS])
        g = float(o["gripper.pos"])
        return q, g

    def fk_cam_pose(q):
        return A_he @ kin.fk(q) @ B_he       # T_cam<-body from encoders

    def build_obs():
        q, g = raw_joints()
        frame = rdr.frame                      # BGR half-res
        obs = {}
        if pose_state:
            M = fk_cam_pose(q)
            rvec, _ = cv2.Rodrigues(M[:3, :3])
            obs["observation.state"] = np.concatenate(
                [M[:3, 3], rvec.ravel(), [g]]).astype(np.float32)
        else:
            obs["observation.state"] = np.concatenate(
                [q, [g]]).astype(np.float32)
        H, W = frame.shape[:2]
        rgb = frame[:, :, ::-1]
        if cam.startswith("forward"):
            sq = min(H, W)
            x0 = (W - sq) // 2
            base = cv2.resize(rgb[:, x0:x0 + sq],
                              (384, 384), interpolation=cv2.INTER_AREA)
            obs["observation.images.b0587"] = np.ascontiguousarray(base)
            if "observation.images.eef" in img_keys:
                M = fk_cam_pose(q)
                p = M[:3, :3] @ eef_tgt + M[:3, 3]
                px, _ = cv2.projectPoints(p.reshape(1, 3), np.zeros(3),
                                          np.zeros(3), K_half, dist)
                cx, cy = px.ravel()
                cr = 600                        # 1200 native px at half-res
                x1 = int(min(max(cx - cr / 2, 0), W - cr))
                y1 = int(min(max(cy - cr / 2, 0), H - cr))
                eef = cv2.resize(rgb[y1:y1 + cr, x1:x1 + cr], (384, 384),
                                 interpolation=cv2.INTER_AREA)
                obs["observation.images.eef"] = np.ascontiguousarray(eef)
            if rdr2 is not None and "observation.images.side" in img_keys:
                f2 = rdr2.frame
                rgb2 = f2[:, :, ::-1]
                H2, W2 = f2.shape[:2]
                M2 = A2_he @ kin.fk(q) @ B_he     # T_twin<-body
                p2 = M2[:3, :3] @ eef_tgt + M2[:3, 3]
                px2, _ = cv2.projectPoints(p2.reshape(1, 3), np.zeros(3),
                                           np.zeros(3), K2_half, dist2)
                cx2, cy2 = px2.ravel()
                cr2 = 600
                x2 = int(min(max(cx2 - cr2 / 2, 0), W2 - cr2))
                y2 = int(min(max(cy2 - cr2 / 2, 0), H2 - cr2))
                side = cv2.resize(rgb2[y2:y2 + cr2, x2:x2 + cr2],
                                  (384, 384), interpolation=cv2.INTER_AREA)
                obs["observation.images.side"] = np.ascontiguousarray(side)
        elif cam == "head":
            # full frame (or centred crop) -> 640x360, as build_dataset
            # --obs-stem / head_crop.py produced for training
            if head_crop < 1.0:
                ch, cw = int(round(H * head_crop)), int(round(W * head_crop))
                y0, x0 = (H - ch) // 2, (W - cw) // 2
                rgb = rgb[y0:y0 + ch, x0:x0 + cw]
            obs["observation.images.b0587b"] = np.ascontiguousarray(
                cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA))
        else:
            obs["observation.images.fish"] = np.ascontiguousarray(
                cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA))
            if rdr2 is not None and "observation.images.b0587" in img_keys:
                rgb2 = rdr2.frame[:, :, ::-1]     # forward full-frame
                obs["observation.images.b0587"] = np.ascontiguousarray(
                    cv2.resize(rgb2, (640, 360),
                               interpolation=cv2.INTER_AREA))
        return obs, q, g

    def act_to_joints(raw, q_seed):
        """Policy action -> (q6, gripper)."""
        if not pose_actions:
            return np.array(raw[:6], dtype=float), float(raw[6])
        R, _ = cv2.Rodrigues(np.array(raw[3:6], dtype=np.float64))
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, raw[:3]
        F_t = Ainv @ M @ Binv
        ok, q = kin.ik(F_t, "grasp_site", init_q=q_seed.copy(),
                       pos_threshold=1e-3, ori_threshold=1e-2, max_iters=60)
        return np.array(q[:6], dtype=float), float(raw[6])

    if args.tracker_ctrl != "none" and ARMS[args.arm]["cam"] == "head":
        import zmq
        want = ARMS[args.arm].get("tracker", "track")
        sock = zmq.Context.instance().socket(zmq.REQ); sock.setsockopt(zmq.LINGER, 0); sock.setsockopt(zmq.RCVTIMEO, 2000); sock.connect(args.tracker_ctrl)
        try:
            sock.send_string(json.dumps({"mode": want})); print("tracker:", sock.recv_string())
        except zmq.Again:
            print(f"WARNING: no tracker answering on {args.tracker_ctrl}; continuing")
        finally:
            sock.close()
    reset_pose = None
    if not args.no_reset or args.spot:
        if args.skeleton_root is None:
            args.skeleton_root = str(
                Path.home() / ".cache/huggingface/lerobot"
                / ARMS[args.arm].get("skeleton", "yam_pilot1"))
        reset_pose = median_first_frame_state(
            args.skeleton_root, JOINTS + ["gripper.pos"], set())
        print("Reset pose:", {k: round(v, 3) for k, v in reset_pose.items()})

    def ramp_to(target, seconds):
        q, g = raw_joints()
        cur = {**{n: q[i] for i, n in enumerate(JOINTS)},
               "gripper.pos": g}
        n = max(1, int(seconds * 30))
        for i in range(1, n + 1):
            w = i / n
            robot.send_action({k: cur[k] + w * (target[k] - cur[k])
                               for k in cur})
            precise_sleep(1 / 30)

    def safe_home():
        """Ramp to the start pose, immune to further Ctrl-C — never leave
        the arm (or a held block) hanging mid-air."""
        if reset_pose is None:
            return
        import signal as _sig
        old = _sig.signal(_sig.SIGINT, _sig.SIG_IGN)
        try:
            ramp_to(reset_pose, args.home_seconds)
        finally:
            _sig.signal(_sig.SIGINT, old)

    interval = 1.0 / args.fps
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        ep = -1
        while True:
            ep += 1
            if not args.spot and ep >= args.num_episodes:
                break
            if reset_pose is not None:
                log_say("Going to start pose")
                safe_home()
            tag = f"{ep+1}" if args.spot else \
                f"{ep+1}/{args.num_episodes}"
            input(f"\nReset scene; ENTER starts rollout {tag} "
                  "(ENTER again stops it; Ctrl-C quits + homes)…")
            if planner is not None:
                planner.reset()
            else:
                policy.reset()
            pre.reset()
            post.reset()
            slew_ticks = slew_clips = gclip_ticks = 0
            slew_worst = 0.0
            slew_last_warn = 0.0
            q_last, g_last = raw_joints()
            rows, writers = [], {}
            stamp = time.strftime("%Y%m%d_%H%M%S")
            t_start = time.perf_counter()
            while time.perf_counter() - t_start < args.episode_time:
                if enter_pressed():
                    print("stopped")
                    break
                t0 = time.perf_counter()
                obs, q_obs, g_obs = build_obs()
                frames = {k: obs[k] for k in obs if "images" in k}
                with torch.inference_mode():
                    o = prepare_observation_for_inference(
                        obs, torch.device(device), args.task_description,
                        "yam")
                    o = pre(o)
                if planner is not None:
                    raw = planner.step(o)
                else:
                    with torch.inference_mode():
                        raw = post(policy.select_action(o)) \
                            .squeeze(0).cpu().numpy()
                q_cmd, g_cmd = act_to_joints(raw, q_last)
                if args.gripper_close_bias > 0 and g_cmd < 0.5:
                    g_cmd = max(g_cmd - args.gripper_close_bias, 0.2)
                if args.action_ema > 0:
                    a = args.action_ema
                    q_cmd = a * q_cmd + (1 - a) * q_last
                if args.max_delta_per_tick > 0:
                    d = args.max_delta_per_tick
                    _q_want, _g_want = q_cmd.copy(), g_cmd
                    q_cmd = np.clip(q_cmd, q_last - d, q_last + d)
                    g_cmd = float(np.clip(g_cmd,
                                          g_last - args.gripper_max_delta,
                                          g_last + args.gripper_max_delta))
                    # live slew-clip instrumentation: a clipped tick means
                    # the policy asked for more speed than the limiter
                    # allows — sustained clipping = the arm lagging its own
                    # plan (under-reach / late-close candidates)
                    _jex = np.abs(_q_want - q_cmd).max()
                    _gex = abs(_g_want - g_cmd)
                    slew_ticks += 1
                    if _jex > 1e-9 or _gex > 1e-9:
                        slew_clips += 1
                        slew_worst = max(slew_worst, _jex)
                        gclip_ticks += _gex > 1e-9
                        if time.time() - slew_last_warn > 1.0:
                            ch = (f"joint +{_jex:.3f}rad/tick over" if _jex > 1e-9
                                  else "")
                            if _gex > 1e-9:
                                ch += (" & " if ch else "") +                                     f"gripper +{_gex:.3f} over"
                            print(f"[slew] CLIPPING {ch} "
                                  f"({slew_clips}/{slew_ticks} ticks so far)",
                                  flush=True)
                            slew_last_warn = time.time()
                prev_q, prev_g = q_last, g_last
                q_last, g_last = q_cmd, g_cmd
                S = max(args.interp_substeps, 1)
                for s in range(1, S + 1):
                    w = s / S
                    cmd = {n: float(prev_q[i] + w * (q_cmd[i] - prev_q[i]))
                           for i, n in enumerate(JOINTS)}
                    cmd["gripper.pos"] = float(prev_g + w * (g_cmd - prev_g))
                    robot.send_action(cmd)
                    if s < S:
                        precise_sleep(interval / S)
                rows.append(np.concatenate(
                    [[time.perf_counter() - t_start], q_obs, [g_obs],
                     np.array(raw, dtype=np.float32)[:7], q_cmd,
                     [g_cmd]]))
                if not args.spot:
                    for k, fr in frames.items():
                        if k not in writers:
                            h, w = fr.shape[:2]
                            writers[k] = cv2.VideoWriter(
                                str(log_dir / f"{args.arm}_{stamp}_ep{ep}_"
                                    f"{k.split('.')[-1]}.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                args.fps, (w, h))
                        writers[k].write(fr[:, :, ::-1])
                precise_sleep(max(interval - (time.perf_counter() - t0), 0))
            print(f"[slew] {slew_clips}/{max(slew_ticks,1)} ticks clipped "
                  f"({100*slew_clips/max(slew_ticks,1):.0f}%), worst joint "
                  f"excess {slew_worst:.3f} rad/tick, gripper clipped "
                  f"{gclip_ticks} ticks", flush=True)
            if planner is not None:
                print(planner.timing_summary(), flush=True)
            for vw in writers.values():
                vw.release()
            if True:  # npz tick logs in spot mode too (diagnosis data)
                tag = f"_{args.tag}" if args.tag else ""
                out = log_dir / f"{args.arm}{tag}_{stamp}_ep{ep}.npz"
                np.savez_compressed(out,
                                    rows=np.array(rows, dtype=np.float32),
                                    config_json=json.dumps(vars(args)))
                print(f"logged {out} ({len(rows)} ticks)")
            log_say("Episode done")
    except KeyboardInterrupt:
        print("\nCtrl-C — homing before exit…")
    finally:
        try:
            safe_home()
        except Exception as e:
            print(f"(homing failed: {e})")
        rdr.stop()
        robot.set_mode("idle")
        robot.disconnect()
        print("Rollout session finished (arm at start pose).")


if __name__ == "__main__":
    main()
