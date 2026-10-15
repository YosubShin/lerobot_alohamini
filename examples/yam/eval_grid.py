#!/usr/bin/env python3
"""Placement-grid evaluation on the YAM: the same scene, every policy,
re-placed to the pixel, scored, logged — the SO-101 grid protocol for the
pilot arms.

Per SCENE (a block placement you choose; the bin stays where it is):
  1. First trial of the scene: place the block, ENTER. The forward-camera
     frame at that moment becomes the scene's REFERENCE and is saved.
  2. Every later trial of the scene (other policies, repeats): the live
     forward view is shown blended 50/50 with the reference, plus a
     misalignment image, in a Rerun window — slide the block until the ghost
     disappears, then ENTER.
  3. The policy runs; press ENTER the moment the block is in the bin (that
     timestamp is the time-to-success) or let it hit the budget; the arm
     homes, and you score the trial with one key:
        ENTER / s  success (block in the bin)       n  never grasped
        d  dropped before the bin                   b  dropped at / on the bin
        t  timed out holding / wandering            r  redo this trial (not logged)
        q  quit (log is append-only; rerun resumes where it stopped)
Policy order is shuffled per scene and per repeat (seeded), so no policy
always goes first on a fresh scene. Live tallies after every trial.

Outputs in --out_dir: trials.jsonl (one line per scored trial), per-scene
reference PNGs, and per-trial tick logs (.npz) + observation videos, the
same format evaluate_yam.py writes.

Usage (host first, policy-capable mode):
  python eval_grid.py --policies Bj,mix_Bj_r5,glove_Bj_r5,Aj,mix_Aj_r5,glove_Aj_r5 --scenes 9 --reps 2
  python eval_grid.py ... --resume     # continue a partially done grid
All ARMS entries are ACT (round-7 deploy recipe: temporal ensemble 0.01,
fps 15, 0.9 rad/s slew). Deploy knobs mirror evaluate_yam.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alohamini"))
sys.path.insert(0, str(Path.home() / "co/pilot/scripts"))
sys.path.insert(0, str(Path.home() / "co/i2rt"))

from evaluate_yam import (ARMS, JOINTS, PILOT, FORWARD_DEV, FISH_DEV, HEAD_DEV,  # noqa: E402
                          CamReader)
from evaluate_so101 import enter_pressed, median_first_frame_state  # noqa: E402
import torch  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.utils import log_say  # noqa: E402
from yam_zmq_client import YamZmqClient, YamZmqClientConfig  # noqa: E402

OUTCOMES = {"": "success", "s": "success", "n": "no_grasp", "d": "dropped_before_bin",
            "b": "dropped_at_bin", "t": "timeout"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policies", required=True, help="comma list of ARMS names (ACT entries)")
    p.add_argument("--scenes", type=int, default=9)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--out_dir", default=str(PILOT / ("rollout_logs/grid_" + time.strftime("%Y%m%d"))))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--episode_time", type=float, default=60.0)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--n_action_steps", type=int, default=15)
    p.add_argument("--act_ensemble", type=float, default=0.01)
    p.add_argument("--max_delta_per_tick", type=float, default=None)
    p.add_argument("--gripper_max_delta", type=float, default=0.2)
    p.add_argument("--interp_substeps", type=int, default=2)
    p.add_argument("--home_seconds", type=float, default=3.0)
    p.add_argument("--skeleton_root", default=str(Path.home() / ".cache/huggingface/lerobot/yam_pilot5"))
    p.add_argument("--task_description", default="pick red block into bin")
    p.add_argument("--remote_ip", default="127.0.0.1")
    p.add_argument("--zmq_cmd_port", type=int, default=5601)
    p.add_argument("--zmq_obs_port", type=int, default=5602)
    p.add_argument("--no_overlay", action="store_true")
    return p.parse_args()


class Overlay:
    """Live forward view blended with the scene's reference frame (Rerun)."""

    def __init__(self, enabled):
        self.ok = False
        if not enabled:
            return
        try:
            import rerun as rr
            import rerun.blueprint as rrb
            rr.init("yam_eval_grid", spawn=True)
            # fixed 2x2 layout, re-sent on every launch so a viewer left open from an
            # earlier run does not keep a stale layout: coverage | live / blend | misalignment
            rr.send_blueprint(rrb.Blueprint(rrb.Grid(
                rrb.Spatial2DView(origin="place/coverage", name="coverage (place: earlier scenes)"),
                rrb.Spatial2DView(origin="live/forward", name="live forward"),
                rrb.Spatial2DView(origin="replace/blend", name="align: blend"),
                rrb.Spatial2DView(origin="replace/misalignment", name="align: misalignment"),
                grid_columns=2), collapse_panels=True), make_active=True)
            self.rr, self.ok = rr, True
            print("[rerun] overlay viewer spawned")
        except Exception as e:
            print(f"[rerun] overlay unavailable ({e}) — align by eye")

    def show(self, live, ref):
        if not self.ok:
            return
        blend = cv2.addWeighted(live, 0.5, ref, 0.5, 0)
        diff = cv2.absdiff(live, ref).max(axis=2)
        self.rr.log("replace/blend", self.rr.Image(blend[:, :, ::-1]).compress(jpeg_quality=70))
        self.rr.log("replace/misalignment", self.rr.Image(diff).compress(jpeg_quality=70))

    def show_live(self, live):
        """Plain live forward view ('live/forward'), half resolution, for watching the workspace."""
        if not self.ok:
            return
        small = cv2.resize(live, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        self.rr.log("live/forward", self.rr.Image(small[:, :, ::-1]).compress(jpeg_quality=60))

    def show_coverage(self, live, placements, current, boxes=()):
        """Live forward view with every earlier scene's block position drawn on it
        (numbered yellow rings), the block as currently detected (green), and the
        forward policy's input regions (centre square, fovea at the current pose)."""
        if not self.ok:
            return
        self.rr.log("place/coverage", self.rr.Image(draw_coverage(live, placements, current, boxes)[:, :, ::-1])
                    .compress(jpeg_quality=70))


def detect_red_block(frame):
    """Centroid (x, y) of the largest red blob in a BGR forward frame, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 120, 70), (8, 255, 255)) | cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < 400:
        return None
    return float(cen[i][0]), float(cen[i][1])


def policy_view_boxes(q, kin, geom, shape):
    """What the forward-camera policy sees, in forward-frame px: the centre square
    (resized to 384x384) and the 600 px fovea around the gripper at joints q."""
    H, W = shape[:2]
    sq = min(H, W); x0 = (W - sq) // 2
    boxes = [(x0, 0, x0 + sq, sq, "policy: centre square", (255, 200, 0))]
    A_he, B_he, eef_tgt, K_half, dist = geom
    M = A_he @ kin.fk(q) @ B_he
    p = M[:3, :3] @ eef_tgt + M[:3, 3]
    px, _ = cv2.projectPoints(p.reshape(1, 3), np.zeros(3), np.zeros(3), K_half, dist)
    cx, cy = px.ravel(); cr = 600
    x1 = int(min(max(cx - cr / 2, 0), W - cr)); y1 = int(min(max(cy - cr / 2, 0), H - cr))
    boxes.append((x1, y1, x1 + cr, y1 + cr, "policy: fovea (now)", (255, 120, 255)))
    return boxes


def draw_coverage(live, placements, current, boxes=()):
    im = live.copy()
    for x1, y1, x2, y2, label, col in boxes:
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(im, label, (int(x1) + 6, int(y1) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    for s, (x, y) in sorted(placements.items()):
        cv2.circle(im, (int(x), int(y)), 28, (0, 220, 255), 2)
        cv2.putText(im, str(s), (int(x) - 10, int(y) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    if current is not None:
        cv2.circle(im, (int(current[0]), int(current[1])), 28, (0, 255, 0), 3)
    return im


class Runner:
    """One loaded policy with its observation builder."""

    def __init__(self, name, device, fwd, fish, kin, geom, args, head=None):
        spec = ARMS[name]
        if "act" not in spec:
            raise SystemExit(f"{name}: no ACT checkpoint in ARMS")
        repo, ck_rel = spec["act"]
        self.name, self.cam = name, spec["cam"]
        self.fwd, self.fish, self.head, self.kin, self.geom, self.args = fwd, fish, head, kin, geom, args
        self.crop = float(spec.get("crop", 1.0))   # head arms: centred crop fraction before the 640x360 resize
        if self.cam not in ("forward", "fish", "head"):
            raise SystemExit(f"{name}: cam '{self.cam}' not supported by eval_grid (forward, fish or head)")
        if self.cam == "head" and head is None:
            raise SystemExit(f"{name}: head arm but no head camera reader")
        model_dir = str(PILOT / "train_runs" / ck_rel / "pretrained_model")
        root = Path.home() / ".cache/huggingface/lerobot" / repo
        meta = LeRobotDatasetMetadata(repo_id=repo, root=str(root))
        self.img_keys = sorted(k for k in meta.features if k.startswith("observation.images"))
        cfg = PreTrainedConfig.from_pretrained(model_dir)
        cfg.pretrained_path = model_dir
        assert cfg.type == "act", f"{name}: eval_grid runs ACT policies only"
        cfg.n_action_steps = min(args.n_action_steps, cfg.chunk_size)
        if args.act_ensemble > 0:
            cfg.temporal_ensemble_coeff = args.act_ensemble
            cfg.n_action_steps = 1
        self.policy = get_policy_class(cfg.type).from_pretrained(model_dir, config=cfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=model_dir, dataset_stats=meta.stats,
            preprocessor_overrides={"device_processor": {"device": device}})
        self.device = device
        print(f"  loaded {name}: {ck_rel}  cam={self.cam}  images={self.img_keys}")

    def reset(self):
        self.policy.reset(); self.pre.reset(); self.post.reset()

    def obs(self, q, g):
        A_he, B_he, eef_tgt, K_half, dist = self.geom
        obs = {"observation.state": np.concatenate([q, [g]]).astype(np.float32)}
        if self.cam == "forward":
            frame = self.fwd.frame
            H, W = frame.shape[:2]
            rgb = frame[:, :, ::-1]
            sq = min(H, W); x0 = (W - sq) // 2
            obs["observation.images.b0587"] = np.ascontiguousarray(
                cv2.resize(rgb[:, x0:x0 + sq], (384, 384), interpolation=cv2.INTER_AREA))
            if "observation.images.eef" in self.img_keys:
                M = A_he @ self.kin.fk(q) @ B_he
                p = M[:3, :3] @ eef_tgt + M[:3, 3]
                px, _ = cv2.projectPoints(p.reshape(1, 3), np.zeros(3), np.zeros(3), K_half, dist)
                cx, cy = px.ravel(); cr = 600
                x1 = int(min(max(cx - cr / 2, 0), W - cr)); y1 = int(min(max(cy - cr / 2, 0), H - cr))
                obs["observation.images.eef"] = np.ascontiguousarray(
                    cv2.resize(rgb[y1:y1 + cr, x1:x1 + cr], (384, 384), interpolation=cv2.INTER_AREA))
        elif self.cam == "head":
            # SO-101 head cam, driven by so101_cam_track.py: full frame (or centred crop) -> 640x360,
            # the same path build_dataset --obs-stem / head_crop.py produced for training
            rgb = self.head.frame[:, :, ::-1]
            if self.crop < 1.0:
                H, W = rgb.shape[:2]; ch, cw = int(round(H * self.crop)), int(round(W * self.crop))
                y0, x0 = (H - ch) // 2, (W - cw) // 2
                rgb = rgb[y0:y0 + ch, x0:x0 + cw]
            obs["observation.images.b0587b"] = np.ascontiguousarray(
                cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA))
        else:
            rgb = self.fish.frame[:, :, ::-1]
            obs["observation.images.fish"] = np.ascontiguousarray(
                cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA))
        return obs

    def act(self, obs):
        with torch.inference_mode():
            o = prepare_observation_for_inference(obs, torch.device(self.device), self.args.task_description, "yam")
            o = self.pre(o)
            return self.post(self.policy.select_action(o)).squeeze(0).cpu().numpy()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = [n.strip() for n in args.policies.split(",") if n.strip()]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    log_p = out / "trials.jsonl"
    done = set()
    if args.resume and log_p.exists():
        for line in open(log_p):
            r = json.loads(line); done.add((r["scene"], r["rep"], r["policy"]))
        print(f"resume: {len(done)} trials already scored")
    if args.max_delta_per_tick is None:
        args.max_delta_per_tick = round(0.9 / args.fps, 4)

    # geometry (robot-side: the ball is on the robot at deployment)
    he = json.load(open(PILOT / "calib/handeye.json"))
    A_he, B_he = np.array(he["T_cam_base"]), np.array(he["T_flange_body"])
    eef_tgt = np.array(json.load(open(PILOT / "calib/eef_target.json"))["p_body_mm"]).reshape(3) / 1000.0
    intr = json.load(open(PILOT / "calib/B0587_intrinsics.json"))
    K_half = np.array(intr["camera_matrix"]) / 2.0; K_half[2, 2] = 1.0
    dist = np.array(intr["dist_coeffs"])
    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
    kin = Kinematics(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER), "grasp_site")

    # cameras: forward always (overlay + forward arms); fish if any wrist arm
    fwd = CamReader(FORWARD_DEV, 1920, 1080, 25)
    fish = CamReader(FISH_DEV, 1920, 1080, 30) if any(ARMS[n]["cam"] == "fish" for n in names) else None
    head = None
    if any(ARMS[n]["cam"] == "head" for n in names):
        # the tracker (so101_cam_track.py --release_cam) must already be parked and have released the camera
        head = CamReader(HEAD_DEV, 1920, 1080, 25)
        print("head camera open — make sure so101_cam_track.py is running (tracking, or --static for the parked ablation)")
    geom = (A_he, B_he, eef_tgt, K_half, dist)
    print("loading policies:")
    runners = {n: Runner(n, device, fwd, fish, kin, geom, args, head=head) for n in names}
    overlay = Overlay(not args.no_overlay)

    robot = YamZmqClient(YamZmqClientConfig(remote_ip=args.remote_ip, zmq_cmd_port=args.zmq_cmd_port,
                                            zmq_obs_port=args.zmq_obs_port, cameras={}))
    robot.connect(); robot.set_mode("policy"); time.sleep(0.5)

    def raw_joints():
        o = robot.get_observation()
        return np.array([float(o[n]) for n in JOINTS]), float(o["gripper.pos"])

    def rotor_max():
        """Hottest follower rotor (degC) from the host's last message, or None if the host is older."""
        o = robot.get_observation()
        t = [float(v) for k, v in o.items() if k.startswith("rotor.")]
        return max(t) if t else None

    def wait_cool(limit=60.0):
        """Hold the next trial until every rotor is below `limit` degC."""
        t = rotor_max()
        if t is None or t < limit:
            return
        print(f"  rotor {t:.0f} degC — waiting for < {limit:.0f} before the next trial", flush=True)
        while (t := rotor_max()) is not None and t >= limit:
            time.sleep(5)
        print(f"  rotor {t:.0f} degC — continuing", flush=True)

    reset_pose = median_first_frame_state(args.skeleton_root, JOINTS + ["gripper.pos"], set())

    def ramp_to(target, seconds):
        q, g = raw_joints(); cur = {**{n: q[i] for i, n in enumerate(JOINTS)}, "gripper.pos": g}
        n = max(1, int(seconds * 30))
        for i in range(1, n + 1):
            w = i / n
            robot.send_action({k: cur[k] + w * (target[k] - cur[k]) for k in cur}); precise_sleep(1 / 30)

    def safe_home():
        import signal as _sig
        old = _sig.signal(_sig.SIGINT, _sig.SIG_IGN)
        try:
            ramp_to(reset_pose, args.home_seconds)
        finally:
            _sig.signal(_sig.SIGINT, old)

    # block placement per scene (forward-frame px), recovered from saved references on resume
    placements = {}
    for p_ in sorted(out.glob("scene*_ref.png")):
        c = detect_red_block(cv2.imread(str(p_)))
        if c is not None:
            placements[int(p_.stem[5:-4])] = c

    def flush_stdin():
        import select
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()

    def wait_enter_with_overlay(ref, prompt):
        """Block until ENTER; a line reading 'q' quits the grid instead. With no
        reference (first trial of a scene) the Rerun 'place/coverage' view shows
        every earlier scene's block position over the live frame, so new
        placements can be spread out."""
        import select
        flush_stdin()  # ENTERs mashed during the previous trial must not start this one
        print(prompt, flush=True)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                if sys.stdin.readline().strip().lower() == "q":
                    raise KeyboardInterrupt
                return
            live = fwd.frame
            overlay.show_live(live)
            if ref is not None:
                overlay.show(live, ref)
            else:
                overlay.show_coverage(live, placements, detect_red_block(live),
                                      policy_view_boxes(raw_joints()[0], kin, geom, live.shape))
            time.sleep(0.1)

    def rollout(run, stamp, scene, rep):
        import threading
        stop_live = threading.Event()

        def live_loop():  # ~7 Hz workspace view off the control thread (encode + send cost ~8 ms)
            while not stop_live.wait(0.15):
                overlay.show_live(fwd.frame)
        threading.Thread(target=live_loop, daemon=True).start()
        run.reset()
        q_last, g_last = raw_joints()
        rows, writers = [], {}
        interval = 1.0 / args.fps; t_start = time.perf_counter()
        while time.perf_counter() - t_start < args.episode_time:
            if enter_pressed():
                print("stopped"); break
            if len(rows) % 15 == 0 and (rt := rotor_max()) is not None and rt > 80.0:
                print(f"  ROTOR {rt:.0f} degC — stall; aborting this rollout", flush=True); break
            t0 = time.perf_counter()
            q_obs, g_obs = raw_joints()
            obs = run.obs(q_obs, g_obs)
            frames = {k: v for k, v in obs.items() if "images" in k}  # numpy copies: act() turns obs into tensors
            raw = run.act(obs)
            q_cmd, g_cmd = np.array(raw[:6], dtype=float), float(raw[6])
            d = args.max_delta_per_tick
            q_cmd = np.clip(q_cmd, q_last - d, q_last + d)
            g_cmd = float(np.clip(g_cmd, g_last - args.gripper_max_delta, g_last + args.gripper_max_delta))
            prev_q, prev_g = q_last, g_last
            q_last, g_last = q_cmd, g_cmd
            S = max(args.interp_substeps, 1)
            for s in range(1, S + 1):
                w = s / S
                cmd = {n: float(prev_q[i] + w * (q_cmd[i] - prev_q[i])) for i, n in enumerate(JOINTS)}
                cmd["gripper.pos"] = float(prev_g + w * (g_cmd - prev_g))
                robot.send_action(cmd)
                if s < S:
                    precise_sleep(interval / S)
            rows.append(np.concatenate([[time.perf_counter() - t_start], q_obs, [g_obs],
                                        np.array(raw, dtype=np.float32)[:7], q_cmd, [g_cmd]]))
            for k, fr in frames.items():
                if k not in writers:
                    h, w = fr.shape[:2]
                    writers[k] = cv2.VideoWriter(str(out / f"{run.name}_S{scene}r{rep}_{stamp}_{k.split('.')[-1]}.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                writers[k].write(fr[:, :, ::-1])
            precise_sleep(max(interval - (time.perf_counter() - t0), 0))
        elapsed = time.perf_counter() - t_start
        stop_live.set()
        for vw in writers.values():
            vw.release()
        npz = out / f"{run.name}_S{scene}r{rep}_{stamp}.npz"
        np.savez_compressed(npz, rows=np.array(rows, dtype=np.float32), config_json=json.dumps(vars(args)))
        return len(rows), npz.name, elapsed

    records = []
    for line in (open(log_p) if log_p.exists() and args.resume else []):
        records.append(json.loads(line))

    def print_tally():
        """Per policy: successes/trials, median and IQR of time-to-success, failure modes."""
        short = {"no_grasp": "n", "dropped_before_bin": "d", "dropped_at_bin": "b", "timeout": "t"}
        print("  tally:")
        for n in names:
            rs = [r for r in records if r["policy"] == n]
            ok = [r["elapsed_s"] for r in rs if r["outcome"] == "success"]
            tts = (f"  tts {np.median(ok):.1f}s" + (f" [{np.percentile(ok, 25):.1f}-{np.percentile(ok, 75):.1f}]"
                                                    if len(ok) >= 4 else "")) if ok else ""
            fails = {k: sum(r["outcome"] == v for r in rs) for v, k in short.items()}
            ft = " ".join(f"{k}{c}" for k, c in fails.items() if c)
            pct = f" ({100 * len(ok) / len(rs):.0f}%)" if rs else ""
            print(f"    {n:14s} {len(ok)}/{len(rs)}{pct}{tts}" + (f"  fails: {ft}" if ft else ""), flush=True)

    try:
        for scene in range(1, args.scenes + 1):
            ref_p = out / f"scene{scene}_ref.png"
            ref = cv2.imread(str(ref_p)) if ref_p.exists() else None
            for rep in range(args.reps):
                order = names[:]
                random.Random(args.seed + 97 * scene + rep).shuffle(order)
                for name in order:
                    if (scene, rep, name) in done:
                        continue
                    run = runners[name]
                    while True:
                        wait_cool()
                        log_say("Going to start pose"); safe_home()
                        if ref is None:
                            wait_enter_with_overlay(None, f"\n=== scene {scene} rep {rep} — {name} (FIRST trial of this scene): "
                                                          "place the block, ENTER (captures the reference)…")
                            ref = fwd.frame.copy(); cv2.imwrite(str(ref_p), ref)
                            c = detect_red_block(ref)
                            if c is not None:
                                placements[scene] = c
                                json.dump({str(k): v for k, v in sorted(placements.items())},
                                          open(out / "placements.json", "w"), indent=1)
                            print(f"  reference saved -> {ref_p.name}; block at "
                                  + (f"({c[0]:.0f}, {c[1]:.0f}) px" if c else "?? (not detected)"))
                        else:
                            wait_enter_with_overlay(ref, f"\n=== scene {scene} rep {rep} — {name}: align the block to the ghost "
                                                         "(Rerun: replace/blend, replace/misalignment), ENTER to run…")
                        stamp = time.strftime("%Y%m%d_%H%M%S")
                        ticks, npz, elapsed = rollout(run, stamp, scene, rep)
                        log_say("Episode done"); safe_home()
                        if ticks < 45:  # < 3 s: a stray ENTER stopped it, not a real trial
                            print(f"  only {ticks} ticks — discarded, redoing this trial")
                            for f_ in out.glob(f"{run.name}_S{scene}r{rep}_{stamp}*"):
                                f_.unlink()
                            continue
                        flush_stdin()  # any ENTERs mashed after the stop must not score the trial
                        while True:
                            key = input("  outcome [ENTER/s success, n no-grasp, d dropped-before-bin, b dropped-at-bin, "
                                        "t timeout, r redo, q quit]: ").strip().lower()
                            if key in OUTCOMES or key in ("r", "q"):
                                break
                        if key == "q":
                            raise KeyboardInterrupt
                        if key == "r":
                            print("  redo"); continue
                        # elapsed = time from start to your ENTER (press it the moment the
                        # block is in the bin) or to the budget: time-to-success on successes
                        rec = {"scene": scene, "rep": rep, "policy": name, "outcome": OUTCOMES[key],
                               "elapsed_s": round(elapsed, 1), "stamp": stamp, "ticks": ticks, "npz": npz, "t": time.time()}
                        with open(log_p, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                        records.append(rec)
                        print_tally()
                        break
        print("\ngrid complete"); print_tally()
    except KeyboardInterrupt:
        print("\nstopping — homing…")
    finally:
        try:
            safe_home()
        except Exception:
            pass
        # a scene whose reference was captured but no trial scored starts over on resume
        logged = set()
        if log_p.exists():
            for line in open(log_p):
                logged.add(json.loads(line)["scene"])
        for scene in range(1, args.scenes + 1):
            ref_p = out / f"scene{scene}_ref.png"
            if ref_p.exists() and scene not in logged:
                ref_p.unlink(); print(f"  discarded {ref_p.name} (no trial scored for scene {scene})")
        fwd.stop()
        if head is not None:
            head.stop()
        if fish is not None:
            fish.stop()
        print(f"log: {log_p}")


if __name__ == "__main__":
    main()
