#!/usr/bin/env python3
"""Grasp-only A/B/N eval harness — matched-block comparison of checkpoints at
the task's weakest link, with rollout-seeded start states.

Per SCENE:
  1. Place the block anywhere (remember the spot — all trials of this scene
     re-use it back-to-back).
  2. The SEEDING policy (round-robin across models per scene) rolls out
     normally from home. When the block first enters the space between the
     gripper fingers, press SPACE — the harness captures the arm pose and the
     last two policy-tick observations. The seeding rollout CONTINUES
     uninterrupted into the grasp: it is that model's first trial.
  3. Every remaining (model, rep) runs in randomized order: re-place the
     block, the arm ramps to the captured pose, the policy's observation
     history is primed with the captured frames (identical moving-context for
     every model; no RTC action prefix — fresh chunk from identical obs), and
     the policy takes over.
  4. Each trial ends via a SETTLED grasp (gripper cmd < --close_threshold held
     continuously for --close_hold_s — reseats reset the timer; budget
     --grasp_budget) followed by a scripted,
     policy-independent stress (lift to the captured pose with grip held,
     hold, pan jiggle). Score: 0 = no grasp (auto), 1 = dropped during
     stress (press d), 2 = held (ENTER). r = redo trial, s = abort scene,
     q = quit (log is append-only).

Live per-model tallies after every trial; JSONL log per trial.
Models default to the five encoder-matrix checkpoints; override with
repeated --model name:hf_model_id:train_dataset_root.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.robot_utils import precise_sleep

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_so101 import AsyncChunkPlanner, median_first_frame_state  # noqa: E402
from record_so101 import (  # noqa: E402
    _MIN_OBS_RATE_FRACTION,
    FocusKeyboard,
    _measure_obs_rate,
    _obs_rate_banner,
)
from so101_zmq_client import SO101ZmqClient, SO101ZmqClientConfig  # noqa: E402


class WristOverlay:
    """Live wrist view blended with the captured reference frame in Rerun,
    for pixel-accurate block re-placement between trials (no table marks)."""

    def __init__(self):
        self.ok = False
        try:
            import cv2  # noqa: F401
            import rerun as rr
            rr.init("grasp_eval_overlay", spawn=True)
            self.rr, self.ok = rr, True
            print("[rerun] overlay viewer spawned")
        except Exception as e:
            print(f"[rerun] overlay unavailable ({e})")

    def log(self, live, ref):
        if not self.ok:
            return
        try:
            import cv2
            blend = cv2.addWeighted(live, 0.5, ref, 0.5, 0)
            diff = cv2.absdiff(live, ref).max(axis=2)
            try:
                self.rr.log("replace/blend", self.rr.Image(blend).compress(jpeg_quality=75))
                self.rr.log("replace/misalignment",
                            self.rr.Image(diff).compress(jpeg_quality=75))
            except Exception:
                self.rr.log("replace/blend", self.rr.Image(blend))
                self.rr.log("replace/misalignment", self.rr.Image(diff))
        except Exception:
            pass

V3_ROOT = "/mnt/nvme/lerobot/yosubshin/so101_grand_v3_visibility_wristonly"
DEFAULT_MODELS = [
    ("GN_s1000",   "/mnt/nvme/lerobot/outputs/dp_grand_v3_8500/pretrained_model",          V3_ROOT),
    ("GN_raw",     "/mnt/nvme/lerobot/outputs/dp_grand_v3_raw8500/pretrained_model",       V3_ROOT),
    ("GN_s2000",   "/mnt/nvme/lerobot/outputs/dp_grand_v3seed2/pretrained_model",          V3_ROOT),
    ("BN_scratch", "/mnt/nvme/lerobot/outputs/dp_grand_v3bnscratch_8000/pretrained_model", V3_ROOT),
    ("PT_slowLR",  "/mnt/nvme/lerobot/outputs/dp_grand_v3ft_8500/pretrained_model",        V3_ROOT),
    ("PT_fullLR",  "/mnt/nvme/lerobot/outputs/dp_grand_v3ftfull_5000/pretrained_model",    V3_ROOT),
]
STATE_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grasp-only multi-checkpoint eval")
    p.add_argument("--model", action="append", default=None, metavar="NAME:CKPT:DATAROOT",
                   help="Repeatable. Defaults to the five encoder-matrix checkpoints.")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="grasp_eval_results")
    p.add_argument("--task_description", type=str, default="Put the red block into the bin")
    # trial protocol
    p.add_argument("--seed_budget", type=float, default=45.0,
                   help="Max seconds for the seeding rollout before scene abort")
    p.add_argument("--grasp_budget", type=float, default=20.0,
                   help="Max seconds from handoff to a settled grasp (slow "
                        "adjust + reseat cycles observed to exceed 10 s)")
    p.add_argument("--close_threshold", type=float, default=60.0,
                   help="Commanded gripper below this counts toward closure (pilot: hold cmds sit at 54-55; approach 88-95)")
    p.add_argument("--close_hold_s", type=float, default=2.5,
                   help="Gripper must stay closed this long continuously before "
                        "the stress starts — a reseat (reopen) resets the timer, "
                        "so the policy finishes securing its grip first")
    p.add_argument("--stress_lift_s", type=float, default=1.5)
    p.add_argument("--stress_hold_s", type=float, default=4.0)
    p.add_argument("--jiggle_amp", type=float, default=3.0)
    p.add_argument("--jiggle_cycles", type=int, default=2)
    p.add_argument("--close_meas_threshold", type=float, default=60.0,
                   help="MEASURED gripper below this also counts toward closure "
                        "(fingers stall at block width; open is much higher)")
    p.add_argument("--empty_gripper_below", type=float, default=32.0,
                   help="Measured gripper below this at stress start suggests the "
                        "jaws closed on air (block width stalls them higher) — "
                        "shown as a scoring hint, never auto-scored")
    # deploy config (project standard)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--interp_substeps", type=int, default=2)
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--chunk_trigger", type=int, default=6)
    p.add_argument("--max_delta_per_tick", type=float, default=7.0)
    p.add_argument("--gripper_max_delta", type=float, default=20.0)
    p.add_argument("--home_seconds", type=float, default=3.0)
    # rig
    p.add_argument("--remote_ip", type=str, default="192.168.0.17")
    p.add_argument("--zmq_cmd_port", type=int, default=5601)
    p.add_argument("--zmq_obs_port", type=int, default=5602)
    p.add_argument("--save_clips", action="store_true",
                   help="Save an mp4 per trial per subscribed camera")
    p.add_argument("--cameras", type=str, default="wrist",
                   help="Host cameras to subscribe (policy uses wrist; add "
                        "'wrist,forward' for external-view clips)")
    p.add_argument("--seed", type=int, default=0, help="Rotation-order shuffle seed")
    p.add_argument("--no_overlay", action="store_true",
                   help="Disable the Rerun re-placement overlay")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    robot = SO101ZmqClient(SO101ZmqClientConfig(
        remote_ip=args.remote_ip, zmq_cmd_port=args.zmq_cmd_port,
        zmq_obs_port=args.zmq_obs_port, cameras={c: (240, 320, 3) for c in cams}))
    robot.connect()
    time.sleep(2.0)
    hz, frozen = _measure_obs_rate(robot)
    needed = args.fps * _MIN_OBS_RATE_FRACTION
    if (hz is not None and hz < needed) or frozen:
        print(_obs_rate_banner(hz or 0.0, needed), flush=True)
        robot.disconnect(); raise SystemExit(1)
    print(f"obs rate OK: {hz:.1f} Hz")

    # ---- load all models ----
    specs = ([tuple(m.split(":", 2)) for m in args.model] if args.model else DEFAULT_MODELS)
    models = {}
    for name, ckpt, root in specs:
        meta = LeRobotDatasetMetadata(repo_id=Path(root).name, root=Path(root))
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.pretrained_path = ckpt
        if cfg.type == "diffusion":
            cfg.noise_scheduler_type = args.scheduler.upper()
            cfg.num_inference_steps = args.num_inference_steps
            cfg.n_action_steps = args.n_action_steps
        policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to(device).eval()
        pre, post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=ckpt, dataset_stats=meta.stats,
            preprocessor_overrides={"device_processor": {"device": device}})
        planner = AsyncChunkPlanner(policy, post, device,
                                    trigger=args.chunk_trigger, blend=False, rtc=True)
        models[name] = dict(pre=pre, post=post, planner=planner,
                            state_names=meta.features["observation.state"]["names"])
        print(f"loaded {name}: {ckpt}")
    names = list(models)
    home_pose = median_first_frame_state(specs[0][2], STATE_NAMES, set())

    overlay = None if args.no_overlay else WristOverlay()
    kb = FocusKeyboard(); kb.start()
    rng = random.Random(args.seed)
    interval = 1.0 / args.fps
    log_path = out_dir / f"grasp_eval_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    tally = {n: [0, 0, 0] for n in names}
    # ---- resume: absorb every prior log in out_dir ----
    scene_start, prior = 0, 0
    for old_log in sorted(out_dir.glob("grasp_eval_*.jsonl")):
        if old_log == log_path:
            continue
        for line in old_log.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("model") in tally and r.get("score") in (0, 1, 2):
                tally[r["model"]][r["score"]] += 1
                prior += 1
                scene_start = max(scene_start, int(r.get("scene", -1)) + 1)
    if prior:
        print(f"RESUME: {prior} prior trials from earlier sessions loaded "
              f"(scenes 0..{scene_start - 1}); continuing at scene {scene_start}")
        print_tally_pending = True
    else:
        print_tally_pending = False

    def get_obs():
        raw = robot.get_observation()
        return raw, {n: float(raw[n]) for n in STATE_NAMES}

    def ramp_to(target, seconds):
        _, cur = get_obs()
        n = max(1, int(seconds * 30))
        for i in range(1, n + 1):
            w = i / n
            robot.send_action({k: cur[k] + w * (target[k] - cur[k]) for k in STATE_NAMES})
            precise_sleep(1.0 / 30)

    def build_obs(raw, M):
        return {"observation.state": np.array([raw[n] for n in M["state_names"]],
                                              dtype=np.float32),
                "observation.images.wrist": np.ascontiguousarray(raw["wrist"])}

    def prep(M, obs):
        with torch.inference_mode():
            o = prepare_observation_for_inference(
                dict(obs), torch.device(device), args.task_description, robot.name)
            return M["pre"](o)

    def print_tally():
        print("  TALLY  " + " | ".join(
            f"{n}: n={sum(t)} >=1:{t[1]+t[2]}/{sum(t)} =2:{t[2]}/{sum(t)}"
            for n, t in tally.items() if sum(t)), flush=True)

    def clip_writers(trial_id):
        if not args.save_clips:
            return {}
        ws = {}
        for c in cams:
            fn = out_dir / f"{trial_id}_{c}.mp4"
            ws[c] = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt",
                 "rgb24", "-s", "320x240", "-r", str(args.fps), "-i", "-", "-c:v",
                 "libx264", "-preset", "fast", "-crf", "24", str(fn)],
                stdin=subprocess.PIPE)
        return ws

    def policy_grasp_loop(M, writers, last_sent, ring=None, capture=None,
                          budget=None, allow_keys=True):
        """Run the policy until a settled grasp, budget expiry, or a manual key.
        Returns (closed_at, aborted, manual) where manual is None | "score2" |
        "score0". Keys during the phase: SPACE=capture (seeding), ENTER=force
        the stress NOW (auto-detect missed a held grasp), 2/1/0=score directly
        and end the episode early, s=abort scene, q=quit."""
        need = int(args.close_hold_s * args.fps)
        kb.clear()  # a buffered ENTER from the start prompt must not fire the
        time.sleep(0.15); kb.clear()  # manual stress trigger instantly
        closure_run, closed_at, t0 = 0, None, time.perf_counter()
        last_telemetry = 0.0
        while time.perf_counter() - t0 < budget or closure_run > 0:
            tick0 = time.perf_counter()
            raw, joints = get_obs()
            for c, w in writers.items():
                if isinstance(raw.get(c), np.ndarray):
                    w.stdin.write(np.ascontiguousarray(raw[c]).tobytes())
            if ring is not None:
                ring.append((raw, dict(joints)))
                if len(ring) > 2:
                    ring.pop(0)
            if allow_keys:
                pressed = kb.get_pressed()
                if "q" in pressed:
                    raise KeyboardInterrupt
                if "s" in pressed:
                    return closed_at, True, None
                if "\r" in pressed or "\n" in pressed:
                    closed_at = time.perf_counter() - t0
                    print("  >>> manual stress trigger", flush=True)
                    return closed_at, False, None
                if "2" in pressed or "g" in pressed:
                    print("  >>> manual score 2 (verified by eye)", flush=True)
                    return time.perf_counter() - t0, False, "score2"
                if "1" in pressed:
                    print("  >>> manual score 1 (grabbed but lost)", flush=True)
                    return time.perf_counter() - t0, False, "score1"
                if "0" in pressed or "f" in pressed:
                    print("  >>> manual score 0 (ended early)", flush=True)
                    return None, False, "score0"
                if capture is not None and not capture and " " in pressed:
                    capture["pose"] = dict(joints)
                    capture["obs"] = [r for r, _ in ring]
                    capture["t"] = time.perf_counter() - t0
                    print("  >>> CAPTURED start state (rollout continues)", flush=True)
            obs = build_obs(raw, M)
            a = M["planner"].step(prep(M, obs))
            cmd = {n: float(a[i]) for i, n in enumerate(M["state_names"])}
            prev = dict(last_sent)
            for n in STATE_NAMES:
                d = args.gripper_max_delta if n == "gripper.pos" else args.max_delta_per_tick
                cmd[n] = min(max(cmd[n], last_sent[n] - d), last_sent[n] + d)
                last_sent[n] = cmd[n]
            S = args.interp_substeps
            for s_ in range(1, S + 1):
                w_ = s_ / S
                robot.send_action({n: prev[n] + w_ * (cmd[n] - prev[n]) for n in STATE_NAMES})
                if s_ < S:
                    precise_sleep(interval / S)
            # dual-signal, flicker-tolerant closure: commanded OR measured gripper
            # below threshold counts; brief flickers above decay the streak
            # instead of resetting it
            meas = joints["gripper.pos"]
            closing = (cmd["gripper.pos"] < args.close_threshold
                       or meas < args.close_meas_threshold)
            closure_run = closure_run + 1 if closing else max(0, closure_run - 3)
            now = time.perf_counter()
            if now - last_telemetry > 1.0:
                print(f"    t={now - t0:5.1f}s grip cmd {cmd['gripper.pos']:5.1f} "
                      f"meas {meas:5.1f} streak {closure_run}/{need}", flush=True)
                last_telemetry = now
            if closure_run >= need:
                closed_at = now - t0
                break
            precise_sleep(max(interval - (now - tick0), 0.0))
        return closed_at, False, None

    def confirm_score(suggest, why):
        print(f"  {why} — suggested score {suggest}. ENTER=accept {suggest} | "
              "2 | 1 | 0 | r=redo", flush=True)
        kb.clear()
        while True:
            pressed = kb.get_pressed()
            if "\r" in pressed or "\n" in pressed:
                return suggest
            for k, v in (("2", 2), ("1", 1), ("0", 0), ("r", -1)):
                if k in pressed:
                    return v
            if "q" in pressed:
                raise KeyboardInterrupt
            precise_sleep(0.05)

    def stress_and_score(anchor_pose, grip_hold):
        _, cur = get_obs()
        meas_grip = cur["gripper.pos"]
        hint = ("EMPTY? (closed on air)" if meas_grip < args.empty_gripper_below
                else "holding (stalled at block width)")
        print(f"  measured gripper at stress start: {meas_grip:.1f} -> {hint}", flush=True)
        nlift = max(1, int(args.stress_lift_s * 30))
        for i in range(1, nlift + 1):
            w_ = i / nlift
            tgt = {k: cur[k] + w_ * (anchor_pose[k] - cur[k]) for k in STATE_NAMES}
            tgt["gripper.pos"] = grip_hold
            robot.send_action(tgt); precise_sleep(1.0 / 30)
        hold = {**anchor_pose, "gripper.pos": grip_hold}
        t_h = time.perf_counter()
        while time.perf_counter() - t_h < args.stress_hold_s:
            robot.send_action(hold); precise_sleep(1.0 / 30)
        for _ in range(args.jiggle_cycles):
            for sgn in (1, -1):
                jp = dict(hold); jp["shoulder_pan.pos"] += sgn * args.jiggle_amp
                for _ in range(8):
                    robot.send_action(jp); precise_sleep(1.0 / 30)
        robot.send_action(hold)
        _, cur_end = get_obs()
        meas_end = cur_end["gripper.pos"]
        # release + retreat BEFORE scoring so ENTER rolls the next trial instantly
        rel = {**hold, "gripper.pos": max(grip_hold, 70.0)}
        for _ in range(10):
            robot.send_action(rel); precise_sleep(1.0 / 30)
        held_start = meas_grip >= args.empty_gripper_below
        held_end = meas_end >= args.empty_gripper_below
        if held_start and held_end:
            suggest, why = 2, f"still holding at stress end (meas {meas_end:.1f})"
        elif held_start:
            suggest, why = 1, f"held at start, empty at stress end (meas {meas_end:.1f})"
        else:
            suggest, why = 0, f"jaws empty throughout (meas {meas_end:.1f})"
        score = confirm_score(suggest, "stress done: " + why)
        return score, meas_grip

    def record(scene, m, rep, role, score, closed_at, seed_model):
        tally[m][score] += 1
        with open(log_path, "a") as f:
            f.write(json.dumps(dict(scene=scene, model=m, rep=rep, role=role,
                                    score=score, closed_at=closed_at,
                                    seed_model=seed_model, t=time.time())) + "\n")
        print_tally()

    scene = scene_start
    trial_no = 0
    if print_tally_pending:
        print_tally()
    try:
        while True:
            seed_model = names[scene % len(names)]
            print(f"\n===== SCENE {scene} — seeding model: {seed_model} =====")
            print("Place the block (remember the spot!), ENTER to start the seeding "
                  "rollout. During it: SPACE the moment the block enters the gripper "
                  "opening. s=abort scene q=quit", flush=True)
            kb.clear()
            while True:
                pressed = kb.get_pressed()
                if "q" in pressed:
                    raise KeyboardInterrupt
                if "\r" in pressed or "\n" in pressed:
                    break
                precise_sleep(0.05)
            robot.enable_torque()
            ramp_to(home_pose, args.home_seconds)
            M = models[seed_model]
            M["planner"].reset(); M["pre"].reset(); M["post"].reset()
            trial_no += 1
            writers = clip_writers(f"sc{scene:02d}_{seed_model}_seed")
            _, last_sent = get_obs()
            ring, capture = [], {}
            closed_at, aborted, manual = policy_grasp_loop(
                M, writers, last_sent, ring=ring, capture=capture,
                budget=args.seed_budget)
            for w in writers.values():
                w.stdin.close(); w.wait()
            if aborted or not capture:
                print("  scene aborted (no capture) — re-place and try again", flush=True)
                continue
            try:
                import cv2
                cv2.imwrite(str(out_dir / f"sc{scene:02d}_ref.png"),
                            capture["obs"][-1]["wrist"][:, :, ::-1])
            except Exception:
                pass
            if manual == "score2":
                score = 2
            elif manual == "score1":
                score = 1
            elif manual == "score0":
                score = 0
            elif closed_at is None:
                score = confirm_score(0, "seeding: no settled grasp detected")
                if score == -1:
                    print("  scene restarts", flush=True)
                    continue
            else:
                score, meas_grip = stress_and_score(capture["pose"], last_sent["gripper.pos"])
                if score == -1:
                    print("  redo not supported for seeding trial — scene restarts", flush=True)
                    continue
            record(scene, seed_model, 0, "seed", score, closed_at, seed_model)

            # remaining trials of this scene, randomized
            order = [(m, r) for m in names for r in range(args.reps)
                     if not (m == seed_model and r == 0)]
            rng.shuffle(order)
            i = 0
            while i < len(order):
                m, rep = order[i]; i += 1
                trial_no += 1
                print(f"\n[scene {scene} | {m} rep {rep}] arm moving to capture pose; "
                      "then re-place via overlay (s=skip scene, q=quit)", flush=True)
                skip = False
                M = models[m]
                robot.enable_torque()
                ramp_to(capture["pose"], 2.0)
                ref = capture["obs"][-1]["wrist"] if capture.get("obs") else None
                print("  align the block to the GHOST in the overlay, then ENTER",
                      flush=True)
                kb.clear()
                while True:
                    pressed = kb.get_pressed()
                    if "q" in pressed:
                        raise KeyboardInterrupt
                    if "s" in pressed:
                        skip = True; break
                    if "\r" in pressed or "\n" in pressed:
                        break
                    raw_l, _ = get_obs()
                    if overlay is not None and ref is not None:
                        overlay.log(np.ascontiguousarray(raw_l["wrist"]), ref)
                    precise_sleep(1.0 / 15)
                if skip:
                    break
                M["planner"].reset(); M["pre"].reset(); M["post"].reset()
                # prime the obs history with the captured pre-handoff frame so every
                # model sees the same moving context (no RTC action prefix — fresh
                # chunk from identical observations)
                if capture["obs"]:
                    M["planner"].step(prep(M, build_obs(capture["obs"][0], M)))
                writers = clip_writers(f"sc{scene:02d}_{m}_r{rep}")
                _, last_sent = get_obs()
                closed_at, aborted_t, manual = policy_grasp_loop(
                    M, writers, last_sent, budget=args.grasp_budget, allow_keys=True)
                for w in writers.values():
                    w.stdin.close(); w.wait()
                if aborted_t:
                    break
                if manual == "score2":
                    score = 2
                elif manual == "score1":
                    score = 1
                elif manual == "score0":
                    score = 0
                elif closed_at is None:
                    score = confirm_score(0, "no settled grasp within budget")
                    if score == -1:
                        print("  redo — trial re-queued", flush=True)
                        order.append((m, rep))
                        continue
                else:
                    score, meas_grip = stress_and_score(capture["pose"], last_sent["gripper.pos"])
                    if score == -1:
                        print("  redo — trial re-queued", flush=True)
                        order.append((m, rep))
                        continue
                record(scene, m, rep, "trial", score, closed_at, seed_model)
            scene += 1
    except KeyboardInterrupt:
        pass
    finally:
        kb.stop()
        try:
            robot.disconnect()
        except Exception:
            pass
        print(f"\nlog: {log_path}")
        print_tally()


if __name__ == "__main__":
    main()
