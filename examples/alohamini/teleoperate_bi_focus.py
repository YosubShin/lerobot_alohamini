#!/usr/bin/env python
"""AlohaMini bimanual teleop — focus-aware keyboard variant.

Differences vs teleoperate_bi.py:
  * Base/lift keyboard input is read from THIS terminal's stdin, so keys only
    register while this terminal window is focused (no system-wide capture).
  * The rerun viewer is OFF by default (pass --rerun to enable) to keep the
    loop lean / reduce latency.
  * Prints a compact live status (loop fps + active keys) instead of the full
    action dict every frame.

Keyboard (base): w/s fwd/back, z/x strafe L/R, a/d rotate L/R, r/f speed +/-,
                 u/j lift up/down, q quit.
Note: stdin has no key-release event, so "hold to move" uses the OS key-repeat
plus a short active-timeout (--key_timeout, default 0.35s = coast after release).
"""
import argparse
import os
import sys
import termios
import time

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep

# ---------------- args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--no_robot", action="store_true", help="Do not connect robot, only print actions")
parser.add_argument("--no_leader", action="store_true", help="Do not connect leader arms")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="Pi host IP (e.g. 192.168.0.50)")
parser.add_argument("--robot_model", type=str, default="alohamini1",
                    choices=["alohamini1", "alohamini2", "alohamini2pro"])
parser.add_argument("--leader_id", type=str, default="so101_leader_bi")
parser.add_argument("--arm_profile", type=str, default="so-arm-5dof",
                    choices=["so-arm-5dof", "am-leader-6dof"])
parser.add_argument("--rerun", action="store_true", help="Enable the rerun viewer (off by default)")
parser.add_argument("--key_timeout", type=float, default=0.35,
                    help="Seconds a key stays 'active' after its last stdin event (coast after release)")
args = parser.parse_args()

NO_ROBOT, NO_LEADER, FPS = args.no_robot, args.no_leader, args.fps


# ---------------- focus-bound keyboard (stdin) ----------------
class FocusKeyboard:
    """Reads keys from this terminal's stdin: only active while the terminal is focused."""

    def __init__(self, active_timeout=0.35):
        self.active_timeout = active_timeout
        self.last_seen = {}
        self.fd = sys.stdin.fileno()
        self._saved = None

    def start(self):
        if not sys.stdin.isatty():
            raise RuntimeError("stdin is not a TTY — run this in an interactive terminal.")
        self._saved = termios.tcgetattr(self.fd)
        new = termios.tcgetattr(self.fd)
        new[3] = new[3] & ~(termios.ICANON | termios.ECHO)  # lflags: no line-buffering, no echo (keep ISIG for Ctrl+C)
        new[6][termios.VMIN] = 0
        new[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, new)
        os.set_blocking(self.fd, False)

    def stop(self):
        if self._saved is not None:
            os.set_blocking(self.fd, True)
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def get_pressed(self):
        now = time.monotonic()
        try:
            data = os.read(self.fd, 4096)
        except (BlockingIOError, OSError):
            data = b""
        for ch in data.decode("utf-8", "ignore"):
            self.last_seen[ch] = now
        return {ch for ch, t in self.last_seen.items() if now - t <= self.active_timeout}


# ---------------- configs ----------------
robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini", robot_model=args.robot_model))
leader = BiSOLeader(BiSOLeaderConfig(
    left_arm_config=SOLeaderConfig(port="/dev/am_arm_leader_left", arm_profile=args.arm_profile),
    right_arm_config=SOLeaderConfig(port="/dev/am_arm_leader_right", arm_profile=args.arm_profile),
    id=args.leader_id,
))
kb = FocusKeyboard(active_timeout=args.key_timeout)

# ---------------- connect (input() prompts happen here, before cbreak) ----------------
if not NO_ROBOT:
    robot.connect()
else:
    print("🧪 NO_ROBOT: robot not connected.")
if not NO_LEADER:
    leader.connect()   # may prompt: ENTER to use each leader's calibration file
else:
    print("🧪 NO_LEADER: leaders not connected.")

log_rerun = None
if args.rerun:
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
    init_rerun(session_name="lekiwi_teleop")
    log_rerun = log_rerun_data

print("\nFocus-aware teleop ready. Keys work ONLY while THIS terminal is focused.")
print("  base: w/s fwd/back  z/x strafe  a/d rotate   speed: r/f   lift: u/j   quit: q\n")

# ---------------- main loop ----------------
kb.start()
frame = 0
try:
    while True:
        t0 = time.perf_counter()

        pressed = kb.get_pressed()
        if "q" in pressed:
            print("\n'q' pressed — quitting.")
            break

        observation = robot.get_observation() if not NO_ROBOT else {}
        arm_actions = leader.get_action() if not NO_LEADER else {}
        arm_actions = {f"arm_{k}": v for k, v in arm_actions.items()}
        base_action = robot._from_keyboard_to_base_action(pressed)
        lift_action = robot._from_keyboard_to_lift_action(pressed)
        action = {**arm_actions, **base_action, **lift_action}

        if log_rerun is not None:
            log_rerun(observation, action)
        if not NO_ROBOT:
            robot.send_action(action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
        loop_dt = time.perf_counter() - t0
        loop_fps = 1.0 / loop_dt if loop_dt > 0 else float("inf")

        frame += 1
        if frame % 5 == 0:
            keys = "".join(sorted(k for k in pressed if k != "q")) or "-"
            warn = "  ⚠️LOW-FPS" if loop_fps < FPS * 0.7 else ""
            sys.stdout.write(f"\rfps={loop_fps:5.1f}  keys=[{keys:<8}]{warn}   ")
            sys.stdout.flush()
except KeyboardInterrupt:
    print("\nCtrl+C — quitting.")
finally:
    kb.stop()
    try:
        if not NO_ROBOT:
            robot.disconnect()
        if not NO_LEADER:
            leader.disconnect()
    except Exception as e:
        print(f"(cleanup warning: {e})")
    print("Teleop stopped, terminal restored.")
