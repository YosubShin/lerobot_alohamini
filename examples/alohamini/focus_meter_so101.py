#!/usr/bin/env python3
"""Live sharpness meter for focusing the wrist fisheye. Ctrl-C to stop.

Put a textured target (printed text / checkerboard / the block) ~20 cm from
the lens and rotate the barrel until the number peaks; then aim at something
>1 m away and confirm it stays high. Sharp scenes read in the hundreds.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
from so101_zmq_client import SO101ZmqClient, SO101ZmqClientConfig

r = SO101ZmqClient(SO101ZmqClientConfig(remote_ip=sys.argv[1] if len(sys.argv) > 1 else "192.168.0.50"))
r.connect()
best = 0.0
try:
    while True:
        o = r.get_observation()
        g = cv2.cvtColor(o["wrist"], cv2.COLOR_RGB2GRAY)
        s = cv2.Laplacian(g, cv2.CV_64F).var()
        best = max(best, s)
        print(f"\rsharpness {s:7.1f}   best {best:7.1f}   {'#' * min(60, int(s / 8))}".ljust(90),
              end="", flush=True)
        time.sleep(0.25)
except KeyboardInterrupt:
    print()
finally:
    r.disconnect()
