"""Live wrist-cam view in Rerun for collection sessions.

Shows the raw wrist feed with the red-block detection overlaid:
green box = block readable (area >= threshold, the observability standard),
orange box = visible but small, no box = blind. Also logs the blob area as
a timeline scalar so acquire-then-act discipline is checkable at a glance.

All logging is best-effort: any Rerun/viewer failure must never break a
collection session.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
    import rerun as rr
    _OK = True
except Exception:  # pragma: no cover
    _OK = False


class RerunWristView:
    def __init__(self, cam: str = "wrist", session: str = "so101_collection",
                 vis_threshold: float = 100.0):
        self.cam = cam
        self.th = vis_threshold
        self.ok = _OK
        if not self.ok:
            print("[rerun] rerun/cv2 unavailable — live view disabled")
            return
        try:
            rr.init(session, spawn=True)
            print(f"[rerun] viewer spawned — watching '{cam}' "
                  f"(green box = readable >= {vis_threshold:.0f} px^2)")
        except Exception as e:
            print(f"[rerun] viewer failed to start ({e}) — live view disabled")
            self.ok = False

    def log(self, obs: dict) -> None:
        if not self.ok:
            return
        fr = obs.get(self.cam)
        if not (isinstance(fr, np.ndarray) and fr.ndim == 3):
            return
        try:
            hsv = cv2.cvtColor(np.ascontiguousarray(fr), cv2.COLOR_RGB2HSV)
            m = (cv2.inRange(hsv, (0, 110, 60), (10, 255, 255))
                 | cv2.inRange(hsv, (170, 110, 60), (180, 255, 255)))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area, box = 0.0, None
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area = float(cv2.contourArea(c))
                box = cv2.boundingRect(c)
            img = rr.Image(fr)
            try:
                img = img.compress(jpeg_quality=80)
            except Exception:
                pass
            rr.log("wrist/image", img)
            if box is not None and area >= 20:
                x, y, w, h = box
                color = [0, 230, 0] if area >= self.th else [255, 160, 0]
                rr.log("wrist/image/block",
                       rr.Boxes2D(mins=[[x, y]], sizes=[[w, h]],
                                  colors=[color], labels=[f"{int(area)}"]))
            else:
                rr.log("wrist/image/block", rr.Clear(recursive=False))
            rr.log("plots/block_area", rr.Scalars(area))
            rr.log("plots/readable", rr.Scalars(1.0 if area >= self.th else 0.0))
        except Exception:
            pass
