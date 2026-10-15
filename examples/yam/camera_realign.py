#!/usr/bin/env python3
"""Put the forward camera back where it was.

Shows the live forward view blended with a reference frame (a grid scene
reference by default) in Rerun, and prints how far each workspace tile
(36h11 ids 2-5, taped to the table) sits from where it was in the reference.
Move the camera until every tile is within ~2 px, then ENTER.

  python camera_realign.py                       # ref = grid_final/scene1_ref.png
  python camera_realign.py --ref path/to.png
"""
from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_yam import FORWARD_DEV, PILOT, CamReader  # noqa: E402

TILES = (2, 3, 4, 5)


def tile_centres(gray, det):
    c, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): cc[0].mean(0) for cc, i in zip(c, ids.ravel()) if int(i) in TILES}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ref", default=str(PILOT / "calib/forward_ref_20261012.png"), help="reference frame; the tiles in it define where the camera belongs relative to the robot")
    p.add_argument("--tol", type=float, default=2.0, help="px: print OK when every tile is within this")
    args = p.parse_args()

    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                  cv2.aruco.DetectorParameters())
    ref = cv2.imread(args.ref)
    ref_t = tile_centres(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), det)
    print(f"reference {Path(args.ref).name}: tiles {sorted(ref_t)}")
    if not ref_t:
        sys.exit("no tiles in the reference")

    import rerun as rr
    import rerun.blueprint as rrb
    rr.init("yam_camera_realign", spawn=True)
    rr.send_blueprint(rrb.Blueprint(rrb.Grid(
        rrb.Spatial2DView(origin="realign/blend", name="blend (live + reference)"),
        rrb.Spatial2DView(origin="realign/misalignment", name="misalignment"),
        grid_columns=2), collapse_panels=True), make_active=True)

    fwd = CamReader(FORWARD_DEV, 1920, 1080, 25)
    print("move the camera until the ghost disappears and every tile reads ~0 px; ENTER to finish")
    last = 0.0
    try:
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline(); break
            live = fwd.frame
            blend = cv2.addWeighted(live, 0.5, ref, 0.5, 0)
            cur = tile_centres(cv2.cvtColor(live, cv2.COLOR_BGR2GRAY), det)
            offs = {k: cur[k] - ref_t[k] for k in cur if k in ref_t}
            for k, (x, y) in ref_t.items():
                cv2.circle(blend, (int(x), int(y)), 40, (0, 220, 255), 2)        # where it should be
                if k in cur:
                    cv2.arrowedLine(blend, (int(cur[k][0]), int(cur[k][1])), (int(x), int(y)),
                                    (0, 255, 0), 3, tipLength=0.2)                  # live -> target
            rr.log("realign/blend", rr.Image(blend[:, :, ::-1]).compress(jpeg_quality=70))
            rr.log("realign/misalignment", rr.Image(cv2.absdiff(live, ref).max(axis=2)).compress(jpeg_quality=70))
            if time.time() - last > 0.5:
                txt = "  ".join(f"tile{k}: {v[0]:+.1f},{v[1]:+.1f}" for k, v in sorted(offs.items()))
                worst = max((float(np.hypot(*v)) for v in offs.values()), default=float("nan"))
                ok = offs and worst <= args.tol and len(offs) >= 2
                print(f"\r{txt or 'no tiles seen'}   worst {worst:.1f} px {'OK' if ok else ''}      ",
                      end="", flush=True)
                last = time.time()
            time.sleep(0.05)
    finally:
        print()
        fwd.stop()


if __name__ == "__main__":
    main()
