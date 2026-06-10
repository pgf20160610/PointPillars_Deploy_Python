#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from pointpillars.utils.bev_ref import load_detections_json, save_bev_image_ref


def main() -> int:
    ap = argparse.ArgumentParser("Replot BEV with zhulf0804/PointPillars-compatible yaw convention")
    ap.add_argument("--pc-path", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--output", default="outputs/bev_angle_fixed.png")
    ap.add_argument("--scale", type=float, default=10.0)
    args = ap.parse_args()

    points = np.fromfile(args.pc_path, dtype=np.float32).reshape(-1, 4)
    boxes = load_detections_json(args.detections)
    save_bev_image_ref(points, boxes, args.output, scale=args.scale)
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
