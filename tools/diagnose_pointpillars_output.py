#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser("Diagnose decoded PointPillars JSON boxes")
    ap.add_argument("json_path")
    args = ap.parse_args()
    boxes = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    print(f"boxes={len(boxes)}")
    for i, b in enumerate(boxes):
        warnings = []
        if b.get("z", 0) < -3.0 or b.get("z", 0) > 1.5:
            warnings.append("z_out_of_range")
        if b.get("w", b.get("dx", 0)) > 5 or b.get("l", b.get("dy", 0)) > 8:
            warnings.append("oversized")
        if b.get("score", 0) < 0.1:
            warnings.append("low_score")
        print(f"[{i:02d}] cls={b.get('class_name', b.get('cls_id'))} score={b.get('score'):.3f} warn={','.join(warnings) or '-'} box={b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
