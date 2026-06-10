#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/home/panguofeng/pgf_ai_deploy/mini-kitti-3d")


def frame_name(frame: str) -> str:
    text = str(frame)
    if text.endswith((".bin", ".png", ".txt")):
        text = Path(text).stem
    return f"{int(text):06d}"


def validate_velodyne_bin(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size == 0 or size % 16 != 0:
        raise ValueError(f"invalid KITTI velodyne bin size: {path} ({size} bytes), expected non-empty multiple of 16")
    return size // 16


def copy_one(src: Path, dst: Path, force: bool, dry_run: bool, required: bool = True) -> bool:
    if not src.exists():
        if required:
            raise FileNotFoundError(f"required source not found: {src}")
        print(f"skip missing optional: {src}")
        return False
    print(f"copy: {src} -> {dst}")
    if dry_run:
        return True
    if dst.exists() and not force:
        raise FileExistsError(f"output exists: {dst}; pass --force to overwrite")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract one KITTI-style frame from mini-kitti-3d into repository data/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="mini KITTI dataset root")
    ap.add_argument("--split", default="training", choices=("training", "testing"), help="KITTI split directory")
    ap.add_argument("--frame", default="000000", help="frame id, e.g. 0 or 000134")
    ap.add_argument("--output-bin", default="data/sample.bin", help="output velodyne bin path")
    ap.add_argument("--output-image", default="data/sample.png", help="output image path")
    ap.add_argument("--output-calib", default="data/sample_calib.txt", help="output calibration path")
    ap.add_argument("--output-label", default="data/sample_label.txt", help="output label path")
    ap.add_argument("--no-image", action="store_true", help="do not copy image_2 PNG")
    ap.add_argument("--no-calib", action="store_true", help="do not copy calib TXT")
    ap.add_argument("--no-label", action="store_true", help="do not copy label_2 TXT")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="print actions without copying")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        root = Path(args.dataset_root).expanduser()
        fid = frame_name(args.frame)
        split = args.split

        src_bin = root / split / "velodyne" / f"{fid}.bin"
        src_img = root / split / "image_2" / f"{fid}.png"
        src_calib = root / split / "calib" / f"{fid}.txt"
        src_label = root / split / "label_2" / f"{fid}.txt"

        points = validate_velodyne_bin(src_bin)
        print(f"frame: {fid}, points: {points}")
        copy_one(src_bin, Path(args.output_bin), args.force, args.dry_run, required=True)
        if not args.no_image:
            copy_one(src_img, Path(args.output_image), args.force, args.dry_run, required=True)
        if not args.no_calib:
            copy_one(src_calib, Path(args.output_calib), args.force, args.dry_run, required=True)
        if not args.no_label:
            copy_one(src_label, Path(args.output_label), args.force, args.dry_run, required=(split == "training"))
        if not args.dry_run:
            copied_points = validate_velodyne_bin(Path(args.output_bin))
            print(f"saved velodyne: {args.output_bin} ({copied_points} points)")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())