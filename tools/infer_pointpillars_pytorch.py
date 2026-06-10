#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PointPillars PyTorch inference + source-style GT visualization.

Drop-in replacement for:
    tools/infer_pointpillars_pytorch.py

Only this script and pointpillars/utils/vis_o3d.py need to be replaced.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
sys.path = [str(PROJECT_ROOT)] + [p for p in sys.path if Path(p or ".").resolve() != TOOLS_DIR]

import torch  # noqa: E402

from pointpillars.model import PointPillars  # noqa: E402
from pointpillars.utils.vis_o3d import (  # noqa: E402
    bbox_camera2lidar_src,
    bbox_lidar2camera_src,
    bbox3d2corners_camera_src,
    points_camera2image_src,
    save_bev_image,
    draw_lidar_boxes_on_image,
    draw_camera_boxes_on_image,
)

CLASSES = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}
LABEL2CLASSES = {v: k for k, v in CLASSES.items()}
DEFAULT_POINT_RANGE = np.array([0.0, -39.68, -3.0, 69.12, 39.68, 1.0], dtype=np.float32)
DEFAULT_PCD_LIMIT_RANGE = np.array([0.0, -40.0, -3.0, 70.4, 40.0, 0.0], dtype=np.float32)


def _to_numpy(x: Any, dtype=None) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=np.float32 if dtype is None else dtype)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def read_points(path: str | Path) -> np.ndarray:
    pts = np.fromfile(str(path), dtype=np.float32)
    if pts.size % 4 != 0:
        raise ValueError(f"Invalid KITTI point file: {path}, float_count={pts.size}")
    return pts.reshape(-1, 4)


def _extend_3x4(data: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, :4] = data.reshape(3, 4)
    return out


def _extend_3x3(data: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = data.reshape(3, 3)
    return out


def read_calib(path: str | Path) -> dict[str, np.ndarray]:
    vals: dict[str, np.ndarray] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, rest = line.split(":", 1)
        arr = np.array([float(v) for v in rest.split()], dtype=np.float32)
        if key.startswith("P") and arr.size == 12:
            vals[key] = _extend_3x4(arr)
        elif key == "Tr_velo_to_cam" and arr.size == 12:
            vals[key] = _extend_3x4(arr)
        elif key == "R0_rect" and arr.size == 9:
            vals[key] = _extend_3x3(arr)
        else:
            vals[key] = arr
    for k in ("P2", "Tr_velo_to_cam", "R0_rect"):
        if k not in vals:
            raise KeyError(f"Missing {k} in calib file: {path}")
    return vals


def read_label_source_style(path: str | Path) -> dict[str, np.ndarray]:
    """Read KITTI label_2 exactly for source-style visualization.

    Raw KITTI dimensions are h,w,l.  Source read_label() reorders them to
    l,h,w before building camera boxes.  We parse manually here and never rely
    on a possibly modified local read_label().
    """
    names: list[str] = []
    bbox2d: list[list[float]] = []
    dims_lhw: list[list[float]] = []
    locs: list[list[float]] = []
    rots: list[float] = []

    p = Path(path)
    if not p.exists():
        return {
            "name": np.asarray([], dtype=object),
            "bbox": np.zeros((0, 4), dtype=np.float32),
            "dimensions": np.zeros((0, 3), dtype=np.float32),
            "location": np.zeros((0, 3), dtype=np.float32),
            "rotation_y": np.zeros((0,), dtype=np.float32),
        }

    for raw in p.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 15:
            continue
        names.append(parts[0])
        bbox2d.append([float(v) for v in parts[4:8]])
        h, w, l = [float(v) for v in parts[8:11]]
        dims_lhw.append([l, h, w])
        locs.append([float(v) for v in parts[11:14]])
        rots.append(float(parts[14]))

    return {
        "name": np.asarray(names, dtype=object),
        "bbox": np.asarray(bbox2d, dtype=np.float32),
        "dimensions": np.asarray(dims_lhw, dtype=np.float32),
        "location": np.asarray(locs, dtype=np.float32),
        "rotation_y": np.asarray(rots, dtype=np.float32),
    }


def point_range_filter(points: np.ndarray, point_range: np.ndarray = DEFAULT_POINT_RANGE) -> np.ndarray:
    x1, y1, z1, x2, y2, z2 = [float(v) for v in point_range]
    mask = (
        (points[:, 0] >= x1) & (points[:, 0] < x2) &
        (points[:, 1] >= y1) & (points[:, 1] < y2) &
        (points[:, 2] >= z1) & (points[:, 2] < z2)
    )
    return points[mask]


def load_gt_source_style(gt_path: str | Path | None, calib: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if gt_path is None or not Path(gt_path).exists():
        return np.zeros((0, 7), np.float32), np.zeros((0, 7), np.float32), np.zeros((0,), np.int64)

    ann = read_label_source_style(gt_path)
    names = ann["name"]
    valid = names != "DontCare"

    gt_camera = np.concatenate(
        [
            ann["location"][valid].astype(np.float32),
            ann["dimensions"][valid].astype(np.float32),  # already l,h,w
            ann["rotation_y"][valid, None].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    gt_labels = np.array([CLASSES.get(str(n), 2) for n in names[valid]], dtype=np.int64)
    gt_lidar = bbox_camera2lidar_src(gt_camera, calib["Tr_velo_to_cam"], calib["R0_rect"])
    return gt_camera, gt_lidar, gt_labels


def load_checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    state: dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        nk = str(k)
        if nk.startswith("module."):
            nk = nk[len("module."):]
        state[nk] = v
    return state


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    try:
        return PointPillars(
            nclasses=3,
            score_thr=float(args.score_thr),
            nms_thr=float(args.nms_thr),
            nms_pre=int(args.nms_pre),
            max_num=int(args.max_num),
        )
    except TypeError:
        model = PointPillars(nclasses=3)
        for name, value in (
            ("score_thr", float(args.score_thr)),
            ("nms_thr", float(args.nms_thr)),
            ("nms_pre", int(args.nms_pre)),
            ("max_num", int(args.max_num)),
        ):
            try:
                setattr(model, name, value)
            except Exception:
                pass
        return model


def result_to_numpy(result: dict[str, Any]) -> dict[str, np.ndarray]:
    out = {
        "lidar_bboxes": _to_numpy(result.get("lidar_bboxes", None), np.float32).reshape(-1, 7),
        "labels": _to_numpy(result.get("labels", None), np.int64).reshape(-1),
        "scores": _to_numpy(result.get("scores", None), np.float32).reshape(-1),
    }
    n = len(out["lidar_bboxes"])
    if out["labels"].size < n:
        out["labels"] = np.concatenate([out["labels"], np.full((n - out["labels"].size,), 2, dtype=np.int64)])
    if out["scores"].size < n:
        out["scores"] = np.concatenate([out["scores"], np.zeros((n - out["scores"].size,), dtype=np.float32)])
    out["labels"] = out["labels"][:n]
    out["scores"] = out["scores"][:n]
    return out


def _filter_by_mask(result: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    return {k: v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v for k, v in result.items()}


def apply_image_range_filter(
    result: dict[str, np.ndarray],
    calib: dict[str, np.ndarray],
    image_shape: tuple[int, int],
    min_depth: float = 0.5,
    max_span_ratio: float = 3.0,
) -> dict[str, np.ndarray]:
    """Filter predicted boxes before image visualization.

    Source keep_bbox_from_image_range keeps boxes that intersect the image, but
    for a Python fallback deployment we also need to remove boxes with corners
    behind/too close to the camera; otherwise perspective projection produces
    huge blue rays across the image.
    """
    boxes = result["lidar_bboxes"]
    if len(boxes) == 0:
        return result
    cam = bbox_lidar2camera_src(boxes, calib["Tr_velo_to_cam"], calib["R0_rect"])
    corners = bbox3d2corners_camera_src(cam)
    pts = points_camera2image_src(corners, calib["P2"])
    H, W = image_shape

    z_ok = np.all(corners[:, :, 2] > float(min_depth), axis=1)
    finite = np.all(np.isfinite(pts), axis=(1, 2))
    x1 = np.nanmin(pts[:, :, 0], axis=1)
    y1 = np.nanmin(pts[:, :, 1], axis=1)
    x2 = np.nanmax(pts[:, :, 0], axis=1)
    y2 = np.nanmax(pts[:, :, 1], axis=1)
    intersects = (x2 >= 0) & (x1 < W) & (y2 >= 0) & (y1 < H)
    span_ok = ((x2 - x1) <= float(max_span_ratio) * W) & ((y2 - y1) <= float(max_span_ratio) * H)
    non_degenerate = ((x2 - x1) > 2) & ((y2 - y1) > 2)
    mask = z_ok & finite & intersects & span_ok & non_degenerate
    return _filter_by_mask(result, mask)

def apply_lidar_range_filter(result: dict[str, np.ndarray], limit_range: np.ndarray = DEFAULT_PCD_LIMIT_RANGE) -> dict[str, np.ndarray]:
    boxes = result["lidar_bboxes"]
    if len(boxes) == 0:
        return result
    x1, y1, z1, x2, y2, z2 = [float(v) for v in limit_range]
    centers = boxes[:, :3]
    mask = (
        (centers[:, 0] >= x1) & (centers[:, 0] <= x2) &
        (centers[:, 1] >= y1) & (centers[:, 1] <= y2) &
        (centers[:, 2] >= z1) & (centers[:, 2] <= z2)
    )
    return _filter_by_mask(result, mask)



def _poly_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        s += p[0] * q[1] - q[0] * p[1]
    return abs(s) * 0.5


def _cross(a, b, p) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _clip_poly(subject, clipper):
    def signed_area(poly):
        if len(poly) < 3:
            return 0.0
        s = 0.0
        for i, p in enumerate(poly):
            q = poly[(i + 1) % len(poly)]
            s += p[0] * q[1] - q[0] * p[1]
        return 0.5 * s

    def inside(p, a, b, ccw):
        c = _cross(a, b, p)
        return c >= -1e-9 if ccw else c <= 1e-9

    def intersect(s, e, a, b):
        x1, y1 = s; x2, y2 = e; x3, y3 = a; x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return e
        px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / den
        py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / den
        return (px, py)

    out = list(subject)
    if len(out) < 3 or len(clipper) < 3:
        return []
    ccw = signed_area(clipper) >= 0
    for i, a in enumerate(clipper):
        b = clipper[(i + 1) % len(clipper)]
        inp = out
        out = []
        if not inp:
            break
        prev = inp[-1]
        for cur in inp:
            cur_in = inside(cur, a, b, ccw)
            prev_in = inside(prev, a, b, ccw)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur, a, b))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur, a, b))
            prev = cur
    return out


def _lidar_box_bev_poly(box: np.ndarray) -> list[tuple[float, float]]:
    # lidar box: [x, y, z, w, l, h, yaw]
    x, y, _, w, l, _, yaw = [float(v) for v in box[:7]]
    local = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32)
    corners = local * np.array([w, l], dtype=np.float32)[None, :]
    c, ss = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, ss], [-ss, c]], dtype=np.float32)  # source-style BEV rotation
    corners = corners @ rot.T
    corners += np.array([x, y], dtype=np.float32)[None, :]
    return [(float(px), float(py)) for px, py in corners]


def _bev_iou_lidar(a: np.ndarray, b: np.ndarray) -> float:
    pa = _lidar_box_bev_poly(a)
    pb = _lidar_box_bev_poly(b)
    aa = _poly_area(pa)
    ab = _poly_area(pb)
    if aa <= 0 or ab <= 0:
        return 0.0
    inter_poly = _clip_poly(pa, pb)
    inter = _poly_area(inter_poly)
    return inter / max(aa + ab - inter, 1e-8)


def final_nms_lidar_result(
    result: dict[str, np.ndarray],
    iou_thr: float = 0.01,
    max_num: int = 50,
    class_agnostic: bool = False,
) -> dict[str, np.ndarray]:
    """Extra safety NMS for Python fallback deployments.

    The source model already calls nms_cuda internally.  This post-NMS is kept
    because a pure-Python replacement of iou3d_op can fail to be loaded or can
    return wrong keep indices, leaving many near-identical boxes in the result.
    """
    boxes = result["lidar_bboxes"]
    labels = result["labels"]
    scores = result["scores"]
    if len(boxes) == 0:
        return result

    keep_all: list[int] = []
    classes = [None] if class_agnostic else sorted(set(int(x) for x in labels.tolist()))
    for cls in classes:
        if cls is None:
            idxs = np.arange(len(boxes))
        else:
            idxs = np.where(labels == cls)[0]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        kept: list[int] = []
        for idx in idxs.tolist():
            suppress = False
            for kept_idx in kept:
                if _bev_iou_lidar(boxes[idx], boxes[kept_idx]) > float(iou_thr):
                    suppress = True
                    break
            if not suppress:
                kept.append(idx)
        keep_all.extend(kept)

    keep_all = sorted(keep_all, key=lambda i: float(scores[i]), reverse=True)[:int(max_num)]
    keep = np.asarray(keep_all, dtype=np.int64)
    return {"lidar_bboxes": boxes[keep], "labels": labels[keep], "scores": scores[keep]}

def save_result_json(path: str | Path, result: dict[str, np.ndarray]) -> None:
    boxes, labels, scores = result["lidar_bboxes"], result["labels"], result["scores"]
    records = []
    for i, box in enumerate(boxes):
        cls_id = int(labels[i]) if i < len(labels) else 2
        records.append({
            "x": float(box[0]), "y": float(box[1]), "z": float(box[2]),
            "w": float(box[3]), "l": float(box[4]), "h": float(box[5]), "yaw": float(box[6]),
            "score": float(scores[i]) if i < len(scores) else 0.0,
            "cls_id": cls_id,
            "class_name": LABEL2CLASSES.get(cls_id, str(cls_id)),
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def diagnose(result: dict[str, np.ndarray], gt_camera: np.ndarray, gt_lidar: np.ndarray, args: argparse.Namespace) -> None:
    print("[diagnose] pred_boxes =", len(result["lidar_bboxes"]))
    print("[diagnose] gt_camera =", len(gt_camera), "gt_lidar =", len(gt_lidar))
    print("[diagnose] thresholds =", args.score_thr, args.nms_thr, args.nms_pre, args.max_num)
    for i, b in enumerate(result["lidar_bboxes"][:10]):
        print(f"[pred {i:02d}] cls={int(result['labels'][i])} score={float(result['scores'][i]):.3f} box={np.round(b, 3).tolist()}")
    for i, b in enumerate(gt_camera[:10]):
        print(f"[gt_camera {i:02d}] [x,y,z,l,h,w,ry]={np.round(b, 3).tolist()}")
    for i, b in enumerate(gt_lidar[:10]):
        print(f"[gt_lidar  {i:02d}] [x,y,z,w,l,h,yaw]={np.round(b, 3).tolist()}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        "PointPillars PyTorch inference with source-style GT visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pc-path", required=True)
    ap.add_argument("--calib-path", default=None)
    ap.add_argument("--img-path", default=None)
    ap.add_argument("--gt-path", default=None)
    ap.add_argument("--save-dir", default="outputs/debug_000000")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--nms-thr", type=float, default=0.01)
    ap.add_argument("--nms-pre", type=int, default=100)
    ap.add_argument("--max-num", type=int, default=50)
    ap.add_argument("--save-bev", action="store_true")
    ap.add_argument("--save-image", action="store_true")
    ap.add_argument("--show-gt", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--no-image-range-filter", action="store_true")
    ap.add_argument("--no-lidar-range-filter", action="store_true")
    ap.add_argument("--no-final-nms", action="store_true", help="Disable extra post-NMS used for Python fallback debugging")
    ap.add_argument("--min-camera-depth", type=float, default=0.5, help="Drop boxes whose camera-space corners are too close/behind camera before image drawing")
    ap.add_argument("--max-proj-span-ratio", type=float, default=3.0, help="Drop boxes whose projected width/height exceeds this multiple of the image size")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    points = read_points(args.pc_path)
    pc_filtered = point_range_filter(points)

    model = build_model(args).to(device)
    state = load_checkpoint_state(args.ckpt)
    missing, unexpected = model.load_state_dict(state, strict=bool(args.strict))
    if missing:
        print("[load_state_dict] missing keys:", len(missing))
    if unexpected:
        print("[load_state_dict] unexpected keys:", len(unexpected))
    model.eval()

    with torch.no_grad():
        raw = model(batched_pts=[torch.from_numpy(pc_filtered).float().to(device)], mode="test")[0]

    result = result_to_numpy(raw)

    calib = read_calib(args.calib_path) if args.calib_path else None
    image = cv2.imread(str(args.img_path)) if args.img_path else None
    if args.img_path and image is None:
        raise FileNotFoundError(f"Failed to read image: {args.img_path}")

    if calib is not None and image is not None and not args.no_image_range_filter:
        result = apply_image_range_filter(
            result,
            calib,
            image.shape[:2],
            min_depth=args.min_camera_depth,
            max_span_ratio=args.max_proj_span_ratio,
        )
    if not args.no_lidar_range_filter:
        result = apply_lidar_range_filter(result)

    gt_camera = np.zeros((0, 7), dtype=np.float32)
    gt_lidar = np.zeros((0, 7), dtype=np.float32)
    gt_labels = np.zeros((0,), dtype=np.int64)
    if calib is not None and args.gt_path:
        gt_camera, gt_lidar, gt_labels = load_gt_source_style(args.gt_path, calib)

    if not args.no_final_nms:
        before_nms = len(result["lidar_bboxes"])
        result = final_nms_lidar_result(result, iou_thr=args.nms_thr, max_num=args.max_num)
        after_nms = len(result["lidar_bboxes"])
        if args.diagnose:
            print(f"[final_nms] {before_nms} -> {after_nms} boxes, iou_thr={args.nms_thr}")

    json_path = save_dir / "detections_pytorch_refactor.json"
    save_result_json(json_path, result)
    print(f"[saved] {json_path}")

    if args.save_bev:
        bev_path = save_dir / "bev_pytorch_refactor.png"
        save_bev_image(
            pc_filtered,
            pred_bboxes=result["lidar_bboxes"],
            pred_labels=result["labels"],
            pred_scores=result["scores"],
            out_path=bev_path,
            gt_bboxes=gt_lidar if args.show_gt else None,
        )
        print(f"[saved] {bev_path}")

    if args.save_image:
        if image is None or calib is None:
            print("[warn] --save-image requires --img-path and --calib-path; skip")
        else:
            img_draw = image.copy()
            img_draw = draw_lidar_boxes_on_image(
                img_draw,
                result["lidar_bboxes"],
                result["labels"],
                calib["Tr_velo_to_cam"],
                calib["R0_rect"],
                calib["P2"],
                scores=result["scores"],
                prefix="Pred",
                min_depth=args.min_camera_depth,
                max_span_ratio=args.max_proj_span_ratio,
            )
            if args.show_gt and len(gt_camera) > 0:
                img_draw = draw_camera_boxes_on_image(
                    img_draw,
                    gt_camera,
                    gt_labels,
                    calib["P2"],
                    prefix="GT",
                )
            img_path = save_dir / "img_3dbox_pytorch_refactor.png"
            cv2.imwrite(str(img_path), img_draw)
            print(f"[saved] {img_path}")

    if args.diagnose:
        diagnose(result, gt_camera, gt_lidar, args)
        print("[path] pointpillars root:", __import__("pointpillars").__file__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
