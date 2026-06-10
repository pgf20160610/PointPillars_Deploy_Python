#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX/MNN inference for split PointPillars models.

Incremental ONNX-side fix only:
  1) do not modify shared pointpillars utils / visualization code;
  2) decode Backbone/Head outputs with the same anchor order and postprocess
     sequence as zhulf0804/PointPillars PointPillars.get_predicted_bboxes_single();
  3) add ONNX-local projection sanity filtering before calling shared visualizers,
     preventing invalid boxes from producing long blue/red rays on image.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
sys.path = [str(PROJECT_ROOT)] + [p for p in sys.path if Path(p or ".").resolve() != TOOLS_DIR]

from pointpillars.utils.io import read_calib, read_points
from pointpillars.utils.process import (
    keep_bbox_from_image_range,
    keep_bbox_from_lidar_range,
    point_range_filter,
    save_result_json,
)

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover
    ort = None
    ort_error = exc
else:
    ort_error = None

try:
    import MNN
except Exception:
    MNN = None

LABEL2CLASSES = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}
CLASSES = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}

# Source repository anchor definition, ordered by class:
#   0 Pedestrian, 1 Cyclist, 2 Car
# Format: [w, l, h]
SOURCE_ANCHOR_RANGES = [
    [0.0, -39.68, -0.60, 69.12, 39.68, -0.60],
    [0.0, -39.68, -0.60, 69.12, 39.68, -0.60],
    [0.0, -39.68, -1.78, 69.12, 39.68, -1.78],
]
SOURCE_ANCHOR_SIZES = [
    [0.6, 0.8, 1.73],   # Pedestrian
    [0.6, 1.76, 1.73],  # Cyclist
    [1.6, 3.9, 1.56],   # Car
]
SOURCE_ANCHOR_ROTATIONS = [0.0, 1.57]
PCD_LIMIT_RANGE = np.asarray([0.0, -40.0, -3.0, 70.4, 40.0, 0.0], dtype=np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def limit_period_np(val: np.ndarray, offset: float = 0.5, period: float = np.pi) -> np.ndarray:
    return val - np.floor(val / period + offset) * period


def build_pillars(
    points: np.ndarray,
    point_cloud_range: np.ndarray,
    voxel_size: np.ndarray,
    grid_size: np.ndarray,
    max_pillars: int,
    max_points_per_pillar: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build split-export PFN inputs.

    The real split PFN exported by tools/export_pointpillars_split.py reuses the
    trained source `pillar_encoder` conv/bn.  In the PyTorch reference forward,
    the encoder first builds 9-D features and then overwrites feature channels
    0/1 with x/y pillar-center offsets before the PFN conv:

      [x_offset_to_center, y_offset_to_center, z, intensity,
       cluster_x, cluster_y, cluster_z, center_x, center_y]

    The exported PFN accepts a 10-D deployment tensor for C++ compatibility, but
    drops channel 9 (`center_z`) internally.  Therefore channels 0/1 must already
    be the overwritten source-style offsets here. Passing raw x/y keeps BEV box
    sizes plausible but corrupts spatial features and causes Y/position errors.
    """
    points = np.asarray(points, dtype=np.float32)
    point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    grid_size = np.asarray(grid_size, dtype=np.int64)

    pillar_features = np.zeros((max_pillars, max_points_per_pillar, 10), dtype=np.float32)
    pillar_mask = np.zeros((max_pillars, max_points_per_pillar, 1), dtype=np.float32)
    coords = np.zeros((max_pillars, 4), dtype=np.int64)
    num_points = np.zeros((max_pillars,), dtype=np.int64)

    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range.tolist()
    vx, vy, vz = voxel_size.tolist()
    grid_x = int(grid_size[0])
    pillar_map: dict[int, int] = {}
    valid_pillars = 0

    for p in points:
        x, y, z, intensity = p.tolist()
        if x < x_min or x >= x_max or y < y_min or y >= y_max or z < z_min or z >= z_max:
            continue
        x_idx = int(np.floor((x - x_min) / vx))
        y_idx = int(np.floor((y - y_min) / vy))
        key = y_idx * grid_x + x_idx
        pillar_id = pillar_map.get(key, -1)
        if pillar_id == -1:
            if valid_pillars >= max_pillars:
                continue
            pillar_id = valid_pillars
            pillar_map[key] = pillar_id
            coords[pillar_id, 0] = 0
            coords[pillar_id, 1] = 0
            coords[pillar_id, 2] = y_idx
            coords[pillar_id, 3] = x_idx
            valid_pillars += 1

        point_id = int(num_points[pillar_id])
        if point_id >= max_points_per_pillar:
            continue
        pillar_features[pillar_id, point_id, :4] = [x, y, z, intensity]
        pillar_mask[pillar_id, point_id, 0] = 1.0
        num_points[pillar_id] += 1

    for pillar_id in range(valid_pillars):
        n = int(num_points[pillar_id])
        if n == 0:
            continue
        points_feat = pillar_features[pillar_id, :n, :3]
        centroid = points_feat.mean(axis=0)
        x_idx = int(coords[pillar_id, 3])
        y_idx = int(coords[pillar_id, 2])
        center_x = x_min + (float(x_idx) + 0.5) * vx
        center_y = y_min + (float(y_idx) + 0.5) * vy
        center_z = z_min + 0.5 * vz
        pillar_features[pillar_id, :n, 4] = points_feat[:, 0] - centroid[0]
        pillar_features[pillar_id, :n, 5] = points_feat[:, 1] - centroid[1]
        pillar_features[pillar_id, :n, 6] = points_feat[:, 2] - centroid[2]
        pillar_features[pillar_id, :n, 7] = points_feat[:, 0] - center_x
        pillar_features[pillar_id, :n, 8] = points_feat[:, 1] - center_y
        pillar_features[pillar_id, :n, 9] = points_feat[:, 2] - center_z
        # Source PillarEncoder overwrites raw x/y with pillar-center offsets
        # before applying the trained PFN conv. Keep z/intensity unchanged.
        pillar_features[pillar_id, :n, 0] = pillar_features[pillar_id, :n, 7]
        pillar_features[pillar_id, :n, 1] = pillar_features[pillar_id, :n, 8]

    return pillar_features, pillar_mask, coords, num_points, valid_pillars


def read_label_source_style(path: str | Path) -> dict[str, np.ndarray]:
    """Read KITTI labels exactly as source-style camera box expects.

    KITTI raw dimensions are h,w,l. Source read_label() returns dimensions as
    l,h,w, so camera boxes become [x,y,z,l,h,w,ry].
    """
    p = Path(path)
    if not p.exists():
        return {
            "name": np.asarray([], dtype=object),
            "bbox": np.zeros((0, 4), dtype=np.float32),
            "dimensions": np.zeros((0, 3), dtype=np.float32),
            "location": np.zeros((0, 3), dtype=np.float32),
            "rotation_y": np.zeros((0,), dtype=np.float32),
        }

    names: list[str] = []
    bbox2d: list[list[float]] = []
    dims_lhw: list[list[float]] = []
    locs: list[list[float]] = []
    rots: list[float] = []
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


def load_gt_source_style(gt_path: str | Path | None, calib: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if gt_path is None or not Path(gt_path).exists():
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    ann = read_label_source_style(gt_path)
    names = ann["name"]
    valid = names != "DontCare"
    camera_gt = np.concatenate(
        [
            ann["location"][valid].astype(np.float32),
            ann["dimensions"][valid].astype(np.float32),
            ann["rotation_y"][valid, None].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    gt_labels = np.array([CLASSES.get(str(n), 2) for n in names[valid]], dtype=np.int64)
    try:
        from pointpillars.utils.vis_o3d import bbox_camera2lidar_src
    except Exception:
        from pointpillars.utils.process import bbox_camera2lidar as bbox_camera2lidar_src
    gt_lidar = bbox_camera2lidar_src(camera_gt, calib["Tr_velo_to_cam"], calib["R0_rect"])
    return camera_gt, gt_lidar, gt_labels


def summarize_labels(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = {name: 0 for name in LABEL2CLASSES.values()}
    for cls_id in labels:
        name = LABEL2CLASSES.get(int(cls_id), str(cls_id))
        counts[name] = counts.get(name, 0) + 1
    return counts


def scatter_bev(
    bev_shape: tuple[int, int, int, int],
    pillar_embed: np.ndarray,
    coords: np.ndarray,
    valid_pillars: int,
    bev_y_reverse: bool,
) -> np.ndarray:
    """Scatter PFN embeddings to BEV feature map.

    Source PillarEncoder writes canvas[x_idx, y_idx] and then permutes to
    (C, y, x), so the source-style default is bev_y_reverse=False.
    Keep --bev-y-reverse only as an escape hatch for older exports.
    """
    bev = np.zeros(bev_shape, dtype=np.float32)
    _, _, H, W = bev_shape
    for i in range(valid_pillars):
        y = int(coords[i, 2])
        x = int(coords[i, 3])
        if bev_y_reverse:
            y = H - 1 - y
        if 0 <= x < W and 0 <= y < H:
            bev[0, :, y, x] = pillar_embed[0, i, :]
    return bev


def make_source_anchors(feature_h: int, feature_w: int) -> np.ndarray:
    """Generate anchors matching source Anchors.get_multi_anchors().

    Returns shape (H, W, 3, 2, 7), ordered as class then rotation.
    """
    all_cls: list[np.ndarray] = []
    rotations = np.asarray(SOURCE_ANCHOR_ROTATIONS, dtype=np.float32)
    for anchor_range, anchor_size in zip(SOURCE_ANCHOR_RANGES, SOURCE_ANCHOR_SIZES):
        x1, y1, z1, x2, y2, z2 = [float(v) for v in anchor_range]
        # Match pointpillars/model/anchors.py exactly.  The verified PyTorch
        # path uses torch.linspace(min, max, W/H), not half-cell shifted centers.
        # A previous ONNX-local implementation used feature_w+1/feature_h+1 and
        # added 0.5 step, which keeps box sizes correct but shifts BEV box
        # locations (most visibly along the y direction).
        x_centers = np.linspace(x1, x2, feature_w, dtype=np.float32)
        y_centers = np.linspace(y1, y2, feature_h, dtype=np.float32)
        z_value = np.float32(z1)

        # Source shape after permute: (H=y, W=x, R, 7)
        yy, xx, rr = np.meshgrid(y_centers, x_centers, rotations, indexing="ij")
        anchors = np.zeros((feature_h, feature_w, len(rotations), 7), dtype=np.float32)
        anchors[..., 0] = xx
        anchors[..., 1] = yy
        anchors[..., 2] = z_value
        anchors[..., 3:6] = np.asarray(anchor_size, dtype=np.float32)
        anchors[..., 6] = rr
        all_cls.append(anchors[:, :, None, :, :])
    return np.concatenate(all_cls, axis=2).astype(np.float32)


def anchors2bboxes_np(anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Source-equivalent anchors2bboxes for [x,y,z,w,l,h,theta]."""
    anchors = np.asarray(anchors, dtype=np.float32).reshape(-1, 7)
    deltas = np.asarray(deltas, dtype=np.float32).reshape(-1, 7)
    da = np.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
    x = deltas[:, 0] * da + anchors[:, 0]
    y = deltas[:, 1] * da + anchors[:, 1]
    z = deltas[:, 2] * anchors[:, 5] + anchors[:, 2] + anchors[:, 5] * 0.5
    w = anchors[:, 3] * np.exp(np.minimum(deltas[:, 3], 5.0))
    l = anchors[:, 4] * np.exp(np.minimum(deltas[:, 4], 5.0))
    h = anchors[:, 5] * np.exp(np.minimum(deltas[:, 5], 5.0))
    z = z - h * 0.5
    theta = anchors[:, 6] + deltas[:, 6]
    return np.stack([x, y, z, w, l, h, theta], axis=1).astype(np.float32)


# ---------- rotated NMS local to ONNX script ----------
Point = tuple[float, float]


def _rect_corners_from_nms_box(box: np.ndarray) -> list[Point]:
    x1, y1, x2, y2, yaw = [float(v) for v in box[:5]]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = max(0.0, x2 - x1)
    l = max(0.0, y2 - y1)
    c = math.cos(yaw)
    s = math.sin(yaw)
    local = [(-w / 2.0, -l / 2.0), (-w / 2.0, l / 2.0), (w / 2.0, l / 2.0), (w / 2.0, -l / 2.0)]
    return [(cx + dx * c - dy * s, cy + dx * s + dy * c) for dx, dy in local]


def _polygon_area(poly: list[Point]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        area += p[0] * q[1] - q[0] * p[1]
    return abs(area) * 0.5


def _is_ccw(poly: list[Point]) -> bool:
    signed = 0.0
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        signed += p[0] * q[1] - q[0] * p[1]
    return signed > 0


def _inside(p: Point, a: Point, b: Point, ccw: bool) -> bool:
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return cross >= -1e-8 if ccw else cross <= 1e-8


def _intersect(p1: Point, p2: Point, a: Point, b: Point) -> Point:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-10:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)


def _clip_polygon(subject: list[Point], clipper: list[Point]) -> list[Point]:
    output = subject[:]
    ccw = _is_ccw(clipper)
    for i, a in enumerate(clipper):
        b = clipper[(i + 1) % len(clipper)]
        input_poly = output
        output = []
        if not input_poly:
            break
        s = input_poly[-1]
        for e in input_poly:
            if _inside(e, a, b, ccw):
                if not _inside(s, a, b, ccw):
                    output.append(_intersect(s, e, a, b))
                output.append(e)
            elif _inside(s, a, b, ccw):
                output.append(_intersect(s, e, a, b))
            s = e
    return output


def rotated_iou_2d(a: np.ndarray, b: np.ndarray) -> float:
    pa = _rect_corners_from_nms_box(a)
    pb = _rect_corners_from_nms_box(b)
    aa = _polygon_area(pa)
    ab = _polygon_area(pb)
    if aa <= 0 or ab <= 0:
        return 0.0
    inter = _polygon_area(_clip_polygon(pa, pb))
    return inter / max(aa + ab - inter, 1e-8)


def rotated_nms_np(boxes2d: np.ndarray, scores: np.ndarray, thresh: float) -> np.ndarray:
    if len(boxes2d) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    suppressed = np.zeros((len(order),), dtype=bool)
    for pos, idx in enumerate(order):
        if suppressed[pos]:
            continue
        keep.append(int(idx))
        for pos_j in range(pos + 1, len(order)):
            if suppressed[pos_j]:
                continue
            j = order[pos_j]
            if rotated_iou_2d(boxes2d[idx], boxes2d[j]) > thresh:
                suppressed[pos_j] = True
    return np.asarray(keep, dtype=np.int64)


def decode_head_output_source_style(
    cls_preds: np.ndarray,
    box_preds: np.ndarray,
    dir_preds: Optional[np.ndarray],
    score_threshold: float,
    nms_threshold: float,
    nms_pre: int,
    max_detections: int,
) -> dict[str, np.ndarray]:
    """Source-equivalent PointPillars.get_predicted_bboxes_single() in NumPy."""
    cls_preds = np.asarray(cls_preds, dtype=np.float32)
    box_preds = np.asarray(box_preds, dtype=np.float32)
    if dir_preds is not None:
        dir_preds = np.asarray(dir_preds, dtype=np.float32)

    if cls_preds.ndim != 3 or box_preds.ndim != 3:
        raise ValueError(f"expected CHW outputs, got cls={cls_preds.shape}, box={box_preds.shape}")
    _, H, W = cls_preds.shape
    nclasses = 3

    cls_flat = np.transpose(cls_preds, (1, 2, 0)).reshape(-1, nclasses)
    box_flat = np.transpose(box_preds, (1, 2, 0)).reshape(-1, 7)
    if dir_preds is not None:
        dir_flat = np.transpose(dir_preds, (1, 2, 0)).reshape(-1, 2)
        dir_cls = np.argmax(dir_flat, axis=1).astype(np.int64)
    else:
        dir_cls = np.zeros((box_flat.shape[0],), dtype=np.int64)

    anchors = make_source_anchors(H, W).reshape(-1, 7)
    if anchors.shape[0] != box_flat.shape[0] or anchors.shape[0] != cls_flat.shape[0]:
        raise RuntimeError(
            f"anchor/output size mismatch: anchors={anchors.shape}, cls={cls_flat.shape}, box={box_flat.shape}; "
            "check ONNX head output layout."
        )

    cls_scores = sigmoid(cls_flat)
    max_scores = cls_scores.max(axis=1)
    k = min(int(nms_pre), max_scores.shape[0]) if nms_pre > 0 else max_scores.shape[0]
    pre_inds = np.argsort(max_scores)[-k:][::-1]

    cls_scores = cls_scores[pre_inds]
    box_flat = box_flat[pre_inds]
    dir_cls = dir_cls[pre_inds]
    anchors = anchors[pre_inds]

    decoded = anchors2bboxes_np(anchors, box_flat)
    boxes2d = np.concatenate(
        [decoded[:, [0, 1]] - decoded[:, [3, 4]] * 0.5, decoded[:, [0, 1]] + decoded[:, [3, 4]] * 0.5, decoded[:, 6:7]],
        axis=1,
    ).astype(np.float32)

    ret_boxes: list[np.ndarray] = []
    ret_labels: list[np.ndarray] = []
    ret_scores: list[np.ndarray] = []
    for cls_id in range(nclasses):
        scores_i = cls_scores[:, cls_id]
        valid = scores_i > float(score_threshold)
        if not np.any(valid):
            continue
        cur_scores = scores_i[valid]
        cur_boxes = decoded[valid].copy()
        cur_boxes2d = boxes2d[valid]
        cur_dir = dir_cls[valid]
        keep = rotated_nms_np(cur_boxes2d, cur_scores, float(nms_threshold))
        if keep.size == 0:
            continue
        cur_scores = cur_scores[keep]
        cur_boxes = cur_boxes[keep]
        cur_dir = cur_dir[keep]

        # Source direction correction:
        #   yaw = limit_period(yaw, 1, pi) + (1 - dir_cls) * pi
        cur_boxes[:, 6] = limit_period_np(cur_boxes[:, 6], offset=1.0, period=np.pi)
        cur_boxes[:, 6] += (1 - cur_dir.astype(np.float32)) * np.pi

        ret_boxes.append(cur_boxes.astype(np.float32))
        ret_labels.append(np.full((cur_boxes.shape[0],), cls_id, dtype=np.int64))
        ret_scores.append(cur_scores.astype(np.float32))

    if not ret_boxes:
        return {
            "lidar_bboxes": np.zeros((0, 7), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "scores": np.zeros((0,), dtype=np.float32),
        }

    boxes = np.concatenate(ret_boxes, axis=0)
    labels = np.concatenate(ret_labels, axis=0)
    scores = np.concatenate(ret_scores, axis=0)
    if boxes.shape[0] > int(max_detections):
        keep_final = np.argsort(scores)[-int(max_detections):][::-1]
        boxes = boxes[keep_final]
        labels = labels[keep_final]
        scores = scores[keep_final]
    return {"lidar_bboxes": boxes, "labels": labels, "scores": scores}


def filter_projection_sane_result(
    result: dict[str, np.ndarray],
    calib: dict[str, np.ndarray] | None,
    image_shape: tuple[int, int] | None,
    min_depth: float,
    max_proj_span_ratio: float,
) -> dict[str, np.ndarray]:
    """ONNX-local visualization safety filter.

    Does not alter shared visualization code. It removes boxes whose corners are
    behind/too close to the camera or whose projected span is implausibly huge.
    This prevents long ray-like lines in camera visualization.
    """
    boxes = np.asarray(result.get("lidar_bboxes", []), dtype=np.float32).reshape(-1, 7)
    if boxes.shape[0] == 0 or calib is None or image_shape is None:
        return result
    try:
        from pointpillars.utils.process import bbox_lidar2camera, bbox3d2corners_camera, points_camera2image
    except Exception:
        return result

    labels = np.asarray(result.get("labels", []), dtype=np.int64).reshape(-1)
    scores = np.asarray(result.get("scores", []), dtype=np.float32).reshape(-1)
    H, W = int(image_shape[0]), int(image_shape[1])
    P2 = np.asarray(calib["P2"], dtype=np.float32)
    if P2.shape == (3, 4):
        P2_4 = np.eye(4, dtype=np.float32)
        P2_4[:3, :4] = P2
        P2 = P2_4

    cam_boxes = bbox_lidar2camera(boxes, calib["Tr_velo_to_cam"], calib["R0_rect"])
    corners_cam = bbox3d2corners_camera(cam_boxes)
    keep = []
    for i in range(boxes.shape[0]):
        depths = corners_cam[i, :, 2]
        if not np.all(np.isfinite(depths)) or np.any(depths <= float(min_depth)):
            continue
        pts = points_camera2image(corners_cam[i : i + 1], P2)[0]
        if not np.all(np.isfinite(pts)):
            continue
        span_x = float(pts[:, 0].max() - pts[:, 0].min())
        span_y = float(pts[:, 1].max() - pts[:, 1].min())
        if span_x > max_proj_span_ratio * W or span_y > max_proj_span_ratio * H:
            continue
        keep.append(i)
    keep_arr = np.asarray(keep, dtype=np.int64)
    return {
        "lidar_bboxes": boxes[keep_arr],
        "labels": labels[keep_arr] if labels.size else labels,
        "scores": scores[keep_arr] if scores.size else scores,
    }


def visualize_results(
    points: np.ndarray,
    result: dict[str, np.ndarray],
    calib: dict[str, np.ndarray] | None,
    img: np.ndarray | None,
    gt_path: str | Path | None,
    save_dir: Path,
    save_bev: bool,
    save_image: bool,
    min_camera_depth: float,
    max_proj_span_ratio: float,
) -> None:
    try:
        from pointpillars.utils.vis_o3d import (
            draw_camera_boxes_on_image,
            draw_lidar_boxes_on_image,
            save_bev_image,
        )
    except Exception as exc:
        print(f"[WARN] visualization disabled: {exc}")
        return

    pred_bboxes = np.asarray(result["lidar_bboxes"], dtype=np.float32).reshape(-1, 7)
    pred_labels = np.asarray(result["labels"], dtype=np.int64).reshape(-1)
    pred_scores = np.asarray(result["scores"], dtype=np.float32).reshape(-1)

    if save_bev:
        bev_path = save_dir / "vis_bev.png"
        gt_lidar = np.zeros((0, 7), dtype=np.float32)
        if gt_path and calib is not None and Path(gt_path).exists():
            _, gt_lidar, _ = load_gt_source_style(gt_path, calib)
        save_bev_image(
            points,
            pred_bboxes=pred_bboxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            out_path=str(bev_path),
            gt_bboxes=gt_lidar,
        )
        print(f"[OK] BEV visualization saved to {bev_path}")

    if save_image and img is not None and calib is not None:
        import cv2

        safe = filter_projection_sane_result(
            result,
            calib=calib,
            image_shape=img.shape[:2],
            min_depth=min_camera_depth,
            max_proj_span_ratio=max_proj_span_ratio,
        )
        if len(safe["scores"]) != len(result["scores"]):
            print(f"[visualize_filter] image boxes {len(result['scores'])} -> {len(safe['scores'])}")
        pred_bboxes = np.asarray(safe["lidar_bboxes"], dtype=np.float32).reshape(-1, 7)
        pred_labels = np.asarray(safe["labels"], dtype=np.int64).reshape(-1)
        pred_scores = np.asarray(safe["scores"], dtype=np.float32).reshape(-1)

        gt_camera = np.zeros((0, 7), dtype=np.float32)
        gt_labels = np.zeros((0,), dtype=np.int64)
        if gt_path and Path(gt_path).exists():
            gt_camera, _, gt_labels = load_gt_source_style(gt_path, calib)
        image_path = save_dir / "vis_image.png"
        image_pred = draw_lidar_boxes_on_image(
            img,
            pred_bboxes,
            pred_labels,
            calib["Tr_velo_to_cam"],
            calib["R0_rect"],
            calib["P2"],
            scores=pred_scores,
            prefix="Pred",
        )
        if len(gt_camera):
            image_pred = draw_camera_boxes_on_image(image_pred, gt_camera, gt_labels, calib["P2"], prefix="GT")
        cv2.imwrite(str(image_path), image_pred)
        print(f"[OK] image visualization saved to {image_path}")


class OnnxEngine:
    def __init__(self, model_path: Path, providers: list[str]):
        if ort is None:
            raise RuntimeError(f"onnxruntime is required for ONNX inference: {ort_error}")
        self.session = ort.InferenceSession(str(model_path), providers=providers)

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        outputs = self.session.run(None, inputs)
        return [np.asarray(output, dtype=np.float32) for output in outputs]


class MnnEngine:
    def __init__(self, model_path: Path, device: str, num_threads: int):
        if MNN is None:
            raise RuntimeError("MNN python package is not installed")
        self.interpreter = MNN.Interpreter(str(model_path))
        if device != "cpu":
            print(f"[WARN] MNN Python backend uses default session settings; requested device='{device}' may not be honored.")
        try:
            self.session = self.interpreter.createSession()
        except Exception as exc:
            raise RuntimeError(f"Failed to create MNN session: {exc}")
        self.num_threads = num_threads

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        for name, data in inputs.items():
            input_tensor = self.interpreter.getSessionInput(self.session, name)
            if input_tensor is None:
                raise RuntimeError(f"MNN input '{name}' not found in session")
            input_tensor.resize(data.shape)
        self.interpreter.resizeSession(self.session)
        for name, data in inputs.items():
            input_tensor = self.interpreter.getSessionInput(self.session, name)
            host_tensor = MNN.Tensor.fromNumpy(np.asarray(data, dtype=np.float32))
            input_tensor.copyFromHostTensor(host_tensor)
        self.interpreter.runSession(self.session)
        outputs: list[np.ndarray] = []
        for out in self.interpreter.getSessionOutputAll(self.session):
            outputs.append(np.asarray(out.getNumpyData(), dtype=np.float32).reshape(out.getShape()))
        return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX/MNN inference for split PointPillars models")
    parser.add_argument("--pc-path", required=True)
    parser.add_argument("--calib-path", default="")
    parser.add_argument("--img-path", default="")
    parser.add_argument("--gt-path", default="")
    parser.add_argument("--pfn", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--backend", choices=["onnx", "mnn"], default="onnx")
    parser.add_argument("--runtime-device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--score-thr", type=float, default=0.1)
    parser.add_argument("--nms-thr", type=float, default=0.01)
    parser.add_argument("--nms-pre", type=int, default=100)
    parser.add_argument("--max-num", type=int, default=50)
    parser.add_argument("--save-dir", default="outputs/debug")
    parser.add_argument("--save-bev", action="store_true", help="Save BEV visualization image")
    parser.add_argument("--save-image", action="store_true", help="Save camera image visualization")
    parser.add_argument("--visualize", action="store_true", help="Save both BEV and camera visualizations when possible")
    parser.add_argument("--no-image-range-filter", action="store_true")
    parser.add_argument("--no-lidar-range-filter", action="store_true")
    parser.add_argument("--bev-y-reverse", action="store_true", help="Legacy export compatibility only. Source-style scatter does not reverse y.")
    parser.add_argument("--min-camera-depth", type=float, default=0.5)
    parser.add_argument("--max-proj-span-ratio", type=float, default=3.0)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pc = read_points(args.pc_path)
    pc_filtered = point_range_filter(pc)
    pillar_features, pillar_mask, coords, _num_points, valid_pillars = build_pillars(
        pc_filtered,
        np.array([0.0, -39.68, -3.0, 69.12, 39.68, 1.0], dtype=np.float32),
        np.array([0.16, 0.16, 4.0], dtype=np.float32),
        np.array([432, 496, 1], dtype=np.int64),
        12000,
        32,
    )
    print(f"[INFO] points={len(pc)} filtered={len(pc_filtered)} valid_pillars={valid_pillars}")

    if args.backend == "onnx":
        pfn_engine = OnnxEngine(Path(args.pfn), providers=["CPUExecutionProvider"])
        backbone_engine = OnnxEngine(Path(args.backbone), providers=["CPUExecutionProvider"])
    else:
        pfn_engine = MnnEngine(Path(args.pfn), args.runtime_device, args.num_threads)
        backbone_engine = MnnEngine(Path(args.backbone), args.runtime_device, args.num_threads)

    pfn_inputs = {
        "pillar_features": pillar_features[np.newaxis, ...],
        "pillar_mask": pillar_mask[np.newaxis, ...],
    }
    pfn_outputs = pfn_engine.run(pfn_inputs)
    if len(pfn_outputs) == 0:
        raise RuntimeError("PFN model returned no outputs")
    pillar_embed = pfn_outputs[0]

    bev_feature = scatter_bev((1, 64, 496, 432), pillar_embed, coords, valid_pillars, bev_y_reverse=args.bev_y_reverse)
    backbone_outputs = backbone_engine.run({"bev_feature": bev_feature})

    cls_preds = np.asarray(backbone_outputs[0][0], dtype=np.float32)
    box_preds = np.asarray(backbone_outputs[1][0], dtype=np.float32)
    dir_preds = np.asarray(backbone_outputs[2][0], dtype=np.float32) if len(backbone_outputs) > 2 else None
    print(f"[INFO] head shapes cls={cls_preds.shape} box={box_preds.shape} dir={None if dir_preds is None else dir_preds.shape}")

    result = decode_head_output_source_style(
        cls_preds,
        box_preds,
        dir_preds,
        score_threshold=args.score_thr,
        nms_threshold=args.nms_thr,
        nms_pre=args.nms_pre,
        max_detections=args.max_num,
    )
    print(f"[INFO] decoded boxes before range filters: {len(result['scores'])}")

    calib_info = None
    if args.calib_path and Path(args.calib_path).exists():
        calib_info = read_calib(args.calib_path)

    img = None
    if args.img_path and Path(args.img_path).exists():
        try:
            import cv2
            img = cv2.imread(str(args.img_path), 1)
        except Exception:
            img = None

    if calib_info is not None and img is not None and not args.no_image_range_filter:
        before = len(result["scores"])
        result = keep_bbox_from_image_range(
            result,
            calib_info["Tr_velo_to_cam"].astype(np.float32),
            calib_info["R0_rect"].astype(np.float32),
            calib_info["P2"].astype(np.float32),
            img.shape[:2],
        )
        print(f"[filter] image range: {before} -> {len(result['scores'])}")
        before = len(result["scores"])
        result = filter_projection_sane_result(
            result,
            calib=calib_info,
            image_shape=img.shape[:2],
            min_depth=args.min_camera_depth,
            max_proj_span_ratio=args.max_proj_span_ratio,
        )
        print(f"[filter] projection sanity: {before} -> {len(result['scores'])}")

    if not args.no_lidar_range_filter:
        before = len(result["scores"])
        result = keep_bbox_from_lidar_range(result, PCD_LIMIT_RANGE)
        print(f"[filter] lidar range: {before} -> {len(result['scores'])}")

    # Save result after regular source-style filters.
    save_result_json(Path(save_dir) / "detections_runtime.json", result, label_map=LABEL2CLASSES)
    print(f"[OK] saved results to {save_dir}")

    pred_counts = summarize_labels(result["labels"])
    print(f"[INFO] prediction counts: {pred_counts}")
    if args.gt_path and calib_info is not None and Path(args.gt_path).exists():
        _, _, gt_labels = load_gt_source_style(args.gt_path, calib_info)
        gt_counts = summarize_labels(gt_labels)
        print(f"[INFO] ground truth counts: {gt_counts}")

    save_bev = args.save_bev or args.visualize
    save_image = args.save_image or args.visualize
    if save_bev or save_image:
        visualize_results(
            pc_filtered,
            result,
            calib_info,
            img,
            args.gt_path,
            save_dir,
            save_bev=save_bev,
            save_image=save_image,
            min_camera_depth=args.min_camera_depth,
            max_proj_span_ratio=args.max_proj_span_ratio,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())