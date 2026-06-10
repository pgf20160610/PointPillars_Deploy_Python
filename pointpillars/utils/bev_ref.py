#!/usr/bin/env python
# coding=utf-8
'''
creater      : PGF
since        : 2026-06-09 16:10:53
lastTime     : 2026-06-09 16:15:36
LastAuthor   : PGF
message      : The function of this file is 
文件相对于项目的路径   : /PointPillars_ONNX_MNN_CPP/pointpillars/utils/bev_ref.py
Copyright (c) 2026 by pgf email: nchu_pgf@163.com, All Rights Reserved.
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BEV visualization aligned with zhulf0804/PointPillars box convention.

Box convention:
    lidar_bboxes: [x, y, z, w, l, h, yaw]
    x: front, y: left, z: up/bottom, yaw follows source bbox3d2bevcorners.

This module fixes common BEV angle errors caused by:
  1) treating [w, l] as [l, w];
  2) using standard CCW rotation instead of source rotation matrix;
  3) rotating yaw again when mapping lidar XY to image UV.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np


DEFAULT_PC_RANGE = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)


def _as_lidar_boxes(boxes: Sequence[Mapping[str, float]] | np.ndarray) -> np.ndarray:
    """Convert list/dict JSON boxes or ndarray to ndarray [x,y,z,w,l,h,yaw]."""
    if isinstance(boxes, np.ndarray):
        arr = boxes.astype(np.float32, copy=False)
        if arr.size == 0:
            return arr.reshape(0, 7)
        if arr.shape[-1] < 7:
            raise ValueError(f"boxes ndarray must have >=7 columns, got {arr.shape}")
        return arr[:, :7]

    out: list[list[float]] = []
    for b in boxes:
        # Prefer source convention keys.
        if all(k in b for k in ("x", "y", "z", "w", "l", "h", "yaw")):
            out.append([float(b["x"]), float(b["y"]), float(b["z"]), float(b["w"]), float(b["l"]), float(b["h"]), float(b["yaw"])])
        # Backward-compatible deployment JSON sometimes uses dx/dy/dz.
        # For this project, dx->w, dy->l, dz->h only if w/l/h are absent.
        elif all(k in b for k in ("x", "y", "z", "dx", "dy", "dz", "yaw")):
            out.append([float(b["x"]), float(b["y"]), float(b["z"]), float(b["dx"]), float(b["dy"]), float(b["dz"]), float(b["yaw"])])
        else:
            raise KeyError(f"box missing fields: {b.keys()}")
    return np.asarray(out, dtype=np.float32).reshape(-1, 7)


def bbox3d2bevcorners_ref(bboxes):
    """
    bboxes: (N, 7), [x, y, z, w, l, h, yaw]
    """
    centers = bboxes[:, :2]
    dims = bboxes[:, 3:5]   # [w, l]
    angles = bboxes[:, 6]

    base = np.array([
        [-0.5, -0.5],
        [-0.5,  0.5],
        [ 0.5,  0.5],
        [ 0.5, -0.5],
    ], dtype=np.float32)

    corners = base[None, :, :] * dims[:, None, :]

    s, c = np.sin(angles), np.cos(angles)

    rot = np.array([
        [ c,  s],
        [-s,  c],
    ], dtype=np.float32).transpose(2, 1, 0)

    corners = corners @ rot
    corners += centers[:, None, :]

    return corners.astype(np.float32)


def lidar_xy_to_bev_uv(xy, pc_range, scale, margin):
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    # x/front 向上，y/left 向左
    u = margin + (y_max - xy[..., 1]) * scale
    v = margin + (x_max - xy[..., 0]) * scale

    return np.stack([u, v], axis=-1).astype(np.int32)


def draw_lidar_boxes_bev(
    canvas: np.ndarray,
    boxes: Sequence[Mapping[str, float]] | np.ndarray,
    pc_range: Sequence[float] = DEFAULT_PC_RANGE,
    scale: float = 10.0,
    margin: int = 20,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    prefix: str = "Pred",
    draw_heading: bool = True,
) -> None:
    """Draw lidar boxes in source convention onto BEV canvas."""
    arr = _as_lidar_boxes(boxes)
    if arr.size == 0:
        return
    corners_xy = bbox3d2bevcorners_ref(arr)
    corners_uv = lidar_xy_to_bev_uv(corners_xy, pc_range, scale, margin)
    centers_uv = lidar_xy_to_bev_uv(arr[:, :2], pc_range, scale, margin)

    for i, poly in enumerate(corners_uv):
        cv2.polylines(canvas, [poly.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        if draw_heading:
            # Source corner order makes edge 0-3 one side and 1-2 the other.  The
            # local +x side is between corners 2 and 3 after source transform.
            # Draw center -> midpoint(corner2, corner3) as an orientation marker.
            front_mid = ((poly[2].astype(np.float32) + poly[3].astype(np.float32)) * 0.5).astype(np.int32)
            cv2.line(canvas, tuple(centers_uv[i]), tuple(front_mid), color, max(1, thickness), cv2.LINE_AA)
        label = prefix
        if isinstance(boxes, list) and i < len(boxes):
            score = boxes[i].get("score") if isinstance(boxes[i], Mapping) else None
            class_name = boxes[i].get("class_name", "") if isinstance(boxes[i], Mapping) else ""
            if score is not None:
                label = f"{prefix} {class_name} {float(score):.2f}".strip()
        cv2.putText(canvas, label, tuple(poly[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def save_bev_image_ref(
    points: np.ndarray,
    pred_boxes: Sequence[Mapping[str, float]] | np.ndarray,
    out_path: str | Path,
    gt_boxes: Sequence[Mapping[str, float]] | np.ndarray | None = None,
    pc_range: Sequence[float] = DEFAULT_PC_RANGE,
    scale: float = 10.0,
    margin: int = 20,
) -> None:
    """Save headless BEV visualization with source-compatible yaw/dim convention."""
    x_min, y_min, _z_min, x_max, y_max, _z_max = [float(v) for v in pc_range]
    width = int(round((y_max - y_min) * scale)) + 2 * margin
    height = int(round((x_max - x_min) * scale)) + 2 * margin
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    if points is not None and len(points) > 0:
        pts = np.asarray(points, dtype=np.float32)
        keep = (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) & (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
        uv = lidar_xy_to_bev_uv(pts[keep, :2], pc_range, scale, margin)
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        canvas[uv[valid, 1], uv[valid, 0]] = (180, 180, 180)

    if gt_boxes is not None:
        draw_lidar_boxes_bev(canvas, gt_boxes, pc_range, scale, margin, color=(0, 200, 200), thickness=2, prefix="GT")
    draw_lidar_boxes_bev(canvas, pred_boxes, pc_range, scale, margin, color=(255, 0, 0), thickness=2, prefix="Pred")

    # Axes: x/front up, y/left left.
    origin = lidar_xy_to_bev_uv(np.array([[0.0, 0.0]], dtype=np.float32), pc_range, scale, margin)[0]
    x_front = lidar_xy_to_bev_uv(np.array([[6.0, 0.0]], dtype=np.float32), pc_range, scale, margin)[0]
    y_left = lidar_xy_to_bev_uv(np.array([[0.0, 6.0]], dtype=np.float32), pc_range, scale, margin)[0]
    cv2.arrowedLine(canvas, tuple(origin), tuple(x_front), (0, 0, 255), 3, tipLength=0.25)
    cv2.arrowedLine(canvas, tuple(origin), tuple(y_left), (0, 180, 0), 3, tipLength=0.25)
    cv2.putText(canvas, "x/front", tuple(x_front + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(canvas, "y/left", tuple(y_left + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 0), 2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def load_detections_json(path: str | Path) -> list[dict[str, float]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
