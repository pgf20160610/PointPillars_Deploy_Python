#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source-style PointPillars visualization helpers.

Drop-in replacement for:
    pointpillars/utils/vis_o3d.py

This file does NOT modify or depend on process.py geometry, because the local
process.py may have been changed during deployment debugging.  Geometry below is
implemented to match zhulf0804/PointPillars source conventions:
    - lidar box:  [x, y, z, w, l, h, yaw]
    - camera box: [x, y, z, l, h, w, ry]
    - BEV: x/front up, y/left left
    - image GT: draw directly from KITTI camera boxes
    - image pred: lidar -> camera -> camera corners -> image
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

LABEL2CLASSES = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}

# OpenCV BGR colors.
COLORS = {
    0: (0, 0, 255),        # Pedestrian red
    1: (0, 200, 0),        # Cyclist green
    2: (255, 0, 0),        # Car blue
    "gt": (0, 220, 220),  # Ground truth yellow
}

EDGES_3D = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _np(x: Any, dtype=np.float32) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=dtype)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _as_4x4(mat: Any, name: str = "matrix") -> np.ndarray:
    m = _np(mat, np.float32)
    if m.shape == (4, 4):
        return m
    if m.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = m
        return out
    if m.shape == (3, 3):
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = m
        return out
    if m.size == 12:
        return _as_4x4(m.reshape(3, 4), name)
    if m.size == 9:
        return _as_4x4(m.reshape(3, 3), name)
    if m.size == 16:
        return m.reshape(4, 4).astype(np.float32)
    raise ValueError(f"{name} must be 3x3/3x4/4x4, got {m.shape}")


def _as_lidar_boxes(boxes: Any) -> np.ndarray:
    arr = _np(boxes, np.float32)
    if arr.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    return arr.reshape(-1, 7).astype(np.float32)


def _as_camera_boxes(boxes: Any) -> np.ndarray:
    arr = _np(boxes, np.float32)
    if arr.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    return arr.reshape(-1, 7).astype(np.float32)


def _as_labels(labels: Any, n: int, default: int = 2) -> np.ndarray:
    if labels is None:
        return np.full((n,), default, dtype=np.int64)
    arr = _np(labels, np.int64).reshape(-1)
    if arr.size < n:
        arr = np.concatenate([arr, np.full((n - arr.size,), default, dtype=np.int64)])
    return arr[:n]


def _as_scores(scores: Any, n: int) -> np.ndarray | None:
    if scores is None:
        return None
    arr = _np(scores, np.float32).reshape(-1)
    if arr.size < n:
        arr = np.concatenate([arr, np.zeros((n - arr.size,), dtype=np.float32)])
    return arr[:n]


# ---------------------------------------------------------------------------
# Source-compatible geometry. Keep this local to avoid accidental changes in
# process.py from affecting visualization.
# ---------------------------------------------------------------------------

def bbox_camera2lidar_src(camera_bboxes: np.ndarray, tr_velo_to_cam: np.ndarray, r0_rect: np.ndarray) -> np.ndarray:
    """camera [x,y,z,l,h,w,ry] -> lidar [x,y,z,w,l,h,yaw].

    This matches the source repository behavior: dimensions are reordered, and
    the last angle column is carried through unchanged.
    """
    b = _as_camera_boxes(camera_bboxes)
    if len(b) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    tr = _as_4x4(tr_velo_to_cam, "Tr_velo_to_cam")
    r0 = _as_4x4(r0_rect, "R0_rect")
    x_size, y_size, z_size = b[:, 3:4], b[:, 4:5], b[:, 5:6]
    xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)  # l,h,w -> w,l,h
    xyz1 = np.pad(b[:, :3], ((0, 0), (0, 1)), constant_values=1.0)
    xyz = xyz1 @ np.linalg.inv(r0 @ tr).T
    return np.concatenate([xyz[:, :3], xyz_size, b[:, 6:7]], axis=1).astype(np.float32)


def bbox_lidar2camera_src(lidar_bboxes: np.ndarray, tr_velo_to_cam: np.ndarray, r0_rect: np.ndarray) -> np.ndarray:
    """lidar [x,y,z,w,l,h,yaw] -> camera [x,y,z,l,h,w,ry]."""
    b = _as_lidar_boxes(lidar_bboxes)
    if len(b) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    tr = _as_4x4(tr_velo_to_cam, "Tr_velo_to_cam")
    r0 = _as_4x4(r0_rect, "R0_rect")
    x_size, y_size, z_size = b[:, 3:4], b[:, 4:5], b[:, 5:6]
    xyz_size = np.concatenate([y_size, z_size, x_size], axis=1)  # w,l,h -> l,h,w
    xyz1 = np.pad(b[:, :3], ((0, 0), (0, 1)), constant_values=1.0)
    xyz = xyz1 @ (r0 @ tr).T
    return np.concatenate([xyz[:, :3], xyz_size, b[:, 6:7]], axis=1).astype(np.float32)


def bbox3d2bevcorners_src(lidar_bboxes: np.ndarray) -> np.ndarray:
    """Source-style BEV corners for lidar [x,y,z,w,l,h,yaw]."""
    b = _as_lidar_boxes(lidar_bboxes)
    if len(b) == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    centers, dims, angles = b[:, :2], b[:, 3:5], b[:, 6]
    bev_corners = np.array(
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
        dtype=np.float32,
    )
    corners = bev_corners[None, :, :] * dims[:, None, :]
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # Source convention: in fact, -angle.
    rot_mat = np.array([[rot_cos, rot_sin], [-rot_sin, rot_cos]], dtype=np.float32)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))
    corners = corners @ rot_mat
    corners += centers[:, None, :]
    return corners.astype(np.float32)


def bbox3d2corners_camera_src(camera_bboxes: np.ndarray) -> np.ndarray:
    """Camera 3D corners for camera [x,y,z,l,h,w,ry].

    KITTI camera y points down and label location is bottom center. This is why
    y coordinates use [0, -1] * h instead of ±h/2.
    """
    b = _as_camera_boxes(camera_bboxes)
    if len(b) == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    centers, dims, angles = b[:, :3], b[:, 3:6], b[:, 6]
    base = np.array(
        [
            [0.5, 0.0, -0.5], [0.5, -1.0, -0.5], [-0.5, -1.0, -0.5], [-0.5, 0.0, -0.5],
            [0.5, 0.0, 0.5],  [0.5, -1.0, 0.5],  [-0.5, -1.0, 0.5],  [-0.5, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    corners = base[None, :, :] * dims[:, None, :]
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    rot_mat = np.array(
        [
            [rot_cos, np.zeros_like(rot_cos), rot_sin],
            [np.zeros_like(rot_cos), np.ones_like(rot_cos), np.zeros_like(rot_cos)],
            [-rot_sin, np.zeros_like(rot_cos), rot_cos],
        ],
        dtype=np.float32,
    )
    rot_mat = np.transpose(rot_mat, (2, 1, 0))
    corners = corners @ rot_mat
    corners += centers[:, None, :]
    return corners.astype(np.float32)


def points_camera2image_src(points: np.ndarray, P2: np.ndarray) -> np.ndarray:
    pts = _np(points, np.float32)
    P2_4 = _as_4x4(P2, "P2")
    extended = np.pad(pts, ((0, 0), (0, 0), (0, 1)), constant_values=1.0)
    projected = extended @ P2_4.T
    z = projected[:, :, 2:3]
    image_points = projected[:, :, :2] / np.where(np.abs(z) < 1e-6, np.nan, z)
    return image_points.astype(np.float32)




def _projection_keep_mask(
    camera_corners: np.ndarray,
    image_points: np.ndarray,
    image_shape: tuple[int, int],
    min_depth: float = 0.5,
    max_span_ratio: float = 3.0,
) -> np.ndarray:
    """Return boxes safe to draw on image.

    A 3D box with any corner behind/too close to camera can produce extremely
    large projected coordinates, which appears as blue rays across the image.
    Keep only boxes whose 8 camera-space corners are in front of the camera and
    whose projected 2D extent is not unreasonably larger than the image.
    """
    if len(camera_corners) == 0:
        return np.zeros((0,), dtype=bool)
    H, W = int(image_shape[0]), int(image_shape[1])
    z_ok = np.all(np.asarray(camera_corners[:, :, 2], dtype=np.float32) > float(min_depth), axis=1)
    pts = np.asarray(image_points, dtype=np.float32)
    finite = np.all(np.isfinite(pts), axis=(1, 2))
    x1 = np.nanmin(pts[:, :, 0], axis=1)
    y1 = np.nanmin(pts[:, :, 1], axis=1)
    x2 = np.nanmax(pts[:, :, 0], axis=1)
    y2 = np.nanmax(pts[:, :, 1], axis=1)
    intersects = (x2 >= 0) & (x1 < W) & (y2 >= 0) & (y1 < H)
    span_ok = ((x2 - x1) <= float(max_span_ratio) * W) & ((y2 - y1) <= float(max_span_ratio) * H)
    non_degenerate = ((x2 - x1) > 2) & ((y2 - y1) > 2)
    return z_ok & finite & intersects & span_ok & non_degenerate

# ---------------------------------------------------------------------------
# Image drawing
# ---------------------------------------------------------------------------

def _cv_point(p: Any) -> tuple[int, int] | None:
    arr = np.asarray(p, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return None
    x, y = float(arr[0]), float(arr[1])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    if abs(x) > 1e6 or abs(y) > 1e6:
        return None
    return int(round(x)), int(round(y))


def _normalize_image_points(image_points: Any) -> np.ndarray:
    pts = _np(image_points, np.float32)
    if pts.ndim == 3 and pts.shape[1:] == (2, 8):
        pts = np.transpose(pts, (0, 2, 1))
    if pts.ndim != 3 or pts.shape[1:] != (8, 2):
        raise ValueError(f"image_points must be (N,8,2), got {pts.shape}")
    return pts


def vis_img_3d(
    img: np.ndarray,
    image_points: np.ndarray,
    labels: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    rt: bool = True,
    prefix: str = "Pred",
    color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    pts_all = _normalize_image_points(image_points)
    labels = _as_labels(labels, len(pts_all), default=2)
    scores = _as_scores(scores, len(pts_all))
    out = img.copy()
    H, W = out.shape[:2]

    for i, pts in enumerate(pts_all):
        if not np.all(np.isfinite(pts)):
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        span_x = float(xs.max() - xs.min())
        span_y = float(ys.max() - ys.min())
        # Skip boxes whose projection explodes due to near/behind-camera depth.
        if span_x > 3.0 * W or span_y > 3.0 * H:
            continue
        if xs.max() < -W or xs.min() > 2 * W or ys.max() < -H or ys.min() > 2 * H:
            continue
        cls_id = int(labels[i]) if i < len(labels) else 2
        draw_color = color if color is not None else COLORS.get(cls_id, COLORS[2])

        for a, b in EDGES_3D:
            p1 = _cv_point(pts[a])
            p2 = _cv_point(pts[b])
            if p1 is not None and p2 is not None:
                cv2.line(out, p1, p2, draw_color, 2, lineType=cv2.LINE_AA)

        p0 = _cv_point(pts[0])
        if p0 is not None:
            name = LABEL2CLASSES.get(cls_id, str(cls_id))
            text = f"{prefix} {name}"
            if scores is not None and i < len(scores):
                text += f" {float(scores[i]):.2f}"
            cv2.putText(out, text, p0, cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 1, cv2.LINE_AA)
    return out


def draw_lidar_boxes_on_image(
    img: np.ndarray,
    lidar_bboxes: np.ndarray,
    labels: np.ndarray,
    tr_velo_to_cam: np.ndarray,
    r0_rect: np.ndarray,
    P2: np.ndarray,
    scores: np.ndarray | None = None,
    prefix: str = "Pred",
    min_depth: float = 0.5,
    max_span_ratio: float = 3.0,
) -> np.ndarray:
    lidar_bboxes = _as_lidar_boxes(lidar_bboxes)
    if len(lidar_bboxes) == 0:
        return img
    labels = _as_labels(labels, len(lidar_bboxes), default=2)
    scores = _as_scores(scores, len(lidar_bboxes))
    camera_bboxes = bbox_lidar2camera_src(lidar_bboxes, tr_velo_to_cam, r0_rect)
    camera_corners = bbox3d2corners_camera_src(camera_bboxes)
    image_points = points_camera2image_src(camera_corners, P2)

    keep = _projection_keep_mask(
        camera_corners,
        image_points,
        img.shape[:2],
        min_depth=min_depth,
        max_span_ratio=max_span_ratio,
    )
    if not np.any(keep):
        return img
    image_points = image_points[keep]
    labels = labels[keep]
    if scores is not None:
        scores = scores[keep]
    return vis_img_3d(img, image_points, labels=labels, scores=scores, prefix=prefix)


def draw_camera_boxes_on_image(
    img: np.ndarray,
    camera_bboxes: np.ndarray,
    labels: np.ndarray | None,
    P2: np.ndarray,
    prefix: str = "GT",
) -> np.ndarray:
    camera_bboxes = _as_camera_boxes(camera_bboxes)
    if len(camera_bboxes) == 0:
        return img
    camera_corners = bbox3d2corners_camera_src(camera_bboxes)
    image_points = points_camera2image_src(camera_corners, P2)
    labels = _as_labels(labels, len(camera_bboxes), default=2)
    keep = _projection_keep_mask(camera_corners, image_points, img.shape[:2], min_depth=0.05, max_span_ratio=4.0)
    if np.any(keep):
        image_points = image_points[keep]
        labels = labels[keep]
    return vis_img_3d(img, image_points, labels=labels, scores=None, prefix=prefix, color=COLORS["gt"])


# ---------------------------------------------------------------------------
# BEV drawing
# ---------------------------------------------------------------------------

def _metric_to_canvas_xy(xy: np.ndarray, scale: float, cx: int, cy: int) -> np.ndarray:
    xy = _np(xy, np.float32)
    u = cx - xy[..., 1] * scale  # y/left is left on screen
    v = cy - xy[..., 0] * scale  # x/front is up on screen
    return np.stack([u, v], axis=-1).astype(np.int32)


def _draw_bev_box(
    canvas: np.ndarray,
    corners_xy: np.ndarray,
    color: tuple[int, int, int],
    text: str | None,
    scale: float,
    cx: int,
    cy: int,
    thickness: int = 2,
) -> None:
    pts = _metric_to_canvas_xy(corners_xy, scale, cx, cy).reshape(-1, 2)
    for i in range(4):
        p1 = tuple(int(v) for v in pts[i])
        p2 = tuple(int(v) for v in pts[(i + 1) % 4])
        cv2.line(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    if text:
        p = tuple(int(v) for v in pts[0])
        cv2.putText(canvas, text, p, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def save_bev_image(
    points: np.ndarray,
    pred_bboxes: np.ndarray | None = None,
    pred_labels: np.ndarray | None = None,
    pred_scores: np.ndarray | None = None,
    out_path: str | Path | None = None,
    gt_bboxes: np.ndarray | None = None,
    width: int = 1024,
    height: int = 768,
    scale: float = 10.0,
    # Backward compatible aliases:
    result: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    gt_lidar_bboxes: np.ndarray | None = None,
    **_: Any,
) -> np.ndarray:
    if result is not None:
        pred_bboxes = result.get("lidar_bboxes", pred_bboxes)
        pred_labels = result.get("labels", pred_labels)
        pred_scores = result.get("scores", pred_scores)
    if output_path is not None and out_path is None:
        out_path = output_path
    if gt_lidar_bboxes is not None and gt_bboxes is None:
        gt_bboxes = gt_lidar_bboxes

    points = _np(points, np.float32)
    pred_bboxes = _as_lidar_boxes(pred_bboxes)
    gt_bboxes = _as_lidar_boxes(gt_bboxes)
    pred_labels = _as_labels(pred_labels, len(pred_bboxes), default=2)
    pred_scores = _as_scores(pred_scores, len(pred_bboxes))

    legend_w = 260
    canvas = np.full((height, width + legend_w, 3), 255, dtype=np.uint8)
    canvas[:, width:, :] = 0
    cx = width // 2
    cy = height - 80

    if points.size > 0:
        pix = _metric_to_canvas_xy(points[:, :2], scale, cx, cy)
        mask = (pix[:, 0] >= 0) & (pix[:, 0] < width) & (pix[:, 1] >= 0) & (pix[:, 1] < height)
        pix = pix[mask]
        canvas[pix[:, 1], pix[:, 0]] = (0, 0, 0)

    if len(gt_bboxes):
        for corners in bbox3d2bevcorners_src(gt_bboxes):
            _draw_bev_box(canvas, corners, COLORS["gt"], "GT", scale, cx, cy, 2)

    if len(pred_bboxes):
        for i, corners in enumerate(bbox3d2bevcorners_src(pred_bboxes)):
            cls_id = int(pred_labels[i]) if i < len(pred_labels) else 2
            name = LABEL2CLASSES.get(cls_id, str(cls_id))
            text = f"Pred {name}"
            if pred_scores is not None and i < len(pred_scores):
                text += f" {float(pred_scores[i]):.2f}"
            _draw_bev_box(canvas, corners, COLORS.get(cls_id, COLORS[2]), text, scale, cx, cy, 2)

    origin = (cx, cy)
    cv2.arrowedLine(canvas, origin, (cx, cy - 105), (0, 0, 255), 5, tipLength=0.25)
    cv2.putText(canvas, "x/front", (cx + 8, cy - 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.arrowedLine(canvas, origin, (cx - 115, cy), (0, 220, 0), 5, tipLength=0.25)
    cv2.putText(canvas, "y/left", (cx - 108, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)

    x0, y0 = width + 30, 300
    cv2.putText(canvas, "Pedestrian:", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[0], 2)
    cv2.putText(canvas, "Cyclist:", (x0, y0 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[1], 2)
    cv2.putText(canvas, "Car:", (x0, y0 + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[2], 2)
    cv2.putText(canvas, "Ground truth:", (x0, y0 + 255), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS["gt"], 2)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas)
    return canvas


def vis_pc(
    points,
    bboxes=None,
    labels=None,
    scores=None,
    save_path=None,
    gt_bboxes=None,
    **kwargs,
):
    if isinstance(bboxes, dict):
        result = bboxes
        bboxes = result.get("lidar_bboxes", None)
        labels = result.get("labels", labels)
        scores = result.get("scores", scores)
    if bboxes is None and "bboxes" in kwargs:
        bboxes = kwargs["bboxes"]
    if gt_bboxes is None and "gt_lidar_bboxes" in kwargs:
        gt_bboxes = kwargs["gt_lidar_bboxes"]
    if save_path is not None:
        return save_bev_image(
            points,
            pred_bboxes=bboxes,
            pred_labels=labels,
            pred_scores=scores,
            out_path=save_path,
            gt_bboxes=gt_bboxes,
        )
    return None
