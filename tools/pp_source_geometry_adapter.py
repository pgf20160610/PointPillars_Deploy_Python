#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-replacement source-geometry adapter for zhulf0804/PointPillars.

Put this file under tools/ and import it from infer_pointpillars_pytorch.py.
It does NOT replace pointpillars.utils.*. It only mirrors the source repository's
KITTI label dimension order, camera/lidar box transforms, BEV corners, and image projection.

Important source contract:
  KITTI label line dimensions are h,w,l.
  Source read_label converts h,w,l -> l,h,w before building camera boxes.
  Source camera box format: [x, y, z, l, h, w, ry]
  Source lidar box format:  [x, y, z, w, l, h, yaw]
  Source bbox_camera2lidar/bbox_lidar2camera DO NOT add pi/2 or negate yaw.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import math
import cv2
import numpy as np

EDGES_3D = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
LABEL2CLASSES = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}
COLORS = {
    0: (0, 0, 255),       # Pedestrian red in BGR
    1: (0, 180, 0),       # Cyclist green
    2: (255, 0, 0),       # Car blue
    -1: (0, 220, 220),    # GT yellow/cyan-ish in BGR
}


def read_kitti_label_source_order(label_path: str | Path) -> Dict[str, np.ndarray]:
    """Read KITTI label_2 with the same dimension reorder as source io.py.

    Returns camera-space boxes in source format [x, y, z, l, h, w, ry].
    """
    names, bbox2d, dims_lhw, locs, rys = [], [], [], [], []
    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            p = raw.strip().split()
            if len(p) < 15 or p[0] == "DontCare":
                continue
            names.append(p[0])
            bbox2d.append([float(v) for v in p[4:8]])
            h, w, l = [float(v) for v in p[8:11]]
            # Source repo read_label(): dimensions = hwl[:, [2,0,1]] => l,h,w.
            dims_lhw.append([l, h, w])
            locs.append([float(v) for v in p[11:14]])
            rys.append(float(p[14]))
    if not names:
        return {
            "name": np.empty((0,), dtype=object),
            "bbox": np.empty((0, 4), dtype=np.float32),
            "dimensions": np.empty((0, 3), dtype=np.float32),
            "location": np.empty((0, 3), dtype=np.float32),
            "rotation_y": np.empty((0,), dtype=np.float32),
            "camera_bboxes": np.empty((0, 7), dtype=np.float32),
        }
    dims_lhw = np.asarray(dims_lhw, dtype=np.float32)
    locs = np.asarray(locs, dtype=np.float32)
    rys = np.asarray(rys, dtype=np.float32)
    return {
        "name": np.asarray(names),
        "bbox": np.asarray(bbox2d, dtype=np.float32),
        "dimensions": dims_lhw,
        "location": locs,
        "rotation_y": rys,
        "camera_bboxes": np.concatenate([locs, dims_lhw, rys[:, None]], axis=1).astype(np.float32),
    }


def ensure_4x4(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = mat
        return out
    if mat.shape == (3, 3):
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = mat
        return out
    raise ValueError(f"matrix must be 3x3, 3x4, or 4x4, got {mat.shape}")


def bbox_camera2lidar_source(bboxes_camera_lhw: np.ndarray, tr_velo_to_cam: np.ndarray, r0_rect: np.ndarray) -> np.ndarray:
    """Source-equivalent bbox_camera2lidar.

    Input camera bboxes are [x,y,z,l,h,w,ry]. Output lidar bboxes are [x,y,z,w,l,h,yaw].
    The yaw column is copied exactly; no `-ry-pi/2` conversion is applied in this repo.
    """
    b = np.asarray(bboxes_camera_lhw, dtype=np.float32).reshape(-1, 7)
    tr = ensure_4x4(tr_velo_to_cam)
    r0 = ensure_4x4(r0_rect)
    x_size, y_size, z_size = b[:, 3:4], b[:, 4:5], b[:, 5:6]  # l,h,w in source camera format
    xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)  # w,l,h for lidar
    extended_xyz = np.pad(b[:, :3], ((0, 0), (0, 1)), constant_values=1.0)
    rt_mat = np.linalg.inv(r0 @ tr)
    xyz = extended_xyz @ rt_mat.T
    return np.concatenate([xyz[:, :3], xyz_size, b[:, 6:]], axis=1).astype(np.float32)


def bbox_lidar2camera_source(bboxes_lidar_wlh: np.ndarray, tr_velo_to_cam: np.ndarray, r0_rect: np.ndarray) -> np.ndarray:
    """Source-equivalent bbox_lidar2camera.

    Input lidar bboxes [x,y,z,w,l,h,yaw]. Output camera bboxes [x,y,z,l,h,w,ry-like].
    The angle column is copied exactly, matching the source repo's visualization/filter path.
    """
    b = np.asarray(bboxes_lidar_wlh, dtype=np.float32).reshape(-1, 7)
    tr = ensure_4x4(tr_velo_to_cam)
    r0 = ensure_4x4(r0_rect)
    x_size, y_size, z_size = b[:, 3:4], b[:, 4:5], b[:, 5:6]  # w,l,h in lidar
    xyz_size = np.concatenate([y_size, z_size, x_size], axis=1)  # l,h,w in camera-source format
    extended_xyz = np.pad(b[:, :3], ((0, 0), (0, 1)), constant_values=1.0)
    xyz = extended_xyz @ (r0 @ tr).T
    return np.concatenate([xyz[:, :3], xyz_size, b[:, 6:]], axis=1).astype(np.float32)


def bbox3d2bevcorners_source(bboxes_lidar_wlh: np.ndarray) -> np.ndarray:
    """Source-equivalent bbox3d2bevcorners for [x,y,z,w,l,h,yaw]."""
    b = np.asarray(bboxes_lidar_wlh, dtype=np.float32).reshape(-1, 7)
    if b.size == 0:
        return np.empty((0, 4, 2), dtype=np.float32)
    centers, dims, angles = b[:, :2], b[:, 3:5], b[:, 6]
    corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32)
    corners = corners[None, :, :] * dims[:, None, :]
    s, c = np.sin(angles), np.cos(angles)
    rot_mat = np.array([[c, s], [-s, c]], dtype=np.float32)  # source says "in fact, -angle"
    rot_mat = np.transpose(rot_mat, (2, 1, 0))
    corners = corners @ rot_mat
    corners += centers[:, None, :]
    return corners.astype(np.float32)


def bbox3d2corners_camera_source(bboxes_camera_lhw: np.ndarray) -> np.ndarray:
    """Source-equivalent camera box corners for [x,y,z,l,h,w,ry]."""
    b = np.asarray(bboxes_camera_lhw, dtype=np.float32).reshape(-1, 7)
    if b.size == 0:
        return np.empty((0, 8, 3), dtype=np.float32)
    centers, dims, angles = b[:, :3], b[:, 3:6], b[:, 6]
    base = np.array([
        [ 0.5,  0.0, -0.5], [ 0.5, -1.0, -0.5], [-0.5, -1.0, -0.5], [-0.5,  0.0, -0.5],
        [ 0.5,  0.0,  0.5], [ 0.5, -1.0,  0.5], [-0.5, -1.0,  0.5], [-0.5,  0.0,  0.5],
    ], dtype=np.float32)
    corners = base[None, :, :] * dims[:, None, :]
    s, c = np.sin(angles), np.cos(angles)
    rot_mat = np.array([[c, np.zeros_like(c), s], [np.zeros_like(c), np.ones_like(c), np.zeros_like(c)], [-s, np.zeros_like(c), c]], dtype=np.float32)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))
    corners = corners @ rot_mat
    corners += centers[:, None, :]
    return corners.astype(np.float32)


def points_camera2image_source(points_camera: np.ndarray, P2: np.ndarray) -> np.ndarray:
    p2 = ensure_4x4(P2)
    pts = np.asarray(points_camera, dtype=np.float32)
    ext = np.pad(pts, ((0, 0), (0, 0), (0, 1)), constant_values=1.0)
    img = ext @ p2.T
    return (img[:, :, :2] / np.maximum(img[:, :, 2:3], 1e-6)).astype(np.float32)


def _safe_pt(p) -> tuple[int, int] | None:
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    if a.size < 2 or not (np.isfinite(a[0]) and np.isfinite(a[1])):
        return None
    if abs(a[0]) > 1e6 or abs(a[1]) > 1e6:
        return None
    return int(round(float(a[0]))), int(round(float(a[1])))


def draw_image_boxes_source(img: np.ndarray, image_points: np.ndarray, labels=None, scores=None, gt: bool = False, prefix: str = "Pred") -> np.ndarray:
    pts_all = np.asarray(image_points, dtype=np.float32)
    if pts_all.ndim == 3 and pts_all.shape[1] == 2 and pts_all.shape[2] == 8:
        pts_all = pts_all.transpose(0, 2, 1)
    out = img.copy()
    labels = np.zeros((len(pts_all),), dtype=np.int64) if labels is None else np.asarray(labels).reshape(-1)
    scores = None if scores is None else np.asarray(scores).reshape(-1)
    for i, pts in enumerate(pts_all):
        color = COLORS[-1] if gt else COLORS.get(int(labels[i]) if i < len(labels) else 2, (255, 0, 0))
        for a, b in EDGES_3D:
            p1, p2 = _safe_pt(pts[a]), _safe_pt(pts[b])
            if p1 is not None and p2 is not None:
                cv2.line(out, p1, p2, color, 2 if gt else 1, lineType=cv2.LINE_AA)
        p0 = _safe_pt(pts[0])
        if p0 is not None:
            if gt:
                text = "GT"
            else:
                cls = LABEL2CLASSES.get(int(labels[i]) if i < len(labels) else 2, "Obj")
                text = f"{prefix} {cls} {float(scores[i]):.2f}" if scores is not None and i < len(scores) else f"{prefix} {cls}"
            cv2.putText(out, text, p0, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def overlay_predictions_and_gt_on_image_source(img, pred_lidar, pred_labels, pred_scores, gt_camera, calib):
    out = img.copy()
    tr, r0, p2 = calib["Tr_velo_to_cam"], calib["R0_rect"], calib["P2"]
    if gt_camera is not None and len(gt_camera):
        gt_img_pts = points_camera2image_source(bbox3d2corners_camera_source(gt_camera), p2)
        out = draw_image_boxes_source(out, gt_img_pts, gt=True)
    if pred_lidar is not None and len(pred_lidar):
        pred_camera = bbox_lidar2camera_source(pred_lidar, tr, r0)
        pred_img_pts = points_camera2image_source(bbox3d2corners_camera_source(pred_camera), p2)
        out = draw_image_boxes_source(out, pred_img_pts, labels=pred_labels, scores=pred_scores, gt=False)
    return out


def save_bev_source_style(points, pred_lidar, pred_labels, pred_scores, gt_lidar, out_path, image_size=(768, 1280), scale=10.0):
    """Save BEV in the source-style display: x/front up, y/left left.

    This only maps source-computed BEV corners to pixels. It never changes yaw.
    """
    H, W = int(image_size[0]), int(image_size[1])
    legend_w = 260
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    canvas[:, W - legend_w:] = 0
    cx = (W - legend_w) // 2
    cy = H - 110

    pts = np.asarray(points, dtype=np.float32).reshape(-1, points.shape[-1]) if points is not None and len(points) else np.empty((0, 4), dtype=np.float32)
    if len(pts):
        u = cx - pts[:, 1] * scale
        v = cy - pts[:, 0] * scale
        ok = (u >= 0) & (u < W - legend_w) & (v >= 0) & (v < H)
        ui = u[ok].astype(np.int32); vi = v[ok].astype(np.int32)
        canvas[vi, ui] = (30, 30, 30)

    def map_xy(xy):
        arr = np.asarray(xy, dtype=np.float32)
        u = cx - arr[:, 1] * scale
        v = cy - arr[:, 0] * scale
        return np.stack([u, v], axis=1).astype(np.int32)

    def draw_boxes(bboxes, labels=None, scores=None, gt=False):
        if bboxes is None or len(bboxes) == 0:
            return
        corners = bbox3d2bevcorners_source(bboxes)
        labels_arr = np.zeros((len(corners),), dtype=np.int64) if labels is None else np.asarray(labels).reshape(-1)
        scores_arr = None if scores is None else np.asarray(scores).reshape(-1)
        for i, c in enumerate(corners):
            pix = map_xy(c)
            color = COLORS[-1] if gt else COLORS.get(int(labels_arr[i]) if i < len(labels_arr) else 2, (255, 0, 0))
            cv2.polylines(canvas, [pix.reshape(-1, 1, 2)], True, color, 2, lineType=cv2.LINE_AA)
            if gt:
                txt = "GT"
            else:
                cls = LABEL2CLASSES.get(int(labels_arr[i]) if i < len(labels_arr) else 2, "Obj")
                txt = f"Pred {cls} {float(scores_arr[i]):.2f}" if scores_arr is not None and i < len(scores_arr) else f"Pred {cls}"
            cv2.putText(canvas, txt, tuple(pix[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    draw_boxes(gt_lidar, gt=True)
    draw_boxes(pred_lidar, pred_labels, pred_scores, gt=False)

    # Axes: x/front up, y/left left.
    origin = (cx, cy)
    cv2.arrowedLine(canvas, origin, (cx, cy - 90), (0, 0, 255), 6, tipLength=0.25)
    cv2.arrowedLine(canvas, origin, (cx - 120, cy), (0, 220, 0), 6, tipLength=0.25)
    cv2.putText(canvas, "x/front", (cx + 10, cy - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "y/left", (cx - 115, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)

    # Legend
    lx = W - legend_w + 35
    cv2.putText(canvas, "Pedestrian:", (lx, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[0], 2, cv2.LINE_AA)
    cv2.putText(canvas, "Cyclist:", (lx, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[1], 2, cv2.LINE_AA)
    cv2.putText(canvas, "Car:", (lx, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[2], 2, cv2.LINE_AA)
    cv2.putText(canvas, "Ground truth:", (lx, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[-1], 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
    return canvas
