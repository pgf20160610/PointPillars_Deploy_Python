"""Pure Python/Torch fallback for zhulf0804/PointPillars iou3d_op.

This file replaces the compiled CUDA extension during debug.  It follows the
extension API used by upstream iou3d_module.py:
  boxes_overlap_bev_gpu(boxes_a, boxes_b, ans_overlap)
  boxes_iou_bev_gpu(boxes_a, boxes_b, ans_iou)
  nms_gpu(boxes, keep, thresh, device_id=None)
  nms_normal_gpu(boxes, keep, thresh, device_id=None)

Boxes are expected as [x1, y1, x2, y2, yaw].  nms_gpu receives boxes already
sorted by score in the upstream wrapper, so it writes kept POSITION indices.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch

Point = tuple[float, float]


def _signed_area(poly: list[Point]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        s += p[0] * q[1] - q[0] * p[1]
    return 0.5 * s


def _area(poly: list[Point]) -> float:
    return abs(_signed_area(poly))


def _rect_corners(box: Iterable[float]) -> list[Point]:
    vals = [float(v) for v in box]
    x1, y1, x2, y2, yaw = vals[:5]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    c, s = math.cos(yaw), math.sin(yaw)
    # CCW local rectangle.
    local = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    return [(cx + px * c - py * s, cy + px * s + py * c) for px, py in local]


def _cross(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _inside(p: Point, a: Point, b: Point, clip_ccw: bool) -> bool:
    c = _cross(a, b, p)
    return c >= -1e-9 if clip_ccw else c <= 1e-9


def _intersection(s: Point, e: Point, a: Point, b: Point) -> Point:
    x1, y1 = s; x2, y2 = e; x3, y3 = a; x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return e
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / den
    return (px, py)


def _clip(subject: list[Point], clipper: list[Point]) -> list[Point]:
    output = list(subject)
    if len(output) < 3 or len(clipper) < 3:
        return []
    clip_ccw = _signed_area(clipper) >= 0.0
    for i, a in enumerate(clipper):
        b = clipper[(i + 1) % len(clipper)]
        input_poly = output
        output = []
        if not input_poly:
            break
        s = input_poly[-1]
        for e in input_poly:
            e_in = _inside(e, a, b, clip_ccw)
            s_in = _inside(s, a, b, clip_ccw)
            if e_in:
                if not s_in:
                    output.append(_intersection(s, e, a, b))
                output.append(e)
            elif s_in:
                output.append(_intersection(s, e, a, b))
            s = e
    return output


def _overlap_one(a, b) -> float:
    pa = _rect_corners(a)
    pb = _rect_corners(b)
    return _area(_clip(pa, pb))


def _area_one(box) -> float:
    x1, y1, x2, y2, _ = [float(v) for v in box[:5]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@torch.no_grad()
def boxes_overlap_bev_gpu(boxes_a: torch.Tensor, boxes_b: torch.Tensor, ans_overlap: torch.Tensor) -> None:
    a_cpu = boxes_a.detach().cpu().float()
    b_cpu = boxes_b.detach().cpu().float()
    out = torch.zeros((a_cpu.shape[0], b_cpu.shape[0]), dtype=ans_overlap.dtype)
    for i in range(a_cpu.shape[0]):
        for j in range(b_cpu.shape[0]):
            out[i, j] = _overlap_one(a_cpu[i].tolist(), b_cpu[j].tolist())
    ans_overlap.copy_(out.to(device=ans_overlap.device, dtype=ans_overlap.dtype))


@torch.no_grad()
def boxes_iou_bev_gpu(boxes_a: torch.Tensor, boxes_b: torch.Tensor, ans_iou: torch.Tensor) -> None:
    a_cpu = boxes_a.detach().cpu().float()
    b_cpu = boxes_b.detach().cpu().float()
    out = torch.zeros((a_cpu.shape[0], b_cpu.shape[0]), dtype=ans_iou.dtype)
    areas_a = [_area_one(a_cpu[i].tolist()) for i in range(a_cpu.shape[0])]
    areas_b = [_area_one(b_cpu[j].tolist()) for j in range(b_cpu.shape[0])]
    for i in range(a_cpu.shape[0]):
        for j in range(b_cpu.shape[0]):
            inter = _overlap_one(a_cpu[i].tolist(), b_cpu[j].tolist())
            union = max(areas_a[i] + areas_b[j] - inter, 1e-8)
            out[i, j] = inter / union
    ans_iou.copy_(out.to(device=ans_iou.device, dtype=ans_iou.dtype))


@torch.no_grad()
def nms_gpu(boxes: torch.Tensor, keep: torch.Tensor, thresh: float, device_id=None) -> int:
    b_cpu = boxes.detach().cpu().float()
    n = int(b_cpu.shape[0])
    areas = [_area_one(b_cpu[i].tolist()) for i in range(n)]
    kept_positions: list[int] = []
    suppressed = [False] * n
    # Upstream wrapper has already sorted boxes by score.  Keep POSITION indices.
    for i in range(n):
        if suppressed[i]:
            continue
        kept_positions.append(i)
        for j in range(i + 1, n):
            if suppressed[j]:
                continue
            inter = _overlap_one(b_cpu[i].tolist(), b_cpu[j].tolist())
            union = max(areas[i] + areas[j] - inter, 1e-8)
            if inter / union > float(thresh):
                suppressed[j] = True
    if kept_positions:
        keep[:len(kept_positions)].copy_(torch.tensor(kept_positions, dtype=keep.dtype, device=keep.device))
    return len(kept_positions)


@torch.no_grad()
def nms_normal_gpu(boxes: torch.Tensor, keep: torch.Tensor, thresh: float, device_id=None) -> int:
    return nms_gpu(boxes, keep, thresh, device_id)
