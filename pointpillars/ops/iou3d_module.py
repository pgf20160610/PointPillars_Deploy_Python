# -*- coding: utf-8 -*-
"""Safe iou3d_module wrapper for Python fallback ops.

Drop-in replacement for pointpillars/ops/iou3d_module.py.
It keeps the upstream API but avoids CPU/CUDA index device mismatches.
"""
from __future__ import annotations

import torch

from pointpillars.ops.iou3d_op import (
    boxes_overlap_bev_gpu,
    boxes_iou_bev_gpu,
    nms_gpu,
    nms_normal_gpu as nms_normal_gpu_op,
)


def boxes_overlap_bev(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    ans_overlap = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_overlap)
    return ans_overlap


def boxes_iou_bev(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)
    return ans_iou


def nms_cuda(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    thresh: float,
    pre_maxsize=None,
    post_max_size=None,
) -> torch.Tensor:
    """Rotated BEV NMS compatible with upstream PointPillars.

    boxes: [N, 5] as [x1, y1, x2, y2, yaw].  The underlying nms_gpu receives
    boxes already sorted by score and writes kept POSITION indices into keep.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:int(pre_maxsize)]
    boxes_sorted = boxes[order].contiguous()
    keep = torch.zeros((boxes_sorted.size(0),), dtype=torch.long, device=boxes_sorted.device)
    device_id = boxes_sorted.device.index if boxes_sorted.is_cuda else None
    num_out = nms_gpu(boxes_sorted, keep, float(thresh), device_id)
    keep_pos = keep[:num_out].to(device=order.device, dtype=torch.long)
    keep_idx = order[keep_pos].contiguous()
    if post_max_size is not None:
        keep_idx = keep_idx[:int(post_max_size)]
    return keep_idx


def nms_normal_cuda(boxes: torch.Tensor, scores: torch.Tensor, thresh: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    order = scores.sort(0, descending=True)[1]
    boxes_sorted = boxes[order].contiguous()
    keep = torch.zeros((boxes_sorted.size(0),), dtype=torch.long, device=boxes_sorted.device)
    device_id = boxes_sorted.device.index if boxes_sorted.is_cuda else None
    num_out = nms_normal_gpu_op(boxes_sorted, keep, float(thresh), device_id)
    keep_pos = keep[:num_out].to(device=order.device, dtype=torch.long)
    return order[keep_pos].contiguous()
