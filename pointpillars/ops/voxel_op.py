"""Compatibility fallback for zhulf0804/PointPillars voxel_op extension.

The upstream repository normally builds a C++/CUDA extension named
``pointpillars.ops.voxel_op`` and imports ``hard_voxelize`` from it.
This pure-PyTorch/Python implementation keeps the same function signature so
inference can run without compiling the extension. It is intended for debug and
ONNX/MNN baseline verification, not high-FPS deployment.
"""
from __future__ import annotations

from typing import Sequence

import torch


def _to_float_tuple(values: Sequence[float] | torch.Tensor) -> tuple[float, ...]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    return tuple(float(v) for v in values)


@torch.no_grad()
def hard_voxelize(
    points: torch.Tensor,
    voxels: torch.Tensor,
    coors: torch.Tensor,
    num_points_per_voxel: torch.Tensor,
    voxel_size,
    coors_range,
    max_points: int,
    max_voxels: int,
    ndim: int = 3,
    deterministic: bool = True,
) -> int:
    """Fill ``voxels/coors/num_points_per_voxel`` like the upstream extension.

    Args follow the upstream C++ extension. ``coors`` is filled in (z, y, x)
    order because upstream ``voxel_module.py`` immediately applies
    ``coors_out = coors[:voxel_num].flip(-1)`` to obtain (x, y, z).
    """
    if points.ndim != 2 or points.size(1) < ndim:
        raise ValueError(f"points must be [N, >= {ndim}], got {tuple(points.shape)}")

    device = points.device
    dtype = points.dtype
    vx, vy, vz = _to_float_tuple(voxel_size)
    x_min, y_min, z_min, x_max, y_max, z_max = _to_float_tuple(coors_range)
    grid_x = int(round((x_max - x_min) / vx))
    grid_y = int(round((y_max - y_min) / vy))
    grid_z = int(round((z_max - z_min) / vz))

    voxel_map: dict[tuple[int, int, int], int] = {}
    voxel_num = 0

    pts_cpu = points.detach().cpu()
    for p_cpu in pts_cpu:
        x = float(p_cpu[0]); y = float(p_cpu[1]); z = float(p_cpu[2])
        if not (x_min <= x < x_max and y_min <= y < y_max and z_min <= z < z_max):
            continue
        x_idx = int((x - x_min) // vx)
        y_idx = int((y - y_min) // vy)
        z_idx = int((z - z_min) // vz)
        if not (0 <= x_idx < grid_x and 0 <= y_idx < grid_y and 0 <= z_idx < grid_z):
            continue

        key = (x_idx, y_idx, z_idx)
        voxel_id = voxel_map.get(key)
        if voxel_id is None:
            if voxel_num >= int(max_voxels):
                continue
            voxel_id = voxel_num
            voxel_map[key] = voxel_id
            # Upstream extension writes (z, y, x); voxel_module flips to (x, y, z).
            coors[voxel_id, 0] = z_idx
            coors[voxel_id, 1] = y_idx
            coors[voxel_id, 2] = x_idx
            voxel_num += 1

        n = int(num_points_per_voxel[voxel_id].item())
        if n < int(max_points):
            voxels[voxel_id, n, : points.size(1)] = p_cpu.to(device=device, dtype=dtype)
            num_points_per_voxel[voxel_id] += 1

    return int(voxel_num)
