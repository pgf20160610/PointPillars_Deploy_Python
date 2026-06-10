
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class VoxelizationConfig:
    voxel_size: Tuple[float, float, float]
    point_cloud_range: Tuple[float, float, float, float, float, float]
    max_num_points: int
    max_voxels: Tuple[int, int] | int


class Voxelization(nn.Module):
    """Pure-PyTorch fallback for the source repository's CUDA voxel op.

    Output format follows zhulf0804/PointPillars:
        voxels:      (M, max_num_points, 4)
        coordinates: (M, 3), stored as (x_idx, y_idx, z_idx) for compatibility
        num_points:  (M,)

    This fallback is intended for correctness/debug and CPU/CUDA portability.
    For high FPS deployment, replace only this class with a C++/CUDA voxelizer.
    """

    def __init__(self, voxel_size, point_cloud_range, max_num_points=32, max_voxels=(16000, 40000)):
        super().__init__()
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.point_cloud_range = tuple(float(v) for v in point_cloud_range)
        self.max_num_points = int(max_num_points)
        if isinstance(max_voxels, Iterable) and not isinstance(max_voxels, (str, bytes)):
            mv = tuple(int(v) for v in max_voxels)
            self.max_voxels = mv[-1] if len(mv) else 40000
        else:
            self.max_voxels = int(max_voxels)

        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        vx, vy, vz = self.voxel_size
        self.grid_size = (
            int(round((x_max - x_min) / vx)),
            int(round((y_max - y_min) / vy)),
            int(round((z_max - z_min) / vz)),
        )

    @torch.no_grad()
    def forward(self, points: torch.Tensor):
        if points.ndim != 2 or points.size(-1) < 4:
            raise ValueError(f"points must have shape (N, >=4), got {tuple(points.shape)}")

        device = points.device
        dtype = points.dtype
        pc_range = torch.tensor(self.point_cloud_range, device=device, dtype=dtype)
        voxel = torch.tensor(self.voxel_size, device=device, dtype=dtype)
        xyz = points[:, :3]
        keep = (xyz[:, 0] >= pc_range[0]) & (xyz[:, 0] < pc_range[3]) & \
               (xyz[:, 1] >= pc_range[1]) & (xyz[:, 1] < pc_range[4]) & \
               (xyz[:, 2] >= pc_range[2]) & (xyz[:, 2] < pc_range[5])
        points = points[keep]
        if points.numel() == 0:
            return (
                torch.zeros((0, self.max_num_points, points.shape[-1] if points.ndim == 2 else 4), device=device, dtype=dtype),
                torch.zeros((0, 3), device=device, dtype=torch.int32),
                torch.zeros((0,), device=device, dtype=torch.int32),
            )

        coords_f = torch.floor((points[:, :3] - pc_range[:3]) / voxel).to(torch.long)
        gx, gy, gz = self.grid_size
        inside = (coords_f[:, 0] >= 0) & (coords_f[:, 0] < gx) & \
                 (coords_f[:, 1] >= 0) & (coords_f[:, 1] < gy) & \
                 (coords_f[:, 2] >= 0) & (coords_f[:, 2] < gz)
        points = points[inside]
        coords_f = coords_f[inside]
        if points.numel() == 0:
            return (
                torch.zeros((0, self.max_num_points, points.shape[-1]), device=device, dtype=dtype),
                torch.zeros((0, 3), device=device, dtype=torch.int32),
                torch.zeros((0,), device=device, dtype=torch.int32),
            )

        # Keep insertion order like a hard voxelizer. Python dict is deterministic in insertion order.
        voxels = []
        coords = []
        counts = []
        voxel_map: dict[tuple[int, int, int], int] = {}
        pts_cpu = points.detach().cpu()
        coords_cpu = coords_f.detach().cpu()
        for p, c in zip(pts_cpu, coords_cpu):
            key = (int(c[0]), int(c[1]), int(c[2]))
            idx = voxel_map.get(key)
            if idx is None:
                if len(voxels) >= self.max_voxels:
                    continue
                idx = len(voxels)
                voxel_map[key] = idx
                voxels.append(torch.zeros((self.max_num_points, points.shape[-1]), dtype=dtype))
                coords.append(key)
                counts.append(0)
            n = counts[idx]
            if n < self.max_num_points:
                voxels[idx][n] = p.to(dtype)
                counts[idx] += 1

        if not voxels:
            return (
                torch.zeros((0, self.max_num_points, points.shape[-1]), device=device, dtype=dtype),
                torch.zeros((0, 3), device=device, dtype=torch.int32),
                torch.zeros((0,), device=device, dtype=torch.int32),
            )
        voxels_t = torch.stack(voxels, dim=0).to(device=device, dtype=dtype)
        coors_t = torch.tensor(coords, dtype=torch.int32, device=device)
        counts_t = torch.tensor(counts, dtype=torch.int32, device=device)
        return voxels_t, coors_t, counts_t
