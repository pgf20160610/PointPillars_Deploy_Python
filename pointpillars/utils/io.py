
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read_points(path: str | Path) -> np.ndarray:
    path = Path(path)
    pts = np.fromfile(str(path), dtype=np.float32)
    if pts.size == 0 or pts.size % 4 != 0:
        raise ValueError(f"invalid KITTI velodyne bin: {path}, float_count={pts.size}")
    return pts.reshape(-1, 4)


def read_calib(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    data: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            key, rest = line.split(":", 1)
            vals = np.array([float(x) for x in rest.strip().split()], dtype=np.float32)
            if key in ("P0", "P1", "P2", "P3"):
                data[key] = vals.reshape(3, 4)
            elif key in ("R0_rect", "R_rect"):
                data["R0_rect"] = vals.reshape(3, 3)
            elif key in ("Tr_velo_to_cam", "Tr_velo_cam"):
                data["Tr_velo_to_cam"] = vals.reshape(3, 4)
            elif key in ("Tr_imu_to_velo",):
                data[key] = vals.reshape(3, 4)
            else:
                data[key] = vals
    # Homogeneous convenience matrices, matching common KITTI utilities.
    if "Tr_velo_to_cam" in data and data["Tr_velo_to_cam"].shape == (3, 4):
        tr = np.eye(4, dtype=np.float32)
        tr[:3, :] = data["Tr_velo_to_cam"]
        data["Tr_velo_to_cam_4x4"] = tr
    if "R0_rect" in data and data["R0_rect"].shape == (3, 3):
        r = np.eye(4, dtype=np.float32)
        r[:3, :3] = data["R0_rect"]
        data["R0_rect_4x4"] = r
    return data


def read_label(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    names: list[str] = []
    truncated: list[float] = []
    occluded: list[int] = []
    alpha: list[float] = []
    bboxes2d: list[list[float]] = []
    dimensions: list[list[float]] = []  # h, w, l in KITTI label
    locations: list[list[float]] = []
    rotation_y: list[float] = []
    if not path.exists():
        return {
            "name": np.array([], dtype=object),
            "truncated": np.zeros((0,), dtype=np.float32),
            "occluded": np.zeros((0,), dtype=np.int64),
            "alpha": np.zeros((0,), dtype=np.float32),
            "bbox": np.zeros((0, 4), dtype=np.float32),
            "dimensions": np.zeros((0, 3), dtype=np.float32),
            "location": np.zeros((0, 3), dtype=np.float32),
            "rotation_y": np.zeros((0,), dtype=np.float32),
        }
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 15 or parts[0] == "DontCare":
                continue
            names.append(parts[0])
            truncated.append(float(parts[1]))
            occluded.append(int(float(parts[2])))
            alpha.append(float(parts[3]))
            bboxes2d.append([float(x) for x in parts[4:8]])
            dimensions.append([float(x) for x in parts[8:11]])
            locations.append([float(x) for x in parts[11:14]])
            rotation_y.append(float(parts[14]))
    return {
        "name": np.array(names, dtype=object),
        "truncated": np.array(truncated, dtype=np.float32),
        "occluded": np.array(occluded, dtype=np.int64),
        "alpha": np.array(alpha, dtype=np.float32),
        "bbox": np.array(bboxes2d, dtype=np.float32).reshape(-1, 4),
        "dimensions": np.array(dimensions, dtype=np.float32).reshape(-1, 3),
        "location": np.array(locations, dtype=np.float32).reshape(-1, 3),
        "rotation_y": np.array(rotation_y, dtype=np.float32).reshape(-1),
    }
