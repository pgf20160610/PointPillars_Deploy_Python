from __future__ import annotations

import copy
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - allow importing utils without torch installed
    torch = None

try:
    import numba
except Exception:  # pragma: no cover
    class _NoNumba:
        def jit(self, *args, **kwargs):
            def deco(fn): return fn
            return deco
    numba = _NoNumba()


def _as_4x4(mat: np.ndarray, name: str = "mat") -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = mat
        return out
    if mat.shape == (3, 3):
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = mat
        return out
    if mat.size == 12:
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = mat.reshape(3, 4)
        return out
    if mat.size == 9:
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = mat.reshape(3, 3)
        return out
    raise ValueError(f"{name} must be 3x3, 3x4 or 4x4, got {mat.shape}")


def bbox_camera2lidar(bboxes, tr_velo_to_cam, r0_rect):
    """Source-compatible camera-box to lidar-box conversion.

    Input camera bboxes are [x, y, z, h, w, l, ry]. Output lidar bboxes are
    [x, y, z, w, l, h, ry]. This intentionally keeps angle unchanged, matching
    zhulf0804/PointPillars process.py.
    """
    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 7)
    if bboxes.size == 0:
        return bboxes.copy()
    tr_velo_to_cam = _as_4x4(tr_velo_to_cam, "tr_velo_to_cam")
    r0_rect = _as_4x4(r0_rect, "r0_rect")
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)  # [l, h, w] -> [w, l, h] per source variable names
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    bboxes_lidar = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return np.array(bboxes_lidar, dtype=np.float32)


def bbox_lidar2camera(bboxes, tr_velo_to_cam, r0_rect):
    """Source-compatible lidar-box to camera-box conversion.

    Input lidar bboxes are [x, y, z, w, l, h, yaw]. Output camera bboxes are
    [x, y, z, l, h, w, yaw] by the source implementation's size reorder.
    """
    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 7)
    if bboxes.size == 0:
        return bboxes.copy()
    tr_velo_to_cam = _as_4x4(tr_velo_to_cam, "tr_velo_to_cam")
    r0_rect = _as_4x4(r0_rect, "r0_rect")
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([y_size, z_size, x_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = r0_rect @ tr_velo_to_cam
    xyz = extended_xyz @ rt_mat.T
    bboxes_camera = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return bboxes_camera.astype(np.float32)


def points_camera2image(points, P2):
    points = np.asarray(points, dtype=np.float32)
    P2 = np.asarray(P2, dtype=np.float32)
    if P2.shape == (3, 4):
        P = P2
    else:
        P = _as_4x4(P2, "P2")
    extended_points = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0)
    image_points = extended_points @ P.T
    z = image_points[:, :, 2:3]
    image_points = image_points[:, :, :2] / np.clip(z, 1e-6, None)
    return image_points.astype(np.float32)


def points_lidar2image(points, tr_velo_to_cam, r0_rect, P2):
    points = np.asarray(points, dtype=np.float32)
    tr_velo_to_cam = _as_4x4(tr_velo_to_cam, "tr_velo_to_cam")
    r0_rect = _as_4x4(r0_rect, "r0_rect")
    P = _as_4x4(P2, "P2") if np.asarray(P2).shape != (3, 4) else np.asarray(P2, dtype=np.float32)
    extended_points = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0)
    camera_points = extended_points @ (r0_rect @ tr_velo_to_cam).T
    image_points = camera_points @ P.T
    image_points = image_points[:, :, :2] / np.clip(image_points[:, :, 2:3], 1e-6, None)
    return image_points.astype(np.float32)


def points_camera2lidar(points, tr_velo_to_cam, r0_rect):
    points = np.asarray(points, dtype=np.float32)
    tr_velo_to_cam = _as_4x4(tr_velo_to_cam, "tr_velo_to_cam")
    r0_rect = _as_4x4(r0_rect, "r0_rect")
    extended_xyz = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    return xyz[..., :3].astype(np.float32)


def bbox3d2bevcorners(bboxes):
    """Exact-source BEV corners for lidar boxes [x, y, z, w, l, h, theta]."""
    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 7)
    centers, dims, angles = bboxes[:, :2], bboxes[:, 3:5], bboxes[:, 6]
    bev_corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32)
    bev_corners = bev_corners[None, ...] * dims[:, None, :]
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)  # source says: in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin], [-rot_sin, rot_cos]], dtype=np.float32)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))
    bev_corners = bev_corners @ rot_mat
    bev_corners += centers[:, None, :]
    return bev_corners.astype(np.float32)


def bbox3d2corners(bboxes):
    """Exact-source 3D lidar corners for [x, y, z, w, l, h, theta]."""
    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 7)
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]
    bboxes_corners = np.array([
        [-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
        [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]
    ], dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :]
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    rot_mat = np.array([
        [rot_cos, rot_sin, np.zeros_like(rot_cos)],
        [-rot_sin, rot_cos, np.zeros_like(rot_cos)],
        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]
    ], dtype=np.float32)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))
    bboxes_corners = bboxes_corners @ rot_mat
    bboxes_corners += centers[:, None, :]
    return bboxes_corners.astype(np.float32)


def bbox3d2corners_camera(bboxes):
    """Exact-source camera corners for camera boxes [x, y, z, l, h, w, ry]."""
    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 7)
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]
    bboxes_corners = np.array([
        [0.5, 0.0, -0.5], [0.5, -1.0, -0.5], [-0.5, -1.0, -0.5], [-0.5, 0.0, -0.5],
        [0.5, 0.0, 0.5], [0.5, -1.0, 0.5], [-0.5, -1.0, 0.5], [-0.5, 0.0, 0.5]
    ], dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :]
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    rot_mat = np.array([
        [rot_cos, np.zeros_like(rot_cos), rot_sin],
        [np.zeros_like(rot_cos), np.ones_like(rot_cos), np.zeros_like(rot_cos)],
        [-rot_sin, np.zeros_like(rot_cos), rot_cos]
    ], dtype=np.float32)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))
    bboxes_corners = bboxes_corners @ rot_mat
    bboxes_corners += centers[:, None, :]
    return bboxes_corners.astype(np.float32)


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


def keep_bbox_from_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape):
    h, w = image_shape
    lidar_bboxes = np.asarray(result['lidar_bboxes'], dtype=np.float32).reshape(-1, 7)
    labels = np.asarray(result['labels'])
    scores = np.asarray(result['scores'])
    if len(lidar_bboxes) == 0:
        return {'lidar_bboxes': lidar_bboxes, 'labels': labels, 'scores': scores, 'bboxes2d': np.zeros((0,4),np.float32), 'camera_bboxes': lidar_bboxes.copy()}
    camera_bboxes = bbox_lidar2camera(lidar_bboxes, tr_velo_to_cam, r0_rect)
    bboxes_points = bbox3d2corners_camera(camera_bboxes)
    image_points = points_camera2image(bboxes_points, P2)
    image_x1y1 = np.min(image_points, axis=1)
    image_x1y1 = np.maximum(image_x1y1, 0)
    image_x2y2 = np.max(image_points, axis=1)
    image_x2y2 = np.minimum(image_x2y2, [w, h])
    bboxes2d = np.concatenate([image_x1y1, image_x2y2], axis=-1)
    keep_flag = (image_x1y1[:, 0] < w) & (image_x1y1[:, 1] < h) & (image_x2y2[:, 0] > 0) & (image_x2y2[:, 1] > 0)
    return {'lidar_bboxes': lidar_bboxes[keep_flag], 'labels': labels[keep_flag], 'scores': scores[keep_flag], 'bboxes2d': bboxes2d[keep_flag], 'camera_bboxes': camera_bboxes[keep_flag]}


def keep_bbox_from_lidar_range(result, pcd_limit_range):
    lidar_bboxes, labels, scores = result['lidar_bboxes'], result['labels'], result['scores']
    if 'bboxes2d' not in result:
        result['bboxes2d'] = np.zeros((len(lidar_bboxes), 4), dtype=np.float32)
    if 'camera_bboxes' not in result:
        result['camera_bboxes'] = np.zeros_like(lidar_bboxes)
    bboxes2d, camera_bboxes = result['bboxes2d'], result['camera_bboxes']
    pcd_limit_range = np.asarray(pcd_limit_range, dtype=np.float32)
    flag1 = lidar_bboxes[:, :3] > pcd_limit_range[:3][None, :]
    flag2 = lidar_bboxes[:, :3] < pcd_limit_range[3:][None, :]
    keep_flag = np.all(flag1, axis=-1) & np.all(flag2, axis=-1)
    return {'lidar_bboxes': lidar_bboxes[keep_flag], 'labels': labels[keep_flag], 'scores': scores[keep_flag], 'bboxes2d': bboxes2d[keep_flag], 'camera_bboxes': camera_bboxes[keep_flag]}


# ---------------------------------------------------------------------------
# Compatibility helpers expected by pointpillars.utils package
# ---------------------------------------------------------------------------

# Default class mapping used in several tools/scripts
CLASSES = {'Pedestrian': 0, 'Cyclist': 1, 'Car': 2}
LABEL2CLASSES = {v: k for k, v in CLASSES.items()}

# Default point range used by tools (KITTI-like range)
POINT_RANGE = np.array([0, -39.68, -3, 69.12, 39.68, 1], dtype=np.float32)


def setup_seed(seed: int) -> None:
    """Setup RNG seeds for reproducible behavior."""
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        # make deterministic where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def point_range_filter(pts, point_range: np.ndarray | None = None):
    """Filter points by `point_range`. Matches implementation in tools/infer_pointpillars_pytorch.py."""
    if point_range is None:
        point_range = POINT_RANGE
    pts = np.asarray(pts)
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    return pts[keep_mask]


def result_to_json(result, label_map: dict | None = None):
    """Convert a detection `result` dict to a JSON-serializable list of dicts.

    `result` is expected to contain 'lidar_bboxes', 'labels', 'scores'.
    """
    import json as _json

    if label_map is None:
        label_map = LABEL2CLASSES
    boxes = np.asarray(result.get('lidar_bboxes', np.zeros((0, 7))), dtype=np.float32).reshape(-1, 7)
    labels = np.asarray(result.get('labels', np.zeros((0,), dtype=np.int64))).reshape(-1)
    scores = np.asarray(result.get('scores', np.zeros((0,), dtype=np.float32))).reshape(-1)
    data = []
    for i, b in enumerate(boxes):
        label = int(labels[i]) if i < len(labels) else -1
        data.append({
            'x': float(b[0]), 'y': float(b[1]), 'z': float(b[2]),
            'w': float(b[3]), 'l': float(b[4]), 'h': float(b[5]), 'yaw': float(b[6]),
            'score': float(scores[i]) if i < len(scores) else 0.0,
            'cls_id': label,
            'class_name': label_map.get(label, str(label)),
        })
    return data


def save_result_json(path, result, label_map: dict | None = None):
    path = str(path)
    import json as _json
    data = result_to_json(result, label_map=label_map)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(_json.dumps(data, ensure_ascii=False, indent=2) + '\n')

