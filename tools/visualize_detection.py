#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

try:
    import cv2
    import numpy as np
except Exception as exc:  # pragma: no cover
    print(f"error: OpenCV visualization requires cv2/numpy: {exc}", file=sys.stderr)
    print("install with: python3 -m pip install opencv-python numpy", file=sys.stderr)
    raise SystemExit(2)


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
COLORS = [(255, 0, 0), (0, 0, 255), (0, 192, 0), (255, 0, 255), (255, 255, 0)]
CLASS_COLORS = {
    "Pedestrian": (0, 0, 255),
    "Cyclist": (0, 192, 0),
    "Car": (255, 0, 0),
}
GT_COLOR = (0, 176, 176)


def box_lidar_dims(box: dict) -> tuple[float, float, float]:
    """Return source-style LiDAR dimensions (w, l, h).

    The Python/source path writes boxes as [x, y, z, w, l, h, yaw], while some
    deployment JSONs use dx/dy/dz aliases.  Keep both formats visualizable, but
    always interpret the result as PointPillars' source convention.
    """
    if all(k in box for k in ("w", "l", "h")):
        return float(box["w"]), float(box["l"]), float(box["h"])
    return float(box["dx"]), float(box["dy"]), float(box["dz"])


def read_points(path: Path) -> np.ndarray:
    pts = np.fromfile(str(path), dtype=np.float32)
    if pts.size == 0 or pts.size % 4 != 0:
        raise ValueError(f"invalid KITTI bin: {path}, float count={pts.size}")
    return pts.reshape(-1, 4)


def read_boxes(path: Path) -> list[dict]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"detection json must be a list: {path}")
    return data


def read_calib(path: Path) -> dict[str, np.ndarray]:
    vals: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, rest = line.split(":", 1)
        arr = np.array([float(x) for x in rest.split()], dtype=np.float32)
        vals[key] = arr
    return vals


def box_corners_lidar(box: dict) -> np.ndarray:
    x, y, z = float(box["x"]), float(box["y"]), float(box["z"])
    w, length, h = box_lidar_dims(box)
    yaw = float(box.get("yaw", 0.0))
    # LiDAR boxes are [x, y, z, w, l, h, yaw] and z is the bottom face, not
    # the geometric center.  Keep bottom corners first so the existing EDGES
    # and BEV rendering remain valid; the XY order matches source BEV corners.
    local = np.array([
        [-w / 2, -length / 2, 0.0], [-w / 2,  length / 2, 0.0], [ w / 2,  length / 2, 0.0], [ w / 2, -length / 2, 0.0],
        [-w / 2, -length / 2, h],   [-w / 2,  length / 2, h],   [ w / 2,  length / 2, h],   [ w / 2, -length / 2, h],
    ], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)
    return local @ rot.T + np.array([x, y, z], dtype=np.float32)


def label_corners_camera(label: dict) -> np.ndarray:
    h, w, length = label["h"], label["w"], label["l"]
    x, y, z = label["x"], label["y"], label["z"]
    ry = label["ry"]
    local = np.array([
        [ length / 2, 0.0,  w / 2], [ length / 2, 0.0, -w / 2], [-length / 2, 0.0, -w / 2], [-length / 2, 0.0,  w / 2],
        [ length / 2,  -h,  w / 2], [ length / 2,  -h, -w / 2], [-length / 2,  -h, -w / 2], [-length / 2,  -h,  w / 2],
    ], dtype=np.float32)
    c, s = math.cos(ry), math.sin(ry)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y, z], dtype=np.float32)


def read_labels(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    labels: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 15 or parts[0] == "DontCare":
            continue
        labels.append({
            "type": parts[0],
            "truncated": float(parts[1]),
            "occluded": int(parts[2]),
            "alpha": float(parts[3]),
            "bbox": [float(v) for v in parts[4:8]],
            "h": float(parts[8]),
            "w": float(parts[9]),
            "l": float(parts[10]),
            "x": float(parts[11]),
            "y": float(parts[12]),
            "z": float(parts[13]),
            "ry": float(parts[14]),
        })
    return labels


def parse_config(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    out: dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = [x.strip() for x in line.split(":", 1)]
        if value.startswith("[") and value.endswith("]"):
            items = [x.strip().strip("'\"") for x in value[1:-1].split(",") if x.strip()]
            out[key] = items
        else:
            out[key] = value.strip("'\"")
    return out


def class_name_for(cls_id: int, class_names: list[str]) -> str:
    if 0 <= cls_id < len(class_names):
        return class_names[cls_id]
    return f"cls_id={cls_id}"


def config_class_names(config: dict[str, object]) -> list[str]:
    value = config.get("class_names", [])
    return [str(x) for x in value] if isinstance(value, list) else []


def calib_mats(calib: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    p2 = calib.get("P2")
    tr = calib.get("Tr_velo_to_cam")
    r0 = calib.get("R0_rect")
    if p2 is None or tr is None or r0 is None:
        return None
    p2m = p2.reshape(3, 4)
    tr4 = np.eye(4, dtype=np.float32); tr4[:3, :] = tr.reshape(3, 4)
    r4 = np.eye(4, dtype=np.float32); r4[:3, :3] = r0.reshape(3, 3)
    return p2m, tr4, r4


def camera_to_lidar(corners_cam: np.ndarray, calib: dict[str, np.ndarray]) -> np.ndarray | None:
    mats = calib_mats(calib)
    if mats is None:
        return None
    _, tr4, r4 = mats
    cam_to_lidar = np.linalg.inv(r4 @ tr4)
    homo = np.concatenate([corners_cam, np.ones((corners_cam.shape[0], 1), dtype=np.float32)], axis=1).T
    lidar = cam_to_lidar @ homo
    return lidar[:3, :].T


def bev_project_xy(x: float, y: float, width: int, height: int, x_range: tuple[float, float], y_range: tuple[float, float], view: str) -> tuple[int, int]:
    if view == "forward-down":
        u = width - 1 - (y - y_range[0]) / (y_range[1] - y_range[0]) * (width - 1)
        v = (x - x_range[0]) / (x_range[1] - x_range[0]) * (height - 1)
    elif view == "left-up":
        u = (x - x_range[0]) / (x_range[1] - x_range[0]) * (width - 1)
        v = height - 1 - (y - y_range[0]) / (y_range[1] - y_range[0]) * (height - 1)
    elif view == "right-up":
        u = width - 1 - (x - x_range[0]) / (x_range[1] - x_range[0]) * (width - 1)
        v = height - 1 - (y - y_range[0]) / (y_range[1] - y_range[0]) * (height - 1)
    else:  # forward-up: LiDAR x points upward, LiDAR y positive/left maps to image left.
        u = width - 1 - (y - y_range[0]) / (y_range[1] - y_range[0]) * (width - 1)
        v = height - 1 - (x - x_range[0]) / (x_range[1] - x_range[0]) * (height - 1)
    return int(u), int(v)


def draw_reference_legend(img: np.ndarray, legend_x: int, class_names: list[str]) -> None:
    cv2.rectangle(img, (legend_x, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 0), -1)
    items = [
        ("Pedestrian:", CLASS_COLORS["Pedestrian"]),
        ("Cyclist:", CLASS_COLORS["Cyclist"]),
        ("Car:", CLASS_COLORS["Car"]),
        ("Ground truth:", GT_COLOR),
    ]
    y0 = max(120, img.shape[0] // 3)
    for i, (text, color) in enumerate(items):
        cv2.putText(img, text, (legend_x + 60, y0 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    if class_names:
        cv2.putText(img, "Config classes: " + ",".join(class_names), (legend_x + 40, img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)


def draw_sensor_axes(img: np.ndarray, bev_width: int, bev_height: int, view: str) -> None:
    origin = (bev_width // 2, bev_height - 45) if view.startswith("forward") else (60, bev_height // 2)
    ox, oy = origin
    cv2.circle(img, origin, 7, (80, 80, 80), -1, cv2.LINE_AA)
    cv2.arrowedLine(img, origin, (ox, max(15, oy - 90)), (0, 0, 255), 5, cv2.LINE_AA, tipLength=0.25)
    y_tip_x = max(15, ox - 90) if view.startswith("forward") else ox
    cv2.arrowedLine(img, origin, (y_tip_x, oy), (0, 255, 0), 5, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(img, "x/front", (ox + 8, max(20, oy - 95)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1, cv2.LINE_AA)
    cv2.putText(img, "y/left", (max(5, ox - 115), oy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1, cv2.LINE_AA)


def render_bev(points: np.ndarray, boxes: list[dict], labels: list[dict], calib: dict[str, np.ndarray] | None, output: Path, width: int, height: int, x_range: tuple[float, float], y_range: tuple[float, float], view: str, reference_style: bool, class_names: list[str]) -> None:
    legend_width = 520 if reference_style else 0
    bev_width = width - legend_width
    if bev_width <= 0:
        raise ValueError("--bev-width is too small for --bev-reference-style")
    bg = 255 if reference_style else 0
    img = np.full((height, width, 3), bg, dtype=np.uint8)
    canvas = img[:, :bev_width]
    mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    pts = points[mask]
    if pts.size:
        uv = np.array([bev_project_xy(float(p[0]), float(p[1]), bev_width, height, x_range, y_range, view) for p in pts], dtype=np.int32)
        u, v = uv[:, 0], uv[:, 1]
        if reference_style:
            inten = np.clip(255.0 - pts[:, 3] * 180.0, 0, 200).astype(np.uint8)
        else:
            inten = np.clip(pts[:, 3] * 255.0, 60, 255).astype(np.uint8)
        canvas[v, u] = np.stack([inten, inten, inten], axis=1)
    for box in boxes:
        cls_name = class_name_for(int(box.get("cls_id", 0)), class_names)
        color = CLASS_COLORS.get(cls_name, COLORS[int(box.get("cls_id", 0)) % len(COLORS)])
        corners = box_corners_lidar(box)[:4]
        uv = [bev_project_xy(float(p[0]), float(p[1]), bev_width, height, x_range, y_range, view) for p in corners]
        cv2.polylines(canvas, [np.array(uv, dtype=np.int32)], True, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Pred {cls_name} {box.get('score', 0):.2f}", tuple(uv[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if calib is not None:
        for label in labels:
            corners_lidar = camera_to_lidar(label_corners_camera(label), calib)
            if corners_lidar is None:
                continue
            uv = [bev_project_xy(float(p[0]), float(p[1]), bev_width, height, x_range, y_range, view) for p in corners_lidar[:4]]
            cv2.polylines(canvas, [np.array(uv, dtype=np.int32)], True, GT_COLOR, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"GT {label['type']}", tuple(uv[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GT_COLOR, 1, cv2.LINE_AA)
    if reference_style:
        draw_sensor_axes(canvas, bev_width, height, view)
        draw_reference_legend(img, bev_width, class_names)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), img)


def diagnose_pedestrian(labels: list[dict], boxes: list[dict], config: dict[str, object], backbone: Path | None) -> None:
    class_names = config_class_names(config)
    num_classes = config.get("num_classes", len(class_names) or "unknown")
    gt_counts = Counter(str(x["type"]) for x in labels)
    det_counts = Counter(class_name_for(int(x.get("cls_id", -1)), class_names) for x in boxes)
    print("[diagnosis] GT label counts:", dict(gt_counts))
    print("[diagnosis] detection counts:", dict(det_counts))
    print(f"[diagnosis] config class_names={class_names or 'unknown'} num_classes={num_classes}")
    if backbone is not None and backbone.exists():
        try:
            import onnx  # type: ignore
            model = onnx.load(str(backbone))
            outs = [(o.name, [d.dim_value or "?" for d in o.type.tensor_type.shape.dim]) for o in model.graph.output]
            print(f"[diagnosis] ONNX outputs: {outs}")
        except Exception as exc:  # pragma: no cover
            print(f"[diagnosis] ONNX output check skipped: {exc}")
    if gt_counts.get("Pedestrian", 0) and "Pedestrian" not in class_names:
        print("[diagnosis] reason: 当前配置/模型类别中没有 Pedestrian；cls_id=0 仅映射为 Car，单类 head 无法输出 Pedestrian 检测框。")
        print("[diagnosis] action: 使用包含 Car/Pedestrian/Cyclist 的训练权重，重新导出匹配的多类 ONNX/MNN，并同步更新 class_names、num_classes 与 anchor sizes。")
    elif gt_counts.get("Pedestrian", 0) and not det_counts.get("Pedestrian", 0):
        print("[diagnosis] reason: 配置包含 Pedestrian 但本帧无 Pedestrian prediction；请继续检查分数阈值、anchor 尺寸/高度、NMS 和模型精度。")


def project_lidar_to_image(corners: np.ndarray, calib: dict[str, np.ndarray]) -> np.ndarray | None:
    mats = calib_mats(calib)
    if mats is None:
        return None
    p2, tr4, r4 = mats
    homo = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=1).T
    cam = r4 @ tr4 @ homo
    if np.any(cam[2, :] <= 0.1):
        return None
    proj = p2 @ cam
    return (proj[:2, :] / proj[2:3, :]).T


def project_lidar_to_image_partial(corners: np.ndarray, calib: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    mats = calib_mats(calib)
    if mats is None:
        return None
    p2, tr4, r4 = mats
    homo = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=1).T
    cam = r4 @ tr4 @ homo
    valid = cam[2, :] > 0.1
    if not np.any(valid):
        return None
    proj = p2 @ cam
    pts = (proj[:2, :] / np.maximum(proj[2:3, :], 1e-6)).T
    return pts, valid


def projected_box_reasonable(img: np.ndarray, pts: np.ndarray, valid: np.ndarray, max_extent_factor: float = 2.5) -> bool:
    h, w = img.shape[:2]
    finite = pts[valid]
    if finite.size == 0 or not np.all(np.isfinite(finite)):
        return False
    x1, y1 = float(finite[:, 0].min()), float(finite[:, 1].min())
    x2, y2 = float(finite[:, 0].max()), float(finite[:, 1].max())
    # Boxes far outside the image can still pass cv2.clipLine and create long
    # misleading lines across the frame. Treat such projections as out-of-FOV.
    if x2 < -w or x1 > 2.0 * w or y2 < -h or y1 > 2.0 * h:
        return False
    if (x2 - x1) > max_extent_factor * w or (y2 - y1) > max_extent_factor * h:
        return False
    return True


def draw_projected_box(img: np.ndarray, pts: np.ndarray, valid: np.ndarray, color: tuple[int, int, int], label: str) -> bool:
    if not projected_box_reasonable(img, pts, valid):
        return False
    h, w = img.shape[:2]
    rect = (0, 0, w, h)
    safe_pts = np.where(np.isfinite(pts), pts, 0.0)
    safe_pts = np.clip(safe_pts, -1.0e6, 1.0e6)
    pi = np.round(safe_pts).astype(np.int32)
    drawn = False
    for a, b in EDGES:
        if not (valid[a] and valid[b]):
            continue
        ok, p0, p1 = cv2.clipLine(rect, tuple(pi[a]), tuple(pi[b]))
        if ok:
            cv2.line(img, p0, p1, color, 3, cv2.LINE_AA)
            drawn = True
    visible = valid & (pi[:, 0] >= 0) & (pi[:, 0] < w) & (pi[:, 1] >= 0) & (pi[:, 1] < h)
    if np.any(visible):
        xs = pi[visible, 0]
        ys = pi[visible, 1]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        drawn = True
    return drawn


def project_camera_to_image(corners_cam: np.ndarray, calib: dict[str, np.ndarray]) -> np.ndarray | None:
    p2 = calib.get("P2")
    if p2 is None:
        return None
    if np.any(corners_cam[:, 2] <= 0.1):
        return None
    p2 = p2.reshape(3, 4)
    homo = np.concatenate([corners_cam, np.ones((corners_cam.shape[0], 1), dtype=np.float32)], axis=1).T
    proj = p2 @ homo
    return (proj[:2, :] / proj[2:3, :]).T


def render_image(image_path: Path, calib_path: Path, boxes: list[dict], labels: list[dict], output: Path) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to read image: {image_path}")
    calib = read_calib(calib_path)
    h, w = img.shape[:2]
    pred_drawn = 0
    pred_skipped_depth = 0
    pred_skipped_fov = 0
    for box in boxes:
        projected = project_lidar_to_image_partial(box_corners_lidar(box), calib)
        if projected is None:
            pred_skipped_depth += 1
            continue
        color = COLORS[int(box.get("cls_id", 0)) % len(COLORS)]
        pts, valid = projected
        if draw_projected_box(img, pts, valid, color, f"Pred {box.get('score', 0):.2f}"):
            pred_drawn += 1
        else:
            pred_skipped_fov += 1
    for label in labels:
        pts = project_camera_to_image(label_corners_camera(label), calib)
        if pts is None:
            continue
        pi = np.round(pts).astype(np.int32)
        if not np.any((pi[:, 0] >= 0) & (pi[:, 0] < w) & (pi[:, 1] >= 0) & (pi[:, 1] < h)):
            continue
        for a, b in EDGES:
            cv2.line(img, tuple(pi[a]), tuple(pi[b]), GT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(img, f"GT {label['type']}", tuple(pi[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), img)
    print(f"image predictions drawn={pred_drawn} skipped_depth={pred_skipped_depth} skipped_fov={pred_skipped_fov}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize PointPillars detections with OpenCV", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--points", default="data/sample.bin", help="KITTI velodyne bin path")
    ap.add_argument("--detections", default="outputs/sample_boxes.json", help="detection JSON path")
    ap.add_argument("--image", default="data/sample.png", help="KITTI image_2 PNG path")
    ap.add_argument("--calib", default="data/sample_calib.txt", help="KITTI calib TXT path")
    ap.add_argument("--label", default="data/sample_label.txt", help="KITTI label_2 TXT path for GT overlay")
    ap.add_argument("--bev-output", default="outputs/pc_pred_sample.png", help="BEV output PNG")
    ap.add_argument("--image-output", default="outputs/img_3dbbox_sample.png", help="camera output PNG")
    ap.add_argument("--no-bev", action="store_true", help="disable BEV output")
    ap.add_argument("--no-image", action="store_true", help="disable camera image output")
    ap.add_argument("--no-label", action="store_true", help="disable KITTI label_2 GT overlay")
    ap.add_argument("--config", default="configs/pointpillars_kitti.yaml", help="deploy config path for class-name diagnostics/legend")
    ap.add_argument("--backbone", default="models/backbone_head_sim.onnx", help="optional Backbone/Head ONNX path for diagnosis")
    ap.add_argument("--diagnose-pedestrian", action="store_true", help="print why Pedestrian GT may not have Pedestrian predictions")
    ap.add_argument("--bev-view", choices=["forward-up", "forward-down", "left-up", "right-up"], default="forward-up", help="BEV orientation mapping")
    ap.add_argument("--bev-reference-style", action="store_true", help="render BEV closer to pc_pred reference: white canvas, right legend, sensor axes")
    ap.add_argument("--bev-width", type=int, default=1786)
    ap.add_argument("--bev-height", type=int, default=1122)
    ap.add_argument("--x-range", nargs=2, type=float, default=[0.0, 70.4], metavar=("MIN", "MAX"))
    ap.add_argument("--y-range", nargs=2, type=float, default=[-40.0, 40.0], metavar=("MIN", "MAX"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        boxes = read_boxes(Path(args.detections))
        print(f"detections: {len(boxes)}")
        labels = [] if args.no_label else read_labels(Path(args.label))
        if not args.no_label:
            print(f"labels: {len(labels)}")
        deploy_config = parse_config(Path(args.config))
        class_names = config_class_names(deploy_config)
        if args.diagnose_pedestrian:
            diagnose_pedestrian(labels, boxes, deploy_config, Path(args.backbone) if args.backbone else None)
        calib = read_calib(Path(args.calib)) if Path(args.calib).exists() else None
        if not args.no_bev:
            points = read_points(Path(args.points))
            render_bev(points, boxes, labels, calib, Path(args.bev_output), args.bev_width, args.bev_height, tuple(args.x_range), tuple(args.y_range), args.bev_view, args.bev_reference_style, class_names)
            print(f"saved BEV: {args.bev_output}")
        if not args.no_image:
            render_image(Path(args.image), Path(args.calib), boxes, labels, Path(args.image_output))
            print(f"saved image: {args.image_output}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())