#!/usr/bin/env python3
"""ONNXRuntime smoke verification for split PointPillars ONNX models."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
# Allow both direct script execution and module-style imports in tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sys
from typing import Sequence
import numpy as np
from model_manifest import get_section, load_manifest


def parse_providers(items: Sequence[str] | str | None):
    if items is None:
        return ["CPUExecutionProvider"]
    if isinstance(items, str):
        return [x.strip() for x in items.split(",") if x.strip()]
    return list(items)


def assert_finite(name: str, arr: np.ndarray) -> None:
    if arr.size == 0:
        raise RuntimeError(f"output {name} is empty")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"output {name} contains NaN/Inf")


def run_pfn(sess, max_pillars: int, max_points: int, feat_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(1, max_pillars, max_points, feat_dim)).astype(np.float32)
    mask = np.ones((1, max_pillars, max_points, 1), dtype=np.float32)
    outs = sess.run(None, {"pillar_features": features, "pillar_mask": mask})
    names = [o.name for o in sess.get_outputs()]
    for name, out in zip(names, outs):
        assert_finite(name, out)
        print(f"PFN output {name}: shape={out.shape} dtype={out.dtype} mean={float(out.mean()):.6f}")
    return outs


def run_backbone(sess, c: int, h: int, w: int, seed: int):
    rng = np.random.default_rng(seed + 1)
    bev = rng.normal(size=(1, c, h, w)).astype(np.float32)
    outs = sess.run(None, {"bev_feature": bev})
    names = [o.name for o in sess.get_outputs()]
    for name, out in zip(names, outs):
        assert_finite(name, out)
        print(f"Backbone output {name}: shape={out.shape} dtype={out.dtype} mean={float(out.mean()):.6f}")
    return outs


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify PointPillars ONNX models with ONNXRuntime Python")
    ap.add_argument("--manifest", default="models/model_manifest.json")
    ap.add_argument("--pfn", help="PFN ONNX path")
    ap.add_argument("--backbone", help="Backbone/Head ONNX path")
    ap.add_argument("--providers", default=None, help="comma-separated ORT providers")
    ap.add_argument("--max-pillars", type=int, default=None)
    ap.add_argument("--max-points", type=int, default=None)
    ap.add_argument("--feat-dim", type=int, default=None)
    ap.add_argument("--bev-c", type=int, default=None)
    ap.add_argument("--bev-h", type=int, default=None)
    ap.add_argument("--bev-w", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--skip-pfn", action="store_true")
    ap.add_argument("--skip-backbone", action="store_true")
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except Exception as e:
        raise SystemExit("onnxruntime is required: python3 -m pip install onnxruntime") from e

    manifest = load_manifest(args.manifest) if Path(args.manifest).exists() else {}
    exp = get_section(manifest, "export") if manifest else {}
    ver = get_section(manifest, "verify") if manifest else {}
    providers = parse_providers(args.providers or ver.get("providers", ["CPUExecutionProvider"]))
    seed = int(args.seed if args.seed is not None else ver.get("seed", 2026))
    pfn_path = Path(args.pfn or ver.get("pfn_model") or exp.get("pfn_output", "models/pfn.onnx"))
    backbone_path = Path(args.backbone or ver.get("backbone_model") or exp.get("backbone_output", "models/backbone_head.onnx"))

    max_pillars = int(args.max_pillars or exp.get("max_pillars", 12000))
    max_points = int(args.max_points or exp.get("max_points", 32))
    feat_dim = int(args.feat_dim or exp.get("pillar_feature_dim", 10))
    bev_c = int(args.bev_c or exp.get("bev_channels", 64))
    bev_h = int(args.bev_h or exp.get("bev_h", 496))
    bev_w = int(args.bev_w or exp.get("bev_w", 432))

    if not args.skip_pfn:
        if not pfn_path.exists():
            raise SystemExit(f"PFN ONNX not found: {pfn_path}")
        print(f"Verifying PFN: {pfn_path}")
        run_pfn(ort.InferenceSession(str(pfn_path), providers=providers), max_pillars, max_points, feat_dim, seed)
    if not args.skip_backbone:
        if not backbone_path.exists():
            raise SystemExit(f"Backbone ONNX not found: {backbone_path}")
        print(f"Verifying Backbone/Head: {backbone_path}")
        run_backbone(ort.InferenceSession(str(backbone_path), providers=providers), bev_c, bev_h, bev_w, seed)
    print("ONNX verification OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
