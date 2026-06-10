#!/usr/bin/env python3
"""Export PointPillars PyTorch checkpoint to split ONNX models.

This script is an orchestration layer. For real OpenPCDet/MMDetection3D checkpoints,
pass --external-command (or configure manifest export.external_export_command) to call
your training project's exporter. For toolchain smoke tests, --dummy exports tiny
PyTorch modules with the same deployment input/output contracts.
"""
from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
import sys
# Allow both direct script execution and module-style imports in tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from typing import Dict
from model_manifest import get_section, load_manifest


def format_command(template: str, values: Dict[str, object]) -> list[str]:
    rendered = template.format(**values)
    return shlex.split(rendered)


def run_external(cmd_template: str, values: Dict[str, object], dry_run: bool) -> None:
    cmd = format_command(cmd_template, values)
    this_script = Path(__file__).resolve()
    for token in cmd:
        try:
            candidate = Path(token).resolve()
        except Exception:
            continue
        if candidate == this_script:
            raise SystemExit(
                "external_export_command points back to tools/export_pointpillars_onnx.py, which would recurse. "
                "Set export.mode=\"dummy\" / pass --dummy for smoke tests, or replace "
                "export.external_export_command with your real OpenPCDet/MMDetection3D/PointPillars exporter."
            )
    print("+", " ".join(shlex.quote(x) for x in cmd))
    if not dry_run:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                f"external exporter failed with exit code {e.returncode}: {' '.join(cmd)}\n"
                "If you only want to validate the deployment toolchain, rerun with --dummy "
                "or prepare_models.sh --dummy-export. For real export, set "
                "POINTPILLARS_PYTORCH_ROOT to a training repo that provides a split exporter."
            ) from None


def export_dummy(pfn_output: Path, backbone_output: Path, cfg: Dict[str, object], dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] export dummy PFN -> {pfn_output}")
        print(f"[dry-run] export dummy BackboneHead -> {backbone_output}")
        return
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise RuntimeError("PyTorch is required for --dummy export. Install torch or use --external-command.") from e

    max_pillars = int(cfg.get("max_pillars", 12000))
    max_points = int(cfg.get("max_points", 32))
    feat_dim = int(cfg.get("pillar_feature_dim", 10))
    pfn_out = int(cfg.get("pfn_out_channels", 64))
    bev_c = int(cfg.get("bev_channels", 64))
    bev_h = int(cfg.get("bev_h", 496))
    bev_w = int(cfg.get("bev_w", 432))
    opset = int(cfg.get("opset", 13))

    class DummyPFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(feat_dim, pfn_out)
        def forward(self, pillar_features, pillar_mask):
            x = pillar_features * pillar_mask
            denom = pillar_mask.sum(dim=2).clamp_min(1.0)
            x = x.sum(dim=2) / denom
            return self.linear(x)

    class DummyBackboneHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(bev_c, 16, kernel_size=1)
            self.cls = nn.Conv2d(16, 2, kernel_size=1)
            self.box = nn.Conv2d(16, 14, kernel_size=1)
            self.dir = nn.Conv2d(16, 4, kernel_size=1)
        def forward(self, bev_feature):
            x = torch.relu(self.conv(bev_feature))
            return self.cls(x), self.box(x), self.dir(x)

    torch.manual_seed(int(cfg.get("seed", 2026)))
    pfn_output.parent.mkdir(parents=True, exist_ok=True)
    backbone_output.parent.mkdir(parents=True, exist_ok=True)
    pfn = DummyPFN().eval()
    pillar_features = torch.randn(1, max_pillars, max_points, feat_dim, dtype=torch.float32)
    pillar_mask = torch.ones(1, max_pillars, max_points, 1, dtype=torch.float32)
    torch.onnx.export(
        pfn, (pillar_features, pillar_mask), str(pfn_output),
        input_names=["pillar_features", "pillar_mask"], output_names=["pillar_embed"],
        opset_version=opset, do_constant_folding=True, dynamic_axes=None)

    backbone = DummyBackboneHead().eval()
    bev = torch.randn(1, bev_c, bev_h, bev_w, dtype=torch.float32)
    torch.onnx.export(
        backbone, bev, str(backbone_output),
        input_names=["bev_feature"], output_names=["cls_preds", "box_preds", "dir_cls_preds"],
        opset_version=opset, do_constant_folding=True, dynamic_axes=None)
    print(f"Exported dummy PFN: {pfn_output}")
    print(f"Exported dummy BackboneHead: {backbone_output}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export PointPillars checkpoint to split ONNX models")
    ap.add_argument("--manifest", default="models/model_manifest.json")
    ap.add_argument("--checkpoint", "--ckpt", dest="checkpoint", help="PyTorch checkpoint path")
    ap.add_argument("--config", help="training/deploy config path")
    ap.add_argument("--pfn-output", "--pfn-out", dest="pfn_output", help="output PFN ONNX path")
    ap.add_argument("--backbone-output", "--backbone-out", dest="backbone_output", help="output Backbone/Head ONNX path")
    ap.add_argument("--external-command", help="external exporter command template")
    ap.add_argument("--opset", type=int, default=None, help="ONNX opset override")
    ap.add_argument("--dummy", action="store_true", help="export built-in dummy models for toolchain smoke test")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest) if Path(args.manifest).exists() else {}
    exp = get_section(manifest, "export") if manifest else {}
    checkpoint = args.checkpoint or exp.get("checkpoint", "models/checkpoints/pointpillars.pth")
    config = args.config or exp.get("config", "configs/pointpillars_kitti.yaml")
    pfn_output = Path(args.pfn_output or exp.get("pfn_output", "models/pfn.onnx"))
    backbone_output = Path(args.backbone_output or exp.get("backbone_output", "models/backbone_head.onnx"))
    external = args.external_command or exp.get("external_export_command", "")
    mode = "dummy" if args.dummy else str(exp.get("mode", "external"))

    values = dict(exp)
    values.update({
        "checkpoint": checkpoint,
        "config": config,
        "pfn_output": str(pfn_output),
        "backbone_output": str(backbone_output),
        "opset": int(args.opset if args.opset is not None else exp.get("opset", 13)),
    })

    if mode == "dummy":
        export_dummy(pfn_output, backbone_output, values, args.dry_run)
    else:
        if not external:
            raise SystemExit("external export mode requires --external-command or manifest export.external_export_command")
        if not args.dry_run and not Path(checkpoint).exists():
            raise SystemExit(f"checkpoint not found: {checkpoint}")
        run_external(external, values, args.dry_run)
        if not args.dry_run:
            for out in (pfn_output, backbone_output):
                if not out.exists():
                    raise SystemExit(f"expected ONNX output missing: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
