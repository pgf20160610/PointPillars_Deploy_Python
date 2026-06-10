#!/usr/bin/env python3
"""Export zhulf0804/PointPillars epoch_160.pth to split ONNX models.

This exporter is self-contained for the deployable neural-network parts and does
not depend on the original repo's CUDA voxelization/NMS ops. It reconstructs the
checkpoint modules:
  - PFN: pillar_encoder conv/bn, exported as pillar_features + pillar_mask -> pillar_embed
  - BackboneHead: backbone + neck + head, exported as bev_feature -> raw preds

C++ deployment keeps voxelization, scatter, anchor decode and NMS outside ONNX.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys


class MissingDependency(RuntimeError):
    pass


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except Exception as e:
        raise MissingDependency("PyTorch is required for real ONNX export: python3 -m pip install torch onnx") from e


def strip_prefix_state(state, prefix: str):
    out = OrderedDict()
    plen = len(prefix)
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
    return out


def load_state(ckpt: str):
    torch, _, _ = require_torch()
    obj = torch.load(ckpt, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def build_modules():
    torch, nn, F = require_torch()

    class PFNExport(nn.Module):
        def __init__(self, in_channel=9, out_channel=64):
            super().__init__()
            self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
            self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

        def forward(self, pillar_features, pillar_mask):
            # Deployment C++ already builds 10-dim features:
            # x,y,z,i + cluster xyz + center xyz.
            # zhulf0804 checkpoint PFN was trained with 9 dims:
            # x,y,z,i + cluster xyz + center xy. Drop center_z for weight compatibility.
            x = pillar_features[..., :9] * pillar_mask
            x = x.permute(0, 1, 3, 2).contiguous()  # [B,N,9,P]
            B, N, C, P = x.shape
            x = x.reshape(B * N, C, P)
            x = F.relu(self.bn(self.conv(x)))
            x = torch.max(x, dim=-1)[0]
            return x.reshape(B, N, -1)

    class Backbone(nn.Module):
        def __init__(self, in_channel=64, out_channels=(64, 128, 256), layer_nums=(3, 5, 5), layer_strides=(2, 2, 2)):
            super().__init__()
            self.multi_blocks = nn.ModuleList()
            for i in range(len(layer_strides)):
                blocks = [
                    nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1),
                    nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ]
                for _ in range(layer_nums[i]):
                    blocks += [
                        nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1),
                        nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True),
                    ]
                in_channel = out_channels[i]
                self.multi_blocks.append(nn.Sequential(*blocks))

        def forward(self, x):
            outs = []
            for block in self.multi_blocks:
                x = block(x)
                outs.append(x)
            return outs

    class Neck(nn.Module):
        def __init__(self, in_channels=(64, 128, 256), upsample_strides=(1, 2, 4), out_channels=(128, 128, 128)):
            super().__init__()
            self.decoder_blocks = nn.ModuleList()
            for i in range(len(in_channels)):
                self.decoder_blocks.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels[i], out_channels[i], upsample_strides[i], stride=upsample_strides[i], bias=False),
                    nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ))

        def forward(self, xs):
            return torch.cat([block(x) for block, x in zip(self.decoder_blocks, xs)], dim=1)

    class Head(nn.Module):
        def __init__(self, in_channel=384, n_anchors=6, n_classes=3):
            super().__init__()
            self.conv_cls = nn.Conv2d(in_channel, n_anchors * n_classes, 1)
            self.conv_reg = nn.Conv2d(in_channel, n_anchors * 7, 1)
            self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors * 2, 1)

        def forward(self, x):
            return self.conv_cls(x), self.conv_reg(x), self.conv_dir_cls(x)

    class BackboneHeadExport(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.neck = Neck()
            self.head = Head()

        def forward(self, bev_feature):
            x = self.neck(self.backbone(bev_feature))
            return self.head(x)

    return PFNExport, BackboneHeadExport


def export_real(args) -> int:
    torch, _, _ = require_torch()
    PFNExport, BackboneHeadExport = build_modules()
    state = load_state(args.ckpt)

    pfn = PFNExport().eval()
    missing, unexpected = pfn.load_state_dict(strip_prefix_state(state, "pillar_encoder."), strict=True)
    if missing or unexpected:
        raise SystemExit(f"PFN state mismatch missing={missing} unexpected={unexpected}")

    backbone_head = BackboneHeadExport().eval()
    bh_state = OrderedDict()
    for prefix in ("backbone.", "neck.", "head."):
        for k, v in strip_prefix_state(state, prefix).items():
            bh_state[prefix + k] = v
    missing, unexpected = backbone_head.load_state_dict(bh_state, strict=True)
    if missing or unexpected:
        raise SystemExit(f"BackboneHead state mismatch missing={missing} unexpected={unexpected}")

    pfn_out = Path(args.pfn_out)
    backbone_out = Path(args.backbone_out)
    pfn_out.parent.mkdir(parents=True, exist_ok=True)
    backbone_out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"[dry-run] export real PFN -> {pfn_out}")
        print(f"[dry-run] export real BackboneHead -> {backbone_out}")
        return 0

    pillar_features = torch.randn(1, args.max_pillars, args.max_points, args.feat_dim, dtype=torch.float32)
    pillar_mask = torch.ones(1, args.max_pillars, args.max_points, 1, dtype=torch.float32)
    torch.onnx.export(
        pfn, (pillar_features, pillar_mask), str(pfn_out),
        input_names=["pillar_features", "pillar_mask"], output_names=["pillar_embed"],
        opset_version=args.opset, do_constant_folding=True, dynamic_axes=None)

    bev = torch.randn(1, args.bev_channels, args.bev_h, args.bev_w, dtype=torch.float32)
    torch.onnx.export(
        backbone_head, bev, str(backbone_out),
        input_names=["bev_feature"], output_names=["cls_preds", "box_preds", "dir_cls_preds"],
        opset_version=args.opset, do_constant_folding=True, dynamic_axes=None)

    print(f"Exported PFN ONNX: {pfn_out}")
    print(f"Exported BackboneHead ONNX: {backbone_out}")
    return 0


def export_dummy(args) -> int:
    # Reuse generic dummy exporter implementation.
    import subprocess
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "export_pointpillars_onnx.py"),
        "--dummy", "--checkpoint", args.ckpt, "--config", args.config,
        "--pfn-output", args.pfn_out, "--backbone-output", args.backbone_out,
        "--opset", str(args.opset),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    print("+", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export zhulf0804 PointPillars checkpoint to split ONNX models")
    parser.add_argument("--ckpt", "--checkpoint", dest="ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pfn-out", "--pfn-output", dest="pfn_out", required=True)
    parser.add_argument("--backbone-out", "--backbone-output", dest="backbone_out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-pillars", type=int, default=12000)
    parser.add_argument("--max-points", type=int, default=32)
    parser.add_argument("--feat-dim", type=int, default=10)
    parser.add_argument("--bev-channels", type=int, default=64)
    parser.add_argument("--bev-h", type=int, default=496)
    parser.add_argument("--bev-w", type=int, default=432)
    parser.add_argument("--pointpillars-root", default="", help="Unused for self-contained exporter; kept for compatibility")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dummy:
        return export_dummy(args)
    return export_real(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MissingDependency as e:
        raise SystemExit(str(e))
