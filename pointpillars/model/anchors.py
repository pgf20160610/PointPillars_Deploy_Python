
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch


def limit_period(val, offset=0.5, period=math.pi):
    if isinstance(val, torch.Tensor):
        return val - torch.floor(val / period + offset) * period
    return val - math.floor(val / period + offset) * period


class Anchors:
    """Anchor generator reproduced for the zhulf0804/PointPillars layout.

    Anchor format is [x, y, z, w, l, h, yaw].
    Output shape is (feature_h, feature_w, num_classes, num_rotations, 7).
    """

    def __init__(self, ranges, sizes, rotations):
        self.ranges = torch.tensor(ranges, dtype=torch.float32)
        self.sizes = torch.tensor(sizes, dtype=torch.float32)
        self.rotations = torch.tensor(rotations, dtype=torch.float32)

    def get_multi_anchors(self, feature_map_size: torch.Tensor | Sequence[int]):
        if isinstance(feature_map_size, torch.Tensor):
            device = feature_map_size.device
            h, w = int(feature_map_size[0].item()), int(feature_map_size[1].item())
        else:
            device = self.ranges.device
            h, w = int(feature_map_size[0]), int(feature_map_size[1])
        ranges = self.ranges.to(device)
        sizes = self.sizes.to(device)
        rotations = self.rotations.to(device)
        anchors_per_class = []
        for class_id in range(sizes.shape[0]):
            x_min, y_min, z_min, x_max, y_max, z_max = ranges[class_id]
            # This follows the original repository's mmdet3d-style range generator:
            # generated anchors cover the given anchor range on the feature map grid.
            xs = torch.linspace(x_min, x_max, w, dtype=torch.float32, device=device)
            ys = torch.linspace(y_min, y_max, h, dtype=torch.float32, device=device)
            if hasattr(torch, 'meshgrid'):
                yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            else:
                yy, xx = torch.meshgrid(ys, xs)
            zz = torch.full_like(xx, float(z_min))
            size = sizes[class_id]
            anchors_rot = []
            for rot in rotations:
                base = torch.stack([
                    xx,
                    yy,
                    zz,
                    torch.full_like(xx, float(size[0])),
                    torch.full_like(xx, float(size[1])),
                    torch.full_like(xx, float(size[2])),
                    torch.full_like(xx, float(rot)),
                ], dim=-1)
                anchors_rot.append(base)
            anchors_per_class.append(torch.stack(anchors_rot, dim=2))  # (h,w,num_rot,7)
        return torch.stack(anchors_per_class, dim=2)  # (h,w,num_class,num_rot,7)


def anchors2bboxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Decode box deltas to LiDAR boxes.

    anchors and output use [x, y, z, w, l, h, yaw]. This is the source-repo
    formula used by the test-time decoder.
    """
    anchors = anchors.to(deltas.device).type_as(deltas)
    xa, ya, za, wa, la, ha, ra = anchors.unbind(dim=-1)
    xt, yt, zt, wt, lt, ht, rt = deltas.unbind(dim=-1)

    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    za_center = za + ha / 2.0

    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg_center = zt * ha + za_center
    wg = torch.exp(wt) * wa
    lg = torch.exp(lt) * la
    hg = torch.exp(ht) * ha
    rg = rt + ra
    zg = zg_center - hg / 2.0
    return torch.stack([xg, yg, zg, wg, lg, hg, rg], dim=-1)


# Training target generation is intentionally a lightweight placeholder in this
# deployment-focused rewrite. Inference/export do not call this function.
def anchor_target(*args, **kwargs):
    raise NotImplementedError(
        "anchor_target is not included in this deployment rewrite. Use the original "
        "training repo for dataset training, or implement target assignment here."
    )
