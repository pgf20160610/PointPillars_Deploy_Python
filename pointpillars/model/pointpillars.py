
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointpillars.model.anchors import Anchors, anchor_target, anchors2bboxes
from pointpillars.ops import Voxelization, nms_cuda
from pointpillars.utils import limit_period


class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )

    @torch.no_grad()
    def forward(self, batched_pts):
        pillars, coors, npoints_per_pillar = [], [], []
        for pts in batched_pts:
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts)
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out.long())
        if len(pillars) == 0:
            raise ValueError("batched_pts must not be empty")
        pillars = torch.cat(pillars, dim=0)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0)
        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = int(out_channel)
        self.vx, self.vy = float(voxel_size[0]), float(voxel_size[1])
        self.x_offset = float(voxel_size[0]) / 2.0 + float(point_cloud_range[0])
        self.y_offset = float(voxel_size[1]) / 2.0 + float(point_cloud_range[1])
        self.x_l = int((float(point_cloud_range[3]) - float(point_cloud_range[0])) / float(voxel_size[0]))
        self.y_l = int((float(point_cloud_range[4]) - float(point_cloud_range[1])) / float(voxel_size[1]))
        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        device = pillars.device
        if pillars.numel() == 0:
            bs = int(coors_batch[-1, 0].item() + 1) if coors_batch.numel() else 1
            return torch.zeros((bs, self.out_channel, self.y_l, self.x_l), dtype=torch.float32, device=device)
        npoints = torch.clamp(npoints_per_pillar.to(pillars.dtype), min=1.0)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints[:, None, None]
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2].to(pillars.dtype) * self.vx + self.x_offset)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3].to(pillars.dtype) * self.vy + self.y_offset)
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1)
        # The upstream repo overwrites x/y with pillar-center offsets to match mmdet3d behavior.
        features[:, :, 0:1] = x_offset_pi_center
        features[:, :, 1:2] = y_offset_pi_center

        voxel_ids = torch.arange(0, pillars.size(1), device=device)
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]
        mask = mask.permute(1, 0).contiguous()
        features = features * mask[:, :, None].to(features.dtype)

        features = features.permute(0, 2, 1).contiguous()
        features = F.relu(self.bn(self.conv(features)))
        pooling_features = torch.max(features, dim=-1)[0]

        bs = int(coors_batch[-1, 0].item() + 1) if coors_batch.numel() else 1
        batched_canvas = []
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]
            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            if cur_coors.numel() > 0:
                x_idx = cur_coors[:, 1].clamp(0, self.x_l - 1)
                y_idx = cur_coors[:, 2].clamp(0, self.y_l - 1)
                canvas[x_idx, y_idx] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        return torch.stack(batched_canvas, dim=0)


class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=(2, 2, 2)):
        super().__init__()
        assert len(out_channels) == len(layer_nums) == len(layer_strides)
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        outs = []
        for block in self.multi_blocks:
            x = block(x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides) == len(out_channels)
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            self.decoder_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels[i], out_channels[i], upsample_strides[i], stride=upsample_strides[i], bias=False),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ))
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        outs = [block(xi) for block, xi in zip(self.decoder_blocks, x)]
        return torch.cat(outs, dim=1)


class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()
        self.conv_cls = nn.Conv2d(in_channel, n_anchors * n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors * 7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors * 2, 1)
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        return self.conv_cls(x), self.conv_reg(x), self.conv_dir_cls(x)


class PointPillars(nn.Module):
    def __init__(
        self,
        nclasses=3,
        voxel_size=(0.16, 0.16, 4),
        point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
        max_num_points=32,
        max_voxels=(16000, 40000),
    ):
        super().__init__()
        self.nclasses = int(nclasses)
        self.pillar_layer = PillarLayer(voxel_size=voxel_size, point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points, max_voxels=max_voxels)
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size, point_cloud_range=point_cloud_range,
                                            in_channel=9, out_channel=64)
        self.backbone = Backbone(in_channel=64, out_channels=[64, 128, 256], layer_nums=[3, 5, 5])
        self.neck = Neck(in_channels=[64, 128, 256], upsample_strides=[1, 2, 4], out_channels=[128, 128, 128])
        self.head = Head(in_channel=384, n_anchors=2 * self.nclasses, n_classes=self.nclasses)

        ranges = [[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                  [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                  [0, -39.68, -1.78, 69.12, 39.68, -1.78]]
        sizes = [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]]
        rotations = [0, 1.57]
        self.anchors_generator = Anchors(ranges=ranges[:self.nclasses], sizes=sizes[:self.nclasses], rotations=rotations)

        self.assigners = [
            {"pos_iou_thr": 0.5, "neg_iou_thr": 0.35, "min_iou_thr": 0.35},
            {"pos_iou_thr": 0.5, "neg_iou_thr": 0.35, "min_iou_thr": 0.35},
            {"pos_iou_thr": 0.6, "neg_iou_thr": 0.45, "min_iou_thr": 0.45},
        ][:self.nclasses]
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)

        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        topk = min(int(self.nms_pre), bbox_cls_pred.shape[0])
        inds = bbox_cls_pred.max(1)[0].topk(topk)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        bbox_pred = anchors2bboxes(anchors, bbox_pred)
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                  bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                  bbox_pred[:, 6:]], dim=-1)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            cur_scores = bbox_cls_pred[:, i]
            score_inds = cur_scores > self.score_thr
            if score_inds.sum() == 0:
                continue
            cur_scores = cur_scores[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_dir = bbox_dir_cls_pred[score_inds]
            keep_inds = nms_cuda(cur_bbox_pred2d, cur_scores, self.nms_thr, pre_maxsize=None, post_max_size=None)
            cur_scores = cur_scores[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_dir = cur_dir[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred)
            cur_bbox_pred[:, -1] += (1 - cur_dir.to(cur_bbox_pred.dtype)) * np.pi
            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_scores)

        if not ret_bboxes:
            return {"lidar_bboxes": np.zeros((0, 7), dtype=np.float32),
                    "labels": np.zeros((0,), dtype=np.int64),
                    "scores": np.zeros((0,), dtype=np.float32)}
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        return {
            "lidar_bboxes": ret_bboxes.detach().cpu().numpy(),
            "labels": ret_labels.detach().cpu().numpy(),
            "scores": ret_scores.detach().cpu().numpy(),
        }

    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        return [self.get_predicted_bboxes_single(bbox_cls_pred[i], bbox_pred[i], bbox_dir_cls_pred[i], batched_anchors[i])
                for i in range(bbox_cls_pred.size(0))]

    def forward(self, batched_pts, mode="test", batched_gt_bboxes=None, batched_gt_labels=None):
        batch_size = len(batched_pts)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        xs = self.backbone(pillar_features)
        x = self.neck(xs)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)

        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size).to(device)
        batched_anchors = [anchors for _ in range(batch_size)]

        if mode == "train":
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors,
                                               batched_gt_bboxes=batched_gt_bboxes,
                                               batched_gt_labels=batched_gt_labels,
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        if mode in ("val", "test"):
            return self.get_predicted_bboxes(bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors)
        if mode == "raw":
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors
        raise ValueError(f"unsupported mode: {mode}")
