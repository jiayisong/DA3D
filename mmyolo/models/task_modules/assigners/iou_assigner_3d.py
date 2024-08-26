# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.structures.bbox import BaseBoxes
from mmdet.utils import ConfigType
from torch import Tensor
from .batch_dsl_assigner import BatchDynamicSoftLabelAssigner, find_inside_points, get_box_center
from mmyolo.registry import TASK_UTILS
from cv_ops.bbox3d import boxes_iou_3d_gpu
from mmengine.logging import MessageHub

INF = 100000000
EPS = 1.0e-7


@TASK_UTILS.register_module()
class IOUAssigner3D(nn.Module):
    def __init__(
            self,
            num_classes,
            soft_center_radius: float = 3.0,
            topk: int = 13,
            iou_weight: float = 3.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight

    @torch.no_grad()
    #@profile
    def forward(self, pred_bboxes: Tensor, pred_scores: Tensor, priors: Tensor,
                gt_labels: Tensor, gt_bboxes: Tensor, gt_bboxes_3d: Tensor,
                pad_bbox_flag: Tensor) -> dict:
        num_gt = gt_bboxes.size(1)
        decoded_bboxes = pred_bboxes
        batch_size, num_bboxes, _ = decoded_bboxes.size()
        box_dim = gt_bboxes.shape[-1]
        if num_gt == 0 or num_bboxes == 0:
            return {
                'assigned_labels': gt_labels.new_full([batch_size, num_bboxes], self.num_classes),
                'assigned_labels_weights': gt_bboxes.new_full([batch_size, num_bboxes], 1),
                'assign_metrics': gt_bboxes.new_full([batch_size, num_bboxes], 0),
                'assigned_batch_index': gt_labels.new_full([0, ], 0),
                'assigned_pred_index': gt_labels.new_full([0, ], 0),
                'assigned_gt_index': gt_labels.new_full([0, ], 0),
            }
        prior_center = priors[:, :2]
        if isinstance(gt_bboxes, BaseBoxes):
            raise NotImplementedError(
                f'type of {type(gt_bboxes)} are not implemented !')
        else:
            is_in_gts = find_inside_points(gt_bboxes, prior_center, box_dim)  # (n_points, batch_size, n_boxes)
        # (N_points, B, N_boxes)
        is_in_gts = is_in_gts * pad_bbox_flag[..., 0][None]
        is_in_gts = is_in_gts.permute(1, 0, 2) # (N_points, B, N_boxes) -> (B, N_points, N_boxes)
        batch_index, pred_index, gt_index = torch.nonzero(is_in_gts, as_tuple=True)
        decoded_bboxes = decoded_bboxes[batch_index, pred_index, :]
        # pred_scores = pred_scores[batch_index, pred_index, :]
        gt_bboxes_3d = gt_bboxes_3d[batch_index, gt_index, :]
        # gt_labels = gt_labels[batch_index, gt_index]
        pairwise_ious = boxes_iou_3d_gpu(decoded_bboxes.unsqueeze(1), gt_bboxes_3d.unsqueeze(1)).view(-1)
        ass_gt_num_per_pred = is_in_gts.sum(2)  # (B, N_points,)
        is_in_mul_gts = ass_gt_num_per_pred > 1
        assign_metrics = gt_bboxes.new_full([batch_size, num_bboxes], 0)
        if torch.any(is_in_mul_gts):
            cost_mat = is_in_gts.float()
            cost_mat[batch_index, pred_index, gt_index] = pairwise_ious
            cost_max, cost_argmax = torch.max(cost_mat[is_in_mul_gts, :], dim=1)
            cost_mat[is_in_mul_gts, :] = 0
            cost_mat[is_in_mul_gts, cost_argmax] = cost_max
            batch_index, pred_index, gt_index = torch.nonzero(cost_mat, as_tuple=True)
            assign_metrics[batch_index, pred_index] = cost_mat[batch_index, pred_index, gt_index]
        else:
            assign_metrics[batch_index, pred_index] = pairwise_ious
        assigned_labels = gt_labels.new_full([batch_size, num_bboxes], self.num_classes)
        assigned_labels[batch_index, pred_index] = gt_labels[batch_index, gt_index, 0]
        assigned_labels_weights = gt_bboxes.new_full([batch_size, num_bboxes], 1)
        return dict(
            assigned_labels=assigned_labels,
            assigned_labels_weights=assigned_labels_weights,
            assigned_batch_index=batch_index,
            assigned_pred_index=pred_index,
            assigned_gt_index=gt_index,
            assign_metrics=assign_metrics)