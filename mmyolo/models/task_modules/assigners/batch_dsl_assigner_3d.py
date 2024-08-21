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
class BatchDynamicSoftLabelAssigner3D(nn.Module):
    def __init__(
            self,
            num_classes,
            soft_center_radius: float = 3.0,
            topk: int = 13,
            iou_weight: float = 3.0,
            iou_type='3d',
            iou3d_compensation=0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        self.iou_type = iou_type
        self.iou3d_compensation = iou3d_compensation

    @torch.no_grad()
    # @profile
    def forward(self, pred_bboxes: Tensor, pred_scores: Tensor, priors: Tensor,
                gt_labels: Tensor, gt_bboxes: Tensor, gt_center, gt_bboxes_3d: Tensor,
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
        # (N_points, B, N_boxes) -> (B, N_points, N_boxes)
        is_in_gts = is_in_gts.permute(1, 0, 2)
        # (B, N_points)
        # valid_mask = (is_in_gts.sum(dim=-1, keepdim=True) > 0)  # (batch_size, n_points, 1)

        # gt_center = get_box_center(gt_bboxes, box_dim)

        strides = priors[None, :, None, 2:]  # (1, n_points, 1, 2)
        distance = torch.norm((prior_center.view(1, -1, 1, 2) - gt_center[:, None, :, :]) / strides, dim=-1,
                              keepdim=False)

        # prevent overflow
        # distance = distance * valid_mask  # (batch_size, n_points, n_boxes)
        distance = distance * is_in_gts
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)  # (batch_size, n_points, n_boxes)
        if self.iou_type == '3d':
            pairwise_ious = boxes_iou_3d_gpu(decoded_bboxes, gt_bboxes_3d)  # (batch_size, n_points, n_boxes)
        elif self.iou_type == 'dist':
            iou_heatmap3d = torch.norm((decoded_bboxes[:, :, None, [0, 2]] - gt_bboxes_3d[:, None, :, [0, 2]]), dim=3,
                                       keepdim=False)
            #iou_heatmap3d = (2.5 - torch.log2(iou_heatmap3d)) / 4
            iou_heatmap3d = 1 - iou_heatmap3d / 4
            pairwise_ious = torch.clamp(iou_heatmap3d, 0, 1)
        else:
            raise NotImplementedError
        # pairwise_ious = (pairwise_ious + 0.15).clamp_max(1) * is_in_gts
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight  # (batch_size, n_points, n_boxes)

        # select the predicted scores corresponded to the gt_labels
        pairwise_pred_scores = torch.gather(pred_scores, dim=2,
                                            index=gt_labels.squeeze(-1).unsqueeze(1).expand(batch_size, num_bboxes,
                                                                                            num_gt))
        # debug_pairwise_pred_scores2 = pairwise_pred_scores2.cpu().numpy()
        # debug_pairwise_pred_scores = pairwise_pred_scores.cpu().numpy()
        # debug_gt_labels = gt_labels.cpu().numpy()
        # debug_pred_scores = pred_scores.cpu().numpy()
        # classification cost
        scale_factor = pairwise_ious - pairwise_pred_scores.sigmoid()
        pairwise_cls_cost = F.binary_cross_entropy_with_logits(
            pairwise_pred_scores, pairwise_ious,
            reduction='none') * scale_factor.pow(2.0)

        cost_matrix = pairwise_cls_cost + iou_cost + soft_center_prior  # (batch_size, n_points, n_boxes)

        # max_pad_value = torch.ones_like(cost_matrix) * INF
        # cost_matrix = torch.where(valid_mask[..., None].repeat(1, 1, num_gt), cost_matrix, max_pad_value)
        cost_matrix = torch.where(is_in_gts, cost_matrix, INF)

        # cost_matrix = cost_matrix + (~valid_mask) * INF
        # (matched_pred_ious, matched_gt_inds, fg_mask_inboxes) = self.dynamic_k_matching(cost_matrix, pairwise_ious, pad_bbox_flag)
        batch_index, pred_index, gt_index = self.dynamic_k_matching(cost_matrix, pairwise_ious, is_in_gts)
        # del pairwise_ious, cost_matrix

        # batch_index = (fg_mask_inboxes > 0).nonzero(as_tuple=True)[0]

        # assigned_labels[fg_mask_inboxes] = gt_labels[batch_index, matched_gt_inds].squeeze(-1)
        # assigned_labels = assigned_labels.long()

        assigned_labels_weights = gt_bboxes.new_full([batch_size, num_bboxes], 1)

        # assign_metrics = gt_bboxes.new_full([batch_size, num_bboxes], 0)
        # assign_metrics, max_index = torch.max(pairwise_ious, dim=2)  # (batch_size, n_points, )
        _, max_index = torch.min(cost_matrix, dim=2)  # (batch_size, n_points, )
        assign_metrics = torch.gather(pairwise_ious, dim=2, index=max_index.unsqueeze(-1)).squeeze(-1)
        assign_metrics[batch_index, pred_index] = pairwise_ious[batch_index, pred_index, gt_index]

        assigned_labels = torch.gather(gt_labels.squeeze(-1), dim=1, index=max_index)
        assigned_labels[assign_metrics < 1e-7] = self.num_classes
        assigned_labels[batch_index, pred_index] = gt_labels[batch_index, gt_index, 0]

        return dict(
            assigned_labels=assigned_labels,
            assigned_labels_weights=assigned_labels_weights,
            assigned_batch_index=batch_index,
            assigned_pred_index=pred_index,
            assigned_gt_index=gt_index,
            assign_metrics=assign_metrics)

    # @profile
    def dynamic_k_matching(
            self, cost_matrix: Tensor, pairwise_ious: Tensor,
            is_in_gts: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets.

        Args:
            cost_matrix (Tensor,  (batch_size, n_points, n_boxes)): Cost matrix.
            pairwise_ious (Tensor,  (batch_size, n_points, n_boxes)): Pairwise iou matrix.
            is_in_gts (Tensor,  (batch_size, n_points, n_boxes)):
        Returns:
            tuple: matched ious and gt indexes.
        """
        # matching_matrix = torch.zeros_like(cost_matrix, dtype=torch.uint8)  # (batch_size, n_points, n_boxes)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(1))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=1)  # (batch_size, top_k, n_boxes)
        # calculate dynamic k for each gt
        if self.iou_type == '3d':
            topk_ious = topk_ious + self.iou3d_compensation
            #topk_ious = topk_ious.clamp_max(1)
        else:
            topk_ious = topk_ious# - 0.15
            #topk_ious = topk_ious.clamp_max(1)
        dynamic_ks = torch.clamp((topk_ious.sum(1)).long(), min=1)  # (batch_size, n_boxes)
        message_hub = MessageHub.get_current_instance()
        message_hub.update_scalar('train/dynamic_k', dynamic_ks.float().mean())
        # num_gts = pad_bbox_flag.sum((1, 2)).int()  # (batch_size, )
        # sorting the batch cost matirx is faster than topk
        # _, sorted_indices = torch.sort(cost_matrix, dim=1)  # (batch_size, n_points, n_boxes)
        # for b in range(pad_bbox_flag.shape[0]):
        #     for gt_idx in range(num_gts[b]):
        #         topk_ids = sorted_indices[b, :dynamic_ks[b, gt_idx], gt_idx]
        #         matching_matrix[b, :, gt_idx][topk_ids] = 1
        # dynamic_ks = torch.ones_like(dynamic_ks) * candidate_topk
        sorted_cost, _ = torch.sort(cost_matrix, dim=1)  # (batch_size, n_points, n_boxes)
        cost_k = torch.gather(sorted_cost, dim=1, index=dynamic_ks.unsqueeze(1))
        cost_k_1 = torch.gather(sorted_cost, dim=1, index=(dynamic_ks - 1).unsqueeze(1))
        cost_threshold = 0.5 * (cost_k + cost_k_1)  # (batch_size, 1, n_boxes)
        cost_threshold = cost_threshold * is_in_gts  # (batch_size, n_points, n_boxes)
        matching_matrix = (cost_matrix <= cost_threshold)  # (batch_size, n_points, n_boxes)
        cost_matrix = torch.where(matching_matrix, cost_matrix, INF)
        # a = ((matching_matrix - matching_matrix2.float()).abs().sum())
        # if a > 0:
        #     print(cost_matrix.max())
        #     print(torch.where(matching_matrix))
        #     print(torch.where(matching_matrix2))
        #     debug_matching_matrix = matching_matrix.float().cpu().numpy()
        #     debug_matching_matrix2 = matching_matrix2.float().cpu().numpy()
        #     debug_matching_matrix_delta = debug_matching_matrix - debug_matching_matrix2
        #     debug_sorted_cost = sorted_cost.cpu().numpy()
        #     debug_cost_matrix = cost_matrix.cpu().numpy()
        #     debug_cost_threshold = cost_threshold.cpu().numpy()
        #     print(a)

        # del topk_ious, dynamic_ks

        prior_match_gt_mask = matching_matrix.sum(2) > 1  # (batch_size, n_points,)
        if torch.any(prior_match_gt_mask):
            _, cost_argmin = torch.min(cost_matrix[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] = False
            matching_matrix[prior_match_gt_mask, cost_argmin] = True

        # get foreground mask inside box and center prior
        batch_index, pred_index, gt_index = torch.nonzero(matching_matrix, as_tuple=True)
        # if batch_index.shape[0] > 1000:
        #     debug_matching_matrix = matching_matrix.float().cpu().numpy()
        #     debug_sorted_cost = sorted_cost.cpu().numpy()
        #     debug_cost_matrix = cost_matrix.cpu().numpy()
        #     debug_cost_threshold = cost_threshold.cpu().numpy()
        #     print(batch_index.shape[0])
        return batch_index, pred_index, gt_index
        # fg_mask_inboxes = matching_matrix.sum(2) > 0
        # matched_pred_ious = (matching_matrix * pairwise_ious).sum(2)[fg_mask_inboxes]
        # matched_gt_inds = matching_matrix[fg_mask_inboxes, :].long().argmax(1)
        # return matched_pred_ious, matched_gt_inds, fg_mask_inboxes
