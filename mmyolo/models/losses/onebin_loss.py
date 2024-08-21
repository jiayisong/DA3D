# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.losses.utils import weighted_loss
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from mmyolo.registry import MODELS


@weighted_loss
def onebin_loss(pred_orientations, gt_orientations):
    """Multi-Bin Loss.

    Args:
        pred_orientations(torch.Tensor): Predicted local vector
            orientation in [axis_cls, head_cls, sin, cos] format.
            shape (N, 2)
        gt_orientations(torch.Tensor): Corresponding gt bboxes,
            shape (N, 1).
        num_dir_bins(int, optional): Number of bins to encode
            direction angle.
            Defaults: 4.

    Return:
        torch.Tensor: Loss tensor.
    """
    alphas = torch.cat((torch.cos(gt_orientations), -torch.sin(gt_orientations), torch.sin(gt_orientations), torch.cos(gt_orientations)), dim=1)
    target = pred_orientations.new_tensor([[0.5, 1],]).expand_as(pred_orientations)
    alphas = alphas.view(-1, 2, 2)
    alpha_pred = torch.bmm(alphas, pred_orientations.unsqueeze(-1)).squeeze(-1)
    alpha_pred = alpha_pred.sigmoid()
    loss = F.binary_cross_entropy(alpha_pred, target, reduction='none')
    return loss


@MODELS.register_module()
class OneBinLoss(nn.Module):
    """Multi-Bin Loss for orientation.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'none'.
        loss_weight (float, optional): The weight of loss. Defaults
            to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(OneBinLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, avg_factor, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            num_dir_bins (int): Number of bins to encode direction angle.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * onebin_loss(
            pred, target,  weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss
