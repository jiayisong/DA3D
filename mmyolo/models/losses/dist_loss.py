# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.losses.utils import weighted_loss
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from mmyolo.registry import MODELS


@weighted_loss
def dist_loss(pred, gt):
    loss = torch.norm(pred - gt, p=2, dim=-1, keepdim=True)
    return loss


@MODELS.register_module()
class DistLoss(nn.Module):
    """Multi-Bin Loss for orientation.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'none'.
        loss_weight (float, optional): The weight of loss. Defaults
            to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DistLoss, self).__init__()
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
        loss = self.loss_weight * dist_loss(
            pred, target, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss
