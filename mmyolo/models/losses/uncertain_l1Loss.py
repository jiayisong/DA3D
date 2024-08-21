# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.losses.utils import weighted_loss
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from mmyolo.registry import MODELS


@weighted_loss
def uncertain_l1Loss(pred, gt, k=1):
    pred, score = torch.chunk(pred, 2, dim=-1)
    loss = 1.4142 * k * torch.exp(-score) * torch.abs(pred - gt) + score
    return loss


@MODELS.register_module()
class UncertainL1Loss(nn.Module):
    """Multi-Bin Loss for orientation.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'none'.
        loss_weight (float, optional): The weight of loss. Defaults
            to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, k=16.32):
        super(UncertainL1Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.k = k
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
        loss = self.loss_weight * uncertain_l1Loss(
            pred, target, k=self.k, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss
