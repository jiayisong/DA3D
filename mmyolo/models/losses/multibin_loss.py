# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.losses.utils import weighted_loss
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from mmyolo.registry import MODELS


@weighted_loss
def multibin_loss(pred_orientations, gt_orientations, bin_margin):
    """Multi-Bin Loss.

    Args:
        pred_orientations(torch.Tensor): Predicted local vector
            orientation in [axis_cls, head_cls, sin, cos] format.
            shape (N, num_dir_bins * 4)
        gt_orientations(torch.Tensor): Corresponding gt bboxes,
            shape (N, 1).
        num_dir_bins(int, optional): Number of bins to encode
            direction angle.
            Defaults: 4.

    Return:
        torch.Tensor: Loss tensor.
    """
    N, num_dir_bins = pred_orientations.shape
    num_dir_bins = int(num_dir_bins / 4)
    if num_dir_bins == 0:
        gt_reg = torch.cat([torch.sin(gt_orientations), torch.cos(gt_orientations)], dim=1)
        reg_loss = F.l1_loss(pred_orientations, gt_reg, reduction='none')
        return reg_loss
    angle_per_class = 2 * np.pi / num_dir_bins
    angle_center = torch.arange(num_dir_bins, device=pred_orientations.device) * angle_per_class
    angle_res = gt_orientations - angle_center.unsqueeze(0)
    angle_res = (angle_res + np.pi) % (2 * np.pi) - np.pi
    angle_cls = (angle_res.abs() < (angle_per_class * (0.5 + bin_margin / 2)))
    gt_cls = angle_cls.long()
    gt_reg = torch.stack([torch.sin(angle_res), torch.cos(angle_res)], dim=2) # (N, num_dir_bins, 2)

    pred_cls = pred_orientations[:, :num_dir_bins * 2].view(N, num_dir_bins, 2)
    pred_reg = pred_orientations[:, num_dir_bins * 2:].view(N, num_dir_bins, 2)
    pred_reg = F.normalize(pred_reg, dim=-1)

    cls_loss = F.cross_entropy(pred_cls.transpose(1, 2), gt_cls, reduction='none') / num_dir_bins
    # pred_cls_debug = pred_cls.transpose(1, 2).detach().cpu().numpy()
    # gt_cls_debug = gt_cls.detach().cpu().numpy()
    # cls_loss_debug = cls_loss.detach().cpu().numpy()

    reg_loss = F.l1_loss(pred_reg, gt_reg, reduction='none') * gt_cls.unsqueeze(2)
    reg_loss = reg_loss / gt_cls.sum(1, keepdim=True).unsqueeze(2)
    reg_loss = reg_loss.sum(2)
    return cls_loss + reg_loss


@MODELS.register_module()
class MultiBinLoss(nn.Module):
    """Multi-Bin Loss for orientation.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'none'.
        loss_weight (float, optional): The weight of loss. Defaults
            to 1.0.
    """

    def __init__(self, bin_margin, reduction='mean', loss_weight=1.0):
        super(MultiBinLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.bin_margin = bin_margin

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
        loss = self.loss_weight * multibin_loss(
            pred, target, bin_margin=self.bin_margin, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss
