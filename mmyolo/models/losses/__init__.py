# Copyright (c) OpenMMLab. All rights reserved.
from .iou_loss import IoULoss, bbox_overlaps
from .multibin_loss import MultiBinLoss
from .onebin_loss import OneBinLoss
from .simple_uncertain_l1Loss import SimpleUncertainL1Loss
from .dist_loss import DistLoss
from .uncertain_l1Loss import UncertainL1Loss
__all__ = ['IoULoss', 'bbox_overlaps']
