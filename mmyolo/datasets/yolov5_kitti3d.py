# Copyright (c) OpenMMLab. All rights reserved.

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS

try:
    from mmdet3d.datasets import KittiDataset
    MMDET3D_AVAILABLE = True
except ImportError:
    from mmengine.dataset import BaseDataset
    KittiDataset = BaseDataset
    MMDET3D_AVAILABLE = False


@DATASETS.register_module()
class YOLOv5KittiDataset(BatchShapePolicyDataset, KittiDataset):
    """Dataset for YOLOv5 DOTA Dataset.

    We only add `BatchShapePolicy` function compared with DOTADataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """

    def __init__(self, *args, **kwargs):
        if not MMDET3D_AVAILABLE:
            raise ImportError(
                'Please run "mim install -r requirements/mmdet3d.txt" '
                'to install mmdet3d first for 3d detection.')

        super().__init__(*args, **kwargs)

