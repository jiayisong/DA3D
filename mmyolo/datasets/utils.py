# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence

import numpy as np
import torch
from mmengine.dataset import COLLATE_FUNCTIONS

from ..registry import TASK_UTILS


@COLLATE_FUNCTIONS.register_module()
def yolov5_collate(data_batch: Sequence,
                   use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    batch_imgs = []
    batch_bboxes_labels = []
    batch_masks = []
    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs = data_batch[i]['inputs']
        batch_imgs.append(inputs)

        gt_bboxes = datasamples.gt_instances.bboxes.tensor
        gt_labels = datasamples.gt_instances.labels
        if 'masks' in datasamples.gt_instances:
            masks = datasamples.gt_instances.masks.to_tensor(
                dtype=torch.bool, device=gt_bboxes.device)
            batch_masks.append(masks)
        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes),
                                  dim=1)
        batch_bboxes_labels.append(bboxes_labels)

    collated_results = {
        'data_samples': {
            'bboxes_labels': torch.cat(batch_bboxes_labels, 0)
        }
    }
    if len(batch_masks) > 0:
        collated_results['data_samples']['masks'] = torch.cat(batch_masks, 0)

    if use_ms_training:
        collated_results['inputs'] = batch_imgs
    else:
        collated_results['inputs'] = torch.stack(batch_imgs, 0)
    return collated_results


@TASK_UTILS.register_module()
class BatchShapePolicy:
    """BatchShapePolicy is only used in the testing phase, which can reduce the
    number of pad pixels during batch inference.

    Args:
       batch_size (int): Single GPU batch size during batch inference.
           Defaults to 32.
       img_size (int): Expected output image size. Defaults to 640.
       size_divisor (int): The minimum size that is divisible
           by size_divisor. Defaults to 32.
       extra_pad_ratio (float):  Extra pad ratio. Defaults to 0.5.
    """

    def __init__(self,
                 batch_size: int = 32,
                 img_size: int = 640,
                 size_divisor: int = 32,
                 extra_pad_ratio: float = 0.5):
        self.batch_size = batch_size
        self.img_size = img_size
        self.size_divisor = size_divisor
        self.extra_pad_ratio = extra_pad_ratio

    def __call__(self, data_list: List[dict]) -> List[dict]:
        image_shapes = []
        for data_info in data_list:
            image_shapes.append((data_info['width'], data_info['height']))

        image_shapes = np.array(image_shapes, dtype=np.float64)

        n = len(image_shapes)  # number of images
        batch_index = np.floor(np.arange(n) / self.batch_size).astype(
            np.int64)  # batch index
        number_of_batches = batch_index[-1] + 1  # number of batches

        aspect_ratio = image_shapes[:, 1] / image_shapes[:, 0]  # aspect ratio
        irect = aspect_ratio.argsort()

        data_list = [data_list[i] for i in irect]

        aspect_ratio = aspect_ratio[irect]
        # Set training image shapes
        shapes = [[1, 1]] * number_of_batches
        for i in range(number_of_batches):
            aspect_ratio_index = aspect_ratio[batch_index == i]
            min_index, max_index = aspect_ratio_index.min(
            ), aspect_ratio_index.max()
            if max_index < 1:
                shapes[i] = [max_index, 1]
            elif min_index > 1:
                shapes[i] = [1, 1 / min_index]

        batch_shapes = np.ceil(
            np.array(shapes) * self.img_size / self.size_divisor +
            self.extra_pad_ratio).astype(np.int64) * self.size_divisor

        for i, data_info in enumerate(data_list):
            data_info['batch_shape'] = batch_shapes[batch_index[i]]

        return data_list


@COLLATE_FUNCTIONS.register_module()
def RTMDet3D_collate(data_batch: Sequence,
                     use_ms_training: bool = False, train=True) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    batch_imgs = []
    metainfo = []
    if train:
        batch_bboxes = []
        batch_bboxes_3d = []
        batch_labels = []
        batch_target_3d = []
        batch_masks = []
        max_gt_bbox_len = max([i['data_samples'].gt_instances_3d.labels_3d.shape[0] for i in data_batch])
        max_gt_bbox_len = max(max_gt_bbox_len, 1)
        pad_bbox_flag = data_batch[0]['inputs']['img'].new_zeros([len(data_batch), max_gt_bbox_len, 1],
                                                                 dtype=torch.bool)
        for i in range(len(data_batch)):
            datasamples = data_batch[i]['data_samples']
            inputs = data_batch[i]['inputs']['img']
            batch_imgs.append(inputs)
            gt_bboxes = datasamples.gt_instances.bboxes
            # gt_labels = datasamples.gt_instances.bboxes_labels
            gt_bboxes_3d = datasamples.gt_instances_3d.bboxes_3d
            gt_labels_3d = datasamples.gt_instances_3d.labels_3d
            gt_target_3d = datasamples.gt_instances_3d.target_3d
            pad_bbox_flag[i, :gt_bboxes.shape[0], :] = True
            if 'masks' in datasamples.gt_instances:
                masks = datasamples.gt_instances.masks.to_tensor(
                    dtype=torch.bool, device=gt_bboxes.device)
                batch_masks.append(masks)
            if gt_bboxes.shape[0] < max_gt_bbox_len:
                fill_tensor = gt_bboxes.new_full([max_gt_bbox_len - gt_bboxes.shape[0], gt_bboxes.shape[1]], 0)
                gt_bboxes = torch.cat((gt_bboxes, fill_tensor), dim=0)
                fill_tensor = gt_bboxes_3d.new_full([max_gt_bbox_len - gt_bboxes_3d.shape[0], gt_bboxes_3d.shape[1]], 3)
                gt_bboxes_3d = torch.cat((gt_bboxes_3d, fill_tensor), dim=0)
                fill_tensor = gt_target_3d.new_full([max_gt_bbox_len - gt_target_3d.shape[0], gt_target_3d.shape[1]], 0)
                gt_target_3d = torch.cat((gt_target_3d, fill_tensor), dim=0)
                fill_tensor = gt_labels_3d.new_full([max_gt_bbox_len - gt_labels_3d.shape[0], ], 0)
                gt_labels_3d = torch.cat((gt_labels_3d, fill_tensor), dim=0)

            batch_bboxes.append(gt_bboxes)
            batch_bboxes_3d.append(gt_bboxes_3d)
            batch_labels.append(gt_labels_3d)
            batch_target_3d.append(gt_target_3d)
            metainfo.append(datasamples.metainfo)
        collated_results = {
            'inputs': {},
            'data_samples': {
                'bboxes': torch.stack(batch_bboxes, 0),
                'bboxes_3d': torch.stack(batch_bboxes_3d, 0),
                'labels': torch.stack(batch_labels, 0).unsqueeze(-1),
                'target_3d': torch.stack(batch_target_3d, 0),
                'pad_bbox_flag': pad_bbox_flag,
                'img_metas': metainfo,
            }
        }
        if len(batch_masks) > 0:
            collated_results['data_samples']['masks'] = torch.cat(batch_masks, 0)

        if use_ms_training:
            collated_results['inputs']['img'] = batch_imgs
        else:
            collated_results['inputs']['img'] = torch.stack(batch_imgs, 0)
        return collated_results
    else:
        for i in range(len(data_batch)):
            datasamples = data_batch[i]['data_samples']
            inputs = data_batch[i]['inputs']['img']
            batch_imgs.append(inputs)
            metainfo.append(datasamples.metainfo)
        collated_results = {
            'inputs': {
                'img': torch.stack(batch_imgs, 0)
            },
            'data_samples':  metainfo,
        }
        return collated_results
