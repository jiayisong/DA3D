# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from typing import Dict, List, Optional, Sequence, Union, Tuple

import numpy as np
import torch
from numbers import Number
import torch.nn.functional as F
from mmdet.models import BatchSyncRandomResize
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine import MessageHub, is_list_of
from mmengine.structures import BaseDataElement
from mmengine.model import stack_batch
from torch import Tensor, nn
from mmdet3d.utils import OptConfigType
from mmyolo.registry import MODELS
from mmcv.ops import Voxelization
from mmdet3d.models.data_preprocessors.utils import multiview_img_stack_batch

CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str,
                 None]


@MODELS.register_module()
class RTMDet3DDataPreprocessor(DetDataPreprocessor):
    """Points / Image pre-processor for point clouds / vision-only / multi-
    modality 3D detection tasks.

    It provides the data pre-processing as follows

    - Collate and move image and point cloud data to the target device.

    - 1) For image data:
    - Pad images in inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``.
    - Stack images in inputs to batch_imgs.
    - Convert images in inputs from bgr to rgb if the shape of input is
      (3, H, W).
    - Normalize images in inputs with defined std and mean.
    - Do batch augmentations during training.

    - 2) For point cloud data:
    - If no voxelization, directly return list of point cloud data.
    - If voxelization is applied, voxelize point cloud according to
      ``voxel_type`` and obtain ``voxels``.

    Args:
        voxel (bool): Whether to apply voxelization to point cloud.
            Defaults to False.
        voxel_type (str): Voxelization type. Two voxelization types are
            provided: 'hard' and 'dynamic', respectively for hard
            voxelization and dynamic voxelization. Defaults to 'hard'.
        voxel_layer (dict or :obj:`ConfigDict`, optional): Voxelization layer
            config. Defaults to None.
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
        bgr_to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): Whether to convert image from RGB to BGR.
            Defaults to False.
        boxtype2tensor (bool): Whether to keep the ``BaseBoxes`` type of
            bboxes data or not. Defaults to True.
        batch_augments (List[dict], optional): Batch-level augmentations.
            Defaults to None.
    """

    def __init__(self,
                 voxel: bool = False,
                 voxel_type: str = 'hard',
                 voxel_layer: OptConfigType = None,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 batch_augments: Optional[List[dict]] = None) -> None:
        super(RTMDet3DDataPreprocessor, self).__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            pad_mask=pad_mask,
            mask_pad_value=mask_pad_value,
            pad_seg=pad_seg,
            seg_pad_value=seg_pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            batch_augments=batch_augments)
        self.voxel = voxel
        self.voxel_type = voxel_type
        if voxel:
            self.voxel_layer = Voxelization(**voxel_layer)

    def forward(self,
                data: Union[dict, List[dict]],
                training: bool = False) -> Union[dict, List[dict]]:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict or List[dict]): Data from dataloader.
                The dict contains the whole batch data, when it is
                a list[dict], the list indicate test time augmentation.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict or List[dict]: Data in the same format as the model input.
        """
        if isinstance(data, list):
            num_augs = len(data)
            aug_batch_data = []
            for aug_id in range(num_augs):
                single_aug_batch_data = self.simple_process(
                    data[aug_id], training)
                aug_batch_data.append(single_aug_batch_data)
            return aug_batch_data

        else:
            return self.simple_process(data, training)

    def simple_process(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.voxel:
                voxel_dict = self.voxelize(inputs['points'])
                batch_inputs['voxels'] = voxel_dict

        if 'imgs' in inputs:
            imgs = inputs['imgs']
            if self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def preprocess_img(self, _batch_img: torch.Tensor) -> torch.Tensor:
        # channel transform
        if self._channel_conversion:
            _batch_img = _batch_img[[2, 1, 0], ...]
        # Convert to float after channel conversion to ensure
        # efficiency
        _batch_img = _batch_img.float()
        # Normalization.
        if self._enable_normalize:
            if self.mean.shape[0] == 3:
                assert _batch_img.dim() == 3 and _batch_img.shape[0] == 3, (
                    'If the mean has 3 values, the input tensor '
                    'should in shape of (3, H, W), but got the '
                    f'tensor with shape {_batch_img.shape}')
            _batch_img = (_batch_img - self.mean) / self.std
        return _batch_img

    def collate_data(self, data: dict) -> dict:
        """Copying data to the target device and Performs normalization,
        padding and bgr2rgb conversion and stack based on
        ``BaseDataPreprocessor``.

        Collates the data sampled from dataloader into a list of dict and
        list of labels, and then copies tensor to the target device.

        Args:
            data (dict): Data sampled from dataloader.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore

        if 'img' in data['inputs']:
            _batch_imgs = data['inputs']['img']
            # Process data with `pseudo_collate`.
            if is_list_of(_batch_imgs, torch.Tensor):
                batch_imgs = []
                img_dim = _batch_imgs[0].dim()
                for _batch_img in _batch_imgs:
                    if img_dim == 3:  # standard img
                        _batch_img = self.preprocess_img(_batch_img)
                    elif img_dim == 4:
                        _batch_img = [
                            self.preprocess_img(_img) for _img in _batch_img
                        ]

                        _batch_img = torch.stack(_batch_img, dim=0)

                    batch_imgs.append(_batch_img)

                # Pad and stack Tensor.
                if img_dim == 3:
                    batch_imgs = stack_batch(batch_imgs, self.pad_size_divisor,
                                             self.pad_value)
                elif img_dim == 4:
                    batch_imgs = multiview_img_stack_batch(
                        batch_imgs, self.pad_size_divisor, self.pad_value)

            # Process data with `default_collate`.
            elif isinstance(_batch_imgs, torch.Tensor):
                assert _batch_imgs.dim() == 4, (
                    'The input of `ImgDataPreprocessor` should be a NCHW '
                    'tensor or a list of tensor, but got a tensor with '
                    f'shape: {_batch_imgs.shape}')
                if self._channel_conversion:
                    _batch_imgs[:, [0, 1, 2], ...] = _batch_imgs[:, [2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_imgs = _batch_imgs.float()
                if self._enable_normalize:
                    _batch_imgs = (_batch_imgs - self.mean) / self.std
                h, w = _batch_imgs.shape[2:]
                target_h = math.ceil(
                    h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(
                    w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                batch_imgs = F.pad(_batch_imgs, (0, pad_w, 0, pad_h),
                                   'constant', self.pad_value)
            else:
                raise TypeError(
                    'Output of `cast_data` should be a list of dict '
                    'or a tuple with inputs and data_samples, but got'
                    f'{type(data)}: {data}')

            data['inputs']['imgs'] = batch_imgs

        data.setdefault('data_samples', None)

        return data

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        # rewrite `_get_pad_shape` for obtaining image inputs.
        _batch_inputs = data['inputs']['img']
        # Process data with `pseudo_collate`.
        if is_list_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                if ori_input.dim() == 4:
                    # mean multiview input, select one of the
                    # image to calculate the pad shape
                    ori_input = ori_input[0]
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[3] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a list of dict '
                            'or a tuple with inputs and data_samples, but got '
                            f'{type(data)}: {data}')
        return batch_pad_shape

    @torch.no_grad()
    def voxelize(self, points: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """

        voxel_dict = dict()

        if self.voxel_type == 'hard':
            voxels, coors, num_points, voxel_centers = [], [], [], []
            for i, res in enumerate(points):
                res_voxels, res_coors, res_num_points = self.voxel_layer(res)
                res_voxel_centers = (
                                            res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                    self.voxel_layer.voxel_size) + res_voxels.new_tensor(
                    self.voxel_layer.point_cloud_range[0:3])
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
                voxel_centers.append(res_voxel_centers)

            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
            num_points = torch.cat(num_points, dim=0)
            voxel_centers = torch.cat(voxel_centers, dim=0)

            voxel_dict['num_points'] = num_points
            voxel_dict['voxel_centers'] = voxel_centers
        elif self.voxel_type == 'dynamic':
            coors = []
            # dynamic voxelization only provide a coors mapping
            for i, res in enumerate(points):
                res_coors = self.voxel_layer(res)
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                coors.append(res_coors)
            voxels = torch.cat(points, dim=0)
            coors = torch.cat(coors, dim=0)
        else:
            raise ValueError(f'Invalid voxelization type {self.voxel_type}')

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict


@MODELS.register_module()
class YOLOXBatchSyncRandomResize(BatchSyncRandomResize):
    """YOLOX batch random resize.

    Args:
        random_size_range (tuple): The multi-scale random range during
            multi-scale training.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
    """

    def forward(self, inputs: Tensor, data_samples: dict) -> Tensor and dict:
        """resize a batch of images and bboxes to shape ``self._input_size``"""
        h, w = inputs.shape[-2:]
        inputs = inputs.float()
        assert isinstance(data_samples, dict)

        if self._input_size is None:
            self._input_size = (h, w)
        scale_y = self._input_size[0] / h
        scale_x = self._input_size[1] / w
        if scale_x != 1 or scale_y != 1:
            inputs = F.interpolate(
                inputs,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)

            data_samples['bboxes_labels'][:, 2::2] *= scale_x
            data_samples['bboxes_labels'][:, 3::2] *= scale_y

        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            self._input_size = self._get_random_size(
                aspect_ratio=float(w / h), device=inputs.device)

        return inputs, data_samples


@MODELS.register_module()
class YOLOv5DetDataPreprocessor(DetDataPreprocessor):
    """Rewrite collate_fn to get faster training speed.

    Note: It must be used together with `mmyolo.datasets.utils.yolov5_collate`
    """

    def __init__(self, *args, non_blocking: Optional[bool] = True, **kwargs):
        super().__init__(*args, non_blocking=non_blocking, **kwargs)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``DetDataPreprocessorr``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            return super().forward(data, training)

        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        assert isinstance(data['data_samples'], dict)

        # TODO: Supports multi-scale training
        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]
        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples_output = {
            'bboxes_labels': data_samples['bboxes_labels'],
            'img_metas': img_metas
        }
        if 'masks' in data_samples:
            data_samples_output['masks'] = data_samples['masks']

        return {'inputs': inputs, 'data_samples': data_samples_output}


@MODELS.register_module()
class PPYOLOEDetDataPreprocessor(DetDataPreprocessor):
    """Image pre-processor for detection tasks.

    The main difference between PPYOLOEDetDataPreprocessor and
    DetDataPreprocessor is the normalization order. The official
    PPYOLOE resize image first, and then normalize image.
    In DetDataPreprocessor, the order is reversed.

    Note: It must be used together with
    `mmyolo.datasets.utils.yolov5_collate`
    """

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``. This class use batch_augments first, and then
        normalize the image, which is different from the `DetDataPreprocessor`
        .

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            return super().forward(data, training)

        assert isinstance(data['inputs'], list) and is_list_of(
            data['inputs'], torch.Tensor), \
            '"inputs" should be a list of Tensor, but got ' \
            f'{type(data["inputs"])}. The possible reason for this ' \
            'is that you are not using it with ' \
            '"mmyolo.datasets.utils.yolov5_collate". Please refer to ' \
            '"cconfigs/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco.py".'

        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        assert isinstance(data['data_samples'], dict)

        # Process data.
        batch_inputs = []
        for _input in inputs:
            # channel transform
            if self._channel_conversion:
                _input = _input[[2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _input = _input.float()
            batch_inputs.append(_input)

        # Batch random resize image.
        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(batch_inputs, data_samples)

        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples = {
            'bboxes_labels': data_samples['bboxes_labels'],
            'img_metas': img_metas
        }

        return {'inputs': inputs, 'data_samples': data_samples}


# TODO: No generality. Its input data format is different
#  mmdet's batch aug, and it must be compatible in the future.
@MODELS.register_module()
class PPYOLOEBatchRandomResize(BatchSyncRandomResize):
    """PPYOLOE batch random resize.

    Args:
        random_size_range (tuple): The multi-scale random range during
            multi-scale training.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
        random_interp (bool): Whether to choose interp_mode randomly.
            If set to True, the type of `interp_mode` must be list.
            If set to False, the type of `interp_mode` must be str.
            Defaults to True.
        interp_mode (Union[List, str]): The modes available for resizing
            are ('nearest', 'bilinear', 'bicubic', 'area').
        keep_ratio (bool): Whether to keep the aspect ratio when resizing
            the image. Now we only support keep_ratio=False.
            Defaults to False.
    """

    def __init__(self,
                 random_size_range: Tuple[int, int],
                 interval: int = 1,
                 size_divisor: int = 32,
                 random_interp=True,
                 interp_mode: Union[List[str], str] = [
                     'nearest', 'bilinear', 'bicubic', 'area'
                 ],
                 keep_ratio: bool = False) -> None:
        super().__init__(random_size_range, interval, size_divisor)
        self.random_interp = random_interp
        self.keep_ratio = keep_ratio
        # TODO: need to support keep_ratio==True
        assert not self.keep_ratio, 'We do not yet support keep_ratio=True'

        if self.random_interp:
            assert isinstance(interp_mode, list) and len(interp_mode) > 1, \
                'While random_interp==True, the type of `interp_mode`' \
                ' must be list and len(interp_mode) must large than 1'
            self.interp_mode_list = interp_mode
            self.interp_mode = None
        else:
            assert isinstance(interp_mode, str), \
                'While random_interp==False, the type of ' \
                '`interp_mode` must be str'
            assert interp_mode in ['nearest', 'bilinear', 'bicubic', 'area']
            self.interp_mode_list = None
            self.interp_mode = interp_mode

    def forward(self, inputs: list,
                data_samples: dict) -> Tuple[Tensor, Tensor]:
        """Resize a batch of images and bboxes to shape ``self._input_size``.

        The inputs and data_samples should be list, and
        ``PPYOLOEBatchRandomResize`` must be used with
        ``PPYOLOEDetDataPreprocessor`` and ``yolov5_collate`` with
        ``use_ms_training == True``.
        """
        assert isinstance(inputs, list), \
            'The type of inputs must be list. The possible reason for this ' \
            'is that you are not using it with `PPYOLOEDetDataPreprocessor` ' \
            'and `yolov5_collate` with use_ms_training == True.'

        bboxes_labels = data_samples['bboxes_labels']

        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            # get current input size
            self._input_size, interp_mode = self._get_random_size_and_interp()
            if self.random_interp:
                self.interp_mode = interp_mode

        # TODO: need to support type(inputs)==Tensor
        if isinstance(inputs, list):
            outputs = []
            for i in range(len(inputs)):
                _batch_input = inputs[i]
                h, w = _batch_input.shape[-2:]
                scale_y = self._input_size[0] / h
                scale_x = self._input_size[1] / w
                if scale_x != 1. or scale_y != 1.:
                    if self.interp_mode in ('nearest', 'area'):
                        align_corners = None
                    else:
                        align_corners = False
                    _batch_input = F.interpolate(
                        _batch_input.unsqueeze(0),
                        size=self._input_size,
                        mode=self.interp_mode,
                        align_corners=align_corners)

                    # rescale boxes
                    indexes = bboxes_labels[:, 0] == i
                    bboxes_labels[indexes, 2] *= scale_x
                    bboxes_labels[indexes, 3] *= scale_y
                    bboxes_labels[indexes, 4] *= scale_x
                    bboxes_labels[indexes, 5] *= scale_y

                    data_samples['bboxes_labels'] = bboxes_labels
                else:
                    _batch_input = _batch_input.unsqueeze(0)

                outputs.append(_batch_input)

            # convert to Tensor
            return torch.cat(outputs, dim=0), data_samples
        else:
            raise NotImplementedError('Not implemented yet!')

    def _get_random_size_and_interp(self) -> Tuple[int, int]:
        """Randomly generate a shape in ``_random_size_range`` and a
        interp_mode in interp_mode_list."""
        size = random.randint(*self._random_size_range)
        input_size = (self._size_divisor * size, self._size_divisor * size)

        if self.random_interp:
            interp_ind = random.randint(0, len(self.interp_mode_list) - 1)
            interp_mode = self.interp_mode_list[interp_ind]
        else:
            interp_mode = None
        return input_size, interp_mode


@MODELS.register_module()
class CatDepth(nn.Module):
    def __init__(self, img_size, cv, fv, relative_depth=False):
        super(CatDepth, self).__init__()
        theta = torch.arange(img_size[1])
        y = torch.arange(img_size[0])
        y, theta = torch.meshgrid(y, theta, indexing='ij')
        y = y.unsqueeze(0)
        self.register_buffer('y', y)
        self.relative_depth = relative_depth
        if relative_depth:
            self.fv = fv
            self.y_max = (img_size[0] - cv)
            self.y_min = - cv
        else:
            self.y_max = (img_size[0] - cv) / fv
            self.y_min = - cv / fv
        # self.fv = fv

        # self.y_range = img_size[0]
        # self.cv = cv

        # self.register_parameter('weight', torch.nn.Parameter(torch.tensor([0.1, ])))

    def forward(self, imgs, data_samples):
        with torch.no_grad():
            if 'img_metas' in data_samples:
                img_metas = data_samples['img_metas']
            else:
                img_metas = data_samples
            img2cam = torch.stack([i['img2cam'] for i in img_metas], 0)  # [b,3,3]

            fv_inv = img2cam[:, 1:2, 1:2]
            cv = -img2cam[:, 1:2, 2:3] / fv_inv
            if self.relative_depth:
                scale = 1 / (fv_inv * self.fv)
                y = (self.y - cv)
                y_range = (self.y_max - self.y_min) * scale
                y_min = self.y_min * scale
                y = (y - y_min) % y_range + y_min
                y = y / self.fv
            else:
                y = (self.y - cv) * fv_inv
                y = (y - self.y_min) % (self.y_max - self.y_min) + self.y_min
            # y1 = y
            # y = (((self.y - cv) + self.cv) % self.y_range - self.cv) * fv_inv
            # print((y-y1).abs().max())
            # y1_debug = y1.detach().cpu().numpy()
            # y2_debug = y.detach().cpu().numpy()
            d = y * 7.5
            # y = y.sign() * y.abs().clamp_min(0.02)
            # d = 1 / y
            # d = d.clamp(0, self.d_max) / self.d_max * 5
            # d = d.clamp(0, self.d_max) * self.weight
            # d_debug = d.detach().cpu().numpy()
            # print(torch.std(imgs[:,0,:,:]), torch.std(imgs[:,1,:,:]), torch.std(imgs[:,2,:,:]))
            # print(torch.std(d))
            imgs = torch.cat([imgs, d.unsqueeze(1)], dim=1)
        return imgs, data_samples



@MODELS.register_module()
class MulDepth(nn.Module):
    def __init__(self, img_size, cv, fv):
        super(MulDepth, self).__init__()
        theta = torch.arange(img_size[1])
        y = torch.arange(img_size[0])
        y, theta = torch.meshgrid(y, theta, indexing='ij')
        y = y.unsqueeze(0)
        self.register_buffer('y', y)
        self.y_max = (img_size[0] - cv) / fv
        self.y_min = - cv / fv


    def forward(self, imgs, data_samples):
        with torch.no_grad():
            if 'img_metas' in data_samples:
                img_metas = data_samples['img_metas']
            else:
                img_metas = data_samples
            img2cam = torch.stack([i['img2cam'] for i in img_metas], 0)  # [b,3,3]

            fv_inv = img2cam[:, 1:2, 1:2]
            cv = -img2cam[:, 1:2, 2:3] / fv_inv

            y = (self.y - cv) * fv_inv
            y = (y - self.y_min) % (self.y_max - self.y_min) + self.y_min
            d = y * 3 + 1
            # d_debug = d.detach().cpu().numpy()
            imgs = imgs * d.unsqueeze(1)
        return imgs, data_samples



@MODELS.register_module()
class CatDepthNus(nn.Module):
    def __init__(self, img_size):
        super(CatDepthNus, self).__init__()
        theta = torch.arange(img_size[1])
        y = torch.arange(img_size[0])
        y, theta = torch.meshgrid(y, theta, indexing='ij')
        y = y.unsqueeze(0)
        self.register_buffer('y', y)

    def forward(self, imgs, data_samples):
        with torch.no_grad():
            if 'img_metas' in data_samples:
                img_metas = data_samples['img_metas']
            else:
                img_metas = data_samples
            img2cam = torch.stack([i['img2cam'] for i in img_metas], 0)  # [b,3,3]
            y_range = torch.stack([i['catdepth'] for i in img_metas], 0)  # [b,2]
            fv_inv = img2cam[:, 1:2, 1:2]
            cv = -img2cam[:, 1:2, 2:3] / fv_inv
            y = (self.y - cv)
            y_range = y_range[:, 1].view(-1, 1, 1)
            y_min = y_range[:, 0].view(-1, 1, 1)
            y = (y - y_min) % y_range + y_min
            d = y * 0.01
            imgs = torch.cat([imgs, d.unsqueeze(1)], dim=1)
        return imgs, data_samples