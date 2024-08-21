# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy
from typing import List, Sequence, Tuple, Union
from mmdet3d.datasets.transforms import Pack3DDetInputs as MMDET3D_Pack3DDetInputs
from mmdet3d.datasets.transforms import RandomFlip3D as MMDET3D_RandomFlip3D
from mmdet3d.datasets.transforms import LoadImageFromFileMono3D as MMDET3D_LoadImageFromFileMono3D
from mmdet3d.datasets.transforms import LoadAnnotations3D as MMDET3D_LoadAnnotations3D
import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose
from mmcv.transforms.utils import cache_randomness
from mmdet.datasets.transforms import LoadAnnotations as MMDET_LoadAnnotations
from mmdet.datasets.transforms import Resize as MMDET_Resize
from mmdet.structures.bbox import (HorizontalBoxes, autocast_box_type,
                                   get_box_type)
from mmdet.structures.mask import PolygonMasks
from mmengine import fileio
from numpy import random

from mmyolo.registry import TRANSFORMS

# TODO: Waiting for MMCV support
TRANSFORMS.register_module(module=Compose, force=True)


@TRANSFORMS.register_module()
class YOLOv5KeepRatioResize(MMDET_Resize):
    """Resize images & bbox(if existed).

    This transform resizes the input image according to ``scale``.
    Bboxes (if existed) are then resized with the same scale factor.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img (np.uint8)
    - img_shape (tuple)
    - gt_bboxes (optional)
    - scale (float)

    Added Keys:

    - scale_factor (np.float32)

    Args:
        scale (Union[int, Tuple[int, int]]): Images scales for resizing.
    """

    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 keep_ratio: bool = True,
                 **kwargs):
        assert keep_ratio is True
        super().__init__(scale=scale, keep_ratio=True, **kwargs)

    @staticmethod
    def _get_rescale_ratio(old_size: Tuple[int, int],
                           scale: Union[float, Tuple[int]]) -> float:
        """Calculate the ratio for rescaling.

        Args:
            old_size (tuple[int]): The old size (w, h) of image.
            scale (float | tuple[int]): The scaling factor or maximum size.
                If it is a float number, then the image will be rescaled by
                this factor, else if it is a tuple of 2 integers, then
                the image will be rescaled as large as possible within
                the scale.

        Returns:
            float: The resize ratio.
        """
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
            scale_factor = scale
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(h, w),
                               max_short_edge / min(h, w))
        else:
            raise TypeError('Scale must be a number or tuple of int, '
                            f'but got {type(scale)}')

        return scale_factor

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        assert self.keep_ratio is True

        if results.get('img', None) is not None:
            image = results['img']
            original_h, original_w = image.shape[:2]
            ratio = self._get_rescale_ratio((original_h, original_w),
                                            self.scale)

            if ratio != 1:
                # resize image according to the ratio
                image = mmcv.imrescale(
                    img=image,
                    scale=ratio,
                    interpolation='area' if ratio < 1 else 'bilinear',
                    backend=self.backend)

            resized_h, resized_w = image.shape[:2]
            scale_ratio = resized_h / original_h

            scale_factor = (scale_ratio, scale_ratio)

            results['img'] = image
            results['img_shape'] = image.shape[:2]
            results['scale_factor'] = scale_factor


@TRANSFORMS.register_module()
class LetterResize(MMDET_Resize):
    """Resize and pad image while meeting stride-multiple constraints.

    Required Keys:

    - img (np.uint8)
    - batch_shape (np.int64) (optional)

    Modified Keys:

    - img (np.uint8)
    - img_shape (tuple)
    - gt_bboxes (optional)

    Added Keys:
    - pad_param (np.float32)

    Args:
        scale (Union[int, Tuple[int, int]]): Images scales for resizing.
        pad_val (dict): Padding value. Defaults to dict(img=0, seg=255).
        use_mini_pad (bool): Whether using minimum rectangle padding.
            Defaults to True
        stretch_only (bool): Whether stretch to the specified size directly.
            Defaults to False
        allow_scale_up (bool): Allow scale up when ratio > 1. Defaults to True
    """

    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 pad_val: dict = dict(img=0, mask=0, seg=255),
                 use_mini_pad: bool = False,
                 stretch_only: bool = False,
                 allow_scale_up: bool = True,
                 **kwargs):
        super().__init__(scale=scale, keep_ratio=True, **kwargs)

        self.pad_val = pad_val
        if isinstance(pad_val, (int, float)):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(
            pad_val, dict), f'pad_val must be dict, but got {type(pad_val)}'

        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        image = results.get('img', None)
        if image is None:
            return

        # Use batch_shape if a batch_shape policy is configured
        if 'batch_shape' in results:
            scale = tuple(results['batch_shape'])  # hw
        else:
            scale = self.scale[::-1]  # wh -> hw

        image_shape = image.shape[:2]  # height, width

        # Scale ratio (new / old)
        ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])

        # only scale down, do not scale up (for better test mAP)
        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)

        ratio = [ratio, ratio]  # float -> (float, float) for (height, width)

        # compute the best size of the image
        no_pad_shape = (int(round(image_shape[0] * ratio[0])),
                        int(round(image_shape[1] * ratio[1])))

        # padding height & width
        padding_h, padding_w = [
            scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]
        ]
        if self.use_mini_pad:
            # minimum rectangle padding
            padding_w, padding_h = np.mod(padding_w, 32), np.mod(padding_h, 32)

        elif self.stretch_only:
            # stretch to the specified size directly
            padding_h, padding_w = 0.0, 0.0
            no_pad_shape = (scale[0], scale[1])
            ratio = [scale[0] / image_shape[0],
                     scale[1] / image_shape[1]]  # height, width ratios

        if image_shape != no_pad_shape:
            # compare with no resize and padding size
            image = mmcv.imresize(
                image, (no_pad_shape[1], no_pad_shape[0]),
                interpolation=self.interpolation,
                backend=self.backend)

        scale_factor = (ratio[1], ratio[0])  # mmcv scale factor is (w, h)

        if 'scale_factor' in results:
            results['scale_factor_origin'] = results['scale_factor']
        results['scale_factor'] = scale_factor

        # padding
        top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(
            round(padding_w // 2 - 0.1))
        bottom_padding = padding_h - top_padding
        right_padding = padding_w - left_padding

        padding_list = [
            top_padding, bottom_padding, left_padding, right_padding
        ]
        if top_padding != 0 or bottom_padding != 0 or \
                left_padding != 0 or right_padding != 0:

            pad_val = self.pad_val.get('img', 0)
            if isinstance(pad_val, int) and image.ndim == 3:
                pad_val = tuple(pad_val for _ in range(image.shape[2]))

            image = mmcv.impad(
                img=image,
                padding=(padding_list[2], padding_list[0], padding_list[3],
                         padding_list[1]),
                pad_val=pad_val,
                padding_mode='constant')

        results['img'] = image
        results['img_shape'] = image.shape
        if 'pad_param' in results:
            results['pad_param_origin'] = results['pad_param'] * \
                                          np.repeat(ratio, 2)
        results['pad_param'] = np.array(padding_list, dtype=np.float32)

    def _resize_masks(self, results: dict):
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is None:
            return

        gt_masks = results['gt_masks']
        assert isinstance(
            gt_masks, PolygonMasks
        ), f'Only supports PolygonMasks, but got {type(gt_masks)}'

        # resize the gt_masks
        gt_mask_h = results['gt_masks'].height * results['scale_factor'][1]
        gt_mask_w = results['gt_masks'].width * results['scale_factor'][0]
        gt_masks = results['gt_masks'].resize(
            (int(round(gt_mask_h)), int(round(gt_mask_w))))

        top_padding, _, left_padding, _ = results['pad_param']
        if int(left_padding) != 0:
            gt_masks = gt_masks.translate(
                out_shape=results['img_shape'][:2],
                offset=int(left_padding),
                direction='horizontal')
        if int(top_padding) != 0:
            gt_masks = gt_masks.translate(
                out_shape=results['img_shape'][:2],
                offset=int(top_padding),
                direction='vertical')
        results['gt_masks'] = gt_masks

    def _resize_bboxes(self, results: dict):
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is None:
            return
        results['gt_bboxes'].rescale_(results['scale_factor'])

        if len(results['pad_param']) != 4:
            return
        results['gt_bboxes'].translate_(
            (results['pad_param'][2], results['pad_param'][0]))

        if self.clip_object_border:
            results['gt_bboxes'].clip_(results['img_shape'])

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if 'scale_factor_origin' in results:
            scale_factor_origin = results.pop('scale_factor_origin')
            results['scale_factor'] = (results['scale_factor'][0] *
                                       scale_factor_origin[0],
                                       results['scale_factor'][1] *
                                       scale_factor_origin[1])
        if 'pad_param_origin' in results:
            pad_param_origin = results.pop('pad_param_origin')
            results['pad_param'] += pad_param_origin
        return results


# TODO: Check if it can be merged with mmdet.YOLOXHSVRandomAug
@TRANSFORMS.register_module()
class YOLOv5HSVRandomAug(BaseTransform):
    """Apply HSV augmentation to image sequentially.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        hue_delta ([int, float]): delta of hue. Defaults to 0.015.
        saturation_delta ([int, float]): delta of saturation. Defaults to 0.7.
        value_delta ([int, float]): delta of value. Defaults to 0.4.
    """

    def __init__(self,
                 hue_delta: Union[int, float] = 0.015,
                 saturation_delta: Union[int, float] = 0.7,
                 value_delta: Union[int, float] = 0.4):
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def transform(self, results: dict) -> dict:
        """The HSV augmentation transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        hsv_gains = \
            random.uniform(-1, 1, 3) * \
            [self.hue_delta, self.saturation_delta, self.value_delta] + 1
        hue, sat, val = cv2.split(
            cv2.cvtColor(results['img'], cv2.COLOR_BGR2HSV))

        table_list = np.arange(0, 256, dtype=hsv_gains.dtype)
        lut_hue = ((table_list * hsv_gains[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(table_list * hsv_gains[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(table_list * hsv_gains[2], 0, 255).astype(np.uint8)

        im_hsv = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat,
                                            lut_sat), cv2.LUT(val, lut_val)))
        results['img'] = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotations(MMDET_LoadAnnotations):
    """Because the yolo series does not need to consider ignore bboxes for the
    time being, in order to speed up the pipeline, it can be excluded in
    advance."""

    def __init__(self,
                 mask2bbox: bool = False,
                 poly2mask: bool = False,
                 **kwargs) -> None:
        self.mask2bbox = mask2bbox
        assert not poly2mask, 'Does not support BitmapMasks considering ' \
                              'that bitmap consumes more memory.'
        super().__init__(poly2mask=poly2mask, **kwargs)
        if self.mask2bbox:
            assert self.with_mask, 'Using mask2bbox requires ' \
                                   'with_mask is True.'
        self._mask_ignore_flag = None

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """
        if self.mask2bbox:
            self._load_masks(results)
            if self.with_label:
                self._load_labels(results)
                self._update_mask_ignore_data(results)
            gt_bboxes = results['gt_masks'].get_bboxes(dst_type='hbox')
            results['gt_bboxes'] = gt_bboxes
        else:
            results = super().transform(results)
            self._update_mask_ignore_data(results)
        return results

    def _update_mask_ignore_data(self, results: dict) -> None:
        if 'gt_masks' not in results:
            return

        if 'gt_bboxes_labels' in results and len(
                results['gt_bboxes_labels']) != len(results['gt_masks']):
            assert len(results['gt_bboxes_labels']) == len(
                self._mask_ignore_flag)
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                self._mask_ignore_flag]

        if 'gt_bboxes' in results and len(results['gt_bboxes']) != len(
                results['gt_masks']):
            assert len(results['gt_bboxes']) == len(self._mask_ignore_flag)
            results['gt_bboxes'] = results['gt_bboxes'][self._mask_ignore_flag]

    def _load_bboxes(self, results: dict):
        """Private function to load bounding box annotations.
        Note: BBoxes with ignore_flag of 1 is not considered.
        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                gt_bboxes.append(instance['bbox'])
                gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)

    def _load_labels(self, results: dict):
        """Private function to load label annotations.

        Note: BBoxes with ignore_flag of 1 is not considered.
        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        gt_masks = []
        gt_ignore_flags = []
        self._mask_ignore_flag = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                if 'mask' in instance:
                    gt_mask = instance['mask']
                    if isinstance(gt_mask, list):
                        gt_mask = [
                            np.array(polygon) for polygon in gt_mask
                            if len(polygon) % 2 == 0 and len(polygon) >= 6
                        ]
                        if len(gt_mask) == 0:
                            # ignore
                            self._mask_ignore_flag.append(0)
                        else:
                            gt_masks.append(gt_mask)
                            gt_ignore_flags.append(instance['ignore_flag'])
                            self._mask_ignore_flag.append(1)
                    else:
                        raise NotImplementedError(
                            'Only supports mask annotations in polygon '
                            'format currently')
                else:
                    # TODO: Actually, gt with bbox and without mask needs
                    #  to be retained
                    self._mask_ignore_flag.append(0)
        self._mask_ignore_flag = np.array(self._mask_ignore_flag, dtype=bool)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

        h, w = results['ori_shape']
        gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results['gt_masks'] = gt_masks

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'mask2bbox={self.mask2bbox}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str


@TRANSFORMS.register_module()
class YOLOv5RandomAffine(BaseTransform):
    """Random affine transform data augmentation in YOLOv5 and YOLOv8. It is
    different from the implementation in YOLOX.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.
    If you set use_mask_refine == True, the code will use the masks
    annotation to refine the bbox.
    Our implementation is slightly different from the official. In COCO
    dataset, a gt may have multiple mask tags.  The official YOLOv5
    annotation file already combines the masks that an object has,
    but our code takes into account the fact that an object has multiple masks.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (PolygonMasks) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_masks (PolygonMasks) (optional)

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int]): Distance from width and height sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Defaults to 2.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Defaults to 0.1.
        use_mask_refine (bool): Whether to refine bbox by mask.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed. Defaults to 20.
        resample_num (int): Number of poly to resample to.
    """

    def __init__(self,
                 max_rotate_degree: float = 10.0,
                 max_translate_ratio: float = 0.1,
                 scaling_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 max_shear_degree: float = 2.0,
                 border: Tuple[int, int] = (0, 0),
                 border_val: Tuple[int, int, int] = (114, 114, 114),
                 bbox_clip_border: bool = True,
                 min_bbox_size: int = 2,
                 min_area_ratio: float = 0.1,
                 use_mask_refine: bool = False,
                 max_aspect_ratio: float = 20.,
                 resample_num: int = 1000):
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.use_mask_refine = use_mask_refine
        self.max_aspect_ratio = max_aspect_ratio
        self.resample_num = resample_num

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """The YOLOv5 random affine transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        img = results['img']
        # self.border is wh format
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        # Note: Different from YOLOX
        center_matrix = np.eye(3, dtype=np.float32)
        center_matrix[0, 2] = -img.shape[1] / 2
        center_matrix[1, 2] = -img.shape[0] / 2

        warp_matrix, scaling_ratio = self._get_random_homography_matrix(
            height, width)
        warp_matrix = warp_matrix @ center_matrix

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape
        img_h, img_w = img.shape[:2]

        bboxes = results['gt_bboxes']
        num_bboxes = len(bboxes)
        if num_bboxes:
            orig_bboxes = bboxes.clone()
            if self.use_mask_refine and 'gt_masks' in results:
                # If the dataset has annotations of mask,
                # the mask will be used to refine bbox.
                gt_masks = results['gt_masks']

                gt_masks_resample = self.resample_masks(gt_masks)
                gt_masks = self.warp_mask(gt_masks_resample, warp_matrix,
                                          img_h, img_w)

                # refine bboxes by masks
                bboxes = gt_masks.get_bboxes(dst_type='hbox')
                # filter bboxes outside image
                valid_index = self.filter_gt_bboxes(orig_bboxes,
                                                    bboxes).numpy()
                results['gt_masks'] = gt_masks[valid_index]
            else:
                bboxes.project_(warp_matrix)
                if self.bbox_clip_border:
                    bboxes.clip_([height, width])

                # filter bboxes
                orig_bboxes.rescale_([scaling_ratio, scaling_ratio])

                # Be careful: valid_index must convert to numpy,
                # otherwise it will raise out of bounds when len(valid_index)=1
                valid_index = self.filter_gt_bboxes(orig_bboxes,
                                                    bboxes).numpy()
                if 'gt_masks' in results:
                    results['gt_masks'] = PolygonMasks(
                        results['gt_masks'].masks, img_h, img_w)

            results['gt_bboxes'] = bboxes[valid_index]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                valid_index]
            results['gt_ignore_flags'] = results['gt_ignore_flags'][
                valid_index]

        return results

    @staticmethod
    def warp_poly(poly: np.ndarray, warp_matrix: np.ndarray, img_w: int,
                  img_h: int) -> np.ndarray:
        """Function to warp one mask and filter points outside image.

        Args:
            poly (np.ndarray): Segmentation annotation with shape (n, ) and
                with format (x1, y1, x2, y2, ...).
            warp_matrix (np.ndarray): Affine transformation matrix.
                Shape: (3, 3).
            img_w (int): Width of output image.
            img_h (int): Height of output image.
        """
        # TODO: Current logic may cause retained masks unusable for
        #  semantic segmentation training, which is same as official
        #  implementation.
        poly = poly.reshape((-1, 2))
        poly = np.concatenate((poly, np.ones(
            (len(poly), 1), dtype=poly.dtype)),
                              axis=-1)
        # transform poly
        poly = poly @ warp_matrix.T
        poly = poly[:, :2] / poly[:, 2:3]

        # filter point outside image
        x, y = poly.T
        valid_ind_point = (x >= 0) & (y >= 0) & (x <= img_w) & (y <= img_h)
        return poly[valid_ind_point].reshape(-1)

    def warp_mask(self, gt_masks: PolygonMasks, warp_matrix: np.ndarray,
                  img_w: int, img_h: int) -> PolygonMasks:
        """Warp masks by warp_matrix and retain masks inside image after
        warping.

        Args:
            gt_masks (PolygonMasks): Annotations of semantic segmentation.
            warp_matrix (np.ndarray): Affine transformation matrix.
                Shape: (3, 3).
            img_w (int): Width of output image.
            img_h (int): Height of output image.

        Returns:
            PolygonMasks: Masks after warping.
        """
        masks = gt_masks.masks

        new_masks = []
        for poly_per_obj in masks:
            warpped_poly_per_obj = []
            # One gt may have multiple masks.
            for poly in poly_per_obj:
                valid_poly = self.warp_poly(poly, warp_matrix, img_w, img_h)
                if len(valid_poly):
                    warpped_poly_per_obj.append(valid_poly.reshape(-1))
            # If all the masks are invalid,
            # add [0, 0, 0, 0, 0, 0,] here.
            if not warpped_poly_per_obj:
                # This will be filtered in function `filter_gt_bboxes`.
                warpped_poly_per_obj = [
                    np.zeros(6, dtype=poly_per_obj[0].dtype)
                ]
            new_masks.append(warpped_poly_per_obj)

        gt_masks = PolygonMasks(new_masks, img_h, img_w)
        return gt_masks

    def resample_masks(self, gt_masks: PolygonMasks) -> PolygonMasks:
        """Function to resample each mask annotation with shape (2 * n, ) to
        shape (resample_num * 2, ).

        Args:
            gt_masks (PolygonMasks): Annotations of semantic segmentation.
        """
        masks = gt_masks.masks
        new_masks = []
        for poly_per_obj in masks:
            resample_poly_per_obj = []
            for poly in poly_per_obj:
                poly = poly.reshape((-1, 2))  # xy
                poly = np.concatenate((poly, poly[0:1, :]), axis=0)
                x = np.linspace(0, len(poly) - 1, self.resample_num)
                xp = np.arange(len(poly))
                poly = np.concatenate([
                    np.interp(x, xp, poly[:, i]) for i in range(2)
                ]).reshape(2, -1).T.reshape(-1)
                resample_poly_per_obj.append(poly)
            new_masks.append(resample_poly_per_obj)
        return PolygonMasks(new_masks, gt_masks.height, gt_masks.width)

    def filter_gt_bboxes(self, origin_bboxes: HorizontalBoxes,
                         wrapped_bboxes: HorizontalBoxes) -> torch.Tensor:
        """Filter gt bboxes.

        Args:
            origin_bboxes (HorizontalBoxes): Origin bboxes.
            wrapped_bboxes (HorizontalBoxes): Wrapped bboxes

        Returns:
            dict: The result dict.
        """
        origin_w = origin_bboxes.widths
        origin_h = origin_bboxes.heights
        wrapped_w = wrapped_bboxes.widths
        wrapped_h = wrapped_bboxes.heights
        aspect_ratio = np.maximum(wrapped_w / (wrapped_h + 1e-16),
                                  wrapped_h / (wrapped_w + 1e-16))

        wh_valid_idx = (wrapped_w > self.min_bbox_size) & \
                       (wrapped_h > self.min_bbox_size)
        area_valid_idx = wrapped_w * wrapped_h / (origin_w * origin_h +
                                                  1e-16) > self.min_area_ratio
        aspect_ratio_valid_idx = aspect_ratio < self.max_aspect_ratio
        return wh_valid_idx & area_valid_idx & aspect_ratio_valid_idx

    @cache_randomness
    def _get_random_homography_matrix(self, height: int,
                                      width: int) -> Tuple[np.ndarray, float]:
        """Get random homography matrix.

        Args:
            height (int): Image height.
            width (int): Image width.

        Returns:
            Tuple[np.ndarray, float]: The result of warp_matrix and
            scaling_ratio.
        """
        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * width
        trans_y = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)
        warp_matrix = (
                translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)
        return warp_matrix, scaling_ratio

    @staticmethod
    def _get_rotation_matrix(rotate_degrees: float) -> np.ndarray:
        """Get rotation matrix.

        Args:
            rotate_degrees (float): Rotate degrees.

        Returns:
            np.ndarray: The rotation matrix.
        """
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        """Get scaling matrix.

        Args:
            scale_ratio (float): Scale ratio.

        Returns:
            np.ndarray: The scaling matrix.
        """
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees: float,
                          y_shear_degrees: float) -> np.ndarray:
        """Get shear matrix.

        Args:
            x_shear_degrees (float): X shear degrees.
            y_shear_degrees (float): Y shear degrees.

        Returns:
            np.ndarray: The shear matrix.
        """
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                                dtype=np.float32)
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x: float, y: float) -> np.ndarray:
        """Get translation matrix.

        Args:
            x (float): X translation.
            y (float): Y translation.

        Returns:
            np.ndarray: The translation matrix.
        """
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
        return translation_matrix

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'max_translate_ratio={self.max_translate_ratio}, '
        repr_str += f'scaling_ratio_range={self.scaling_ratio_range}, '
        repr_str += f'max_shear_degree={self.max_shear_degree}, '
        repr_str += f'border={self.border}, '
        repr_str += f'border_val={self.border_val}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@TRANSFORMS.register_module()
class PPYOLOERandomDistort(BaseTransform):
    """Random hue, saturation, contrast and brightness distortion.

    Required Keys:

    - img

    Modified Keys:

    - img (np.float32)

    Args:
        hue_cfg (dict): Hue settings. Defaults to dict(min=-18,
            max=18, prob=0.5).
        saturation_cfg (dict): Saturation settings. Defaults to dict(
            min=0.5, max=1.5, prob=0.5).
        contrast_cfg (dict): Contrast settings. Defaults to dict(
            min=0.5, max=1.5, prob=0.5).
        brightness_cfg (dict): Brightness settings. Defaults to dict(
            min=0.5, max=1.5, prob=0.5).
        num_distort_func (int): The number of distort function. Defaults
            to 4.
    """

    def __init__(self,
                 hue_cfg: dict = dict(min=-18, max=18, prob=0.5),
                 saturation_cfg: dict = dict(min=0.5, max=1.5, prob=0.5),
                 contrast_cfg: dict = dict(min=0.5, max=1.5, prob=0.5),
                 brightness_cfg: dict = dict(min=0.5, max=1.5, prob=0.5),
                 num_distort_func: int = 4):
        self.hue_cfg = hue_cfg
        self.saturation_cfg = saturation_cfg
        self.contrast_cfg = contrast_cfg
        self.brightness_cfg = brightness_cfg
        self.num_distort_func = num_distort_func
        assert 0 < self.num_distort_func <= 4, \
            'num_distort_func must > 0 and <= 4'
        for cfg in [
            self.hue_cfg, self.saturation_cfg, self.contrast_cfg,
            self.brightness_cfg
        ]:
            assert 0. <= cfg['prob'] <= 1., 'prob must >=0 and <=1'

    def transform_hue(self, results):
        """Transform hue randomly."""
        if random.uniform(0., 1.) >= self.hue_cfg['prob']:
            return results
        img = results['img']
        delta = random.uniform(self.hue_cfg['min'], self.hue_cfg['max'])
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        delta_iq = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        rgb2yiq_matrix = np.array([[0.114, 0.587, 0.299],
                                   [-0.321, -0.274, 0.596],
                                   [0.311, -0.523, 0.211]])
        yiq2rgb_matric = np.array([[1.0, -1.107, 1.705], [1.0, -0.272, -0.647],
                                   [1.0, 0.956, 0.621]])
        t = np.dot(np.dot(yiq2rgb_matric, delta_iq), rgb2yiq_matrix).T
        img = np.dot(img, t)
        results['img'] = img
        return results

    def transform_saturation(self, results):
        """Transform saturation randomly."""
        if random.uniform(0., 1.) >= self.saturation_cfg['prob']:
            return results
        img = results['img']
        delta = random.uniform(self.saturation_cfg['min'],
                               self.saturation_cfg['max'])

        # convert bgr img to gray img
        gray = img * np.array([[[0.114, 0.587, 0.299]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        results['img'] = img
        return results

    def transform_contrast(self, results):
        """Transform contrast randomly."""
        if random.uniform(0., 1.) >= self.contrast_cfg['prob']:
            return results
        img = results['img']
        delta = random.uniform(self.contrast_cfg['min'],
                               self.contrast_cfg['max'])
        img *= delta
        results['img'] = img
        return results

    def transform_brightness(self, results):
        """Transform brightness randomly."""
        if random.uniform(0., 1.) >= self.brightness_cfg['prob']:
            return results
        img = results['img']
        delta = random.uniform(self.brightness_cfg['min'],
                               self.brightness_cfg['max'])
        img += delta
        results['img'] = img
        return results

    def transform(self, results: dict) -> dict:
        """The hue, saturation, contrast and brightness distortion function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        results['img'] = results['img'].astype(np.float32)

        functions = [
            self.transform_brightness, self.transform_contrast,
            self.transform_saturation, self.transform_hue
        ]
        distortions = random.permutation(functions)[:self.num_distort_func]
        for func in distortions:
            results = func(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(hue_cfg={self.hue_cfg}, '
        repr_str += f'saturation_cfg={self.saturation_cfg}, '
        repr_str += f'contrast_cfg={self.contrast_cfg}, '
        repr_str += f'brightness_cfg={self.brightness_cfg}, '
        repr_str += f'num_distort_func={self.num_distort_func})'
        return repr_str


@TRANSFORMS.register_module()
class PPYOLOERandomCrop(BaseTransform):
    """Random crop the img and bboxes. Different thresholds are used in PPYOLOE
    to judge whether the clipped image meets the requirements. This
    implementation is different from the implementation of RandomCrop in mmdet.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Added Keys:
    - pad_param (np.float32)

    Args:
        aspect_ratio (List[float]): Aspect ratio of cropped region. Default to
             [.5, 2].
        thresholds (List[float]): Iou thresholds for deciding a valid bbox crop
            in [min, max] format. Defaults to [.0, .1, .3, .5, .7, .9].
        scaling (List[float]): Ratio between a cropped region and the original
            image in [min, max] format. Default to [.3, 1.].
        num_attempts (int): Number of tries for each threshold before
            giving up. Default to 50.
        allow_no_crop (bool): Allow return without actually cropping them.
            Default to True.
        cover_all_box (bool): Ensure all bboxes are covered in the final crop.
            Default to False.
    """

    def __init__(self,
                 aspect_ratio: List[float] = [.5, 2.],
                 thresholds: List[float] = [.0, .1, .3, .5, .7, .9],
                 scaling: List[float] = [.3, 1.],
                 num_attempts: int = 50,
                 allow_no_crop: bool = True,
                 cover_all_box: bool = False):
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box

    def _crop_data(self, results: dict, crop_box: Tuple[int, int, int, int],
                   valid_inds: np.ndarray) -> Union[dict, None]:
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_box (Tuple[int, int, int, int]): Expected absolute coordinates
                for cropping, (x1, y1, x2, y2).
            valid_inds (np.ndarray): The indexes of gt that needs to be
                retained.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        # crop the image
        img = results['img']
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        results['img'] = img
        img_shape = img.shape
        results['img_shape'] = img.shape

        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-crop_x1, -crop_y1])
            bboxes.clip_(img_shape[:2])

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_inds.nonzero()[0]].crop(
                    np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                    crop_x1:crop_x2]

        return results

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """The random crop transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if results.get('gt_bboxes', None) is None or len(
                results['gt_bboxes']) == 0:
            return results

        orig_img_h, orig_img_w = results['img'].shape[:2]
        gt_bboxes = results['gt_bboxes']

        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        random.shuffle(thresholds)

        for thresh in thresholds:
            # Determine the coordinates for cropping
            if thresh == 'no_crop':
                return results

            found = False
            for i in range(self.num_attempts):
                crop_h, crop_w = self._get_crop_size((orig_img_h, orig_img_w))
                if self.aspect_ratio is None:
                    if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                        continue

                # get image crop_box
                margin_h = max(orig_img_h - crop_h, 0)
                margin_w = max(orig_img_w - crop_w, 0)
                offset_h, offset_w = self._rand_offset((margin_h, margin_w))
                crop_y1, crop_y2 = offset_h, offset_h + crop_h
                crop_x1, crop_x2 = offset_w, offset_w + crop_w

                crop_box = [crop_x1, crop_y1, crop_x2, crop_y2]
                # Calculate the iou between gt_bboxes and crop_boxes
                iou = self._iou_matrix(gt_bboxes,
                                       np.array([crop_box], dtype=np.float32))
                # If the maximum value of the iou is less than thresh,
                # the current crop_box is considered invalid.
                if iou.max() < thresh:
                    continue

                # If cover_all_box == True and the minimum value of
                # the iou is less than thresh, the current crop_box
                # is considered invalid.
                if self.cover_all_box and iou.min() < thresh:
                    continue

                # Get which gt_bboxes to keep after cropping.
                valid_inds = self._get_valid_inds(
                    gt_bboxes, np.array(crop_box, dtype=np.float32))
                if valid_inds.size > 0:
                    found = True
                    break

            if found:
                results = self._crop_data(results, crop_box, valid_inds)
                return results
        return results

    @cache_randomness
    def _rand_offset(self, margin: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generate crop offset.

        Args:
            margin (Tuple[int, int]): The upper bound for the offset generated
                randomly.

        Returns:
            Tuple[int, int]: The random offset for the crop.
        """
        margin_h, margin_w = margin
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        return (offset_h, offset_w)

    @cache_randomness
    def _get_crop_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generates the crop size based on `image_size`.

        Args:
            image_size (Tuple[int, int]): (h, w).

        Returns:
            crop_size (Tuple[int, int]): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        scale = random.uniform(*self.scaling)
        if self.aspect_ratio is not None:
            min_ar, max_ar = self.aspect_ratio
            aspect_ratio = random.uniform(
                max(min_ar, scale ** 2), min(max_ar, scale ** -2))
            h_scale = scale / np.sqrt(aspect_ratio)
            w_scale = scale * np.sqrt(aspect_ratio)
        else:
            h_scale = random.uniform(*self.scaling)
            w_scale = random.uniform(*self.scaling)
        crop_h = h * h_scale
        crop_w = w * w_scale
        return int(crop_h), int(crop_w)

    def _iou_matrix(self,
                    gt_bbox: HorizontalBoxes,
                    crop_bbox: np.ndarray,
                    eps: float = 1e-10) -> np.ndarray:
        """Calculate iou between gt and image crop box.

        Args:
            gt_bbox (HorizontalBoxes): Ground truth bounding boxes.
            crop_bbox (np.ndarray): Image crop coordinates in
                [x1, y1, x2, y2] format.
            eps (float): Default to 1e-10.
        Return:
            (np.ndarray): IoU.
        """
        gt_bbox = gt_bbox.tensor.numpy()
        lefttop = np.maximum(gt_bbox[:, np.newaxis, :2], crop_bbox[:, :2])
        rightbottom = np.minimum(gt_bbox[:, np.newaxis, 2:], crop_bbox[:, 2:])

        overlap = np.prod(
            rightbottom - lefttop,
            axis=2) * (lefttop < rightbottom).all(axis=2)
        area_gt_bbox = np.prod(gt_bbox[:, 2:] - crop_bbox[:, :2], axis=1)
        area_crop_bbox = np.prod(gt_bbox[:, 2:] - crop_bbox[:, :2], axis=1)
        area_o = (area_gt_bbox[:, np.newaxis] + area_crop_bbox - overlap)
        return overlap / (area_o + eps)

    def _get_valid_inds(self, gt_bbox: HorizontalBoxes,
                        img_crop_bbox: np.ndarray) -> np.ndarray:
        """Get which Bboxes to keep at the current cropping coordinates.

        Args:
            gt_bbox (HorizontalBoxes): Ground truth bounding boxes.
            img_crop_bbox (np.ndarray): Image crop coordinates in
                [x1, y1, x2, y2] format.

        Returns:
            (np.ndarray): Valid indexes.
        """
        cropped_box = gt_bbox.tensor.numpy().copy()
        gt_bbox = gt_bbox.tensor.numpy().copy()

        cropped_box[:, :2] = np.maximum(gt_bbox[:, :2], img_crop_bbox[:2])
        cropped_box[:, 2:] = np.minimum(gt_bbox[:, 2:], img_crop_bbox[2:])
        cropped_box[:, :2] -= img_crop_bbox[:2]
        cropped_box[:, 2:] -= img_crop_bbox[:2]

        centers = (gt_bbox[:, :2] + gt_bbox[:, 2:]) / 2
        valid = np.logical_and(img_crop_bbox[:2] <= centers,
                               centers < img_crop_bbox[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return np.where(valid)[0]

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(aspect_ratio={self.aspect_ratio}, '
        repr_str += f'thresholds={self.thresholds}, '
        repr_str += f'scaling={self.scaling}, '
        repr_str += f'num_attempts={self.num_attempts}, '
        repr_str += f'allow_no_crop={self.allow_no_crop}, '
        repr_str += f'cover_all_box={self.cover_all_box})'
        return repr_str


@TRANSFORMS.register_module()
class YOLOv5CopyPaste(BaseTransform):
    """Copy-Paste used in YOLOv5 and YOLOv8.

    This transform randomly copy some objects in the image to the mirror
    position of the image.It is different from the `CopyPaste` in mmdet.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (PolygonMasks) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (optional)
    - gt_masks (optional)

    Args:
        ioa_thresh (float): Ioa thresholds for deciding valid bbox.
        prob (float): Probability of choosing objects.
            Defaults to 0.5.
    """

    def __init__(self, ioa_thresh: float = 0.3, prob: float = 0.5):
        self.ioa_thresh = ioa_thresh
        self.prob = prob

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """The YOLOv5 and YOLOv8 Copy-Paste transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if len(results.get('gt_masks', [])) == 0:
            return results
        gt_masks = results['gt_masks']
        assert isinstance(gt_masks, PolygonMasks), \
            'only support type of PolygonMasks,' \
            ' but get type: %s' % type(gt_masks)
        gt_bboxes = results['gt_bboxes']
        gt_bboxes_labels = results.get('gt_bboxes_labels', None)
        img = results['img']
        img_h, img_w = img.shape[:2]

        # calculate ioa
        gt_bboxes_flip = deepcopy(gt_bboxes)
        gt_bboxes_flip.flip_(img.shape)

        ioa = self.bbox_ioa(gt_bboxes_flip, gt_bboxes)
        indexes = torch.nonzero((ioa < self.ioa_thresh).all(1))[:, 0]
        n = len(indexes)
        valid_inds = random.choice(
            indexes, size=round(self.prob * n), replace=False)
        if len(valid_inds) == 0:
            return results

        if gt_bboxes_labels is not None:
            # prepare labels
            gt_bboxes_labels = np.concatenate(
                (gt_bboxes_labels, gt_bboxes_labels[valid_inds]), axis=0)

        # prepare bboxes
        copypaste_bboxes = gt_bboxes_flip[valid_inds]
        gt_bboxes = gt_bboxes.cat([gt_bboxes, copypaste_bboxes])

        # prepare images
        copypaste_gt_masks = gt_masks[valid_inds]
        copypaste_gt_masks_flip = copypaste_gt_masks.flip()
        # convert poly format to bitmap format
        # example: poly: [[array(0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]]
        #  -> bitmap: a mask with shape equal to (1, img_h, img_w)
        # # type1 low speed
        # copypaste_gt_masks_bitmap = copypaste_gt_masks.to_ndarray()
        # copypaste_mask = np.sum(copypaste_gt_masks_bitmap, axis=0) > 0

        # type2
        copypaste_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for poly in copypaste_gt_masks.masks:
            poly = [i.reshape((-1, 1, 2)).astype(np.int32) for i in poly]
            cv2.drawContours(copypaste_mask, poly, -1, (1,), cv2.FILLED)

        copypaste_mask = copypaste_mask.astype(bool)

        # copy objects, and paste to the mirror position of the image
        copypaste_mask_flip = mmcv.imflip(
            copypaste_mask, direction='horizontal')
        copypaste_img = mmcv.imflip(img, direction='horizontal')
        img[copypaste_mask_flip] = copypaste_img[copypaste_mask_flip]

        # prepare masks
        gt_masks = copypaste_gt_masks.cat([gt_masks, copypaste_gt_masks_flip])

        if 'gt_ignore_flags' in results:
            # prepare gt_ignore_flags
            gt_ignore_flags = results['gt_ignore_flags']
            gt_ignore_flags = np.concatenate(
                [gt_ignore_flags, gt_ignore_flags[valid_inds]], axis=0)
            results['gt_ignore_flags'] = gt_ignore_flags

        results['img'] = img
        results['gt_bboxes'] = gt_bboxes
        if gt_bboxes_labels is not None:
            results['gt_bboxes_labels'] = gt_bboxes_labels
        results['gt_masks'] = gt_masks

        return results

    @staticmethod
    def bbox_ioa(gt_bboxes_flip: HorizontalBoxes,
                 gt_bboxes: HorizontalBoxes,
                 eps: float = 1e-7) -> np.ndarray:
        """Calculate ioa between gt_bboxes_flip and gt_bboxes.

        Args:
            gt_bboxes_flip (HorizontalBoxes): Flipped ground truth
                bounding boxes.
            gt_bboxes (HorizontalBoxes): Ground truth bounding boxes.
            eps (float): Default to 1e-10.
        Return:
            (Tensor): Ioa.
        """
        gt_bboxes_flip = gt_bboxes_flip.tensor
        gt_bboxes = gt_bboxes.tensor

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = gt_bboxes_flip.T
        b2_x1, b2_y1, b2_x2, b2_y2 = gt_bboxes.T

        # Intersection area
        inter_area = (torch.minimum(b1_x2[:, None],
                                    b2_x2) - torch.maximum(b1_x1[:, None],
                                                           b2_x1)).clip(0) * \
                     (torch.minimum(b1_y2[:, None],
                                    b2_y2) - torch.maximum(b1_y1[:, None],
                                                           b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

        # Intersection over box2 area
        return inter_area / box2_area

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(ioa_thresh={self.ioa_thresh},'
        repr_str += f'prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class RemoveDataElement(BaseTransform):
    """Remove unnecessary data element in results.

    Args:
        keys (Union[str, Sequence[str]]): Keys need to be removed.
    """

    def __init__(self, keys: Union[str, Sequence[str]]):
        self.keys = [keys] if isinstance(keys, str) else keys

    def transform(self, results: dict) -> dict:
        for key in self.keys:
            results.pop(key, None)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        return repr_str


@TRANSFORMS.register_module()
class RegularizeRotatedBox(BaseTransform):
    """Regularize rotated boxes.

    Due to the angle periodicity, one rotated box can be represented in
    many different (x, y, w, h, t). To make each rotated box unique,
    ``regularize_boxes`` will take the remainder of the angle divided by
    180 degrees.

    For convenience, three angle_version can be used here:

    - 'oc': OpenCV Definition. Has the same box representation as
        ``cv2.minAreaRect`` the angle ranges in [-90, 0).
    - 'le90': Long Edge Definition (90). the angle ranges in [-90, 90).
        The width is always longer than the height.
    - 'le135': Long Edge Definition (135). the angle ranges in [-45, 135).
        The width is always longer than the height.

    Required Keys:

    - gt_bboxes (RotatedBoxes[torch.float32])

    Modified Keys:

    - gt_bboxes

    Args:
        angle_version (str): Angle version. Can only be 'oc',
            'le90', or 'le135'. Defaults to 'le90.
    """

    def __init__(self, angle_version='le90') -> None:
        self.angle_version = angle_version
        try:
            from mmrotate.structures.bbox import RotatedBoxes
            self.box_type = RotatedBoxes
        except ImportError:
            raise ImportError(
                'Please run "mim install -r requirements/mmrotate.txt" '
                'to install mmrotate first for rotated detection.')

    def transform(self, results: dict) -> dict:
        assert isinstance(results['gt_bboxes'], self.box_type)
        results['gt_bboxes'] = self.box_type(
            results['gt_bboxes'].regularize_boxes(self.angle_version))
        return results


@TRANSFORMS.register_module()
class RandomFlip3D(MMDET3D_RandomFlip3D):

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str, optional): Flip direction.
                Default: 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if self.flip_box3d:
            if 'gt_bboxes_3d' in input_dict:
                if 'points' in input_dict:
                    input_dict['points'] = input_dict['gt_bboxes_3d'].flip(
                        direction, points=input_dict['points'])
                else:
                    # vision-only detection
                    input_dict['gt_bboxes_3d'].flip(direction)
            else:
                input_dict['points'].flip(direction)

        if 'centers_2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['img_shape'][1]
            input_dict['centers_2d'][..., 0] = w - 1 - input_dict['centers_2d'][..., 0]
        if 'cam2img' in input_dict:
            w = input_dict['img_shape'][1]
            input_dict['cam2img'][0][2] = w - 1 - input_dict['cam2img'][0][2]
            if len(input_dict['cam2img'][0]) == 4:
                input_dict['cam2img'][0][3] = (w - 1) * input_dict['cam2img'][2][3] - input_dict['cam2img'][0][3]


@TRANSFORMS.register_module()
class LoadImageFromFileMono3D(MMDET3D_LoadImageFromFileMono3D):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def transform(self, results: dict) -> dict:
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        # TODO: load different camera image from data info,
        # for kitti dataset, we load 'CAM2' image.
        # for nuscenes dataset, we load 'CAM_FRONT' image.

        if 'img_path' in results:
            filename = results['img_path']
        else:
            raise RuntimeError('no img_path')
        if 'cam2img' not in results:
            raise RuntimeError('no cam2img')

        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results


@TRANSFORMS.register_module()
class LoadAnnotations3D(MMDET3D_LoadAnnotations3D):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Required Keys:

    - ann_info (dict)

        - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes` |
          :obj:`DepthInstance3DBoxes` | :obj:`CameraInstance3DBoxes`):
          3D ground truth bboxes. Only when `with_bbox_3d` is True
        - gt_labels_3d (np.int64): Labels of ground truths.
          Only when `with_label_3d` is True.
        - gt_bboxes (np.float32): 2D ground truth bboxes.
          Only when `with_bbox` is True.
        - gt_labels (np.ndarray): Labels of ground truths.
          Only when `with_label` is True.
        - depths (np.ndarray): Only when
          `with_bbox_depth` is True.
        - centers_2d (np.ndarray): Only when
          `with_bbox_depth` is True.
        - attr_labels (np.ndarray): Attribute labels of instances.
          Only when `with_attr_label` is True.

    - pts_instance_mask_path (str): Path of instance mask file.
      Only when `with_mask_3d` is True.
    - pts_semantic_mask_path (str): Path of semantic mask file.
      Only when `with_seg_3d` is True.

    Added Keys:

    - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes` |
      :obj:`DepthInstance3DBoxes` | :obj:`CameraInstance3DBoxes`):
      3D ground truth bboxes. Only when `with_bbox_3d` is True
    - gt_labels_3d (np.int64): Labels of ground truths.
      Only when `with_label_3d` is True.
    - gt_bboxes (np.float32): 2D ground truth bboxes.
      Only when `with_bbox` is True.
    - gt_labels (np.int64): Labels of ground truths.
      Only when `with_label` is True.
    - depths (np.float32): Only when
      `with_bbox_depth` is True.
    - centers_2d (np.ndarray): Only when
      `with_bbox_depth` is True.
    - attr_labels (np.int64): Attribute labels of instances.
      Only when `with_attr_label` is True.
    - pts_instance_mask (np.int64): Instance mask of each point.
      Only when `with_mask_3d` is True.
    - pts_semantic_mask (np.int64): Semantic mask of each point.
      Only when `with_seg_3d` is True.

    Args:
        with_bbox_3d (bool): Whether to load 3D boxes. Defaults to True.
        with_label_3d (bool): Whether to load 3D labels. Defaults to True.
        with_attr_label (bool): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool): Whether to load 3D instance masks for points.
            Defaults to False.
        with_seg_3d (bool): Whether to load 3D semantic masks for points.
            Defaults to False.
        with_bbox (bool): Whether to load 2D boxes. Defaults to False.
        with_label (bool): Whether to load 2D labels. Defaults to False.
        with_mask (bool): Whether to load 2D instance masks. Defaults to False.
        with_seg (bool): Whether to load 2D semantic masks. Defaults to False.
        with_bbox_depth (bool): Whether to load 2.5D boxes. Defaults to False.
        poly2mask (bool): Whether to convert polygon annotations to bitmasks.
            Defaults to True.
        seg_3d_dtype (dtype): Dtype of 3D semantic masks. Defaults to int64.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to dict(backend='disk').
    """

    def __init__(
            self,
            with_difficulty: bool = False, with_visibility=False, with_pt_num=False,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.with_difficulty = with_difficulty
        self.with_visibility = with_visibility
        self.with_pt_num = with_pt_num

    def _load_visibility(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        # if results['ann_info']['gt_bboxes_labels'].shape[0] != results['ann_info']['gt_labels_3d'].shape[0]:
        #     print(0)
        if 'visibility_token' in results['ann_info']:
            visibility_token = results['ann_info']['visibility_token']
            visibility_token = visibility_token.astype(np.int32)
            mask = (visibility_token >= 2)
            #if 1 > 0:
            if (~mask).sum() > 0:
                for k in results['ann_info'].keys():
                    if k != 'instances':
                        results['ann_info'][k] = results['ann_info'][k][mask]
        return results

    def _load_difficulty(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        # if results['ann_info']['gt_bboxes_labels'].shape[0] != results['ann_info']['gt_labels_3d'].shape[0]:
        #     print(0)
        if 'difficulty' in results['ann_info']:
            difficulty = results['ann_info']['difficulty']
            mask = (difficulty >= 0)
            if (~mask).sum() > 0:
                for k in results['ann_info'].keys():
                    if k != 'instances':
                        results['ann_info'][k] = results['ann_info'][k][mask]
        return results

    def _load_pt_num(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        # if results['ann_info']['gt_bboxes_labels'].shape[0] != results['ann_info']['gt_labels_3d'].shape[0]:
        #     print(0)
        if 'num_pts' in results['ann_info']:
            num_pts = results['ann_info']['num_pts']
            mask = (num_pts > 0)
            #if 1 > 0:
            if (~mask).sum() > 0:
                for k in results['ann_info'].keys():
                    if k != 'instances':
                        results['ann_info'][k] = results['ann_info'][k][mask]
        return results

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
            semantic segmentation annotations.
        """
        if self.with_difficulty:
            results = self._load_difficulty(results)
        if self.with_visibility:
            results = self._load_visibility(results)
        if self.with_pt_num:
            results = self._load_pt_num(results)
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results


@TRANSFORMS.register_module()
class Update2Dattr(BaseTransform):
    def __init__(self, cylinder=False):
        self.cylinder = cylinder

    def _cylinder_bboxes(self, results):
        if 'gt_bboxes_3d' in results:
            gt_bbox_3d = results['gt_bboxes_3d']
            intrinsics = results['cam2img']
            h, w = results['img_shape']
            Y_min = (-intrinsics[1, 2]) / intrinsics[1, 1]
            Y_max = (h - 1 - intrinsics[1, 2]) / intrinsics[1, 1]
            theta_min_val = (-intrinsics[0, 2]) / intrinsics[0, 0]
            theta_max_val = (w - 1 - intrinsics[0, 2]) / intrinsics[0, 0]
            centers3d = gt_bbox_3d.gravity_center + results['K_out'][None, :]
            centers3d = centers3d.numpy()
            theta = np.arctan2(centers3d[:, 0], centers3d[:, 2])
            depths = np.linalg.norm(centers3d[:, 0::2], axis=1, keepdims=False)
            yy = centers3d[:, 1] / depths
            if 'depths' in results:
                results['depths'] = depths
            if 'centers_2d' in results:
                u = theta * intrinsics[0, 0] + intrinsics[0, 2]
                v = yy * intrinsics[1, 1] + intrinsics[1, 2]
                results['centers_2d'] = np.stack((u, v), axis=1)
            if 'gt_bboxes' in results:
                corners = gt_bbox_3d.corners.view(-1, 3) + results['K_out'][None, :]
                corners = corners.numpy()
                theta = np.arctan2(corners[:, 0], corners[:, 2])
                depths = np.linalg.norm(corners[:, 0::2], axis=1, keepdims=False)
                yy = corners[:, 1] / depths
                theta_min = np.min(theta.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[1] - 1)
                theta_max = np.max(theta.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[1] - 1)
                yy_min = np.min(yy.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[0] - 1)
                yy_max = np.max(yy.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[0] - 1)
                x_min = theta_min * intrinsics[0, 0] + intrinsics[0, 2]
                x_max = theta_max * intrinsics[0, 0] + intrinsics[0, 2]
                y_min = yy_min * intrinsics[1, 1] + intrinsics[1, 2]
                y_max = yy_max * intrinsics[1, 1] + intrinsics[1, 2]
                results['gt_bboxes_area'] = (y_max - y_min) * (x_max - x_min)
                # clip bbox2d
                theta_min = np.clip(theta_min, theta_min_val, theta_max_val)
                theta_max = np.clip(theta_max, theta_min_val, theta_max_val)
                y_max_theta = np.where(theta_max <= 0, theta_max, 0)
                y_max_theta = np.where(theta_min >= 0, theta_min, y_max_theta)
                yy_min = np.clip(yy_min, Y_min * np.cos(y_max_theta), Y_max * np.cos(y_max_theta))
                yy_max = np.clip(yy_max, Y_min * np.cos(y_max_theta), Y_max * np.cos(y_max_theta))
                x_min = theta_min * intrinsics[0, 0] + intrinsics[0, 2]
                x_max = theta_max * intrinsics[0, 0] + intrinsics[0, 2]
                y_min = yy_min * intrinsics[1, 1] + intrinsics[1, 2]
                y_max = yy_max * intrinsics[1, 1] + intrinsics[1, 2]
                results['gt_bboxes'] = np.stack((x_min, y_min, x_max, y_max), axis=1)
        return results

    def show_result(self, results, name='cutmix3d'):
        img = results['img'].copy()
        bbox = results['gt_bboxes']
        P2 = results['cam2img']
        K_out = results['K_out'].numpy()
        for i, b in enumerate(bbox):
            # img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
            label = results['gt_bboxes_3d'].tensor[i, :]
            h, w, l = float(label[4]), float(label[5]), float(label[3])
            x, y, z = float(label[0]), float(label[1]), float(label[2])
            yaw = float(label[6])  # 朝向角度
            # 计算3D框的8个顶点坐标
            corners = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                [0, 0, 0, 0, -h, -h, -h, -h],
                                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                                ])
            R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                          [0, 1, 0],
                          [-np.sin(yaw), 0, np.cos(yaw)]])
            corners = R @ corners + np.array([[x], [y], [z]])
            # 投影到图像上
            # corners_h = np.vstack((corners, np.ones((1, 8))))
            corners = corners + K_out[:, None]
            if self.cylinder:
                corners = np.stack([np.arctan2(corners[0, :], corners[2, :]),
                                    corners[1, :] / np.linalg.norm(corners[0::2, :], axis=0, keepdims=False),
                                    np.ones([8, ])], axis=0)
            corners_2d = P2 @ corners
            corners_2d /= corners_2d[2:, :]
            corners_2d = corners_2d[:2, :].transpose()
            # 绘制3D框
            imgpts = np.int32(corners_2d).reshape(-1, 2)
            cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 2)  # 底部矩形
            for i in range(4):
                cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i + 4]), (0, 255, 0), 2)  # 框体纵��线
            cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)  # 顶部矩形
            img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.imwrite(f'{results["sample_idx"]}_{name}.jpg', img)
        return 0

    def _compute_2d(self, results):
        if 'gt_bboxes_3d' in results:
            gt_bbox_3d = results['gt_bboxes_3d']
            intrinsics = results['cam2img']
            centers3d = gt_bbox_3d.gravity_center + results['K_out'][None, :]
            centers3d = centers3d.numpy()
            d = centers3d[:, 2]
            u = centers3d[:, 0] * intrinsics[0, 0] / d + intrinsics[0, 2]
            v = centers3d[:, 1] * intrinsics[1, 1] / d + intrinsics[1, 2]
            centers_2d = np.stack((u, v), axis=1)
            # print((np.abs(results['centers_2d'] - centers_2d)).max())
            # if np.abs(results['centers_2d'] - centers_2d).max() > 0.1:
            #     print(results['centers_2d'], centers_2d)
            #     print(0)
            results['centers_2d'] = centers_2d
            # print(np.abs(results['depths'] - d).max())
            # if np.abs(results['depths'] - d).max() > 0.1:
            #     print(results['depths'], d)
            #     print(0)
            results['depths'] = d
            corners = gt_bbox_3d.corners.view(-1, 3) + results['K_out'][None, :]
            corners = corners.numpy()
            d = corners[:, 2].clip(min=0.1)
            u = corners[:, 0] * intrinsics[0, 0] / d + intrinsics[0, 2]
            v = corners[:, 1] * intrinsics[1, 1] / d + intrinsics[1, 2]
            x_min = np.min(u.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[1] - 1)
            x_max = np.max(u.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[1] - 1)
            y_min = np.min(v.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[0] - 1)
            y_max = np.max(v.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[0] - 1)
            results['gt_bboxes_area'] = (y_max - y_min) * (x_max - x_min)
            x_min = np.clip(x_min, 0, results['img_shape'][1] - 1)
            x_max = np.clip(x_max, 0, results['img_shape'][1] - 1)
            y_min = np.clip(y_min, 0, results['img_shape'][0] - 1)
            y_max = np.clip(y_max, 0, results['img_shape'][0] - 1)
            gt_bboxes = np.stack((x_min, y_min, x_max, y_max), axis=1)
            # print(np.abs(results['gt_bboxes'] - gt_bboxes).max())
            # if np.abs(results['gt_bboxes'] - gt_bboxes).max() > 0.1:
            #     print(results['gt_bboxes'], gt_bboxes)
            #     print(0)
            results['gt_bboxes'] = gt_bboxes
        return results

    def transform(self, results):
        if self.cylinder:
            # self.show_result(results, 'update2dattr')
            return self._cylinder_bboxes(results)
        else:
            return self._compute_2d(results)


@TRANSFORMS.register_module()
class UnifiedIntrinsics(BaseTransform):
    def __init__(self, size=(384, 1280), intrinsics=((721.5377, 0.0, 639.5), (0.0, 721.5377, 172.854), (0.0, 0.0, 1.0)),
                 random_shift=(0, 0), cycle=False, pad_val=(114, 114, 114), cylinder=False):
        self.intrinsics_origin = np.array(intrinsics, dtype=np.float32)
        self.size = size
        self.random_shift = random_shift
        self.cycle = cycle
        self.pad_val = pad_val
        self.cylinder = cylinder

    def _cylinder_img(self, results):
        """Resize images with ``results['scale']``."""
        theta = (np.arange(self.size[1], dtype=np.float32) - self.intrinsics[0][2]) / self.intrinsics[0][0]
        y = (np.arange(self.size[0], dtype=np.float32) - self.intrinsics[1][2]) / self.intrinsics[1][1]
        theta, y = np.meshgrid(theta, y)
        cam2img = results['cam2img']
        for key in results.get('img_fields', ['img', ]):
            img = results[key]
            h, w, _ = img.shape
            self.Y_min = (-cam2img[1, 2]) / cam2img[1, 1]
            self.Y_max = (h - 1 - cam2img[1, 2]) / cam2img[1, 1]
            self.theta_min = math.atan((-cam2img[0, 2]) / cam2img[0, 0])
            self.theta_max = math.atan((w - 1 - cam2img[0, 2]) / cam2img[0, 0])
            theta_min = (-self.intrinsics_origin[0, 2]) / self.intrinsics_origin[0, 0]
            y_min = (-self.intrinsics_origin[1, 2]) / self.intrinsics_origin[1, 1]
            delta_theta = self.size[1] / self.intrinsics_origin[0, 0]
            delta_y = self.size[0] / self.intrinsics_origin[1, 1]
            if self.cycle:
                theta = (theta - theta_min) % delta_theta + theta_min
                y = (y - y_min) % delta_y + y_min
            X = np.tan(theta)
            Y = y / np.cos(theta)
            u = (cam2img[0, 0] * X + cam2img[0, 2])
            v = (cam2img[1, 1] * Y + cam2img[1, 2])
            # uv = np.stack([u, v], axis=-1)
            # print(self.theta_min, self.theta_max, self.y_min, self.y_max)
            # cv2.imwrite(f'{results["sample_idx"]}_ori.jpg', img)
            img = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderValue=self.pad_val)
            # cv2.imwrite(f'{results["sample_idx"]}.jpg', img)
            results[key] = img
            results['img_shape'] = img.shape[:2]
            # in case that there is no padding
            results['pad_shape'] = img.shape[:2]
        return results

    def _cylinder_bboxes(self, results):
        if 'gt_bboxes_3d' in results:
            gt_bbox_3d = results['gt_bboxes_3d']
            centers3d = gt_bbox_3d.gravity_center + results['K_out'][None, :]
            centers3d = centers3d.numpy()
            theta = np.arctan2(centers3d[:, 0], centers3d[:, 2])
            depths = np.linalg.norm(centers3d[:, 0::2], axis=1, keepdims=False)
            yy = centers3d[:, 1] / depths
            if 'depths' in results:
                results['depths'] = depths
            if 'centers_2d' in results:
                u = theta * self.intrinsics[0, 0] + self.intrinsics[0, 2]
                v = yy * self.intrinsics[1, 1] + self.intrinsics[1, 2]
                results['centers_2d'] = np.stack((u, v), axis=1)
            if 'gt_bboxes' in results:
                corners = gt_bbox_3d.corners.view(-1, 3) + results['K_out'][None, :]
                corners = corners.numpy()
                theta = np.arctan2(corners[:, 0], corners[:, 2])
                depths = np.linalg.norm(corners[:, 0::2], axis=1, keepdims=False)
                yy = corners[:, 1] / depths
                theta_min = np.min(theta.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[1] - 1)
                theta_max = np.max(theta.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[1] - 1)
                yy_min = np.min(yy.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[0] - 1)
                yy_max = np.max(yy.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[0] - 1)
                x_min = theta_min * self.intrinsics[0, 0] + self.intrinsics[0, 2]
                x_max = theta_max * self.intrinsics[0, 0] + self.intrinsics[0, 2]
                y_min = yy_min * self.intrinsics[1, 1] + self.intrinsics[1, 2]
                y_max = yy_max * self.intrinsics[1, 1] + self.intrinsics[1, 2]
                results['gt_bboxes_area'] = (y_max - y_min) * (x_max - x_min)
                # clip bbox2d
                theta_min = np.clip(theta_min, self.theta_min, self.theta_max)
                theta_max = np.clip(theta_max, self.theta_min, self.theta_max)
                y_max_theta = np.where(theta_max <= 0, theta_max, 0)
                y_max_theta = np.where(theta_min >= 0, theta_min, y_max_theta)
                yy_min = np.clip(yy_min, self.Y_min * np.cos(y_max_theta), self.Y_max * np.cos(y_max_theta))
                yy_max = np.clip(yy_max, self.Y_min * np.cos(y_max_theta), self.Y_max * np.cos(y_max_theta))
                x_min = theta_min * self.intrinsics[0, 0] + self.intrinsics[0, 2]
                x_max = theta_max * self.intrinsics[0, 0] + self.intrinsics[0, 2]
                y_min = yy_min * self.intrinsics[1, 1] + self.intrinsics[1, 2]
                y_max = yy_max * self.intrinsics[1, 1] + self.intrinsics[1, 2]
                results['gt_bboxes'] = np.stack((x_min, y_min, x_max, y_max), axis=1)
        return results

    def _cylinder_cam2img(self, results):
        results['cam2img'][:3, :3] = self.intrinsics
        return results

    def _compute_scale(self, results):
        cam2img = results['cam2img']
        w_scale = self.intrinsics[0, 0] / cam2img[0, 0]
        h_scale = self.intrinsics[1, 1] / cam2img[1, 1]
        results['scale_factor'] = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        return results

    def _compute_bias(self, results):
        cam2img = results['cam2img']
        left = self.intrinsics[0, 2] - cam2img[0, 2]
        top = self.intrinsics[1, 2] - cam2img[1, 2]
        results['pad_bias'] = np.array([left, top, left, top], dtype=np.float32)
        return results

    def _affine_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img', ]):
            img = results[key]
            A1 = np.zeros([2, 3])
            A1[0, 0] = results['scale_factor'][0]
            A1[1, 1] = results['scale_factor'][1]
            A1[:, 2] = results['pad_bias'][:2]
            # cv2.imwrite(f'{results["sample_idx"]}_ori.jpg', img)
            if self.cycle:
                h, w, _ = img.shape
                delta_u = self.intrinsics[0, 2] - self.intrinsics_origin[0, 2]
                delta_v = self.intrinsics[1, 2] - self.intrinsics_origin[1, 2]
                A1[0, 2] -= delta_u
                A1[1, 2] -= delta_v
                img = cv2.warpAffine(img, A1, self.size[::-1], borderValue=self.pad_val)

                img = np.concatenate([img, img, img], axis=1)
                img = np.concatenate([img, img, img], axis=0)
                # A2 = np.zeros([2, 3])
                A1[0, 0] = 1
                A1[1, 1] = 1
                A1[0, 2] = delta_u - self.size[1]
                A1[1, 2] = delta_v - self.size[0]

            img = cv2.warpAffine(img, A1, self.size[::-1], borderValue=self.pad_val)
            # cv2.imwrite(f'{results["sample_idx"]}.jpg', img)
            results[key] = img
            results['img_shape'] = img.shape[:2]
            # in case that there is no padding
            results['pad_shape'] = img.shape[:2]
        return results

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', ['gt_bboxes', ]):
            if key in results.keys():
                bboxes = results[key] * results['scale_factor'][None, :]
                results[key] = bboxes
        if 'centers_2d' in results:
            results['centers_2d'] = results['centers_2d'] * results['scale_factor'][None, :2]
        if 'gt_bboxes_area' in results:
            results['gt_bboxes_area'] = results['gt_bboxes_area'] * results['scale_factor'][0] * \
                                        results['scale_factor'][1]
        return results

    def _resize_cam2img(self, results):
        wh_scale = results['scale_factor'][:2]
        results['cam2img'][:2, :] *= wh_scale[:, None]
        return results

    def _crop_bboxes(self, results):
        bias = results['pad_bias']
        for key in results.get('bbox_fields', ['gt_bboxes', ]):
            if key in results.keys():
                bbox = results[key] + bias[None, :]
                results[key] = bbox
        if 'centers_2d' in results:
            results['centers_2d'] = results['centers_2d'] + bias[None, :2]
        return results

    def _crop_cam2img(self, results):
        bias = results['pad_bias'][:2]
        results['cam2img'][:2, 2] += bias
        # results['cam2img'][:2, 3] += (results['cam2img'][2, 3] * bias)
        return results

    def _move_object_x(self, gt_bbox_3d, centers2d, gt_bboxes, depths, k, delta_u, K_out, cam2img):
        bbox3d1 = gt_bbox_3d.clone()
        centers2d1 = centers2d + k * delta_u[:, :2]
        gt_bboxes1 = gt_bboxes + k * delta_u
        xyz = bbox3d1[:, :3] + K_out
        rays = torch.atan2(xyz[:, 0], xyz[:, 2])
        if self.cylinder:
            theta_delta = k * delta_u[0, 0] / cam2img[0, 0]
            rays_new = rays + theta_delta
            bbox3d1[:, 0] = torch.sin(rays_new) * depths - K_out[0]
            bbox3d1[:, 2] = torch.cos(rays_new) * depths - K_out[2]
            bbox3d1[:, 6] = (bbox3d1[:, 6] + theta_delta + math.pi) % (2 * math.pi) - math.pi
        else:
            bbox3d1[:, 0] = k * delta_u[0, 0] * torch.from_numpy(depths) / cam2img[0, 0] + bbox3d1[:, 0]
            rays_new = torch.atan2(bbox3d1[:, 0] + K_out[0], xyz[:, 2])
            bbox3d1[:, 6] = (bbox3d1[:, 6] + rays_new - rays + math.pi) % (2 * math.pi) - math.pi
        return bbox3d1, centers2d1, gt_bboxes1

    def _move_object_y(self, gt_bbox_3d, centers2d, gt_bboxes, depths, k, delta_v, K_out, cam2img):
        bbox3d1 = gt_bbox_3d.clone()
        centers2d1 = centers2d + k * delta_v[:, :2]
        bbox3d1[:, 1] = k * delta_v[0, 1] * torch.from_numpy(depths) / cam2img[1, 1] + bbox3d1[:, 1]
        gt_bboxes1 = gt_bboxes + k * delta_v
        return bbox3d1, centers2d1, gt_bboxes1

    def _cycle_bboxes_xy(self, results, cx_min, cx_max, delta_u, axis='x'):
        if axis == 'x':
            move_object = self._move_object_x
        else:
            move_object = self._move_object_y
        gt_bbox_3d = results['gt_bboxes_3d'].tensor
        depths = results['depths']
        centers2d = results['centers_2d']
        gt_bboxes = results['gt_bboxes']
        gt_bbox_3d_list = [gt_bbox_3d, ]
        centers2d_list = [centers2d, ]
        gt_bboxes_list = [gt_bboxes, ]
        for cx in range(cx_min, cx_max + 1):
            if cx == 0:
                continue
            gt_bbox_3d1, centers2d1, gt_bboxes1 = move_object(gt_bbox_3d, centers2d, gt_bboxes, depths, cx, delta_u,
                                                              results['K_out'], results['cam2img'])
            gt_bbox_3d_list.append(gt_bbox_3d1)
            centers2d_list.append(centers2d1)
            gt_bboxes_list.append(gt_bboxes1)
        if len(gt_bboxes_list) != (cx_max - cx_min + 1):
            raise RuntimeError('缩放范围不合适')
        results['gt_bboxes_3d'].tensor = torch.cat(gt_bbox_3d_list, 0)
        results['centers_2d'] = np.concatenate(centers2d_list, 0)
        results['gt_bboxes'] = np.concatenate(gt_bboxes_list, 0)
        for key in ['gt_bboxes_labels', 'depths', 'gt_labels_3d', 'gt_bboxes_area']:
            if key in results:
                results[key] = np.concatenate([results[key] for _ in range(cx_max - cx_min + 1)], 0)
        return results

    def _cycle_bboxes(self, results):
        if self.cycle:
            delta_u = np.array([[self.size[1], 0, self.size[1], 0], ], dtype=np.float32)
            delta_v = np.array([[0, self.size[0], 0, self.size[0]], ], dtype=np.float32)
            results = self._cycle_bboxes_xy(results, -1, 1, delta_u, 'x')
            results = self._cycle_bboxes_xy(results, -1, 1, delta_v, 'y')
        return results

    def _compute_2d(self, results):
        if 'gt_bboxes_3d' in results:
            gt_bbox_3d = results['gt_bboxes_3d']
            intrinsics = results['cam2img']
            centers3d = gt_bbox_3d.gravity_center + results['K_out'][None, :]
            centers3d = centers3d.numpy()
            d = centers3d[:, 2]
            u = centers3d[:, 0] * intrinsics[0, 0] / d + intrinsics[0, 2]
            v = centers3d[:, 1] * intrinsics[1, 1] / d + intrinsics[1, 2]
            centers_2d = np.stack((u, v), axis=1)
            # print((np.abs(results['centers_2d'] - centers_2d)).max())
            # if np.abs(results['centers_2d'] - centers_2d).max() > 0.1:
            #     print(results['centers_2d'], centers_2d)
            #     print(0)
            results['centers_2d'] = centers_2d
            # print(np.abs(results['depths'] - d).max())
            # if np.abs(results['depths'] - d).max() > 0.1:
            #     print(results['depths'], d)
            #     print(0)
            results['depths'] = d
            corners = gt_bbox_3d.corners.view(-1, 3) + results['K_out'][None, :]
            corners = corners.numpy()
            d = corners[:, 2].clip(min=0.1)
            u = corners[:, 0] * intrinsics[0, 0] / d + intrinsics[0, 2]
            v = corners[:, 1] * intrinsics[1, 1] / d + intrinsics[1, 2]
            x_min = np.min(u.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[1] - 1)
            x_max = np.max(u.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[1] - 1)
            y_min = np.min(v.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[0] - 1)
            y_max = np.max(v.reshape(-1, 8), axis=1, keepdims=False)  # .clamp(0, self.size[0] - 1)
            results['gt_bboxes_area'] = (y_max - y_min) * (x_max - x_min)
            x_min = np.clip(x_min, 0, results['img_shape'][1] - 1)
            x_max = np.clip(x_max, 0, results['img_shape'][1] - 1)
            y_min = np.clip(y_min, 0, results['img_shape'][0] - 1)
            y_max = np.clip(y_max, 0, results['img_shape'][0] - 1)
            gt_bboxes = np.stack((x_min, y_min, x_max, y_max), axis=1)
            # print(np.abs(results['gt_bboxes'] - gt_bboxes).max())
            # if np.abs(results['gt_bboxes'] - gt_bboxes).max() > 0.1:
            #     print(results['gt_bboxes'], gt_bboxes)
            #     print(0)
            results['gt_bboxes'] = gt_bboxes
        return results

    def _clip_bboxes(self, results):
        if 'gt_bboxes' in results:
            results['gt_bboxes'][:, 0::2] = results['gt_bboxes'][:, 0::2].clip(0, self.size[1] - 1)
            results['gt_bboxes'][:, 1::2] = results['gt_bboxes'][:, 1::2].clip(0, self.size[0] - 1)
        return results

    def _init_intrinsics(self, results):
        self.intrinsics = self.intrinsics_origin.copy()
        if self.random_shift[0] != 0 or self.random_shift[1] != 0:
            x_shift = random.uniform(-self.random_shift[0], self.random_shift[0])
            y_shift = random.uniform(-self.random_shift[1], self.random_shift[1])
            self.intrinsics[0, 2] += x_shift
            self.intrinsics[1, 2] += y_shift

    def show_result(self, results):
        img = results['img']
        bbox = results['gt_bboxes']
        for b in bbox:
            img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.imwrite(f'{results["sample_idx"]}.jpg', img)
        return 0

    def catdepth(self, results):
        results['catdepth'] = np.array([-results['cam2img'][1, 2], results['img_shape'][0]], dtype=np.float32)
        results['catdepth'] = torch.from_numpy(results['catdepth'])
        return results

    def transform(self, results):
        self._init_intrinsics(results)
        if self.cylinder:
            results = self._cylinder_img(results)
            results = self._cylinder_bboxes(results)
            results = self._cylinder_cam2img(results)
        else:
            results = self._compute_2d(results)
            results = self._compute_scale(results)
            results = self._resize_bboxes(results)
            results = self._resize_cam2img(results)
            results = self._compute_bias(results)
            results = self._crop_bboxes(results)
            results = self._crop_cam2img(results)
            results = self._affine_img(results)
        results = self._cycle_bboxes(results)
        results = self._clip_bboxes(results)
        results = self.catdepth(results)
        # self.show_result(results)
        return results


@TRANSFORMS.register_module()
class RandomResizeCrop(UnifiedIntrinsics):
    def __init__(self, size, random_scale=(1, 1), unbias_sampling=True, cycle=False, pad_val=(114, 114, 114),
                 cylinder=False):
        self.size = size
        self.random_scale = random_scale
        self.cycle = cycle
        self.pad_val = pad_val
        self.cylinder = cylinder
        self.unbias_sampling = unbias_sampling

    def _compute_scale(self, results):
        if self.random_scale[0] != 1 or self.random_scale[1] != 1:
            if self.unbias_sampling:
                xy_scale = random.uniform(self.random_scale[0] ** 3, self.random_scale[1] ** 3)
                xy_scale = xy_scale ** (1 / 3)
            else:
                xy_scale = random.uniform(self.random_scale[0], self.random_scale[1])
        else:
            xy_scale = 1
        results['catdepth'] = results['catdepth'] * xy_scale
        results['scale_factor'] = np.array([xy_scale, xy_scale, xy_scale, xy_scale], dtype=np.float32)
        results['img_shape_scale'] = [i * xy_scale for i in results['img_shape']]
        return results

    def _compute_bias(self, results):
        h, w = results['img_shape_scale']
        x_shift = self.size[1] * 0.5 + random.uniform(-w, 0)
        y_shift = self.size[0] * 0.5 + random.uniform(-h, 0)
        results['pad_bias'] = np.array([x_shift, y_shift, x_shift, y_shift], dtype=np.float32)
        return results

    def _affine_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img', ]):
            img = results[key]
            A1 = np.zeros([2, 3])
            A1[0, 0] = results['scale_factor'][0]
            A1[1, 1] = results['scale_factor'][1]
            A1[:, 2] = results['pad_bias'][:2]
            # cv2.imwrite(f'{results["sample_idx"]}_ori.jpg', img)
            if self.cycle:
                borderMode = cv2.BORDER_WRAP
                borderValue = None
            else:
                borderMode = cv2.BORDER_CONSTANT
                borderValue = self.pad_val
            img = cv2.warpAffine(img, A1, self.size[::-1], borderMode=borderMode, borderValue=borderValue)
            # cv2.imwrite(f'{results["sample_idx"]}.jpg', img)
            results[key] = img
            results['img_shape'] = img.shape[:2]
            # in case that there is no padding
            results['pad_shape'] = img.shape[:2]
        return results

    def _cycle_bboxes(self, results):
        if self.cycle:
            h, w = results.pop('img_shape_scale')
            x_shift = results['pad_bias'][0]
            y_shift = results['pad_bias'][1]
            cx_min = int((-x_shift) // w)
            cx_max = int((-x_shift + self.size[1]) // w)
            cy_min = int((-y_shift) // h)
            cy_max = int((-y_shift + self.size[0]) // h)
            delta_u = np.array([[w, 0, w, 0], ], dtype=np.float32)
            delta_v = np.array([[0, h, 0, h, ]], dtype=np.float32)
            results = self._cycle_bboxes_xy(results, cx_min, cx_max, delta_u, 'x')
            results = self._cycle_bboxes_xy(results, cy_min, cy_max, delta_v, 'y')
        return results

    def transform(self, results):
        results = self._compute_scale(results)
        results = self._resize_bboxes(results)
        results = self._resize_cam2img(results)
        results = self._compute_bias(results)
        results = self._crop_bboxes(results)
        results = self._crop_cam2img(results)
        results = self._affine_img(results)
        results = self._cycle_bboxes(results)
        results = self._clip_bboxes(results)
        return results


@TRANSFORMS.register_module()
class RandomCrop(RandomResizeCrop):
    def __init__(self, size, random_shift=(0, 0), cycle=False, pad_val=(114, 114, 114), cylinder=False):
        self.size = size
        self.random_shift = random_shift
        self.cycle = cycle
        self.pad_val = pad_val
        self.cylinder = cylinder

    def _compute_bias(self, results):
        x_shift = random.uniform(-self.random_shift[0], self.random_shift[0])
        y_shift = random.uniform(-self.random_shift[1], self.random_shift[1])
        results['pad_bias'] = np.array([x_shift, y_shift, x_shift, y_shift], dtype=np.float32)
        return results

    def _affine_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img', ]):
            img = results[key]
            A1 = np.zeros([2, 3])
            A1[0, 0] = 1
            A1[1, 1] = 1
            A1[:, 2] = results['pad_bias'][:2]
            # cv2.imwrite(f'{results["sample_idx"]}_ori.jpg', img)
            if self.cycle:
                borderMode = cv2.BORDER_WRAP
                borderValue = None
            else:
                borderMode = cv2.BORDER_CONSTANT
                borderValue = self.pad_val
            img = cv2.warpAffine(img, A1, self.size[::-1], borderMode=borderMode, borderValue=borderValue)
            # cv2.imwrite(f'{results["sample_idx"]}.jpg', img)
            results[key] = img
            results['img_shape'] = img.shape[:2]
            # in case that there is no padding
            results['pad_shape'] = img.shape[:2]
        return results

    def _cycle_bboxes(self, results):
        if self.cycle:
            h, w = results['img_shape']
            x_shift = results['pad_bias'][0]
            y_shift = results['pad_bias'][1]
            cx_min = int((-x_shift) // w)
            cx_max = int((-x_shift + self.size[1]) // w)
            cy_min = int((-y_shift) // h)
            cy_max = int((-y_shift + self.size[0]) // h)
            delta_u = np.array([[w, 0, w, 0], ], dtype=np.float32)
            delta_v = np.array([[0, h, 0, h, ]], dtype=np.float32)
            results = self._cycle_bboxes_xy(results, cx_min, cx_max, delta_u, 'x')
            results = self._cycle_bboxes_xy(results, cy_min, cy_max, delta_v, 'y')
        return results

    def transform(self, results):
        results = self._compute_bias(results)
        results = self._crop_bboxes(results)
        results = self._crop_cam2img(results)
        results = self._cycle_bboxes(results)
        results = self._clip_bboxes(results)
        results = self._affine_img(results)
        # self.show_result(results)
        return results


@TRANSFORMS.register_module()
class PitchCam(BaseTransform):
    def __init__(self, random_theta=3,
                 cylinder=False,
                 pad_val=(114, 114, 114)):
        self.random_theta = random_theta
        # self.cylinder = cylinder
        self.pad_val = pad_val
        self.cylinder = cylinder

    def show_result(self, results, name='cutmix3d'):
        img = results['img'].copy()
        bbox = results['gt_bboxes']
        P2 = results['cam2img']
        K_out = results['K_out'].numpy()
        for i, b in enumerate(bbox):
            # img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
            label = results['gt_bboxes_3d'].tensor[i, :]
            h, w, l = float(label[4]), float(label[5]), float(label[3])
            x, y, z = float(label[0]), float(label[1]), float(label[2])
            yaw = float(label[6])  # 朝向角度
            # 计算3D框的8个顶点坐标
            corners = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                [0, 0, 0, 0, -h, -h, -h, -h],
                                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                                ])
            R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                          [0, 1, 0],
                          [-np.sin(yaw), 0, np.cos(yaw)]])
            corners = R @ corners + np.array([[x], [y], [z]])
            # 投影到图像上
            # corners_h = np.vstack((corners, np.ones((1, 8))))
            corners = corners + K_out[:, None]
            if self.cylinder:
                corners = np.stack([np.arctan2(corners[0, :], corners[2, :]),
                                    corners[1, :] / np.linalg.norm(corners[0::2, :], axis=0, keepdims=False),
                                    np.ones([8, ])], axis=0)
            corners_2d = P2 @ corners
            corners_2d /= corners_2d[2:, :]
            corners_2d = corners_2d[:2, :].transpose()
            # 绘制3D框
            imgpts = np.int32(corners_2d).reshape(-1, 2)
            cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 2)  # 底部矩形
            for i in range(4):
                cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i + 4]), (0, 255, 0), 2)  # 框体纵��线
            cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)  # 顶部矩形
            img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.imwrite(f'{results["sample_idx"]}_{name}.jpg', img)
        return 0

    def _compute_bias(self, results):
        theta = random.uniform(-self.random_theta, self.random_theta)
        # trans_order = random.randint(0, 2)
        trans_order = 1
        # theta = np.sign(theta) * 40
        theta = theta * np.pi / 180
        results['theta_bias'] = theta
        results['trans_order'] = trans_order
        return results

    def _compute_img(self, results):
        cam2img = results['cam2img']
        theta = results.pop('theta_bias')
        trans_order = results.pop('trans_order')
        for key in results.get('img_fields', ['img', ]):
            img = results[key]
            h, w, _ = img.shape
            x = np.arange(w, dtype=np.float32)
            y = np.arange(h, dtype=np.float32)
            x, y = np.meshgrid(x, y)
            X = (x - cam2img[0, 2]) / cam2img[0, 0]
            Y = (y - cam2img[1, 2]) / cam2img[1, 1]
            if self.cylinder:
                Y = Y / np.cos(X)
                X = np.tan(X)
            if trans_order == 1:
                Y = Y - np.tan(theta)
            X = X / (-np.sin(theta) * Y + np.cos(theta))
            Y = (Y * np.cos(theta) + np.sin(theta)) / (-Y * np.sin(theta) + np.cos(theta))
            if trans_order == 0:
                Y = Y - np.tan(theta)
            if self.cylinder:
                Y = Y / np.sqrt(X ** 2 + 1)
                X = np.arctan2(X, 1)
            u = (cam2img[0, 0] * X + cam2img[0, 2])
            v = (cam2img[1, 1] * Y + cam2img[1, 2])
            img = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderValue=self.pad_val)
            # cv2.imwrite(f'{results["sample_idx"]}.jpg', img)
            results[key] = img
            results['img_shape'] = img.shape[:2]
            # in case that there is no padding
            results['pad_shape'] = img.shape[:2]
        return results

    def _compute_bboxes(self, results):
        if 'gt_bboxes_3d' in results:
            theta = results['theta_bias']
            trans_order = results['trans_order']
            rot = np.array([[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]],
                           dtype=np.float32)
            gt_bbox_3d = results['gt_bboxes_3d']
            # intrinsics = results['cam2img']
            centers3d = gt_bbox_3d.gravity_center + results['K_out'][None, :]

            angles = gt_bbox_3d.tensor[:, -1]
            dims_l = gt_bbox_3d.tensor[:, 3:4]
            rot_sin = torch.sin(angles)
            rot_cos = torch.cos(angles)
            zeros = torch.zeros_like(rot_cos)
            front_center3d = centers3d + torch.stack([rot_cos, zeros, -rot_sin], dim=1) * dims_l * 0.5
            centers3d = centers3d.numpy()
            front_center3d = front_center3d.numpy()
            if trans_order == 0:
                centers3d[:, 1] = np.tan(theta) * centers3d[:, 2] + centers3d[:, 1]
            centers3d = np.dot(centers3d, rot)
            front_center3d = np.dot(front_center3d, rot)
            dir_vec = front_center3d - centers3d
            angles2 = np.arctan2(-dir_vec[:, 2], dir_vec[:, 0])
            # if self.cylinder:
            #     results['depths'] = np.linalg.norm(centers3d[:, 0::2], axis=1, keepdims=False)
            # else:
            if trans_order == 1:
                centers3d[:, 1] = np.tan(theta) * centers3d[:, 2] + centers3d[:, 1]
            centers3d = torch.from_numpy(centers3d)
            angles2 = torch.from_numpy(angles2)
            centers3d += gt_bbox_3d.tensor[:, 3:6] * centers3d.new_tensor((0.0, 0.5, 0.0))
            results['gt_bboxes_3d'].tensor[:, :3] = centers3d - results['K_out'][None, :]
            results['gt_bboxes_3d'].tensor[:, -1] = (angles2 + np.pi) % (2 * np.pi) - np.pi
        return results

    def transform(self, results):
        # self.show_result(results, 'ori')
        results = self._compute_bias(results)
        results = self._compute_bboxes(results)
        results = self._compute_img(results)
        # self.show_result(results, 'pitchcam')
        return results


@TRANSFORMS.register_module()
class RollCam(PitchCam):
    def _compute_bias(self, results):
        theta = random.uniform(-self.random_theta, self.random_theta)
        # trans_order = random.randint(0, 2)
        trans_order = 1
        # theta = np.sign(theta) * 40
        theta = theta * np.pi / 180
        results['theta_bias'] = theta
        results['trans_order'] = trans_order
        return results

    def _compute_img(self, results):
        cam2img = results['cam2img']
        theta = results.pop('theta_bias')
        trans_order = results.pop('trans_order')
        for key in results.get('img_fields', ['img', ]):
            img = results[key]
            h, w, _ = img.shape
            x = np.arange(w, dtype=np.float32)
            y = np.arange(h, dtype=np.float32)
            x, y = np.meshgrid(x, y)
            X = (x - cam2img[0, 2]) / cam2img[0, 0]
            Y = (y - cam2img[1, 2]) / cam2img[1, 1]
            if self.cylinder:
                Y = Y / np.cos(X)
                X = np.tan(X)
            if trans_order == 1:
                Y = Y + np.tan(theta) * X
            X, Y = np.cos(theta) * X + np.sin(theta) * Y, Y * np.cos(theta) - np.sin(theta) * X
            if trans_order == 0:
                Y = Y + np.tan(theta) * X
            if self.cylinder:
                Y = Y / np.sqrt(X ** 2 + 1)
                X = np.arctan2(X, 1)
            u = (cam2img[0, 0] * X + cam2img[0, 2])
            v = (cam2img[1, 1] * Y + cam2img[1, 2])
            img = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderValue=self.pad_val)
            # cv2.imwrite(f'{results["sample_idx"]}.jpg', img)
            results[key] = img
            results['img_shape'] = img.shape[:2]
            # in case that there is no padding
            results['pad_shape'] = img.shape[:2]
        return results

    def _compute_bboxes(self, results):
        if 'gt_bboxes_3d' in results:
            theta = results['theta_bias']
            trans_order = results['trans_order']
            rot = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
                           dtype=np.float32)
            gt_bbox_3d = results['gt_bboxes_3d']
            # intrinsics = results['cam2img']
            centers3d = gt_bbox_3d.gravity_center + results['K_out'][None, :]
            angles = gt_bbox_3d.tensor[:, -1]
            dims_l = gt_bbox_3d.tensor[:, 3:4]
            rot_sin = torch.sin(angles)
            rot_cos = torch.cos(angles)
            zeros = torch.zeros_like(rot_cos)
            front_center3d = centers3d + torch.stack([rot_cos, zeros, -rot_sin], dim=1) * dims_l * 0.5
            if trans_order == 0:
                centers3d[:, 1] = -np.tan(theta) * centers3d[:, 0] + centers3d[:, 1]
                front_center3d[:, 1] = -np.tan(theta) * front_center3d[:, 0] + front_center3d[:, 1]
            centers3d = centers3d.numpy()
            front_center3d = front_center3d.numpy()
            centers3d = np.dot(centers3d, rot)
            front_center3d = np.dot(front_center3d, rot)
            dir_vec = front_center3d - centers3d
            angles2 = np.arctan2(-dir_vec[:, 2], dir_vec[:, 0])
            if trans_order == 1:
                centers3d[:, 1] = -np.tan(theta) * centers3d[:, 0] + centers3d[:, 1]
            centers3d = torch.from_numpy(centers3d)
            angles2 = torch.from_numpy(angles2)
            centers3d += gt_bbox_3d.tensor[:, 3:6] * centers3d.new_tensor((0.0, 0.5, 0.0))
            results['gt_bboxes_3d'].tensor[:, :3] = centers3d - results['K_out'][None, :]
            results['gt_bboxes_3d'].tensor[:, -1] = (angles2 + np.pi) % (2 * np.pi) - np.pi
        return results

    def transform(self, results):
        # self.show_result(results, 'ori')
        results = self._compute_bias(results)
        results = self._compute_bboxes(results)
        results = self._compute_img(results)
        # results = self.update2d(results)
        # self.show_result(results, 'rollcam')
        return results


@TRANSFORMS.register_module()
class PitchRollCam(BaseTransform):
    def __init__(self, random_theta=(2, 2),
                 cylinder=False,
                 pad_val=(114, 114, 114)):
        self.random_theta = random_theta
        # self.cylinder = cylinder
        self.pad_val = pad_val
        self.cylinder = cylinder

    def show_result(self, results, name='cutmix3d'):
        img = results['img'].copy()
        bbox = results['gt_bboxes']
        P2 = results['cam2img']
        K_out = results['K_out'].numpy()
        for i, b in enumerate(bbox):
            # img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
            label = results['gt_bboxes_3d'].tensor[i, :]
            h, w, l = float(label[4]), float(label[5]), float(label[3])
            x, y, z = float(label[0]), float(label[1]), float(label[2])
            yaw = float(label[6])  # 朝向角度
            # 计算3D框的8个顶点坐标
            corners = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                [0, 0, 0, 0, -h, -h, -h, -h],
                                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                                ])
            R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                          [0, 1, 0],
                          [-np.sin(yaw), 0, np.cos(yaw)]])
            corners = R @ corners + np.array([[x], [y], [z]])
            # 投影到图像上
            # corners_h = np.vstack((corners, np.ones((1, 8))))
            corners = corners + K_out[:, None]
            if self.cylinder:
                corners = np.stack([np.arctan2(corners[0, :], corners[2, :]),
                                    corners[1, :] / np.linalg.norm(corners[0::2, :], axis=0, keepdims=False),
                                    np.ones([8, ])], axis=0)
            corners_2d = P2 @ corners
            corners_2d /= corners_2d[2:, :]
            corners_2d = corners_2d[:2, :].transpose()
            # 绘制3D框
            imgpts = np.int32(corners_2d).reshape(-1, 2)
            cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 2)  # 底部矩形
            for i in range(4):
                cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i + 4]), (0, 255, 0), 2)  # 框体纵��线
            cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)  # 顶部矩形
            img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.imwrite(f'{results["sample_idx"]}_{name}.jpg', img)
        return 0

    def _compute_bias(self, results):
        theta = random.uniform(-1, 1, size=[2, ])
        # theta = random.uniform(0, 2 * np.pi)
        # theta = random.uniform(0, 1) * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        # trans_order = random.randint(0, 2)
        trans_order = 1
        # theta = np.sign(theta) * 20
        theta = theta * self.random_theta * np.pi / 180
        theta[0] = np.arctan(np.sin(theta[0]) / np.sqrt(1 - (np.sin(theta[0])) ** 2 - (np.sin(theta[1])) ** 2))
        results['theta_bias'] = theta
        results['trans_order'] = trans_order
        return results

    def _compute_img(self, results):
        cam2img = results['cam2img']
        theta = results.pop('theta_bias')
        trans_order = results.pop('trans_order')
        for key in results.get('img_fields', ['img', ]):
            img = results[key]
            h, w, _ = img.shape
            x = np.arange(w, dtype=np.float32)
            y = np.arange(h, dtype=np.float32)
            x, y = np.meshgrid(x, y)
            X = (x - cam2img[0, 2]) / cam2img[0, 0]
            Y = (y - cam2img[1, 2]) / cam2img[1, 1]
            if self.cylinder:
                Y = Y / np.cos(X)
                X = np.tan(X)
            if trans_order == 0:
                Y = Y + np.tan(theta[1]) * X - np.tan(theta[0]) / np.cos(theta[1])
                X, Y = np.cos(theta[1]) * X + np.sin(theta[1]) * Y, Y * np.cos(theta[1]) - np.sin(theta[1]) * X
                X = X / (-np.sin(theta[0]) * Y + np.cos(theta[0]))
                Y = (Y * np.cos(theta[0]) + np.sin(theta[0])) / (-Y * np.sin(theta[0]) + np.cos(theta[0]))
            else:
                Y = Y + np.tan(theta[1]) * X / np.cos(theta[0]) - np.tan(theta[0])
                X = X / (-np.sin(theta[0]) * Y + np.cos(theta[0]))
                Y = (Y * np.cos(theta[0]) + np.sin(theta[0])) / (-Y * np.sin(theta[0]) + np.cos(theta[0]))
                X, Y = np.cos(theta[1]) * X + np.sin(theta[1]) * Y, Y * np.cos(theta[1]) - np.sin(theta[1]) * X
            if self.cylinder:
                Y = Y / np.sqrt(X ** 2 + 1)
                X = np.arctan2(X, 1)
            u = (cam2img[0, 0] * X + cam2img[0, 2])
            v = (cam2img[1, 1] * Y + cam2img[1, 2])
            img = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderValue=self.pad_val)
            # cv2.imwrite(f'{results["sample_idx"]}.jpg', img)
            results[key] = img
            results['img_shape'] = img.shape[:2]
            # in case that there is no padding
            results['pad_shape'] = img.shape[:2]
        return results

    def _compute_bboxes(self, results):
        if 'gt_bboxes_3d' in results:
            theta = results['theta_bias']
            trans_order = results['trans_order']
            rot1 = np.array(
                [[1, 0, 0], [0, np.cos(theta[0]), np.sin(theta[0])], [0, -np.sin(theta[0]), np.cos(theta[0])]],
                dtype=np.float32)
            rot2 = np.array(
                [[np.cos(theta[1]), np.sin(theta[1]), 0], [-np.sin(theta[1]), np.cos(theta[1]), 0], [0, 0, 1]],
                dtype=np.float32)
            gt_bbox_3d = results['gt_bboxes_3d']
            # intrinsics = results['cam2img']
            centers3d = gt_bbox_3d.gravity_center + results['K_out'][None, :]
            angles = gt_bbox_3d.tensor[:, -1]
            dims_l = gt_bbox_3d.tensor[:, 3:4]
            rot_sin = torch.sin(angles)
            rot_cos = torch.cos(angles)
            zeros = torch.zeros_like(rot_cos)
            front_center3d = centers3d + torch.stack([rot_cos, zeros, -rot_sin], dim=1) * dims_l * 0.5
            if trans_order == 0:
                rot = np.dot(rot1, rot2)
            else:
                rot = np.dot(rot2, rot1)
            centers3d = centers3d.numpy()
            front_center3d = front_center3d.numpy()
            centers3d = np.dot(centers3d, rot)
            front_center3d = np.dot(front_center3d, rot)
            dir_vec = front_center3d - centers3d
            angles2 = np.arctan2(-dir_vec[:, 2], dir_vec[:, 0])
            if trans_order == 0:
                centers3d[:, 1] = np.tan(theta[0]) * centers3d[:, 2] / np.cos(theta[1]) - \
                                  np.tan(theta[1]) * centers3d[:, 0] + centers3d[:, 1]
            else:
                centers3d[:, 1] = np.tan(theta[0]) * centers3d[:, 2] - \
                                  np.tan(theta[1]) * centers3d[:, 0] / np.cos(theta[0]) + centers3d[:, 1]
            centers3d = torch.from_numpy(centers3d)
            angles2 = torch.from_numpy(angles2)
            centers3d += gt_bbox_3d.tensor[:, 3:6] * centers3d.new_tensor((0.0, 0.5, 0.0))
            results['gt_bboxes_3d'].tensor[:, :3] = centers3d - results['K_out'][None, :]
            results['gt_bboxes_3d'].tensor[:, -1] = (angles2 + np.pi) % (2 * np.pi) - np.pi
        return results

    def transform(self, results):
        # self.show_result(results, 'ori')
        results = self._compute_bias(results)
        results = self._compute_bboxes(results)
        results = self._compute_img(results)
        # self.show_result(results, 'pitchrollcam')
        return results


@TRANSFORMS.register_module()
class BackgroundNoise(BaseTransform):
    def __init__(self, prob=1.0):
        self.prob = prob

    def show_result(self, results):
        img = results['img']
        bbox = results['gt_bboxes']
        P2 = results['cam2img']
        K_out = results['K_out']
        for i, b in enumerate(bbox):
            # img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
            label = results['gt_bboxes_3d'].tensor[i, :]
            h, w, l = float(label[4]), float(label[5]), float(label[3])
            x, y, z = float(label[0]), float(label[1]), float(label[2])
            yaw = float(label[6])  # 朝向角度
            # 计算3D框的8个顶点坐标
            corners = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                [0, 0, 0, 0, -h, -h, -h, -h],
                                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                                ])
            R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                          [0, 1, 0],
                          [-np.sin(yaw), 0, np.cos(yaw)]])
            corners = R @ corners + np.array([[x], [y], [z]])

            # 投影到图像上
            # corners_h = np.vstack((corners, np.ones((1, 8))))
            corners_h = corners + K_out[None, :]
            corners_2d = P2 @ corners_h
            corners_2d /= corners_2d[2:, :]
            corners_2d = corners_2d[:2, :].transpose()

            # 绘制3D框
            imgpts = np.int32(corners_2d).reshape(-1, 2)
            cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 2)  # 底部矩形
            for i in range(4):
                cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i + 4]), (0, 255, 0), 2)  # 框体纵��线
            cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)  # 顶部矩形
        cv2.imwrite(f'{results["sample_idx"]}_BackgroundNoise.jpg', img)
        return 0

    def _compute_img(self, results):
        bbox = results['gt_bboxes']
        # cam2img = results['cam2img']
        for key in results.get('img_fields', ['img', ]):
            img = results[key]
            h, w, _ = img.shape
            mask = np.ones([h, w, 1], dtype=np.float32) * np.random.uniform(0, 1, size=[1, 1, 1])
            # mask = np.random.uniform(0, 1, size=img.shape)
            noise = np.random.uniform(0, 255, size=img.shape)
            # noise = 114
            kernel = np.random.randint(1, 5, size=(2,))
            for b in bbox:
                x_min, y_min, x_max, y_max = int(b[0]), int(b[1]), int(b[2]) + 1, int(b[3]) + 1
                # bbox_h, bbox_w = y_max - y_min, x_max - x_min
                x_min = max(x_min - kernel[0], 0)
                y_min = max(y_min - kernel[1], 0)
                x_max = min(x_max + kernel[0], w - 1)
                y_max = min(y_max + kernel[1], h - 1)
                mask[y_min:y_max, x_min:x_max, :] = 0
            mask = cv2.blur(mask, kernel * 2 - 1, borderType=cv2.BORDER_REPLICATE)
            mask = mask.reshape([h, w, -1])
            # mask = mask[:,:,None]
            img = img * (1 - mask) + noise * mask
            # img = img.clip(0, 255)
            results[key] = img.astype(np.uint8)
            results['img_shape'] = img.shape[:2]
            # in case that there is no padding
            results['pad_shape'] = img.shape[:2]
        return results

    def transform(self, results):
        if np.random.uniform() < self.prob:
            results = self._compute_img(results)
        self.show_result(results)
        return results


@TRANSFORMS.register_module()
class RandomErasing(BaseTransform):
    def __init__(
            self,
            n_patches: Union[int, Tuple[int, int]],
            ratio: Union[float, Tuple[float, float]],
            squared: bool = False, border=False, fill_val=(114, 114, 114),
    ) -> None:
        if isinstance(n_patches, tuple):
            assert len(n_patches) == 2 and 0 <= n_patches[0] <= n_patches[1]
        else:
            n_patches = (n_patches, n_patches)
        if isinstance(ratio, tuple):
            assert len(ratio) == 2 and 0 <= ratio[0] < ratio[1] <= 1
        else:
            ratio = (ratio, ratio)
        self.n_patches = n_patches
        self.ratio = ratio
        self.squared = squared
        self.border = border
        self.fill_val = fill_val

    def show_result(self, results, name='RandomErasing'):
        img = results['img'].copy()
        bbox = results['gt_bboxes']
        P2 = results['cam2img']
        K_out = results['K_out'].numpy()
        for i, b in enumerate(bbox):
            # img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
            label = results['gt_bboxes_3d'].tensor[i, :]
            h, w, l = float(label[4]), float(label[5]), float(label[3])
            x, y, z = float(label[0]), float(label[1]), float(label[2])
            yaw = float(label[6])  # 朝向角度
            # 计算3D框的8个顶点坐标
            corners = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                [0, 0, 0, 0, -h, -h, -h, -h],
                                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                                ])
            R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                          [0, 1, 0],
                          [-np.sin(yaw), 0, np.cos(yaw)]])
            corners = R @ corners + np.array([[x], [y], [z]])
            # 投影到图像上
            # corners_h = np.vstack((corners, np.ones((1, 8))))
            corners = corners + K_out[:, None]
            if True:
                corners = np.stack([np.arctan2(corners[0, :], corners[2, :]),
                                    corners[1, :] / np.linalg.norm(corners[0::2, :], axis=0, keepdims=False),
                                    np.ones([8, ])], axis=0)
            corners_2d = P2 @ corners
            corners_2d /= corners_2d[2:, :]
            corners_2d = corners_2d[:2, :].transpose()
            # 绘制3D框
            imgpts = np.int32(corners_2d).reshape(-1, 2)
            cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 2)  # 底部矩形
            for i in range(4):
                cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i + 4]), (0, 255, 0), 2)  # 框体纵��线
            cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)  # 顶部矩形
            img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.imwrite(f'{results["sample_idx"]}_{name}.jpg', img)
        return 0

    def _get_patches(self, bboxes2d: Tuple[int, int]) -> List[list]:
        """Get patches for random erasing."""
        patches = []
        for bbox2d in bboxes2d:
            bbox2d = bbox2d.astype(np.int32)
            n_patches = np.random.randint(self.n_patches[0], self.n_patches[1] + 1)
            for _ in range(n_patches):
                ratio = np.random.uniform(*self.ratio)
                if self.squared:
                    ratio = np.sqrt(ratio)
                    ratio = (ratio, ratio)
                else:
                    a = np.random.uniform(ratio, 1)
                    b = ratio + 1 - a
                    ratio = (np.sqrt(ratio * a / b), np.sqrt(ratio * b / a))
                ph, pw = int((bbox2d[3] - bbox2d[1]) * ratio[0]), int((bbox2d[2] - bbox2d[0]) * ratio[1])
                if self.border:
                    if np.random.randint(2):
                        px1 = bbox2d[0]
                    else:
                        px1 = bbox2d[2] - pw
                    if np.random.randint(2):
                        py1 = bbox2d[1]
                    else:
                        py1 = bbox2d[3] - ph
                else:
                    px1 = np.random.randint(bbox2d[0], bbox2d[2] - pw)
                    py1 = np.random.randint(bbox2d[1], bbox2d[3] - ph)
                px2, py2 = px1 + pw, py1 + ph
                patches.append([px1, py1, px2, py2])
        return patches

    def _transform_img(self, results: dict, patches: List[list]) -> None:
        """Random erasing the image."""
        for patch in patches:
            px1, py1, px2, py2 = patch
            if self.fill_val is None:
                results['img'][py1:py2, px1:px2, :] = np.random.randint(0, 256, size=[py2 - py1, px2 - px1, 3],
                                                                        dtype=np.uint8)
            else:
                results['img'][py1:py2, px1:px2, :] = self.fill_val
        return results

    def transform(self, results: dict) -> dict:
        """Transform function to erase some regions of image."""
        patches = self._get_patches(results['gt_bboxes'])
        results = self._transform_img(results, patches)
        # self.show_result(results)
        return results


@TRANSFORMS.register_module()
class K_out(BaseTransform):
    def transform(self, results):
        results['cam2img'] = np.array(results['cam2img'], dtype=np.float32)
        results['cam2img_ori'] = results['cam2img'].copy()
        results['img_shape_ori'] = results['img_shape'][:]
        cam2img = results['cam2img']
        fu = cam2img[0, 0]
        fv = cam2img[1, 1]
        cu = cam2img[0, 2]
        cv = cam2img[1, 2]
        if results['cam2img'].shape[1] == 4:
            K_out = np.array(
                [(cam2img[0, 3] - cu * cam2img[2, 3]) / fu, (cam2img[1, 3] - cv * cam2img[2, 3]) / fv, cam2img[2, 3]],
                dtype=np.float32)
            results['K_out'] = torch.from_numpy(K_out)
            results['cam2img'] = cam2img[:3, :3]
        else:
            results['K_out'] = torch.zeros([3, ])
        return results


@TRANSFORMS.register_module()
class Img2Cam(BaseTransform):
    def transform(self, results):
        cam2img = results['cam2img']
        results['img2cam'] = torch.from_numpy(np.linalg.inv(cam2img))
        return results


@TRANSFORMS.register_module()
class FilterObject(BaseTransform):
    def __init__(self, wo_small=8, wo_ratio=0, wo_occ=0, wo_centerout=False):
        self.wo_small = wo_small
        self.wo_ratio = wo_ratio
        self.wo_occ = wo_occ
        self.wo_centerout = wo_centerout

    def show_result(self, results, name='FilterObject'):
        img = results['img'].copy()
        bbox = results['gt_bboxes']
        P2 = results['cam2img']
        K_out = results['K_out'].numpy()
        for i, b in enumerate(bbox):
            # img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
            label = results['gt_bboxes_3d'].tensor[i, :]
            h, w, l = float(label[4]), float(label[5]), float(label[3])
            x, y, z = float(label[0]), float(label[1]), float(label[2])
            yaw = float(label[6])  # 朝向角度
            # 计算3D框的8个顶点坐标
            corners = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                [0, 0, 0, 0, -h, -h, -h, -h],
                                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                                ])
            R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                          [0, 1, 0],
                          [-np.sin(yaw), 0, np.cos(yaw)]])
            corners = R @ corners + np.array([[x], [y], [z]])
            # 投影到图像上
            # corners_h = np.vstack((corners, np.ones((1, 8))))
            corners = corners + K_out[:, None]
            if False:
                corners = np.stack([np.arctan2(corners[0, :], corners[2, :]),
                                    corners[1, :] / np.linalg.norm(corners[0::2, :], axis=0, keepdims=False),
                                    np.ones([8, ])], axis=0)
            corners_2d = P2 @ corners
            corners_2d /= corners_2d[2:, :]
            corners_2d = corners_2d[:2, :].transpose()
            # 绘制3D框
            imgpts = np.int32(corners_2d).reshape(-1, 2)
            cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 2)  # 底部矩形
            for i in range(4):
                cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i + 4]), (0, 255, 0), 2)  # 框体纵��线
            cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)  # 顶部矩形
            img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.imwrite(f'{results["sample_idx"]}_{name}.jpg', img)
        return 0

    def transform(self, results):
        if 'gt_bboxes' in results:
            if results['gt_bboxes'].shape[0] > 0:
                bbox_w = results['gt_bboxes'][:, 2] - results['gt_bboxes'][:, 0]
                bbox_h = results['gt_bboxes'][:, 3] - results['gt_bboxes'][:, 1]
                valid_mask = (bbox_w > self.wo_small)
                valid_mask *= (bbox_h > self.wo_small)
                bbox_ratio = bbox_h / bbox_w.clip(min=0.1)
                if self.wo_ratio > 0:
                    valid_mask *= ((bbox_ratio < self.wo_ratio) * (bbox_ratio > (1 / self.wo_ratio)))
                if 'gt_bboxes_area' in results and self.wo_occ > 0:
                    bbox_occ = (bbox_w * bbox_h) / results['gt_bboxes_area']
                    valid_mask *= (bbox_occ > self.wo_occ)
                if self.wo_centerout:
                    temp1 = results['centers_2d'][:, 0] > results['gt_bboxes'][:, 0]
                    temp2 = results['centers_2d'][:, 0] < results['gt_bboxes'][:, 2]
                    temp3 = results['centers_2d'][:, 1] > results['gt_bboxes'][:, 1]
                    temp4 = results['centers_2d'][:, 1] < results['gt_bboxes'][:, 3]
                    valid_mask *= (temp1 * temp2 * temp3 * temp4)
                results['gt_bboxes'] = results['gt_bboxes'][valid_mask]
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][valid_mask]
                results['centers_2d'] = results['centers_2d'][valid_mask]
                results['depths'] = results['depths'][valid_mask]
                results['gt_labels_3d'] = results['gt_labels_3d'][valid_mask]
                results['gt_bboxes_labels'] = results['gt_bboxes_labels'][valid_mask]
                # self.show_result(results)
        return results


@TRANSFORMS.register_module()
class GetTarget(BaseTransform):
    def __init__(self, relative_depth=False):
        self.relative_depth = relative_depth

    def transform(self, results):
        results['box_type_3d'] = type(results['gt_bboxes_3d'])
        xyz = results['gt_bboxes_3d'].gravity_center
        lhw = results['gt_bboxes_3d'].dims
        ry = results['gt_bboxes_3d'].yaw
        corners = results['gt_bboxes_3d'].corners.flatten(1, -1)
        centers_2d = torch.from_numpy(results['centers_2d'])
        if self.relative_depth:
            depths = torch.from_numpy(results['depths'] / results['cam2img'][1, 1])
        else:
            depths = torch.from_numpy(results['depths'])
        K_out = results['K_out']
        alphas = ry - torch.atan2(xyz[:, 0] + K_out[0], xyz[:, 2] + K_out[2])
        alphas = alphas % (2 * np.pi)
        results['target_3d'] = torch.cat([centers_2d, depths.unsqueeze(1), lhw, alphas.unsqueeze(1), corners], dim=1)
        if 'velocity' in results:
            velocity = torch.from_numpy(results['velocity'])
            results['target_3d'] = torch.cat([results['target_3d'], velocity], dim=1)
        if 'attr_label' in results:
            attr_label = torch.from_numpy(results['attr_label'])
            results['target_3d'] = torch.cat([results['target_3d'], attr_label.unsqueeze(1)], dim=1)
        results['gt_bboxes_3d'] = torch.cat([xyz, lhw, ry.unsqueeze(1)], dim=1)

        return results


@TRANSFORMS.register_module()
class Pack3DDetInputs(MMDET3D_Pack3DDetInputs):
    INPUTS_KEYS = ['points', 'img']
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'depths', 'centers_2d', 'target_3d'
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_bboxes',
        'gt_bboxes_labels',
    ]

    SEG_KEYS = [
        'gt_seg_map', 'pts_instance_mask', 'pts_semantic_mask',
        'gt_semantic_seg'
    ]


@TRANSFORMS.register_module()
class Resize3D(UnifiedIntrinsics):
    def __init__(self, size=(384, 1280), random_shift=(0, 0), cycle=False, pad_val=(114, 114, 114), cylinder=False):
        self.size = size
        self.random_shift = random_shift
        self.cycle = cycle
        self.pad_val = pad_val
        self.cylinder = cylinder

    def _init_intrinsics(self, results):
        h, w = results['img'].shape[:2]
        h_new, w_new = self.size
        w_scale, h_scale = w_new / w, h_new / h
        self.intrinsics = results['cam2img'].copy()
        self.intrinsics[0, :] *= w_scale
        self.intrinsics[1, :] *= h_scale
        if self.cylinder:
            X_min = -self.intrinsics[0, 2] / self.intrinsics[0, 0]
            X_max = (w_new - 1 - self.intrinsics[0, 2]) / self.intrinsics[0, 0]
            X_abs_max = max(-X_min, X_max)
            theta_scale = np.arctan2(X_abs_max, 1) / X_abs_max
            self.intrinsics[0, 0] *= theta_scale
            theta_min = np.arctan2(X_min, 1)
            theta_max = np.arctan2(X_max, 1)
            y_scale = (np.tan(theta_max) * np.abs(np.cos(theta_max)) - np.tan(theta_min) * np.abs(
                np.cos(theta_min))) / (theta_max - theta_min)
            self.intrinsics[1, 1] *= y_scale
            # print(theta_scale, y_scale)
        self.intrinsics_origin = self.intrinsics.copy()
        if self.random_shift[0] != 0 or self.random_shift[1] != 0:
            x_shift = random.uniform(-self.random_shift[0], self.random_shift[0])
            y_shift = random.uniform(-self.random_shift[1], self.random_shift[1])
            self.intrinsics[0, 2] += x_shift
            self.intrinsics[1, 2] += y_shift
