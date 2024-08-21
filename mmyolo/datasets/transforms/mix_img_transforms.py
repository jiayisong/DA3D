# Copyright (c) OpenMMLab. All rights reserved.
import collections
import copy
import math
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Tuple, Union
import time
import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet.structures.bbox import autocast_box_type
from mmengine.dataset import BaseDataset
from mmengine.dataset.base_dataset import Compose
from numpy import random
from .transforms import RandomCrop
from mmyolo.registry import TRANSFORMS


class BaseMixImageTransform(BaseTransform, metaclass=ABCMeta):
    """A Base Transform of multiple images mixed.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup.

    Cached mosaic transform will random select images from the cache
    and combine them into one output image if use_cached is True.

    Args:
        pre_transform(Sequence[str]): Sequence of transform object or
            config dict to be composed. Defaults to None.
        prob(float): The transformation probability. Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Defaults to 15.
    """

    def __init__(self,
                 pre_transform: Optional[Sequence[str]] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 40,
                 random_pop: bool = True,
                 max_refetch: int = 15):

        self.max_refetch = max_refetch
        self.prob = prob

        self.use_cached = use_cached
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.results_cache = []

        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    @abstractmethod
    def get_indexes(self, dataset: Union[BaseDataset,
                                         list]) -> Union[list, int]:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            list or int: indexes.
        """
        pass

    @abstractmethod
    def mix_img_transform(self, results: dict) -> dict:
        """Mixed image data transformation.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        pass

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Data augmentation function.

        The transform steps are as follows:
        1. Randomly generate index list of other images.
        2. Before Mosaic or MixUp need to go through the necessary
            pre_transform, such as MixUp' pre_transform pipeline
            include: 'LoadImageFromFile','LoadAnnotations',
            'Mosaic' and 'RandomAffine'.
        3. Use mix_img_transform function to implement specific
            mix operations.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """

        if random.uniform(0, 1) > self.prob:
            return results

        if self.use_cached:
            # Be careful: deep copying can be very time-consuming
            # if results includes dataset.
            dataset = results.pop('dataset', None)
            self.results_cache.append(copy.deepcopy(results))
            if len(self.results_cache) > self.max_cached_images:
                if self.random_pop:
                    index = random.randint(0, len(self.results_cache) - 1)
                else:
                    index = 0
                self.results_cache.pop(index)

            if len(self.results_cache) <= 4:
                return results
        else:
            assert 'dataset' in results
            # Be careful: deep copying can be very time-consuming
            # if results includes dataset.
            dataset = results.pop('dataset', None)

        for _ in range(self.max_refetch):
            # get index of one or three other images
            if self.use_cached:
                indexes = self.get_indexes(self.results_cache)
            else:
                indexes = self.get_indexes(dataset)

            if not isinstance(indexes, collections.abc.Sequence):
                indexes = [indexes]

            if self.use_cached:
                mix_results = [
                    copy.deepcopy(self.results_cache[i]) for i in indexes
                ]
            else:
                # get images information will be used for Mosaic or MixUp
                mix_results = [
                    copy.deepcopy(dataset.get_data_info(index))
                    for index in indexes
                ]

            if self.pre_transform is not None:
                for i, data in enumerate(mix_results):
                    # pre_transform may also require dataset
                    data.update({'dataset': dataset})
                    # before Mosaic or MixUp need to go through
                    # the necessary pre_transform
                    _results = self.pre_transform(data)
                    _results.pop('dataset')
                    mix_results[i] = _results

            if None not in mix_results:
                results['mix_results'] = mix_results
                break
            print('Repeated calculation')
        else:
            raise RuntimeError(
                'The loading pipeline of the original dataset'
                ' always return None. Please check the correctness '
                'of the dataset and its pipeline.')

        # Mosaic or MixUp
        results = self.mix_img_transform(results)

        if 'mix_results' in results:
            results.pop('mix_results')
        results['dataset'] = dataset

        return results


@TRANSFORMS.register_module()
class Mosaic(BaseMixImageTransform):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |           |
                |      +-----------+    pad    |
                |      |           |           |
                |      |  image1   +-----------+
                |      |           |           |
                |      |           |   image2  |
     center_y   |----+-+-----------+-----------+
                |    |   cropped   |           |
                |pad |   image3    |   image4  |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        pre_transform(Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Defaults to 15.
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 bbox_clip_border: bool = True,
                 pad_val: float = 114.0,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 40,
                 random_pop: bool = True,
                 max_refetch: int = 15):
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'
        if use_cached:
            assert max_cached_images >= 4, 'The length of cache must >= 4, ' \
                                           f'but got {max_cached_images}.'

        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch)

        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val

    def get_indexes(self, dataset: Union[BaseDataset, list]) -> list:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            list: indexes.
        """
        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
        return indexes

    def mix_img_transform(self, results: dict) -> dict:
        """Mixed image data transformation.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        assert 'mix_results' in results
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        mosaic_masks = []
        with_mask = True if 'gt_masks' in results else False
        # self.img_scale is wh format
        img_scale_w, img_scale_h = self.img_scale

        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(img_scale_h * 2), int(img_scale_w * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full((int(img_scale_h * 2), int(img_scale_w * 2)),
                                 self.pad_val,
                                 dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * img_scale_w)
        center_y = int(random.uniform(*self.center_ratio_range) * img_scale_h)
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = results
            else:
                results_patch = results['mix_results'][i - 1]

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(img_scale_h / h_i, img_scale_w / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)
            if with_mask and results_patch.get('gt_masks', None) is not None:
                gt_masks_i = results_patch['gt_masks']
                gt_masks_i = gt_masks_i.rescale(float(scale_ratio_i))
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2),
                               int(self.img_scale[1] * 2)),
                    offset=padw,
                    direction='horizontal')
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2),
                               int(self.img_scale[1] * 2)),
                    offset=padh,
                    direction='vertical')
                mosaic_masks.append(gt_masks_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * img_scale_h, 2 * img_scale_w])
            if with_mask:
                mosaic_masks = mosaic_masks[0].cat(mosaic_masks)
                results['gt_masks'] = mosaic_masks
        else:
            # remove outside bboxes
            inside_inds = mosaic_bboxes.is_inside(
                [2 * img_scale_h, 2 * img_scale_w]).numpy()
            mosaic_bboxes = mosaic_bboxes[inside_inds]
            mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
            mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]
            if with_mask:
                mosaic_masks = mosaic_masks[0].cat(mosaic_masks)[inside_inds]
                results['gt_masks'] = mosaic_masks

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        return results

    def _mosaic_combine(
            self, loc: str, center_position_xy: Sequence[float],
            img_shape_wh: Sequence[int]) -> Tuple[Tuple[int], Tuple[int]]:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                    y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class Mosaic9(BaseMixImageTransform):
    """Mosaic9 augmentation.

    Given 9 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                +-------------------------------+------------+
                | pad           |      pad      |            |
                |    +----------+               |            |
                |    |          +---------------+  top_right |
                |    |          |      top      |   image2   |
                |    | top_left |     image1    |            |
                |    |  image8  o--------+------+--------+---+
                |    |          |        |               |   |
                +----+----------+        |     right     |pad|
                |               | center |     image3    |   |
                |     left      | image0 +---------------+---|
                |    image7     |        |               |   |
            +---+-----------+---+--------+               |   |
            |   |  cropped  |            |  bottom_right |pad|
            |   |bottom_left|            |    image4     |   |
            |   |  image6   |   bottom   |               |   |
            +---|-----------+   image5   +---------------+---|
                |    pad    |            |        pad        |
                +-----------+------------+-------------------+

     The mosaic transform steps are as follows:

         1. Get the center image according to the index, and randomly
            sample another 8 images from the custom dataset.
         2. Randomly offset the image after Mosaic

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        pre_transform(Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 5 caches for each image suffices for
            randomness. Defaults to 50.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Defaults to 15.
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 bbox_clip_border: bool = True,
                 pad_val: Union[float, int] = 114.0,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 50,
                 random_pop: bool = True,
                 max_refetch: int = 15):
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'
        if use_cached:
            assert max_cached_images >= 9, 'The length of cache must >= 9, ' \
                                           f'but got {max_cached_images}.'

        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch)

        self.img_scale = img_scale
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val

        # intermediate variables
        self._current_img_shape = [0, 0]
        self._center_img_shape = [0, 0]
        self._previous_img_shape = [0, 0]

    def get_indexes(self, dataset: Union[BaseDataset, list]) -> list:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            list: indexes.
        """
        indexes = [random.randint(0, len(dataset)) for _ in range(8)]
        return indexes

    def mix_img_transform(self, results: dict) -> dict:
        """Mixed image data transformation.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        assert 'mix_results' in results

        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []

        img_scale_w, img_scale_h = self.img_scale

        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(img_scale_h * 3), int(img_scale_w * 3), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full((int(img_scale_h * 3), int(img_scale_w * 3)),
                                 self.pad_val,
                                 dtype=results['img'].dtype)

        # index = 0 is mean original image
        # len(results['mix_results']) = 8
        loc_strs = ('center', 'top', 'top_right', 'right', 'bottom_right',
                    'bottom', 'bottom_left', 'left', 'top_left')

        results_all = [results, *results['mix_results']]
        for index, results_patch in enumerate(results_all):
            img_i = results_patch['img']
            # keep_ratio resize
            img_i_h, img_i_w = img_i.shape[:2]
            scale_ratio_i = min(img_scale_h / img_i_h, img_scale_w / img_i_w)
            img_i = mmcv.imresize(
                img_i,
                (int(img_i_w * scale_ratio_i), int(img_i_h * scale_ratio_i)))

            paste_coord = self._mosaic_combine(loc_strs[index],
                                               img_i.shape[:2])

            padw, padh = paste_coord[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in paste_coord)
            mosaic_img[y1:y2, x1:x2] = img_i[y1 - padh:, x1 - padw:]

            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)

        # Offset
        offset_x = int(random.uniform(0, img_scale_w))
        offset_y = int(random.uniform(0, img_scale_h))
        mosaic_img = mosaic_img[offset_y:offset_y + 2 * img_scale_h,
                     offset_x:offset_x + 2 * img_scale_w]

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes.translate_([-offset_x, -offset_y])
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * img_scale_h, 2 * img_scale_w])
        else:
            # remove outside bboxes
            inside_inds = mosaic_bboxes.is_inside(
                [2 * img_scale_h, 2 * img_scale_w]).numpy()
            mosaic_bboxes = mosaic_bboxes[inside_inds]
            mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
            mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags
        return results

    def _mosaic_combine(self, loc: str,
                        img_shape_hw: Tuple[int, int]) -> Tuple[int, ...]:
        """Calculate global coordinate of mosaic image.

        Args:
            loc (str): Index for the sub-image.
            img_shape_hw (Sequence[int]): Height and width of sub-image

        Returns:
             paste_coord (tuple): paste corner coordinate in mosaic image.
        """
        assert loc in ('center', 'top', 'top_right', 'right', 'bottom_right',
                       'bottom', 'bottom_left', 'left', 'top_left')

        img_scale_w, img_scale_h = self.img_scale

        self._current_img_shape = img_shape_hw
        current_img_h, current_img_w = self._current_img_shape
        previous_img_h, previous_img_w = self._previous_img_shape
        center_img_h, center_img_w = self._center_img_shape

        if loc == 'center':
            self._center_img_shape = self._current_img_shape
            #  xmin, ymin, xmax, ymax
            paste_coord = img_scale_w, \
                          img_scale_h, \
                          img_scale_w + current_img_w, \
                          img_scale_h + current_img_h
        elif loc == 'top':
            paste_coord = img_scale_w, \
                          img_scale_h - current_img_h, \
                          img_scale_w + current_img_w, \
                          img_scale_h
        elif loc == 'top_right':
            paste_coord = img_scale_w + previous_img_w, \
                          img_scale_h - current_img_h, \
                          img_scale_w + previous_img_w + current_img_w, \
                          img_scale_h
        elif loc == 'right':
            paste_coord = img_scale_w + center_img_w, \
                          img_scale_h, \
                          img_scale_w + center_img_w + current_img_w, \
                          img_scale_h + current_img_h
        elif loc == 'bottom_right':
            paste_coord = img_scale_w + center_img_w, \
                          img_scale_h + previous_img_h, \
                          img_scale_w + center_img_w + current_img_w, \
                          img_scale_h + previous_img_h + current_img_h
        elif loc == 'bottom':
            paste_coord = img_scale_w + center_img_w - current_img_w, \
                          img_scale_h + center_img_h, \
                          img_scale_w + center_img_w, \
                          img_scale_h + center_img_h + current_img_h
        elif loc == 'bottom_left':
            paste_coord = img_scale_w + center_img_w - \
                          previous_img_w - current_img_w, \
                          img_scale_h + center_img_h, \
                          img_scale_w + center_img_w - previous_img_w, \
                          img_scale_h + center_img_h + current_img_h
        elif loc == 'left':
            paste_coord = img_scale_w - current_img_w, \
                          img_scale_h + center_img_h - current_img_h, \
                          img_scale_w, \
                          img_scale_h + center_img_h
        elif loc == 'top_left':
            paste_coord = img_scale_w - current_img_w, \
                          img_scale_h + center_img_h - \
                          previous_img_h - current_img_h, \
                          img_scale_w, \
                          img_scale_h + center_img_h - previous_img_h

        self._previous_img_shape = self._current_img_shape
        #  xmin, ymin, xmax, ymax
        return paste_coord

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class YOLOv5MixUp(BaseMixImageTransform):
    """MixUp data augmentation for YOLOv5.

    .. code:: text

    The mixup transform steps are as follows:

        1. Another random image is picked by dataset.
        2. Randomly obtain the fusion ratio from the beta distribution,
            then fuse the target
        of the original image and mixup image through this ratio.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)


    Args:
        alpha (float): parameter of beta distribution to get mixup ratio.
            Defaults to 32.
        beta (float):  parameter of beta distribution to get mixup ratio.
            Defaults to 32.
        pre_transform (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of iterations. If the number of
            iterations is greater than `max_refetch`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
    """

    def __init__(self,
                 alpha: float = 32.0,
                 beta: float = 32.0,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 20,
                 random_pop: bool = True,
                 max_refetch: int = 15):
        if use_cached:
            assert max_cached_images >= 2, 'The length of cache must >= 2, ' \
                                           f'but got {max_cached_images}.'
        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch)
        self.alpha = alpha
        self.beta = beta

    def get_indexes(self, dataset: Union[BaseDataset, list]) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            int: indexes.
        """
        return random.randint(0, len(dataset))

    def mix_img_transform(self, results: dict) -> dict:
        """YOLOv5 MixUp transform function.

        Args:
            results (dict): Result dict

        Returns:
            results (dict): Updated result dict.
        """
        assert 'mix_results' in results

        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']
        ori_img = results['img']
        assert ori_img.shape == retrieve_img.shape

        # Randomly obtain the fusion ratio from the beta distribution,
        # which is around 0.5
        ratio = np.random.beta(self.alpha, self.beta)
        mixup_img = (ori_img * ratio + retrieve_img * (1 - ratio))

        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes_labels = retrieve_results['gt_bboxes_labels']
        retrieve_gt_ignore_flags = retrieve_results['gt_ignore_flags']

        mixup_gt_bboxes = retrieve_gt_bboxes.cat(
            (results['gt_bboxes'], retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = np.concatenate(
            (results['gt_bboxes_labels'], retrieve_gt_bboxes_labels), axis=0)
        mixup_gt_ignore_flags = np.concatenate(
            (results['gt_ignore_flags'], retrieve_gt_ignore_flags), axis=0)
        if 'gt_masks' in results:
            assert 'gt_masks' in retrieve_results
            mixup_gt_masks = results['gt_masks'].cat(
                [results['gt_masks'], retrieve_results['gt_masks']])
            results['gt_masks'] = mixup_gt_masks

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_bboxes_labels'] = mixup_gt_bboxes_labels
        results['gt_ignore_flags'] = mixup_gt_ignore_flags

        return results


@TRANSFORMS.register_module()
class YOLOXMixUp(BaseMixImageTransform):
    """MixUp data augmentation for YOLOX.

    .. code:: text

                         mixup transform
                +---------------+--------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                +---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      +-----------------+     |
                |             pad              |
                +------------------------------+

    The mixup transform steps are as follows:

        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)


    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (width, height). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (int): Pad value. Defaults to 114.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pre_transform(Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of iterations. If the number of
            iterations is greater than `max_refetch`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 ratio_range: Tuple[float, float] = (0.5, 1.5),
                 flip_ratio: float = 0.5,
                 pad_val: float = 114.0,
                 bbox_clip_border: bool = True,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 20,
                 random_pop: bool = True,
                 max_refetch: int = 15):
        assert isinstance(img_scale, tuple)
        if use_cached:
            assert max_cached_images >= 2, 'The length of cache must >= 2, ' \
                                           f'but got {max_cached_images}.'
        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch)
        self.img_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.bbox_clip_border = bbox_clip_border

    def get_indexes(self, dataset: Union[BaseDataset, list]) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            int: indexes.
        """
        return random.randint(0, len(dataset))

    def mix_img_transform(self, results: dict) -> dict:
        """YOLOX MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        assert 'mix_results' in results
        assert len(
            results['mix_results']) == 1, 'MixUp only support 2 images now !'

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            # empty bbox
            return results

        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']

        jit_factor = random.uniform(*self.ratio_range)
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones((self.img_scale[1], self.img_scale[0], 3),
                              dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.img_scale[::-1], dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.img_scale[1] / retrieve_img.shape[0],
                          self.img_scale[0] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.ones((max(origin_h, target_h), max(
            origin_w, target_w), 3)) * self.pad_val
        padded_img = padded_img.astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                             x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes.rescale_([scale_ratio, scale_ratio])
        if self.bbox_clip_border:
            retrieve_gt_bboxes.clip_([origin_h, origin_w])

        if is_filp:
            retrieve_gt_bboxes.flip_([origin_h, origin_w],
                                     direction='horizontal')

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.clone()
        cp_retrieve_gt_bboxes.translate_([-x_offset, -y_offset])
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes.clip_([target_h, target_w])

        # 8. mix up
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img

        retrieve_gt_bboxes_labels = retrieve_results['gt_bboxes_labels']
        retrieve_gt_ignore_flags = retrieve_results['gt_ignore_flags']

        mixup_gt_bboxes = cp_retrieve_gt_bboxes.cat(
            (results['gt_bboxes'], cp_retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = np.concatenate(
            (results['gt_bboxes_labels'], retrieve_gt_bboxes_labels), axis=0)
        mixup_gt_ignore_flags = np.concatenate(
            (results['gt_ignore_flags'], retrieve_gt_ignore_flags), axis=0)

        if not self.bbox_clip_border:
            # remove outside bbox
            inside_inds = mixup_gt_bboxes.is_inside([target_h,
                                                     target_w]).numpy()
            mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
            mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]
            mixup_gt_ignore_flags = mixup_gt_ignore_flags[inside_inds]

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_bboxes_labels'] = mixup_gt_bboxes_labels
        results['gt_ignore_flags'] = mixup_gt_ignore_flags

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'max_refetch={self.max_refetch}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@TRANSFORMS.register_module()
class MosaicMixUp3D(BaseTransform):

    def __init__(self,
                 mosaic_num=(2, 2),
                 mixup_num=1,
                 alpha: float = 32.0,
                 beta: float = 32.0,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 40,
                 cylinder=False,
                 max_refetch: int = 15):
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'
        self.mix_num = mosaic_num[0] * mosaic_num[1] * mixup_num - 1
        if use_cached:
            assert max_cached_images >= self.mix_num, 'The length of cache must >= mosaic_num[0] * mosaic_num[1]-1, ' \
                                                      f'but got {max_cached_images}.'

        self.max_refetch = max_refetch
        self.prob = prob
        self.alpha = alpha
        self.beta = beta
        self.use_cached = use_cached
        self.max_cached_images = max_cached_images
        self.results_cache = []

        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)
        self.mosaic_num = mosaic_num
        self.mixup_num = mixup_num
        self.cylinder = cylinder
        self.random_crop = RandomCrop((0, 0), random_shift=(0, 0), cycle=True, pad_val=(114, 114, 114),
                                      cylinder=cylinder)
        self.results_cache_timestamp = np.array([], dtype=np.float32)
        self.start_step = 0

    def transform(self, results: dict) -> dict:
        """Data augmentation function.

        The transform steps are as follows:
        1. Randomly generate index list of other images.
        2. Before Mosaic or MixUp need to go through the necessary
            pre_transform, such as MixUp' pre_transform pipeline
            include: 'LoadImageFromFile','LoadAnnotations',
            'Mosaic' and 'RandomAffine'.
        3. Use mix_img_transform function to implement specific
            mix operations.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        dataset = results.pop('dataset', None)
        mix_results = None
        indexes = None
        for _ in range(self.max_refetch):
            # get index of one or three other images
            if self.use_cached:
                if self.start_step < self.max_cached_images:
                    self.start_step += 1
                    break
                if len(self.results_cache) < self.mix_num:
                    break
                if random.uniform(0, 1) > self.prob:
                    break
                indexes = self.get_indexes(self.results_cache, results)
            else:
                if random.uniform(0, 1) > self.prob:
                    break
                indexes = self.get_indexes(dataset, results)
            if not isinstance(indexes, collections.abc.Sequence):
                indexes = [indexes]

            if self.use_cached:
                mix_results = [
                    copy.deepcopy(self.results_cache[i]) for i in indexes
                ]
            else:
                # get images information will be used for Mosaic or MixUp
                mix_results = [
                    copy.deepcopy(dataset.get_data_info(index))
                    for index in indexes
                ]

            if self.pre_transform is not None:
                for i, data in enumerate(mix_results):
                    # pre_transform may also require dataset
                    data.update({'dataset': dataset})
                    # before Mosaic or MixUp need to go through
                    # the necessary pre_transform
                    _results = self.pre_transform(data)
                    _results.pop('dataset')
                    mix_results[i] = _results

            if None not in mix_results:
                break
            print('Repeated calculation')
        else:
            raise RuntimeError(
                'The loading pipeline of the original dataset'
                ' always return None. Please check the correctness '
                'of the dataset and its pipeline.')
        self.update_cache(results, indexes)
        if mix_results is not None:
            results['mix_results'] = mix_results
            results = self.mix_img_transform(results)
            if 'mix_results' in results:
                results.pop('mix_results')
            # self.show_result(results)
        results['dataset'] = dataset
        return results

    def update_cache(self, results, indexes):
        if self.use_cached:
            if indexes is not None:
                self.results_cache_timestamp[indexes] = self.results_cache_timestamp[indexes] - (1 / self.prob)
                results_cache = []
                for i in range(len(self.results_cache)):
                    if self.results_cache_timestamp[i] > 1e-4:
                        results_cache.append(self.results_cache[i])
                self.results_cache = results_cache
                self.results_cache_timestamp = self.results_cache_timestamp[self.results_cache_timestamp > 1e-4]
            results = self.save_cache(results)
            if isinstance(results, list):
                self.results_cache.extend(self.save_cache(results))
                self.results_cache_timestamp = np.append(self.results_cache_timestamp, self.mix_num)
            else:
                self.results_cache.append(self.save_cache(results))
                self.results_cache_timestamp = np.concatenate(self.results_cache_timestamp,
                                                              np.ones([len(results), ]) * self.mix_num)

    def save_cache(self, results):
        return copy.deepcopy(results)

    def get_indexes(self, dataset: Union[BaseDataset, list], results) -> list:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            list: indexes.
        """
        indexes = np.random.choice(len(dataset), self.mix_num, replace=False).tolist()
        return indexes

    def mixup_img_transform(self, results_list) -> dict:
        """YOLOv5 MixUp transform function.

        Args:
            results (dict): Result dict

        Returns:
            results (dict): Updated result dict.
        """
        results = results_list[0]
        ori_img = results['img']
        # Randomly obtain the fusion ratio from the beta distribution,
        # which is around 0.5
        prob = 1
        mixup_img = 0
        mosaic_result = []
        delta_uv = np.array([[0, 0, 0, 0], ], dtype=np.float32)
        for i in range(self.mixup_num):
            # start = time.time()
            if i == 0:
                results_patch = results
            else:
                results_patch = results_list[i]
                self.random_crop.size = ori_img.shape[:2]
                self.random_crop.random_shift = [ori_img.shape[1], 0]
                results_patch = self.random_crop(results_patch)
                results_patch = self._move_object_xy(results_patch, delta_uv, results['cam2img'], results['K_out'])
            img_i = results_patch['img']
            assert ori_img.shape == img_i.shape
            if (self.mixup_num - i) == 1:
                ratio = prob
            else:
                # ratio = np.random.beta(self.alpha, self.beta) * prob * 2 / (self.mixup_num - i)
                ratio = 0.5 * prob * 2 / (self.mixup_num - i)
                prob = prob - ratio
            mixup_img = mixup_img + img_i * ratio
            mosaic_result.append(results_patch)
            # end = time.time()
            # print(i, end - start)  # 1.0257349014282227 (秒)
        results['img'] = mixup_img.astype(np.uint8)
        results['gt_bboxes_3d'].tensor = torch.cat([i['gt_bboxes_3d'].tensor for i in mosaic_result], 0)
        for key in ['gt_bboxes_labels', 'depths', 'gt_labels_3d', 'gt_bboxes_area', 'centers_2d', 'gt_bboxes']:
            if key in results:
                results[key] = np.concatenate([i[key] for i in mosaic_result], 0)
        return results

    def mix_img_transform(self, results: dict) -> dict:
        """Mixed image data transformation.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        assert 'mix_results' in results

        img_scale_h, img_scale_w = results['img_shape']
        delta_u = np.array([[img_scale_w, 0, img_scale_w, 0], ], dtype=np.float32)
        delta_v = np.array([[0, img_scale_h, 0, img_scale_h], ], dtype=np.float32)

        mosaic_result = []
        for i in range(self.mosaic_num[0]):
            mixup_result = []
            for k in range(self.mixup_num):
                row_result = []
                for j in range(self.mosaic_num[1]):
                    if (i + j + k) == 0:
                        results_patch = results
                    else:
                        results_patch = results['mix_results'][
                            i * self.mosaic_num[1] * self.mixup_num + k * self.mosaic_num[1] + j - 1]
                    if j == 0:
                        row_dict = results_patch
                    else:
                        delta_uv = delta_u * j  # + delta_v * i
                        results_patch = self._move_object_xy(results_patch, delta_uv, row_dict['cam2img'],
                                                             row_dict['K_out'])
                    img_i = results_patch['img']
                    h_i, w_i = img_i.shape[:2]
                    assert img_scale_h == h_i and img_scale_w == w_i
                    row_result.append(results_patch)
                row_img = np.concatenate([i['img'] for i in row_result], axis=1)
                row_dict['img'] = row_img
                row_dict['img_shape'] = row_img.shape[:2]
                row_dict['gt_bboxes_3d'].tensor = torch.cat([i['gt_bboxes_3d'].tensor for i in row_result], 0)
                for key in ['gt_bboxes_labels', 'depths', 'gt_labels_3d', 'gt_bboxes_area', 'centers_2d', 'gt_bboxes']:
                    if key in results:
                        row_dict[key] = np.concatenate([i[key] for i in row_result], 0)
                if i > 0:
                    self.random_crop.size = [img_scale_h, img_scale_w * self.mosaic_num[1]]
                    self.random_crop.random_shift = [img_scale_w * self.mosaic_num[1], 0]
                    row_dict = self.random_crop(row_dict)
                mixup_result.append(row_dict)
            mixup_result = self.mixup_img_transform(mixup_result)

            if i == 0:
                mixup_result_0 = mixup_result
            else:
                delta_uv = delta_v * i
                mixup_result = self._move_object_xy(mixup_result, delta_uv, mixup_result_0['cam2img'],
                                                    mixup_result_0['K_out'])
            mosaic_result.append(mixup_result)
        mosaic_img = np.concatenate([i['img'] for i in mosaic_result], axis=0)
        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape[:2]
        results['gt_bboxes_3d'].tensor = torch.cat([i['gt_bboxes_3d'].tensor for i in mosaic_result], 0)
        for key in ['gt_bboxes_labels', 'depths', 'gt_labels_3d', 'gt_bboxes_area', 'centers_2d', 'gt_bboxes']:
            if key in results:
                results[key] = np.concatenate([i[key] for i in mosaic_result], 0)
        return results

    def _move_object_xy(self, results, delta_uv, cam2img, K_out):
        K_out_ori = results['K_out']
        gt_bbox_3d = results['gt_bboxes_3d'].tensor
        cam2img_ori = results['cam2img']
        assert cam2img[0, 0] == cam2img_ori[0, 0]
        assert cam2img[1, 1] == cam2img_ori[1, 1]
        depths = torch.from_numpy(results['depths'])
        centers2d = results['centers_2d']
        gt_bboxes = results['gt_bboxes']
        bbox3d1 = gt_bbox_3d.clone()
        centers2d1 = centers2d + delta_uv[:, :2]
        gt_bboxes1 = gt_bboxes + delta_uv
        bbox3d1[:, :3] = bbox3d1[:, :3] + K_out_ori
        rays = torch.atan2(bbox3d1[:, 0], bbox3d1[:, 2])
        x_delta = (delta_uv[0, 0] + cam2img_ori[0, 2] - cam2img[0, 2]) / cam2img[0, 0]
        y_delta = (delta_uv[0, 1] + cam2img_ori[1, 2] - cam2img[1, 2]) / cam2img[1, 1]
        if self.cylinder:
            rays_new = rays + x_delta
            bbox3d1[:, 0] = torch.sin(rays_new) * depths
            bbox3d1[:, 2] = torch.cos(rays_new) * depths
            bbox3d1[:, 6] = (bbox3d1[:, 6] + x_delta + math.pi) % (2 * math.pi) - math.pi
        else:
            bbox3d1[:, 0] = x_delta * depths + bbox3d1[:, 0]
            rays_new = torch.atan2(bbox3d1[:, 0], bbox3d1[:, 2])
            bbox3d1[:, 6] = (bbox3d1[:, 6] + rays_new - rays + math.pi) % (2 * math.pi) - math.pi
        bbox3d1[:, 1] = y_delta * depths + bbox3d1[:, 1]
        bbox3d1[:, :3] = bbox3d1[:, :3] - K_out
        results['gt_bboxes_3d'].tensor = bbox3d1
        results['centers_2d'] = centers2d1
        results['gt_bboxes'] = gt_bboxes1
        return results

    def show_result(self, results):
        img = results['img']
        bbox = results['gt_bboxes']
        for b in bbox:
            img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.imwrite(f'{results["sample_idx"]}_mosaic.jpg', img)
        return 0


@TRANSFORMS.register_module()
class MixUp3D(MosaicMixUp3D):
    def __init__(self,
                 mixup_num=2,
                 alpha: float = 32.0,
                 beta: float = 32.0,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 cylinder: bool = False,
                 max_cached_images: int = 20,
                 max_cached_time=None,
                 max_refetch: int = 15):
        self.mix_num = mixup_num - 1
        if use_cached:
            assert max_cached_images >= self.mix_num, 'The length of cache must >= mixup_num, ' \
                                                      f'but got {max_cached_images}.'
        self.max_refetch = max_refetch
        self.prob = prob

        self.use_cached = use_cached
        self.max_cached_images = max_cached_images
        self.max_cached_time = max_cached_images if max_cached_time is None else max_cached_time
        self.results_cache = []

        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)
        self.mixup_num = mixup_num

        self.cylinder = cylinder
        self.results_cache_timestamp = np.array([], dtype=np.int32)
        self.start_step = 0
        self.alpha = alpha
        self.beta = beta
        self.random_crop = RandomCrop((0, 0), random_shift=(0, 0), cycle=True, pad_val=(114, 114, 114),
                                      cylinder=cylinder)

    def _move_object_xy(self, results, cam2img, K_out):
        K_out_ori = results['K_out']
        gt_bbox_3d = results['gt_bboxes_3d'].tensor
        cam2img_ori = results['cam2img']
        assert cam2img[0, 0] == cam2img_ori[0, 0]
        assert cam2img[1, 1] == cam2img_ori[1, 1]
        assert cam2img[1, 2] == cam2img_ori[1, 2]
        depths = torch.from_numpy(results['depths'])
        bbox3d1 = gt_bbox_3d.clone()
        bbox3d1[:, :3] = bbox3d1[:, :3] + K_out_ori
        rays = torch.atan2(bbox3d1[:, 0], bbox3d1[:, 2])
        x_delta = (cam2img_ori[0, 2] - cam2img[0, 2]) / cam2img[0, 0]
        # y_delta = (cam2img_ori[1, 2] - cam2img[1, 2]) / cam2img[1, 1]
        if self.cylinder:
            rays_new = rays + x_delta
            bbox3d1[:, 0] = torch.sin(rays_new) * depths
            bbox3d1[:, 2] = torch.cos(rays_new) * depths
            bbox3d1[:, 6] = (bbox3d1[:, 6] + x_delta + math.pi) % (2 * math.pi) - math.pi
        else:
            bbox3d1[:, 0] = x_delta * depths + bbox3d1[:, 0]
            rays_new = torch.atan2(bbox3d1[:, 0], bbox3d1[:, 2])
            bbox3d1[:, 6] = (bbox3d1[:, 6] + rays_new - rays + math.pi) % (2 * math.pi) - math.pi
        # bbox3d1[:, 1] = y_delta * depths + bbox3d1[:, 1]
        bbox3d1[:, :3] = bbox3d1[:, :3] - K_out
        results['gt_bboxes_3d'].tensor = bbox3d1
        return results

    def mix_img_transform(self, results) -> dict:
        """YOLOv5 MixUp transform function.

        Args:
            results (dict): Result dict

        Returns:
            results (dict): Updated result dict.
        """
        ori_img = results['img']
        # Randomly obtain the fusion ratio from the beta distribution,
        # which is around 0.5
        prob = 1
        mixup_img = 0
        mosaic_result = []
        for i in range(self.mixup_num):
            # start = time.time()
            if i == 0:
                results_patch = results
            else:
                results_patch = results['mix_results'][i - 1]
                self.random_crop.size = ori_img.shape[:2]
                self.random_crop.random_shift = [results_patch['img'].shape[1], 0]
                results_patch = self.random_crop(results_patch)
                results_patch = self._move_object_xy(results_patch, results['cam2img'], results['K_out'])
            img_i = results_patch['img']
            assert ori_img.shape == img_i.shape
            if (self.mixup_num - i) == 1:
                ratio = prob
            else:
                # ratio = np.random.beta(self.alpha, self.beta) * prob * 2 / (self.mixup_num - i)
                ratio = 0.5 * prob * 2 / (self.mixup_num - i)
                prob = prob - ratio
            mixup_img = mixup_img + img_i * ratio
            mosaic_result.append(results_patch)
            # end = time.time()
            # print(i, end - start)  # 1.0257349014282227 (秒)
        results['img'] = mixup_img.astype(np.uint8)
        results['gt_bboxes_3d'].tensor = torch.cat([i['gt_bboxes_3d'].tensor for i in mosaic_result], 0)
        for key in ['gt_bboxes_labels', 'depths', 'gt_labels_3d', 'gt_bboxes_area', 'centers_2d', 'gt_bboxes']:
            if key in results:
                results[key] = np.concatenate([i[key] for i in mosaic_result], 0)
        return results

    def show_result(self, results):
        img = results['img']
        bbox = results['gt_bboxes']
        for b in bbox:
            img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.imwrite(f'{results["sample_idx"]}_mixup.jpg', img)
        return 0


@TRANSFORMS.register_module()
class CutMix3D(BaseTransform):
    def __init__(self,
                 mix_num=2, max_cache_num=10,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 cylinder: bool = False):
        self.mix_num = mix_num - 1
        # self.gt_num = gt_num
        self.max_cache_num = max_cache_num
        self.prob = prob
        self.results_cache = []
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)
        self.cylinder = cylinder
        self.results_cache_timestamp = np.array([], dtype=np.float32)
        self.step = 0

    def _move_object_xy(self, results, delta_uv, cam2img, K_out):
        results_new = {}
        results_new['gt_bboxes'] = results['gt_bboxes'] + delta_uv
        results_new['centers_2d'] = results['centers_2d'] + delta_uv[:, :2]
        results_new['gt_bboxes_3d'] = copy.deepcopy(results['gt_bboxes_3d'])
        gt_bbox_3d = results_new['gt_bboxes_3d'].tensor
        depths = torch.from_numpy(results['depths'])
        bbox3d1 = gt_bbox_3d
        rays = torch.atan2(bbox3d1[:, 0], bbox3d1[:, 2])
        x_delta = (delta_uv[0, 0]) / cam2img[0, 0]
        if self.cylinder:
            rays_new = rays + x_delta
            bbox3d1[:, 0] = torch.sin(rays_new) * depths
            bbox3d1[:, 2] = torch.cos(rays_new) * depths
            bbox3d1[:, 6] = (bbox3d1[:, 6] + x_delta + math.pi) % (2 * math.pi) - math.pi
        else:
            bbox3d1[:, 0] = x_delta * depths + bbox3d1[:, 0]
            rays_new = torch.atan2(bbox3d1[:, 0], bbox3d1[:, 2])
            bbox3d1[:, 6] = (bbox3d1[:, 6] + rays_new - rays + math.pi) % (2 * math.pi) - math.pi
        bbox3d1[:, :3] = bbox3d1[:, :3] - K_out
        results_new['gt_bboxes_3d'].tensor = bbox3d1
        return results_new

    def get_indexes(self, results) -> list:
        if self.results_cache_timestamp.shape[0] > 0:
            p = 1 / (1 + self.max_cache_num - self.results_cache_timestamp).clip(min=1)
            indexes = np.where(np.random.uniform(size=p.shape) < p)[0].tolist()
            if indexes == []:
                indexes = None
        else:
            indexes = None
        return indexes

    # def get_indexes(self, results) -> list:
    #     gt_num = results['gt_bboxes'].shape[0]
    #     if gt_num >= self.gt_num or len(self.results_cache) < (self.gt_num - gt_num):
    #         return None
    #     else:
    #         indexes = np.random.choice(len(self.results_cache), self.gt_num - gt_num, replace=False,
    #                                    p=self.results_cache_timestamp / self.results_cache_timestamp.sum()).tolist()
    #         return indexes

    # def get_indexes(self, results) -> list:
    #     if len(self.results_cache) == 0:
    #         return None
    #     elif results['gt_bboxes'].shape[0] >= self.gt_num:
    #         return None
    #     else:
    #         select_num = min((self.gt_num - results['gt_bboxes'].shape[0]) * 10, len(self.results_cache))
    #         indexes = np.random.choice(len(self.results_cache), select_num, replace=False,
    #                                    p=self.results_cache_timestamp / self.results_cache_timestamp.sum()).tolist()
    #         return indexes

    def transform(self, results: dict) -> dict:
        dataset = results.pop('dataset', None)
        if random.uniform(0, 1) > self.prob:
            results_cache = self.add_cache(results)
            results['dataset'] = dataset
            return results
        indexes = self.get_indexes(results)
        if indexes is None:
            results_cache = self.add_cache(results)
            results['dataset'] = dataset
            return results
        if not isinstance(indexes, collections.abc.Sequence):
            indexes = [indexes]
        mix_results = [self.results_cache[i] for i in indexes]
        results_cache = self.add_cache(results)
        results['mix_results'] = mix_results
        results['results_cache'] = results_cache
        results, index_valid = self.mix_img_transform(results)
        indexes_new = []
        for i, v in zip(indexes, index_valid):
            if v:
                indexes_new.append(i)
        self.remove_cache(indexes_new)
        # print(len(indexes_new) / len(indexes))
        if 'mix_results' in results:
            results.pop('mix_results')
        if 'results_cache' in results:
            results.pop('results_cache')
        # self.show_result(results)
        results['dataset'] = dataset
        return results

    def add_cache(self, results):
        results = self.save_cache(results)
        if isinstance(results, list):
            self.results_cache.extend(results * self.mix_num)
            self.results_cache_timestamp = self.results_cache_timestamp + 1
            self.results_cache_timestamp = np.concatenate([self.results_cache_timestamp,
                                                           np.ones([len(results) * self.mix_num, ], dtype=np.float32)],
                                                          axis=0)
        else:
            self.results_cache.append(results)
            self.results_cache_timestamp = np.append(self.results_cache_timestamp, self.mix_num)
        # assert len(self.results_cache) == len(self.results_cache_timestamp)
        return results

    def remove_cache(self, indexes):
        if indexes is not None:
            indexes = sorted(indexes, reverse=True)
            for i in indexes:
                self.results_cache.pop(i)
            self.results_cache_timestamp = np.delete(self.results_cache_timestamp, indexes)
            # assert len(self.results_cache) == len(self.results_cache_timestamp)

    def mix_img_transform(self, results) -> dict:
        index_valid = np.zeros([len(results['mix_results'])], np.bool)
        # gt_num = len(results['results_cache'])
        ori_img = results['img'].astype(np.float32)
        h, w, _ = ori_img.shape
        # gt_bboxes = results['gt_bboxes']
        mask = np.zeros([h, w], dtype=np.float32)
        for obj in results['results_cache']:
            gt_bboxes_int = obj['gt_bboxes_int'][0]
            obj_mask = obj['mask']
            temp = mask[gt_bboxes_int[1]:gt_bboxes_int[3], gt_bboxes_int[0]:gt_bboxes_int[2]]
            mask[gt_bboxes_int[1]:gt_bboxes_int[3], gt_bboxes_int[0]:gt_bboxes_int[2]] = np.maximum(temp, obj_mask)
        # mask_ori = mask.copy()
        obj_list = [results, ]
        for i, obj in enumerate(results['mix_results']):
            img_i = obj['img']
            gt_bboxes_obj = obj['gt_bboxes'][0]
            gt_bboxes_int = obj['gt_bboxes_int'][0]
            obj_mask = obj['mask']
            gt_bboxes_obj_int = gt_bboxes_obj.astype(np.int32)
            target_width = gt_bboxes_obj_int[2] - gt_bboxes_obj_int[0]
            # loc = np.random.randint(0, w - target_width)
            # '''
            iszero = np.sum(mask[gt_bboxes_obj_int[1]:gt_bboxes_obj_int[3], :] >= 1, axis=0, keepdims=False) == 0
            iszero = np.concatenate(([0], iszero, [0]))
            absdiff = np.abs(np.diff(iszero))
            # Runs start and end where absdiff is 1.
            ranges_zero = np.where(absdiff == 1)[0].reshape(-1, 2)
            space_length = ranges_zero[:, 1] - ranges_zero[:, 0]
            ranges = ranges_zero[space_length > target_width]
            if ranges.shape[0] > 0:
                index_valid[i] = True
                p = (ranges[:, 1] - ranges[:, 0])
                ranges_id = np.random.choice(ranges.shape[0], 1, p=p / p.sum())
                ranges = ranges[ranges_id]
                loc = np.random.randint(ranges[0][0], ranges[0][1] - target_width)
            else:
                # continue
                loc = np.random.randint(0, w - target_width)
                index_valid[i] = True
            # '''
            x_min = loc - (gt_bboxes_obj_int[0] - gt_bboxes_int[0])
            y_min = gt_bboxes_int[1]
            x_max = x_min + gt_bboxes_int[2] - gt_bboxes_int[0]
            y_max = gt_bboxes_int[3]
            if x_min < 0:
                mask_x_min = -x_min
                x_min = 0
            else:
                mask_x_min = 0
            if x_max > w:
                mask_x_max = w - x_max
                x_max = w
            else:
                mask_x_max = gt_bboxes_int[2] - gt_bboxes_int[0]
            obj_mask = obj_mask[:, mask_x_min:mask_x_max]
            img_i = img_i[:, mask_x_min:mask_x_max, :]
            # ori_img_mask = ori_img[y_min:y_max, x_min:x_max, :]
            # mask[y_min:y_max, x_min:x_max] = np.maximum(mask[y_min:y_max, x_min:x_max], obj_mask)
            mask[y_min:y_max, x_min:x_max] = mask[y_min:y_max, x_min:x_max] + obj_mask
            # mask[y_min:y_max,
            # x_min + (gt_bboxes_obj_int[0] - gt_bboxes_int[0]):x_max + (gt_bboxes_obj_int[2] - gt_bboxes_int[2])] = 1
            # obj_mask = obj_mask[:, :, None]
            # ori_img[y_min:y_max, x_min:x_max, :] = ori_img_mask * (1 - obj_mask) + img_i * obj_mask
            delta_u = x_min - mask_x_min - gt_bboxes_int[0]
            delta_u = np.array([[delta_u, 0, delta_u, 0], ], dtype=np.float32)
            obj_new = self._move_object_xy(obj, delta_u, results['cam2img'], results['K_out'])
            obj_new['mask'] = obj_mask
            obj_new['img'] = img_i
            obj_new['cood'] = [y_min, y_max, x_min, x_max]
            for key in ['gt_bboxes_labels', 'depths', 'gt_labels_3d', 'gt_bboxes_area', ]:
                if key in obj:
                    obj_new[key] = obj[key]
            obj_list.append(obj_new)
            # gt_num = gt_num + 1
            # if gt_num >= self.gt_num:
            #     break
        # mask_background = (1 - mask).clip(min=0)
        # mask = mask.clip(min=1)
        # mask_ori = mask_ori / mask + mask_background
        # ori_img = ori_img * mask_ori[:, :, None]
        for obj in obj_list[1:]:
            y_min, y_max, x_min, x_max = obj['cood']
            # mask_obj = obj['mask'] / mask[y_min:y_max, x_min:x_max]
            # ori_img[y_min:y_max, x_min:x_max, :] += obj['img'] * mask_obj[:, :, None]
            mask_obj = obj['mask'][:, :, None]
            ori_img[y_min:y_max, x_min:x_max, :] = ori_img[y_min:y_max, x_min:x_max, :] * (1 - mask_obj) \
                                                   + obj['img'] * mask_obj
        results['img'] = ori_img.astype(np.uint8)
        results['gt_bboxes_3d'].tensor = torch.cat([i['gt_bboxes_3d'].tensor for i in obj_list], 0)
        for key in ['gt_bboxes_labels', 'depths', 'gt_labels_3d', 'gt_bboxes_area', 'centers_2d', 'gt_bboxes']:
            if key in results:
                results[key] = np.concatenate([i[key] for i in obj_list], 0)
        return results, index_valid

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

    def save_cache(self, results):
        img = results['img']
        h, w, _ = img.shape
        gt_bboxes = results['gt_bboxes']
        kernel = 8
        gt_bboxes_int = gt_bboxes.astype(np.int32)
        gt_bboxes_int[:, :2] -= kernel
        gt_bboxes_int[:, 2:] += (kernel + 1)
        gt_bboxes_int[:, 0::2] = gt_bboxes_int[:, 0::2].clip(0, w - 1)
        gt_bboxes_int[:, 1::2] = gt_bboxes_int[:, 1::2].clip(0, h - 1)
        object_results = []
        kernel = kernel * 2 - 1
        for i, b in enumerate(gt_bboxes_int):
            x_min, y_min, x_max, y_max = b[0], b[1], b[2], b[3]
            bbox = (gt_bboxes[i] - np.array([x_min, y_min, x_min, y_min, ])).astype(np.int32)
            img_i = img[y_min:y_max, x_min:x_max, :]
            mask = np.zeros(img_i.shape[:2], dtype=np.float32)
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            mask = cv2.blur(mask, (kernel, kernel), borderType=cv2.BORDER_REPLICATE)
            object_result = {'img': img_i, 'gt_bboxes_int': b[None, :], 'mask': mask}
            for key in ['gt_bboxes_labels', 'depths', 'gt_labels_3d', 'gt_bboxes_area', 'centers_2d', 'gt_bboxes',
                        'gt_bboxes_3d']:
                if key in results:
                    object_result[key] = results[key][i:i + 1]
            object_result['gt_bboxes_3d'].tensor[:, :3] += results['K_out'][None, :]
            object_results.append(object_result)
        return object_results
