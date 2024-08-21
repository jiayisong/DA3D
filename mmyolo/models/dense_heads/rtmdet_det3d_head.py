# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Sequence, Tuple, Union, Optional
import torch.distributed as dist
import numpy as np
from mmcv.ops import batched_nms
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, is_norm
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.structures.bbox import distance2bbox
from mmengine.structures import InstanceData
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig, reduce_mean)
from mmengine.model import (BaseModule, bias_init_with_prob, constant_init,
                            normal_init)
from torch import Tensor
from mmengine.config import ConfigDict

from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import gt_instances_preprocess
from .yolov5_head import YOLOv5Head
from mmdet3d.models import BaseMono3DDenseHead
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from cv_ops.bbox3d import nms_3d_gpu, nms_bev_gpu, nms_dist_gpu
from mmengine.logging import MessageHub


@MODELS.register_module()
class RTMDet3DHead(BaseMono3DDenseHead):
    """Anchor-free head for monocular 3D object detection.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels.
            Used in child classes. Defaults to 256.
        stacked_convs (int): Number of stacking convs of the head.
        strides (Sequence[int] or Sequence[Tuple[int, int]]): Downsample
            factor of each feature map.
        dcn_on_last_conv (bool): If true, use dcn in the last
            layer of towers. Default: False.
        conv_bias (bool or str): If specified as `auto`, it will be
            decided by the norm_cfg. Bias of conv will be set as True
            if `norm_cfg` is None, otherwise False. Default: 'auto'.
        background_label (bool, Optional): Label ID of background,
            set as 0 for RPN and num_classes for other heads.
            It will automatically set as `num_classes` if None is given.
        diff_rad_by_sin (bool): Whether to change the difference
            into sin difference for box regression loss. Defaults to True.
        dir_offset (float): Parameter used in direction
            classification. Defaults to 0.
        dir_limit_offset (float): Parameter used in direction
            classification. Defaults to 0.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dir (:obj:`ConfigDict` or dict): Config of direction classifier
            loss.
        loss_attr (:obj:`ConfigDict` or dict): Config of attribute classifier
            loss, which is only active when `pred_attrs=True`.
        bbox_code_size (int): Dimensions of predicted bounding boxes.
        pred_attrs (bool): Whether to predict attributes.
            Defaults to False.
        num_attrs (int): The number of attributes to be predicted.
            Default: 9.
        pred_velo (bool): Whether to predict velocity.
            Defaults to False.
        pred_bbox2d (bool): Whether to predict 2D boxes.
            Defaults to False.
        group_reg_dims (tuple[int], optional): The dimension of each regression
            target group. Default: (2, 1, 3, 1, 2).
        cls_branch (tuple[int], optional): Channels for classification branch.
            Default: (128, 64).
        reg_branch (tuple[tuple], optional): Channels for regression branch.
            Default: (
                (128, 64),  # offset
                (128, 64),  # depth
                (64, ),  # size
                (64, ),  # rot
                ()  # velo
            ),
        dir_branch (Sequence[int]): Channels for direction
            classification branch. Default: (64, ).
        attr_branch (Sequence[int]): Channels for classification branch.
            Default: (64, ).
        conv_cfg (:obj:`ConfigDict` or dict, Optional): Config dict for
            convolution layer. Default: None.
        norm_cfg (:obj:`ConfigDict` or dict, Optional): Config dict for
            normalization layer. Default: None.
        train_cfg (:obj:`ConfigDict` or dict, Optional): Training config
            of anchor head.
        test_cfg (:obj:`ConfigDict` or dict, Optional): Testing config of
            anchor head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
    """  # noqa: W605

    _version = 1

    def __init__(
            self,
            num_classes,
            in_channels,
            widen_factor,
            feat_channels,
            stacked_convs,
            strides,
            group_reg_dims,
            cls_branch,
            reg_branch,
            attr_branch,
            loss_cls,
            loss_offset,
            loss_depth,
            loss_dim,
            loss_dir,
            prior_generator,
            bbox_coder,
            loss_offset_corner=None,
            loss_depth_corner=None,
            loss_dim_corner=None,
            loss_dir_corner=None,
            loss_corner=None,
            loss_center=None,
            dcn_on_last_conv=False,
            num_attrs: int = 0,  # For nuscenes = 9
            loss_bbox_weight_compensation=0.0,
            loss_velo=None,
            loss_attr=None,
            conv_bias='auto',
            conv_cfg: OptConfigType = None,
            norm_cfg: OptConfigType = None,
            act_cfg: OptConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = int(in_channels * widen_factor)
        if stacked_convs == 0:
            self.feat_channels = self.in_channels
        else:
            self.feat_channels = int(feat_channels * widen_factor)
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.num_levels = len(strides)
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_depth = MODELS.build(loss_depth)
        self.loss_offset = MODELS.build(loss_offset)
        self.loss_dim = MODELS.build(loss_dim)
        self.loss_dir = MODELS.build(loss_dir)
        self.loss_depth_corner = None if loss_depth_corner is None else MODELS.build(loss_depth_corner)
        self.loss_offset_corner = None if loss_offset_corner is None else MODELS.build(loss_offset_corner)
        self.loss_dim_corner = None if loss_dim_corner is None else MODELS.build(loss_dim_corner)
        self.loss_dir_corner = None if loss_dir_corner is None else MODELS.build(loss_dir_corner)
        self.loss_corner = None if loss_corner is None else MODELS.build(loss_corner)
        self.loss_center = None if loss_center is None else MODELS.build(loss_center)
        self.group_reg_dims = list(group_reg_dims)
        self.cls_branch = [int(i * widen_factor) for i in cls_branch]
        self.reg_branch = [[int(j * widen_factor) for j in i] for i in reg_branch]
        assert len(reg_branch) == len(group_reg_dims), 'The number of ' \
                                                       'element in reg_branch and group_reg_dims should be the same.'
        self.pred_velo = loss_velo is not None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.loss_bbox_weight_compensation = loss_bbox_weight_compensation
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self.featmap_sizes = [torch.empty(1)] * self.num_levels
        self.fp16_enabled = False
        self.pred_attrs = num_attrs > 0
        if self.pred_attrs:
            self.num_attrs = num_attrs
            self.attr_background_label = num_attrs
            self.loss_attr = MODELS.build(loss_attr)
            self.attr_branch = attr_branch
        if self.pred_velo:
            self.loss_velo = MODELS.build(loss_velo)
        self._init_layers()
        self.special_init()

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg.sampler, default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler(context=self)

            self.featmap_sizes_train = None
            self.flatten_priors_train = None

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    act_cfg=self.act_cfg,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    act_cfg=self.act_cfg,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_branch(self, conv_channels=(64), conv_strides=(1)):
        """Initialize conv layers as a prediction branch."""
        conv_before_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.feat_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else:
            conv_channels = [self.feat_channels] + list(conv_channels)
            conv_strides = list(conv_strides)
        for i in range(len(conv_strides)):
            conv_before_pred.append(
                ConvModule(
                    conv_channels[i],
                    conv_channels[i + 1],
                    3,
                    stride=conv_strides[i],
                    padding=1,
                    act_cfg=self.act_cfg,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        return conv_before_pred

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls_prev = self._init_branch(
            conv_channels=self.cls_branch,
            conv_strides=(1,) * len(self.cls_branch))
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.cls_out_channels,
                                  1)
        self.conv_reg_prevs = nn.ModuleList()
        self.conv_regs = nn.ModuleList()
        for i in range(len(self.group_reg_dims)):
            reg_dim = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]
            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs.append(
                    self._init_branch(
                        conv_channels=reg_branch_channels,
                        conv_strides=(1,) * len(reg_branch_channels)))
                self.conv_regs.append(nn.Conv2d(reg_branch_channels[-1], reg_dim, 1))
            else:
                self.conv_reg_prevs.append(None)
                self.conv_regs.append(
                    nn.Conv2d(self.feat_channels, reg_dim, 1))
        if self.pred_attrs:
            self.conv_attr_prev = self._init_branch(
                conv_channels=self.attr_branch,
                conv_strides=(1,) * len(self.attr_branch))
            self.conv_attr = nn.Conv2d(self.attr_branch[-1], self.num_attrs, 1)

    def init_weights(self):
        """Initialize weights of the head.

        We currently still use the customized defined init_weights because the
        default init of DCN triggered by the init_cfg will init
        conv_offset.weight, which mistakenly affects the training stability.
        """
        for modules in [self.cls_convs, self.reg_convs, self.conv_cls_prev]:
            for m in modules:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        for conv_reg_prev in self.conv_reg_prevs:
            if conv_reg_prev is None:
                continue
            for m in conv_reg_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        if self.pred_attrs:
            for m in self.conv_attr_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        for conv_reg in self.conv_regs:
            normal_init(conv_reg, std=0.01)
        if self.pred_attrs:
            normal_init(self.conv_attr, std=0.01, bias=bias_cls)

    def forward(self, x):
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually contain classification scores, bbox predictions,
                and direction class predictions.
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * bbox_code_size.
                dir_cls_preds (list[Tensor]): Box scores for direction class
                    predictions on each scale level, each is a 4D-tensor,
                    the channel number is num_points * 2. (bin = 2)
                attr_preds (list[Tensor]): Attribute scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_attrs.
        """
        return multi_apply(self.forward_single, x)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: Scores for each class, bbox predictions, direction class,
                and attributes, features after classification and regression
                conv layers, some models needs these features like FCOS.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        # clone the cls_feat for reusing the feature map afterwards
        clone_cls_feat = cls_feat  # .clone()
        for conv_cls_prev_layer in self.conv_cls_prev:
            clone_cls_feat = conv_cls_prev_layer(clone_cls_feat)
        cls_score = self.conv_cls(clone_cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = []
        for i in range(len(self.group_reg_dims)):
            # clone the reg_feat for reusing the feature map afterwards
            clone_reg_feat = reg_feat  # .clone()
            if len(self.reg_branch[i]) > 0:
                for conv_reg_prev_layer in self.conv_reg_prevs[i]:
                    clone_reg_feat = conv_reg_prev_layer(clone_reg_feat)
            bbox_pred.append(self.conv_regs[i](clone_reg_feat))
        bbox_pred = torch.cat(bbox_pred, dim=1)

        attr_pred = None
        if self.pred_attrs:
            # clone the cls_feat for reusing the feature map afterwards
            clone_cls_feat = cls_feat  # .clone()
            for conv_attr_prev_layer in self.conv_attr_prev:
                clone_cls_feat = conv_attr_prev_layer(clone_cls_feat)
            attr_pred = self.conv_attr(clone_cls_feat)

        return cls_score, bbox_pred, attr_pred

    # @profile
    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list,
                                                               dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        outs = self(x)
        losses = self.loss_by_feat(*outs, **batch_data_samples)
        return losses

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        # results_list = []
        # for i, (labels, bboxes, scores, img_meta) in enumerate(
        #         zip(batch_data_samples['labels'], batch_data_samples['bboxes_3d'], batch_data_samples['pad_bbox_flag'],
        #             batch_data_samples['img_metas'])):
        #     results = InstanceData()
        #     results.bboxes_3d = bboxes
        #     results.labels_3d = labels.view(-1).int()
        #     scores = scores * 0.9 + torch.rand_like(scores, dtype=torch.float32) * 0.1
        #     results.scores_3d = scores.view(-1).float()
        #     results.bboxes_3d = img_meta['box_type_3d'](
        #         results.bboxes_3d, box_dim=results.bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5))
        #     results_list.append(results)
        # return results_list
        # batch_data_samples = batch_data_samples['img_metas']
        outs = self(x)
        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_data_samples, rescale=rescale)
        return predictions

    # @profile
    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            attr_preds: List[Tensor],
            bboxes, bboxes_3d, labels, target_3d, pad_bbox_flag, img_metas, **kwargs,
    ) -> dict:
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            attr_preds (list[Tensor]): Attribute scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_attrs.

            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes``、``labels``.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        assert len(cls_scores) == len(bbox_preds)
        img2cams = torch.stack([input_meta['img2cam'] for input_meta in img_metas])
        K_out = torch.stack([input_meta['K_out'] for input_meta in img_metas])
        device = cls_scores[0].device

        # If the shape does not equal, generate new one
        if featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = featmap_sizes
            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                featmap_sizes, device=device, with_stride=True)
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ], 1).contiguous()

        flatten_bboxes = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, sum(self.group_reg_dims))
            for bbox_pred in bbox_preds
        ], 1)

        if self.group_reg_dims[1] == 2:
            score_depth = flatten_bboxes[:, :, self.group_reg_dims[0] + 1:self.group_reg_dims[0] + 2]
            score_depth = (1 * torch.exp(-score_depth) - 0.5).clamp_min(1e-4).log().detach()
            flatten_cls_scores = flatten_cls_scores + score_depth

        with torch.no_grad():
            flatten_bboxes_decode = self.bbox_coder.decode(flatten_bboxes, self.group_reg_dims, flatten_cls_scores,
                                                           self.flatten_priors_train, K_out, img2cams)
        assigned_result = self.assigner(flatten_bboxes_decode.detach(),
                                        flatten_cls_scores.detach(),
                                        self.flatten_priors_train, labels,
                                        bboxes, target_3d[:, :, :2], bboxes_3d, pad_bbox_flag)

        labels = assigned_result['assigned_labels']
        label_weights = assigned_result['assigned_labels_weights']  # .view(-1)
        assign_metrics = assigned_result['assign_metrics']  # .view(-1)
        batch_index = assigned_result['assigned_batch_index']
        pred_index = assigned_result['assigned_pred_index']
        gt_index = assigned_result['assigned_gt_index']
        cls_preds = flatten_cls_scores.view(-1, self.cls_out_channels)
        # bbox_preds = flatten_bboxes.view(-1, sum(self.group_reg_dims))

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        # bg_class_ind = self.num_classes
        # pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        avg_factor = assign_metrics.sum()
        if dist.is_initialized():
            dist.all_reduce(avg_factor)
        avg_factor = avg_factor.clamp_(min=1).item()
        # avg_factor = batch_index.shape[0]
        loss_cls = self.loss_cls(cls_preds, (labels.view(-1), assign_metrics.view(-1)), label_weights.view(-1),
                                 avg_factor=avg_factor)
        message_hub = MessageHub.get_current_instance()
        message_hub.update_scalar('train/pos_num', batch_index.shape[0] / pad_bbox_flag.sum())
        # if batch_index.shape[0] / pad_bbox_flag.sum() < 1:
        #     print(batch_index.shape[0], pad_bbox_flag.sum())
        #     print(0)
        if len(batch_index) > 0:
            bbox_preds = flatten_bboxes[batch_index, pred_index]
            bbox_targets = target_3d[batch_index, gt_index]

            weight = assign_metrics[batch_index, pred_index, None]
            message_hub.update_scalar('train/iou3d', weight.mean())
            # weight = 1
            # weight_debug = weight.detach().cpu().numpy()
            if self.assigner.iou_type == '3d':
                # weight = torch.ones_like(weight)
                weight = weight + self.loss_bbox_weight_compensation
                # weight = weight.clamp_max(1)
            else:
                weight = weight  # + 0.3
                # weight = weight.clamp_max(1)

            avg_factor = weight.sum()
            if dist.is_initialized():
                dist.all_reduce(avg_factor)
            bbox_preds_split = torch.split(bbox_preds, self.group_reg_dims, dim=1)
            labels = labels[batch_index, pred_index]
            priors = self.flatten_priors_train[pred_index]
            center_target, depths_target, dims_target, dir_target = self.bbox_coder.encode(bbox_targets, labels, priors)
            loss_offset = self.loss_offset(bbox_preds_split[0], center_target, weight=weight, avg_factor=avg_factor)
            # print('offset', (center_target).min(), (center_target).max(),(center_target).mean())
            loss_depth = self.loss_depth(bbox_preds_split[1], depths_target, weight=weight, avg_factor=avg_factor)
            # print('depth',(depths_target).min(), (depths_target).max(), (depths_target).mean())
            loss_size = self.loss_dim(bbox_preds_split[2], dims_target, weight=weight, avg_factor=avg_factor)
            # print('size',(dims_target).min(), (dims_target).max(), (dims_target).mean())
            loss_rotsin = self.loss_dir(bbox_preds_split[3], dir_target, weight=weight, avg_factor=avg_factor)
            # print('rot',(dir_target).min(), (dir_target).max(), (dir_target).mean())
            # debug_true_bbox3d = torch.cat()
            K_out = K_out[None, batch_index]
            img2cams = img2cams[None, batch_index]
            bbox_gt_split = [center_target, depths_target, dims_target,
                             torch.cat([torch.sin(dir_target), torch.cos(dir_target)], 1), ]
            if self.loss_corner is not None:
                # bbox_preds = torch.cat([bbox_gt_split[j] for j in range(4)], 1).unsqueeze(0)
                bbox_preds = bbox_preds.unsqueeze(0)
                bbox3d_preds = self.bbox_coder.decode(bbox_preds, self.group_reg_dims, labels.unsqueeze(0),
                                                      priors, K_out, img2cams)
                bbox3d_preds = bbox3d_preds.squeeze(0)
                bbox3d_preds = img_metas[0]['box_type_3d'](bbox3d_preds, box_dim=bbox3d_preds.shape[-1],
                                                           origin=(0.5, 0.5, 0.5))

                loss_corner = self.loss_corner(bbox3d_preds.corners.flatten(1, -1), bbox_targets[:, 7:7 + 24],
                                               weight=weight, avg_factor=avg_factor)
                # loss_corner = loss_rotsin.new_tensor(0)
            else:
                loss_corner = loss_rotsin.new_tensor(0)
            if self.loss_center is not None:
                bboxes_3d_targets = bboxes_3d[batch_index, gt_index]
                center3d_preds = self.bbox_coder.decode_center(bbox_preds_split[0].unsqueeze(0),
                                                               bbox_preds_split[1].unsqueeze(0), img2cams, priors,
                                                               K_out)
                center3d_preds = center3d_preds.squeeze(0)
                loss_center = self.loss_center(center3d_preds[:, 0::2], bboxes_3d_targets[:, [0, 2]], weight=weight,
                                               avg_factor=avg_factor)
            else:
                loss_center = loss_rotsin.new_tensor(0)
            loss_var_list = []
            for i, loss_func in enumerate(
                    [self.loss_offset_corner, self.loss_depth_corner, self.loss_dim_corner, self.loss_dir_corner, ]):
                if loss_func is not None:
                    bbox_gt_preds = torch.cat([bbox_preds_split[j] if j == i else bbox_gt_split[j] for j in range(4)],
                                              1).unsqueeze(0)
                    # bbox_gt_preds = torch.cat([bbox_gt_split[j] for j in range(4)], 1).unsqueeze(0)
                    bbox_gt_preds = self.bbox_coder.decode(bbox_gt_preds, self.group_reg_dims, labels.unsqueeze(0),
                                                           priors,
                                                           K_out, img2cams).squeeze(0)
                    # bboxes_3d = bboxes_3d[batch_index, gt_index]
                    # bboxes_3d[:, -1] = bboxes_3d[:, -1] % (2 * np.pi)
                    # bbox_gt_preds[:, -1] = bbox_gt_preds[:, -1] % (2 * np.pi)
                    # print((bbox_gt_preds-bboxes_3d).abs().max())
                    # debug_bbox3d_preds = bbox_gt_preds.detach().cpu().numpy()
                    # debug_bboxes_3d = bboxes_3d.detach().cpu().numpy()
                    # debug_bboxes_delta = debug_bboxes_3d - debug_bbox3d_preds
                    # print(bbox_gt_preds.shape, bboxes_3d.shape)
                    # print(bbox3d_preds.corners.flatten(1, -1).shape, bbox_targets[:, 7:7 + 24].shape)
                    # debug_cor1 = bbox3d_preds.corners.flatten(1, -1).detach().cpu().numpy()
                    # debug_cor2 = bbox_targets[:, 7:7 + 24].detach().cpu().numpy()
                    # debug_cor21 = (bbox3d_preds.corners.flatten(1, -1) - bbox_targets[:, 7:7 + 24]).detach().cpu().numpy()
                    # print((bbox3d_preds.corners.flatten(1, -1) - bbox_targets[:, 7:7 + 24]).abs().max())
                    bbox_gt_preds = img_metas[0]['box_type_3d'](bbox_gt_preds, box_dim=bbox_gt_preds.shape[-1],
                                                                origin=(0.5, 0.5, 0.5))
                    loss_var = loss_func(bbox_gt_preds.corners.flatten(1, -1),
                                         bbox_targets[:, 7:7 + 24],
                                         weight=weight, avg_factor=avg_factor)
                else:
                    loss_var = loss_rotsin.new_tensor(0)
                loss_var_list.append(loss_var)
            loss_offset_corner, loss_depth_corner, loss_dim_corner, loss_dir_corner = loss_var_list
            loss_velo = loss_rotsin.new_tensor(0)
            if self.pred_velo:
                loss_velo = self.loss_velo(bbox_preds_split[4], bbox_targets[7:9], weight=weight, avg_factor=avg_factor)
            loss_attr = loss_rotsin.new_tensor(0)
            if self.pred_attrs:
                flatten_attr_preds = [
                    attr_pred.permute(0, 2, 3, 1).reshape(-1, self.num_attrs)
                    for attr_pred in attr_preds
                ]
                flatten_attr_preds = torch.cat(flatten_attr_preds)
                pos_attr_preds = flatten_attr_preds[batch_index, pred_index]
                loss_attr = self.loss_attr(pos_attr_preds, bbox_targets[9:10], weight=weight, avg_factor=avg_factor)
        else:
            loss_offset = loss_cls.new_tensor(0)
            loss_depth = loss_cls.new_tensor(0)
            loss_size = loss_cls.new_tensor(0)
            loss_rotsin = loss_cls.new_tensor(0)
            loss_velo = loss_cls.new_tensor(0)
            loss_attr = loss_cls.new_tensor(0)
            loss_corner = loss_cls.new_tensor(0)
            loss_center = loss_cls.new_tensor(0)
            loss_offset_corner = loss_cls.new_tensor(0)
            loss_depth_corner = loss_cls.new_tensor(0)
            loss_dim_corner = loss_cls.new_tensor(0)
            loss_dir_corner = loss_cls.new_tensor(0)
        loss_dict = dict(
            loss_cls=loss_cls,
            loss_offset=loss_offset,
            loss_depth=loss_depth,
            loss_size=loss_size,
            loss_rotsin=loss_rotsin,
            loss_velo=loss_velo,
            loss_attr=loss_attr,
            loss_corner=loss_corner,
            loss_center=loss_center,
            loss_offset_corner=loss_offset_corner,
            loss_depth_corner=loss_depth_corner,
            loss_dim_corner=loss_dim_corner,
            loss_dir_corner=loss_dir_corner,
        )

        return loss_dict

    # @profile
    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        attr_preds: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        img2cams = torch.stack([input_meta['img2cam'] for input_meta in batch_img_metas])
        K_out = torch.stack([input_meta['K_out'] for input_meta in batch_img_metas])
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device, with_stride=True)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, sum(self.group_reg_dims))
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)

        if self.group_reg_dims[1] == 2:
            score_depth = flatten_bbox_preds[:, :, self.group_reg_dims[0] + 1:self.group_reg_dims[0] + 2]
            score_depth = (1 * torch.exp(-score_depth) - 0.5).clamp_min(1e-4).log()
            flatten_cls_scores = flatten_cls_scores + score_depth
        flatten_cls_scores = flatten_cls_scores.sigmoid()
        flatten_decoded_bboxes = self.bbox_coder.decode(flatten_bbox_preds, self.group_reg_dims, flatten_cls_scores,
                                                        flatten_priors,
                                                        K_out, img2cams)

        if self.pred_velo:
            flatten_decoded_bboxes = torch.cat([flatten_decoded_bboxes,
                                                flatten_bbox_preds[:, :,
                                                6 + self.bbox_coder.num_dir_bins * 4:8 + self.bbox_coder.num_dir_bins * 4]],
                                               dim=2)
        if self.pred_attrs:
            flatten_attr_preds = [
                attr_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_attrs)
                for attr_pred in attr_preds
            ]
            flatten_attr_preds = torch.cat(flatten_attr_preds)
        # score_depth = None
        # if self.group_reg_dims[1] == 2:
        #     score_depth = flatten_bbox_preds[:, :, self.group_reg_dims[0] + 1:self.group_reg_dims[0] + 2]
        #     score_depth = score_depth.exp()
        #     score_depth = (3 - score_depth) / (3 + score_depth)
        #     score_depth = score_depth.clamp(0, 1)
        results_list = []
        for i, (bboxes, scores, img_meta, K_o) in enumerate(
                zip(flatten_decoded_bboxes, flatten_cls_scores, batch_img_metas, K_out)):
            # ori_shape = img_meta['ori_shape']
            # scale_factor = img_meta['scale_factor']
            # if 'pad_param' in img_meta:
            #     pad_param = img_meta['pad_param']
            # else:
            #     pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes_3d = bboxes
                empty_results.scores_3d = scores[:, 0]
                empty_results.labels_3d = scores[:, 0].int()
                results_list.append(empty_results)
                continue
            # if score_depth is not None:
            #     scores = scores * score_depth[i]
            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                topk_scores, topk_inds = torch.topk(scores, nms_pre // self.num_classes, dim=0)
                keep_idxs = topk_inds.view(-1)
                scores = topk_scores.view(-1)
                labels = torch.arange(self.num_classes, device=scores.device, dtype=torch.long).unsqueeze(0).expand_as(
                    topk_scores).reshape(-1)
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)
            bboxes = bboxes[keep_idxs]
            results = InstanceData()
            results.bboxes_3d = bboxes
            results.labels_3d = labels
            results.scores_3d = scores
            if self.pred_attrs:
                attrs = flatten_attr_preds[i][keep_idxs]
                results.attr_labels = attrs
            results = self._bbox_post_process(results, cfg, with_nms, K_o)
            results.bboxes_3d = img_meta['box_type_3d'](
                results.bboxes_3d, box_dim=results.bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5))
            results_list.append(results)
        return results_list

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           with_nms, K_o) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # if hasattr(results, 'score_factors'):
        #     # TODO： Add sqrt operation in order to be consistent with
        #     #  the paper.
        #     score_factors = results.pop('score_factors')
        #     results.scores = results.scores * score_factors

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg

        if hasattr(cfg, 'dm3d') and results.bboxes_3d.numel() > 0:
            results = self.dm3d(results, cfg, K_o)

        if hasattr(cfg, 'nms') and results.bboxes_3d.numel() > 0:
            keep_idxs = self.box3d_nms(results.bboxes_3d, results.scores_3d, results.labels_3d, cfg)
            results = results[keep_idxs]

        return results

    def box3d_nms(self, bboxes3d, scores, labels, cfg):
        if cfg.nms.type == '3d':
            keep = nms_3d_gpu(bboxes3d, scores, labels, cfg.nms.iou_threshold,
                              fol_maxsize=cfg.max_per_img)
        elif cfg.nms.type == 'bev':
            keep = nms_bev_gpu(bboxes3d, scores, labels, cfg.nms.iou_threshold,
                               fol_maxsize=cfg.max_per_img)
        elif cfg.nms.type == 'dist':
            keep = nms_dist_gpu(bboxes3d, scores, labels, cfg.nms.iou_threshold,
                                fol_maxsize=cfg.max_per_img)
        else:
            raise NotImplementedError
        return keep

    def dm3d(self, results, cfg, K_o):
        K_o = K_o.unsqueeze(0)

        keep = ((results.bboxes_3d[:, 2] + K_o[0, 2]) < 10)
        # keep = (results.scores_3d > 0.7)
        old_results = results[keep]
        results = results[~keep]
        xyz = results.bboxes_3d[:, :3] + K_o
        lhwa = results.bboxes_3d[:, 3:]
        z = xyz[:, 2:]
        xy = xyz[:, :2]
        sigma = torch.exp(z / cfg.dm3d.lamda)
        # sigma = (z - 0) * 0.1
        # sigma = 2.5
        # sigma = -torch.log(results.scores_3d).unsqueeze(-1)
        # sigma = 10 * (1 - torch.pow(results.scores_3d, 2)).unsqueeze(-1)

        depth_offset = results.bboxes_3d.new_tensor(cfg.dm3d.depth_offset).unsqueeze(0)  # [1, 7]
        ts = torch.exp(-torch.square(depth_offset / sigma))  # [N, 7]

        # ts = results.bboxes_3d.new_tensor([0.7, 0.8, 0.9, 1, 0.9, 0.8, 0.7]).unsqueeze(0)  # [1, 7]
        # sign = results.bboxes_3d.new_tensor([-1, -1, -1, 0, 1, 1, 1]).unsqueeze(0)  # [1, 7]
        # depth_offset = torch.sqrt(-torch.log(ts) * sigma.square()) * sign

        new_z = (z + depth_offset).unsqueeze(-1)  # [N, 7, 1]
        new_xy = (xy / z).unsqueeze(1) * new_z  # [N, 7, 2]
        new_lhwa = lhwa.unsqueeze(1).repeat(1, len(cfg.dm3d.depth_offset), 1)  # [N, 7, 4]
        new_bboxes_3d = torch.cat([new_xy, new_z, new_lhwa], dim=-1).flatten(0, 1)
        new_bboxes_3d[:, :3] -= K_o
        new_scores = results.scores_3d.unsqueeze(1) * ts  # [N, 7]
        new_labels = results.labels_3d.unsqueeze(1).repeat(1, len(cfg.dm3d.depth_offset))  # [N, 7]

        new_results = InstanceData()
        new_results.bboxes_3d = new_bboxes_3d
        new_results.scores_3d = new_scores.view(-1)
        new_results.labels_3d = new_labels.view(-1)
        if self.pred_attrs:
            new_results.attr_labels = results.attr_labels.unsqueeze(1).repeat(1, len(cfg.dm3d.depth_offset), 1).flatten(
                0, 1)  # [N, 7, 4]
        new_results = new_results.cat([old_results, new_results])
        return new_results
