# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmyolo.registry import TASK_UTILS
from mmdet.models.task_modules import BaseBBoxCoder


@TASK_UTILS.register_module()
class AnchorFreeBBox3DCoder(torch.nn.Module):

    def __init__(self, base_depth, base_offset, base_dims, cylinder, relative_depth):
        super(AnchorFreeBBox3DCoder, self).__init__()
        self.base_depth = base_depth
        self.base_offset = base_offset
        self.register_buffer('base_dims', torch.tensor(base_dims))
        self.cylinder = cylinder
        self.relative_depth = relative_depth

    def encode(self, gt_bboxes_target, gt_labels_3d, priors):
        """Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes): Ground truth bboxes
                with shape (n, 7).
            gt_labels_3d (torch.Tensor): Ground truth classes. (n, ).

        Returns:
            tuple: Targets of center, size and direction.
        """
        # generate center target
        # gt_bboxes_3d = gt_bboxes_target[:, :7]
        centers_2d = gt_bboxes_target[:, :2]
        depths = gt_bboxes_target[:, 2:3]
        dims = gt_bboxes_target[:, 3:6]
        alphas = gt_bboxes_target[:, 6:7]
        # bboxes3d_target = gt_bboxes_target[:, 7:7 + 24]
        center_target = self.encode_offset(centers_2d, priors[:, :2], priors[:, 2:])
        depths_target = self.encode_depth(depths)
        dims_target = self.encode_dimension(dims, gt_labels_3d)
        dir_target = self.encode_orientation(alphas)
        return center_target, depths_target, dims_target, dir_target

    # @profile
    def decode(self, bbox_out, group_reg_dims, cls_out, priors, K_out, img2cams):
        '''

        Args:
            bbox_out: [b,n,7+*]
            cls_out: [b,n,c] or cls_out: [b,n]
            priors: [n,4]
             img2cams: [b,3,3] or img2cams: [b,n,3,3]
        Returns:
            bbox3d:[b,n,7]
        '''
        bbox_out = torch.split(bbox_out, group_reg_dims, dim=2)
        centers2d_offsets = bbox_out[0]
        depth_offsets = bbox_out[1]
        dimensions_offsets = bbox_out[2]
        angle_offsets = bbox_out[3]
        if angle_offsets.shape[2] > 2:
            angle_cls, angle_res = torch.chunk(angle_offsets, 2, 2)
        else:
            angle_cls = None
            angle_res = angle_offsets
        pred_locations = self.decode_center(centers2d_offsets, depth_offsets, img2cams, priors)
        pred_dimensions = self.decode_dimension(dimensions_offsets, cls_out)
        pred_orientations = self.decode_orientation(angle_res, pred_locations, angle_cls)
        if K_out is not None:
            if len(K_out.shape) == 2:
                K_out = K_out.unsqueeze(1)
            pred_locations = pred_locations - K_out
        bbox3d = torch.cat([pred_locations, pred_dimensions, pred_orientations], dim=-1)
        return bbox3d

    def decode_center(self, centers2d_offsets, depth_offsets, img2cams, priors, K_out=None):
        depths = self.decode_depth(depth_offsets, img2cams)
        centers2d = self.decode_offset(centers2d_offsets, priors[:, :2], priors[:, 2:])
        # get the 3D Bounding box's center location.
        pred_locations = self.decode_location(centers2d, depths, img2cams)
        if K_out is not None:
            if len(K_out.shape) == 2:
                K_out = K_out.unsqueeze(1)
            pred_locations = pred_locations - K_out
        return pred_locations

    def decode_depth(self, depth_offsets, img2cams):
        '''
        depth_offsets: [b,n,1]
        img2cams: [b, 3, 3] or img2cams: [b, n, 3, 3]
        '''
        if depth_offsets.shape[2] == 2:
            depth_offsets = depth_offsets[:, :, :1]
        depths = depth_offsets * self.base_depth[1] + self.base_depth[0]
        if self.relative_depth:
            if len(img2cams.shape) == 3:
                img2cams = img2cams[:, None, :, :]
            depths = depths / img2cams[:, :, 1:2, 1]
        return depths

    def encode_depth(self, depth, ):
        """Transform depth offset to depth."""
        depth_offsets = (depth - self.base_depth[0]) / self.base_depth[1]
        return depth_offsets

    def decode_offset(self, centers2d_offsets, points, stride):
        '''

        Args:
            centers2d_offsets: [b,n,2]
            points: [n,2]
            stride: [n,2]

        Returns:
            [b,n,2]
        '''
        centers2d_offsets = centers2d_offsets * self.base_offset[1] + self.base_offset[0]
        centers2d = points.unsqueeze(0) + centers2d_offsets * stride.unsqueeze(0)
        return centers2d

    def encode_offset(self, centers2d, points, stride):
        '''

        Args:
            centers2d_offsets: [n,2]
            points: [n,2]
            stride: [n,2]

        Returns:
            [n,2]
        '''
        centers2d_offsets = ((centers2d - points) / stride - self.base_offset[0]) / self.base_offset[1]
        return centers2d_offsets

    def decode_location(self, centers2d, depths, img2cams):
        '''

        Args:
            centers2d: [b,n,2]
            depths: [b,n,1]
            img2cams: [b,3,3] or img2cams: [b,n,3,3]
        Returns:
            centers3d: [b,n,3]
        '''
        # number of points
        N = centers2d.shape[1]
        # batch_size
        B = img2cams.shape[0]

        centers2d_extend = torch.cat((centers2d, centers2d.new_ones(B, N, 1)), dim=2)

        centers2d_img = centers2d_extend.unsqueeze(-1)  # [B, N, 3, 1]
        if len(img2cams.shape) == 3:
            img2cams = img2cams[:, None, :, :]  # [B, 1, 3, 3]
        locations = torch.matmul(img2cams, centers2d_img).squeeze(-1)  # [B, N, 3]
        if self.cylinder:
            theta = locations[:, :, :1]
            yy = locations[:, :, 1:2]
            locations = torch.cat((torch.sin(theta), yy, torch.cos(theta)), dim=2)  # [B, N, 3]
        xyz = locations * depths
        return xyz

    # @profile
    def decode_dimension(self, dims_offset, labels):
        """Transform dimension offsets to dimension according to its category.

        Args:
            labels (Tensor): Each points' category id.
                shape: (B, N, Class) or (B, N)
            dims_offset (Tensor): Dimension offsets.
                shape: (B, N, 3)
        """
        B, N = dims_offset.shape[:2]
        if len(labels.shape) == 3:
            labels = torch.argmax(labels, dim=2).flatten()
        else:
            labels = labels.flatten()
        # base_dims = dims_offset.new_tensor(self.base_dims)
        base_dims = self.base_dims
        dims_select = base_dims[:, labels, :]
        dimensions = (dims_offset.flatten(0, 1) * dims_select[1]).exp() * dims_select[0]
        return dimensions.view(B, N, 3)

    def encode_dimension(self, dims, labels):
        """Transform dimension offsets to dimension according to its category.

        Args:
            labels (Tensor): Each points' category id.
                shape: (N, )
            dims (Tensor): Dimension offsets.
                shape: (N, 3)
        """
        N, _ = dims.shape
        # dims_ori_debug = dims.cpu().numpy()
        labels = labels.flatten().long()
        # base_dims = dims.new_tensor(self.base_dims)
        base_dims = self.base_dims
        dims_select = base_dims[:, labels, :]
        # dims_select_debug = dims_select.cpu().numpy()
        dims = (dims.clamp_min(0.1) / dims_select[0]).log() / dims_select[1]
        # dims_debug = dims.cpu().numpy()
        return dims

    def decode_orientation(self, ori_vector, locations, ori_cls=None):
        """Retrieve object orientation.

        Args:
            ori_vector (Tensor): Local orientation in [sin, cos] format.
                shape: (B, N, 2*bin_num)
            locations (Tensor): Object location.
                shape: (B, N, 3)
            K_out: [B, 3]
            ori_cls: [B, N, 2*bin_num]
        Return:
            Tensor: yaw(Orientation). Notice that the yaw's
                range is [-np.pi, np.pi].
                shape：(B, N, 1）
        """
        B, N, num_dir_bins = ori_vector.shape
        num_dir_bins = num_dir_bins / 2
        # locations = locations + K_out.unsqueeze(1)
        rays = torch.atan2(locations[:, :, 0], locations[:, :, 2])
        if ori_cls is None:
            alphas = torch.atan2(ori_vector[:, :, 0], ori_vector[:, :, 1])
        else:
            ori_cls = ori_cls.view(B, N, -1, 2)
            ori_cls = torch.softmax(ori_cls, dim=3)
            ori_cls = ori_cls[:, :, :, 1]
            ori_cls = torch.argmax(ori_cls, dim=2, keepdim=False)
            ori_cls_index = ori_cls.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1, 2)
            ori_vector = ori_vector.view(B, N, -1, 2)
            ori_vector = torch.gather(ori_vector, dim=2, index=ori_cls_index)
            alphas = torch.atan2(ori_vector[:, :, 0, 0], ori_vector[:, :, 0, 1])
            angle_per_class = 2 * np.pi / num_dir_bins
            angle_center = ori_cls.float() * angle_per_class
            alphas = angle_center + alphas
        yaws = (alphas + rays + np.pi) % (2 * np.pi) - np.pi
        yaws = yaws.unsqueeze(-1)
        return yaws

    def encode_orientation(self, angle):
        """Convert continuous angle to a discrete class and a residual.

        Convert continuous angle to a discrete class and a small
        regression number from class center angle to current angle.

        Args:
            angle (torch.Tensor): [N, 1] Angle is from 0-2pi (or -pi~pi),
                class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N).

        Returns:
            angle_cls_res: [N, 2*num_dir_bins] or [N, 1] (if num_dir_bins=1)
        """
        return angle
