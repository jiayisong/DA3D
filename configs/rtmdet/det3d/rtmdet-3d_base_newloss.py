_base_ = 'rtmdet-3d_base.py'

loss_bbox_weight = 1.0

model = dict(
    bbox_head=dict(
        loss_bbox_weight_compensation=0.3,
        group_reg_dims=(2, 1, 3, 32),  # offset, depth, size, rot, velo
        loss_offset=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 1.0),
        loss_depth=dict(type='SimpleUncertainL1Loss', loss_weight=loss_bbox_weight * 1.0),
        loss_dim=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 1.0),
        loss_dir=dict(type='MultiBinLoss', bin_margin=1 / 3, loss_weight=loss_bbox_weight * 1.0),
        loss_corner=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 0.2),

        loss_offset_corner=None,
        loss_depth_corner=None,
        loss_dim_corner=None,
        loss_dir_corner=None,

        # loss_velo=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight*1),
        # loss_attr=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=loss_cls_weight),
    ),
)
