_base_ = '../../_base_/default_runtime.py'

# checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth'  # noqa

# ========================Frequently modified parameters======================

work_dir = '/mnt/jys/mmyolo/work_dirs/large_1/'

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
input_modality = dict(use_lidar=False, use_camera=True)
metainfo = dict(classes=class_names)
num_classes = len(class_names)  # Number of classes for classification
# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 1xb8=8 bs
base_lr = 0.00025  # 0.004 / 16
max_epochs = 12  # Maximum training epochs
num_epochs_stage2 = 3
loss_cls_weight = 1.0
loss_bbox_weight = 1.0
# ========================Possible modified parameters========================
# -----data related-----
img_scale = (544, 960)  # height  width
# Dataset type, this will be used to define the dataset
dataset_type = 'mmdet3d.NuScenesDataset'
data_root = '/mnt/DataSets/nuScenes/'
# data_root = '/dataset/kitti/object_detection/'
# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 1.0
# The scaling factor that controls the width of the network structure
widen_factor = 1.0
# Strides of multi-scale prior box
strides = [8, 16, 32]

norm_cfg = dict(type='BN')  # Normalization config

# -----train val related-----

# single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
model = dict(
    type='SingleStageMono3DDetector',
    data_preprocessor=dict(
        type='RTMDet3DDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        input_channels=4,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
        # init_cfg=dict(
        #    type='Pretrained', prefix='backbone.', checkpoint=checkpoint)
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDet3DHead',
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=norm_cfg,
        num_classes=num_classes,
        widen_factor=widen_factor,
        in_channels=256,
        stacked_convs=0,
        feat_channels=256,
        strides=strides,
        group_reg_dims=(2, 1, 3, 2,),  # offset, depth, size, rot, velo
        cls_branch=(256, 256,),
        reg_branch=(
            (256, 256,),  # offset
            (256, 256,),  # depth
            (256, 256,),  # size
            (256, 256,),  # rot
            # (256, 256, ),  # velo
        ),
        attr_branch=(256, 256,),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=strides),
        bbox_coder=dict(
            type='AnchorFreeBBox3DCoder', cylinder=True,
            base_depth=(28.01, 16.32),
            # base_depth=(0, 1),
            base_offset=(0.50, 0.29),
            base_dims=(((4.62, 1.73, 1.96), (6.93, 2.83, 2.51), (12.56, 3.89, 2.94),
                        (11.22, 3.5, 2.95), (6.68, 3.21, 2.85), (6.68, 3.21, 2.85),
                        (2.11, 1.46, 0.78), (0.73, 1.77, 0.67), (0.41, 1.08, 0.41),
                        (0.5, 0.99, 2.52)),
                       ((0.2, 0.2, 0.2), (0.2, 0.2, 0.2), (0.2, 0.2, 0.2),
                        (0.2, 0.2, 0.2), (0.2, 0.2, 0.2), (0.2, 0.2, 0.2),
                        (0.2, 0.2, 0.2), (0.2, 0.2, 0.2), (0.2, 0.2, 0.2),
                        (0.2, 0.2, 0.2))), ),
        loss_cls=dict(type='mmdet.QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=loss_cls_weight),
        loss_offset=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 1.0),
        loss_depth=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 1.0),
        loss_dim=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 1.0),
        loss_dir=dict(type='MultiBinLoss', bin_margin=1 / 3, loss_weight=loss_bbox_weight * 1.0),
        # loss_dir=dict(type='OneBinLoss', loss_weight=loss_bbox_weight * 0.0),
        # loss_offset_corner=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 4),
        # loss_depth_corner=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 0.2),
        # loss_dim_corner=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 1),
        # loss_dir_corner=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 0.2),
        loss_center=dict(type='DistLoss', loss_weight=loss_bbox_weight * 1.0),
        # loss_corner=None,
        # loss_corner=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 0.2),
        # loss_velo=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight*1),
        # loss_attr=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=loss_cls_weight),
    ),
    train_cfg=dict(
        assigner=dict(
            # type='IOUAssigner3D',
            type='BatchDynamicSoftLabelAssigner3D',
            num_classes=num_classes,
            topk=13,
            iou_type='dist',
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        # The config of multi-label for multi-class prediction.
        multi_label=False,
        # The number of boxes before NMS
        nms_pre=512,
        score_thr=0.00,  # Threshold to filter out boxes.
        # nms=dict(type='3d', iou_threshold=0.7),  # NMS type and threshold  2.83=2^1.5
        max_per_img=128)  # Max number of detections of each image,
)

train_pipeline = []
test_pipeline = []
val_pipeline = test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    collate_fn=dict(type='RTMDet3D_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    # persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            pts='',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        ann_file='nuscenes_infos_train.pkl',
        load_type='mv_image_based',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        serialize_data=False,
        # we use box_type_3d='Camera' in monocular 3d
        # detection task
        box_type_3d='Camera',
        use_valid_flag=True))

val_dataloader = dict(
    batch_size=16,
    num_workers=32,
    pin_memory=True,
    collate_fn=dict(type='RTMDet3D_collate', train=False),
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            pts='',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        ann_file='nuscenes_infos_val.pkl',
        load_type='mv_image_based',
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        test_mode=True,
        filter_empty_gt=False,
        box_type_3d='Camera',
        use_valid_flag=True))

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    test_cfg=dict(
        nms=dict(type='dist', iou_threshold=1,
                 # nms_rescale_factor=None,
                 nms_rescale_factor=[3.5, 6, 3.5, 7.5, 4.5, 4.5, 2.5, 1.3, 1.5, 3.5],
                 ),
        # NMS type and threshold  2.83=2^1.5
        max_per_img=128),
    metric='bbox')

# Inference on val dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# Inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
# test_dataloader = dict(
#     batch_size=val_batch_size_per_gpu,
#     num_workers=val_num_workers,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path=test_data_prefix),
#         test_mode=True,
#         batch_shapes_cfg=batch_shapes_cfg,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='mmrotate.DOTAMetric',
#     format_only=True,
#     merge_patches=True,
#     outfile_prefix=submission_dir)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=5),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1,
                    save_best="NuScenes metric/pred_instances_3d_NuScenes/mAP", rule='greater'
                    ),
    visualization=dict(type='mmdet3d.Det3DVisualizationHook'))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[(max_epochs - 3, 1), ],
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='mmdet3d.Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

runner_type = 'SaveRandomStateRunner'
randomness = dict(seed=0, diff_rank_seed=True, deterministic=True)
