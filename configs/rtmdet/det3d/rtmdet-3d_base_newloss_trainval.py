_base_ = '../../_base_/default_runtime.py'

# checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth'  # noqa

# ========================Frequently modified parameters======================

work_dir = '/mnt/jys/mmyolo/work_dirs/large_1/'

class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=False, use_camera=True)
metainfo = dict(classes=class_names)
num_classes = len(class_names)  # Number of classes for classification
# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 1xb8=8 bs
base_lr = 0.00025  # 0.004 / 16
max_epochs = 125  # Maximum training epochs
num_epochs_stage2 = 20
loss_cls_weight = 1.0
loss_bbox_weight = 1.0
# ========================Possible modified parameters========================
# -----data related-----
img_scale = (384, 1280)  # height  width
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5KittiDataset'
data_root = '/home/jys/DataSets/kitti/'
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
        batch_augments=[dict(type='CatDepth', img_size=img_scale, d_max=50, cv=172.854, ), ],
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
        group_reg_dims=(2, 1, 3, 32),  # offset, depth, size, rot, velo
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
            # base_offset=(0, 1),
            base_offset=(0.50, 0.29),
            base_dims=(((0.84, 1.76, 0.66), (1.76, 1.74, 0.60), (3.88, 1.53, 1.63)),
                       #     ((0.2, 0.2, 0.2), (0.2, 0.2, 0.2), (0.2, 0.2, 0.2)
                       ((0.33, 0.07, 0.22), (0.10, 0.06, 0.22), (0.11, 0.09, 0.06)
                        )), ),
        loss_cls=dict(type='mmdet.QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=loss_cls_weight),
        loss_offset=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 1.0),
        loss_depth=dict(type='SimpleUncertainL1Loss', loss_weight=loss_bbox_weight * 1.0),
        loss_dim=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 1.0),
        loss_dir=dict(type='MultiBinLoss', bin_margin=1 / 3, loss_weight=loss_bbox_weight * 1.0),
        # loss_dir=dict(type='OneBinLoss', loss_weight=loss_bbox_weight * 0.0),
        # loss_offset_corner=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 4),
        # loss_depth_corner=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 0.2),
        # loss_dim_corner=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 1),
        # loss_dir_corner=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 0.2),
        # loss_corner=None,
        loss_corner=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight * 0.2),
        # loss_velo=dict(type='mmdet.L1Loss', loss_weight=loss_bbox_weight*1),
        # loss_attr=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=loss_cls_weight),
    ),
    train_cfg=dict(
        assigner=dict(
            # type='IOUAssigner3D',
            type='BatchDynamicSoftLabelAssigner3D',
            num_classes=num_classes,
            topk=13,
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        # The config of multi-label for multi-class prediction.
        multi_label=True,
        # The number of boxes before NMS
        nms_pre=512,
        score_thr=0.00,  # Threshold to filter out boxes.
        nms=dict(type='3d', iou_threshold=0.7),  # NMS type and threshold
        max_per_img=32)  # Max number of detections of each image,
)

train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='mmdet3d.LoadAnnotations3D', with_difficulty=False,
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='K_out'),
    dict(type='CylinderUnifiedIntrinsics', size=img_scale,
         intrinsics=((881.8, 0.0, 639.5), (0.0, 801.7, 172.854), (0.0, 0.0, 1.0)), pad_val=(114, 114, 114),
         random_shift=(639.5, 0), random_scale=(1, 1), cycle=True),
    # dict(type='UnifiedIntrinsics', size=img_scale,
    #      intrinsics=((721.5377, 0.0, 639.5), (0.0, 721.5377, 172.854), (0.0, 0.0, 1.0)), pad_val=(114, 114, 114),
    #      random_shift=(0, 191.5), random_scale=(1, 1), cycle=True),
    # dict(type='Pad', size=IMG_SIZE),
    dict(type='GetTarget'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d', 'gt_labels_3d', 'target_3d'
        ],
        meta_keys=('K_out', 'img2cam', 'box_type_3d')
    ),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(type='K_out'),
    dict(type='CylinderUnifiedIntrinsics', size=img_scale,
         intrinsics=((881.8, 0.0, 639.5), (0.0, 801.7, 172.854), (0.0, 0.0, 1.0)), pad_val=(114, 114, 114), ),
    # dict(type='UnifiedIntrinsics', size=img_scale,
    #      intrinsics=((721.5377, 0.0, 639.5), (0.0, 721.5377, 172.854), (0.0, 0.0, 1.0))),
    # dict(type='Pad', size=IMG_SIZE),
    dict(type='Img2Cam'),
    dict(
        type='Pack3DDetInputs',
        keys=['img', ],
        meta_keys=('K_out', 'img2cam', 'box_type_3d', 'sample_idx')
    ),
]
val_pipeline = test_pipeline

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    collate_fn=dict(type='RTMDet3D_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    # persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='new/kitti_infos_trainval.pkl',
        data_prefix=dict(img='training/image_2'),
        pipeline=train_pipeline,
        modality=input_modality,
        load_type='fov_image_based',
        test_mode=False,
        metainfo=metainfo,
        # we use box_type_3d='Camera' in monocular 3d
        # detection task
        box_type_3d='Camera'))

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    collate_fn=dict(type='RTMDet3D_collate', train=False),
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='training/image_2'),
        ann_file='new/kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        load_type='fov_image_based',
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Camera'))

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'new/kitti_infos_val.pkl',
    metric=['3d', 'bev'],
    pred_box_type_3d='Camera')

# Inference on val dataset
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    collate_fn=dict(type='RTMDet3D_collate', train=False),
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        load_eval_anns=False,
        data_prefix=dict(img='testing/image_2'),
        ann_file='new/kitti_infos_test.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        load_type='fov_image_based',
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Camera'))
test_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'new/kitti_infos_test.pkl',
    metric=['3d', 'bev'],
    format_only=True,
    submission_prefix='/mnt/jys/mmyolo/work_dirs/small_259/result/',
    pred_box_type_3d='Camera')

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
                    save_best="Kitti metric/pred_instances_3d/KITTI/Car_3D_AP40_moderate_strict", rule='greater'
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
    dynamic_intervals=[(max_epochs - 10, 1), ],
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='mmdet3d.Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

runner_type = 'SaveRandomStateRunner'
randomness = dict(seed=0, diff_rank_seed=True, deterministic=True)
