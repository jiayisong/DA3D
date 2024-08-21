_base_ = './rtmdet-3d_base_nus.py'
# ========================modified parameters======================
work_dir = '/mnt/jys/mmyolo/work_dirs/small_nus_19/'
# work_dir = './work_dirs/small_206/'
# =======================Unmodified in most cases==================
deepen_factor = 0.33
widen_factor = 0.5

relative_depth = True
cylinder = False
ppp = False
if ppp:
    checkpoint = './model_weight/cspnext-s_imagenet_600e_channel-4.pth'
    in_channels = 4
    batch_augments = [dict(type='CatDepthNus', img_size=_base_.img_scale), ]
else:
    checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'
    in_channels = 3
    batch_augments = []
base_depth = (30.11, 19.01)
if relative_depth:
    base_depth = (base_depth[0] / 759.63, base_depth[1] / 759.63)

model = dict(
    data_preprocessor=dict(
        batch_augments=batch_augments,
    ),
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor,
                  init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=checkpoint, map_location='cpu'),
                  input_channels=in_channels,
                  ),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(widen_factor=widen_factor, bbox_coder=dict(cylinder=cylinder, relative_depth=relative_depth,
                                                              base_depth=base_depth, )))

train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(type='LoadAnnotations3D', with_visibility=False, with_pt_num=True, with_bbox=True, with_label=True,
         with_attr_label=False, with_bbox_3d=True, with_label_3d=True, with_bbox_depth=True),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='K_out'),
    # dict(type='UnifiedIntrinsics', size=_base_.img_scale, intrinsics=intrinsics, cylinder=cylinder,
    # cycle=True, random_shift=(640, 192)
    #     ),
    dict(type='Resize3D', size=_base_.img_scale, cylinder=cylinder, ),
    # dict(type='MosaicMixUp3D', mosaic_num=(2, 2), use_cached=True, max_cached_images=10, cylinder=cylinder),
    # dict(type='RandomCrop', size=(_base_.img_scale[0], _base_.img_scale[1] * 2), random_shift=(_base_.img_scale[1], 0),
    #      cycle=True, cylinder=cylinder),
    # dict(type='Mosaic3D', mosaic_num=(2, 1), use_cached=True, max_cached_images=10, cylinder=cylinder),
    # dict(type='MixUp3D', use_cached=True, max_cached_images=10, cylinder=cylinder),
    # dict(type='CutMix3D', mix_num=2, max_cache_num=10, cylinder=cylinder),
    # dict(type='PitchCam', random_theta=10, cylinder=cylinder),
    # dict(type='RollCam', random_theta=10, cylinder=cylinder),
    # dict(type='PitchRollCam', random_theta=(2, 2), cylinder=cylinder),
    # dict(type='Update2Dattr', cylinder=cylinder),
    # dict(type='RandomResizeCrop', size=_base_.img_scale, random_scale=(0.5, 1.23), cycle=True, cylinder=cylinder),
    dict(type='FilterObject', wo_small=8, wo_ratio=0, wo_occ=0, wo_centerout=True),
    dict(type='Img2Cam'),
    # dict(type='RandomErasing', n_patches=(1, 1), ratio=(0, 0.2), border=False, fill_val=(114, 114, 114)),
    # dict(type='BackgroundNoise', prob=1),
    # dict(type='CutMix', prob=1),
    dict(type='GetTarget', relative_depth=relative_depth),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d', 'gt_labels_3d', 'target_3d'
        ],
        meta_keys=('K_out', 'img2cam', 'box_type_3d', 'catdepth')
    ),
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFileMono3D'),
    dict(type='LoadAnnotations3D', with_visibility=False, with_pt_num=True, with_bbox=True, with_label=True,
         with_attr_label=False, with_bbox_3d=True, with_label_3d=True, with_bbox_depth=True),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='K_out'),
    dict(type='Resize3D', size=_base_.img_scale, cylinder=cylinder, ),
    # dict(type='RandomResizeCrop', size=_base_.img_scale, random_scale=(0.5, 1.23), cycle=True, cylinder=cylinder,
    #      pad_val=(114, 114, 114)),
    dict(type='FilterObject', wo_small=8, wo_ratio=0, wo_occ=0, wo_centerout=True),
    dict(type='Img2Cam'),
    dict(type='GetTarget', relative_depth=relative_depth),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d', 'gt_labels_3d', 'target_3d'
        ],
        meta_keys=('K_out', 'img2cam', 'box_type_3d', 'sample_idx', 'catdepth')
    ),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(type='K_out'),
    dict(type='Resize3D', size=_base_.img_scale, cylinder=cylinder, ),
    dict(type='Img2Cam'),
    dict(
        type='Pack3DDetInputs',
        keys=['img', ],
        meta_keys=('K_out', 'img2cam', 'box_type_3d', 'sample_idx', 'catdepth')
    ),
]
val_pipeline = test_pipeline
train_dataloader = dict(dataset=dict(pipeline=train_pipeline, ))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline, ))
test_dataloader = val_dataloader
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=_base_.max_epochs - _base_.num_epochs_stage2,
        switch_pipeline=train_pipeline_stage2)
]
