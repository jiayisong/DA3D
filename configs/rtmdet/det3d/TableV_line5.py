_base_ = './rtmdet-3d_base_newloss_dla.py'


checkpoint = 'http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth'

# ========================modified parameters======================
work_dir = './mmyolo/work_dirs/TableV_line5/'
# =======================Unmodified in most cases==================

relative_depth = False
cylinder = False

if cylinder:
    intrinsics = ((881.8, 0.0, 639.5), (0.0, 801.7, 172.854), (0.0, 0.0, 1.0))
else:
    intrinsics = ((721.5377, 0.0, 639.5), (0.0, 721.5377, 172.854), (0.0, 0.0, 1.0))
base_depth = (28.01, 16.32)
if relative_depth:
    base_depth = (base_depth[0] / intrinsics[1][1], base_depth[1] / intrinsics[1][1])

model = dict(
    data_preprocessor=dict(
        batch_augments=[],
    ),
    backbone=dict(init_cfg=dict(type='Pretrained',
                                #            prefix='backbone.',
                                checkpoint=checkpoint, map_location='cpu'),
                  input_channels=3, ),
    bbox_head=dict(bbox_coder=dict(cylinder=cylinder, relative_depth=relative_depth, base_depth=base_depth, ))
)

train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(type='LoadAnnotations3D', with_bbox=True, with_label=True, with_attr_label=False, with_difficulty=False,
         with_bbox_3d=True, with_label_3d=True, with_bbox_depth=True),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='K_out'),
    dict(type='UnifiedIntrinsics', size=_base_.img_scale, intrinsics=intrinsics, cylinder=cylinder),
    # dict(type='CutMix3D', mix_num=2, max_cache_num=10, cylinder=cylinder),
    # dict(type='PitchRollCam', random_theta=(2, 2), cylinder=cylinder),
    # dict(type='Update2Dattr', cylinder=cylinder),
    # dict(type='RandomResizeCrop', size=_base_.img_scale, random_scale=(0.5, 1.23), cycle=True, cylinder=cylinder),
    dict(type='FilterObject', wo_small=8, wo_ratio=0, wo_occ=0, wo_centerout=True),
    dict(type='Img2Cam'),
    dict(type='GetTarget', relative_depth=relative_depth),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d', 'gt_labels_3d', 'target_3d'
        ],
        meta_keys=('K_out', 'img2cam', 'box_type_3d')
    ),
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFileMono3D'),
    dict(type='LoadAnnotations3D', with_bbox=True, with_label=True, with_attr_label=False, with_difficulty=False,
         with_bbox_3d=True, with_label_3d=True, with_bbox_depth=True),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='K_out'),
    dict(type='UnifiedIntrinsics', size=_base_.img_scale, intrinsics=intrinsics, pad_val=(114, 114, 114),
         cylinder=cylinder, ),
    dict(type='FilterObject', wo_small=8, wo_ratio=0, wo_occ=0, wo_centerout=True),
    dict(type='Img2Cam'),
    dict(type='GetTarget', relative_depth=relative_depth),
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
    dict(type='UnifiedIntrinsics', size=_base_.img_scale, intrinsics=intrinsics, pad_val=(114, 114, 114),
         cylinder=cylinder, ),
    dict(type='Img2Cam'),
    dict(
        type='Pack3DDetInputs',
        keys=['img', ],
        meta_keys=('K_out', 'img2cam', 'box_type_3d', 'sample_idx')
    ),
]
val_pipeline = test_pipeline
train_dataloader = dict(dataset=dict(pipeline=train_pipeline, ))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline, ))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, ))
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
