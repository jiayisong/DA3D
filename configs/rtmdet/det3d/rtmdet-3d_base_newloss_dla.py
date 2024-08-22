_base_ = 'rtmdet-3d_base_newloss.py'


strides = [4, ]
base_lr = 0.0001
max_epochs = 125

# ===============================Unmodified in most cases====================
model = dict(
    backbone=dict(
        _delete_=True,
        type='DLANet',
        depth=34,
        input_channels=3,
        norm_cfg=dict(type='BN'),
    ),
    neck=dict(
        _delete_=True,
        type='mmdet3d.DLANeck',
        in_channels=[16, 32, 64, 128, 256, 512],
        start_level=2,
        end_level=5,
        norm_cfg=dict(type='BN')),
    bbox_head=dict(
        prior_generator=dict(strides=strides),
        widen_factor=0.25,
        strides=strides,
    )
)
optim_wrapper = dict(optimizer=dict(lr=base_lr), )

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
