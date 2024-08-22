_base_ = 'rtmdet-3d_base_dla.py'

base_lr = 0.00002
max_epochs = 60

# The model is too large and requires dual GPU training, so the batch size is 4.
train_dataloader = dict(
    batch_size=4,
    num_workers=2, )

# ===============================Unmodified in most cases====================
model = dict(
    backbone=dict(
        _delete_=True,
        type='VoVNet',
        depth='V-99-eSE',
        input_channels=3,
        norm_cfg=dict(type='BN'),
    ),
    neck=dict(
        in_channels=[32, 64, 256, 512, 768, 1024],
    ),
    bbox_head=dict(
        widen_factor=1.0,
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
    # dict(
    #     type='LinearLR',
    #     start_factor=0.05,
    #     by_epoch=True,
    #     convert_to_iter_based=True,
    #     begin=0,
    #     end=max_epochs // 2),
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

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=500,
    # dynamic_intervals=[(max_epochs - 10, 1), ],
)


