_base_ = './TableV_line6.py'

work_dir = './work_dirs/TableVI_line1/'

train_dataloader = dict(dataset=dict(ann_file='kitti_infos_trainval.pkl',),
                        drop_last=True
                        )
train_cfg = dict(
    val_interval=5000,
    dynamic_intervals=None,
)
test_evaluator = dict(
    submission_prefix=work_dir + 'result/',
)
