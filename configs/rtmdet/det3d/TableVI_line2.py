_base_ = './TableVI_line1.py'

model = dict(
    test_cfg=dict(
        dm3d=dict(lamda=40, depth_offset=(-2, -1, -0.5, 0, 0.5, 1, 2)),
    ) 
)
