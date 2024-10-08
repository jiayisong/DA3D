# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmyolo.models import PPYOLOECSPPAFPN
from mmyolo.utils import register_all_modules

register_all_modules()


class TestPPYOLOECSPPAFPN(TestCase):

    def test_forward(self):
        s = 64
        in_channels = [8, 16, 32]
        feat_sizes = [s // 2**i for i in range(4)]  # [32, 16, 8]
        out_channels = [8, 16, 32]
        feats = [
            torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
            for i in range(len(in_channels))
        ]
        neck = PPYOLOECSPPAFPN(
            in_channels=in_channels, out_channels=out_channels)
        outs = neck(feats)
        assert len(outs) == len(feats)
        for i in range(len(feats)):
            assert outs[i].shape[1] == out_channels[i]
            assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    def test_drop_block(self):
        s = 64
        in_channels = [8, 16, 32]
        feat_sizes = [s // 2**i for i in range(4)]  # [32, 16, 8]
        out_channels = [8, 16, 32]
        feats = [
            torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
            for i in range(len(in_channels))
        ]
        neck = PPYOLOECSPPAFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            drop_block_cfg=dict(
                type='mmdet.DropBlock',
                drop_prob=0.1,
                block_size=3,
                warm_iters=0))
        neck.train()
        outs = neck(feats)
        assert len(outs) == len(feats)
        for i in range(len(feats)):
            assert outs[i].shape[1] == out_channels[i]
            assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)
