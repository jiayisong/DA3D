# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms import Compose
from mmengine.hooks import Hook
from typing import List, Sequence, Tuple, Union
from mmyolo.registry import HOOKS


@HOOKS.register_module()
class PipelineSwitchHook(Hook):
    """Switch data pipeline at switch_epoch.

    Args:
        switch_epoch (int): switch pipeline at this epoch.
        switch_pipeline (list[dict]): the pipeline to switch to.
    """

    def __init__(self, switch_epoch, switch_pipeline):
        self.switch_epoch = switch_epoch
        self.switch_pipeline = switch_pipeline
        self._restart_dataloader = False

    def before_train_epoch(self, runner):
        """switch pipeline."""
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        if epoch == self.switch_epoch:
            runner.logger.info('Switch pipeline now!')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            if hasattr(train_loader.dataset, 'datasets'):
                for i in range(len(train_loader.dataset.datasets)):
                    train_loader.dataset.datasets[i].pipeline = Compose(self.switch_pipeline)
            elif hasattr(train_loader.dataset, 'dataset'):
                train_loader.dataset.dataset.pipeline = Compose(self.switch_pipeline)
            else:
                train_loader.dataset.pipeline = Compose(self.switch_pipeline)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True

        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True
