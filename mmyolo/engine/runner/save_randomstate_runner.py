import random
import logging
import numpy as np
from mmyolo.registry import RUNNERS
from mmengine.runner import Runner
import pickle
import time, os
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Union
import torch
import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.device import get_device
from mmengine.dist import master_only
from mmengine.fileio import FileClient, join_path
from mmengine.model import is_model_wrapper
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)

from mmengine.utils import digit_version, get_git_hash, is_seq_of

from mmengine.runner.checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                                        find_latest_checkpoint, get_state_dict,
                                        save_checkpoint, weights_to_cpu)
from mmengine.dist import get_rank, sync_random_seed
from mmengine.logging import print_log
from mmengine.utils import digit_version, is_list_of
from mmengine.utils.dl_utils import TORCH_VERSION


@RUNNERS.register_module()
class SaveRandomStateRunner(Runner):
    # @master_only
    # def save_checkpoint(
    #         self,
    #         out_dir: str,
    #         filename: str,
    #         file_client_args: Optional[dict] = None,
    #         save_optimizer: bool = True,
    #         save_param_scheduler: bool = True,
    #         meta: dict = None,
    #         by_epoch: bool = True,
    #         backend_args: Optional[dict] = None,
    # ):
    #     """Save checkpoints.
    #     ``CheckpointHook`` invokes this method to save checkpoints
    #     periodically.
    #     Args:
    #         out_dir (str): The directory that checkpoints are saved.
    #         filename (str): The checkpoint filename.
    #         file_client_args (dict, optional): Arguments to instantiate a
    #             FileClient. See :class:`mmengine.fileio.FileClient` for
    #             details. Defaults to None. It will be deprecated in future.
    #             Please use `backend_args` instead.
    #         save_optimizer (bool): Whether to save the optimizer to
    #             the checkpoint. Defaults to True.
    #         save_param_scheduler (bool): Whether to save the param_scheduler
    #             to the checkpoint. Defaults to True.
    #         meta (dict, optional): The meta information to be saved in the
    #             checkpoint. Defaults to None.
    #         by_epoch (bool): Whether the scheduled momentum is updated by
    #             epochs. Defaults to True.
    #         backend_args (dict, optional): Arguments to instantiate the
    #             prefix of uri corresponding backend. Defaults to None.
    #             New in v0.2.0.
    #     """
    #     if meta is None:
    #         meta = {}
    #     elif not isinstance(meta, dict):
    #         raise TypeError(
    #             f'meta should be a dict or None, but got {type(meta)}')
    #
    #     if by_epoch:
    #         # self.epoch increments 1 after
    #         # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
    #         # called by `after_train_epoch`` method of `CheckpointHook` so
    #         # `epoch` should be `self.epoch + 1`
    #         meta.update(epoch=self.epoch + 1, iter=self.iter)
    #     else:
    #         meta.update(epoch=self.epoch, iter=self.iter + 1)
    #
    #     if file_client_args is not None:
    #         warnings.warn(
    #             '"file_client_args" will be deprecated in future. '
    #             'Please use "backend_args" instead', DeprecationWarning)
    #         if backend_args is not None:
    #             raise ValueError(
    #                 '"file_client_args" and "backend_args" cannot be set at '
    #                 'the same time.')
    #
    #         file_client = FileClient.infer_client(file_client_args, out_dir)
    #         filepath = file_client.join_path(out_dir, filename)
    #     else:
    #         filepath = join_path(  # type: ignore
    #             out_dir, filename, backend_args=backend_args)
    #
    #     meta.update(
    #         cfg=self.cfg.pretty_text,
    #         seed=self.seed,
    #         experiment_name=self.experiment_name,
    #         time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
    #         mmengine_version=mmengine.__version__ + get_git_hash())
    #
    #     if hasattr(self.train_dataloader.dataset, 'metainfo'):
    #         meta.update(dataset_meta=self.train_dataloader.dataset.metainfo)
    #     meta.update(
    #         random_state=(random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()))
    #     if is_model_wrapper(self.model):
    #         model = self.model.module
    #     else:
    #         model = self.model
    #
    #     checkpoint = {
    #         'meta': meta,
    #         'state_dict': weights_to_cpu(get_state_dict(model)),
    #         'message_hub': self.message_hub.state_dict()
    #     }
    #     # save optimizer state dict to checkpoint
    #     if save_optimizer:
    #         if isinstance(self.optim_wrapper, OptimWrapper):
    #             checkpoint['optimizer'] = self.optim_wrapper.state_dict()
    #         else:
    #             raise TypeError(
    #                 'self.optim_wrapper should be an `OptimWrapper` '
    #                 'or `OptimWrapperDict` instance, but got '
    #                 f'{self.optim_wrapper}')
    #
    #     # save param scheduler state dict
    #     if save_param_scheduler and self.param_schedulers is None:
    #         self.logger.warning(
    #             '`save_param_scheduler` is True but `self.param_schedulers` '
    #             'is None, so skip saving parameter schedulers')
    #         save_param_scheduler = False
    #     if save_param_scheduler:
    #         if isinstance(self.param_schedulers, dict):
    #             checkpoint['param_schedulers'] = dict()
    #             for name, schedulers in self.param_schedulers.items():
    #                 checkpoint['param_schedulers'][name] = []
    #                 for scheduler in schedulers:
    #                     state_dict = scheduler.state_dict()
    #                     checkpoint['param_schedulers'][name].append(state_dict)
    #         else:
    #             checkpoint['param_schedulers'] = []
    #             for scheduler in self.param_schedulers:  # type: ignore
    #                 state_dict = scheduler.state_dict()  # type: ignore
    #                 checkpoint['param_schedulers'].append(state_dict)
    #
    #     self.call_hook('before_save_checkpoint', checkpoint=checkpoint)
    #     save_checkpoint(checkpoint, filepath)
    #
    # def resume(self,
    #            filename: str,
    #            resume_optimizer: bool = True,
    #            resume_param_scheduler: bool = True,
    #            map_location: Union[str, Callable] = 'default') -> None:
    #     """Resume model from checkpoint.
    #     Args:
    #         filename (str): Accept local filepath, URL, ``torchvision://xxx``,
    #             ``open-mmlab://xxx``.
    #         resume_optimizer (bool): Whether to resume optimizer state.
    #             Defaults to True.
    #         resume_param_scheduler (bool): Whether to resume param scheduler
    #             state. Defaults to True.
    #         map_location (str or callable):A string or a callable function to
    #             specifying how to remap storage locations.
    #             Defaults to 'default'.
    #     """
    #     if map_location == 'default':
    #         device = get_device()
    #         checkpoint = self.load_checkpoint(filename, map_location=device)
    #     else:
    #         checkpoint = self.load_checkpoint(
    #             filename, map_location=map_location)
    #
    #     self.train_loop._epoch = checkpoint['meta']['epoch']
    #     self.train_loop._iter = checkpoint['meta']['iter']
    #
    #     # check whether the number of GPU used for current experiment
    #     # is consistent with resuming from checkpoint
    #     if 'config' in checkpoint['meta']:
    #         config = mmengine.Config.fromstring(
    #             checkpoint['meta']['config'], file_format='.py')
    #         previous_gpu_ids = config.get('gpu_ids', None)
    #         if (previous_gpu_ids is not None and len(previous_gpu_ids) > 0
    #                 and len(previous_gpu_ids) != self._world_size):
    #             # TODO, should we modify the iteration?
    #             self.logger.info(
    #                 'Number of GPU used for current experiment is not '
    #                 'consistent with resuming from checkpoint')
    #             if (self.auto_scale_lr is None
    #                     or not self.auto_scale_lr.get('enable', False)):
    #                 raise RuntimeError(
    #                     'Cannot automatically rescale lr in resuming. Please '
    #                     'make sure the number of GPU is consistent with the '
    #                     'previous training state resuming from the checkpoint '
    #                     'or set `enable` in `auto_scale_lr to False.')
    #
    #     # resume random seed
    #     resumed_seed = checkpoint['meta'].get('seed', None)
    #     current_seed = self._randomness_cfg.get('seed')
    #     if resumed_seed is not None and resumed_seed != current_seed:
    #         if current_seed is not None:
    #             self.logger.warning(f'The value of random seed in the '
    #                                 f'checkpoint "{resumed_seed}" is '
    #                                 f'different from the value in '
    #                                 f'`randomness` config "{current_seed}"')
    #         self._randomness_cfg.update(seed=resumed_seed)
    #         self.set_randomness(**self._randomness_cfg)
    #
    #     resumed_dataset_meta = checkpoint['meta'].get('dataset_meta', None)
    #     dataset_meta = getattr(self.train_dataloader.dataset, 'metainfo', None)
    #
    #     # `resumed_dataset_meta` and `dataset_meta` could be object like
    #     # np.ndarray, which cannot be directly judged as equal or not,
    #     # therefore we just compared their dumped results.
    #     if pickle.dumps(resumed_dataset_meta) != pickle.dumps(dataset_meta):
    #         self.logger.warning(
    #             'The dataset metainfo from the resumed checkpoint is '
    #             'different from the current training dataset, please '
    #             'check the correctness of the checkpoint or the training '
    #             'dataset.')
    #
    #     self.message_hub.load_state_dict(checkpoint['message_hub'])
    #
    #     # resume optimizer
    #     if 'optimizer' in checkpoint and resume_optimizer:
    #         self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
    #         self.optim_wrapper.load_state_dict(  # type: ignore
    #             checkpoint['optimizer'])
    #
    #     # resume param scheduler
    #     if resume_param_scheduler and self.param_schedulers is None:
    #         self.logger.warning(
    #             '`resume_param_scheduler` is True but `self.param_schedulers` '
    #             'is None, so skip resuming parameter schedulers')
    #         resume_param_scheduler = False
    #     if 'param_schedulers' in checkpoint and resume_param_scheduler:
    #         self.param_schedulers = self.build_param_scheduler(  # type: ignore
    #             self.param_schedulers)  # type: ignore
    #         if isinstance(self.param_schedulers, dict):
    #             for name, schedulers in self.param_schedulers.items():
    #                 for scheduler, ckpt_scheduler in zip(
    #                         schedulers, checkpoint['param_schedulers'][name]):
    #                     scheduler.load_state_dict(ckpt_scheduler)
    #         else:
    #             for scheduler, ckpt_scheduler in zip(
    #                     self.param_schedulers,  # type: ignore
    #                     checkpoint['param_schedulers']):
    #                 scheduler.load_state_dict(ckpt_scheduler)
    #
    #     self._has_loaded = True
    #
    #     self.logger.info(f'resumed epoch: {self.epoch}, iter: {self.iter}')
    #
    #     # random_state, np_random_state, torch_random_state, torch_cuda_random_state = checkpoint['meta'].pop(
    #     #     'random_state')
    #     # torch_random_state = torch_random_state.cpu()
    #     # torch_cuda_random_state = torch_cuda_random_state.cpu()
    #     # random.setstate(random_state)
    #     # np.random.set_state(np_random_state)
    #     # torch.set_rng_state(torch_random_state)
    #     # torch.cuda.set_rng_state(torch_cuda_random_state)

    def set_randomness(self,
                       seed,
                       diff_rank_seed: bool = False,
                       deterministic: bool = False) -> None:
        """Set random seed to guarantee reproducible results.

        Args:
            seed (int): A number to set random modules.
            diff_rank_seed (bool): Whether or not set different seeds according
                to global rank. Defaults to False.
            deterministic (bool): Whether to set the deterministic option for
                CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
                to True and `torch.backends.cudnn.benchmark` to False.
                Defaults to False.
                See https://pytorch.org/docs/stable/notes/randomness.html for
                more details.
        """
        self._deterministic = deterministic
        if seed is None:
            seed = sync_random_seed()

        if diff_rank_seed:
            rank = get_rank()
            seed += rank

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)
        if deterministic:
            if torch.backends.cudnn.benchmark:
                print_log(
                    'torch.backends.cudnn.benchmark is going to be set as '
                    '`False` to cause cuDNN to deterministically select an '
                    'algorithm',
                    logger='current',
                    level=logging.WARNING)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            if digit_version(TORCH_VERSION) >= digit_version('1.10.0'):
                torch.use_deterministic_algorithms(True)
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        self._seed = seed
