# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, Optional

import torch
from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.codebase.mmdet.deploy import ObjectDetection
from mmdeploy.utils import Codebase, Task
from mmengine import Config
from mmengine.registry import Registry

MMYOLO_TASK = Registry('mmyolo_tasks')


@CODEBASE.register_module(Codebase.MMYOLO.value)
class MMYOLO(MMCodebase):
    """MMYOLO codebase class."""

    task_registry = MMYOLO_TASK

    @classmethod
    def register_deploy_modules(cls):
        """register all rewriters for mmdet."""
        import mmdeploy.codebase.mmdet.models  # noqa: F401
        import mmdeploy.codebase.mmdet.ops  # noqa: F401
        import mmdeploy.codebase.mmdet.structures  # noqa: F401

    @classmethod
    def register_all_modules(cls):
        """register all modules."""
        from mmdet.utils.setup_env import \
            register_all_modules as register_all_modules_mmdet

        from mmyolo.utils.setup_env import \
            register_all_modules as register_all_modules_mmyolo

        cls.register_deploy_modules()
        register_all_modules_mmyolo(True)
        register_all_modules_mmdet(False)


def _get_dataset_metainfo(model_cfg: Config):
    """Get metainfo of dataset.

    Args:
        model_cfg Config: Input model Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
    """
    from mmyolo import datasets  # noqa
    from mmyolo.registry import DATASETS

    module_dict = DATASETS.module_dict
    for dataloader_name in [
            'test_dataloader', 'val_dataloader', 'train_dataloader'
    ]:
        if dataloader_name not in model_cfg:
            continue
        dataloader_cfg = model_cfg[dataloader_name]
        dataset_cfg = dataloader_cfg.dataset
        dataset_cls = module_dict.get(dataset_cfg.type, None)
        if dataset_cls is None:
            continue
        if hasattr(dataset_cls, '_load_metainfo') and isinstance(
                dataset_cls._load_metainfo, Callable):
            meta = dataset_cls._load_metainfo(
                dataset_cfg.get('metainfo', None))
            if meta is not None:
                return meta
        if hasattr(dataset_cls, 'METAINFO'):
            return dataset_cls.METAINFO

    return None


@MMYOLO_TASK.register_module(Task.OBJECT_DETECTION.value)
class YOLOObjectDetection(ObjectDetection):
    """YOLO Object Detection task."""

    def get_visualizer(self, name: str, save_dir: str):
        """Get visualizer.

        Args:
            name (str): Name of visualizer.
            save_dir (str): Directory to save visualization results.

        Returns:
            Visualizer: A visualizer instance.
        """
        from mmdet.visualization import DetLocalVisualizer  # noqa: F401,F403
        metainfo = _get_dataset_metainfo(self.model_cfg)
        visualizer = super().get_visualizer(name, save_dir)
        if metainfo is not None:
            visualizer.dataset_meta = metainfo
        return visualizer

    def build_pytorch_model(self,
                            model_checkpoint: Optional[str] = None,
                            cfg_options: Optional[Dict] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize torch model.

        Args:
            model_checkpoint (str): The checkpoint file of torch model,
                defaults to `None`.
            cfg_options (dict): Optional config key-pair parameters.
        Returns:
            nn.Module: An initialized torch model generated by other OpenMMLab
                codebases.
        """
        from copy import deepcopy

        from mmengine.model import revert_sync_batchnorm
        from mmengine.registry import MODELS

        from mmyolo.utils import switch_to_deploy

        model = deepcopy(self.model_cfg.model)
        preprocess_cfg = deepcopy(self.model_cfg.get('preprocess_cfg', {}))
        preprocess_cfg.update(
            deepcopy(self.model_cfg.get('data_preprocessor', {})))
        model.setdefault('data_preprocessor', preprocess_cfg)
        model = MODELS.build(model)
        if model_checkpoint is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(model, model_checkpoint, map_location=self.device)

        model = revert_sync_batchnorm(model)
        switch_to_deploy(model)
        model = model.to(self.device)
        model.eval()
        return model
