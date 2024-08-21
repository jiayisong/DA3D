# Copyright (c) OpenMMLab. All rights reserved.
import logging, sys, os
import tempfile
from math import sqrt
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
from mmengine.logging import MMLogger, print_log
import mmengine
import numpy as np
import pyquaternion
import torch
from mmengine import Config, load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.utils.data_classes import Box as NuScenesBox
from terminaltables import AsciiTable
from mmdet3d.models.layers import box3d_multiclass_nms
from mmyolo.registry import METRICS
from mmdet3d.structures import (CameraInstance3DBoxes, LiDARInstance3DBoxes,
                                bbox3d2result, xywhr2xyxyr)
from cv_ops.bbox3d import nms_3d_gpu, nms_bev_gpu, nms_dist_gpu
import random
import time
import json
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import calc_ap, calc_tp, accumulate
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)


@METRICS.register_module()
class NuScenesMetric(BaseMetric):
    """Nuscenes evaluation metric.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metric (str | list[str]): Metrics to be evaluated.
            Default to 'bbox'.
        modality (dict): Modality to specify the sensor data used
            as input. Defaults to dict(use_camera=False, use_lidar=True).
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        jsonfile_prefix (str, optional): The prefix of json files including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        eval_version (str): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        collect_device (str): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }

    def __init__(
            self,
            data_root: str,
            ann_file: str,
            metric: Union[str, List[str]] = 'bbox',
            modality: Dict = dict(use_camera=False, use_lidar=True),
            prefix: Optional[str] = None,
            jsonfile_prefix: Optional[str] = None,
            eval_version: str = 'detection_cvpr_2019',
            collect_device: str = 'cpu',
            file_client_args: dict = dict(backend='disk'),
            test_cfg=None,
    ) -> None:
        self.default_prefix = 'NuScenes metric'
        super(NuScenesMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        if modality is None:
            modality = dict(
                use_camera=False,
                use_lidar=True,
            )
        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.jsonfile_prefix = jsonfile_prefix
        self.file_client_args = file_client_args
        self.test_cfg = test_cfg
        self.metrics = metric if isinstance(metric, list) else [metric]

        self.eval_version = eval_version
        self.eval_detection_configs = config_factory(self.eval_version)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            # for attr_name in pred_3d:
            #     pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d  # .to('cpu')
            # for attr_name in pred_2d:
            #     pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d  # .to('cpu')
            result['sample_idx'] = data_sample['sample_idx']
            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        classes = self.dataset_meta['classes']
        self.version = self.dataset_meta['version']
        # load annotations
        self.data_infos = load(
            self.ann_file, file_client_args=self.file_client_args)['data_list']
        result_dict, tmp_dir = self.format_results(results, classes, self.jsonfile_prefix)

        metric_dict = {}
        for metric in self.metrics:
            ap_dict = self.nus_evaluate(
                result_dict, classes=classes, metric=metric, logger=logger)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        results = collect_results(self.results, size, 'gpu')

        if is_main_process():
            # cast all tensors in results list to cpu
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]

    def nus_evaluate(self,
                     result_dict: dict,
                     metric: str = 'bbox',
                     classes: List[str] = None,
                     logger: logging.Logger = None) -> dict:
        """Evaluation in Nuscenes protocol.

        Args:
            result_dict (dict): Formatted results of the dataset.
            metric (str): Metrics to be evaluated.
                Default: None.
            classes (list[String], optional): A list of class name. Defaults
                to None.
            logger (MMLogger, optional): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        metric_dict = dict()
        for name in result_dict:
            print(f'Evaluating bboxes of {name}')
            ret_dict = self._evaluate_single(
                result_dict[name], classes=classes, result_name=name, logger=logger)
        metric_dict.update(ret_dict)
        return metric_dict

    def _evaluate_single(self,
                         result_path: str,
                         classes: List[None] = None,
                         result_name: str = 'pred_instances_3d', logger=None) -> dict:
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
                Default: 'bbox'.
            classes (list[String], optional): A list of class name. Defaults
                to None.
            result_name (str): Result name in the metric prefix.
                Default: 'pred_instances_3d'.

        Returns:
            dict: Dictionary of evaluation details.
        """

        # output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            verbose=True)

        metrics = nusc_eval.main(render_curves=False)
        detail = dict()

        # Print high-level metrics.

        print_log('Eval time: %.1fs' % metrics['eval_time'], logger=logger)

        table_data = [['Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE', 'NDS'], ]

        class_aps = metrics['mean_dist_aps']
        class_tps = metrics['label_tp_errors']
        for class_name in class_aps.keys():
            cls_data = [class_name,
                        f'{class_aps[class_name]:.3f}',
                        f'{class_tps[class_name]["trans_err"]:.3f}',
                        f'{class_tps[class_name]["scale_err"]:.3f}',
                        f'{class_tps[class_name]["orient_err"]:.3f}',
                        f'{class_tps[class_name]["vel_err"]:.3f}',
                        f'{class_tps[class_name]["attr_err"]:.3f}',
                        '-', ]
            table_data.append(cls_data)
        all_cls_data = ['-',
                        f'{metrics["mean_ap"]:.3f}',
                        f'{metrics["tp_errors"]["trans_err"]:.3f}',
                        f'{metrics["tp_errors"]["scale_err"]:.3f}',
                        f'{metrics["tp_errors"]["orient_err"]:.3f}',
                        f'{metrics["tp_errors"]["vel_err"]:.3f}',
                        f'{metrics["tp_errors"]["attr_err"]:.3f}',
                        f'{metrics["nd_score"]:.3f}', ]
        table_data.append(all_cls_data)
        table = AsciiTable(table_data)
        print_log('\n' + table.table, logger=logger)

        metric_prefix = f'{result_name}_NuScenes'
        for name in classes:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail[f'{metric_prefix}/{name}_{k}'] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = val

        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
        return detail

    # @profile
    def format_results(self,
                       results: List[dict],
                       classes: List[str] = None,
                       jsonfile_prefix: str = None) -> Tuple:
        """Format the mmdet3d results to standard NuScenes json file.

        Args:
            results (list[dict]): Testing results of the dataset.
            classes (list[String], optional): A list of class name. Defaults
                to None.
            jsonfile_prefix (str, optional): The prefix of json files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: Returns (result_dict, tmp_dir), where `result_dict` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'

        tmp_dir = None
        result_dict = dict()
        sample_id_list = [result['sample_idx'] for result in results]

        for name in results[0]:
            if 'pred' in name and '3d' in name and name[0] != '_':
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                if jsonfile_prefix is not None:
                    tmp_file_ = osp.join(jsonfile_prefix, name)
                else:
                    tmp_file_ = None
                box_type_3d = type(results_[0]['bboxes_3d'])
                if box_type_3d == LiDARInstance3DBoxes:
                    result_dict[name] = self._format_lidar_bbox(
                        results_, sample_id_list, classes, tmp_file_)
                elif box_type_3d == CameraInstance3DBoxes:
                    result_dict[name] = self._format_camera_bbox(
                        results_, sample_id_list, classes, tmp_file_)

        return result_dict, tmp_dir

    def get_attr_name(self, attr_idx, label_name):
        """Get attribute from predicted index.

        This is a workaround to predict attribute when the predicted velocity
        is not reliable. We map the predicted attribute index to the one
        in the attribute set. If it is consistent with the category, we will
        keep it. Otherwise, we will use the default attribute.

        Args:
            attr_idx (int): Attribute index.
            label_name (str): Predicted category name.

        Returns:
            str: Predicted attribute name.
        """
        # TODO: Simplify the variable name
        AttrMapping_rev2 = [
            'cycle.with_rider', 'cycle.without_rider', 'pedestrian.moving',
            'pedestrian.standing', 'pedestrian.sitting_lying_down',
            'vehicle.moving', 'vehicle.parked', 'vehicle.stopped', 'None'
        ]
        if label_name == 'car' or label_name == 'bus' \
                or label_name == 'truck' or label_name == 'trailer' \
                or label_name == 'construction_vehicle':
            if AttrMapping_rev2[attr_idx] == 'vehicle.moving' or \
                    AttrMapping_rev2[attr_idx] == 'vehicle.parked' or \
                    AttrMapping_rev2[attr_idx] == 'vehicle.stopped':
                return AttrMapping_rev2[attr_idx]
            else:
                return self.DefaultAttribute[label_name]
        elif label_name == 'pedestrian':
            if AttrMapping_rev2[attr_idx] == 'pedestrian.moving' or \
                    AttrMapping_rev2[attr_idx] == 'pedestrian.standing' or \
                    AttrMapping_rev2[attr_idx] == \
                    'pedestrian.sitting_lying_down':
                return AttrMapping_rev2[attr_idx]
            else:
                return self.DefaultAttribute[label_name]
        elif label_name == 'bicycle' or label_name == 'motorcycle':
            if AttrMapping_rev2[attr_idx] == 'cycle.with_rider' or \
                    AttrMapping_rev2[attr_idx] == 'cycle.without_rider':
                return AttrMapping_rev2[attr_idx]
            else:
                return self.DefaultAttribute[label_name]
        else:
            return self.DefaultAttribute[label_name]

    # @profile
    def _format_camera_bbox(self,
                            results: List[dict],
                            sample_id_list: List[int],
                            classes: List[str] = None,
                            jsonfile_prefix: str = None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}

        print('Start to convert detection format...')

        # Camera types in Nuscenes datasets
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]

        CAM_NUM = 6
        boxes_per_frame = []
        for i, det in enumerate(mmengine.track_iter_progress(results)):

            sample_id = sample_id_list[i]

            frame_sample_id = sample_id // CAM_NUM
            camera_type_id = sample_id % CAM_NUM

            if camera_type_id == 0:
                boxes_per_frame = []
            # need to merge results from images of the same sample

            camera_type = camera_types[camera_type_id]
            det = output_to_global_nusc(det, self.data_infos[frame_sample_id], camera_type)

            boxes_per_frame.append(det)

            # Remove redundant predictions caused by overlap of images
            if (sample_id + 1) % CAM_NUM != 0:
                continue
            dets = {}
            for n in boxes_per_frame[0].keys():
                if n != 'metainfo':
                    dets[n] = torch.cat([i[n] for i in boxes_per_frame], dim=0)
            nusc_anno = merge_global_box(dets, self.test_cfg)
            sample_token = self.data_infos[frame_sample_id]['token']
            annos = []
            for i, box in enumerate(nusc_anno):
                name = classes[box.pop('label')]
                attr = self.get_attr_name(box.pop('attribute'), name)
                box.update(dict(
                    sample_token=sample_token,
                    detection_name=name,
                    attribute_name=attr))
                annos.append(box)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        if jsonfile_prefix is not None:
            mmengine.mkdir_or_exist(jsonfile_prefix)
            res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
            print('Results are writing to', res_path)
            mmengine.dump(nusc_submissions, res_path)
            print('Write finish', res_path)
            return res_path
        else:
            return nusc_submissions

    def _format_lidar_bbox(self,
                           results: List[dict],
                           sample_id_list: List[int],
                           classes: List[str] = None,
                           jsonfile_prefix: str = None) -> str:
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            sample_id_list (list[int]): List of result sample id.
            classes (list[String], optional): A list of class name. Defaults
                to None.
            jsonfile_prefix (str, optional): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}

        print('Start to convert detection format...')
        for i, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes, attrs = output_to_nusc_box(det)
            sample_id = sample_id_list[i]
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             classes,
                                             self.eval_detection_configs)
            for i, box in enumerate(boxes):
                name = classes[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        'car',
                        'construction_vehicle',
                        'bus',
                        'truck',
                        'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmengine.dump(nusc_submissions, res_path)
        return res_path


# @profile
def output_to_global_nusc(detection, info: dict, camera_type: str = 'CAM_FRONT', ):
    """Convert the box from camera to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        attrs (list[str]): List of attributes.
        camera_type (str): Type of camera.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    bbox3d = detection['bboxes_3d']
    box_gravity_center = bbox3d.gravity_center  # .numpy()
    box_dims = bbox3d.dims  # .numpy()
    box_yaw = bbox3d.yaw  # .numpy()
    if type(bbox3d) == CameraInstance3DBoxes:
        # our Camera coordinate system -> nuScenes box coordinate system
        # convert the dim/rot to nuscbox convention
        nus_box_dims = box_dims[:, [2, 0, 1]]
        cam2ego = torch.tensor(info['images'][camera_type]['cam2ego'], device=box_gravity_center.device)
        ego2global = torch.tensor(info['images'][camera_type]['ego2global'], device=box_gravity_center.device)
        cam2global = torch.matmul(ego2global, cam2ego)
        box_center = torch.matmul(cam2global[None, :3, :3], box_gravity_center.unsqueeze(-1)).squeeze(-1) + cam2global[
                                                                                                            None, :3, 3]
        cosi = torch.cos(box_yaw)
        sini = torch.sin(box_yaw)
        R0 = torch.stack([cosi, torch.zeros_like(box_yaw), -sini], dim=1).unsqueeze(-1)
        R1 = torch.matmul(cam2global[None, :3, :3], R0)
        if bbox3d.tensor.shape[1] == 9:
            velocity = torch.stack([bbox3d.tensor[:, 7], torch.zeros_like(box_yaw), bbox3d.tensor[:, 8]],
                                   dim=1).unsqueeze(-1)
            velocity = torch.matmul(cam2global[None, :3, :3], velocity)
        else:
            velocity = torch.zeros_like(box_center)
        theta = torch.atan2(R1[:, 1, :], R1[:, 0, :])
        box_3d_new = torch.cat([box_center, nus_box_dims, theta], dim=1)
        detection['bboxes_3d'] = box_3d_new
        detection['velocity'] = velocity
    else:
        raise NotImplementedError(
            f'Do not support convert {type(bbox3d)} bboxes'
            'to standard NuScenesBoxes.')
    return detection


# @profile
def merge_global_box(detection, nms_cfg):
    bbox3d = detection['bboxes_3d']
    scores = detection['scores_3d']
    labels = detection['labels_3d']
    bboxes3d = bbox3d[:, [1, 2, 0, 4, 5, 3, 6]]

    # bboxes3d_debug = bboxes3d[scores > 0.5]
    # bboxes3d_debug = bboxes3d_debug[:, [0, 2]]
    # mat_debug = torch.norm(bboxes3d_debug[:, None, :] - bboxes3d_debug[None, :, :], dim=2, keepdim=False).cpu().numpy()
    # bboxes3d_debug = bboxes3d_debug.cpu().numpy()

    if 'attrs_3d' not in detection:
        detection['attrs_3d'] = torch.zeros_like(labels)
    if hasattr(nms_cfg, 'nms'):
        if nms_cfg.nms.type == '3d':
            keep = nms_3d_gpu(bboxes3d, scores, labels, nms_cfg.nms.iou_threshold,
                              nms_rescale_factor=nms_cfg.nms.nms_rescale_factor, fol_maxsize=nms_cfg.max_per_img)
        elif nms_cfg.nms.type == 'bev':
            keep = nms_bev_gpu(bboxes3d, scores, labels, nms_cfg.nms.iou_threshold,
                               nms_rescale_factor=nms_cfg.nms.nms_rescale_factor, fol_maxsize=nms_cfg.max_per_img)
        elif nms_cfg.nms.type == 'dist':
            keep = nms_dist_gpu(bboxes3d, scores, labels, nms_cfg.nms.iou_threshold,
                                nms_rescale_factor=nms_cfg.nms.nms_rescale_factor, fol_maxsize=nms_cfg.max_per_img)
        else:
            raise NotImplementedError
        for n in detection.keys():
            detection[n] = detection[n][keep]
    elif hasattr(nms_cfg, 'max_per_img'):
        keep = scores.sort(0, descending=True)[1]
        keep = keep[:nms_cfg.max_per_img]
        for n in detection.keys():
            detection[n] = detection[n][keep]
    quat = detection['bboxes_3d'][:, 6] / 2
    cosi = torch.cos(quat)
    sini = torch.sin(quat)
    quat = torch.stack([cosi, torch.zeros_like(cosi), torch.zeros_like(cosi), sini], dim=1)
    nusc_anno = dict(
        translation=detection['bboxes_3d'][:, :3].cpu().numpy(),
        size=detection['bboxes_3d'][:, 3:6].cpu().numpy(),
        rotation=quat.cpu().numpy(),
        velocity=detection['velocity'][:, :2].cpu().numpy(),
        label=detection['labels_3d'].cpu().numpy(),
        detection_score=detection['scores_3d'].cpu().numpy(),
        attribute=detection['attrs_3d'].cpu().numpy(), )
    nusc_anno_list = []
    for i in range(quat.shape[0]):
        nusc_anno_list.append(dict(
            translation=nusc_anno['translation'][i].tolist(),
            size=nusc_anno['size'][i].tolist(),
            rotation=nusc_anno['rotation'][i].tolist(),
            velocity=nusc_anno['velocity'][i].tolist(),
            label=nusc_anno['label'][i].tolist(),
            detection_score=nusc_anno['detection_score'][i].tolist(),
            attribute=nusc_anno['attribute'][i].tolist(),
        ))
    return nusc_anno_list


# @profile
def output_to_global_nusc_box(detection,
                              info: dict,
                              classes: List[str],
                              eval_configs: DetectionConfig,
                              camera_type: str = 'CAM_FRONT', score_thd=-1,
                              ) -> List[NuScenesBox]:
    """Convert the box from camera to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        attrs (list[str]): List of attributes.
        camera_type (str): Type of camera.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    bbox3d = detection['bboxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    if 'attrs_3d' in detection:
        attr_list = detection['attrs_3d'].numpy()
    else:
        attr_list = np.zeros_like(labels, dtype=np.int32)
    box_gravity_center = bbox3d.gravity_center.numpy()
    box_dims = bbox3d.dims.numpy()
    box_yaw = bbox3d.yaw.numpy()

    nusc_anno_list = []

    if type(bbox3d) == LiDARInstance3DBoxes:
        # our LiDAR coordinate system -> nuScenes box coordinate system
        raise NotImplementedError(
            f'Do not support convert {type(bbox3d)} bboxes'
            'to standard NuScenesBoxes.')
        nus_box_dims = box_dims[:, [1, 0, 2]]
        for i in range(len(bbox3d)):
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            velocity = (*bbox3d.tensor[i, 7:9], 0.0)
            # velo_val = np.linalg.norm(box3d[i, 7:9])
            # velo_ori = box3d[i, 6]
            # velocity = (
            # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)
    elif type(bbox3d) == CameraInstance3DBoxes:
        # our Camera coordinate system -> nuScenes box coordinate system
        # convert the dim/rot to nuscbox convention
        nus_box_dims = box_dims[:, [2, 0, 1]]
        nus_box_yaw = -box_yaw
        cam2ego = np.array(info['images'][camera_type]['cam2ego'])
        ego2global = np.array(info['ego2global'])
        for i in range(len(bbox3d)):
            if scores[i] <= score_thd:
                continue
            box_center = np.dot(cam2ego[:3, :3], box_gravity_center[i]) + cam2ego[:3, 3]
            radius = box_center[0] * box_center[0] + box_center[1] * box_center[1]
            det_range = eval_configs.class_range[classes[labels[i]]]
            if radius > det_range * det_range:
                continue
            box_center = np.dot(ego2global[:3, :3], box_center) + ego2global[:3, 3]
            cosi = np.cos(nus_box_yaw[i])
            sini = np.sin(nus_box_yaw[i])
            R0 = np.array([[cosi, -sini, 0], [0, 0, -1], [sini, cosi, 0]], dtype=np.float32)
            R1 = np.matmul(ego2global[:3, :3], cam2ego[:3, :3])
            if bbox3d.tensor.shape[1] == 9:
                velocity = (bbox3d.tensor[i, 7], 0.0, bbox3d.tensor[i, 8])
                velocity = np.dot(R1, velocity)
            else:
                velocity = np.zeros([3, ], dtype=np.float32)
            R = np.matmul(R1, R0)
            # if not np.allclose(np.dot(R, R.transpose()), np.eye(3), rtol=1e-05, atol=1e-06):
            #     print(np.dot(R, R.transpose()))
            #     print('a')
            # raise ValueError("Matrix must be orthogonal, i.e. its transpose should be its inverse")
            quat = trace_method(R)
            nusc_anno = dict(
                translation=box_center.tolist(),
                size=nus_box_dims[i].tolist(),
                rotation=quat,
                velocity=velocity[:2].tolist(),
                label=int(labels[i]),
                detection_score=float(scores[i]),
                attribute=int(attr_list[i]))
            nusc_anno_list.append(nusc_anno)
    else:
        raise NotImplementedError(
            f'Do not support convert {type(bbox3d)} bboxes'
            'to standard NuScenesBoxes.')

    return nusc_anno_list


# @profile
def trace_method(m):
    """
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    Altered to work with the column vector convention instead of row vectors
    """

    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [m[2, 1] - m[1, 2], t, m[1, 0] + m[0, 1], m[0, 2] + m[2, 0]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[0, 2] - m[2, 0], m[1, 0] + m[0, 1], t, m[2, 1] + m[1, 2]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[1, 0] - m[0, 1], m[0, 2] + m[2, 0], m[2, 1] + m[1, 2], t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [t, m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]
    s = 0.5 / sqrt(t)
    q = [float(i * s) for i in q]
    return q


# @profile
def output_to_nusc_box(detection: dict) -> List[NuScenesBox]:
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - bboxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    bbox3d = detection['bboxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    if 'attr_labels' in detection:
        attrs = detection['attr_labels'].numpy()
    else:
        attrs = np.zeros_like(labels, dtype=np.int32)
    box_gravity_center = bbox3d.gravity_center.numpy()
    box_dims = bbox3d.dims.numpy()
    box_yaw = bbox3d.yaw.numpy()

    box_list = []

    if type(bbox3d) == LiDARInstance3DBoxes:
        # our LiDAR coordinate system -> nuScenes box coordinate system
        nus_box_dims = box_dims[:, [1, 0, 2]]
        for i in range(len(bbox3d)):
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            velocity = (*bbox3d.tensor[i, 7:9], 0.0)
            # velo_val = np.linalg.norm(box3d[i, 7:9])
            # velo_ori = box3d[i, 6]
            # velocity = (
            # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)
    elif type(bbox3d) == CameraInstance3DBoxes:
        # our Camera coordinate system -> nuScenes box coordinate system
        # convert the dim/rot to nuscbox convention
        nus_box_dims = box_dims[:, [2, 0, 1]]
        nus_box_yaw = -box_yaw
        for i in range(len(bbox3d)):
            q1 = pyquaternion.Quaternion(
                axis=[0, 0, 1], radians=nus_box_yaw[i])
            q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
            quat = q2 * q1
            if bbox3d.tensor.shape[1] == 9:
                velocity = (bbox3d.tensor[i, 7], 0.0, bbox3d.tensor[i, 8])
            else:
                velocity = (0.0, 0.0, 0.0)
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)
    else:
        raise NotImplementedError(
            f'Do not support convert {type(bbox3d)} bboxes'
            'to standard NuScenesBoxes.')

    return box_list, attrs


def lidar_nusc_box_to_global(
        info: dict, boxes: List[NuScenesBox], classes: List[str],
        eval_configs: DetectionConfig) -> List[NuScenesBox]:
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        lidar2ego = np.array(info['lidar_points']['lidar2ego'])
        q = pyquaternion.Quaternion(matrix=lidar2ego, rtol=1e-05, atol=1e-07)
        box.rotate(q)
        box.translate(lidar2ego[:3, 3])
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        ego2global = np.array(info['ego2global'])
        q = pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07)
        box.rotate(q)
        box.translate(ego2global[:3, 3])
        box_list.append(box)
    return box_list


# @profile
def cam_nusc_box_to_global(
        info: dict,
        boxes: List[NuScenesBox],
        attrs: List[str],
        classes: List[str],
        eval_configs: DetectionConfig,
        camera_type: str = 'CAM_FRONT',
) -> List[NuScenesBox]:
    """Convert the box from camera to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        attrs (list[str]): List of attributes.
        camera_type (str): Type of camera.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    attr_list = []
    for (box, attr) in zip(boxes, attrs):
        # Move box to ego vehicle coord system
        cam2ego = np.array(info['images'][camera_type]['cam2ego'])
        box.rotate(
            pyquaternion.Quaternion(matrix=cam2ego, rtol=1e-05, atol=1e-07))
        box.translate(cam2ego[:3, 3])
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        ego2global = np.array(info['ego2global'])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
        box.translate(ego2global[:3, 3])
        box_list.append(box)
        attr_list.append(attr)
    return box_list, attr_list


def global_nusc_box_to_cam(info: dict, boxes: List[NuScenesBox],
                           classes: List[str],
                           eval_configs: DetectionConfig) -> List[NuScenesBox]:
    """Convert the box from global to camera coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        ego2global = np.array(info['ego2global'])
        box.translate(-ego2global[:3, 3])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05,
                                    atol=1e-07).inverse)
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to camera coord system
        cam2ego = np.array(info['images']['CAM_FRONT']['cam2ego'])
        box.translate(-cam2ego[:3, 3])
        box.rotate(
            pyquaternion.Quaternion(matrix=cam2ego, rtol=1e-05,
                                    atol=1e-07).inverse)
        box_list.append(box)
    return box_list


def nusc_box_to_cam_box3d(boxes: List[NuScenesBox]):
    """Convert boxes from :obj:`NuScenesBox` to :obj:`CameraInstance3DBoxes`.

    Args:
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.

    Returns:
        tuple (:obj:`CameraInstance3DBoxes` | torch.Tensor | torch.Tensor):
            Converted 3D bounding boxes, scores and labels.
    """
    locs = torch.Tensor([b.center for b in boxes]).view(-1, 3)
    dims = torch.Tensor([b.wlh for b in boxes]).view(-1, 3)
    rots = torch.Tensor([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).view(-1, 1)
    velocity = torch.Tensor([b.velocity[0::2] for b in boxes]).view(-1, 2)

    # convert nusbox to cambox convention
    dims[:, [0, 1, 2]] = dims[:, [1, 2, 0]]
    rots = -rots

    boxes_3d = torch.cat([locs, dims, rots, velocity], dim=1).cuda()
    cam_boxes3d = CameraInstance3DBoxes(
        boxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5))
    scores = torch.Tensor([b.score for b in boxes]).cuda()
    labels = torch.LongTensor([b.label for b in boxes]).cuda()
    nms_scores = scores.new_zeros(scores.shape[0], 10 + 1)
    indices = labels.new_tensor(list(range(scores.shape[0])))
    nms_scores[indices, labels] = scores
    return cam_boxes3d, nms_scores, labels


class HiddenPrints:
    def __init__(self, activated=True):
        # activated参数表示当前修饰类是否被激活
        self.activated = activated
        self.original_stdout = None

    def open(self):
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # 这里的os.devnull实际上就是Linux系统中的“/dev/null”
        # /dev/null会使得发送到此目标的所有数据无效化，就像“被删除”一样
        # 这里使用/dev/null对sys.stdout输出流进行重定向

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()


class NuScenesEval(DetectionEval):
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """

    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = './',
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')

        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)
        # Check result file exists.
        if isinstance(result_path, str):
            assert os.path.exists(result_path), 'Error: The result file does not exist!'
            self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                         verbose=verbose)
        else:
            assert 'results' in result_path, 'Error: No field `results` in result file. Please note that the result format changed.' \
                                             'See https://www.nuscenes.org/object-detection for more information.'

            # Deserialize results and get meta data.
            all_results = EvalBoxes.deserialize(result_path['results'], DetectionBox)
            meta = result_path['meta']
            if verbose:
                print("Loaded results finish. Found detections for {} samples.".format(len(all_results.sample_tokens)))

            # Check that each sample has no more than x predicted boxes.
            for sample_token in self.gt_boxes.sample_tokens:
                if sample_token in all_results.sample_tokens:
                    assert len(all_results.boxes[sample_token]) <= self.cfg.max_boxes_per_sample, \
                        "Error: Only <= %d boxes per sample allowed!" % self.cfg.max_boxes_per_sample
                else:
                    all_results.add_boxes(sample_token, [])
            self.pred_boxes = all_results
            self.meta = meta
        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        return metrics_summary
