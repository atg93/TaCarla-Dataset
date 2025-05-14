from abc import ABC
from typing import Optional

import torch
from torchmetrics.metric import Metric
from torchmetrics.classification import StatScores
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
import os
from pyquaternion import Quaternion
import json

from nuscenes.nuscenes import NuScenes
from tools.nuscenes_tools.eval.detection.config import config_factory
from tools.nuscenes_tools.eval.detection.evaluate import NuScenesEval
from tairvision.models.bev.common.nuscenes.process import FilterClasses


class IntersectionOverUnion(Metric, ABC):
    """Computes intersection-over-union."""
    def __init__(
        self,
        n_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        reduction: str = 'none',
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)

        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.absent_score = absent_score
        self.reduction = reduction

        self.add_state('true_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('support', default=torch.zeros(n_classes), dist_reduce_fx='sum')

        self.stat_class = StatScores(task="multiclass", num_classes=self.n_classes, average="none",
                                     ignore_index=self.ignore_index)

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        result = self.stat_class(prediction, target)

        self.true_positive += result[:, 0]
        self.false_positive += result[:, 1]
        self.false_negative += result[:, 3]
        self.support += result[:, 4]

    def compute(self):
        scores = torch.zeros(self.n_classes, device=self.true_positive.device, dtype=torch.float32)

        for class_idx in range(self.n_classes):
            if class_idx == self.ignore_index:
                continue

            tp = self.true_positive[class_idx]
            fp = self.false_positive[class_idx]
            fn = self.false_negative[class_idx]
            sup = self.support[class_idx]

            # If this class is absent in the target (no support) AND absent in the pred (no true or false
            # positives), then use the absent_score for this class.
            if sup + tp + fp == 0:
                scores[class_idx] = self.absent_score
                continue

            denominator = tp + fp + fn
            score = tp.to(torch.float) / denominator
            scores[class_idx] = score

        # Remove the ignored class index from the scores.
        if (self.ignore_index is not None) and (0 <= self.ignore_index < self.n_classes):
            scores = torch.cat([scores[:self.ignore_index], scores[self.ignore_index+1:]])

        # return reduce(scores, reduction=self.reduction) #TODO, check this functionality later if needed
        return {'iou': scores}


class PanopticMetric(Metric):
    def __init__(
        self,
        n_classes: int,
        temporally_consistent: bool = True,
        vehicles_id: int = 1,
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)

        self.n_classes = n_classes
        self.temporally_consistent = temporally_consistent
        self.vehicles_id = vehicles_id
        self.keys = ['iou', 'true_positive', 'false_positive', 'false_negative']

        self.add_state('iou', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('true_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.zeros(n_classes), dist_reduce_fx='sum')

    def update(self, pred_instance, gt_instance):
        """
        Update state with predictions and targets.

        Parameters
        ----------
            pred_instance: (b, s, h, w)
                Temporally consistent instance segmentation prediction.
            gt_instance: (b, s, h, w)
                Ground truth instance segmentation.
        """
        batch_size, sequence_length = gt_instance.shape[:2]
        # Process labels
        assert gt_instance.min() == 0, 'ID 0 of gt_instance must be background'
        pred_segmentation = (pred_instance > 0).long()
        gt_segmentation = (gt_instance > 0).long()

        for b in range(batch_size):
            unique_id_mapping = {}
            for t in range(sequence_length):
                result = self.panoptic_metrics(
                    pred_segmentation[b, t].detach(),
                    pred_instance[b, t].detach(),
                    gt_segmentation[b, t],
                    gt_instance[b, t],
                    unique_id_mapping,
                )

                self.iou += result['iou']
                self.true_positive += result['true_positive']
                self.false_positive += result['false_positive']
                self.false_negative += result['false_negative']

    def compute(self):
        denominator = torch.maximum(
            (self.true_positive + self.false_positive / 2 + self.false_negative / 2),
            torch.ones_like(self.true_positive)
        )
        pq = self.iou / denominator
        sq = self.iou / torch.maximum(self.true_positive, torch.ones_like(self.true_positive))
        rq = self.true_positive / denominator

        return {'pq': pq,
                'sq': sq,
                'rq': rq,
                # If 0, it means there wasn't any detection.
                'denominator': (self.true_positive + self.false_positive / 2 + self.false_negative / 2),
                }

    def panoptic_metrics(self, pred_segmentation, pred_instance, gt_segmentation, gt_instance, unique_id_mapping):
        """
        Computes panoptic quality metric components.

        Parameters
        ----------
            pred_segmentation: [H, W] range {0, ..., n_classes-1} (>= n_classes is void)
            pred_instance: [H, W] range {0, ..., n_instances} (zero means background)
            gt_segmentation: [H, W] range {0, ..., n_classes-1} (>= n_classes is void)
            gt_instance: [H, W] range {0, ..., n_instances} (zero means background)
            unique_id_mapping: instance id mapping to check consistency
        """
        n_classes = self.n_classes

        result = {key: torch.zeros(n_classes, dtype=torch.float32, device=gt_instance.device) for key in self.keys}

        assert pred_segmentation.dim() == 3
        assert pred_segmentation.shape == pred_instance.shape == gt_segmentation.shape == gt_instance.shape

        n_instances = int(torch.cat([pred_instance, gt_instance]).max().item())
        n_all_things = n_instances + n_classes  # Classes + instances.
        n_things_and_void = n_all_things + 1

        # Now 1 is background; 0 is void (not used). 2 is vehicle semantic class but since it overlaps with
        # instances, it is not present.
        # and the rest are instance ids starting from 3
        prediction, pred_to_cls = self.combine_mask(pred_segmentation, pred_instance, n_classes, n_all_things)
        target, target_to_cls = self.combine_mask(gt_segmentation, gt_instance, n_classes, n_all_things)

        # Compute ious between all stuff and things
        # hack for bincounting 2 arrays together
        x = prediction + n_things_and_void * target
        bincount_2d = torch.bincount(x.long(), minlength=n_things_and_void ** 2)
        if bincount_2d.shape[0] != n_things_and_void ** 2:
            raise ValueError('Incorrect bincount size.')
        conf = bincount_2d.reshape((n_things_and_void, n_things_and_void))
        # Drop void class
        conf = conf[1:, 1:]

        # Confusion matrix contains intersections between all combinations of classes
        union = conf.sum(0).unsqueeze(0) + conf.sum(1).unsqueeze(1) - conf
        iou = torch.where(union > 0, (conf.float() + 1e-9) / (union.float() + 1e-9), torch.zeros_like(union).float())

        # In the iou matrix, first dimension is target idx, second dimension is pred idx.
        # Mapping will contain a tuple that maps prediction idx to target idx for segments matched by iou.
        mapping = (iou > 0.5).nonzero(as_tuple=False)

        # Check that classes match.
        is_matching = pred_to_cls[mapping[:, 1]] == target_to_cls[mapping[:, 0]]
        mapping = mapping[is_matching]
        tp_mask = torch.zeros_like(conf, dtype=torch.bool)
        tp_mask[mapping[:, 0], mapping[:, 1]] = True

        # First ids correspond to "stuff" i.e. semantic seg.
        # Instance ids are offset accordingly
        for target_id, pred_id in mapping:
            cls_id = pred_to_cls[pred_id]

            if self.temporally_consistent and cls_id == self.vehicles_id:
                if target_id.item() in unique_id_mapping and unique_id_mapping[target_id.item()] != pred_id.item():
                    # Not temporally consistent
                    result['false_negative'][target_to_cls[target_id]] += 1
                    result['false_positive'][pred_to_cls[pred_id]] += 1
                    unique_id_mapping[target_id.item()] = pred_id.item()
                    continue

            result['true_positive'][cls_id] += 1
            result['iou'][cls_id] += iou[target_id][pred_id]
            unique_id_mapping[target_id.item()] = pred_id.item()

        for target_id in range(n_classes, n_all_things):
            # If this is a true positive do nothing.
            if tp_mask[target_id, n_classes:].any():
                continue
            # If this target instance didn't match with any predictions and was present set it as false negative.
            if target_to_cls[target_id] != -1:
                result['false_negative'][target_to_cls[target_id]] += 1

        for pred_id in range(n_classes, n_all_things):
            # If this is a true positive do nothing.
            if tp_mask[n_classes:, pred_id].any():
                continue
            # If this predicted instance didn't match with any prediction, set that predictions as false positive.
            if pred_to_cls[pred_id] != -1 and (conf[:, pred_id] > 0).any():
                result['false_positive'][pred_to_cls[pred_id]] += 1

        return result

    def combine_mask(self, segmentation: torch.Tensor, instance: torch.Tensor, n_classes: int, n_all_things: int):
        """Shifts all things ids by num_classes and combines things and stuff into a single mask

        Returns a combined mask + a mapping from id to segmentation class.
        """
        instance = instance.view(-1)
        instance_mask = instance > 0
        instance = instance - 1 + n_classes

        segmentation = segmentation.clone().view(-1)
        segmentation_mask = segmentation < n_classes  # Remove void pixels.

        # Build an index from instance id to class id.
        instance_id_to_class_tuples = torch.cat(
            (
                instance[instance_mask & segmentation_mask].unsqueeze(1),
                segmentation[instance_mask & segmentation_mask].unsqueeze(1),
            ),
            dim=1,
        )
        instance_id_to_class = -instance_id_to_class_tuples.new_ones((n_all_things,))
        instance_id_to_class[instance_id_to_class_tuples[:, 0]] = instance_id_to_class_tuples[:, 1]
        instance_id_to_class[torch.arange(n_classes, device=segmentation.device)] = torch.arange(
            n_classes, device=segmentation.device
        )

        segmentation[instance_mask] = instance[instance_mask]
        segmentation += 1  # Shift all legit classes by 1.
        segmentation[~segmentation_mask] = 0  # Shift void class to zero.

        return segmentation, instance_id_to_class


class MeanAP(Metric):
    def __init__(
        self,
    ):
        super().__init__()
        self.mean_ap = MeanAveragePrecision()
        self.detections = {}

    def update(self, prediction, gt):
        result = self.mean_ap(prediction, gt)

        self.detections = {
            "map": [result["map"]],
            "map_50": [result["map_50"]],
            # "map_75": [result["map_75"]],
            # "map_small": [result["map_small"]],
            # "map_medium": [result["map_medium"]],
            # "map_large": [result["map_large"]],
            # "map_per_class": [result["map_per_class"]],
            # "mar_1": [result["mar_1"]],
            # "mar_10": [result["mar_10"]],
            # "mar_100": [result["mar_100"]],
            # "mar_small": [result["mar_small"]],
            # "mar_medium": [result["mar_medium"]],
            # "mar_large": [result["mar_large"]],
            # "mar_100_per_class": [result["mar_100_per_class"]],
        }

    def compute(self):
        return self.detections


class EvaluateDetection3D:
    def __init__(self, cfg, resultdir=None):

        self.cfg = cfg
        self.nusc = None
        self.filter_classes = FilterClasses(cfg.DATASET.FILTER_CLASSES, cfg.DATASET.BOX_RESIZING_COEF)

        self.resultdir = resultdir
        if self.resultdir is not None and not os.path.exists(self.resultdir):
            os.makedirs(self.resultdir)

        self._import_lidar_box_func()
        self.world_size = len(self.cfg.GPUS)
        self.annotations_dict = {}

    def reset(self):
        self.annotations_dict = {}

    def init_nusc(self):
        self.nusc = NuScenes(version='v1.0-{}'.format(self.cfg.DATASET.VERSION),
                             dataroot=self.cfg.DATASET.DATAROOT,
                             verbose=False,
                             )

    def update(self, prediction, batch):
        for i in range(len(prediction)):
            sample_token = batch['sample_token'][i][0]
            lidar_boxes = self.view_boxes_to_lidar_boxes(prediction[i], batch, self.filter_classes, is_eval=True)

            lidar_to_world = np.asarray(batch['lidar_to_world'][i, 0, 0].cpu())
            translation = lidar_to_world[0:3, 3]
            rotation = Quaternion._from_matrix(matrix=lidar_to_world, atol=1e-07)

            accumulate_det3d_metrics(sample_token, lidar_boxes, translation, rotation, self.annotations_dict)

    def gather(self, rank):
        dict_list = [{} for _ in range(self.world_size)]
        torch.distributed.all_gather_object(dict_list, self.annotations_dict)
        if rank == 0:
            for r in range(1, self.world_size):
                self.annotations_dict.update(dict_list[r])

    def compute(self):
        if self.nusc is None:
            self.init_nusc()

        annotations_dict_all = self.annotations_dict

        nusc_submissions = {
            'meta': dict(use_lidar=False,
                         use_camera=True,
                         use_radar=False,
                         use_map=False,
                         use_external=False),
            'results': annotations_dict_all
        }

        if self.resultdir is not None:
            result_path = os.path.join(self.resultdir, 'results_nusc.json')
            print('Results written to', result_path)
            with open(result_path, "w") as outfile:
                json.dump(nusc_submissions, outfile)

            output_path = os.path.join(*os.path.split(result_path)[:-1])
            kwargs = {}
        else:
            result_path = None
            output_path = None
            kwargs = {
                'result_dict': nusc_submissions,
            }

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }

        nusc_eval = NuScenesEval(
            self.nusc,
            config=config_factory(self.filter_classes.eval_config),
            result_path=result_path,
            eval_set=eval_set_map["v1.0-" + self.cfg.DATASET.VERSION],
            output_dir=output_path,
            detection_mapping=self.filter_classes.name_mapping,
            limit_period=True,
            verbose=False,
            **kwargs
        )
        metrics = nusc_eval.main(render_curves=False)

        classes = self.filter_classes.classes[1:]
        return metrics, classes

    def _import_lidar_box_func(self):
        from tairvision.models.bev.lss.utils.bbox import view_boxes_to_lidar_boxes_xdyd, view_boxes_to_lidar_boxes_yaw
        target_type = self.cfg.MODEL.HEAD3D.TARGET_TYPE
        if target_type == "yaw":
            self.view_boxes_to_lidar_boxes = view_boxes_to_lidar_boxes_yaw
        else:  # target_type="xdyd"
            self.view_boxes_to_lidar_boxes = view_boxes_to_lidar_boxes_xdyd


def accumulate_det3d_metrics(sample_token, lidar_boxes, translation, rotation, annotations_dict):
    default_attribute = {
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
        'vehicle': 'vehicle.parked',
        'dynamic': 'vehicle.parked',
        'cycle': 'cycle.without_rider',
    }

    annotations = []
    for j in range(len(lidar_boxes)):
        # Lidar boxes to global boxes
        lidar_boxes[j].rotate(rotation)
        lidar_boxes[j].translate(translation)
        name = lidar_boxes[j].name
        score = lidar_boxes[j].score
        if np.sqrt(lidar_boxes[j].velocity[0] ** 2 + lidar_boxes[j].velocity[1] ** 2) > 0.2:
            if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                attr = 'vehicle.moving'
            elif name in ['bicycle', 'motorcycle']:
                attr = 'cycle.with_rider'
            else:
                attr = default_attribute[name]
        else:
            if name in ['pedestrian']:
                attr = 'pedestrian.standing'
            elif name in ['bus']:
                attr = 'vehicle.stopped'
            else:
                attr = default_attribute[name]
        # Values of w, l, and h all must > 0
        if lidar_boxes[j].wlh[0] == 0:
            continue
        elif lidar_boxes[j].wlh[1] == 0:
            continue
        elif lidar_boxes[j].wlh[2] == 0:
            continue
        nusc_anno = dict(
            sample_token=sample_token,
            translation=lidar_boxes[j].center.tolist(),
            size=lidar_boxes[j].wlh.tolist(),
            rotation=lidar_boxes[j].orientation.elements.tolist(),
            velocity=lidar_boxes[j].velocity[:2].tolist(),
            detection_name=name,
            detection_score=score,
            attribute_name=attr)
        annotations.append(nusc_anno)
    annotations_dict[sample_token] = annotations


def load_metrics_from_file(out_path, classes):
    if os.path.exists(out_path):
        with open(os.path.join(out_path, 'metrics_summary.json')) as file:
            metrics = json.load(file)

        metric_3d = {
            "mean_ap": torch.Tensor([metrics['mean_ap']])
        }
        classes = metrics['mean_dist_aps']
        for class_name in classes.keys():
            metric_3d[class_name] = torch.Tensor([classes[class_name]])
        print("3D Detection - Mean Average Precision: ", metric_3d["mean_ap"])
    else:
        metric_3d = {
            "mean_ap": torch.Tensor([0.])
        }
        for class_name in classes:
            metric_3d[class_name] = torch.Tensor([0.])
        print("Cannot find the metrics_summary file in this path: ", out_path)
    return metric_3d


def load_metrics(metrics):

    metric_3d = {
        "mean_ap": torch.Tensor([metrics['mean_ap']])
    }
    classes = metrics['mean_dist_aps']
    for class_name in classes.keys():
        metric_3d[class_name] = torch.Tensor([classes[class_name]])
    print("3D Detection - Mean Average Precision: ", metric_3d["mean_ap"])
    return metric_3d
