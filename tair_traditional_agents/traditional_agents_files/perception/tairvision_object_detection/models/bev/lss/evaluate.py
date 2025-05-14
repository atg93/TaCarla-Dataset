from argparse import ArgumentParser
from tqdm import tqdm

import torch
import numpy as np
from pyquaternion import Quaternion
import os
import json

from nuscenes.nuscenes import NuScenes
from tools.nuscenes_tools.eval.detection.config import config_factory
from tools.nuscenes_tools.eval.detection.evaluate import NuScenesEval

from tairvision_object_detection.models.bev.lss.training.metrics import IntersectionOverUnion, PanopticMetric
from tairvision_object_detection.models.bev.lss.utils.network import preprocess_batch
from tairvision_object_detection.models.bev.lss.train import TrainingInterface


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
        'vehicle': 'vehicle.parked',
        'dynamic': 'vehicle.parked',
        'cycle': 'cycle.without_rider',
    }


class EvaluationInterface:
    def __init__(self, checkpoint_path, dataroot='/datasets/nu/nuscenes',
                 version='trainval', resultdir=None, **kwargs):
        
        self.args_dict = kwargs

        self.args_dict.update(
            {
                'checkpoint': checkpoint_path, 
                'dataroot': dataroot,
                'version': version,
                'resultdir': resultdir
            }
        )

        device = torch.device('cuda:0')
        self.device = device

        module = self._import_training_module().load_from_checkpoint(checkpoint_path, strict=True)
        print(f'Loaded weights from \n {checkpoint_path}')
        module.eval()

        module.to(device)
        self.module = module
        model = module.model

        cfg = model.cfg
        cfg = self.load_cfg_settings(cfg)
        self.cfg = cfg

        valloader, filter_classes = self.get_loaders(cfg)
        self.valloader = valloader
        self.filter_classes = filter_classes

        self.checkpoint_path = checkpoint_path
        self.resultdir = resultdir

        self._init_metrics()

    def _set_eval_frames(self):
        self.evaluation_frames = {'T=0': 0}

    def _sef_eval_ranges(self):
        # 30mx30m, 100mx100m
        self.evaluation_ranges = {
            '30x30': (70, 130),
            '100x100': (0, 200)
            }

    def get_loaders(self, cfg):
        _, valloader, filter_classes, _ = TrainingInterface.get_loaders(cfg)
        return valloader, filter_classes
        
    def load_cfg_settings(self, cfg):
        cfg.GPUS = "[0]"
        cfg.BATCHSIZE = 1
        cfg.DATASET.DATAROOT = self.args_dict['dataroot']
        cfg.DATASET.VERSION = self.args_dict['version']
        return cfg
        
    def _init_metrics(self):
        self._import_lidar_box_func()
        self._init_det3d_metrics()
        self._set_eval_frames()
        self._sef_eval_ranges()
        self._init_dynamic_metrics()

    @staticmethod
    def _import_training_module():
        from tairvision_object_detection.models.bev.lss.training.trainer import TrainingModule

        return TrainingModule

    def _import_lidar_box_func(self):
        from tairvision_object_detection.models.bev.lss.utils.bbox import view_boxes_to_lidar_boxes_xdyd, view_boxes_to_lidar_boxes_yaw
        target_type = self.cfg.MODEL.HEAD3D.TARGET_TYPE
        if target_type == "yaw":
            self.view_boxes_to_lidar_boxes = view_boxes_to_lidar_boxes_yaw
        else:  # target_type="xdyd"
            self.view_boxes_to_lidar_boxes = view_boxes_to_lidar_boxes_xdyd

    def _init_det3d_metrics(self):
        self.nusc = NuScenes(version='v1.0-{}'.format(self.cfg.DATASET.VERSION),
                             dataroot=self.cfg.DATASET.DATAROOT,
                             verbose=True,
                             )

        self.det3d_annotations = {}

    def _init_dynamic_metrics(self):

        # Init dynamic metrics
        panoptic_metrics = {}
        iou_metrics = {}
        n_classes = len(self.cfg.MODEL.HEADDYNAMIC.SEMANTIC_SEG.WEIGHTS)
        for key in self.evaluation_ranges.keys():
            panoptic_metrics[key] = {}
            iou_metrics[key] = {}
            for frame, index in self.evaluation_frames.items():
                panoptic_metrics[key][frame] = PanopticMetric(
                    n_classes=n_classes, 
                    temporally_consistent=True).to(self.device)
                iou_metrics[key][frame] = IntersectionOverUnion(n_classes).to(self.device)

        self.panoptic_metrics = panoptic_metrics
        self.iou_metrics = iou_metrics

    def evaluate(self):
        for i, batch in enumerate(tqdm(self.valloader)):
            lidar_to_world = batch['lidar_to_world'][0, 0, 0].numpy()
            sample_token = batch['sample_token'][0][0]
            preprocess_batch(batch, self.device)

            # Forward pass
            output = self.module.predict_step(batch)
            targets = self.module.prepare_targets(batch)

            # Detection 3d evaluation
            if 'head3d' in output.keys():
                lidar_boxes = self.view_boxes_to_lidar_boxes(output['head3d'][0], batch, self.filter_classes,
                                                             score_threshold=0.40, is_eval=True)

                translation = lidar_to_world[0:3, 3]
                rotation = Quaternion._from_matrix(matrix=lidar_to_world, atol=1e-07)

                self.accumulate_det3d_metrics(sample_token, lidar_boxes, translation, rotation, self.det3d_annotations)

            self.accumulate_dynamic_metrics(output, targets, self.panoptic_metrics, self.iou_metrics)

        if 'head3d' in output.keys():
            self.compute_det3d_metrics(self.det3d_annotations)
        self.compute_dynamic_metrics(self.panoptic_metrics, self.iou_metrics)

    @staticmethod
    def accumulate_det3d_metrics(sample_token, lidar_boxes, translation, rotation, annotations_dict):
        annotations = []
        for j in range(len(lidar_boxes)):
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
                    attr = DefaultAttribute[name]
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = DefaultAttribute[name]
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
        # end of the loop over the lidar boxes
        annotations_dict[sample_token] = annotations

    def accumulate_dynamic_metrics(self, output, targets, panoptic_metrics, iou_metrics):

        # Dynamic segmentation evaluation
        for key, grid in self.evaluation_ranges.items():
            for frame, index in self.evaluation_frames.items():
                limits = slice(grid[0], grid[1])
                panoptic_metrics[key][frame](output['inst'][:, index:index+1, :, limits, limits].contiguous(),
                                             targets['instance'][:, index:index+1, :, limits, limits].contiguous())

                iou_metrics[key][frame](output['segm'][:, index:index+1, :, limits, limits].contiguous(),
                                        targets['segmentation'][:, index:index+1, :, limits, limits].contiguous())

    def compute_det3d_metrics(self, annotations_dict):
        # Finalizing detection 3d evaluation
        nusc_submissions = {
            'meta': dict(use_lidar=False,
                         use_camera=True,
                         use_radar=False,
                         use_map=False,
                         use_external=False),
            'results': annotations_dict
        }

        if self.resultdir is None:
            resultdir = os.path.join(os.path.dirname(self.checkpoint_path), 'eval_3d')
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)
        result_path = os.path.join(resultdir, 'results_nusc.json')

        print('Results writen to', result_path)
        with open(result_path, "w") as outfile:
            json.dump(nusc_submissions, outfile)

        output_dir = os.path.join(*os.path.split(result_path)[:-1])
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            self.nusc,
            config=config_factory(self.filter_classes.eval_config),
            result_path=result_path,
            eval_set=eval_set_map["v1.0-" + self.cfg.DATASET.VERSION],
            output_dir=output_dir,
            detection_mapping=self.filter_classes.name_mapping,
            limit_period=True,
            verbose=True)
        nusc_eval.main(render_curves=False)

    def compute_dynamic_metrics(self, panoptic_metrics, iou_metrics):
        # finalizing dynamic segmentation evaluation
        results = {}
        for key, grid in self.evaluation_ranges.items():
            for frame, index in self.evaluation_frames.items():
                panoptic_scores = panoptic_metrics[key][frame].compute()
                for panoptic_key, value in panoptic_scores.items():
                    results[f'{panoptic_key}'] = results.get(f'{panoptic_key}', []) + [100 * value[1].item()]

                iou_scores = iou_metrics[key][frame].compute()
                results['iou'] = results.get('iou', []) + [100 * iou_scores['iou'][1].item()]

        for panoptic_key in ['iou', 'pq', 'sq', 'rq']:
            print(panoptic_key)
            print(' & '.join([f'{x:.1f}' for x in results[panoptic_key]]))


def get_parser():
    parser = ArgumentParser(description='LSS evaluation')
    parser.add_argument('--checkpoint', default='./lss.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='/datasets/nu/nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')
    parser.add_argument('--resultdir', default=None, type=str, help='path to result directory')
    parser.add_argument('--batchsize', type=int, help='batch size for evaluation', default=1, required=False)     
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    evaluation_interface = EvaluationInterface(args.checkpoint, args.dataroot, args.version, args.resultdir)
    evaluation_interface.evaluate()
