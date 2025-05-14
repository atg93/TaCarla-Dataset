import torch
#import wandb
import pytorch_lightning as pl

from tairvision.models.bev.common.nuscenes.process import PCloudListCollator
from tairvision.models.bev.lss.lss import LiftSplatTemporal, LiftSplat, LiftSplatLinear
from tairvision.models.bev.lss.training.metrics import (IntersectionOverUnion, PanopticMetric, MeanAP,
                                                        EvaluateDetection3D, load_metrics)


class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # see config.py for details
        self.hparams.update(hparams)

        self._import_get_cfg()
        self.cfg = self.get_cfg(cfg_dict=self.hparams)

        self._import_target_functions()

        self.n_classes = len(self.cfg.MODEL.HEADDYNAMIC.SEMANTIC_SEG.WEIGHTS)
        self.final_size = (self.cfg.IMAGE.FINAL_DIM[1], self.cfg.IMAGE.FINAL_DIM[0])

        # Bird's-eye view extent in meters
        assert self.cfg.LIFT.X_BOUND[1] > 0 and self.cfg.LIFT.Y_BOUND[1] > 0

        self.model = self._init_model()

        self._initialize_metrics()

        self.factor = {}
        self.pcloud_list_collator = PCloudListCollator(self.cfg.PCLOUD.N_FEATS)
        self.save_hyperparameters()

        self.visualization_module = self._import_visualization_module()(self.cfg)

    def _import_get_cfg(self):
        from tairvision.models.bev.lss.training.config import get_cfg
        self.get_cfg = get_cfg

    def _initialize_metrics(self):
        self.metric_iou_val = IntersectionOverUnion(self.n_classes)
        self.metric_panoptic_val = PanopticMetric(n_classes=self.n_classes)
        if self.model.head3d is not None:
            self.evaluate_detection3d = EvaluateDetection3D(self.cfg)

    def _init_model(self):
        depth_channels = (self.cfg.LIFT.D_BOUND[1] - self.cfg.LIFT.D_BOUND[0]) / self.cfg.LIFT.D_BOUND[2]
        # Model
        if self.cfg.TIME_RECEPTIVE_FIELD > 1:
            model = LiftSplatTemporal(self.cfg)
        else:
            model = LiftSplat(self.cfg) if depth_channels > 1 else LiftSplatLinear(self.cfg)
        return model

    @staticmethod
    def _import_visualization_module():
        from tairvision.models.bev.lss.utils.visualization import VisualizationModule

        return VisualizationModule

    def _import_target_functions(self):
        from tairvision.models.bev.common.utils.instance import get_targets_dynamic
        from tairvision.models.bev.lss.utils.static import get_targets_static
        from tairvision.models.bev.lss.utils.bbox import get_targets2d
        from tairvision.models.bev.lss.utils.bbox import get_targets3d_xdyd, get_targets3d_yaw

        self.get_targets_dynamic = get_targets_dynamic
        self.get_targets_static = get_targets_static
        self.get_targets2d = get_targets2d
        if self.cfg.MODEL.HEAD3D.TARGET_TYPE == "yaw":
            self.get_targets3d = get_targets3d_yaw
        else:  # target_type="xdyd"
            self.get_targets3d = get_targets3d_xdyd

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(
            params, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
        )

        return optimizer

    def forward(self, batch):
        image = batch['images']
        intrinsics = batch['intrinsics']
        extrinsics = batch['cams_to_lidar']
        view = batch['view']
        future_egomotion = batch['future_egomotion']
        pcloud_list = self.pcloud_list_collator(batch)

        # Forward pass
        output = self.model(image, intrinsics, extrinsics, view, future_egomotion, pcloud_list)

        return output

    def shared_step(self, batch, is_train):
        # Forward pass
        output = self.forward(batch)

        # Get targets for available heads
        targets = self.prepare_targets(batch)

        # Get losses and factors for available heads
        loss, factor = self.get_losses(output, targets)
        self.factor = factor

        # Metrics
        if not is_train and not self.trainer.sanity_checking:
            self.accumulate_metrics(output, targets, batch)

        return output, targets, loss

    def training_step(self, batch, batch_idx):
        output, targets, loss = self.shared_step(batch, True)
        batch_size = batch["images"].shape[0]

        self.log_losses(loss, panel='train', batch_size=batch_size, logged_prefixes=['loss'])
        self.log_losses(self.factor, panel='factors', batch_size=batch_size)

        unfactored_losses = self.create_unfactored_losses_dict(loss)
        self.log_losses(unfactored_losses, panel='train_wo_factors', batch_size=batch_size, logged_prefixes=['loss'])

        with torch.no_grad():
            self.visualize(batch, output, targets, batch_idx, prefix='train')

        return sum(loss.values())

    def training_epoch_end(self, step_outputs):
        pass

    def create_unfactored_losses_dict(self, loss):
        unfactored_losses = {}
        for loss_key, loss_value in loss.items():
            if "loss" in loss_key:
                factor_key = loss_key[5:]
                if factor_key in self.factor:
                    factor_value = self.factor[factor_key]
                    unfactored_losses[loss_key] = loss_value / factor_value

        return unfactored_losses

    def validation_step(self, batch, batch_idx):
        output, targets, loss = self.shared_step(batch, False)
        batch_size = batch["images"].shape[0]
        self.log_losses(loss, panel='val', batch_size=batch_size, logged_prefixes=['loss'])

        unfactored_losses = self.create_unfactored_losses_dict(loss)
        self.log_losses(unfactored_losses, panel='val_wo_factors', batch_size=batch_size, logged_prefixes=['loss'])

        output_dict = dict(model_output=output, batch=batch, targets=targets, batch_idx=batch_idx)
        return output_dict

    def validation_step_end(self, output_dict):
        self.visualize(output_dict['batch'], output_dict['model_output'], output_dict['targets'],
                       output_dict['batch_idx'], prefix='val')
        # since validation_epoch_end is overriden, pl stores outputs and cuda memory increases
        # we avoid cuda out of memory by returning dummy int output
        return 0

    def validation_epoch_end(self, step_outputs):
        if not self.trainer.sanity_checking:
            self.compute_metrics()

    def test_step(self, batch, batch_idx):
        output, targets, loss = self.shared_step(batch, False)

        output_dict = dict(model_output=output, batch=batch, targets=targets, batch_idx=batch_idx)
        return output_dict

    def test_step_end(self, output_dict):
        # since test_epoch_end is overriden, pl stores outputs and cuda memory increases
        # we avoid cuda out of memory by returning dummy int output
        return 0

    def test_epoch_end(self, step_outputs):
        if not self.trainer.sanity_checking:
            self.compute_metrics()

    def prepare_targets(self, batch):
        targets = {}
        # if self.model.dynamic is not None:
        targets_dynamic = self.get_targets_dynamic(batch,
                                                   receptive_field=self.model.receptive_field,
                                                   spatial_extent=self.model.spatial_extent,
                                                   )
        targets.update(targets_dynamic)

        # if self.model.static is not None:
        targets_static = self.get_targets_static(batch, receptive_field=self.model.receptive_field)
        targets.update(targets_static)

        # if self.model.head2d is not None:
        targets2d, _ = self.get_targets2d(batch,
                                          receptive_field=self.model.receptive_field,
                                          image_size=self.final_size,
                                          )
        targets['head2d'] = targets2d

        # if self.model.head3d is not None:
        targets3d, _ = self.get_targets3d(batch,
                                          receptive_field=self.model.receptive_field,
                                          spatial_extent=self.model.spatial_extent,
                                          )
        targets['head3d'] = targets3d

        return targets

    def get_losses(self, output, targets):
        loss, factor = {}, {}
        if self.model.dynamic is not None:
            loss_dynamic_segm, factor_dynamic_segm = self.model.dynamic.get_loss(output, targets)
            loss.update(loss_dynamic_segm)
            factor.update(factor_dynamic_segm)

        if self.model.static is not None:
            loss_static_segm, factor_static_segm = self.model.static.get_loss(output, targets)
            loss.update(loss_static_segm)
            factor.update(factor_static_segm)

        if self.model.head2d is not None:
            loss_detection2d, factor_detection2d = self.model.head2d.get_loss(output['head2d'], targets['head2d'])
            loss.update(loss_detection2d)
            factor.update(factor_detection2d)

        if self.model.head3d is not None:
            loss_detection3d, factor_detection3d = self.model.head3d.get_loss(output['head3d'], targets['head3d'])
            loss.update(loss_detection3d)
            factor.update(factor_detection3d)

        return loss, factor

    def log_losses(self, loss, panel='train', batch_size=1, logged_prefixes=None):
        for key, value in loss.items():
            prefix = key.split('_')[0]
            if logged_prefixes is None or prefix in logged_prefixes:
                name = panel + '/' + key
                self.log(name, value.item(), prog_bar=False, sync_dist=True, batch_size=batch_size)

    def accumulate_metrics(self, output, targets, batch):
        if self.model.dynamic is not None:
            post_output = self.model.dynamic.post_process(output)
            self.metric_iou_val(post_output["segm"], targets['segmentation'])
            self.metric_panoptic_val(post_output["inst"], targets['instance'])
        """
        if self.model.head3d is not None:
            detections3d = self.model.head3d.get_detections(output['head3d'])
            self.metric_bbox_bev(detections3d, targets['head3d'])
        """
        if self.model.head3d is not None:
            detections3d = self.model.head3d.get_detections(output['head3d'])
            self.evaluate_detection3d.update(detections3d, batch)

    def compute_metrics(self):
        if self.model.dynamic is not None:
            # log per class iou metrics
            class_names = ['background', 'dynamic']
            logged_classes = ['dynamic']
            scores = self.metric_iou_val.compute()
            self.log_metrics(scores, class_names, panel='metrics/val', logged_classes=logged_classes)
            self.metric_iou_val.reset()

            scores = self.metric_panoptic_val.compute()
            self.log_metrics(scores, class_names, panel='metrics/val', logged_classes=logged_classes)
            self.metric_panoptic_val.reset()

        if self.model.head3d is not None:
            self.evaluate_detection3d.gather(self.local_rank)
            if self.local_rank == 0:
                metrics, classes = self.evaluate_detection3d.compute()
                scores = load_metrics(metrics)
                class_names = scores.keys()
                logged_classes = scores.keys()
                self.log_metrics(scores, class_names, panel='metrics/val_3d', logged_classes=logged_classes,
                                 sync_dist=False)
            self.evaluate_detection3d.reset()
            """
            # Bev 2d eval
            scores = self.metric_bbox_bev.compute()
            class_names = scores.keys()
            logged_classes = scores.keys()
            self.log_metrics(scores, class_names, panel='metrics/val_bev', logged_classes=logged_classes)
            self.metric_bbox_bev.reset()
            """

    def log_metrics(self, scores, class_names, panel='metrics/val', logged_classes=None, sync_dist=True):
        for score_name, score_all_classes in scores.items():
            for class_name, score_per_class in zip(class_names, score_all_classes):
                if logged_classes is None or class_name in logged_classes:
                    name = f'{panel}/{score_name}_{class_name}'
                    self.log(name, score_per_class.item(), prog_bar=False, sync_dist=sync_dist)

    def post_process_outputs(self, output):
        with torch.no_grad():
            vis_output = output.copy()
            if self.model.dynamic is not None:
                post_output = self.model.dynamic.post_process(output)
                vis_output.update(post_output)
            if self.model.head3d is not None:
                vis_output['head3d'] = self.model.head3d.get_detections(output['head3d'])
        return vis_output

    def visualize(self, batch, output, targets, batch_idx, prefix='train'):
        if prefix == 'train':
            vis_interval = self.cfg.VIS_INTERVAL_TRAIN
        else:
            vis_interval = self.cfg.VIS_INTERVAL_VAL

        if batch_idx % vis_interval == 0:
            name = f'media_{prefix}/{batch_idx}'

            vis_output = self.post_process_outputs(output)
            bev_map, _, _ = self.visualization_module.get_bev_maps(batch, vis_output, targets)

            for logger in self.loggers:
                if "TensorBoardLogger" in logger.__str__():
                    logger.experiment.add_image(name, bev_map.transpose(2, 0, 1), self.global_step)
                #if "WandbLogger" in logger.__str__():
                #    logger.experiment.log({name: [wandb.Image(bev_map)]})

    def predict_step(self, batch, **kwargs):
        with torch.no_grad():
            output = self.forward(batch)
            if self.model.dynamic is not None:
                post_output = self.model.dynamic.post_process(output)
                output.update(post_output)
            if self.model.static is not None:
                post_output = self.model.static.post_process(output)
                output.update(post_output)
            if self.model.head2d is not None:
                output['head2d'] = self.model.head2d.get_detections(output['head2d'])
            if self.model.head3d is not None:
                output['head3d'] = self.model.head3d.get_detections(output['head3d'])

        return output
