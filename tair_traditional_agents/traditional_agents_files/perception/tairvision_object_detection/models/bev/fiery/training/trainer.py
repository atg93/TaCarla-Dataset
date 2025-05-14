import torch
from tairvision.models.bev.fiery.training.losses import ProbabilisticLoss
from tairvision.models.bev.lss.training.trainer import TrainingModule

class TrainingModulePrediction(TrainingModule):
    def __init__(self, hparams):
        super().__init__(hparams)

    def _import_get_cfg(self):
        from tairvision.models.bev.fiery.training.config import get_cfg
        self.get_cfg = get_cfg

    def _init_model(self):
        # Model
        if self.cfg.MODEL.FUTURE_PREDICTOR == "fiery":
            from tairvision.models.bev.fiery.fiery import Fiery as Model
        else:
            from tairvision.models.bev.fiery.beverse import Beverse as Model

        return Model(self.cfg)

    def forward(self, batch):
        image = batch['images']
        intrinsics = batch['intrinsics']
        extrinsics = batch['cams_to_lidar']
        view = batch['view']
        future_egomotion = batch['future_egomotion']
        pcloud_list = self.pcloud_list_collator(batch)

        # Warp labels
        labels = self.get_targets_dynamic(batch,
                                          receptive_field=self.model.receptive_field,
                                          spatial_extent=self.model.spatial_extent)

        future_distribution_inputs = self.get_future_distributions(labels)

        # Forward pass
        output = self.model(image, intrinsics, extrinsics, view, future_egomotion,
                            future_distribution_inputs, pcloud_list=pcloud_list)

        return output

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

        if self.model.flow is not None:
            loss_flow, factor_flow = self.model.flow.get_loss(output, targets)
            loss.update(loss_flow)
            factor.update(factor_flow)

        if self.cfg.PROBABILISTIC.ENABLED:
            loss['probabilistic'] = self.cfg.PROBABILISTIC.WEIGHT * ProbabilisticLoss()(output)

        return loss, factor

    def get_future_distributions(self, labels):
        future_distribution_inputs = [labels['segmentation'], labels['centerness'], labels['offset']]
        if self.cfg.INSTANCE_FLOW.ENABLED:
            future_distribution_inputs.append(labels['flow'])

        future_distribution_inputs = torch.cat(future_distribution_inputs, dim=2)

        return  future_distribution_inputs

