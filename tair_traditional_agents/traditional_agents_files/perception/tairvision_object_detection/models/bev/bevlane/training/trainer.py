from tairvision.models.bev.lss.training.trainer import TrainingModule
from tairvision.models.bev.bevlane.lss_lane import LiftSplatLane
from tairvision.models.bev.bevlane.utils.visualization import VisualizationModuleLane
from tairvision.models.bev.bevlane.training.metrics import ChamferDistanceAP
import torch


class TrainingModuleLane(TrainingModule):
    def __init__(self, hparams, **kwargs):
        super(TrainingModuleLane, self).__init__(hparams)
        self.metric_chamfer_val = ChamferDistanceAP(n_classes=2, distance_thresholds=[0.5, 1.0, 1.5])

    @staticmethod
    def _import_visualization_module():
        return VisualizationModuleLane

    def _import_target_functions(self):
        super()._import_target_functions()
        from tairvision.models.bev.bevlane.utils.line import get_targets_line
        self.get_targets_line = get_targets_line

    def prepare_targets(self, batch):
        targets = super().prepare_targets(batch)
        targets_line = self.get_targets_line(batch, receptive_field=self.model.receptive_field)
        targets.update(targets_line)
        targets['view'] = batch['view']
        return targets

    def validation_step(self, batch, batch_idx):
        output_dict = super().validation_step(batch, batch_idx)
        model_output = self.post_process_outputs(output_dict['model_output'])
        self.metric_chamfer_val.update(model_output['line_instances'], model_output['line_probs'],
                                       model_output['line_classes'],
                                       batch['line_instances'], batch['line_classes'])
        return output_dict

    def test_step(self, batch, batch_idx):
        output_dict = super().validation_step(batch, batch_idx)  # TODO: replace validation_step with test_step
        model_output = self.post_process_outputs(output_dict['model_output'])
        self.metric_chamfer_val.update(model_output['line_instances'], model_output['line_probs'],
                                       model_output['line_classes'],
                                       batch['line_instances'], batch['line_classes'])

    def compute_metrics(self):
        super().compute_metrics()
        scores = self.metric_chamfer_val.compute()
        self.metric_chamfer_val.reset()
        if self.trainer.local_rank == 0:
            self.log('val/chamfer_mAP', scores['mAP'], prog_bar=False)
            del scores['mAP']
            for class_id, score_dict in scores.items():
                for score_name, score_value in score_dict.items():
                    name = f'val/{class_id}_{score_name}'
                    self.log(name, score_value, prog_bar=False)

    def post_process_outputs(self, output):
        vis_output = super().post_process_outputs(output)
        with torch.no_grad():
            if self.model.static is not None:
                line_output = self.model.static.post_process(output)
                vis_output.update(line_output)
        return vis_output

    def _init_model(self):
        model = LiftSplatLane(self.cfg)
        return model
