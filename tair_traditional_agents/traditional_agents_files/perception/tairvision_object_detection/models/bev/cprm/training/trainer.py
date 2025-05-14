import torch
from tairvision.models.bev.lss.training.trainer import TrainingModule
from tairvision.models.bev.cprm.cprm import CPRM
from tairvision.models.bev.lss.training.metrics import EvaluateDetection3D

class EvaluateDetection3DCPRM(EvaluateDetection3D):
    def __init__(self, cfg, eval_version, resultdir="./eval_3d", use_gpu=True):
        super().__init__(cfg, resultdir)

    def _import_lidar_box_func(self):
        from tairvision.models.bev.cprm.utils.bbox import view_boxes_to_lidar_boxes_3d
        self.view_boxes_to_lidar_boxes = view_boxes_to_lidar_boxes_3d

class TrainingModuleCPRM(TrainingModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.evaluate_detection3d = EvaluateDetection3DCPRM(self.cfg, eval_version="mini", resultdir="./eval_3d")

    def _import_get_cfg(self):
        from tairvision.models.bev.cprm.training.config import get_cfg
        self.get_cfg = get_cfg

    def _init_model(self):
        model = CPRM(self.cfg)

        return model

    def _import_target_functions(self):
        from tairvision.models.bev.common.utils.instance import get_targets_dynamic
        from tairvision.models.bev.lss.utils.static import get_targets_static
        from tairvision.models.bev.lss.utils.bbox import get_targets2d
        from tairvision.models.bev.cprm.utils.bbox import get_targets3d

        self.get_targets_dynamic = get_targets_dynamic
        self.get_targets_static = get_targets_static
        self.get_targets2d = get_targets2d
        self.get_targets3d = get_targets3d

    @staticmethod
    def _import_visualization_module():
        from tairvision.models.bev.cprm.utils.visualization import VisualizationModuleCP

        return VisualizationModuleCP

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.OPTIMIZER.LR_STEPS)

        return [optimizer], [lr_scheduler]

