from tairvision.models.bev.fiery.training.config import get_cfg
from tairvision.models.bev.lss.training.metrics import IntersectionOverUnion, PanopticMetric
from tairvision.models.bev.fiery.training.trainer import TrainingModulePrediction


class TrainingModuleDemo(TrainingModulePrediction):
    def __init__(self, hparams):
        super().__init__(hparams)

        # see config.py for details
        self.hparams = hparams
        # pytorch lightning does not support saving YACS CfgNone
        cfg = get_cfg(cfg_dict=self.hparams)
        self.cfg = cfg
        self.n_classes = len(cfg.MODEL.HEADDYNAMIC.SEMANTIC_SEG.WEIGHTS)

        # Bird's-eye view extent in meters
        assert self.cfg.LIFT.X_BOUND[1] > 0 and self.cfg.LIFT.Y_BOUND[1] > 0

        # Model
        if cfg.MODEL.FUTURE_PREDICTOR == "fiery":
            from tairvision.models.bev.fiery.fiery_cache import FieryCache as Model
        else:
            from tairvision.models.bev.fiery.beverse import Beverse as Model

        self.model = Model(cfg)

        self.metric_iou_val = IntersectionOverUnion(self.n_classes)
        self.metric_panoptic_val = PanopticMetric(n_classes=self.n_classes)

        self.training_step_count = 0
        self.factor = {}
