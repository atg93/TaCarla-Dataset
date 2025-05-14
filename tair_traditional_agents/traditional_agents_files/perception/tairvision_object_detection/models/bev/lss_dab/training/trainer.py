from tairvision.models.bev.lss_mask2former.training.trainer import TrainingModuleMask2former
from tairvision.models.bev.lss_dab.blocks.lss_dab import LiftSplatDAB, LiftSplatLinearDAB, LiftSplatTemporalDAB


class TrainingModuleDAB(TrainingModuleMask2former):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.param_keys = None

    def _init_model(self):
        depth_channels = (self.cfg.LIFT.D_BOUND[1] - self.cfg.LIFT.D_BOUND[0]) / self.cfg.LIFT.D_BOUND[2]
        if self.cfg.TIME_RECEPTIVE_FIELD > 1:
            model = LiftSplatTemporalDAB(self.cfg)
        else:
            model = LiftSplatDAB(self.cfg) if depth_channels > 1 else LiftSplatLinearDAB(self.cfg)
        return model

    def _import_get_cfg(self):
        from tairvision.models.bev.lss_dab.configs.config import get_cfg as get_cfg_dab
        self.get_cfg = get_cfg_dab
