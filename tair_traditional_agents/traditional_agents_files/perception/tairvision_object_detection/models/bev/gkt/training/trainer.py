import torch
from tairvision.models.bev.gkt.gkt import GKT
from tairvision.models.bev.lss.training.trainer import TrainingModule


class TrainingModule(TrainingModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        # self.param_keys = None

    def _init_model(self):
        model = GKT(self.cfg)
        return model

    @staticmethod
    def import_cfg():
        from tairvision.models.bev.gkt.training.config import get_cfg
        return get_cfg



