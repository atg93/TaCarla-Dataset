import torch
from tairvision.models.bev.cvt.cvt import CVT
from tairvision.models.bev.lss.training.trainer import TrainingModule


class TrainingModule(TrainingModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        # self.param_keys = None

    def _init_model(self):
        model = CVT(self.cfg)
        return model

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(
            params, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.OPTIMIZER.LR,
                                                        pct_start=0.3,
                                                        div_factor=10.0,
                                                        final_div_factor=10,
                                                        cycle_momentum=False,
                                                        total_steps=30001)
        return [optimizer], [scheduler]

    @staticmethod
    def import_cfg():
        from tairvision.models.bev.cvt.training.config import get_cfg
        return get_cfg



