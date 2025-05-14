import os
import time
import torch

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from tairvision_object_detection.models.bev.lss.train import TrainingInterface
from pytorch_lightning.callbacks import ModelCheckpoint
from tairvision_object_detection.models.bev.cvt.training.trainer import TrainingModule
from tairvision_object_detection.models.bev.common.utils.network import load_pretrained_weights


class TrainingInterface(TrainingInterface):
    def __init__(self):
        super().__init__()

        # configuration initialization
        get_parser, get_cfg = self.import_parser_and_cfg()

        args = get_parser().parse_args()
        cfg = get_cfg(args)

        self.args = args
        self.cfg = cfg

    @staticmethod
    def import_parser_and_cfg():
        from tairvision_object_detection.models.bev.lss.training.config import get_parser
        from tairvision_object_detection.models.bev.cvt.training.config import get_cfg

        return get_parser, get_cfg

    def get_strategy(self):
        if self.args.disable_distributed_training:
            strategy = None
        else:
            strategy = DDPStrategy(find_unused_parameters=True)
            pl.seed_everything()

        return strategy

    def get_callbacks(self):
        callbacks = []
        save_dir = os.path.join(self.cfg.LOG_DIR, time.strftime('%Y_%m_%d_at_%H%M') + '_' + self.cfg.TAG)
        checkpoint_callback = ModelCheckpoint(monitor='metrics/val/iou_dynamic', dirpath=save_dir, filename=self.cfg.TAG,
                                              mode='max', auto_insert_metric_name=True, verbose=True)
        callbacks.append(checkpoint_callback)
        return callbacks

    def initialize_module(self):
        module = TrainingModule(self.cfg.convert_to_dict())

        if self.cfg.PRETRAINED.LOAD_WEIGHTS:
            # Load pretrained weights
            pretrained_weights = torch.load(os.path.join(self.cfg.DATASET.DATAROOT, self.cfg.PRETRAINED.PATH),
                                            map_location='cpu')
            state_dict = module.state_dict()

            updated_state_dict = load_pretrained_weights(pretrained_weights, state_dict)
            module.load_state_dict(updated_state_dict, strict=False)
            print(f'Loaded single-image model weights from {self.cfg.PRETRAINED.PATH}')

        return module


if __name__ == "__main__":
    training_interface = TrainingInterface()
    training_interface.train()
