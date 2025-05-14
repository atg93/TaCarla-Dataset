import os
import torch

from tairvision_object_detection.models.bev.fiery.training.trainer import TrainingModulePrediction
from tairvision_object_detection.models.bev.lss.train import TrainingInterface
from tairvision_object_detection.models.bev.common.utils.network import load_pretrained_weights


class TrainingInterfacePrediction(TrainingInterface):
    def __init__(self):
        super().__init__()

    def initialize_module(self):
        module = TrainingModulePrediction(self.cfg.convert_to_dict())
        if self.cfg.PRETRAINED.LOAD_WEIGHTS:
            if self.cfg.PRETRAINED.MODE == 'LSS':
                # Load pretrained LSS/LSS_linear model
                pretrained_weights = torch.load(self.cfg.PRETRAINED.PATH, map_location='cpu')
                model_dict = module.state_dict()
                updated_model_dict = load_pretrained_weights(pretrained_weights, model_dict)
                module.load_state_dict(updated_model_dict, strict=False)
                print(f'Loaded pretrained LSS model weights from {self.cfg.PRETRAINED.PATH}')
            else:
                # Load single-image instance segmentation model.
                pretrained_model_weights = torch.load(os.path.join(self.cfg.DATASET.DATAROOT, self.cfg.PRETRAINED.PATH),
                                                      map_location='cpu')
                module.load_state_dict(pretrained_model_weights, strict=False)
                print(f'Loaded single-image model weights from {self.cfg.PRETRAINED.PATH}')

        return module

    @staticmethod
    def import_parser_and_cfg():
        from tairvision_object_detection.models.bev.fiery.training.config import get_parser, get_cfg

        return get_parser, get_cfg


if __name__ == "__main__":
    training_interface = TrainingInterfacePrediction()
    training_interface.train()
