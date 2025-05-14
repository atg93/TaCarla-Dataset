import torch
import os
from tairvision_object_detection.models.bev.lss.train import TrainingInterface
from tairvision_object_detection.models.bev.cprm.training.trainer import TrainingModuleCPRM
from tairvision_object_detection.models.bev.common.utils.network import load_pretrained_weights

class TrainingInterfaceCPRM(TrainingInterface):
    def __init__(self):
        super().__init__()

    @staticmethod
    def import_parser_and_cfg():
        from tairvision_object_detection.models.bev.cprm.training.config import get_parser, get_cfg

        return get_parser, get_cfg

    def initialize_module(self):
        module = TrainingModuleCPRM(self.cfg.convert_to_dict())

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
    training_interface = TrainingInterfaceCPRM()
    training_interface.train()