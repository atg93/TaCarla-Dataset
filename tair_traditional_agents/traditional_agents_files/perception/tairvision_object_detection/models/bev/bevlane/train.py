import os
import torch
import time

from tairvision_object_detection.models.bev.lss.train import TrainingInterface
from tairvision_object_detection.models.bev.bevlane.training.trainer import TrainingModuleLane
from tairvision_object_detection.models.bev.common.utils.network import load_pretrained_weights

from pytorch_lightning.callbacks import ModelCheckpoint


class TrainingInterfaceLane(TrainingInterface):
    def __init__(self):
        super(TrainingInterfaceLane, self).__init__()

    def get_callbacks(self):
        save_dir = self.save_dir
        # TODO change checkpoint saving frequency
        checkpoint_callback = ModelCheckpoint(dirpath=save_dir, every_n_train_steps=1700)

        callbacks = [checkpoint_callback]

        return callbacks

    def initialize_module(self):
        module = TrainingModuleLane(self.cfg.convert_to_dict())

        if self.cfg.PRETRAINED.LOAD_WEIGHTS:
            # Load pretrained weights
            pretrained_weights = torch.load(os.path.join(self.cfg.DATASET.DATAROOT, self.cfg.PRETRAINED.PATH),
                                            map_location='cpu')
            state_dict = module.state_dict()

            updated_state_dict = load_pretrained_weights(pretrained_weights, state_dict)
            module.load_state_dict(updated_state_dict, strict=False)
            print(f'Loaded single-image model weights from {self.cfg.PRETRAINED.PATH}')

        return module

    def test(self):
        self.trainer.test(self.module, dataloaders=self.valloader)


if __name__ == "__main__":
    training_interface = TrainingInterfaceLane()
    training_interface.train()
