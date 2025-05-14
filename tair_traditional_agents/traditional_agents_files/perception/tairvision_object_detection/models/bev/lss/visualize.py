from argparse import ArgumentParser
from tqdm import tqdm

import cv2
import torch

from tairvision_object_detection.models.bev.lss.utils.network import preprocess_batch
from tairvision_object_detection.models.bev.lss.utils.visualization import LayoutControl
from tairvision_object_detection.models.bev.lss.training.config import get_cf_from_yaml


class VisualizationInterface:
    def __init__(self, path, dataroot, version, gt_only=False):

        device = torch.device('cuda:0')
        self.device = device

        if gt_only:
            cfg = get_cf_from_yaml(path)
            module = None
        else:
            module = self._import_training_module().load_from_checkpoint(path, strict=True)
            print(f'Loaded weights from \n {path}')
            module.eval()

            module.to(device)
            model = module.model

            cfg = model.cfg

        self.module = module

        cfg.GPUS = "[0]"
        cfg.BATCHSIZE = 1

        cfg.DATASET.DATAROOT = dataroot
        cfg.DATASET.VERSION = version

        _, valloader, _, _ = self._import_training_interface().get_loaders(cfg)
        self.valloader = valloader

        visualization_module = self._import_visualization_module()(cfg)
        self.visualization_module = visualization_module

        layout_control = LayoutControl()
        self.layout_control = layout_control

    @staticmethod
    def _import_training_module():
        from tairvision_object_detection.models.bev.lss.training.trainer import TrainingModule

        return TrainingModule

    @staticmethod
    def _import_visualization_module():
        from tairvision_object_detection.models.bev.lss.utils.visualization import VisualizationModule

        return VisualizationModule

    @staticmethod
    def _import_training_interface():
        from tairvision_object_detection.models.bev.lss.train import TrainingInterface

        return TrainingInterface

    def visualize(self):

        iterator = iter(tqdm(self.valloader))
        batch = next(iterator)
        i_batch = 1
        while i_batch < len(self.valloader):
            preprocess_batch(batch, self.device)
            if self.module is None:
                output = {}
            else:
                # Forward pass
                output = self.module.predict_step(batch)

            self.visualization_module.plot_all(batch, output, self.layout_control.get_show())

            ch = cv2.waitKey(1)
            if self.layout_control(ch):
                break

            if not self.layout_control.pause:
                batch = next(iterator)
                i_batch += 1
            elif self.layout_control.next:
                batch = next(iterator)
                i_batch += 1
                self.layout_control.next = False

        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser(description='LSS visualisation')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='/datasets/nu/nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')
    parser.add_argument('--config-file', default=None, type=str, help='path to checkpoint')

    args = parser.parse_args()

    if args.checkpoint is not None:
        visualization_interface = VisualizationInterface(args.checkpoint, args.dataroot, args.version)
        visualization_interface.visualize()
    elif args.config_file is not None:
        visualization_interface = VisualizationInterface(args.config_file, args.dataroot, args.version, gt_only=True)
        visualization_interface.visualize()
    else:
        ValueError("Provide either a checkpoint or a config file.")
