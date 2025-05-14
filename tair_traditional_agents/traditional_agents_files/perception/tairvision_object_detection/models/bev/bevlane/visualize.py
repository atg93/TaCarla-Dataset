from argparse import ArgumentParser

from tairvision_object_detection.models.bev.lss.visualize import VisualizationInterface


class VisualizationInterfaceLane(VisualizationInterface):
    def __init__(self, path, dataroot, version, gt_only=False):
        super(VisualizationInterfaceLane, self).__init__(path, dataroot, version, gt_only)

    @staticmethod
    def _import_training_module():
        from tairvision_object_detection.models.bev.bevlane.training.trainer import TrainingModuleLane

        return TrainingModuleLane

    @staticmethod
    def _import_visualization_module():
        from tairvision_object_detection.models.bev.bevlane.utils.visualization import VisualizationModuleLane

        return VisualizationModuleLane

    @staticmethod
    def _import_training_interface():
        from tairvision_object_detection.models.bev.bevlane.train import TrainingInterfaceLane

        return TrainingInterfaceLane


if __name__ == '__main__':
    parser = ArgumentParser(description='LSS visualisation')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='/datasets/nu/nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')
    parser.add_argument('--config-file', default=None, type=str, help='path to checkpoint')

    args = parser.parse_args()

    if args.checkpoint is not None:
        visualization_interface = VisualizationInterfaceLane(args.checkpoint, args.dataroot, args.version)
        visualization_interface.visualize()
    elif args.config_file is not None:
        visualization_interface = VisualizationInterfaceLane(args.config_file, args.dataroot, args.version, gt_only=True)
        visualization_interface.visualize()
    else:
        ValueError("Provide either a checkpoint or a config file.")
