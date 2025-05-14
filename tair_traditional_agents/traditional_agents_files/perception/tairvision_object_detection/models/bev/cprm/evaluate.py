from argparse import ArgumentParser
from tairvision_object_detection.models.bev.lss.evaluate import EvaluationInterface

class EvaluationInterfaceCPRM(EvaluationInterface):
    def __init__(self,
                 checkpoint_path,
                 dataroot='/datasets/nu/nuscenes',
                 version='trainval',
                 resultdir=None):
        super().__init__(checkpoint_path, dataroot, version, resultdir)

    @staticmethod
    def _import_training_module():
        from tairvision_object_detection.models.bev.cprm.training.trainer import TrainingModuleCPRM

        return TrainingModuleCPRM

    def _import_lidar_box_func(self):
        from tairvision_object_detection.models.bev.cprm.utils.bbox import view_boxes_to_lidar_boxes_3d
        self.view_boxes_to_lidar_boxes = view_boxes_to_lidar_boxes_3d


if __name__ == '__main__':
    parser = ArgumentParser(description='CPRM evaluation')
    parser.add_argument('--checkpoint', default='./lss.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='/datasets/nu/nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')
    parser.add_argument('--resultdir', default=None, type=str, help='path to result directory')
    args = parser.parse_args()

    evaluation_interface = EvaluationInterfaceCPRM(args.checkpoint, args.dataroot, args.version, args.resultdir)
    evaluation_interface.evaluate()
