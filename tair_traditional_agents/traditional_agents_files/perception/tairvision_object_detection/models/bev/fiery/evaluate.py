from argparse import ArgumentParser
from tairvision_object_detection.models.bev.lss.evaluate import EvaluationInterface


class EvaluationInterfacePrediction(EvaluationInterface):
    def __init__(self, checkpoint_path, dataroot='/datasets/nu/nuscenes', version='trainval', resultdir=None):
        super().__init__(checkpoint_path, dataroot, version, resultdir)

    def _set_eval_frames(self):
        self.evaluation_frames = {'T=0': 0,
                                  'T=1': 1,
                                  'T=2': 2,
                                  'T=3': 3,
                                  'T=4': 4,
                                  }

    @staticmethod
    def _import_training_module():
        from tairvision_object_detection.models.bev.fiery.training.trainer import TrainingModulePrediction

        return TrainingModulePrediction


if __name__ == '__main__':
    parser = ArgumentParser(description='Prediction evaluation')
    parser.add_argument('--checkpoint', default='./fiery.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='/datasets/nu/nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')
    parser.add_argument('--resultdir', default=None, type=str, help='path to result directory')

    args = parser.parse_args()

    evaluation_interface = EvaluationInterfacePrediction(args.checkpoint, args.dataroot, args.version, args.resultdir)
    evaluation_interface.evaluate()

