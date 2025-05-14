from tairvision.models.bev.lss.evaluate import EvaluationInterface
from tairvision.models.bev.lss.evaluate import get_parser as get_parser_lss
from argparse import ArgumentParser


class EvaluationInterfaceMask2Former(EvaluationInterface):
    def __init__(self, *args, **kwargs):
        super(EvaluationInterfaceMask2Former, self).__init__(*args, **kwargs)


    @staticmethod
    def _import_training_module():
        from tairvision.models.bev.lss_mask2former.training.trainer import TrainingModuleMask2former
        return TrainingModuleMask2former
    

if __name__ == '__main__':
    parser = get_parser_lss()
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["checkpoint_path"] = args_dict.pop("checkpoint")

    evaluation_interface = EvaluationInterfaceMask2Former(**args_dict)
    evaluation_interface.evaluate()