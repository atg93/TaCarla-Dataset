from tairvision.models.bev.lss_mask2former.train import TrainingInterfaceMask2former


class TrainingInterfaceDAB(TrainingInterfaceMask2former):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def import_parser_and_cfg(self):
        from tairvision.models.bev.lss_dab.configs.config import get_parser, get_cfg
        return get_parser, get_cfg

    def get_module_class(self):
        from tairvision.models.bev.lss_dab.training.trainer import TrainingModuleDAB
        return TrainingModuleDAB


if __name__ == "__main__":
    trainer = TrainingInterfaceDAB()
    trainer.train()
