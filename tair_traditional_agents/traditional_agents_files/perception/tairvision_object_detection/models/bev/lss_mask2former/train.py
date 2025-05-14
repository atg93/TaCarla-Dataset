from tairvision.models.bev.lss.train import TrainingInterface
from tairvision.models.bev.lss_mask2former.training import callbacks


class TrainingInterfaceMask2former(TrainingInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def import_parser_and_cfg(self):
        from tairvision.models.bev.lss_mask2former.configs.config import get_parser, get_cfg
        return get_parser, get_cfg

    def get_callbacks(self):
        callback_dict = callbacks.__dict__
        callback_list = []
        for callback_type, callback_config in self.cfg.CALLBACKS.items():
            callback_config_dict = callback_config["CONFIG"].convert_to_dict()
            callback_config_dict = {k.lower(): v for k, v in callback_config_dict.items()}
            callback = callback_dict[callback_config["NAME"]](**callback_config_dict)
            callback_list.append(callback)
        return callback_list
    
    def get_module_class(self):
        from tairvision.models.bev.lss_mask2former.training.trainer import TrainingModuleMask2former
        return TrainingModuleMask2former


if __name__ == "__main__":
    trainer = TrainingInterfaceMask2former()
    trainer.train()
