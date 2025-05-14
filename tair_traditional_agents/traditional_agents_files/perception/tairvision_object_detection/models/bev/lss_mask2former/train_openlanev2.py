from tairvision.models.bev.lss_mask2former.train import TrainingInterfaceMask2former
from tairvision.datasets.openlane_v2 import prepare_dataloaders
from tairvision.models.bev.common.openlanev2.process import ResizeCropRandomFlipNormalizeCameraKeyBased, get_resizing_and_cropping_parameters_modified
from tairvision.references.detection.presets import DetectionPresetTrain, DetectionPresetEval



class TrainingInterfaceMask2formerOpenLaneV2(TrainingInterfaceMask2former):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_module_class(self):
        from tairvision.models.bev.lss_mask2former.training.trainer_openlaneV2 import TrainingModuleMask2FormerOpenLaneV2
        return TrainingModuleMask2FormerOpenLaneV2
    
    @staticmethod
    def import_parser_and_cfg():
        from tairvision.models.bev.lss_mask2former.configs.config_openlanev2 import get_parser, get_cfg
        return get_parser, get_cfg

    @staticmethod
    def get_loaders(cfg, return_testloader=False):
        augmentation_parameters = get_resizing_and_cropping_parameters_modified(cfg.IMAGE, cfg.PRETRAINED)
        
        augmentation_dict = {}
        for name in cfg.IMAGE.NAMES:
            augmentation_dict[name] = augmentation_parameters
        
        augmentation_parameters_front_center = get_resizing_and_cropping_parameters_modified(cfg.FRONT_CENTER_IMAGE, cfg.PRETRAINED)
        augmentation_dict["RING_FRONT_CENTER"] = augmentation_parameters_front_center
        
        transforms_val = ResizeCropRandomFlipNormalizeCameraKeyBased(cfg, augmentation_parameters, augmentation_dict, enable_random_transforms=False)
        transforms_train = ResizeCropRandomFlipNormalizeCameraKeyBased(cfg, augmentation_parameters, augmentation_dict, enable_random_transforms=True)

        if cfg.MODEL.HEAD2D.SHARED_ENCODER:
            mean_2d = transforms_val.mean
            std_2d = transforms_val.std
        else:
            mean_2d = cfg.IMAGE.BACKBONE_MEAN_2D
            std_2d = cfg.IMAGE.BACKBONE_STD_2D
        
        transforms2d_train = DetectionPresetTrain(cfg.DATASET.PERSPECTIVE_TRANSFORM_TRAIN, mean=mean_2d, std=std_2d)
        transforms2d_val = DetectionPresetEval(cfg.DATASET.PERSPECTIVE_TRANSFORM_VAL, mean=mean_2d, std=std_2d)

        all_loaders = prepare_dataloaders(
            cfg, transforms_train=transforms_train, transforms_val=transforms_val,
            transforms2d_train=transforms2d_train, transforms2d_val=transforms2d_val,
            trainval=cfg.DATASET.ENABLE_TRAINVAL_TRAINING, return_testloader=return_testloader
        )

        if return_testloader:
            return all_loaders[0], all_loaders[1], all_loaders[2], None
        else:
            return all_loaders[0], all_loaders[1], None, None


if __name__ == "__main__":
    trainer = TrainingInterfaceMask2formerOpenLaneV2()
    trainer.train()
