import os
import time

import torch

from tairvision_object_detection.models.bev.common.nuscenes.process import (ResizeCropRandomFlipNormalize, FilterClasses,
                                                           get_resizing_and_cropping_parameters)
from tairvision_object_detection.models.bev.common.utils.network import load_pretrained_weights
from tairvision_object_detection.datasets.nuscenes import prepare_dataloaders

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, MLFlowLogger
from tairvision_object_detection.models.bev.common.utils.strategy import MyDDPStrategy, DDPStrategy


class TrainingInterface:
    def __init__(self):

        # configuration initialization
        get_parser, get_cfg = self.import_parser_and_cfg()

        args = get_parser().parse_args()
        cfg = get_cfg(args)

        self.args = args
        self.cfg = cfg

        # hour and minute can change during loggers and callbacks creation, to use same save dir, we must create here
        self.save_dir = os.path.join(self.cfg.LOG_DIR, time.strftime('%Y_%m_%d_at_%H%M') + '_' + self.cfg.TAG)

        # trainer initialization
        logger_list = self.get_loggers()
        strategy = self.get_strategy()
        callbacks = self.get_callbacks()
        self.checkpoint_module, self.checkpoint_trainer = self.initialize_load_checkpoints()

        trainer = Trainer(
            devices=cfg.GPUS,
            accelerator='gpu',
            strategy=strategy,
            precision=cfg.PRECISION,
            sync_batchnorm=True,
            gradient_clip_val=cfg.GRAD_NORM_CLIP,
            max_epochs=cfg.EPOCHS,
            enable_model_summary=True,
            logger=logger_list,
            callbacks=callbacks,
            log_every_n_steps=cfg.LOGGING_INTERVAL,
            profiler='simple',
            num_nodes=args.num_nodes,
        )
        self.trainer = trainer

        # loader initialization
        trainloader, valloader, _, _ = self.get_loaders(cfg)

        self.trainloader = trainloader
        self.valloader = valloader

        # model initialization
        module = self.initialize_module()

        self.module = module

    @staticmethod
    def import_parser_and_cfg():
        from tairvision_object_detection.models.bev.lss.training.config import get_parser, get_cfg

        return get_parser, get_cfg

    def get_loggers(self):
        save_dir = self.save_dir
        logger_list = []
        if self.args.enable_tensorboard:
            tb_logger = TensorBoardLogger(save_dir=save_dir)
            logger_list.append(tb_logger)
        if self.args.enable_wandb:
            if self.args.wandb_entity_name is None:
                raise ValueError("wandb_entity_name is not set")
            if self.args.wandb_project_name is None:
                raise ValueError("wandb_project_name is not set")
            
            wandb_logger = WandbLogger(entity=self.args.wandb_entity_name,
                                       project=self.args.wandb_project_name,
                                       name=self.cfg.TAG,
                                       save_dir=self.cfg.LOG_DIR,
                                       id=save_dir.split('/')[-1])
            logger_list.append(wandb_logger)
        if self.args.enable_mlflow:
            mlflow_logger = MLFlowLogger(experiment_name="lss_trial",
                                         tags={"mlflow.runName": self.cfg.TAG, "id": save_dir.split('/')[-1]},
                                         save_dir=os.path.join(self.cfg.LOG_DIR, 'mlruns'))
            logger_list.append(mlflow_logger)

        return logger_list

    def get_strategy(self):
        if self.args.disable_distributed_training:
            strategy = None
        elif self.args.init_method == "env://":
            strategy = DDPStrategy(find_unused_parameters=self.args.find_unused_parameters)
        else:
            strategy = MyDDPStrategy(
                find_unused_parameters=self.args.find_unused_parameters, 
                init_method=self.args.init_method,
                )

        return strategy

    def get_callbacks(self):
        callbacks = None

        return callbacks

    def initialize_load_checkpoints(self):
        checkpoint_trainer = None
        checkpoint_module = None

        if self.args.checkpoint is not None:
            if self.args.fresh_start_for_checkpoint:
                checkpoint_module = self.args.checkpoint
                print("weight load mode with fresh start")
            else:
                checkpoint_trainer = self.args.checkpoint
                print("weight load mode with resume from the remaining settings")

        return checkpoint_module, checkpoint_trainer

    @staticmethod
    def get_loaders(cfg):
        augmentation_parameters = get_resizing_and_cropping_parameters(cfg)

        transforms_train = ResizeCropRandomFlipNormalize(augmentation_parameters, enable_random_transforms=True)
        transforms_val = ResizeCropRandomFlipNormalize(augmentation_parameters, enable_random_transforms=False)

        filter_classes = FilterClasses(cfg.DATASET.FILTER_CLASSES, cfg.DATASET.BOX_RESIZING_COEF)
        filter_classes.visibility_list = []
        filter_classes_segm = FilterClasses(cfg.DATASET.FILTER_CLASSES_SEGM, cfg.DATASET.BOX_RESIZING_COEF_SEGM)

        trainloader, valloader = prepare_dataloaders(cfg, transforms_train, transforms_val, filter_classes,
                                                     filter_classes_segm=filter_classes_segm)

        return trainloader, valloader, filter_classes, filter_classes_segm

    def initialize_module(self):
        module_class = self.get_module_class()

        if self.checkpoint_module is None:
            module = module_class(self.cfg.convert_to_dict())
        else:
            print("weights are loaded from checkpoint")
            module = module_class.load_from_checkpoint(self.checkpoint_module, strict=True)

        if self.cfg.PRETRAINED.LOAD_WEIGHTS and not self.checkpoint_module:
            # Load pretrained weights
            pretrained_weights = torch.load(os.path.join(self.cfg.DATASET.DATAROOT, self.cfg.PRETRAINED.PATH),
                                            map_location='cpu')
            state_dict = module.state_dict()

            updated_state_dict = load_pretrained_weights(pretrained_weights, state_dict)
            module.load_state_dict(updated_state_dict, strict=False)
            print(f'Loaded single-image model weights from {self.cfg.PRETRAINED.PATH}')

        return module
    
    def get_module_class(self):
        from tairvision_object_detection.models.bev.lss.training.trainer import TrainingModule
        return TrainingModule

    def train(self):
        self.trainer.fit(self.module, self.trainloader, self.valloader, ckpt_path=self.checkpoint_trainer)

    def test(self):
        self.trainer.test(self.module, self.valloader)


if __name__ == "__main__":
    training_interface = TrainingInterface()
    training_interface.train()
