import logging

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import Accuracy

from carla_gym.core.obs_manager.birdview.planning.training.PlanT.model import HFLM
import pickle
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class LitHFLM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.last_epoch = 0
        model_path = "/home/tg22/remote-pycharm/roach/carla-roach/carla_gym/core/obs_manager/birdview/planning/checkpoints/PlanT/3x/PlanT_medium/cfg.pickle"

        with open(model_path, 'rb') as fp:
            cfg_model = pickle.load(fp)
        cfg = {'user': {'working_dir': '/home/geiger/krenz73/coding/02_sequential_driving/release/plant'}, 'model': {'name': 'PlanT', 'training': {'max_epochs': 49, 'batch_size': 32, 'learning_rate': 0.0001, 'betas': [0.9, 0.95], 'grad_norm_clip': 1.0, 'weight_decay': 0.1, 'ckpt_path': 'log/', 'num_workers': 8, 'pred_len': 4, 'seq_len': 1, 'max_NextRouteBBs': 2, 'input_ego': False, 'remove_velocity': 'None', 'route_only_wp': False, 'remove_back': False, 'pretraining_path': 'none', 'add_noise': False}, 'pre_training': {'pretraining': 'forecast', 'multitask': True, 'forecastLoss_weight': 0.2, 'future_timestep': 1, 'quantize': True, 'precision_pos': 7, 'precision_speed': 4, 'precision_angle': 5}, 'network': {'hf_checkpoint': 'prajjwal1/bert-medium', 'embd_pdrop': 0.1}}, 'exp_folder_name': '1_PlanT_release', 'lrDecay_epoch': 45, 'seed': 1234, 'debug': False, 'visualize': True, 'overfit': 0, 'resume': True, 'use_caching': True, 'custom_sampler': False, 'gpus': 4, 'trainset_size': 3, 'benchmark': 'longest6', 'expname': '20_multitask_forecasting', 'wandb_name': 'training_pamidata_onlybrakeloss_${hydra:job.override_dirname}', 'save_dir': '${hydra:run.dir}', 'data_dir': '/home/geiger/krenz73/coding/02_sequential_driving/release/plant/data/carla/pami_bb_dataset_27_09_22_v4_1'}
        self.cfg = cfg
        self.cfg = OmegaConf.create(cfg)

        self.cfg_train = self.cfg.model.training#tugrul
        self.model = HFLM(self.cfg.model.network, self.cfg)

        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_forecast = nn.CrossEntropyLoss(ignore_index=-999)
        
        # Metrics
        if self.cfg.model.pre_training.get("pretraining", "none") == "forecast":
            self.metrics_forecasting_acc = nn.ModuleList(
                [Accuracy() for i in range(self.model.num_attributes)]
            )
            

    def forward(self, x, y=None, target_point=None, light_hazard=None):
        return self.model(x, y, target_point, light_hazard)


    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(self.cfg.model.training)
        scheduler = MultiStepLR(
            optimizer,
            milestones=[self.cfg.lrDecay_epoch, self.cfg.lrDecay_epoch + 10],
            gamma=0.1,
        )
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        x, y, wp, tp, light = batch

        # training with only waypoint loss
        if self.cfg.model.pre_training.get("pretraining", "none") == "none":
            logits, pred_wp, _ = self(x, y, tp, light)

            loss_pred = F.l1_loss(pred_wp, wp)
            loss_all = loss_pred

            self.log(
                "train/loss_pred",
                loss_pred,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )

        elif self.cfg.model.pre_training.get("pretraining", "none") == "forecast":

            if self.cfg.model.pre_training.get("multitask", False) == True:
                # multitask training
                logits, targets, pred_wp, _ = self(x, y, tp, light)
                loss_wp = F.l1_loss(pred_wp, wp)
                losses_forecast = [
                    torch.mean(self.criterion_forecast(logits[i], targets[i].squeeze()))
                    for i in range(len(logits))
                ]
                loss_forecast = torch.mean(torch.stack(losses_forecast))

                loss_all = (
                    1                                                           * loss_wp
                    + self.cfg.model.pre_training.get("forecastLoss_weight", 0) * loss_forecast
                )
                self.log(
                    "train/loss_forecast",
                    loss_forecast,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )
                self.log(
                    "train/loss_wp",
                    loss_wp,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )

            else:
                # 2 stage training (pre-training only on forecasting - no waypoint loss)
                logits, targets = self(x, y)

                losses_forecast = [
                    torch.mean(self.criterion_forecast(logits[i], targets[i].squeeze()))
                    for i in range(len(logits))
                ]
                loss_all = torch.mean(torch.stack(losses_forecast))

            self.log(
                "train/loss_all",
                loss_all,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )

            for i, name in enumerate(
                ["x", "y", "yaw", "speed", "extent_x", "extent_y"]
            ):
                if i > self.model.num_attributes:
                    break
                self.log(
                    f"train/loss_{name}",
                    losses_forecast[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )

                mask = targets[i].squeeze() != -999
                self.metrics_forecasting_acc[i](
                    logits[i][mask], targets[i][mask].squeeze()
                )
                self.log(
                    f"train/acc_{name}",
                    self.metrics_forecasting_acc[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )

        return loss_all


    def validation_step(self, batch, batch_idx):

        if self.cfg.model.pre_training.get("pretraining", "none") == "forecast":
            # TODO: add proper validation set for multitask
            pass

        else:
            x, y, wp, tp, light = batch

            self.y = y
            logits, pred_wp, _ = self(x, y, tp, light)

            loss_pred = F.l1_loss(pred_wp, wp)
            loss_all = loss_pred

            self.log(
                "val/loss_all",
                loss_all,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )
            self.log(
                "val/loss_pred",
                loss_pred,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )

            self.last_epoch = self.current_epoch


    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg_train.grad_norm_clip
        )
