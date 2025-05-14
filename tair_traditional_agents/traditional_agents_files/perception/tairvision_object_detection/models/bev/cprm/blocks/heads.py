import torch
import torch.nn as nn
from tairvision.models.bev.cprm.blocks.centerpoint_head import CenterHead


class BEVDETAdaptor(CenterHead):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        self.use_lidar_head = cfg.USE_LIDAR_HEAD
        self.task_num = len(cfg.MODEL.CPHEAD.NMS_THR)
        self.task_combined_weights = False
        self.combined_for_loss_types = False
        if self.task_combined_weights:
            for i in range(self.task_num):
                assert self.combined_for_loss_types is False
                setattr(self, 'task_weight' + str(i), nn.Parameter(torch.tensor(0.0), requires_grad=True))
                setattr(self, 'heatmap_weight' + str(i), nn.Parameter(torch.tensor(1.386), requires_grad=True))
        elif self.combined_for_loss_types:
            self.yaw_weight_combined = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.xy_weight_combined = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.z_weight_combined = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.whl_weight_combined = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.heatmap_weight_combined = nn.Parameter(torch.tensor(1.386), requires_grad=True)
        else:
            for i in range(self.task_num):
                setattr(self, 'xy_weight' + str(i), nn.Parameter(torch.tensor(0.0), requires_grad=True))
                setattr(self, 'z_weight' + str(i), nn.Parameter(torch.tensor(0.0), requires_grad=True))
                setattr(self, 'whl_weight' + str(i), nn.Parameter(torch.tensor(0.0), requires_grad=True))
                setattr(self, 'yaw_weight' + str(i), nn.Parameter(torch.tensor(0.0), requires_grad=True))
                setattr(self, 'heatmap_weight' + str(i), nn.Parameter(torch.tensor(1.386), requires_grad=True))


    def get_head_outputs(self, features):
        features = features['y']
        B, S, C, H, W = features.shape
        features = features.view(B*S, C, H, W)
        head_outputs = self(features)
        return head_outputs

    def get_loss(self, head_outputs, gts):
        loss = self.loss(gts, head_outputs)
        factor = {}

        if self.task_combined_weights:
            assert self.combined_for_loss_types is False
            for i in range(self.task_num):
                exec(f'task_factor{i} = 1 / torch.exp(self.task_weight{i})')
                exec(f'heatmap_factor{i} = 1 / torch.exp(self.heatmap_weight{i})')

                exec(f"loss_yaw{i} = loss.pop('loss_task{i}.loss_yaw')")
                exec(f"loss_heatmap{i} = loss.pop('loss_task{i}.loss_heatmap')")
                exec(f"loss_xy{i} = loss.pop('loss_task{i}.loss_xy')")
                exec(f"loss_z{i} = loss.pop('loss_task{i}.loss_z')")
                exec(f"loss_whl{i} = loss.pop('loss_task{i}.loss_whl')")

                exec(f"loss['loss_task{i}.loss_heatmap'] = loss_heatmap{i} * heatmap_factor{i}")
                exec(f"loss['loss_task{i}.loss_yaw'] = loss_yaw{i} * task_factor{i}")
                exec(f"loss['loss_task{i}.loss_xy'] = loss_xy{i} * task_factor{i}")
                exec(f"loss['loss_task{i}.loss_z'] = loss_z{i} * task_factor{i}")
                exec(f"loss['loss_task{i}.loss_whl'] = loss_whl{i} * task_factor{i}")

                exec(f"loss['uncertainty_task{i}'] = 0.5 * self.task_weight{i}")
                exec(f"loss['uncertainty_heatmap{i}'] = 0.5 * self.heatmap_weight{i}")
                exec(f"factor['task{i}'] = task_factor{i}")
                exec(f"factor['heatmap{i}'] = heatmap_factor{i}")
        elif self.combined_for_loss_types:
            yaw_factor_combined = 1 / torch.exp(self.yaw_weight_combined)
            xy_factor_combined = 1 / torch.exp(self.xy_weight_combined)
            z_factor_combined = 1 / torch.exp(self.z_weight_combined)
            whl_factor_combined = 1 / torch.exp(self.whl_weight_combined)
            heatmap_factor_combined = 1 / torch.exp(self.heatmap_weight_combined)
            for i in range(self.task_num):
                exec(f"loss_yaw{i} = loss.pop('loss_task{i}.loss_yaw')")
                exec(f"loss_heatmap{i} = loss.pop('loss_task{i}.loss_heatmap')")
                exec(f"loss_xy{i} = loss.pop('loss_task{i}.loss_xy')")
                exec(f"loss_z{i} = loss.pop('loss_task{i}.loss_z')")
                exec(f"loss_whl{i} = loss.pop('loss_task{i}.loss_whl')")

                exec(f"loss['loss_task{i}.loss_heatmap'] = loss_heatmap{i} * heatmap_factor_combined")
                exec(f"loss['loss_task{i}.loss_yaw'] = loss_yaw{i} * yaw_factor_combined")
                exec(f"loss['loss_task{i}.loss_xy'] = loss_xy{i} * xy_factor_combined")
                exec(f"loss['loss_task{i}.loss_z'] = loss_z{i} * z_factor_combined")
                exec(f"loss['loss_task{i}.loss_whl'] = loss_whl{i} * whl_factor_combined")

            loss['uncertainty_yaw_combined'] = 0.5 * self.yaw_weight_combined
            loss['uncertainty_xy_combined'] = 0.5 * self.xy_weight_combined
            loss['uncertainty_z_combined'] = 0.5 * self.z_weight_combined
            loss['uncertainty_whl_combined'] = 0.5 * self.whl_weight_combined
            loss['uncertainty_heatmap_combined'] = 0.5 * self.heatmap_weight_combined
            factor['heatmap'] = heatmap_factor_combined
            factor['yaw'] = yaw_factor_combined
            factor['xy'] = xy_factor_combined
            factor['z'] = z_factor_combined
            factor['whl'] = whl_factor_combined
        else:
            for i in range(self.task_num):
                exec(f'yaw_factor{i} = 1 / torch.exp(self.yaw_weight{i})')
                exec(f'heatmap_factor{i} = 1 / torch.exp(self.heatmap_weight{i})')
                exec(f'xy_factor{i} = 1 / torch.exp(self.xy_weight{i})')
                exec(f'z_factor{i} = 1 / torch.exp(self.z_weight{i})')
                exec(f'whl_factor{i} = 1 / torch.exp(self.whl_weight{i})')

                exec(f"loss_yaw{i} = loss.pop('loss_task{i}.loss_yaw')")
                exec(f"loss_heatmap{i} = loss.pop('loss_task{i}.loss_heatmap')")
                exec(f"loss_xy{i} = loss.pop('loss_task{i}.loss_xy')")
                exec(f"loss_z{i} = loss.pop('loss_task{i}.loss_z')")
                exec(f"loss_whl{i} = loss.pop('loss_task{i}.loss_whl')")

                exec(f"loss['loss_task{i}.loss_yaw'] = loss_yaw{i} * yaw_factor{i}")
                exec(f"loss['loss_task{i}.loss_heatmap'] = loss_heatmap{i} * heatmap_factor{i}")
                exec(f"loss['loss_task{i}.loss_xy'] = loss_xy{i} * xy_factor{i}")
                exec(f"loss['loss_task{i}.loss_z'] = loss_z{i} * z_factor{i}")
                exec(f"loss['loss_task{i}.loss_whl'] = loss_whl{i} * whl_factor{i}")

                exec(f"loss['uncertainty_yaw{i}'] = 0.5 * self.yaw_weight{i}")
                exec(f"loss['uncertainty_heatmap{i}'] = 0.5 * self.heatmap_weight{i}")
                exec(f"loss['uncertainty_xy{i}'] = 0.5 * self.xy_weight{i}")
                exec(f"loss['uncertainty_z{i}'] = 0.5 * self.z_weight{i}")
                exec(f"loss['uncertainty_whl{i}'] = 0.5 * self.whl_weight{i}")

                exec(f"factor['yaw{i}'] = yaw_factor{i}")

        return loss, factor

    def get_detections(self, head_outputs):
        detections = self.get_bboxes(head_outputs)
        return detections
