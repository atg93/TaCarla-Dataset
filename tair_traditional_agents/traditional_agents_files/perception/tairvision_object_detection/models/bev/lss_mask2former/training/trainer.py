from tairvision.models.bev.lss.training.trainer import TrainingModule as TrainingModuleLss
import torch
from tairvision.models.bev.lss_mask2former.blocks.lss_mask2former import LiftSplatLinearMask2Former, LiftSplatTemporalMask2Former, LiftSplatMask2Former
from tairvision.models.bev.lss.utils.bbox import get_targets2d, get_targets3d_xdyd
import torch
import torch.nn.functional as F
from tairvision.ops.boxes import box_xyxy_to_cxcywh, masks_to_boxes


class TrainingModuleMask2former(TrainingModuleLss):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.param_keys = None

    @staticmethod
    def _import_visualization_module():
        from tairvision.models.bev.lss_mask2former.utils_sub.visualization import VisualizationModuleTransformer
        return VisualizationModuleTransformer

    def _import_get_cfg(self):
        from tairvision.models.bev.lss_mask2former.configs.config import get_cfg as get_cfg_mask2former
        self.get_cfg = get_cfg_mask2former

    def _init_model(self):
        depth_channels = (self.cfg.LIFT.D_BOUND[1] - self.cfg.LIFT.D_BOUND[0]) / self.cfg.LIFT.D_BOUND[2]
        if self.cfg.TIME_RECEPTIVE_FIELD > 1:
            model = LiftSplatTemporalMask2Former(self.cfg)
        else:
            model = LiftSplatMask2Former(self.cfg) if depth_channels > 1 else LiftSplatLinearMask2Former(self.cfg)
        return model

    def prepare_targets(self, batch):
        targets = TrainingModuleLss.prepare_targets(self, batch)

        targets_static = self.get_targets_static(batch, receptive_field=self.model.receptive_field)
        number_of_classes = len(self.cfg.MODEL.HEADDYNAMIC.SEMANTIC_SEG.WEIGHTS)

        targets_list = []
        batch_size = targets["segmentation"].shape[0]
        for i in range(batch_size):
            segmentation_mask = targets["segmentation"][i]
            background_mask = torch.zeros_like(segmentation_mask)
            background_mask[segmentation_mask == 0] = 1
            masks = background_mask[0]
            labels = torch.zeros([1], dtype=torch.int64, device=segmentation_mask.device)[:, None]
            for j in range(number_of_classes - 1):
                masks_class = F.one_hot(targets["instance"][i, 0, j]).permute(2, 0, 1).bool()
                masks_class = masks_class[1:]
                
                if masks_class.shape[0] > 0:
                    filter_count = torch.sum(masks_class.view(masks_class.shape[0], -1), 1)
                    masks_class = masks_class[filter_count > 0]
                labels_class = (j + 1) * torch.ones([masks_class.shape[0]], dtype=torch.int64, device=masks.device)[:, None]

                if masks_class.shape[0] > 0:
                    masks = torch.vstack([masks, masks_class])
                    labels = torch.vstack([labels, labels_class])

            target_dict = {"masks": masks, "labels": labels[:, 0]}

            # TODO, add regression targets for future 3D detection task from here
            # box_targets = masks_to_boxes(background_mask[0])
            # box_targets = torch.vstack([box_targets, targets3d[i]["boxes"]])

            # regressions = self._target_regressions_for_z_yaw_loss(targets, masks, i)
            # target_dict.update({"regressions": regressions})

            box_targets = masks_to_boxes(masks)
            boxes = box_xyxy_to_cxcywh(box_targets)
            boxes = boxes / torch.tensor([200, 200, 200, 200], dtype=boxes.dtype, device=boxes.device)
            target_dict.update({"boxes": boxes})
            targets_list.append(target_dict)

        targets_added = {"dab": targets_list, "segmentation": targets["segmentation"], "instance": targets["instance"], "targets_static": targets_static}
        return targets_added
    
    def _target_regressions_for_z_yaw_loss(self, targets, masks, index):
            z_values = targets["z_position"][index][0]
            z_values[z_values == 255] = 0
            yaw_angles = targets["yaw_angles"][index][0]
            nonzero_indices = torch.nonzero(masks)
            split_sizes = torch.bincount(nonzero_indices[:, 0], minlength=masks.shape[0])
            selected_elements_z = torch.split(z_values[0, nonzero_indices[:, 1], nonzero_indices[:, 2]],
                                                split_sizes.tolist())
            selected_elements_h = torch.split(z_values[1, nonzero_indices[:, 1], nonzero_indices[:, 2]],
                                                split_sizes.tolist())
            selected_elements_yaw = torch.split(yaw_angles[0, nonzero_indices[:, 1], nonzero_indices[:, 2]],
                                                split_sizes.tolist())
            zs = torch.stack([elem[0] for elem in selected_elements_z])[None, :]
            hs = torch.stack([elem[0] for elem in selected_elements_h])[None, :]
            yaw_values = torch.stack([elem[0] for elem in selected_elements_yaw])[None, :]

            sin_yaw_values = torch.sin(yaw_values)
            cos_yaw_values = torch.cos(yaw_values)

            zh_values = torch.cat([zs, hs], 0)
            yaw_per_object = torch.cat([sin_yaw_values, cos_yaw_values], 0)
            regressions = torch.cat([zh_values, yaw_per_object], 0)

            regressions = regressions.transpose(1, 0).contiguous()
            return regressions

    def configure_optimizers(self):
        params = {}
        optimizer_main_config = self.cfg.OPTIMIZER.CONFIG
        optimizer_main_config = {k.lower(): v for k, v in optimizer_main_config.items()}

        if len(self.cfg.OPTIMIZER.PARAMS.NAMES) > 0:
            for index, key in enumerate(self.cfg.OPTIMIZER.PARAMS.NAMES):
                config = self.cfg.OPTIMIZER.PARAMS.CONFIGS[index]
                for config_key in config.keys():
                    config[config_key] = float(config[config_key])
                params[key] = self.cfg.OPTIMIZER.PARAMS.CONFIGS[index]

        param_keys = ["global"]
        param_keys += list(params.keys())
        self.param_keys = param_keys
        optimizer_dict = torch.optim.__dict__
        if params == {}:
            param_list = []
            for model_param_name, model_param in self.model.named_parameters():
                if model_param.requires_grad:
                    param_list.append(model_param)
                    print(model_param_name)
            optimizer = optimizer_dict[self.cfg.OPTIMIZER.NAME](param_list, **optimizer_main_config)
        else:
            param_dict_of_list = {}
            for param_key in params.keys():
                param_dict_of_list[param_key] = []
            param_dict_of_list["global"] = []

            for model_param_name, model_param in self.model.named_parameters():
                added = False
                for param_key in params.keys():
                    if param_key in model_param_name and model_param.requires_grad:
                        param_dict_of_list[param_key].append(model_param)
                        added = True
                if not added and model_param.requires_grad:
                    param_dict_of_list["global"].append(model_param)

            param_list = []
            param_list.append({"params": param_dict_of_list["global"]})
            for param_key, param_config in params.items():
                param_list.append({"params": param_dict_of_list[param_key], **param_config})

            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            optimizer_n_params = 0
            for param_key in params.keys():
                optimizer_n_params += sum(param.numel() for param in param_dict_of_list[param_key])

            optimizer_n_params += sum(param.numel() for param in param_dict_of_list["global"])
            assert n_parameters == optimizer_n_params, \
                "total number of trainable parameters does not check, please check the parameter keys again"

            optimizer = optimizer_dict[self.cfg.OPTIMIZER.NAME](param_list, **optimizer_main_config)

        return optimizer
