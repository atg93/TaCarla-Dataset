from tairvision.models.bev.lss_mask2former.training.trainer import TrainingModuleMask2former
import torch
from tairvision.models.bev.lss_mask2former.blocks.lss_mask2former_topology import LiftSplatLinearMask2FormerTopology, LiftSplatTemporalMask2FormerTopology, LiftSplatMask2FormerTopology
import torch
import torch.nn.functional as F
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import BinaryStatScores
from tairvision.models.bev.lss.training.metrics import IntersectionOverUnion, PanopticMetric


class TrainingModuleMask2FormerOpenLaneV2(TrainingModuleMask2former):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.param_keys = None
        # This is the direction labels of the proposed method. 
        # The labels correspond to up down left right, respectively. 
        # Centerline concept has the direction information and 
        # the proposed method encodes the direction information in this way.
        self.class_names = ['background', 'dir1', 'dir2', 'dir3', 'dir4']

        self.lclc_metrics_target = None
        self.lcte_metrics_target = None

    @staticmethod
    def _import_visualization_module():
        from tairvision.models.bev.lss_mask2former.utils_sub.visualization_openlanev2 import VisualizationModuleOpenLaneV2
        return VisualizationModuleOpenLaneV2
    
    def _import_get_cfg(self):
        from tairvision.models.bev.lss_mask2former.configs.config_openlanev2 import get_cfg as get_cfg_mask2former_openlanev2
        self.get_cfg = get_cfg_mask2former_openlanev2
    
    def _initialize_metrics(self):
        self.metric_iou_val = IntersectionOverUnion(self.n_classes)
        self.metric_panoptic_val = PanopticMetric(n_classes=self.n_classes)

        # Panoptic metric design has a capability of measuring only single class. 
        # Therefore, separate metrics are needed for each class.
        self.metric_panoptic_class_specific_dict = torch.nn.ModuleDict(
            {
                "dir1": PanopticMetric(n_classes=2),
                "dir2": PanopticMetric(n_classes=2),
                "dir3": PanopticMetric(n_classes=2),
                "dir4": PanopticMetric(n_classes=2),
            }
        )

        # This is for measuring the performance after the relationship module.
        # Relationship module (Query Interaction between traffic elements and centerline). 
        # This is still in R&D phase.
        self.metric_panoptic_class_specific_dict_after_centerline = torch.nn.ModuleDict(
            {
                "dir1": PanopticMetric(n_classes=2),
                "dir2": PanopticMetric(n_classes=2),
                "dir3": PanopticMetric(n_classes=2),
                "dir4": PanopticMetric(n_classes=2),
            }
        )

        # For the traffic element performance. 
        if self.model.head2d is not None:
            self.metric_ap = MeanAveragePrecision()

        # 
        if self.model.optimal_transport_lclc is not None:
            self.metric_lclc = BinaryStatScores()
            self.metric_lclc_only_targets = BinaryStatScores()

        if self.model.optimal_transport_lcte is not None:
            self.metric_lcte = BinaryStatScores()
            self.metric_lcte_only_targets = BinaryStatScores()

        if self.model.centerline_last is not None:
            self.metric_iou_val_centerline_last = IntersectionOverUnion(self.n_classes)
            self.metric_panoptic_val_centerline_last = PanopticMetric(n_classes=self.n_classes)

    def prepare_targets(self, targets):
        current_frame_idx = self.cfg.TIME_RECEPTIVE_FIELD - 1

        center_line_segmentation = targets["center_line_segmentation"].contiguous().long()
        center_line_instance = targets["center_line_instance"].contiguous().long()
        
        targets_list, target_list_ordered = self._create_instance_targets(targets, current_frame_idx)
        targets2d_loss, targets2d_metrics, orig_target_sizes = self._create_2D_targets(targets, current_frame_idx)

        targets_added = {
            "dab": targets_list, 
            "dab_ordered": target_list_ordered,
            "segmentation": center_line_segmentation[:, current_frame_idx][:, None],
            "view_inv": targets["view"].inverse()[:, current_frame_idx][:, None],
            "instance": center_line_instance[:, current_frame_idx][:, None], 
            "lclc_list" : [[target[current_frame_idx]] for target in targets["lclc_list"]],
            "lcte_list" : [[target[current_frame_idx]] for target in targets["lcte_list"]],
            "targets2d_loss": targets2d_loss,
            "targets2d_metrics": targets2d_metrics,
            "orig_target_sizes": torch.vstack(orig_target_sizes)
            }
        return targets_added
    
    def _create_instance_targets(self, targets, current_frame_idx):
        number_of_classes = 5
        targets_list = []
        target_list_ordered = []
        center_line_segmentation = targets["center_line_segmentation"].contiguous().long()
        center_line_instance = targets["center_line_instance"].contiguous().long()
        center_line_instance_ordered = targets["center_line_instance_ordered"].contiguous().long()
        ordered_attirbute_list = targets["ordered_attribute_list"]
        bezier_curves_list = targets["bezier_list"]
        batch_size = center_line_segmentation.shape[0]
        for i in range(batch_size):
            segmentation_mask = center_line_segmentation[i]
            bezier_curves = bezier_curves_list[i][current_frame_idx]
            background_mask = torch.zeros_like(segmentation_mask)
            background_mask[segmentation_mask == 0] = 1
            masks = background_mask[current_frame_idx] # The last one in the sequence is the current frame
            labels = torch.zeros([1], dtype=torch.int64, device=segmentation_mask.device)[:, None]
            beziers = torch.zeros([1, bezier_curves.shape[1]], dtype=torch.float32, device=segmentation_mask.device)
            for j in range(number_of_classes - 1):
                masks_class = F.one_hot(center_line_instance[i, current_frame_idx, j]).permute(2, 0, 1).bool() # -1 is to get the current frame
                masks_class = masks_class[1:]
                
                unique_indices = torch.unique(center_line_instance[i, current_frame_idx, j]) - 1
                beziers_class = bezier_curves[unique_indices[1:]]
                if masks_class.shape[0] > 0:
                    filter_count = torch.sum(masks_class.view(masks_class.shape[0], -1), 1)
                    masks_class = masks_class[filter_count > 0]
                labels_class = (j + 1) * torch.ones([masks_class.shape[0]], dtype=torch.int64, device=masks.device)[:, None]

                if masks_class.shape[0] > 0:
                    masks = torch.vstack([masks, masks_class])
                    labels = torch.vstack([labels, labels_class])
                    beziers = torch.vstack([beziers, beziers_class])

            target_dict = {"masks": masks, "labels": labels[:, 0], "regressions": beziers}
            targets_list.append(target_dict)

            masks_class_ordered = F.one_hot(center_line_instance_ordered[i, current_frame_idx, 0]).permute(2, 0, 1).bool()
            masks_class_ordered = masks_class_ordered[1:]
            ordered_labels = torch.tensor(ordered_attirbute_list[i][current_frame_idx], dtype=torch.int64, device=masks.device)
            target_dict_ordered = {"masks": masks_class_ordered, "labels": ordered_labels}
            target_list_ordered.append(target_dict_ordered)
        
        return targets_list, target_list_ordered
    
    def _create_2D_targets(self, targets, current_frame_idx):
        targets2d_raw = targets["targets2d"]
        targets2d_loss = []
        targets2d_metrics = []
        orig_target_sizes = []
        for i in range(len(targets2d_raw)):
            targets2d_loss_dict = {
                "boxes": targets2d_raw[i][current_frame_idx]["boxes"],
                "labels": targets2d_raw[i][current_frame_idx]["labels"], 
                }
            targets2d_metrics_dict = {
                "boxes": targets2d_raw[i][current_frame_idx]["correctly_resized_boxes"],
                "labels": targets2d_raw[i][current_frame_idx]["labels"], 
                "areas": targets2d_raw[i][current_frame_idx]["areas"],
                "orig_target_sizes": targets2d_raw[i][current_frame_idx]["orig_target_sizes"]
            }
            orig_target_sizes.append(targets2d_raw[i][current_frame_idx]["orig_target_sizes"])
            targets2d_loss.append(targets2d_loss_dict)
            targets2d_metrics.append(targets2d_metrics_dict)

        return targets2d_loss, targets2d_metrics, orig_target_sizes

    def forward(self, batch):
        image = batch['images']
        intrinsics = batch['intrinsics']
        extrinsics = batch['cams_to_lidar']
        view = batch['view']
        front_view_images = batch["front_view_images"]
        future_egomotion = batch['future_egomotion']
        front_view_images_processed = []
        for i in range(len(front_view_images)):
            front_view_images_processed.append(front_view_images[i][0])

        # Forward pass
        output = self.model(image, intrinsics, extrinsics, view,
                             front_view_image=front_view_images_processed, future_egomotion=future_egomotion)

        return output
    
    def get_losses(self, output, targets):
        loss, factor = {}, {}

        loss_centerline_segm, factor_centerline_segm = self.model.centerline_head.get_loss(output, targets)
        loss.update(loss_centerline_segm)
        factor.update(factor_centerline_segm)

        if self.model.head2d is not None:
            loss_detection2d = self.model.head2d.get_loss(output['head2d'], targets["targets2d_loss"])
            loss_detection2d_with_new_keys = {}
            for key in loss_detection2d:
                loss_detection2d_with_new_keys["head2d/" + key] = loss_detection2d[key]
            loss.update(loss_detection2d_with_new_keys)
            # factor.update(factor_detection2d)

        if self.model.optimal_transport_lclc:
            loss_lclc, lclc_metrics = self.model.optimal_transport_lclc.get_loss(output, targets)
            self.lclc_metrics_targets = lclc_metrics

            if isinstance(loss_lclc, list):
                loss_dict = {}
                for i in range(len(loss_lclc) - 1):
                    loss_dict.update({"relation_loss/loss_lclc_" + str(i): loss_lclc[i]})
                loss_dict.update({"relation_loss/loss_lclc": loss_lclc[-1]})
            else:
                loss_dict = {"relation_loss/loss_lclc": loss_lclc}
            loss.update(loss_dict)
        

        if self.model.optimal_transport_lcte:
            loss_lcte, lcte_metrics = self.model.optimal_transport_lcte.get_loss(output, targets)
            self.lcte_metrics_targets = lcte_metrics
            
            if isinstance(loss_lcte, list):
                loss_dict = {}
                for i in range(len(loss_lcte) - 1):
                    loss_dict.update({"relation_loss/loss_lcte_" + str(i): loss_lcte[i]})
                loss_dict.update({"relation_loss/loss_lcte": loss_lcte[-1]})
            else:
                loss_dict = {"relation_loss/loss_lcte": loss_lcte}
            loss.update(loss_dict)

        if self.model.centerline_last is not None:
            loss_centerline_last = self.model.centerline_last.get_loss(output, targets)
            loss_centerline_last_new_keys = {}
            for key, value in loss_centerline_last.items():
                loss_centerline_last_new_keys["centerline_last/" + key] = value
            loss.update(loss_centerline_last_new_keys)

        return loss, factor

    def accumulate_metrics(self, output, targets, batch):
        post_output = self.model.centerline_head.post_process(output)            
        self.metric_iou_val(post_output["segm"], targets['segmentation'])
        self.metric_panoptic_val(post_output["inst"], targets['instance'])

        for i in range(1, 5):
            self.metric_panoptic_class_specific_dict[f"dir{i}"](
                post_output["inst"][:, :, i-1].unsqueeze(2), 
                targets['instance'][:, :, i-1].unsqueeze(2)
            )

        if self.model.head2d is not None:
            output['head2d'].update({"orig_target_sizes": targets["orig_target_sizes"]})
            post_outputs_2d = self.model.head2d.postprocess(output['head2d'])
            self.metric_ap(post_outputs_2d, targets["targets2d_metrics"])

        if self.model.optimal_transport_lclc:
            for i in range(len(self.lclc_metrics_targets["predictions"])):
                self.metric_lclc_only_targets(self.lclc_metrics_targets["predictions"][i], self.lclc_metrics_targets["targets"][i])

            for i in range(len(self.lclc_metrics_targets["predictions_all_queries"])):
                self.metric_lclc(
                    self.lclc_metrics_targets["predictions_all_queries"][i],
                    self.lclc_metrics_targets["targets_all_queries"][i]
                )
            
            self.lclc_metrics_targets = None

        if self.model.optimal_transport_lcte:
            for i in range(len(self.lcte_metrics_targets["predictions"])):
                self.metric_lcte_only_targets(self.lcte_metrics_targets["predictions"][i], self.lcte_metrics_targets["targets"][i])
            
            for i in range(len(self.lcte_metrics_targets["predictions_all_queries"])):
                self.metric_lcte(
                    self.lcte_metrics_targets["predictions_all_queries"][i],
                    self.lcte_metrics_targets["targets_all_queries"][i]
                )

            self.lcte_metrics_targets = None

        if self.model.centerline_last is not None:
            post_output = self.model.centerline_last.post_process(output["centerline_after_relation"])
            self.metric_iou_val_centerline_last(post_output["segm"], targets['segmentation'])
            self.metric_panoptic_val_centerline_last(post_output["inst"], targets['instance'])

            for i in range(1, 5):
                self.metric_panoptic_class_specific_dict_after_centerline[f"dir{i}"](
                    post_output["inst"][:, :, i-1].unsqueeze(2), 
                    targets['instance'][:, :, i-1].unsqueeze(2)
                )

    def visualize(self, batch, output, targets, batch_idx, prefix='train'):
        super().visualize(batch, output, targets, batch_idx, prefix)
        
    def visualize_extention(self, batch, output, targets, batch_idx, prefix='train'):
        # TODO, this part will be extended on top of the main visualization function
        """
            0: 'ring_front_center'
            1: 'ring_front_left'
            2: 'ring_front_right'
            3: 'ring_rear_left'
            4: 'ring_rear_right'
            5: 'ring_side_left'
            6: 'ring_side_right'

        """
        name = f'media_{prefix}/{batch_idx}'

        image_index_list = [5, 1, 0, 2, 6]
        front_view_images = None
        image_index_list = [6, 4, 3, 5]
        rear_view_images = None

        front_view_images = self.visualization_module._visualize_images_with_centerlines(batch, targets, image_index_list)
        rear_view_images = self.visualization_module._visualize_images_with_centerlines(batch, targets, image_index_list)

        for logger in self.loggers:
            if "WandbLogger" in logger.__str__():
                if front_view_images is not None:
                    logger.experiment.log({f"{name}_front_view_gt": [wandb.Image(front_view_images)]})
                if rear_view_images is not None:
                    logger.experiment.log({f"{name}_rear_view_gt": [wandb.Image(rear_view_images)]})

    def post_process_outputs(self, output):
        # TODO, this part will be enlargened in the future
        post_output = self.model.centerline_head.post_process(output)
        return post_output
    
    def _init_model(self):
        depth_channels = (self.cfg.LIFT.D_BOUND[1] - self.cfg.LIFT.D_BOUND[0]) / self.cfg.LIFT.D_BOUND[2]
        if self.cfg.TIME_RECEPTIVE_FIELD > 1:
            model = LiftSplatTemporalMask2FormerTopology(self.cfg)
        else:
            model = LiftSplatMask2FormerTopology(self.cfg) if depth_channels > 1 else LiftSplatLinearMask2FormerTopology(self.cfg)
        return model
    
    def compute_metrics(self):  
        class_names = ['background', 'dir1', 'dir2', 'dir3', 'dir4']

        scores = self.metric_iou_val.compute()
        for key, value in zip(class_names, scores["iou"]):
            self.log('metrics/val_iou_' + key, value.item(), prog_bar=False, sync_dist=True)
        self.metric_iou_val.reset()

        scores = self.metric_panoptic_val.compute()
        for key, value in scores.items():
            for instance_name, score in zip(['background', 'centerlines'], value):
                if instance_name != 'background':
                    self.log(f'metrics/val_{key}_{instance_name}', score.item(), prog_bar=False, sync_dist=True)
        self.metric_panoptic_val.reset()

        for class_name in class_names:
            if class_name == 'background':
                continue

            scores = self.metric_panoptic_class_specific_dict[class_name].compute()
            for key, value in scores.items():
                for instance_name, score in zip(['background', class_name], value):
                    if instance_name != 'background':
                        self.log(f'metrics/val_{key}_{instance_name}', score.item(), prog_bar=False, sync_dist=True)
            self.metric_panoptic_class_specific_dict[class_name].reset()

        if self.model.head2d is not None:
            scores = self.metric_ap.compute()
            for key, value in scores.items():
                self.log(f'metrics_ap/val_{key}', value.item(), prog_bar=False, sync_dist=True)
            self.metric_ap.reset()


        if self.model.optimal_transport_lclc:
            result = self.metric_lclc.compute()
            result_dict = self._calculate_f1_related_metrics(result)
            for key, value in result_dict.items():
                self.log(f'metrics_relation/lclc_{key}', value.item(), prog_bar=False, sync_dist=True)
            self.metric_lclc.reset()

            result = self.metric_lclc_only_targets.compute()
            result_dict = self._calculate_f1_related_metrics(result)
            for key, value in result_dict.items():
                self.log(f'metrics_relation/lclc_targets_{key}', value.item(), prog_bar=False, sync_dist=True)
            self.metric_lclc_only_targets.reset()
        
        if self.model.optimal_transport_lcte:
            result = self.metric_lcte.compute()
            result_dict = self._calculate_f1_related_metrics(result)
            for key, value in result_dict.items():
                self.log(f'metrics_relation/lcte_{key}', value.item(), prog_bar=False, sync_dist=True)
            self.metric_lcte.reset()

            result = self.metric_lcte_only_targets.compute()
            result_dict = self._calculate_f1_related_metrics(result)
            for key, value in result_dict.items():
                self.log(f'metrics_relation/lcte_targets_{key}', value.item(), prog_bar=False, sync_dist=True)
            self.metric_lcte_only_targets.reset()

        if self.model.centerline_last is not None:
            scores = self.metric_iou_val_centerline_last.compute()
            for key, value in zip(class_names, scores["iou"]):
                self.log('centerline_last/val_iou_' + key, value.item(), prog_bar=False, sync_dist=True)
                # self.logger.experiment.add_scalar('val_iou_' + key, value, global_step=self.training_step_count)
            self.metric_iou_val_centerline_last.reset()

            scores = self.metric_panoptic_val_centerline_last.compute()
            for key, value in scores.items():
                for instance_name, score in zip(['background', 'centerlines'], value):
                    if instance_name != 'background':
                        self.log(f'centerline_last/val_{key}_{instance_name}', score.item(), prog_bar=False, sync_dist=True)
            self.metric_panoptic_val_centerline_last.reset()

            for class_name in class_names:
                if class_name == 'background':
                    continue
                
                scores = self.metric_panoptic_class_specific_dict_after_centerline[class_name].compute()
                for key, value in scores.items():
                    for instance_name, score in zip(['background', class_name], value):
                        if instance_name != 'background':
                            self.log(f'centerline_last/val_{key}_{instance_name}', score.item(), prog_bar=False, sync_dist=True)
                self.metric_panoptic_class_specific_dict_after_centerline[class_name].reset()

    def _calculate_f1_related_metrics(self, result):
        true_positive = result[0]
        false_positive = result[1]
        false_negative = result[3]
        precision = true_positive / (true_positive + false_positive + 1e-6)
        recall = true_positive / (true_positive + false_negative + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        result_dict = {"precision": precision, "recall": recall, "f1": f1}
        return result_dict

