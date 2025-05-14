import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Dict, List

from tairvision_object_detection.utils import reduce_mean
from tairvision_object_detection.ops import sigmoid_focal_loss

from tairvision_object_detection.models.detection.fcos import (FCOSNet, FCOSClassificationHead, FCOSRegressionCenternessHead, Scale,
                                              permute_head_outputs, compute_ious, compute_centerness_targets, ml_nms,
                                              get_sample_region)

__all__ = [
    "FCOSBevNet", "FCOSBevHead",
]

INF = 100000000


class FCOSBevHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_classes, regression_out_channels, use_scale=False, num_feat_levels=5,
                 use_deformable=False, regression_functions=None, **kwargs):
        super().__init__()

        self.classification_head = FCOSBevClassificationHeadPerClass(in_channels, num_classes,
                                                                     use_deformable=use_deformable, **kwargs)
        self.regression_head = FCOSBevRegressionHeadPerClass(in_channels,
                                                             regression_out_channels,
                                                             use_scale=use_scale,
                                                             num_feat_levels=num_feat_levels,
                                                             use_deformable=use_deformable,
                                                             regression_functions=regression_functions
                                                             )

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        all_features = []

        for features in x:
            all_features.append(features)

        cls_logits = self.classification_head(all_features)
        bbox_regression, centerness, other_regressions = self.regression_head(all_features)
        out_dict = {
            'cls_logits': cls_logits,
            'centerness': centerness,
            'bbox_regression': bbox_regression,
            'other_regressions': other_regressions
        }
        return out_dict

    def compute_loss(self, targets: Dict[str, Tensor], head_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        cls_loss = self.classification_head.compute_loss(targets, head_outputs)
        bbox_loss, ctr_loss, other_losses = self.regression_head.compute_loss(targets, head_outputs)
        loss_dict = {
            'cls_loss': cls_loss,
            'ctr_loss': ctr_loss,
            'bbox_loss': bbox_loss,
            'other_losses': other_losses
        }

        return loss_dict


class FCOSBevRegressionHead(FCOSRegressionCenternessHead):
    """
    A regression head for use in FCOSBevNet.

    Args:
        in_channels (int): number of channels of the input feature
    """

    def __init__(self, in_channels, regression_out_channels, use_scale=False, num_feat_levels=5,
                 loss_type='giou', use_deformable=False, regression_functions=None):
        super().__init__(in_channels, use_scale, num_feat_levels, loss_type, use_deformable)

        # regression head
        self.regression_heads = nn.ModuleList()
        self.regression_scales = nn.ModuleList()
        if regression_functions is not None:
            self.regression_functions = regression_functions
        else:
            self.regression_functions = ["relu" for _ in regression_out_channels]

        for out_channel in regression_out_channels:
            last_conv = nn.Conv2d(in_channels, np.abs(out_channel), kernel_size=3, stride=1, padding=1)
            torch.nn.init.normal_(last_conv.weight, std=0.01)
            torch.nn.init.zeros_(last_conv.bias)
            self.regression_heads.append(last_conv)

            scale = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_feat_levels)]) if use_scale else None
            self.regression_scales.append(scale)

        self.regression_out_channels = regression_out_channels

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        all_bbox_regression = []
        all_centerness = []

        for i_feature, features in enumerate(x):
            shared_features = self.conv(features)

            bbox_regression = self.bbox_reg(shared_features)
            if self.scales is not None:
                bbox_regression = self.scales[i_feature](bbox_regression)
            bbox_regression = F.relu(bbox_regression)
            bbox_regression = permute_head_outputs(bbox_regression)
            all_bbox_regression.append(bbox_regression)

            centerness = self.centerness(shared_features)
            centerness = permute_head_outputs(centerness)
            all_centerness.append(centerness)

        output = [torch.cat(all_bbox_regression, dim=1), torch.cat(all_centerness, dim=1)]

        other_regressions = []
        for i_head, regression_head in enumerate(self.regression_heads):
            all_regression = []
            for i_feature, features in enumerate(x):
                shared_features = self.conv(features)

                regression = regression_head(shared_features)
                scale = self.regression_scales[i_head]
                if scale is not None:
                    regression = scale[i_feature](regression)
                if self.regression_functions[i_head] == "relu":
                    regression = F.relu(regression)
                regression = permute_head_outputs(regression)
                all_regression.append(regression)

            other_regressions.append(torch.cat(all_regression, dim=1))

        output.append(other_regressions)

        return output

    def compute_loss(self, targets, head_outputs):
        pos_inds = head_outputs['pos_inds']
        num_pos_avg = head_outputs['num_pos_avg']

        bbox_pred = head_outputs['bbox_regression'].reshape(-1, 4)[pos_inds]
        ctr_pred = head_outputs['centerness'].reshape(-1, 1)[pos_inds]
        bbox_target = targets['bbox_regression'].reshape(-1, 4)[pos_inds]

        ious, gious = compute_ious(bbox_pred, bbox_target)

        ctr_target = compute_centerness_targets(bbox_target)
        loss_denorm = max(reduce_mean(ctr_target.sum()).item(), 1e-6)
        targets['centerness'] = ctr_target

        bbox_loss = self.loss_func(ious, gious, ctr_target) / loss_denorm
        ctr_loss = F.binary_cross_entropy_with_logits(ctr_pred, ctr_target.unsqueeze(1),
                                                      reduction="sum") / num_pos_avg

        output = [bbox_loss, ctr_loss]

        other_pred_reg = head_outputs['other_regressions']
        other_target_reg = targets['other_regressions']
        out_channel = self.regression_out_channels

        other_preds = []
        other_targets = []
        other_losses = []
        for i in range(len(other_pred_reg)):
            other_pred = other_pred_reg[i].reshape(-1, out_channel[i])[pos_inds]
            other_target = other_target_reg[i].reshape(-1, out_channel[i])[pos_inds]

            valid = other_target < INF
            other_pred = other_pred[valid]
            other_target = other_target[valid]
            other_preds.append(other_pred)
            other_targets.append(other_target)
            if self.regression_functions[i] == "sigmoid":
                other_loss = F.binary_cross_entropy_with_logits(other_pred, other_target, reduction='sum')
            else:
                other_loss = F.smooth_l1_loss(other_pred, other_target, reduction='sum')
            other_loss /= (loss_denorm * out_channel[i])
            other_losses.append(other_loss)

        output.append(other_losses)

        return output


class FCOSBevRegressionHeadPerClass(FCOSBevRegressionHead):

    def __init__(self, in_channels, regression_out_channels,
                 use_scale=False,
                 num_feat_levels=5,
                 loss_type='giou',
                 use_deformable=False,
                 regression_functions=None
                 ):

        super().__init__(in_channels, regression_out_channels,
                         use_scale=use_scale,
                         num_feat_levels=num_feat_levels,
                         loss_type=loss_type,
                         use_deformable=use_deformable,
                         regression_functions=regression_functions,
                         )

    def compute_loss(self, targets, head_outputs):

        num_classes = len(head_outputs['pos_inds_per_class'])

        bbox_loss = 0
        ctr_loss = 0
        for i_class in range(num_classes):

            pos_inds_per_class = head_outputs['pos_inds_per_class'][i_class]
            num_pos_per_class_avg = head_outputs['num_pos_per_class_avg'][i_class]

            bbox_pred = head_outputs['bbox_regression'].reshape(-1, 4)[pos_inds_per_class]
            ctr_pred = head_outputs['centerness'].reshape(-1, 1)[pos_inds_per_class]
            bbox_target = targets['bbox_regression'].reshape(-1, 4)[pos_inds_per_class]

            ious, gious = compute_ious(bbox_pred, bbox_target)

            ctr_target = compute_centerness_targets(bbox_target)
            loss_denorm = max(reduce_mean(ctr_target.sum()).item(), 1e-6)

            bbox_loss += self.loss_func(ious, gious, ctr_target) / loss_denorm
            ctr_loss += F.binary_cross_entropy_with_logits(ctr_pred, ctr_target.unsqueeze(1),
                                                           reduction="sum") / num_pos_per_class_avg

        output = [bbox_loss, ctr_loss]

        other_pred_reg = head_outputs['other_regressions']
        other_target_reg = targets['other_regressions']
        out_channel = self.regression_out_channels

        other_preds = []
        other_targets = []
        other_losses = []
        for i in range(len(other_pred_reg)):
            other_loss_cumm = 0
            for i_class in range(num_classes):

                pos_inds_per_class = head_outputs['pos_inds_per_class'][i_class]
                bbox_target = targets['bbox_regression'].reshape(-1, 4)[pos_inds_per_class]

                ctr_target = compute_centerness_targets(bbox_target)
                loss_denorm = max(reduce_mean(ctr_target.sum()).item(), 1e-6)

                other_pred = other_pred_reg[i].reshape(-1, out_channel[i])[pos_inds_per_class]
                other_target = other_target_reg[i].reshape(-1, out_channel[i])[pos_inds_per_class]

                valid = other_target < INF
                other_pred = other_pred[valid]
                other_target = other_target[valid]
                other_preds.append(other_pred)
                other_targets.append(other_target)
                if self.regression_functions[i] == "sigmoid":
                    other_loss = F.binary_cross_entropy_with_logits(other_pred, other_target, reduction='sum')
                else:
                    other_loss = F.smooth_l1_loss(other_pred, other_target, reduction='sum')
                other_loss /= (loss_denorm * out_channel[i])

                other_loss_cumm += other_loss

            other_losses.append(other_loss_cumm)

        output.append(other_losses)

        return output


class FCOSBevClassificationHead(FCOSClassificationHead):
    """
    A classification head for use in FCOSBevNet.

    Args:
        in_channels (int): number of channels of the input feature
    """

    def __init__(self, in_channels, num_classes,
                 prior_probability=0.01,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 use_deformable=False,
                 **kwargs
                 ):
        super().__init__(in_channels, num_classes,
                         prior_probability=prior_probability,
                         focal_alpha=focal_alpha,
                         focal_gamma=focal_gamma,
                         use_deformable=use_deformable,
                         **kwargs)


class FCOSBevClassificationHeadPerClass(FCOSBevClassificationHead):
    """
    A classification head for use in FCOSBevNet.

    Args:
        in_channels (int): number of channels of the input feature
    """

    def __init__(self, in_channels, num_classes,
                 prior_probability=0.01,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 use_deformable=False,
                 **kwargs
                 ):
        super().__init__(in_channels, num_classes,
                         prior_probability=prior_probability,
                         focal_alpha=focal_alpha,
                         focal_gamma=focal_gamma,
                         use_deformable=use_deformable,
                         **kwargs)

    def compute_loss(self, targets, head_outputs):

        cls_pred = head_outputs['cls_logits'].reshape(-1, self.num_classes)

        labels = targets['labels'].flatten()

        pos_inds = torch.nonzero(labels != self.num_classes).squeeze(1)
        head_outputs['pos_inds'] = pos_inds

        num_pos_local = torch.ones_like(pos_inds).sum()
        num_pos_avg = max(reduce_mean(num_pos_local).item(), 1.0)
        head_outputs['num_pos_avg'] = num_pos_avg

        head_outputs['pos_inds_per_class'] = []
        head_outputs['num_pos_per_class_avg'] = []

        cls_loss = 0
        for i_class in range(self.num_classes):
            # prepare one_hot
            cls_pred_per_class = cls_pred[:, i_class:i_class + 1]
            pos_inds_per_class = torch.nonzero(labels == i_class).squeeze(1)

            num_pos_per_class_local = torch.ones_like(pos_inds_per_class).sum()
            num_pos_per_class_avg = max(reduce_mean(num_pos_per_class_local).item(), 1.0)

            cls_target_per_class = torch.zeros_like(cls_pred_per_class)
            cls_target_per_class[pos_inds_per_class] = 1

            cls_loss_per_class = sigmoid_focal_loss(cls_pred_per_class, cls_target_per_class,
                                                    alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="sum"
                                                    )

            cls_loss_per_class = cls_loss_per_class / num_pos_per_class_avg

            cls_loss += cls_loss_per_class

            head_outputs['pos_inds_per_class'].append(pos_inds_per_class)
            head_outputs['num_pos_per_class_avg'].append(num_pos_per_class_avg)

        return cls_loss


class FCOSBevNet(FCOSNet):
    def __init__(self, backbone, num_classes, regression_out_channels,
                 fpn_strides=(1,),
                 sois=(-1, INF),
                 use_scale=True,
                 use_deformable=False,
                 regression_functions=None,
                 nms_thresh=0.3,
                 **kwargs
                 ):
        in_channels = kwargs.pop("in_channels", backbone.out_channels)
        super().__init__(backbone, num_classes, fpn_strides=fpn_strides, sois=sois, nms_thresh=nms_thresh, **kwargs)

        self.head = FCOSBevHead(in_channels, num_classes, regression_out_channels,
                                use_scale=use_scale, num_feat_levels=len(fpn_strides),
                                use_deformable=use_deformable, regression_functions=regression_functions,
                                **kwargs)

        self.regression_out_channels = regression_out_channels

    def compute_relative_targets(self, targets, locations, strides, sois):
        labels = []
        reg_targets = []
        other_targets = []

        for i in range(len(targets)):
            targets_per_im = targets[i]
            bboxes = targets_per_im['boxes']
            labels_per_im = targets_per_im['labels']
            others_per_im = targets_per_im['others']

            # no gt
            if bboxes.numel() == 0:
                labels.append(torch.zeros((locations.size(0)), dtype=torch.int64, device=locations.device) +
                              self.num_classes)
                reg_targets.append(torch.zeros((locations.size(0), 4), dtype=locations.dtype, device=locations.device))
                other_target_list = []
                for j in range(len(self.regression_out_channels)):
                    other_target = torch.zeros((locations.size(0), self.regression_out_channels[j]),
                                               dtype=locations.dtype, device=locations.device)
                    other_target_list.append(other_target)
                other_targets.append(other_target_list)

                continue

            area = targets_per_im['area']

            reg_targets_per_im = torch.cat([locations[:, None, :] - bboxes[None, :, 0:2],
                                            bboxes[None, :, 2::] - locations[:, None, :]], dim=-1)

            if self.center_sample:
                is_in_boxes = get_sample_region(bboxes, strides, locations,
                                                bitmasks=None, radius=self.radius
                                                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= sois[:, [0]]) & \
                (max_reg_targets_per_im <= sois[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes
            other_targets_per_im = []
            for other in others_per_im:
                others_per_im_temp = other[locations_to_gt_inds]
                others_per_im_temp[locations_to_min_area == INF] = INF
                other_targets_per_im.append(others_per_im_temp)

            other_targets.append(other_targets_per_im)
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        stacked_others = []
        for i in range(len(other_targets[0])):
            stacked_others.append(torch.stack(list(zip(*other_targets))[i], dim=0))

        relative_targets = {"labels": torch.stack(labels, dim=0),
                            "bbox_regression": torch.stack(reg_targets, dim=0) / strides[None, :, :],
                            "other_regressions": stacked_others
                            }

        return relative_targets

    def predict_proposals(self,
                          predictions: Dict[str, Tensor],
                          locations: Tensor,
                          strides: Tensor,
                          top_feats: Tensor = None):

        cls_logits = predictions['cls_logits'].sigmoid()
        bbox_regression = predictions['bbox_regression'] * strides[None, :, :]
        centerness = predictions['centerness'].sigmoid()
        other_regressions = predictions['other_regressions']

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            cls_logits = cls_logits * centerness
        candidate_inds = cls_logits > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.sum([1, 2])
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)

        if not self.thresh_with_ctr:
            cls_logits = cls_logits * centerness

        results = []
        for i in range(len(candidate_inds)):
            per_cls_logits = cls_logits[i]
            per_candidate_inds = candidate_inds[i]
            per_cls_logits = per_cls_logits[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = bbox_regression[i]
            per_box_regression = per_box_regression[per_box_loc]

            per_other_regressions_list = []
            for j in range(len(other_regressions)):
                per_other_regressions = other_regressions[j][i]
                per_other_regressions = per_other_regressions[per_box_loc]
                per_other_regressions_list.append(per_other_regressions)

            per_strides = strides[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_cls_logits, top_k_indices = per_cls_logits.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                for k in range(len(per_other_regressions_list)):
                    per_other_regressions_list[k] = per_other_regressions_list[k][top_k_indices]
                per_strides = per_strides[top_k_indices]
                per_locations = per_locations[top_k_indices]

            per_box_regression = torch.stack([per_locations[:, 0] - per_box_regression[:, 0],
                                              per_locations[:, 1] - per_box_regression[:, 1],
                                              per_locations[:, 0] + per_box_regression[:, 2],
                                              per_locations[:, 1] + per_box_regression[:, 3],
                                              ], dim=1)

            result = {}
            result['boxes'] = per_box_regression
            result['scores'] = torch.sqrt(per_cls_logits)
            result['labels'] = per_class
            result['others'] = []
            for i_other, per_other_regression in enumerate(per_other_regressions_list):
                if self.head.regression_head.regression_functions[i_other] == "sigmoid":
                    per_other_regression = per_other_regression.sigmoid()
                result['others'].append(per_other_regression)
            result['locations'] = per_locations
            result['strides'] = per_strides

            result = ml_nms(result, self.nms_thresh)
            number_of_detections = len(result['boxes'])

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.post_nms_topk > 0:
                cls_scores = result['scores']
                image_thresh, _ = torch.kthvalue(cls_scores, number_of_detections - self.post_nms_topk + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                for key in result.keys():
                    if isinstance(result[key], list):
                        result[key] = [r[keep] for r in result[key]]
                    else:
                        result[key] = result[key][keep]
            results.append(result)

        return results
