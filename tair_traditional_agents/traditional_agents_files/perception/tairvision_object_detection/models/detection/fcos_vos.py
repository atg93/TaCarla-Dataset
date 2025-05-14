import math
from collections import OrderedDict

import torch
from torch import nn, Tensor
from typing import Dict, List, Tuple, Optional, Union

from .backbone_utils import resnet_fpn_backbone, _validate_trainable_layers, regnet_fpn_backbone, convnext_fpn_backbone
from tairvision_object_detection.ops.feature_pyramid_network import LastLevelP6, LastLevelP6P7, LastLevelMaxPool

from tairvision_object_detection.models.detection.fcos import FCOSNet, FCOSHead, FCOSClassificationHead, sum_features
from tairvision_object_detection.models.detection.fcos import eager_outputs, permute_head_outputs, ml_nms
from tairvision_object_detection.models.detection.vos import VOSHeadAF
from tairvision_object_detection.models.detection.process import batch_images, postprocess, get_original_image_sizes

__all__ = [
    "FCOSVOSNet", "FCOSVOSHead", "fcosvos_regnet_fpn", "fcosvos_resnet_fpn", "fcosvos_convnext_fpn",
]

INF = 100000000


class FCOSVOSHead(FCOSHead):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_classes, num_keypoints=0, use_scale=False, num_feat_levels=5,
                 loss_weights=(1.0, 1.0, 1.0, 0.25), use_deformable=False, **kwargs):
        super().__init__(in_channels, num_classes, num_keypoints, use_scale, num_feat_levels,
                         loss_weights, use_deformable, **kwargs)

        self.classification_head = FCOSVOSClassificationHead(in_channels, num_classes,
                                                             use_deformable=use_deformable, **kwargs)

    def forward(self, all_features: List[Tensor]) -> Dict[str, Tensor]:
        cls_logits, cls_feats = self.classification_head(all_features)
        centerness = self.centerness_head(all_features)
        bbox_regression = self.regression_head(all_features)
        out_list = {
            'cls_logits': cls_logits,
            'centerness': centerness,
            'bbox_regression': bbox_regression,
            'cls_feats': cls_feats
        }

        if self.num_keypoints > 0:
            out_list['kpoint_regression'] = self.regression_keypoint_head(all_features)

        return out_list


class FCOSVOSClassificationHead(FCOSClassificationHead):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_classes, prior_probability=0.01,
                 loss_normalizer="fg",
                 moving_fg=100,
                 moving_fg_momentum=0.9,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 use_deformable=False,
                 ap_delta=0.8,
                 cls_loss=None,
                 **kwargs
                 ):
        super().__init__(in_channels, num_classes, prior_probability, loss_normalizer, moving_fg,
                         moving_fg_momentum, focal_alpha, focal_gamma, use_deformable, ap_delta, cls_loss, **kwargs)

        # TODO: had to change this conv from 3x3 to 1x1, check later if this can be done with 3x3,
        #  i.e how to find 9 responsible input location for each output location
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

    def forward(self, x: List[Tensor]) -> Tuple[Tensor, Tensor]:
        all_cls_logits = []
        all_cls_feats = []

        for features in x:
            cls_feats = self.conv(features)
            cls_logits = self.cls_logits(cls_feats)
            cls_logits = permute_head_outputs(cls_logits)

            all_cls_logits.append(cls_logits)

            cls_feats = permute_head_outputs(cls_feats)
            all_cls_feats.append(cls_feats)

        return torch.cat(all_cls_logits, dim=1), torch.cat(all_cls_feats, dim=1)

    def get_score(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.cls_logits(x)
        return x.squeeze(3).squeeze(2)


class FCOSVOSNet(FCOSNet):

    def __init__(self, backbone, num_classes, num_keypoints=0,
                 # transform parameters
                 min_size=800, max_size=1333,
                 context_module=None,
                 fpn_strides=(8, 16, 32, 64, 128),
                 sois=(-1, 64, 128, 256, 512, INF),
                 center_sample=True,
                 radius=1.5,
                 thresh_with_ctr=False,
                 use_scale=True,
                 use_deformable=False,
                 loss_weights=(1.0, 1.0, 1.0, 0.25),
                 pre_nms_thresh=0.05,
                 pre_nms_topk=1000,
                 nms_thresh=0.6,
                 post_nms_topk=100,
                 trainer_side=False,
                 **kwargs
                 ):
        super().__init__(backbone, num_classes, num_keypoints, min_size, max_size,
                         context_module, fpn_strides, sois, center_sample, radius,
                         thresh_with_ctr, use_scale, use_deformable,
                         loss_weights, pre_nms_thresh, pre_nms_topk, nms_thresh, post_nms_topk, trainer_side, **kwargs)

        self.head = FCOSVOSHead(backbone.out_channels, num_classes, num_keypoints,
                                use_scale=use_scale, num_feat_levels=len(fpn_strides), loss_weights=loss_weights,
                                use_deformable=use_deformable, **kwargs)

        score_func = self.head.classification_head.get_score

        self.vos_head = VOSHeadAF(num_classes, score_func, representation_size=256)

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None,
                calc_val_loss: bool = False) \
            -> Union[Dict[str, Tensor], List[Dict[str, Tensor]], Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]]:
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
            calc_val_loss (bool): enable val_loss calculation

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                          boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # transform the input
        images = batch_images(images)

        if targets is not None:
            original_image_sizes = get_original_image_sizes(targets)
        else:
            original_image_sizes = images.image_sizes

        if not self.training and not self.trainer_side:
            targets = None

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # get the features from the backbone

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        if self.context is not None:
            features = self.context(features)

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        all_features = sum_features(features, is_sum=False)

        # compute the fcos heads outputs using the features
        head_outputs = self.head(all_features)
        locations, strides, sois = self.compute_locations(all_features)
        loss, detections = None, None

        if targets is not None:
            targets = self.compute_relative_targets(targets, locations, strides, sois)
            loss = self.head.compute_loss(targets, head_outputs)

            if self.vos_head is not None:
                with torch.no_grad():
                    proposals = self.get_valid_pixels(targets['labels'], head_outputs)

                vos_dist_loss = self.vos_head(proposals, targets)
                loss['vos_dist_loss'] = vos_dist_loss

        if not self.training:
            detections = self.predict_proposals(head_outputs, locations, strides, top_feats=None)
            if self.vos_head is not None:
                detections = self.vos_head(detections)
            detections = self.postprocess(detections, images.image_sizes, original_image_sizes)
        else:
            if self.trainer_side:
                detections = head_outputs

        return eager_outputs(loss, detections)

    def get_valid_pixels(self, labels, head_outputs):

        cls_feats = head_outputs['cls_feats']
        cls_logits = head_outputs['cls_logits']
        batch_size = len(cls_feats)
        pos_inds_1d = torch.nonzero(labels.flatten() != self.num_classes).squeeze(1)
        pos_inds_2d = torch.nonzero(labels != self.num_classes)

        cls_feats = cls_feats.view(-1, 256)[pos_inds_1d, :]
        cls_logits = cls_logits.view(-1, self.num_classes)[pos_inds_1d, :]
        results = []
        for i in range(batch_size):
            idx = pos_inds_2d[:, 0] == i
            result = {'cls_feats': cls_feats[idx],
                      'cls_logits': cls_logits[idx]
                      }
            results.append(result)

        return results

    def predict_proposals(self,
                          predictions: Dict[str, Tensor],
                          locations: Tensor,
                          strides: Tensor,
                          top_feats: Tensor = None):

        cls_logits = predictions['cls_logits'].sigmoid()
        cls_logits_bypass = predictions['cls_logits']
        bbox_regression = predictions['bbox_regression'] * strides[None, :, :]
        centerness = predictions['centerness'].sigmoid()
        if self.num_keypoints > 0:
            kpoint_regression = predictions['kpoint_regression'] * strides[None, :, :]

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

            per_cls_logits_bypass = cls_logits_bypass[i]
            per_cls_logits_bypass = per_cls_logits_bypass[per_box_loc]

            per_strides = strides[per_box_loc]

            per_locations = locations[per_box_loc]

            if self.num_keypoints > 0:
                per_kpoint_regression = kpoint_regression[i]
                per_kpoint_regression = per_kpoint_regression[per_box_loc]

            if top_feats is not None:
                per_top_feats = top_feats[i]
                per_top_feats = per_top_feats[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_cls_logits, top_k_indices = per_cls_logits.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_strides = per_strides[top_k_indices]
                per_locations = per_locations[top_k_indices]

                per_cls_logits_bypass = per_cls_logits_bypass[top_k_indices]

                if self.num_keypoints > 0:
                    per_kpoint_regression = per_kpoint_regression[top_k_indices]

                if top_feats is not None:
                    per_top_feats = per_top_feats[top_k_indices]

            per_box_regression = torch.stack([per_locations[:, 0] - per_box_regression[:, 0],
                                              per_locations[:, 1] - per_box_regression[:, 1],
                                              per_locations[:, 0] + per_box_regression[:, 2],
                                              per_locations[:, 1] + per_box_regression[:, 3],
                                              ], dim=1)

            if self.num_keypoints > 0:
                per_kpoint_regression = per_kpoint_regression.reshape(-1, self.num_keypoints, 2)
                per_kpoint_regression = torch.stack([per_locations[:, None, 0] - per_kpoint_regression[:, :, 0],
                                                     per_locations[:, None, 1] - per_kpoint_regression[:, :, 1],
                                                     ], dim=2)

            result = {}
            result['boxes'] = per_box_regression
            result['scores'] = torch.sqrt(per_cls_logits)
            result['labels'] = per_class
            result['locations'] = per_locations
            result['strides'] = per_strides
            result['cls_logits'] = per_cls_logits_bypass
            if self.num_keypoints:
                result['keypoints'] = per_kpoint_regression

            if top_feats is not None:
                result['top_feats'] = per_top_feats

            result = ml_nms(result, self.nms_thresh)
            number_of_detections = len(result['boxes'])

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.post_nms_topk > 0:
                cls_scores = result['scores']
                image_thresh, _ = torch.kthvalue(cls_scores, number_of_detections - self.post_nms_topk + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                for key in result.keys():
                    result[key] = result[key][keep]
            results.append(result)

        return results


def fcosvos_resnet_fpn(type="resnet50", pretrained=False, progress=True,
                       num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None,
                       use_P2=False, no_extra_blocks=False, extra_before=False,
                       pyramid_type="bifpn", depthwise=True, repeats=3, fusion_type='fastnormed', bifpn_norm_layer=None,
                       **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

    if use_P2:
        returned_layers = [1, 2, 3, 4]
        extra_blocks = None if no_extra_blocks else LastLevelP6 if pyramid_type == "bifpn" else LastLevelMaxPool
    else:
        # skip P2 because it generates too many anchors (according to their paper)
        returned_layers = [2, 3, 4]
        extra_blocks = None if no_extra_blocks else LastLevelP6P7

    backbone = resnet_fpn_backbone(type, pretrained_backbone, returned_layers=returned_layers,
                                   extra_blocks=extra_blocks, trainable_layers=trainable_backbone_layers,
                                   extra_before=extra_before,
                                   pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats,
                                   fusion_type=fusion_type, bifpn_norm_layer=bifpn_norm_layer, **kwargs)
    model = FCOSVOSNet(backbone, num_classes, **kwargs)
    return model


def fcosvos_regnet_fpn(type="regnet_y_1_6gf", pretrained=False, progress=True,
                       num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None,
                       use_P2=False, no_extra_blocks=False, extra_before=False,
                       pyramid_type="bifpn", depthwise=True, repeats=3, fusion_type='fastnormed', bifpn_norm_layer=None,
                       **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

    if use_P2:
        returned_layers = [1, 2, 3, 4]
        extra_blocks = None if no_extra_blocks else LastLevelP6 if pyramid_type == "bifpn" else LastLevelMaxPool
    else:
        # skip P2 because it generates too many anchors (according to their paper)
        returned_layers = [2, 3, 4]
        extra_blocks = None if no_extra_blocks else LastLevelP6P7

    backbone = regnet_fpn_backbone(type, pretrained_backbone, returned_layers=returned_layers,
                                   extra_blocks=extra_blocks, trainable_layers=trainable_backbone_layers,
                                   extra_before=extra_before,
                                   pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats,
                                   fusion_type=fusion_type, bifpn_norm_layer=bifpn_norm_layer)
    model = FCOSVOSNet(backbone, num_classes, **kwargs)
    return model


def fcosvos_convnext_fpn(type="convnext_tiny", pretrained=False, progress=True,
                         num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None,
                         use_P2=False, no_extra_blocks=False, extra_before=False,
                         pyramid_type="bifpn", depthwise=True, repeats=3, fusion_type='fastnormed',
                         bifpn_norm_layer=None,
                         **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

    if use_P2:
        returned_layers = [1, 2, 3, 4]
        extra_blocks = None if no_extra_blocks else LastLevelP6 if pyramid_type == "bifpn" else LastLevelMaxPool
    else:
        # skip P2 because it generates too many anchors (according to their paper)
        returned_layers = [2, 3, 4]
        extra_blocks = None if no_extra_blocks else LastLevelP6P7

    backbone = convnext_fpn_backbone(type, pretrained_backbone, returned_layers=returned_layers,
                                     extra_blocks=extra_blocks, trainable_layers=trainable_backbone_layers,
                                     extra_before=extra_before,
                                     pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats,
                                     fusion_type=fusion_type, bifpn_norm_layer=bifpn_norm_layer)
    model = FCOSVOSNet(backbone, num_classes, **kwargs)
    return model
