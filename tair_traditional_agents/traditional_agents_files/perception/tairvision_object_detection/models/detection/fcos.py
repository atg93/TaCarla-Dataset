import math
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Dict, List, Tuple, Optional, Union

from .backbone_utils import (resnet_fpn_backbone, _validate_trainable_layers, regnet_fpn_backbone,
                             convnext_fpn_backbone, efficientnet_fpn_backbone)
from .process import batch_images, postprocess, get_original_image_sizes

from tairvision_object_detection.ops.feature_pyramid_network import LastLevelP6, LastLevelP6P7, LastLevelMaxPool
from tairvision_object_detection.ops import sigmoid_focal_loss, APLoss
from tairvision_object_detection.ops import boxes as box_ops
from tairvision_object_detection.ops.deform_conv import DFConv2d
from tairvision_object_detection.utils import reduce_mean

from tairvision_object_detection.models.detection.context import IndependentContextNetwork, SharedContextNetwork

__all__ = [
    "FCOSNet", "FCOSHead", "fcos_regnet_fpn", "fcos_resnet_fpn", "fcos_convnext_fpn", "fcos_efficientnet_fpn"
]

INF = 100000000


class FCOSHead(nn.Module):
    """
    A regression and classification head for use in FCOSNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_classes, num_keypoints=0, use_scale=False, num_feat_levels=5,
                 loss_weights=(1.0, 1.0, 1.0, 0.25), use_deformable=False, **kwargs):
        super().__init__()

        self.classification_head = FCOSClassificationHead(in_channels, num_classes,
                                                          use_deformable=use_deformable, **kwargs)
        self.centerness_head = FCOSCenternessHead(in_channels,
                                                  use_deformable=use_deformable)
        self.regression_head = FCOSRegressionHead(in_channels,
                                                  use_scale=use_scale,
                                                  num_feat_levels=num_feat_levels,
                                                  use_deformable=use_deformable)

        self.loss_weights = loss_weights
        self.num_keypoints = num_keypoints
        if self.num_keypoints > 0:
            self.regression_keypoint_head = FCOSKpointRegressionHead(in_channels, num_keypoints,
                                                                     use_scale=use_scale,
                                                                     num_feat_levels=num_feat_levels,
                                                                     use_deformable=use_deformable)

    def forward(self, all_features: List[Tensor]) -> Dict[str, Tensor]:
        cls_logits = self.classification_head(all_features)
        centerness = self.centerness_head(all_features)
        bbox_regression = self.regression_head(all_features)
        out_list = {
            'cls_logits': cls_logits,
            'centerness': centerness,
            'bbox_regression': bbox_regression
        }

        if self.num_keypoints > 0:
            out_list['kpoint_regression'] = self.regression_keypoint_head(all_features)

        return out_list

    def compute_loss(self, targets: Dict[str, Tensor], head_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        cls_weight, ctr_weight, bbox_weight, kpoint_weight = self.loss_weights

        cls_loss = self.classification_head.compute_loss(targets, head_outputs)
        ctr_loss = self.centerness_head.compute_loss(targets, head_outputs)
        bbox_loss = self.regression_head.compute_loss(targets, head_outputs)
        loss_dict = {
            'cls_loss': cls_loss * cls_weight,
            'ctr_loss': ctr_loss * ctr_weight,
            'bbox_loss': bbox_loss * bbox_weight
        }

        if self.num_keypoints > 0:
            loss_dict['kpoint_loss'] = self.regression_keypoint_head.compute_loss(targets, head_outputs) * kpoint_weight

        return loss_dict


class FCOSClassificationHead(nn.Module):
    """
    A classification head for use in FCOSNet.

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
        super().__init__()

        self.loss_normalizer = loss_normalizer
        self.moving_fg = moving_fg
        self.moving_fg_momentum = moving_fg_momentum
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ap_delta = ap_delta
        self.loss = cls_loss

        assert loss_normalizer in ("moving_fg", "fg", "all"), 'loss_normalizer can only be "moving_fg", "fg" or "all"'

        conv = []
        for i in range(4):
            conv_func = DFConv2d if (i == 3 and use_deformable) else nn.Conv2d
            conv.append(conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.GroupNorm(32, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes

    def forward(self, x: List[Tensor]) -> Tensor:
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)
            cls_logits = permute_head_outputs(cls_logits)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)

    def compute_loss(self, targets, head_outputs):

        cls_pred = head_outputs['cls_logits'].reshape(-1, self.num_classes)

        labels = targets['labels'].flatten()

        pos_inds = torch.nonzero(labels != self.num_classes).squeeze(1)
        head_outputs['pos_inds'] = pos_inds

        num_pos_local = torch.ones_like(pos_inds).sum()
        num_pos_avg = max(reduce_mean(num_pos_local).item(), 1.0)
        head_outputs['num_pos_avg'] = num_pos_avg

        # prepare one_hot
        cls_target = torch.zeros_like(cls_pred)
        cls_target[pos_inds, labels[pos_inds]] = 1

        if self.loss == "ap_loss":
            cls_loss = APLoss.apply(cls_pred, cls_target, delta=self.ap_delta, interpolate=0)
            return cls_loss

        cls_loss = sigmoid_focal_loss(cls_pred, cls_target,
                                      alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="sum"
                                      )

        if self.loss_normalizer == "moving_fg":
            self.moving_fg = self.moving_fg_momentum * self.moving_fg + (1 - self.moving_fg_momentum) * num_pos_avg
            cls_loss = cls_loss / self.moving_fg
        elif self.loss_normalizer == "fg":
            cls_loss = cls_loss / num_pos_avg
        else:
            num_samples_local = torch.ones_like(labels).sum()
            num_samples_avg = max(reduce_mean(num_samples_local).item(), 1.0)
            cls_loss = cls_loss / num_samples_avg

        return cls_loss


class FCOSCenternessHead(nn.Module):
    """
    A regression head for use in FCOSNet.

    Args:
        in_channels (int): number of channels of the input feature
    """

    def __init__(self, in_channels, box_quality="centerness", use_deformable=False):
        super().__init__()

        self.box_quality = box_quality

        conv = []
        for i in range(4):
            conv_func = DFConv2d if (i == 3 and use_deformable) else nn.Conv2d
            conv.append(conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.GroupNorm(32, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.centerness.weight, std=0.01)
        torch.nn.init.zeros_(self.centerness.bias)

    def forward(self, x: List[Tensor]) -> Tuple[Tensor, Tensor]:
        all_centerness = []
        for i_feature, features in enumerate(x):
            centerness = self.conv(features)
            centerness = self.centerness(centerness)
            centerness = permute_head_outputs(centerness)

            all_centerness.append(centerness)

        return torch.cat(all_centerness, dim=1)

    def compute_loss(self, targets, head_outputs):

        pos_inds = head_outputs['pos_inds']
        num_pos_avg = head_outputs['num_pos_avg']

        ctr_pred = head_outputs['centerness'].reshape(-1, 1)[pos_inds]
        bbox_target = targets['bbox_regression'].reshape(-1, 4)[pos_inds]

        if self.box_quality == "centerness":
            ctr_target = compute_centerness_targets(bbox_target)
            targets['centerness'] = ctr_target
            ctr_loss = F.binary_cross_entropy_with_logits(ctr_pred, ctr_target.unsqueeze(1),
                                                          reduction="sum") / num_pos_avg

        elif self.box_quality == "iou":
            bbox_pred = head_outputs['bbox_regression'].reshape(-1, 4)[pos_inds]
            ious, gious = compute_ious(bbox_pred, bbox_target)
            ctr_loss = F.binary_cross_entropy_with_logits(ctr_pred, ious.detach(), reduction="sum") / num_pos_avg
        else:
            raise NotImplementedError

        return ctr_loss


class FCOSRegressionHead(nn.Module):
    """
    A regression head for use in FCOSNet.

    Args:
        in_channels (int): number of channels of the input feature
    """

    def __init__(self, in_channels, use_scale=False, num_feat_levels=5,
                 loss_type='giou',
                 box_quality="centerness",
                 use_deformable=False):
        super().__init__()

        self.loss_type = loss_type
        self.box_quality = box_quality

        # regression head
        conv = []
        for i in range(4):
            conv_func = DFConv2d if (i == 3 and use_deformable) else nn.Conv2d
            conv.append(conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.GroupNorm(32, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.bbox_reg = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        if use_scale:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_feat_levels)])
        else:
            self.scales = None

    def forward(self, x: List[Tensor]) -> Tuple[Tensor, Tensor]:
        all_bbox_regression = []

        for i_feature, features in enumerate(x):
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)
            if self.scales is not None:
                bbox_regression = self.scales[i_feature](bbox_regression)
            bbox_regression = F.relu(bbox_regression)
            bbox_regression = permute_head_outputs(bbox_regression)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)

    def compute_loss(self, targets, head_outputs):

        pos_inds = head_outputs['pos_inds']
        num_pos_avg = head_outputs['num_pos_avg']

        bbox_pred = head_outputs['bbox_regression'].reshape(-1, 4)[pos_inds]
        bbox_target = targets['bbox_regression'].reshape(-1, 4)[pos_inds]

        ious, gious = compute_ious(bbox_pred, bbox_target)

        if self.box_quality == "centerness":
            ctr_target = targets['centerness']
            loss_denorm = max(reduce_mean(ctr_target.sum()).item(), 1e-6)

            bbox_loss = self.loss_func(ious, gious, ctr_target) / loss_denorm

        elif self.box_quality == "iou":
            bbox_loss = self.loss_func(ious, gious) / num_pos_avg

        else:
            raise NotImplementedError

        return bbox_loss

    def loss_func(self, ious, gious=None, weight=None):

        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            assert gious is not None
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()


class FCOSRegressionCenternessHead(nn.Module):
    """
    A regression head for use in FCOSNet.

    Args:
        in_channels (int): number of channels of the input feature
    """

    def __init__(self, in_channels, use_scale=False, num_feat_levels=5,
                 loss_type='giou',
                 box_quality="centerness",
                 use_deformable=False):
        super().__init__()

        self.loss_type = loss_type
        self.box_quality = box_quality

        # regression head
        conv = []
        for i in range(4):
            conv_func = DFConv2d if (i == 3 and use_deformable) else nn.Conv2d
            conv.append(conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.GroupNorm(32, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.bbox_reg = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.centerness.weight, std=0.01)
        torch.nn.init.zeros_(self.centerness.bias)

        if use_scale:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_feat_levels)])
        else:
            self.scales = None

    def forward(self, x: List[Tensor]) -> Tuple[Tensor, Tensor]:
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

        return torch.cat(all_bbox_regression, dim=1), torch.cat(all_centerness, dim=1)

    def compute_loss(self, targets, head_outputs):

        pos_inds = head_outputs['pos_inds']
        num_pos_avg = head_outputs['num_pos_avg']

        bbox_pred = head_outputs['bbox_regression'].reshape(-1, 4)[pos_inds]
        ctr_pred = head_outputs['centerness'].reshape(-1, 1)[pos_inds]
        bbox_target = targets['bbox_regression'].reshape(-1, 4)[pos_inds]

        ious, gious = compute_ious(bbox_pred, bbox_target)

        if self.box_quality == "centerness":
            ctr_target = compute_centerness_targets(bbox_target)
            loss_denorm = max(reduce_mean(ctr_target.sum()).item(), 1e-6)
            targets['centerness'] = ctr_target

            bbox_loss = self.loss_func(ious, gious, ctr_target) / loss_denorm
            ctr_loss = F.binary_cross_entropy_with_logits(ctr_pred, ctr_target.unsqueeze(1),
                                                          reduction="sum") / num_pos_avg

        elif self.box_quality == "iou":
            bbox_loss = self.loss_func(ious, gious) / num_pos_avg
            ctr_loss = F.binary_cross_entropy_with_logits(ctr_pred, ious.detach().unsqueeze(1),
                                                          reduction="sum") / num_pos_avg

        else:
            raise NotImplementedError

        return bbox_loss, ctr_loss

    def loss_func(self, ious, gious=None, weight=None):

        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            assert gious is not None
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()


class FCOSKpointRegressionHead(nn.Module):
    """
    A keypoint regression head for use in FCOSNet.

    Args:
        in_channels (int): number of channels of the input feature
    """

    def __init__(self, in_channels, num_keypoints, use_scale=False, num_feat_levels=5, use_deformable=False):
        super().__init__()

        conv = []
        for i in range(4):
            conv_func = DFConv2d if (i == 3 and use_deformable) else nn.Conv2d
            conv.append(conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.GroupNorm(32, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.kpoint_reg = nn.Conv2d(in_channels, num_keypoints * 2, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.kpoint_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.kpoint_reg.bias)

        self.num_keypoints = num_keypoints

        if use_scale:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_feat_levels)])
        else:
            self.scales = None

    def forward(self, x: List[Tensor]) -> Tensor:
        all_kpoint_regression = []

        for i_feature, features in enumerate(x):
            kpoint_regression = self.conv(features)
            kpoint_regression = self.kpoint_reg(kpoint_regression)
            if self.scales is not None:
                kpoint_regression = self.scales[i_feature](kpoint_regression)
            kpoint_regression = permute_head_outputs(kpoint_regression)

            all_kpoint_regression.append(kpoint_regression)

        return torch.cat(all_kpoint_regression, dim=1)

    def compute_loss(self, targets, head_outputs):

        pos_inds = head_outputs['pos_inds']
        kpoint_pred = head_outputs['kpoint_regression'].reshape(-1, self.num_keypoints * 2)[pos_inds]
        kpoint_target = targets['keypoints'].reshape(-1, self.num_keypoints * 2)[pos_inds]

        ctr_target = targets['centerness']
        loss_denorm = max(reduce_mean(ctr_target.sum()).item(), 1e-6)
        loss_denorm *= (self.num_keypoints * 2)

        valid = kpoint_target < INF
        kpoint_pred = kpoint_pred[valid]
        kpoint_target = kpoint_target[valid]

        kpoint_loss = F.smooth_l1_loss(kpoint_pred, kpoint_target, reduction='sum') / loss_denorm

        return kpoint_loss


class FCOSNet(nn.Module):
    def __init__(self, backbone, num_classes, num_keypoints=0,
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
        super().__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.center_sample = center_sample
        self.radius = radius
        self.thresh_with_ctr = thresh_with_ctr

        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.nms_thresh = nms_thresh
        self.post_nms_topk = post_nms_topk

        self.fpn_strides = fpn_strides
        self.sois = [sois[i:i + 2] for i in range(len(sois) - 1)]

        self.backbone = backbone
        self.trainer_side = trainer_side

        if context_module is not None:
            context_module, module_sharing = context_module.split('-')
            if module_sharing == 'shared':
                self.context = SharedContextNetwork(backbone.out_channels, context_module)
            elif module_sharing == 'independent':
                self.context = IndependentContextNetwork(backbone.out_channels, context_module, num_feature_layers=5)
            else:
                raise ValueError("context module sharing should be either -shared or -independent")
        else:
            self.context = None

        self.head = FCOSHead(backbone.out_channels, num_classes, num_keypoints,
                             use_scale=use_scale, num_feat_levels=len(fpn_strides), loss_weights=loss_weights,
                             use_deformable=use_deformable, **kwargs)

        # used only on torchscript mode
        self._has_warned = False

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

        if not self.training:
            detections = self.predict_proposals(head_outputs, locations, strides, top_feats=None)
            detections = self.postprocess(detections, images.image_sizes, original_image_sizes)
        else:
            if self.trainer_side:
                detections = head_outputs

        return eager_outputs(loss, detections)

    def compute_locations(self, x: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        locations = []
        strides = []
        sois = []
        for i_feature, features in enumerate(x):
            N, C, H, W = features.size()
            stride = self.fpn_strides[i_feature]
            soi = self.sois[i_feature]
            device = features.device

            shifts_x = torch.arange(0, W * stride, step=stride, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, H * stride, step=stride, dtype=torch.float32, device=device)

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            locations_per_level = torch.stack((shift_x, shift_y), dim=1)
            locations_per_level += torch.tensor(stride // 2, dtype=torch.float32, device=device)
            locations.append(locations_per_level)

            stride = torch.tensor(stride, dtype=torch.float32, device=device)
            stride = stride.repeat((H * W, 1))
            strides.append(stride)

            soi = torch.tensor(soi, dtype=torch.float32, device=device)
            soi = soi.repeat((H * W, 1))
            sois.append(soi)

        return torch.cat(locations, dim=0), torch.cat(strides, dim=0), torch.cat(sois, dim=0)

    def compute_relative_targets(self, targets, locations, strides, sois):
        labels = []
        reg_targets = []
        kpoint_targets = []
        target_inds = []

        num_targets = 0
        for i in range(len(targets)):
            targets_per_im = targets[i]
            bboxes = targets_per_im['boxes']
            labels_per_im = targets_per_im['labels']

            # no gt
            if bboxes.numel() == 0:
                labels.append(torch.zeros((locations.size(0)), dtype=torch.int64, device=locations.device) +
                              self.num_classes)
                reg_targets.append(torch.zeros((locations.size(0), 4), dtype=locations.dtype, device=locations.device))
                target_inds.append(torch.zeros((locations.size(0)), dtype=torch.int64, device=locations.device) - 1)

                if self.num_keypoints > 0:
                    kpoint_targets.append(torch.zeros((locations.size(0), self.num_keypoints, 2), dtype=locations.dtype,
                                                      device=locations.device))
                continue

            area = targets_per_im['area']

            reg_targets_per_im = torch.cat([locations[:, None, :] - bboxes[None, :, 0:2],
                                            bboxes[None, :, 2::] - locations[:, None, :]], dim=-1)

            if self.center_sample:
                if "bitmasks_full" in targets_per_im.keys():
                    bitmasks = targets_per_im["bitmasks_full"]
                else:
                    bitmasks = None

                is_in_boxes = get_sample_region(bboxes, strides, locations,
                                                bitmasks=bitmasks, radius=self.radius
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
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            target_inds.append(target_inds_per_im)

            if self.num_keypoints > 0:
                keypoints = targets_per_im['keypoints']
                kpoint_targets_per_im = locations[:, None, None, :] - keypoints[None, :, :, :]
                kpoint_targets_per_im[:, keypoints < 0] = INF
                kpoint_targets_per_im = kpoint_targets_per_im[range(len(locations)), locations_to_gt_inds]

                kpoint_targets.append(kpoint_targets_per_im)

        relative_targets = {"labels": torch.stack(labels, dim=0),
                            "bbox_regression": torch.stack(reg_targets, dim=0) / strides[None, :, :],
                            "target_inds": torch.stack(target_inds, dim=0)
                            }
        if self.num_keypoints > 0:
            kpoint_targets = torch.stack(kpoint_targets, dim=0)
            idx_noninf = kpoint_targets < INF
            kpoint_targets[idx_noninf] /= strides[None, :, :, None].expand_as(kpoint_targets)[idx_noninf]
            relative_targets['keypoints'] = kpoint_targets

        return relative_targets

    def compute_absolute_boxes(self, labels, bbox_regression, locations, strides):

        batch_size = len(bbox_regression)
        pos_inds_1d = torch.nonzero(labels.flatten() != self.num_classes).squeeze(1)
        pos_inds_2d = torch.nonzero(labels != self.num_classes)

        bbox_regression = bbox_regression * strides[None, :, :]
        bbox_regression = torch.stack([locations[None, :, 0] - bbox_regression[:, :, 0],
                                       locations[None, :, 1] - bbox_regression[:, :, 1],
                                       locations[None, :, 0] + bbox_regression[:, :, 2],
                                       locations[None, :, 1] + bbox_regression[:, :, 3],
                                       ], dim=-1)

        bbox_regression = bbox_regression.view(-1, 4)[pos_inds_1d, :]
        results = []
        for i in range(batch_size):
            idx = pos_inds_2d[:, 0] == i
            result = {'boxes': bbox_regression[idx]}
            results.append(result)

        return results

    def predict_proposals(self,
                          predictions: Dict[str, Tensor],
                          locations: Tensor,
                          strides: Tensor,
                          top_feats: Tensor = None):

        cls_logits = predictions['cls_logits'].sigmoid()
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

    def postprocess(self,
                    detections: List[Dict[str, Tensor]],
                    image_shapes: List[Tuple[int, int]],
                    original_image_sizes: List[Tuple[int, int]]
                    ) -> List[Dict[str, Tensor]]:

        for detection in detections:
            detection['correctly_resized_boxes'] = detection["boxes"]

        detections = postprocess(detections, image_shapes, original_image_sizes)
        for detection, image_shape in zip(detections, image_shapes):
            detection['boxes'] = box_ops.clip_boxes_to_image(detection['boxes'], image_shape)

        return detections


def sum_features(features, is_sum=False):
    all_features = []
    if is_sum:
        size = features[0].shape[-2:]
        num_channel = features[0].shape[1]
        device = features[0].device
        upsampled_features = []
        upsample_layer = nn.Sequential(
            nn.Upsample(size=size, mode='bilinear', align_corners=False),
            nn.Conv2d(num_channel, num_channel, kernel_size=1, padding=0, bias=False, device=device),
            nn.BatchNorm2d(num_channel, device=device),
        )
        for i in range(1, len(features)):
            upsampled_feature = upsample_layer(features[i])
            upsampled_features.append(torch.tensor(upsampled_feature, device=device))
        upsampled_features.insert(0, features[0])
        all_features.append(torch.sum(torch.stack(upsampled_features), dim=0))

    else:
        for feature in features:
            all_features.append(feature)

    return all_features


def eager_outputs(loss, detections):
    if loss is None:
        return detections
    elif detections is None:
        return loss
    else:
        return loss, detections


def permute_head_outputs(x):
    # Permute bbox regression output from (N, C, H, W) to (N, HW, C).
    N, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(N, -1, C)  # Size=(N, HW, C)
    return x


def compute_centerness_targets(bbox_target):
    if len(bbox_target) == 0:
        return bbox_target.new_zeros(len(bbox_target))
    left_right = bbox_target[:, [0, 2]]
    top_bottom = bbox_target[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)


def get_sample_region(boxes, strides, locations, bitmasks=None, radius=1.0):
    K = len(locations)

    if bitmasks is not None:
        _, h, w = bitmasks.size()

        ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
        xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

        m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
        m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
        m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
        center_x = m10 / m00
        center_y = m01 / m00
    else:
        center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
        center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

    center_x = center_x.repeat(K, 1)
    center_y = center_y.repeat(K, 1)
    center = torch.stack([center_x, center_y], dim=-1)

    # no gt
    if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
        return torch.zeros_like(locations[:, 0]).to(dtype=torch.utils)

    mins = center - strides.unsqueeze(-1) * radius
    maxs = center + strides.unsqueeze(-1) * radius

    boxes = boxes.repeat(K, 1, 1)
    center_gt = torch.zeros_like(boxes)
    center_gt[:, :, 0:2] = torch.where(mins > boxes[:, :, 0:2], mins, boxes[:, :, 0:2])
    center_gt[:, :, 2::] = torch.where(maxs > boxes[:, :, 2::], boxes[:, :, 2::], maxs)
    center_bbox = torch.cat([locations[:, None, :] - center_gt[:, :, 0:2],
                             center_gt[:, :, 2::] - locations[:, None, :]], dim=-1)
    inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

    return inside_gt_bbox_mask


def ml_nms(boxdict, nms_thresh, max_proposals=-1):
    """
    Performs non-maximum suppression on a boxdict

    Args:
        boxdict (dict of tensors):
        nms_thresh (float):
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
    """

    if nms_thresh <= 0:
        keep = torch.arange(len(boxdict['boxes']), device=boxdict['boxes'].device)
    else:
        assert boxdict['boxes'].shape[-1] == 4
        keep = box_ops.batched_nms(boxdict['boxes'].float(), boxdict['scores'], boxdict['labels'], nms_thresh)
        if max_proposals > 0:
            keep = keep[: max_proposals]

    for key in boxdict.keys():
        if isinstance(boxdict[key], list):
            boxdict[key] = [b[keep] for b in boxdict[key]]
        else:
            boxdict[key] = boxdict[key][keep]

    return boxdict


def compute_ious(pred, target):
    """
    Args:
        pred: Nx4 predicted bounding boxes
        target: Nx4 target bounding boxes
        Both are in the form of FCOS prediction (l, t, r, b)
    """
    pred_left, pred_top, pred_right, pred_bottom = pred.unbind(-1)
    target_left, target_top, target_right, target_bottom = target.unbind(-1)

    target_area = (target_left + target_right) * (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

    g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
    ac_uion = g_w_intersect * g_h_intersect

    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect

    ious = area_intersect / area_union
    gious = ious - (ac_uion - area_union) / ac_uion

    return ious, gious


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def fcos_resnet_fpn(type="resnet50", pretrained=False, progress=True,
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
    model = FCOSNet(backbone, num_classes, **kwargs)
    return model


def fcos_regnet_fpn(type="regnet_y_1_6gf", pretrained=False, progress=True,
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
    model = FCOSNet(backbone, num_classes, **kwargs)
    return model


def fcos_convnext_fpn(type="convnext_tiny", pretrained=False, progress=True,
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

    backbone = convnext_fpn_backbone(type, pretrained_backbone, returned_layers=returned_layers,
                                     extra_blocks=extra_blocks, trainable_layers=trainable_backbone_layers,
                                     extra_before=extra_before,
                                     pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats,
                                     fusion_type=fusion_type, bifpn_norm_layer=bifpn_norm_layer)
    model = FCOSNet(backbone, num_classes, **kwargs)
    return model


def fcos_efficientnet_fpn(type="efficientnet_b4", num_classes=91, pretrained_backbone=True,
                          trainable_backbone_layers=None, use_P2=False, no_extra_blocks=False, extra_before=False,
                          pyramid_type="bifpn", depthwise=True, repeats=3, fusion_type='fastnormed',
                          bifpn_norm_layer=None, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

    if use_P2:
        returned_layers = [1, 2, 3, 4]
        extra_blocks = None if no_extra_blocks else LastLevelP6 if pyramid_type == "bifpn" else LastLevelMaxPool
    else:
        # skip P2 because it generates too many anchors (according to their paper)
        returned_layers = [2, 3, 4]
        extra_blocks = None if no_extra_blocks else LastLevelP6P7

    backbone = efficientnet_fpn_backbone(type, pretrained_backbone, returned_layers=returned_layers,
                                         extra_blocks=extra_blocks, trainable_layers=trainable_backbone_layers,
                                         extra_before=extra_before, pyramid_type=pyramid_type, depthwise=depthwise,
                                         repeats=repeats, fusion_type=fusion_type, bifpn_norm_layer=bifpn_norm_layer)
    model = FCOSNet(backbone, num_classes, **kwargs)
    return model
