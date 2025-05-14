from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Union

import torch
from torch import nn, Tensor

from .backbone_utils import resnet_fpn_backbone, _validate_trainable_layers, regnet_fpn_backbone, convnext_fpn_backbone
from .process import batch_images, postprocess, get_original_image_sizes
from tairvision_object_detection.ops.feature_pyramid_network import LastLevelP6P7, FeaturePyramidNetwork
from tairvision_object_detection.ops import boxes as box_ops
from tairvision_object_detection.ops import MultiScaleRoIAlign

from tairvision_object_detection.models.detection.context import IndependentContextNetwork, SharedContextNetwork
from tairvision_object_detection.models.detection.fcos import FCOSHead, ml_nms, get_sample_region, eager_outputs, sum_features
from tairvision_object_detection.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from tairvision_object_detection.models.detection.roi_heads import maskrcnn_loss, maskrcnn_inference

__all__ = [
    "MASKFCOSNet", "MaskHead", "maskfcos_regnet_fpn", "maskfcos_resnet_fpn", "maskfcos_convnext_fpn"
]

INF = 100000000


class MASKFCOSNet(nn.Module):

    def __init__(self, backbone, num_classes, num_keypoints=0,
                 context_module=None,
                 fpn_strides=(8, 16, 32, 64, 128),
                 sois=(-1, 64, 128, 256, 512, INF),
                 center_sample=True,
                 radius=1.5,
                 thresh_with_ctr=False,
                 use_scale=True,
                 use_deformable=False,
                 loss_weights=(1.0, 1.0, 1.0, 1.0),
                 pre_nms_thresh=0.05,
                 pre_nms_topk=1000,
                 nms_thresh=0.6,
                 post_nms_topk=100,
                 roi_output_size=14,
                 roi_sampling_ratio=2,
                 use_P2=False,
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
        if use_P2:
            self.box_feat_range = slice(1, 6)  # use P3-P7
            self.seg_feat_range = slice(0, 4)  # use P2-P5
            self.neck_sub = FeaturePyramidNetwork(in_channels_list=[64, 256], out_channels=256)
        else:
            self.box_feat_range = slice(0, 5)  # use P3-P7
            self.seg_feat_range = slice(0, 3)  # use P3-P5

        self.use_P2 = use_P2

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
                             use_deformable=use_deformable)

        self.mask_head = MaskHead(backbone.out_channels, num_classes,
                                  roi_output_size=roi_output_size,
                                  roi_sampling_ratio=roi_sampling_ratio)

        # used only on torchscript mode
        self._has_warned = False

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None) \
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
        if self.use_P2:
            features['0'] = self.neck_sub({'0': features['0'], '1': features['1']})['0']

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        if self.context is not None:
            features = self.context(features)

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        all_features = sum_features(features, is_sum=False)

        # compute the fcos heads outputs using the features
        head_outputs = self.head(all_features[self.box_feat_range])

        locations, strides, sois = self.compute_locations(all_features[self.box_feat_range])
        loss, detections = None, None

        if targets is not None:
            targets = self.compute_relative_targets(targets, locations, strides, sois)
            loss = self.head.compute_loss(targets, head_outputs)

            with torch.no_grad():
                proposals = self.compute_absolute_boxes(targets['labels'], head_outputs['bbox_regression'],
                                                        locations, strides)

            mask_loss = self.mask_head(all_features[self.seg_feat_range], proposals, images.image_sizes, targets)
            loss['mask_loss'] = mask_loss

        if not self.training:
            detections = self.predict_proposals(head_outputs, locations, strides, top_feats=None)
            detections = self.mask_head(all_features[self.seg_feat_range], detections, images.image_sizes)
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
        masks = []
        mask_labels = []
        target_inds = []
        matched_idxs = []

        num_targets = 0
        for i in range(len(targets)):
            targets_per_im = targets[i]
            bboxes = targets_per_im['boxes']
            labels_per_im = targets_per_im['labels']

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                if self.num_keypoints > 0:
                    kpoint_targets.append(locations.new_zeros((locations.size(0), self.num_keypoints, 2)))
                continue

            area = torch.tensor(targets_per_im['area']).to(bboxes.device)

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

            # matched idx and labels to remove the need for matcher
            matched_idxs_per_im = locations_to_gt_inds[locations_to_min_area != INF]
            matched_idxs.append(matched_idxs_per_im)

            if self.num_keypoints > 0:
                keypoints = targets_per_im['keypoints']
                kpoint_targets_per_im = locations[:, None, None, :] - keypoints[None, :, :, :]
                kpoint_targets_per_im[:, keypoints < 0] = INF
                kpoint_targets_per_im = kpoint_targets_per_im[range(len(locations)), locations_to_gt_inds]

                kpoint_targets.append(kpoint_targets_per_im)

            if 'masks' in targets_per_im.keys():
                masks.append(targets_per_im['masks'])
                mask_labels.append(targets_per_im['labels'])

        relative_targets = {"labels": torch.stack(labels, dim=0),
                            "bbox_regression": torch.stack(reg_targets, dim=0) / strides[None, :, :],
                            "target_inds": torch.stack(target_inds, dim=0)
                            }
        if self.num_keypoints > 0:
            kpoint_targets = torch.stack(kpoint_targets, dim=0)
            idx_noninf = kpoint_targets < INF
            kpoint_targets[idx_noninf] /= strides[None, :, :, None].expand_as(kpoint_targets)[idx_noninf]
            relative_targets['keypoints'] = kpoint_targets

        if len(masks) > 0:
            relative_targets['masks'] = masks
            relative_targets['mask_labels'] = mask_labels
        else:
            relative_targets['masks'] = None
            relative_targets['mask_labels'] = None

        relative_targets['matched_idxs'] = matched_idxs

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

        for detection, image_shape in zip(detections, image_shapes):
            detection['correctly_resized_boxes'] = detection["boxes"]
            detection['correctly_resized_masks'] = detection["masks"]
            detection["image_shape"] = image_shape

        detections = postprocess(detections, image_shapes, original_image_sizes)
        for detection, image_shape in zip(detections, image_shapes):
            detection['boxes'] = box_ops.clip_boxes_to_image(detection['boxes'], image_shape)
            detection['masks'] = detection['masks'].unsqueeze(1) > 0.10

        return detections


class MaskHead(nn.Module):
    def __init__(self, in_channels, num_classes,
                 roi_output_size=14, roi_sampling_ratio=2,
                 ):
        super(MaskHead, self).__init__()

        self.roi_align = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                            output_size=roi_output_size,
                                            sampling_ratio=roi_sampling_ratio)
        mask_layers = [256, 256, 256, 256]
        self.head = MaskRCNNHeads(in_channels, mask_layers, 1)
        self.predictor = MaskRCNNPredictor(in_channels, 256, num_classes)

    def forward(self,
                features,           # type: Dict[str, Tensor]
                detections,         # type: List[Dict[str, Tensor]]
                image_shapes,       # type: List[Tuple[int, int]]
                targets=None,       # type: Optional[List[Dict[str, Tensor]]]
                ):

        boxes = [detection['boxes'] for detection in detections]

        mask_features = self.roi_align(features, boxes, image_shapes)
        mask_features = self.head(mask_features)
        mask_logits = self.predictor(mask_features)

        if targets is not None:
            matched_idxs = targets['matched_idxs']
            loss = maskrcnn_loss(mask_logits, boxes,
                                 targets["masks"], targets['mask_labels'],
                                 matched_idxs)
            return loss
        else:
            labels = [detection['labels'] for detection in detections]
            masks_probs = maskrcnn_inference(mask_logits, labels)
            for detection, masks_prob, image_shape in zip(detections, masks_probs, image_shapes):
                detection['masks'] = masks_prob
        return detections


def maskfcos_resnet_fpn(type="resnet50", pretrained=False, progress=True,
                        num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None,
                        use_P2=False, no_extra_blocks=False, extra_before=False,
                        pyramid_type="bifpn", depthwise=True, repeats=3, fusion_type='fastnormed',
                        bifpn_norm_layer=None, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

    if use_P2:
        returned_layers = [1, 2, 3, 4]
    else:
        # skip P2 because it generates too many anchors (according to their paper)
        returned_layers = [2, 3, 4]
    extra_blocks = None if no_extra_blocks else LastLevelP6P7

    backbone = resnet_fpn_backbone(type, pretrained_backbone, returned_layers=returned_layers,
                                   extra_blocks=extra_blocks, trainable_layers=trainable_backbone_layers,
                                   extra_before=extra_before,
                                   pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats,
                                   fusion_type=fusion_type, bifpn_norm_layer=bifpn_norm_layer)
    model = MASKFCOSNet(backbone, num_classes, use_P2=use_P2, **kwargs)
    return model


def maskfcos_regnet_fpn(type="regnet_y_1_6gf", pretrained=False, progress=True,
                        num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None,
                        use_P2=False, no_extra_blocks=False, extra_before=False,
                        pyramid_type="bifpn", depthwise=True, repeats=3, fusion_type='fastnormed',
                        bifpn_norm_layer=None, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

    if use_P2:
        returned_layers = [1, 2, 3, 4]
        kwargs['skip_fpn'] = [True, False, False, False]
    else:
        # skip P2 because it generates too many anchors (according to their paper)
        returned_layers = [2, 3, 4]
        kwargs['skip_fpn'] = None
    extra_blocks = None if no_extra_blocks else LastLevelP6P7

    backbone = regnet_fpn_backbone(type, pretrained_backbone, returned_layers=returned_layers,
                                   extra_blocks=extra_blocks, trainable_layers=trainable_backbone_layers,
                                   extra_before=extra_before,
                                   pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats,
                                   fusion_type=fusion_type, bifpn_norm_layer=bifpn_norm_layer, **kwargs)
    model = MASKFCOSNet(backbone, num_classes, use_P2=use_P2, **kwargs)
    return model


def maskfcos_convnext_fpn(type="convnext_tiny", pretrained=False, progress=True,
                          num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None,
                          use_P2=False, no_extra_blocks=False, extra_before=False,
                          pyramid_type="bifpn", depthwise=True, repeats=3, fusion_type='fastnormed',
                          bifpn_norm_layer=None, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

    if use_P2:
        returned_layers = [1, 2, 3, 4]
    else:
        # skip P2 because it generates too many anchors (according to their paper)
        returned_layers = [2, 3, 4]
    extra_blocks = None if no_extra_blocks else LastLevelP6P7

    backbone = convnext_fpn_backbone(type, pretrained_backbone, returned_layers=returned_layers,
                                     extra_blocks=extra_blocks, trainable_layers=trainable_backbone_layers,
                                     extra_before=extra_before,
                                     pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats,
                                     fusion_type=fusion_type, bifpn_norm_layer=bifpn_norm_layer)
    model = MASKFCOSNet(backbone, num_classes, use_P2=use_P2, **kwargs)
    return model
