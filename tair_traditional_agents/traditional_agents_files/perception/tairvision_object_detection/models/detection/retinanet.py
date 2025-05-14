import math
from collections import OrderedDict
import warnings

import torch
from torch import nn, Tensor
from typing import Dict, List, Tuple, Optional

from ._utils import overwrite_eps
from tairvision_object_detection._internally_replaced_utils import load_state_dict_from_url

from . import _utils as det_utils
from .anchor_utils import AnchorGenerator
from .process import batch_images, postprocess, get_original_image_sizes
from .backbone_utils import resnet_fpn_backbone, _validate_trainable_layers, regnet_fpn_backbone
from tairvision_object_detection.ops.feature_pyramid_network import LastLevelP6, LastLevelP6P7, LastLevelMaxPool
from tairvision_object_detection.ops import sigmoid_focal_loss
from tairvision_object_detection.ops import boxes as box_ops

from tairvision_object_detection.models.detection.context import IndependentContextNetwork, SharedContextNetwork

__all__ = [
    "RetinaNet", "retinanet_regnet_fpn", "retinanet_resnet_fpn",
]


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


class RetinaNetHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_keypoints=0, loss_weights=(1.0, 1.0, 1.0)):
        super().__init__()
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)
        self.loss_weights = loss_weights
        self.num_keypoints = num_keypoints
        if self.num_keypoints > 0:
            self.regression_keypoint_head = RetinaNetKpointRegressionHead(in_channels, num_anchors, num_keypoints)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]

        cls_weight, bbox_weight, kpoint_weight, _ = self.loss_weights

        cls_loss = self.classification_head.compute_loss(targets, head_outputs, matched_idxs)
        bbox_loss = self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs)

        loss_dict = {
            'classification': cls_loss * cls_weight,
            'bbox_regression': bbox_loss * bbox_weight
        }

        if self.num_keypoints > 0:
            kpoint_loss = self.regression_keypoint_head.compute_loss(targets, head_outputs, anchors, matched_idxs)
            loss_dict['kpoint_regression'] = kpoint_loss * kpoint_weight

        return loss_dict

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        out_list = {
            'cls_logits': self.classification_head(x),
            'bbox_regression': self.regression_head(x)
        }

        if self.num_keypoints > 0:
            out_list['kpoint_regression'] = self.regression_keypoint_head(x)

        return out_list


class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    def compute_loss(self, targets, head_outputs, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        losses = []

        cls_logits = head_outputs['cls_logits']

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image['labels'][matched_idxs_per_image[foreground_idxs_per_image]]
            ] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            losses.append(sigmoid_focal_loss(
                cls_logits_per_image[valid_idxs_per_image],
                gt_classes_target[valid_idxs_per_image],
                reduction='sum',
            ) / max(1, num_foreground))

        return _sum(losses) / len(targets)

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
    }

    def __init__(self, in_channels, num_anchors):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        bbox_regression = head_outputs['bbox_regression']

        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in \
                zip(targets, bbox_regression, anchors, matched_idxs):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image['boxes'][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the regression targets
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)

            # compute the loss
            losses.append(torch.nn.functional.l1_loss(
                bbox_regression_per_image,
                target_regression,
                reduction='sum'
            ) / max(1, num_foreground))

        return _sum(losses) / max(1, len(targets))

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)


class RetinaNetKpointRegressionHead(nn.Module):
    """
    A keypoint regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """
    __annotations__ = {
        'keypoint_coder': det_utils.KeypointCoder
    }

    def __init__(self, in_channels, num_anchors, num_keypoints):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.num_keypoints = num_keypoints
        self.kpoint_reg = nn.Conv2d(in_channels, num_anchors * num_keypoints * 2, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.kpoint_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.kpoint_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.keypoint_coder = det_utils.KeypointCoder()

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        kpoint_regression = head_outputs['kpoint_regression']

        for targets_per_image, kpoint_regression_per_image, anchors_per_image, matched_idxs_per_image in \
                zip(targets, kpoint_regression, anchors, matched_idxs):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground keypoints
            matched_gt_keypoints_per_image = targets_per_image['keypoints'][
                matched_idxs_per_image[foreground_idxs_per_image]]
            kpoint_regression_per_image = kpoint_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the regression targets
            target_regression = self.keypoint_coder.encode_single(matched_gt_keypoints_per_image, anchors_per_image)

            # ignore negative valued keypoints
            valid_idxs_per_image = (matched_gt_keypoints_per_image >= 0).all(2)

            # compute the loss
            losses.append(torch.nn.functional.smooth_l1_loss(
                kpoint_regression_per_image.reshape(target_regression.shape)[valid_idxs_per_image],
                target_regression[valid_idxs_per_image],
                reduction='sum'
            ) / max(1, num_foreground))

        return _sum(losses) / max(1, len(targets))

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_kpoint_regression = []

        for features in x:
            kpoint_regression = self.conv(features)
            kpoint_regression = self.kpoint_reg(kpoint_regression)

            # Permute bbox regression output from (N, K * A, H, W) to (N, HWA, K).
            N, _, H, W = kpoint_regression.shape
            K = self.num_keypoints * 2
            kpoint_regression = kpoint_regression.view(N, -1, K, H, W)
            kpoint_regression = kpoint_regression.permute(0, 3, 4, 1, 2)
            kpoint_regression = kpoint_regression.reshape(N, -1, K)  # Size=(N, HWA, K)

            all_kpoint_regression.append(kpoint_regression)

        return torch.cat(all_kpoint_regression, dim=1)


class RetinaNet(nn.Module):
    """
    Implements RetinaNet.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
        pre_nms_thresh (float): Score threshold used for postprocessing the detections.
        pre_nms_topk (int): Number of best detections to keep before NMS.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        post_nms_topk (int): Number of best detections to keep after NMS.

    Example:

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import RetinaNet
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # RetinaNet needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the network generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(
        >>>     sizes=((32, 64, 128, 256, 512),),
        >>>     aspect_ratios=((0.5, 1.0, 2.0),)
        >>> )
        >>>
        >>> # put the pieces together inside a RetinaNet model
        >>> model = RetinaNet(backbone,
        >>>                   num_classes=2,
        >>>                   anchor_generator=anchor_generator)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'keypoint_coder': det_utils.KeypointCoder
    }

    def __init__(self, backbone, num_classes, num_keypoints=0,
                 head=None,
                 context_module=None,
                 # Anchor parameters
                 anchor_generator=None, anchor_sizes=[32, 64, 128, 256, 512],
                 # Proposal matcher parameters
                 proposal_matcher=None, fg_iou_thresh=0.5, bg_iou_thresh=0.4,
                 loss_weights=(1.0, 1.0, 1.0),
                 pre_nms_thresh=0.05,
                 pre_nms_topk=1000,
                 nms_thresh=0.6,
                 post_nms_topk=100
                 ):
        super().__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")
        self.backbone = backbone

        assert isinstance(anchor_generator, (AnchorGenerator, type(None)))

        if anchor_generator is None:
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in anchor_sizes)
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        self.anchor_generator = anchor_generator
        self.num_keypoints = num_keypoints

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

        if head is None:
            head = RetinaNetHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0],
                                 num_classes, num_keypoints, loss_weights=loss_weights)
        self.head = head

        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.proposal_matcher = proposal_matcher

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        if num_keypoints > 0:
            self.keypoint_coder = det_utils.KeypointCoder()

        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.nms_thresh = nms_thresh
        self.post_nms_topk = post_nms_topk

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image['boxes'].numel() == 0:
                matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64,
                                               device=anchors_per_image.device))
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image['boxes'], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs['cls_logits']
        box_regression = head_outputs['bbox_regression']
        if self.num_keypoints > 0:
            kpoint_regression = head_outputs['kpoint_regression']

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            if self.num_keypoints > 0:
                kpoint_regression_per_image = [lm[index] for lm in kpoint_regression]
            else:
                kpoint_regression_per_image = [None for br in box_regression]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []
            image_keypoints = []

            for box_regression_per_level, logits_per_level, kpoint_regression_per_level, anchors_per_level in \
                    zip(box_regression_per_image, logits_per_image, kpoint_regression_per_image, anchors_per_image):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.pre_nms_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = min(self.pre_nms_topk, topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode='floor')
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(box_regression_per_level[anchor_idxs],
                                                               anchors_per_level[anchor_idxs])
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

                if self.num_keypoints > 0:
                    keypoints_per_level = self.keypoint_coder.decode_single(
                        kpoint_regression_per_level[anchor_idxs].reshape(-1, self.num_keypoints, 2),
                        anchors_per_level[anchor_idxs])
                    # TODO: Check if this is needed
                    keypoints_per_level = det_utils.clip_keypoints_to_image(keypoints_per_level, image_shape)

                    image_keypoints.append(keypoints_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            if self.num_keypoints > 0:
                image_keypoints = torch.cat(image_keypoints, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.post_nms_topk]

            det_dict = {
                'boxes': image_boxes[keep],
                'scores': image_scores[keep],
                'labels': image_labels[keep]
            }
            if self.num_keypoints > 0:
                det_dict['keypoints'] = image_keypoints[keep]

            detections.append(det_dict)

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

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

        original_image_sizes = get_original_image_sizes(targets) if targets is not None else []

        if not self.training:
            targets = None

        # transform the input
        images = batch_images(images)

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

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None

            # compute the losses
            losses = self.compute_loss(targets, head_outputs, anchors)
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs['cls_logits'].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)


def retinanet_resnet_fpn(type="resnet50", pretrained=False, progress=True,
                         num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None,
                         use_P2=False, no_extra_blocks=False, extra_before=False,
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

    backbone = resnet_fpn_backbone(type, pretrained_backbone, returned_layers=returned_layers,
                                   extra_blocks=extra_blocks, trainable_layers=trainable_backbone_layers,
                                   extra_before=extra_before,
                                   pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats,
                                   fusion_type=fusion_type, bifpn_norm_layer=bifpn_norm_layer)
    model = RetinaNet(backbone, num_classes, **kwargs)
    return model


def retinanet_regnet_fpn(type="regnet_y_1_6gf", pretrained=False, progress=True,
                         num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None,
                         use_P2=False, no_extra_blocks=False, extra_before=False,
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

    backbone = regnet_fpn_backbone(type, pretrained_backbone, returned_layers=returned_layers,
                                   extra_blocks=extra_blocks, trainable_layers=trainable_backbone_layers,
                                   extra_before=extra_before,
                                   pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats,
                                   fusion_type=fusion_type, bifpn_norm_layer=bifpn_norm_layer)
    model = RetinaNet(backbone, num_classes, **kwargs)
    return model
