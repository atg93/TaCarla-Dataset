# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py

from torch import nn, Tensor
from ._utils import _SimpleSegmentationModel
from .mask2former_sub import MSDeformAttnPixelDecoder
from .mask2former_sub import ShapeSpec
from .mask2former_sub import MultiScaleMaskedTransformerDecoder
from collections import OrderedDict
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, NamedTuple


class Mask2Former(_SimpleSegmentationModel):
    def __init__(self, *args, **kwargs):
        super(Mask2Former, self).__init__(*args, **kwargs)

    def forward(self, image):
        features = self.backbone(image)

        result = OrderedDict()
        self._get_result_from_classifiers(features, result, self.classifier_keys, self.classifier_list, image)

        if self.aux_classifier_list is not None:
            self._get_result_from_classifiers(features, result, self.aux_classifier_keys, self.aux_classifier_list, image)

        return result

    def _get_result_from_classifiers(self, features, result, classifier_keys, classifier_list, image):
        size = image.shape[-2:]
        for key, classifier in zip(classifier_keys, classifier_list):
            x = classifier(features)
            # TODO, check whether the resolution is true or not during the training. Resolution vs memory tradeoff,
            #  the auxiliaries also??
            if self.training:
                continue
            classifier_output_value = x['pred_masks']
            classifier_output_value_resize = \
                F.interpolate(classifier_output_value, size=size, mode='bilinear', align_corners=False)
            x['pred_masks'] = classifier_output_value_resize

            # Resizing aux outputs are not necessary ??
            # for aux_out in x['aux_outputs']:
            #     aux_classifier_output_value = aux_out['pred_masks']
            #     aux_classifier_output_value_resize = \
            #         F.interpolate(aux_classifier_output_value, size=size, mode='bilinear', align_corners=False)
            #     aux_out['pred_masks'] = aux_classifier_output_value_resize

        result[key] = x


class Mask2FormerHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes, size,
                 number_of_channel_levels,
                 scale_factor_list,
                 level_sizes,
                 activate_levels,
                 **kwargs):
        super(Mask2FormerHead, self).__init__()

        input_shape = {}
        for i in range(3):
            level_shape = ShapeSpec(
                channels=number_of_channel_levels[i],
                height=level_sizes[i][0],
                width=level_sizes[i][1],
                stride=scale_factor_list[i]
            )
            input_shape.update({f"stage{i+1}": level_shape})

        # Sizes in ShapeSpec seems to be not utilized, therefore it is not crucial to set it correctly
        out_shape = ShapeSpec(channels=in_channels, height=size[0], width=size[1], stride=scale_factor_list[-1])
        input_shape.update({"out": out_shape})
        self.pixel_encoder = MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            **kwargs["pixel_encoder_config"]
        )

        self.transformer_decoder = MultiScaleMaskedTransformerDecoder(
            num_classes=num_classes,
            **kwargs["masked_transformer_decoder_config"]
        )

    def forward(self, features):
        mask_features, out, multi_scale_features = self.pixel_encoder.forward_features(features)
        out = self.transformer_decoder(x=multi_scale_features, mask_features=mask_features)
        return out


