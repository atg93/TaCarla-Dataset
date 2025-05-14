import torch.nn as nn
import torch
import torch.nn.functional as F
from ._utils import _SimpleSegmentationModel
from collections import OrderedDict
from .deeplabv3 import ASPP


class OCRNet(_SimpleSegmentationModel):
    def __init__(self, *args, **kwargs):
        super(OCRNet, self).__init__(*args, **kwargs)
        if self.aux_classifier_list is None:
            raise ValueError("Auxilarry classifier cannot be None in OCRNet implementation")

    def forward(self, image):
        features = self.backbone(image)
        result = OrderedDict()
        for out_key, aux_key, aux_classifier, classifier in zip(self.classifier_keys, self.aux_classifier_keys,
                                                                self.aux_classifier_list, self.classifier_list):
            raw_segmentation_results = aux_classifier(features)
            x = classifier(features, raw_segmentation_results)
            if not self.deployment_mode:
                size = image.shape[-2:]
                x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
                raw_segmentation_results = F.interpolate(raw_segmentation_results, size=size, mode='bilinear', align_corners=False)
            else:
                x = self.upsample(x)
                raw_segmentation_results = self.upsample(raw_segmentation_results)
            result[out_key] = x
            result[aux_key] = raw_segmentation_results

        return result


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )


class CreateObjectRegionRepresentation(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, features_in_channels, output_size, deployment_mode, cls_num=0, scale=1):
        super(CreateObjectRegionRepresentation, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.output_size = output_size
        self.hw = self.output_size[0] * self.output_size[1]
        self.features_in_channels = features_in_channels
        self.deployment_mode = deployment_mode

    def forward(self, features, raw_segmentation_results):
        if not self.deployment_mode:
            batch_size, c, h, w = raw_segmentation_results.size(0), raw_segmentation_results.size(1), \
                                  raw_segmentation_results.size(2), raw_segmentation_results.size(3)

            raw_segmentation_results = raw_segmentation_results.view(batch_size, c, -1)
            features = features.view(batch_size, features.size(1), -1)
        else:
            raw_segmentation_results = raw_segmentation_results.view(-1, self.cls_num, self.hw)
            features = features.view(-1, self.features_in_channels, self.hw)
        features = features.permute(0, 2, 1)  # batch x hw x c
        raw_segmentation_probs = F.softmax(self.scale * raw_segmentation_results, dim=2)  # batch x k x hw
        object_region_representation = torch.matmul(raw_segmentation_probs, features).permute(0, 2, 1).unsqueeze(
            3)  # batch x k x c
        return object_region_representation


class _CreateObjectContextualRepresentation(nn.Module):
    """
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 cls_num,
                 output_size,
                 deployment_mode,
                 scale=1,
                 bn_type=None):
        super(_CreateObjectContextualRepresentation, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.cls_num = cls_num
        self.scale = scale
        self.output_size = output_size
        self.deployment_mode = deployment_mode
        self.hw = self.output_size[0] * self.output_size[1]
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        if not self.deployment_mode:
            # Pixel representations are utilized as query
            query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
            # object representations are utilized as key an value in the transformer concept
            key = self.f_object(proxy).view(batch_size, self.key_channels, -1)  # (batch x dimension x k)
            value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        else:
            query = self.f_pixel(x).view(-1, self.key_channels, self.hw)
            key = self.f_object(proxy).view(-1, self.key_channels, self.cls_num)  # (batch x dimension x k)
            value = self.f_down(proxy).view(-1, self.key_channels, self.cls_num)
        query = query.permute(0, 2, 1)  # (batch x hw x dimension)
        value = value.permute(0, 2, 1)  # (batch x k x dimension)

        pixel_region_relation = torch.matmul(query, key)  # (batch x hw x k)
        # in other words, the relation of every pixel to the every contextual representation
        normalized_pixel_region_relation = (self.key_channels ** -.5) * pixel_region_relation
        # Normalized with respect to the dimension of features
        normalized_pixel_region_relation_attention = F.softmax(normalized_pixel_region_relation, dim=-1)

        # add bg context ...
        context = torch.matmul(normalized_pixel_region_relation_attention, value)
        context = context.permute(0, 2, 1).contiguous()
        if not self.deployment_mode:
            context = context.view(batch_size, self.key_channels, *x.size()[2:])
        else:
            context = context.view(-1, self.key_channels, *self.output_size)
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear')

        return context


class CreateObjectContextualRepresentation(_CreateObjectContextualRepresentation):
    def __init__(self,
                 in_channels,
                 key_channels,
                 cls_num,
                 output_size,
                 deployment_mode,
                 scale=1,
                 bn_type=None):
        super(CreateObjectContextualRepresentation, self).__init__(in_channels,
                                                                   key_channels,
                                                                   cls_num,
                                                                   output_size,
                                                                   deployment_mode,
                                                                   scale,
                                                                   bn_type=bn_type)


class OCRHead(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 size,
                 object_contextual_in_channels,
                 object_contextual_key_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None,
                 **kwargs):
        super(OCRHead, self).__init__()
        self.dropout = dropout
        self.object_contextual_in_channels = object_contextual_in_channels
        self._in_channels = self.return_number_of_in_channels()
        self.output_size = size
        self.deployment_mode = kwargs.get('deployment_mode', False)
        self.in_channels = in_channels
        self.size = size

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(in_channels, object_contextual_in_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(object_contextual_in_channels),
            nn.ReLU(inplace=True),
        )

        self.object_region_representation_block = \
            CreateObjectRegionRepresentation(features_in_channels=object_contextual_in_channels,
                                             output_size=self.output_size,
                                             deployment_mode=self.deployment_mode,
                                             cls_num=num_classes, scale=scale)

        self.object_context_representation_block = \
            CreateObjectContextualRepresentation(in_channels=object_contextual_in_channels,
                                                 key_channels=object_contextual_key_channels,
                                                 cls_num=num_classes,
                                                 output_size=self.output_size,
                                                 deployment_mode=self.deployment_mode,
                                                 scale=scale,
                                                 bn_type=bn_type)

        self.conv_bn_dropout = self.create_conv_bn_dropout_layer()

        self.cls_head = nn.Conv2d(
            object_contextual_in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)

        self.output_key = kwargs.get('output_key', 'out')

        self._initialize_weights()

    def forward(self, features, raw_segmentation_result):
        contextual_representation, features_ocr = self.forward_ocr_context(features, raw_segmentation_result)
        output = self.conv_bn_dropout(torch.cat([contextual_representation, features_ocr], 1))
        ocr_segmentation_result = self.cls_head(output)
        return ocr_segmentation_result

    def forward_ocr_context(self, features, raw_segmentation_result):
        features = features[self.output_key]
        features = self.conv3x3_ocr(features)
        object_region_representation = self.object_region_representation_block(
            features, raw_segmentation_result)
        contextual_representation = self.object_context_representation_block(features, object_region_representation)
        return contextual_representation, features

    def create_conv_bn_dropout_layer(self):
        conv_bn_dropout = nn.Sequential(
            nn.Conv2d(self._in_channels, self.object_contextual_in_channels, kernel_size=(1, 1), padding=0, bias=False),
            ModuleHelper.BNReLU(self.object_contextual_in_channels),
            nn.Dropout2d(self.dropout)
        )
        return conv_bn_dropout

    def return_number_of_in_channels(self):
        return 2 * self.object_contextual_in_channels

    def _initialize_weights(self):
        """
        Initialize Model Weights
        """

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class OCRASPPHead(OCRHead):
    def __init__(self,
                 atrous_rates,
                 aspp_number_of_channels,
                 *args, **kwargs):
        self.aspp_number_of_channels = aspp_number_of_channels
        super(OCRASPPHead, self).__init__(*args, **kwargs)
        self.aspp = ASPP(self.in_channels, atrous_rates, self.size,
                         self.deployment_mode, out_channels=aspp_number_of_channels)
        # TODO, default number of ASPP output channels is 256, should we make parametric

        self._initialize_weights()

    def forward(self, features, raw_segmentation_result):
        contextual_representation, features_ocr = self.forward_ocr_context(features, raw_segmentation_result)
        features = features[self.output_key]
        aspp_outputs = self.aspp(features)
        output = self.conv_bn_dropout(torch.cat([contextual_representation, features_ocr, aspp_outputs], 1))
        ocr_segmentation_result = self.cls_head(output)
        return ocr_segmentation_result

    def return_number_of_in_channels(self):
        return 2 * self.object_contextual_in_channels + self.aspp_number_of_channels
