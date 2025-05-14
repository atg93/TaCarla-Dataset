from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, NamedTuple
from torch import nn
from torch.nn import functional as F


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier_list: List, aux_classifier_list: Optional[List] = None,
                 size=(300, 600), **kwargs):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone

        self.classifier_list = classifier_list
        self.aux_classifier_list = aux_classifier_list
        self.classifier_keys = []
        self.aux_classifier_keys = []

        for i, module in enumerate(self.classifier_list):
            if i == 0:
                self.add_module('classifier', module)
                self.classifier_keys.append("out")
            else:
                self.add_module('classifier_' + str(i+1), module)
                self.classifier_keys.append(f"out_{i+1}")

        if self.aux_classifier_list is not None:
            for i, module in enumerate(self.aux_classifier_list):
                if i == 0:
                    self.add_module('aux_classifier', module)
                    self.aux_classifier_keys.append("aux")
                else:
                    self.add_module('aux_classifier_' + str(i+1), module)
                    self.aux_classifier_keys.append(f"aux_{i+1}")

        # TODO Different upsample is needed for aux layer,
        #  but there is also a need that deployment should not have aux layer actually.
        self.upsample = nn.Upsample(size=size, mode='bilinear', align_corners=False)

        self.deployment_mode = kwargs.get('deployment_mode', False)

    def forward(self, image):
        features = self.backbone(image)
        result = OrderedDict()
        self._get_result_from_classifiers(features, result, self.classifier_keys, self.classifier_list, image)

        if self.aux_classifier_list is not None:
            self._get_result_from_classifiers(features, result, self.aux_classifier_keys, self.aux_classifier_list, image)

        return result


    def _get_result_from_classifiers(self, features, result, classifier_keys, classifier_list, image):
        for key, classifier in zip(classifier_keys, classifier_list):
            x = classifier(features)
            if isinstance(x, Dict):
                for classifier_output_key in x.keys():
                    if not self.deployment_mode:
                        size = image.shape[-2:]
                        classifier_output_value = x[classifier_output_key]
                        classifier_output_value_resize = \
                            F.interpolate(classifier_output_value, size=size, mode='bilinear', align_corners=False)
                        if classifier_output_key == "offset":
                            scale = (size[0] - 1) // (classifier_output_value.shape[2] - 1)
                            classifier_output_value_resize *= scale
                        x[classifier_output_key] = classifier_output_value_resize
                    else:
                        raise ValueError("Not implemented for the time being")
            else:
                if not self.deployment_mode:
                    size = image.shape[-2:]
                    x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
                else:
                    x = self.upsample(x)
            result[key] = x


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module


def preprocess_temporal_info(image):
    image = image.permute(0, 2, 1, 3, 4)
    image_shape = image.shape
    batch_size = image_shape[0]
    image = image.view(-1, image_shape[2], image_shape[3], image_shape[4])
    return image, batch_size


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    from the implementation:
    DeeplabV3Plus - https://github.com/VainF/DeepLabV3Plus-Pytorch

    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            # TODO, is there a need for activation between two convolutions??
            #  Detectron2 seems to have it in its separable convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
