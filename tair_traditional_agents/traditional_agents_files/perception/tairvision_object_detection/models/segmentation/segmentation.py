import torch.nn as nn
from .._utils import IntermediateLayerGetter
from tairvision_object_detection._internally_replaced_utils import load_state_dict_from_url
from .. import mobilenetv3
from .. import resnet,swin_transformer
from .. import hrnet
from .. import regnet, convnext
from .deeplabv3 import DeepLabHead, PanopticDeepLabHeadPlusGenericDepthWise, DeepLabHeadPlusGeneric, \
    DeepLabHeadPlusGenericDepthWise, DeepLabV3, DeepLabV3_mc ,PanopticDeepLabHeadPlusGeneric
from .deeplabv3 import DeepLabHeadX #TODO this is a temporary solution

from .lanefit import LaneFitSegmentationModel
try:
    from .temporal import TemporalEncoder, TemporalDeformableEncoder, TemporalDeformableTransformer
    temporal_activated = True
    from .maskformer import Mask2Former, Mask2FormerHead
except:
    print("temporal is not activated")
    temporal_activated = False

from .OCR import OCRNet, OCRHead, OCRASPPHead
from .fcn import FCN, FCNHead
from .lraspp import LRASPP
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, NamedTuple
from math import ceil

from .fpn import FPNSegmentationHead
from tairvision_object_detection.models.detection.backbone_utils import resnet_fpn_backbone, regnet_fpn_backbone, _validate_trainable_layers, \
    swin_fpn_backbone, convnext_fpn_backbone
from tairvision_object_detection.ops.feature_pyramid_network import LastLevelP6,LastLevelP6P7
from tairvision_object_detection.ops import feature_pyramid_network
from tairvision_object_detection.models.detection.backbone_utils import resnet_fpn_backbone, regnet_fpn_backbone, _validate_trainable_layers
from tairvision_object_detection.ops.feature_pyramid_network import LastLevelP6
import tairvision_object_detection.ops.misc as msc

__all__ = ["generic_model"]

model_urls = {
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
    'deeplabv3_mobilenet_v3_large_coco':
        'https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth',
    'lraspp_mobilenet_v3_large_coco': 'https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth',
}


def _segm_model(name: str,
                backbone_name: str,
                num_classes: Union[List[int], int],
                aux: Optional[bool],
                pretrained_backbone=True,
                size: Optional[Tuple[int]] = (304, 600),
                **kwargs):
    replace_stride_with_dilation = kwargs.get('replace_stride_with_dilation')

    if 'swin' not in backbone_name:
        norm = kwargs.get('norm', "BatchNorm2d")
        norm_layer = msc.__dict__[norm]

    if 'swin' in backbone_name:

        weight = kwargs.get('weight')
        # Default was [False, True, True]
        backbone = swin_transformer.__dict__[backbone_name](weights=weight)

        backbone.stem = backbone.features[0]
        backbone.block1 = nn.Sequential(backbone.features[1])
        backbone.block2 = nn.Sequential(backbone.features[2], backbone.features[3])
        backbone.block3 = nn.Sequential(backbone.features[4], backbone.features[5])
        backbone.block4 = nn.Sequential(backbone.features[6], backbone.features[7], backbone.norm)
        del backbone.features
        del backbone.norm
        del backbone.avgpool
        del backbone.head

        scale_factor = 4
        scale_factor_list = [4, 8, 16, 32] #TODO this was just a guess

        stage4 = 'block4'
        stage3 = 'block3'
        stage2 = 'block2'
        stage1 = 'block1'

        returned_layers = [1, 2, 3, 4]
        in_channels_list = []
        if backbone_name == "swin_t" or backbone_name == "swin_s":
            c = 96
        elif backbone_name == "swin_b":
            c = 128
        else:
            print("WRONG MODEL NAME")
            return
        for r in returned_layers:
            in_channels_list.append(c * (2 ** (r - 1)))

        if 'swin_t' in backbone_name or 'swin_s' in backbone_name:
            stage4_channels = 768
            stage3_channels = 384
            stage2_channels = 192
            stage1_channels = 96
        elif 'swin_b' in backbone_name:
            stage4_channels = 1024
            stage3_channels = 512
            stage2_channels = 256
            stage1_channels = 128

    elif 'resnet' in backbone_name:

        # Default was [False, True, True]
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer)

        scale_factor = 4
        scale_factor_list = [4]

        for dilation_check in replace_stride_with_dilation:
            if dilation_check is False:
                scale_factor *= 2
            scale_factor_list.append(scale_factor)

        stage4 = 'layer4'
        stage3 = 'layer3'
        stage2 = 'layer2'
        stage1 = 'layer1'

        if 'resnet101' in backbone_name or 'resnet50' in backbone_name:
            stage4_channels = 2048
            stage3_channels = 1024
            stage2_channels = 512
            stage1_channels = 256
        elif 'resnet18' in backbone_name:
            stage4_channels = 512
            stage3_channels = 256
            stage2_channels = 128
            stage1_channels = 64

    elif 'mobilenet_v3' in backbone_name:
        scale_factor = 16
        # TODO implement the scale8 version of this
        backbone = mobilenetv3.__dict__[backbone_name](pretrained=pretrained_backbone, dilated=True).features

        # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
        # The first and last blocks are always included because they are the C0 (conv1) and Cn.
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
        out_layer = str(out_pos)
        out_inplanes = backbone[out_pos].out_channels
        aux_pos = stage_indices[-3]  # changed to C3 which has output stride of 16,
        # it was C2 previously here which has output_stride = 8
        aux_layer = str(aux_pos)
        aux_inplanes = backbone[aux_pos].out_channels

        second_low_level_pos = stage_indices[-4]
        low_level_pos = stage_indices[-5]

        second_low_level = str(second_low_level_pos)
        low_level = str(low_level_pos)

        second_low_inplanes = backbone[second_low_level_pos].out_channels
        low_inplanes = backbone[low_level_pos].out_channels

        low_level_scale_factor = 4
        second_low_level_scale_factor = 8

    elif 'hrnet' in backbone_name:
        assert replace_stride_with_dilation == [False, False, False], "Stride is not supported for Hrnet backbones"

        backbone = hrnet.__dict__[backbone_name](
            pretrained=pretrained_backbone, size=size, **kwargs)

        if 'plus' in backbone_name:

            scale_factor_list = [4, 8, 16, 32]
            stage4_channels = 2048
            stage3_channels = 512
            stage2_channels = 256
            stage1_channels = 128
        else:

            if 'hrnet18' in backbone_name:
                number_of_planes = 270
            elif 'hrnet32' in backbone_name:
                number_of_planes = 480
            elif 'hrnet48' in backbone_name:
                number_of_planes = 720
            else:
                raise ValueError("Not a supported hrnet backbone...")
            stage4_channels = number_of_planes
            stage3_channels = number_of_planes
            stage2_channels = number_of_planes
            stage1_channels = number_of_planes
            scale_factor_list = [4, 4, 4, 4]

    elif 'regnet' in backbone_name:
        assert replace_stride_with_dilation == [False, False, False], "Stride is not supported for RegNet backbones"
        backbone = regnet.__dict__[backbone_name](
            pretrained=pretrained_backbone)

        backbone.block1 = backbone.trunk_output.block1
        backbone.block2 = backbone.trunk_output.block2
        backbone.block3 = backbone.trunk_output.block3
        backbone.block4 = backbone.trunk_output.block4
        del backbone.trunk_output
        del backbone.avgpool
        del backbone.fc

        stage4 = 'block4'
        stage3 = 'block3'
        stage2 = 'block2'
        stage1 = 'block1'

        return_layers = [stage1, stage2, stage3, stage4]
        in_channels_list = []
        for i in return_layers:
            stage = getattr(backbone, i)
            block = getattr(stage, i + '-0')
            out_channels = getattr(block.proj, 'out_channels')
            in_channels_list.append(out_channels)

        stage4_channels = in_channels_list[3]
        stage3_channels = in_channels_list[2]
        stage2_channels = in_channels_list[1]
        stage1_channels = in_channels_list[0]

        scale_factor_list = [4, 8, 16, 32]

    elif 'convnext' in backbone_name:
        assert replace_stride_with_dilation == [False, False, False], "Stride is not supported for convnext backbones"

        backbone = convnext.__dict__[backbone_name](
            pretrained=pretrained_backbone)

        backbone.stem = backbone.features[0]
        backbone.block1 = backbone.features[1]
        backbone.block2 = nn.Sequential(backbone.features[2], backbone.features[3])
        backbone.block3 = nn.Sequential(backbone.features[4], backbone.features[5])
        backbone.block4 = nn.Sequential(backbone.features[6], backbone.features[7])
        del backbone.features
        del backbone.avgpool
        del backbone.classifier

        stage4 = 'block4'
        stage3 = 'block3'
        stage2 = 'block2'
        stage1 = 'block1'
        returned_layers = [1, 2, 3, 4]
        in_channels_list = [backbone.stem.out_channels * 2 ** (i - 1) for i in returned_layers]

        stage4_channels = in_channels_list[3]
        stage3_channels = in_channels_list[2]
        stage2_channels = in_channels_list[1]
        stage1_channels = in_channels_list[0]

        scale_factor_list = [4, 8, 16, 32]

    else:
        raise NotImplementedError('backbone {} is not supported as of now'.format(backbone_name))

    channel_end = False
    if isinstance(backbone,swin_transformer.SwinTransformer):
        channel_end = True

    kwargs["channel_end"] = channel_end

    if not "hrnet" in backbone_name:
        return_layers = {stage4: 'out', stage3: 'stage3', stage2: 'stage2', stage1: 'stage1'}
        if aux:
            # TODO aux overrides the stage3 problem, therefore assertation can be meaningful for that condition
            return_layers[stage3] = 'aux'
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    number_of_channel_list = [stage1_channels, stage2_channels, stage3_channels]
    out_inplanes = kwargs.get('aspp_number_of_input_channels', stage4_channels)
    aux_inplanes = stage3_channels

    size_list = []
    for scale_factor in scale_factor_list:
        size_scaled = ceil(size[0] / scale_factor), ceil(size[1] / scale_factor)
        size_list.append(size_scaled)

    if scale_factor == 32:
        atrous_rates = [3, 6, 9]
    elif scale_factor == 16:
        atrous_rates = [6, 12, 18]
    elif scale_factor == 8:
        atrous_rates = [12, 24, 36]
    elif scale_factor == 4:
        atrous_rates = [24, 48, 72]
    else:
        assert "Not implemented for the time being"

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'deeplabv3_plus': (DeepLabHeadPlusGeneric, DeepLabV3),
        'deeplabv3_plus_depthwise': (DeepLabHeadPlusGenericDepthWise, DeepLabV3),
        'deeplabv3_mc': (DeepLabHead, DeepLabV3_mc),
        'panoptic_deeplabv3_plus_depthwise': (PanopticDeepLabHeadPlusGenericDepthWise, DeepLabV3),
        'panoptic_deeplabv3_plus': (PanopticDeepLabHeadPlusGeneric, DeepLabV3),
        'ocrnet': (OCRHead, OCRNet),
        'ocrnet_aspp': (OCRASPPHead, OCRNet),
        'fcn': (FCNHead, FCN),
    }

    if temporal_activated:
        model_map.update({
            'deeplabv3_temporal': (DeepLabHead, TemporalEncoder),
            'deeplabv3_temporal_deformable_encoder': (DeepLabHead, TemporalDeformableEncoder),
            'deeplabv3_temporal_deformable_transformer': (DeepLabHead, TemporalDeformableTransformer),
            'fcn_temporal_deformable_encoder': (FCNHead, TemporalDeformableEncoder),
            'fcn_temporal_deformable_transformer': (FCNHead, TemporalDeformableTransformer),
            'mask2Former': (Mask2FormerHead, Mask2Former),
        })

    if not isinstance(num_classes, List):
        num_classes = [num_classes]

    classifier_list = []
    for num_class in num_classes:
        classifier: Union[DeepLabHeadPlusGeneric, DeepLabV3, OCRHead, OCRASPPHead] = \
            model_map[name][0](in_channels=out_inplanes,
                               num_classes=num_class,
                               size=size_list[-1],
                               atrous_rates=atrous_rates,
                               number_of_channel_levels=number_of_channel_list,
                               level_sizes=size_list[:-1],
                               scale_factor_list=scale_factor_list,
                               **kwargs
                               )
        classifier_list.append(classifier)
    base_model = model_map[name][1]

    aux_classifier_list = None
    if aux:
        aux_classifier_list = []
        for num_class in num_classes:
            if "ocrnet" in name:
                aux_classifier = FCNHead(out_inplanes, num_class, output_key="out")
            else:
                aux_classifier = FCNHead(aux_inplanes, num_class, output_key="aux")

            aux_classifier_list.append(aux_classifier)

    model = base_model(backbone, classifier_list, aux_classifier_list, size, **kwargs)

    return model


def _load_model(
        arch_type: str,
        backbone: str,
        pretrained: bool,
        progress: bool,
        num_classes: list,
        aux_loss: Optional[bool],
        **kwargs: Any
) -> nn.Module:
    if pretrained:
        aux_loss = True
        kwargs["pretrained_backbone"] = False

    kwargs["train_generic"] = kwargs.get('train_generic', False)

    if kwargs["train_generic"] :
        print("--- generic version ---")
        model = _segm_model_generic(arch_type, backbone, num_classes, aux_loss, **kwargs)
    else:
        print("--- old version ---")
        if 'fpn' in arch_type:
            model = _segm_model_fpn(arch_type, backbone, num_classes, aux_loss, **kwargs)
        else:
            model = _segm_model(arch_type, backbone, num_classes, aux_loss, **kwargs)
        if pretrained:
            _load_weights(model, arch_type, backbone, progress)
    return model


def _load_weights(model: nn.Module, arch_type: str, backbone: str, progress: bool) -> None:
    arch = arch_type + '_' + backbone + '_coco'
    model_url = model_urls.get(arch, None)
    if model_url is None:
        raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
    else:
        state_dict = load_state_dict_from_url(model_url, progress=progress)
        model.load_state_dict(state_dict)


def generic_model(
        head: str,
        backbone: str,
        pretrained: bool = False,
        progress: bool = True,
        num_classes: Union[List[int], int] = 21,
        aux_loss: Optional[bool] = None,
        **kwargs: Any
) -> nn.Module:
    if pretrained:
        raise ValueError("coco pretrained ??")

    return _load_model(head, backbone, pretrained, progress, num_classes, aux_loss, **kwargs)


def _segm_model_fpn(name: str,
                    backbone_name: str,
                    num_classes: Union[List[int], int],
                    aux: Optional[bool],
                    pretrained_backbone=True,
                    size: Optional[Tuple[int]] = (304, 600),
                    **kwargs):

    trainable_backbone_layers = None
    trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

    if 'resnet' in backbone_name:
        backbone_model = resnet_fpn_backbone
    elif 'regnet' in backbone_name:
        backbone_model = regnet_fpn_backbone
    else:
        raise NotImplementedError('backbone {} is not supported with fpn'.format(backbone_name))

    backbone = backbone_model(backbone_name=backbone_name,
                              pretrained=True,
                              returned_layers=[1, 2, 3, 4],
                              extra_blocks=LastLevelP6,
                              trainable_layers=trainable_backbone_layers,
                              extra_before=False,
                              pyramid_type="bifpn",
                              depthwise=False, repeats=3,
                              fusion_type='fastnormed',
                              bifpn_norm_layer=None)

    out_channels = 256
    number_of_channel_list = [128 for _ in range(5)]
    out_inplanes = out_channels
    aux_inplanes = out_channels

    if not isinstance(num_classes, List):
        num_classes = [num_classes]

    model_map = {
        'fpnsegm': (FPNSegmentationHead, DeepLabV3),
    }

    classifier_list = []
    for num_class in num_classes:
        classifier = model_map[name][0](in_channels=out_inplanes, num_classes=num_class,
                                        number_of_channel_list=number_of_channel_list)
        classifier_list.append(classifier)

    aux_classifier_list = None

    model = model_map[name][1](backbone, classifier_list, aux_classifier_list, size, **kwargs)

    return model


def _segm_model_generic(name: str,
                    backbone_name: str,
                    num_classes: Union[List[int], int],
                    aux: Optional[bool],
                    pretrained_backbone=True,
                    size: Optional[Tuple[int]] = (304, 600),
                    **kwargs):

    trainable_backbone_layers = None
    trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

    if 'resnet' in backbone_name:
        backbone_model = resnet_fpn_backbone
    elif 'regnet' in backbone_name:
        backbone_model = regnet_fpn_backbone
    elif 'swin' in backbone_name:
        backbone_model = swin_fpn_backbone
    elif 'mobilenetv3' in backbone_name:
        pass
        #TODO check mobilenet_backbone
    elif 'hrnet' in backbone_name:
        pass
        #TODO create hrnet_backbone
    elif  'convnext' in backbone_name:
        backbone_model = convnext_fpn_backbone
    else:
        raise NotImplementedError('backbone {} is not supported with fpn'.format(backbone_name))

    if 'swin' not in backbone_name:
        norm = kwargs.get('norm', "BatchNorm2d")
        kwargs["norm_layer"] = msc.__dict__[norm]

    extra_blocks  =  kwargs.get('extra_blocks', None)
    kwargs["pyramid_type"]  = kwargs.get('pyramid_type', None)

    num_level = 4
    if extra_blocks=="LastLevelP6P7" :
        kwargs["extra_blocks"] = LastLevelP6P7
        num_level = 6
    elif extra_blocks=="LastLevelP6" :
        kwargs["extra_blocks"] = LastLevelP6
        num_level = 5

    backbone = backbone_model(backbone_name=backbone_name,
                              pretrained=True,
                              returned_layers=[1, 2, 3, 4],
                              trainable_layers=trainable_backbone_layers,
                              **kwargs)

    return_layers_map = {'0': 'stage1', '1': 'stage2', '2': 'stage3', '3': 'out'}

    if aux:
        return_layers_map['2'] = 'aux'

    for key, value in backbone.body.return_layers.items():
        for key_map, value_map in return_layers_map.items():
            if value == key_map:
                backbone.body.return_layers[key] = value_map

    in_channel_list = backbone.in_channels_list

    if kwargs["pyramid_type"]:
        out_inplanes = backbone.out_channels
        aux_inplanes = backbone.out_channels
    else:
        stage4_channels   = backbone.in_channels_list[3]
        stage3_channels   = backbone.in_channels_list[2]
        # TODO: bunu backbone utils in icine attt
        """"
        if not "hrnet" in backbone_name:
            backbone.body.return_layers = {stage4: 'out', stage3: 'stage3', stage2: 'stage2', stage1: 'stage1'}
            if aux:
                # TODO aux overrides the stage3 problem, therefore assertation can be meaningful for that condition
                return_layers[stage3] = 'aux'"""

        out_inplanes = kwargs.get('aspp_number_of_input_channels', stage4_channels)
        aux_inplanes = stage3_channels

    number_of_channel_list   = [128 for _ in range(num_level)]
    number_of_channel_levels = in_channel_list[:4]
    scale_factor_list = backbone.strides

    size_list = []
    for scale_factor in scale_factor_list:
        size_scaled = ceil(size[0] / scale_factor), ceil(size[1] / scale_factor)
        size_list.append(size_scaled)

    if scale_factor == 32:
        atrous_rates = [3, 6, 9]
    elif scale_factor == 16:
        atrous_rates = [6, 12, 18]
    elif scale_factor == 8:
        atrous_rates = [12, 24, 36]
    elif scale_factor == 4:
        atrous_rates = [24, 48, 72]
    else:
        assert "Not implemented for the time being"

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'deeplabv3x': (DeepLabHeadX, DeepLabV3),#TODO this is a temporary solution
        'deeplabv3_plus': (DeepLabHeadPlusGeneric, DeepLabV3),
        'deeplabv3_plus_depthwise': (DeepLabHeadPlusGenericDepthWise, DeepLabV3),
        'panoptic_deeplabv3_plus_depthwise': (PanopticDeepLabHeadPlusGenericDepthWise, DeepLabV3),
        'panoptic_deeplabv3_plus': (PanopticDeepLabHeadPlusGeneric, DeepLabV3),
        'ocrnet': (OCRHead, OCRNet),
        'ocrnet_aspp': (OCRASPPHead, OCRNet),
        'fcn': (FCNHead, FCN),
        'deeplabv3_lane_fit': (DeepLabHead, LaneFitSegmentationModel)
    }

    if not isinstance(num_classes, List):
        num_classes = [num_classes]

    kwargs['size'] = size_list[-1]
    kwargs['atrous_rates'] = atrous_rates
    kwargs['number_of_channel_levels'] = number_of_channel_levels
    kwargs['level_sizes'] = size_list[:-1]
    kwargs['scale_factor_list'] = scale_factor_list
    kwargs['number_of_channel_list'] = number_of_channel_list

    classifier_list = []
    for num_class in num_classes:
        classifier: Union[DeepLabHeadPlusGeneric, DeepLabV3, OCRHead, OCRASPPHead] = \
            model_map[name][0](in_channels=out_inplanes,
                               num_classes=num_class,
                               **kwargs
                               )
        classifier_list.append(classifier)
    base_model = model_map[name][1]

    aux_classifier_list = None
    if aux:
        aux_classifier_list = []
        for num_class in num_classes:
            if "ocrnet" in name:
                aux_classifier = FCNHead(out_inplanes, num_class, output_key="out")
            else:
                aux_classifier = FCNHead(aux_inplanes, num_class, output_key="aux")

            aux_classifier_list.append(aux_classifier)

    kwargs["num_classes"] = num_class
    model = base_model(backbone, classifier_list, aux_classifier_list, **kwargs)

    return model
