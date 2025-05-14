import warnings
import numpy as np
from torch import nn
from tairvision_object_detection.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, PathAggregationNetwork, \
    RepeatedBidirectionalFPN
from tairvision_object_detection.ops.deformable_encoder_neck import DeformableEncoderNeck
from tairvision_object_detection.ops import misc as misc_nn_ops
from .._utils import IntermediateLayerGetter
from .. import mobilenet
from .. import resnet, regnet, convnext, swin_transformer, efficientnet

from typing import Optional

from ..swin_transformer import Swin_B_Weights, WeightsEnum, Swin_S_Weights, Swin_T_Weights
try:
    import timm
except ImportError:
    warnings.warn("timm is not installed. Please install it for future utilization.")


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, backbone, return_layers, in_channels_list, out_channels, extra_blocks=None,
                 pyramid_type="fpn", depthwise=False, repeats=3, fusion_type='fastnormed', norm_layer=None, **kwargs):
        super(BackboneWithFPN, self).__init__()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.pyramid_type = pyramid_type
        if pyramid_type is None:
            self.num_channels = in_channels_list
        else:
            self.num_channels = len(in_channels_list) * [out_channels]
        self.strides = kwargs.get('scale_factor_list', [4, 8, 16, 32])
        self.channel_end = False
        self.return_layers = return_layers

        skip_fpn = kwargs.pop('skip_fpn', None)
        in_channels_list = remove_skipped_layers(in_channels_list, skip_fpn)

        self.in_channels_list = in_channels_list
        self.skip_fpn = skip_fpn

        if isinstance(backbone, swin_transformer.SwinTransformer):
            self.channel_end = True

        if pyramid_type == "fpn":
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=extra_blocks,
            )
        elif pyramid_type == "panet":
            self.fpn = PathAggregationNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=extra_blocks,
            )
        elif pyramid_type == "bifpn":
            self.fpn = RepeatedBidirectionalFPN(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=extra_blocks,
                depthwise=depthwise,
                repeats=repeats,
                fusion_type=fusion_type,
            )
        elif pyramid_type == "den":
            self.fpn = DeformableEncoderNeck(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=extra_blocks,
                strides=self.strides,
            )
        elif pyramid_type is None:
            pass
        else:
            raise ValueError("Only FPN, PANET, BiFPN, and den(DeformableEncoderNeck) have been implemented.")

        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        if self.channel_end:
            for i, x_i in x.items():
                x[i] = x_i.permute(0, 3, 1, 2)
        if self.pyramid_type is not None:
            if self.skip_fpn is None:
                x = self.fpn(x)
            else:
                skip_fpn = self.skip_fpn
                keys = list(x.keys())

                # create a new dict for skipped features and pop them from x
                skipped_x = {}
                for i, key in enumerate(keys):
                    if skip_fpn[i]:
                        skipped_x[key] = x.pop(key)

                # pass x that are not skipped from fpn
                x = self.fpn(x)

                # fpn may add new features to x, add them to keys and add False to skip_fpn for each extra
                extra_keys = x.keys() - keys
                for _ in extra_keys:
                    skip_fpn.append(False)
                keys.extend(extra_keys)

                # create new dict combining both skipped features and fpn passed features
                x = {key: skipped_x[key] if skip_fpn[i] else x[key] for i, key in enumerate(keys)}
        return x


def remove_skipped_layers(in_channels_list, skip_fpn):
    skip_fpn = [False for _ in in_channels_list] if skip_fpn is None else skip_fpn
    # Handle extra levels
    num_extra_levels = len(in_channels_list) - len(skip_fpn)
    for _ in range(num_extra_levels):
        skip_fpn.append(False)

    in_channels_list_skipped = []
    for i in range(len(in_channels_list)):
        if not skip_fpn[i]:
            in_channels_list_skipped.append(in_channels_list[i])

    return in_channels_list_skipped


def timm_generic_fpn_backbone(
        backbone_name,
        pretrained,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
        trainable_layers=3,
        returned_layers=None,
        extra_blocks=None,
        extra_before=False,
        pyramid_type="fpn",
        depthwise=False,
        repeats=3,
        fusion_type='fastnormed',
        bifpn_norm_layer=None,
        **kwargs
):
    """
    Constructs a specified timm backbone with FPN on top. Freezes the specified number of layers in the backbone.

    Args:
        backbone_name (string): name of the model (without the prefix "timm-")
            pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        norm_layer (torch.nn.module): This is not suitable for all backbones, therefore ignored for generic structure
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values might change depending on the backbone.
        returned_layers (list[int]): The layers of the model that should be returned. Each entry corresponds to the
            order of the blocks in the backbone. Default: [1, 2, 3, 4]
        extra_blocks (ExtraFPNBlock): Adds extra conv layers on top of the original feature pyramid.
        extra_before (bool): If True, extra blocks are added before the original FPN blocks, otherwise they are added
            after.
        pyramid_type (string): Type of pyramid to build on top of the backbone. One of {None, "fpn", "panet", "bifpn"}
        depthwise (bool): If True, uses depthwise separable convolutions in BiFPN
        repeats (int): Number of times to repeat BiFPN
        fusion_type (string): Type of fusion to use in BiFPN. One of {"fastnormed", "fast", "sum"}
        bifpn_norm_layer (torch.nn.module): Normalization layer to use in BiFPN. If None, uses BatchNorm2d
        **kwargs: Other parameters that are passed to the BackboneWithFPN constructor

    """
    # create the specified backbone
    backbone_kwargs = kwargs.get('backbone_kwargs', {})
    backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                 features_only=True, **backbone_kwargs)
    # get the feature names and number of channels
    feature_names = backbone.feature_info.module_name()
    number_of_channels = backbone.feature_info.channels()

    # select layers that won't be frozen (trainable)
    assert 0 <= trainable_layers <= 5
    layers_to_train = feature_names[::-1][:trainable_layers]

    # freeze layers
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    # get the layers that should be returned
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]

    if len(feature_names) == 4:
        if len(returned_layers) == 4:
            returned_layers = [0, 1, 2, 3]
        elif len(returned_layers) == 3:
            returned_layers = [1, 2, 3]

    # create the return layers dictionary, in channels list and scale factor list
    return_layers = {f'{feature_names[k].replace(".", "_")}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_list = [number_of_channels[k] for k in returned_layers]
    scale_factor_list = backbone.feature_info.reduction()[-4:]  #TODO, only gets 4 last layers, might not be suitable for all backbones
    kwargs["scale_factor_list"] = scale_factor_list

    out_channels = kwargs.pop("out_channels", 256)

    if extra_blocks != None:
        extra_in_channels = in_channels_list[-1] if extra_before else out_channels

        extra_blocks = extra_blocks() if extra_blocks == LastLevelMaxPool else \
            extra_blocks(extra_in_channels, out_channels, before_pyramid=extra_before)

        in_channels_list.extend(extra_blocks.in_channels_list)

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks,
                           pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats, fusion_type=fusion_type,
                           norm_layer=bifpn_norm_layer, **kwargs)



def resnet_fpn_backbone(
        backbone_name,
        pretrained,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
        trainable_layers=3,
        returned_layers=None,
        extra_blocks=None,
        extra_before=False,
        pyramid_type="fpn",
        depthwise=False,
        repeats=3,
        fusion_type='fastnormed',
        bifpn_norm_layer=None,
        **kwargs
):
    """
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.

    Examples::

        >>> from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        >>> backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
        >>> # get some dummy image
        >>> x = torch.rand(1,3,64,64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('0', torch.Size([1, 256, 16, 16])),
        >>>    ('1', torch.Size([1, 256, 8, 8])),
        >>>    ('2', torch.Size([1, 256, 4, 4])),
        >>>    ('3', torch.Size([1, 256, 2, 2])),
        >>>    ('pool', torch.Size([1, 256, 1, 1]))]

    Args:
        backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        pretrained (bool): If True, returns a model with backbone pretrained on Imagenet
        norm_layer (torchvision.ops): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
        returned_layers (list of int): The layers of the network to return. Each entry must be in ``[1, 4]``.
            By default, all layers are returned.
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names. By default, a ``LastLevelMaxPool`` is used.
    """
    replace_stride_with_dilation = kwargs.get('replace_stride_with_dilation', [False, False, False])

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer,
        replace_stride_with_dilation=replace_stride_with_dilation)

    # select layers that will not be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]

    out_channels = kwargs.pop("out_channels", 256)

    scale_factor = 4
    scale_factor_list = [4]

    for dilation_check in replace_stride_with_dilation:
        if dilation_check is False:
            scale_factor *= 2
        scale_factor_list.append(scale_factor)

    kwargs["scale_factor_list"] = scale_factor_list

    if extra_blocks is not None:
        extra_in_channels = in_channels_list[-1] if extra_before else out_channels

        extra_blocks = extra_blocks() if extra_blocks == LastLevelMaxPool else \
            extra_blocks(extra_in_channels, out_channels, before_pyramid=extra_before)

        in_channels_list.extend(extra_blocks.in_channels_list)

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks,
                           pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats, fusion_type=fusion_type,
                           norm_layer=bifpn_norm_layer, **kwargs)


def swin_fpn_backbone(
        backbone_name="swin_b",
        trainable_layers=3,
        swin_weight: Optional[WeightsEnum] = Swin_B_Weights.DEFAULT,
        returned_layers=None,
        extra_blocks=None,
        extra_before=False,
        pyramid_type="fpn",
        depthwise=False,
        repeats=3,
        fusion_type='fastnormed',
        **kwargs
):
    """
    Constructs a specified Swin Transformers backbone with FPN on top.

    Args:
        backbone_name (string): swin architecture. Possible values are 'swin_t', 'swin_b', 'swin_s'
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
        returned_layers (list of int): The layers of the network to return. Each entry must be in ``[1, 4]``.
            By default, all layers are returned.
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names. By default, a ``LastLevelMaxPool`` is used.
    """
    if swin_weight is None:
        if backbone_name == "swin_b":
            swin_weight = Swin_B_Weights.DEFAULT
        elif backbone_name == "swin_t":
            swin_weight = Swin_T_Weights.DEFAULT
        elif backbone_name == "swin_s":
            swin_weight = Swin_S_Weights.DEFAULT
        else:
            raise ValueError("Not an available backbone")

    backbone = swin_transformer.__dict__[backbone_name](weights=swin_weight)

    backbone.stem = backbone.features[0]
    backbone.block1 = nn.Sequential(backbone.features[1])
    backbone.block2 = nn.Sequential(backbone.features[2], backbone.features[3])
    backbone.block3 = nn.Sequential(backbone.features[4], backbone.features[5])
    backbone.block4 = nn.Sequential(backbone.features[6], backbone.features[7], backbone.norm)
    del backbone.features
    del backbone.norm
    del backbone.avgpool
    del backbone.head

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 4
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5

    return_layers = {f'block{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = []
    if backbone_name == "swin_t" or backbone_name == "swin_s":
        c = 96
    elif backbone_name == "swin_b":
        c = 128
    else:
        print("WRONG MODEL NAME")
        return
    for r in returned_layers:
        in_channels_list.append(c * (2 ** (r-1)))

    out_channels = kwargs.pop("out_channels", 256)
    kwargs["scale_factor_list"] = [4, 8, 16, 32]

    if extra_blocks is not None:
        extra_in_channels = in_channels_list[-1] if extra_before else out_channels

        extra_blocks = extra_blocks() if extra_blocks == LastLevelMaxPool else \
            extra_blocks(extra_in_channels, out_channels, before_pyramid=extra_before)

        in_channels_list.extend(extra_blocks.in_channels_list)

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks,
                           pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats, fusion_type=fusion_type,
                           **kwargs)


def _validate_trainable_layers(pretrained, trainable_backbone_layers, max_value, default_value):
    # Do not freeze any layers if pretrained model or backbone is not used
    if not pretrained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has not effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                "falling back to trainable_backbone_layers={} so that all layers are trainable".format(max_value))
        trainable_backbone_layers = max_value

    # By default, freeze first blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    assert 0 <= trainable_backbone_layers <= max_value
    return trainable_backbone_layers


def mobilenet_backbone(
        backbone_name,
        pretrained,
        fpn,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
        trainable_layers=2,
        returned_layers=None,
        extra_blocks=None,
        pyramid_type="fpn",
        **kwargs
):
    backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained, norm_layer=norm_layer).features

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)

    # Find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    out_channels = kwargs.pop("out_channels", 256)
    if fpn:
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if returned_layers is None:
            returned_layers = [num_stages - 2, num_stages - 1]
        assert min(returned_layers) >= 0 and max(returned_layers) < num_stages
        return_layers = {f'{stage_indices[k]}': str(v) for v, k in enumerate(returned_layers)}

        in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
        return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks,
                               pyramid_type=pyramid_type)
    else:
        m = nn.Sequential(
            backbone,
            # Depthwise linear combination of channels to reduce their size
            nn.Conv2d(backbone[-1].out_channels, out_channels, 1),
        )
        m.out_channels = out_channels
        return m


def regnet_fpn_backbone(
        backbone_name,
        pretrained,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
        trainable_layers=3,
        returned_layers=None,
        extra_blocks=None,
        extra_before=False,
        pyramid_type="fpn",
        depthwise=False,
        repeats=3,
        fusion_type='fastnormed',
        bifpn_norm_layer=None,
        **kwargs
):
    backbone = regnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)

    # This part is required for IntermediateLaterGetter to work properly
    backbone.block1 = backbone.trunk_output.block1
    backbone.block2 = backbone.trunk_output.block2
    backbone.block3 = backbone.trunk_output.block3
    backbone.block4 = backbone.trunk_output.block4
    del backbone.trunk_output
    del backbone.avgpool
    del backbone.fc

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['block4', 'block3', 'block2', 'block1', 'stem'][:trainable_layers]

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'block{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = []
    for i, layer in enumerate(return_layers):
        stage = getattr(backbone, layer)
        block = getattr(stage, layer + '-0')
        out_channels = getattr(block.proj, 'out_channels')
        in_channels_list.append(out_channels)

    out_channels = kwargs.pop("out_channels", 256)
    kwargs["scale_factor_list"] = [4, 8, 16, 32]

    if extra_blocks is not None:
        extra_in_channels = in_channels_list[-1] if extra_before else out_channels

        extra_blocks = extra_blocks() if extra_blocks == LastLevelMaxPool else \
            extra_blocks(extra_in_channels, out_channels, before_pyramid=extra_before)

        in_channels_list.extend(extra_blocks.in_channels_list)

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks,
                           pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats, fusion_type=fusion_type,
                           norm_layer=bifpn_norm_layer, **kwargs)


def convnext_fpn_backbone(
        backbone_name,
        pretrained,
        trainable_layers=3,
        returned_layers=None,
        extra_blocks=None,
        extra_before=False,
        pyramid_type="fpn",
        depthwise=False,
        repeats=3,
        fusion_type='fastnormed',
        bifpn_norm_layer=None,
        **kwargs
):
    backbone = convnext.__dict__[backbone_name](pretrained=pretrained)

    # This part is required for IntermediateLaterGetter to work properly
    backbone.stem = backbone.features[0]
    backbone.block1 = backbone.features[1]
    backbone.block2 = nn.Sequential(backbone.features[2], backbone.features[3])
    backbone.block3 = nn.Sequential(backbone.features[4], backbone.features[5])
    backbone.block4 = nn.Sequential(backbone.features[6], backbone.features[7])
    del backbone.features
    del backbone.avgpool
    del backbone.classifier

    # Select layers that will not be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['block4', 'block3', 'block2', 'block1', 'stem'][:trainable_layers]

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'block{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = [backbone.stem.out_channels * 2 ** (i - 1) for i in returned_layers]

    out_channels = kwargs.pop("out_channels", 256)

    kwargs["scale_factor_list"] = [4, 8, 16, 32]

    extra_in_channels = in_channels_list[-1] if extra_before else out_channels

    extra_blocks = extra_blocks() if extra_blocks == LastLevelMaxPool else \
        extra_blocks(extra_in_channels, out_channels, before_pyramid=extra_before)

    in_channels_list.extend(extra_blocks.in_channels_list)

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks,
                           pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats, fusion_type=fusion_type,
                           norm_layer=bifpn_norm_layer, **kwargs)


def efficientnet_fpn_backbone(
        backbone_name="efficientnet-b4",
        pretrained=True,
        trainable_layers=5,
        returned_layers=None,
        extra_blocks=None,
        extra_before=False,
        pyramid_type="fpn",
        depthwise=False,
        repeats=3,
        fusion_type='fastnormed',
        bifpn_norm_layer=None,
        **kwargs
):
    """
    Constructs a specified EfficientNet backbone with FPN on top.
    Args:
        backbone_name (string): efficientnet architecture. Possible values are 'efficientnet_b4', 'efficientnet_b0', ..
        pretrained (bool): If True, returns a model pretrained on ImageNet to backbone
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
        returned_layers (list of int): The layers of the network to return. Each entry must be in ``[1, 4]``.
            By default, all layers are returned.
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names. By default, a ``LastLevelMaxPool`` is used.
    """
    backbone = efficientnet.__dict__[backbone_name](pretrained=pretrained)

    backbone.stem = backbone.features[0]
    backbone.block1 = nn.Sequential(backbone.features[1], backbone.features[2])
    backbone.block2 = backbone.features[3]
    backbone.block3 = nn.Sequential(backbone.features[4], backbone.features[5])
    backbone.block4 = nn.Sequential(backbone.features[6], backbone.features[7], backbone.features[8])
    del backbone.features
    del backbone.avgpool
    del backbone.classifier

    # select layers that will not be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['block4', 'block3', 'block2', 'block1', 'stem'][:trainable_layers]

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'block{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = []
    for i, layer in enumerate(return_layers):
        block = getattr(backbone, layer)
        if not type(block[-1]) is nn.modules.container.Sequential:
            out_channels = getattr(block[-1], 'out_channels')
        else:
            out_channels = getattr(block[-1][-1], 'out_channels')
        in_channels_list.append(out_channels)

    out_channels = kwargs.pop("out_channels", 256)
    kwargs["scale_factor_list"] = [4, 8, 16, 32]

    extra_in_channels = in_channels_list[-1] if extra_before else out_channels

    extra_blocks = extra_blocks() if extra_blocks == LastLevelMaxPool else \
        extra_blocks(extra_in_channels, out_channels, before_pyramid=extra_before)

    in_channels_list.extend(extra_blocks.in_channels_list)

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks,
                           pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats, fusion_type=fusion_type,
                           norm_layer=bifpn_norm_layer, **kwargs)
