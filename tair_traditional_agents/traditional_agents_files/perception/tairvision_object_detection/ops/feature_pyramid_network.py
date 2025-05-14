from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class ExtraFPNBlock(nn.Module):
    """
    Base class for the extra block in the FPN.

    Args:
        results (List[Tensor]): the result of the FPN
        x (List[Tensor]): the original feature maps
        names (List[str]): the names for each one of the
            original feature maps

    Returns:
        results (List[Tensor]): the extended set of results
            of the FPN
        names (List[str]): the extended set of names for the results
    """

    def forward(
            self,
            results: List[Tensor],
            x: List[Tensor],
            names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        pass


class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.

    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names

    Examples::

        >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]

    """

    def __init__(
            self,
            in_channels_list: List[int],
            out_channels: int,
            extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super(FeaturePyramidNetwork, self).__init__()

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks
        if self.extra_blocks is not None and not self.extra_blocks.before_pyramid:
            in_channels_list = in_channels_list[:-self.extra_blocks.num_levels]

        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        if self.extra_blocks is not None and self.extra_blocks.before_pyramid:
            x, names = self.extra_blocks(x, x, names)

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None and not self.extra_blocks.before_pyramid:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def __init__(self):
        super(LastLevelMaxPool, self).__init__()
        self.in_channels_list = []
        self.before_pyramid = False
        self.num_levels = None

    def forward(
            self,
            x: List[Tensor],
            y: List[Tensor],
            names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))
        return x, names


class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels: int, out_channels: int, use_P5=True, before_pyramid=False):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = False if before_pyramid else use_P5
        self.in_channels_list = [out_channels, out_channels]
        self.before_pyramid = before_pyramid
        self.num_levels = 2

    def forward(
            self,
            p: List[Tensor],
            c: List[Tensor],
            names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        p5, c5 = p[-1], c[-1]
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        p.extend([p6, p7])
        names.extend(["p6", "p7"])
        return p, names


class LastLevelP6(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layer P6 when using P2.
    """

    def __init__(self, in_channels: int, out_channels: int, use_P5=True, before_pyramid=False):
        super(LastLevelP6, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        for module in [self.p6]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = False if before_pyramid else use_P5
        self.in_channels_list = [out_channels]
        self.before_pyramid = before_pyramid
        self.num_levels = 1

    def forward(
            self,
            p: List[Tensor],
            c: List[Tensor],
            names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        p5, c5 = p[-1], c[-1]
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p.extend([p6])
        names.extend(["p6"])
        return p, names


class PathAggregationNetwork(nn.Module):
    """
    Module that adds a PA-NET from on top of a set of feature maps. This is based on
    `"Path Aggregation Network for Instance Segmentation <https://arxiv.org/abs/1803.01534v4>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the PA-NET will be added.

    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names

    Examples::

        >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]

    """

    def __init__(
            self,
            in_channels_list: List[int],
            out_channels: int,
            extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super(PathAggregationNetwork, self).__init__()

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks
        if self.extra_blocks is not None and not self.extra_blocks.before_pyramid:
            in_channels_list = in_channels_list[:-self.extra_blocks.num_levels]

        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.pan_in_blocks = nn.ModuleList()
        self.pan_out_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            pan_in_block_module = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
                                                nn.ReLU(inplace=True))
            pan_out_block_module = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                                 nn.ReLU(inplace=True))
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
            self.pan_in_blocks.append(pan_in_block_module)
            self.pan_out_blocks.append(pan_out_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_pan_in_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.pan_in_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.pan_in_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.pan_in_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_pan_out_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.pan_out_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.pan_out_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.pan_out_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        if self.extra_blocks is not None and self.extra_blocks.before_pyramid:
            x, names = self.extra_blocks(x, x, names)

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        last_inner = results[0]
        results_pan = []
        results_pan.append(last_inner)
        for idx in range(1, len(results)):
            inner_bottom_up = self.get_result_from_pan_in_blocks(last_inner, idx - 1)
            inner_lateral = results[idx]
            last_inner = inner_bottom_up + inner_lateral
            last_inner = self.get_result_from_pan_out_blocks(last_inner, idx - 1)
            results_pan.append(last_inner)

        if self.extra_blocks is not None and not self.extra_blocks.before_pyramid:
            results_pan, names = self.extra_blocks(results_pan, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results_pan)])

        return out


class BidirectionalFPN(nn.Module):
    def __init__(
            self,
            in_channels_list: List[int],
            out_channels: int,
            extra_blocks: Optional[ExtraFPNBlock] = None,
            depthwise: Optional[Any] = False,
            fusion_type: Optional[str] = 'fastnormed',
            norm_layer: Optional[Any] = None
    ):
        super(BidirectionalFPN, self).__init__()

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks
        if self.extra_blocks is not None and not self.extra_blocks.before_pyramid:
            in_channels_list = in_channels_list[:-self.extra_blocks.num_levels]

        self.lateral_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()
        self.td_conv_blocks = nn.ModuleList()
        self.bu_conv_blocks = nn.ModuleList()
        self.feature_fusion = FeatureFusion(fusion_type, num_feat_layers=len(in_channels_list))
        for i_channel, in_channels in enumerate(in_channels_list):
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            lateral_block_module = nn.Conv2d(in_channels, out_channels, 1)
            skip_block_module = nn.Conv2d(in_channels, out_channels, 1)
            td_conv_module = conv3x3(out_channels, out_channels, depthwise=depthwise, norm_layer=norm_layer)
            bu_conv_module = conv3x3(out_channels, out_channels, depthwise=depthwise, norm_layer=norm_layer)

            self.lateral_blocks.append(lateral_block_module)
            if 0 < i_channel < len(in_channels_list) - 1:
                self.skip_blocks.append(skip_block_module)

            if i_channel < len(in_channels_list) - 1:
                self.td_conv_blocks.append(td_conv_module)
                self.bu_conv_blocks.append(bu_conv_module)
        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def get_result_from_skip_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.skip_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.skip_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_lateral_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.lateral_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.lateral_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_td_conv_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.pan_in_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.td_conv_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.td_conv_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_bu_conv_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.pan_out_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.bu_conv_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.bu_conv_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        if self.extra_blocks is not None and self.extra_blocks.before_pyramid:
            x, names = self.extra_blocks(x, x, names)

        last_top_down = self.get_result_from_lateral_blocks(x[-1], -1)
        results = [last_top_down]
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_lateral_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_top_down, size=feat_shape,
                                           mode="nearest")  # interpolation for upscaling
            last_top_down = self.feature_fusion(idx, [0, 1, None], inner_lateral, inner_top_down)
            last_top_down = self.get_result_from_td_conv_blocks(last_top_down, idx)  # 3x3 conv
            results.insert(0, last_top_down)  # P_td

        last_bottom_up = results[0]
        results_bifpn = [last_bottom_up]
        for idx in range(1, len(results)):
            inner_lateral = results[idx]
            feat_shape = inner_lateral.shape[-2:]
            inner_bottom_up = F.interpolate(last_bottom_up, size=feat_shape,
                                            mode="nearest")  # interpolation for downscaling
            if idx == len(results) - 1:
                last_bottom_up = self.feature_fusion(idx, [2, 3, None], inner_lateral, inner_bottom_up)
            else:
                inner_skip = self.get_result_from_skip_blocks(x[idx], idx - 1)  # 1x1 conv for skip connections
                last_bottom_up = self.feature_fusion(idx, [2, 3, 4], inner_lateral, inner_bottom_up, inner_skip)
            last_bottom_up = self.get_result_from_bu_conv_blocks(last_bottom_up, idx - 1)  # 3x3 conv
            results_bifpn.append(last_bottom_up)

        if self.extra_blocks is not None and not self.extra_blocks.before_pyramid:
            results_bifpn, names = self.extra_blocks(results_bifpn, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results_bifpn)])

        return out


class RepeatedBidirectionalFPN(nn.Module):
    def __init__(
            self,
            in_channels_list: List[int],
            out_channels: int,
            extra_blocks: Optional[ExtraFPNBlock] = None,
            depthwise: Optional[Any] = False,
            fusion_type: Optional[str] = 'fastnormed',
            repeats: Optional[int] = 1,
            norm_layer: Optional[Any] = None
    ):
        super(RepeatedBidirectionalFPN, self).__init__()

        self.bifpn_modules = nn.ModuleList()
        self.bifpn_modules.append(BidirectionalFPN(in_channels_list, out_channels, extra_blocks, depthwise,
                                                   fusion_type, norm_layer))

        in_channels_list_for_repeats = [out_channels] * len(in_channels_list)
        for _ in range(1, repeats):
            # extra_blocks_for_repeats = extra_blocks.__class__(out_channels, out_channels)
            self.bifpn_modules.append(BidirectionalFPN(in_channels_list_for_repeats, out_channels, None, depthwise,
                                                       fusion_type, norm_layer))

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:

        for module in self.bifpn_modules:
            x = module(x)

        return x


class FeatureFusion(nn.Module):
    def __init__(self,
                 fusion_type: str = 'fastnormed',
                 num_feat_layers: int = 5,
                 eps: Optional[float] = 1e-4
                 ):
        super(FeatureFusion, self).__init__()

        self.fusion_type = fusion_type
        self.fusion_weights = nn.Parameter(torch.rand((num_feat_layers, 5)))
        self.eps = eps

    def forward(self, l_idx, w_idx, in1, in2, in3=None):

        w1 = self.fusion_weights[l_idx, w_idx[0]]
        w2 = self.fusion_weights[l_idx, w_idx[1]]
        w3 = None if w_idx[2] is None else self.fusion_weights[l_idx, w_idx[2]]

        if self.fusion_type == 'softmax':
            w1 = torch.exp(w1)
            w2 = torch.exp(w2)
            denum = w1 + w2
            if w3 is not None:
                w3 = torch.exp(w3)
                denum += w3
        elif self.fusion_type == 'fastnormed':
            w1 = nn.functional.relu(w1)
            w2 = nn.functional.relu(w2)
            denum = self.eps + w1 + w2
            if w3 is not None:
                w3 = nn.functional.relu(w3)
                denum += w3
        elif self.fusion_type == 'unbounded':
            denum = 1.
        elif self.fusion_type == 'nofusion':
            w1, w2, w3 = 1., 1., 1.
            denum = 1.
        else:
            raise NotImplementedError

        out = in1 * w1 + in2 * w2
        if in3 is not None:
            out += in3 * w3

        return out / denum


def conv3x3(in_channels, out_channels, depthwise=False, norm_layer=None) -> nn.Module:
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    if depthwise:
        conv_module = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                                    nn.Conv2d(in_channels, out_channels, 1),
                                    norm_layer(out_channels),
                                    nn.ReLU(inplace=True))
    else:
        conv_module = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                    norm_layer(out_channels),
                                    nn.ReLU(inplace=True))
    return conv_module
