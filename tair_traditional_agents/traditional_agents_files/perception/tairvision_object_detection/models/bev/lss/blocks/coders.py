import torch.nn as nn

from tairvision.models.bev.lss.layers.convolutions import UpsamplingAdd, UpsamplingConcat
from tairvision.models.detection.backbone_utils import (regnet_fpn_backbone, resnet_fpn_backbone,
                                                        efficientnet_fpn_backbone)
from tairvision.ops.feature_pyramid_network import LastLevelP6P7, LastLevelP6


class EncoderReXnetFpn(nn.Module):
    def __init__(self, cfg, D):
        super().__init__()
        self.D = D
        self.C = cfg.OUT_CHANNELS
        self.use_depth_distribution = cfg.USE_DEPTH_DISTRIBUTION
        self.downsample = cfg.DOWNSAMPLE
        self.backbone_type = cfg.BACKBONE.TYPE
        self.backbone_version = cfg.BACKBONE.VERSION
        self.backbone_layers = cfg.BACKBONE.LAYERS
        self.backbone_layers_to_bev = cfg.BACKBONE_LAYERS_TO_BEV

        if self.backbone_type == 'regnet':
            backbone_fn = regnet_fpn_backbone
        elif self.backbone_type == 'resnet':
            backbone_fn = resnet_fpn_backbone
            if isinstance(self.backbone_version, str):
                if self.backbone_version[0] == "_":
                    self.backbone_version = self.backbone_version[1:]
        elif self.backbone_type == 'efficientnet':
            backbone_fn = efficientnet_fpn_backbone
        else:
            raise ValueError('Only regnet, resnet and efficientnet options are available.')

        backbone_pyramid_channels = cfg.BACKBONE.PYRAMID_CHANNELS
        pyramid_type = cfg.BACKBONE.PYRAMID_TYPE

        returned_layers = [int(layer) for layer in self.backbone_layers if 'p' not in layer]
        returned_layers = [layer + 5 - len(returned_layers) for layer in returned_layers]
        extra_layers = [layer for layer in self.backbone_layers if 'p' in layer]

        if extra_layers == ['p6', 'p7']:
            extra_blocks = LastLevelP6P7
        elif extra_layers == ['p6']:
            extra_blocks = LastLevelP6
        elif extra_layers == []:
            extra_blocks = None
        else:
            raise ValueError('Only LastLevelP6P7, LastLevelP6 or no extra block options are available.')

        backbone_name = self.backbone_type + str(self.backbone_version)
        self.backbone = backbone_fn(backbone_name, pretrained=True, returned_layers=returned_layers,
                                    extra_blocks=extra_blocks, trainable_layers=4,
                                    extra_before=False,
                                    pyramid_type=pyramid_type, depthwise=False, repeats=3,
                                    fusion_type='fastnormed', bifpn_norm_layer=None,
                                    out_channels=backbone_pyramid_channels)

        upsampling_out_channels = backbone_pyramid_channels * 2

        self.upsampling_layers = nn.ModuleList()
        self.upsampling_layers.append(UpsamplingConcat(backbone_pyramid_channels + backbone_pyramid_channels,
                                                       upsampling_out_channels, scale_factor=None))
        for _ in range(len(self.backbone_layers_to_bev) - 2):
            self.upsampling_layers.append(UpsamplingConcat(upsampling_out_channels + backbone_pyramid_channels,
                                                           upsampling_out_channels, scale_factor=None))

        if self.use_depth_distribution:
            self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C + self.D, kernel_size=1, padding=0)
        else:
            self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def upscale_and_concat_features(self, features):
        layer_names = self.backbone_layers_to_bev[::-1]

        x = self.upsampling_layers[0](features[layer_names[0]], features[layer_names[1]])
        for i_layer in range(1, len(self.upsampling_layers)):
            x = self.upsampling_layers[i_layer](x, features[layer_names[i_layer + 1]])
        return x

    def forward(self, x):
        feats_2d = self.backbone(x)                         # get feature dict of tensor
        x = self.upscale_and_concat_features(feats_2d)      # get feature vector

        x = self.depth_layer(x)                             # feature and depth head

        if self.use_depth_distribution:
            depth = x[:, : self.D].softmax(dim=1)
            x = depth.unsqueeze(1) * x[:, self.D: (self.D + self.C)].unsqueeze(2)  # outer product depth and features
        else:
            x = x.unsqueeze(2).repeat(1, 1, self.D, 1, 1)

        return x, feats_2d


class DecoderReXnetFpn(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()

        self.backbone_type = cfg.BACKBONE.TYPE
        self.backbone_version = cfg.BACKBONE.VERSION
        self.backbone_layers = cfg.BACKBONE.LAYERS

        if self.backbone_type == 'regnet':
            backbone_fn = regnet_fpn_backbone
        elif self.backbone_type == 'resnet':
            backbone_fn = resnet_fpn_backbone
        else:
            raise ValueError('Only regnet, resnet and efficientnet options are available.')

        backbone_pyramid_channels = cfg.BACKBONE.PYRAMID_CHANNELS
        pyramid_type = cfg.BACKBONE.PYRAMID_TYPE

        returned_layers = [int(layer) for layer in self.backbone_layers if 'p' not in layer]
        returned_layers = [layer + 5 - len(returned_layers) for layer in returned_layers]
        extra_layers = [layer for layer in self.backbone_layers if 'p' in layer]

        if extra_layers == ['p6', 'p7']:
            extra_blocks = LastLevelP6P7
        elif extra_layers == ['p6']:
            extra_blocks = LastLevelP6
        elif extra_layers == []:
            extra_blocks = None
        else:
            raise ValueError('Only LastLevelP6P7, LastLevelP6 or no extra block options are available.')

        backbone_name = self.backbone_type + str(self.backbone_version)
        self.backbone = backbone_fn(backbone_name, pretrained=False, norm_layer=None, returned_layers=returned_layers,
                                    extra_blocks=extra_blocks, trainable_layers=5,
                                    extra_before=False,
                                    pyramid_type=pyramid_type, depthwise=False, repeats=3,
                                    fusion_type='fastnormed', bifpn_norm_layer=None,
                                    out_channels=backbone_pyramid_channels)

        if self.backbone_type == 'regnet':
            stem_out_channels = self.backbone.body.stem[0].out_channels
            self.backbone.body.stem[0] = nn.Conv2d(in_channels, stem_out_channels, kernel_size=3, stride=1, padding=1,
                                                   bias=False)
            self.backbone.body.stem[1] = nn.BatchNorm2d(stem_out_channels)
            self.backbone.body.stem[2] = nn.ReLU(inplace=True)
        elif self.backbone_type == 'resnet':
            stem_out_channels = self.backbone.body.conv1.out_channels
            self.backbone.body.conv1 = nn.Conv2d(in_channels, stem_out_channels, kernel_size=7, stride=2, padding=3,
                                                 bias=False)
            self.backbone.body.bn1 = nn.BatchNorm2d(stem_out_channels)
            self.backbone.body.relu = nn.ReLU(inplace=True)
            self.backbone.body.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        else:
            raise ValueError('Only regnet, resnet and efficientnet options are available.')

        out_channels = in_channels
        if pyramid_type is None:
            upsampling_out_channels = self.backbone.in_channels_list
            self.conv1x1s = nn.ModuleList()
            for in_channel in self.backbone.in_channels_list:
                conv1x1 = nn.Sequential(nn.Conv2d(in_channel, backbone_pyramid_channels, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(backbone_pyramid_channels),
                                        nn.ReLU(inplace=True)
                                        )
                self.conv1x1s.append(conv1x1)
        else:
            upsampling_out_channels = [backbone_pyramid_channels for _ in self.backbone.in_channels_list]
            self.conv1x1s = None

        self.up3_skip = UpsamplingAdd(upsampling_out_channels[2], upsampling_out_channels[1], scale_factor=2)
        self.up2_skip = UpsamplingAdd(upsampling_out_channels[1], upsampling_out_channels[0], scale_factor=2)
        self.up1_skip = UpsamplingAdd(upsampling_out_channels[0], out_channels, scale_factor=2)

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)

        # feats = {'x': x}
        # feats.update(self.backbone(x))
        feats = self.backbone(x)

        # First upsample to (H/4, W/4)
        y = self.up3_skip(feats['2'], feats['1'])               # /4 <- /8 || /4
        # Second upsample to (H/2, W/2)
        y = self.up2_skip(y, feats['0'])                        # /2 <- /4 || /2
        # Third upsample to (H, W)
        y = self.up1_skip(y, x)                                 # /1 <- /2 || /1

        feats['y'] = y

        feats_out = {}
        keys = list(feats.keys())
        for i_feat, feat in enumerate(feats.values()):
            if self.conv1x1s is not None and keys[i_feat] != 'y':
                feat = self.conv1x1s[i_feat](feat)
            feats_out[keys[i_feat]] = feat.view(b, s, *feat.shape[1:])

        return feats_out
