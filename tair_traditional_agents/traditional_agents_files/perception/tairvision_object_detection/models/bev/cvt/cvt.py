import torch
import torch.nn as nn
from einops import rearrange, repeat

from tairvision_object_detection.models.bev.common.utils.geometry import calculate_birds_eye_view_parameters
from tairvision_object_detection.models.bev.lss.utils.geometry import map_pcloud_to_bev
from tairvision_object_detection.models.bev.lss.layers.convolutions import SimpleFusion
from tairvision_object_detection.models.bev.lss.blocks.coders import DecoderReXnetFpn
from tairvision_object_detection.models.bev.cvt.blocks.encoder import CrossViewAttention, BEVEmbedding
from tairvision_object_detection.models.bev.lss.blocks.coders import EncoderReXnetFpn
from tairvision_object_detection.models.bev.lss.blocks.heads import (
    FCOSNetAdaptor, FCOSBevNetAdaptor, DynamicSegmentationHead, StaticSegmentationHead, )
from tairvision_object_detection.models.bev.lss.blocks.temporal import TemporalModelIdentity
from tairvision_object_detection.models.bev.lss.utils.network import pack_sequence_dim, set_bn_momentum
from tairvision_object_detection.models.resnet import Bottleneck

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def get_pcloud_features(pcloud, bev_size=(200, 200, 1)):
    b, s = pcloud.shape[0:2]

    pcloud = pack_sequence_dim(pcloud, pack_n=True)
    pcloud_feats = map_pcloud_to_bev(pcloud, bev_size)
    pcloud_feats = pcloud_feats.reshape(b * s, bev_size[0], bev_size[1], -1)
    pcloud_feats = pcloud_feats.permute(0, 3, 1, 2).contiguous()

    return pcloud_feats


class CVT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(cfg.LIFT.X_BOUND,
                                                                                                cfg.LIFT.Y_BOUND,
                                                                                                cfg.LIFT.Z_BOUND
                                                                                                )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.encoder_out_channels = cfg.MODEL.ENCODER.OUT_CHANNELS

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (cfg.LIFT.X_BOUND[1], cfg.LIFT.Y_BOUND[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item(), self.bev_dimension[2].item())

        # BEV & Cross View Attention parameters
        self.cv_cfg = self.cfg.MODEL.ENCODER.BACKBONE.CROSS_VIEW

        self.cross_view = {
            'heads': self.cv_cfg.HEADS, 'dim_head': self.cv_cfg.DIM_HEAD,
            'qkv_bias': self.cv_cfg.QKV_BIAS, 'skip': self.cv_cfg.SKIP,
            'no_image_features': self.cv_cfg.NO_IMG_FEATS,
            'image_height': self.cfg.IMAGE.FINAL_DIM[0], 'image_width': self.cfg.IMAGE.FINAL_DIM[1]
        }

        self.bv_cfg = self.cfg.MODEL.ENCODER.BACKBONE.BEV_EMBEDDING

        self.bev_embedding = {
            'sigma': self.bv_cfg.SIGMA,
            'bev_height': self.bev_dimension[0].item(), 'bev_width': self.bev_dimension[0].item(),
            'h_meters': self.bv_cfg.H_METERS, 'w_meters': self.bv_cfg.W_METERS,
            'offset': self.bv_cfg.OFFSET, 'decoder_blocks': self.bv_cfg.DECODER_BLOCKS
        }

        self.encoder = EncoderReXnetFpn(cfg=cfg.MODEL.ENCODER, D=1)

        self.encoder.output_shapes = [torch.Size(
            [1, self.encoder_out_channels, int(self.cfg.IMAGE.FINAL_DIM[0] / i), int(self.cfg.IMAGE.FINAL_DIM[1] / i)])
            for i in self.cfg.MODEL.ENCODER.BACKBONE.DOWNSAMPLE]
        self.down = lambda x: x

        cross_views = list()
        layers = list()

        for feat_shape, num_layers in zip(self.encoder.output_shapes, self.cv_cfg.MID):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewAttention(feat_height, feat_width, feat_dim,
                                     self.cfg.MODEL.ENCODER.BACKBONE.PYRAMID_CHANNELS, **self.cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(
                *[ResNetBottleNeck(self.cfg.MODEL.ENCODER.BACKBONE.PYRAMID_CHANNELS) for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(self.cfg.MODEL.ENCODER.BACKBONE.PYRAMID_CHANNELS, **self.bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

        # Temporal model
        self.temporal_in_channels = self.encoder_out_channels
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD
        self.temporal_model = TemporalModelIdentity(self.temporal_in_channels, self.receptive_field)

        self.up = nn.Upsample(scale_factor=2 ** len(self.bv_cfg.DECODER_BLOCKS), mode='bilinear', align_corners=False)

        self.pc_mlp = nn.Sequential(nn.Linear(cfg.PCLOUD.N_FEATS, self.encoder_out_channels),
                                    nn.GELU(),
                                    nn.Linear(self.encoder_out_channels, int(self.encoder_out_channels / 2)))

        # Decoder
        self.decoder = DecoderReXnetFpn(cfg.MODEL.DECODER, in_channels=self.temporal_model.out_channels)
        self._init_heads(cfg)

        # Pointcloud Fusion
        self.nb_pcloud_feats = cfg.PCLOUD.N_FEATS
        self.fusion = SimpleFusion(in_channels_pixels=self.encoder_out_channels,
                                   in_channels_pcloud=self.nb_pcloud_feats * self.bev_dimension[2].item(),
                                   out_channels=self.encoder_out_channels)

        set_bn_momentum(self, cfg.MODEL.BN_MOMENTUM)

    def _init_heads(self, cfg):

        if cfg.MODEL.USE_HEADDYNAMIC:
            self.dynamic = self._init_dynamic_head(cfg)
        else:
            self.dynamic = None

        if cfg.MODEL.USE_HEADSTATIC:
            self.static = StaticSegmentationHead(cfg, in_channels=self.temporal_model.out_channels)
        else:
            self.static = None

        if cfg.MODEL.USE_HEAD2D:
            self.head2d = FCOSNetAdaptor(self.encoder.backbone, num_classes=cfg.MODEL.HEAD2D.NUM_CLASSES)
        else:
            self.head2d = None

        if cfg.MODEL.USE_HEAD3D:
            self.head3d = FCOSBevNetAdaptor(self.decoder.backbone,
                                            num_classes=cfg.MODEL.HEAD3D.NUM_CLASSES,
                                            regression_out_channels=cfg.MODEL.HEAD3D.REGRESSION_CHANNELS,
                                            in_channels=self.temporal_model.out_channels
                                            )
        else:
            self.head3d = None

    def _init_dynamic_head(self, cfg):
        return DynamicSegmentationHead(
            cfg,
            n_classes=len(cfg.MODEL.HEADDYNAMIC.SEMANTIC_SEG.WEIGHTS),
            in_channels=self.temporal_model.out_channels
        )

    def forward(self, image, intrinsics, extrinsics, view, future_egomotion=None, pcloud_list=None):

        # Only process features from the past and present
        image, intrinsics, extrinsics, view, future_egomotion, pcloud_list = self.filter_inputs(image,
                                                                                                intrinsics,
                                                                                                extrinsics,
                                                                                                view,
                                                                                                future_egomotion,
                                                                                                pcloud_list)

        image = pack_sequence_dim(image)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)
        view = pack_sequence_dim(view)
        # feats_pcloud = self.collate_pcloud_features(pcloud_list, bev_size=(25, 25, 1))
        # feats_pcloud = feats_pcloud.clone().permute(0, 2, 3, 1)
        # feats_pcloud = self.pc_mlp(feats_pcloud)
        # feats_pcloud = feats_pcloud.clone().permute(0, 3, 1, 2)
        #
        # x = torch.cat((x, feats_pcloud), dim=1)

        feats_3d, feats_2d = self.get_features(image)
        feats_bev = self.calculate_bev_features(feats_3d, intrinsics, extrinsics, view, pcloud_feat=None)

        # Temporal model
        states = self.temporal_model(feats_bev)

        states = self.up(states)

        feats_pcloud = self.collate_pcloud_features(pcloud_list)
        states = self.fusion(states, feats_pcloud)

        # Predict bird's-eye view outputs
        b, c, h, w = states.shape
        feats_dec = self.decoder(states.view(b, 1, c, h, w))

        output = self.get_head_outputs(feats_dec, feats_2d)

        return output

    def get_head_outputs(self, feats_dec, feats_2d, output=None):

        if output is None:
            output = {}

        # compute dynamic segmentation outputs using the bev features
        if self.dynamic is not None:
            output_dynamic = self.dynamic.get_head_outputs(feats_dec)
            output.update(output_dynamic)

        # compute static segmentation outputs using the bev features
        if self.static is not None:
            output_static = self.static.get_head_outputs(feats_dec)
            output.update(output_static)

        # compute the fcos heads outputs using the 2d features
        if self.head2d is not None:
            output['head2d'] = self.head2d.get_head_outputs(feats_2d)

        # Compute the fcos3d heads outputs using the bev features
        if self.head3d is not None:
            output['head3d'] = self.head3d.get_head_outputs(feats_dec)

        return output

    def filter_inputs(self, image, intrinsics, extrinsics, view, future_egomotion=None, pcloud_list=None):
        image = image[:, :self.receptive_field].contiguous()
        intrinsics = intrinsics[:, :self.receptive_field].contiguous()
        extrinsics = extrinsics[:, :self.receptive_field].contiguous()
        view = view[:, :self.receptive_field].contiguous()
        if future_egomotion is not None:
            future_egomotion = future_egomotion[:, :self.receptive_field].contiguous()
        if pcloud_list is not None:
            pcloud_list = [p[:, :self.receptive_field].contiguous() for p in pcloud_list]

        return image, intrinsics, extrinsics, view, future_egomotion, pcloud_list

    def get_features(self, x):
        # batch, n_sequences, n_cameras, channels, height, width
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w)
        x, feats_2d = self.encoder(x)
        y = [feats_2d[i] for i in self.cfg.MODEL.ENCODER.BACKBONE.LAYERS_TO_CVA]
        features = [self.down(z) for z in y]

        feats_2d.pop('0', None)

        return features, feats_2d

    def calculate_bev_features(self, x, intrinsics, extrinsics, view, pcloud_feat=None):

        b, n, _, _ = intrinsics.shape
        I_inv = intrinsics.inverse()
        E_inv = extrinsics
        q = self.bev_embedding.get_prior()  # d H W
        q = repeat(q, '... -> b ...', b=b)  # b d H W

        for cross_view, feature, layer in zip(self.cross_views, x, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(q, self.bev_embedding, feature, I_inv, E_inv, view, pcloud_feat=pcloud_feat)
            x = layer(x)

        return x

    def collate_pcloud_features(self, pcloud_list, use_prob_train=(1.0, 0.5), use_prob_infer=(1.0, 0.),
                                bev_size=(200, 200, 1)):

        if pcloud_list is None:
            return None

        feats_pcloud_list = []
        for i_pcloud, pcloud in enumerate(pcloud_list):
            threshold = use_prob_train[i_pcloud] if self.training else use_prob_infer[i_pcloud]

            feats_pcloud = get_pcloud_features(pcloud, bev_size=bev_size)
            feats_pcloud_coef = torch.rand_like(feats_pcloud[:, 0, 0, 0]).view(-1, 1, 1, 1)
            feats_pcloud_coef = feats_pcloud_coef < threshold
            feats_pcloud *= feats_pcloud_coef

            feats_pcloud_list.append(feats_pcloud)

        return torch.cat(feats_pcloud_list, dim=1)
