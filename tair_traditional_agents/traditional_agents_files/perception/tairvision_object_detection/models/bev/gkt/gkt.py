import torch
import torch.nn as nn
from einops import rearrange, repeat

from tairvision_object_detection.models.bev.cvt.cvt import CVT, ResNetBottleNeck
from tairvision_object_detection.models.bev.gkt.blocks.encoder import GeometryKernelAttention


class GKT(CVT):
    def __init__(self, cfg):
        super().__init__(cfg)
        # BEV & Cross View Attention parameters
        self.cv_cfg = self.cfg.MODEL.ENCODER.BACKBONE.CROSS_VIEW

        self.cross_view = {
            'heads': self.cv_cfg.HEADS, 'dim_head': self.cv_cfg.DIM_HEAD,
            'qkv_bias': self.cv_cfg.QKV_BIAS, 'skip': self.cv_cfg.SKIP,
            'no_image_features': self.cv_cfg.NO_IMG_FEATS,
            'image_height': self.cfg.IMAGE.FINAL_DIM[0], 'image_width': self.cfg.IMAGE.FINAL_DIM[1],
            'bev_z': self.cv_cfg.BEV_Z, 'kernel_h': self.cv_cfg.KERNEL_H, 'kernel_w': self.cv_cfg.KERNEL_W,
        }

        cross_views = list()
        layers = list()

        for feat_shape, num_layers in zip(self.encoder.output_shapes, self.cv_cfg.MID):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = GeometryKernelAttention(feat_height, feat_width, feat_dim,
                                          self.cfg.MODEL.ENCODER.BACKBONE.PYRAMID_CHANNELS, **self.cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(
                *[ResNetBottleNeck(self.cfg.MODEL.ENCODER.BACKBONE.PYRAMID_CHANNELS) for _ in range(num_layers)])
            layers.append(layer)

        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def calculate_bev_features(self, x, intrinsics, extrinsics, view, pcloud_feat=None):

        b, n, _, _ = intrinsics.shape
        I_inv = intrinsics.inverse()
        E_inv = extrinsics
        I = intrinsics
        E = extrinsics.inverse()
        q = self.bev_embedding.get_prior()  # d H W
        q = repeat(q, '... -> b ...', b=b)  # b d H W

        for cross_view, feature, layer in zip(self.cross_views, x, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(q, self.bev_embedding, feature, I_inv, E_inv, I, E, view, pcloud_feat)
            x = layer(x)

        return x
