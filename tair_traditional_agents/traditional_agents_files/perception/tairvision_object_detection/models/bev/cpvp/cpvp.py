import torch

from tairvision_object_detection.models.bev.lss.lss import LiftSplat
from tairvision_object_detection.models.bev.cprm.blocks.heads import BEVDETAdaptor
from tairvision_object_detection.models.bev.lss.blocks.coders import EncoderReXnetFpn
from tairvision_object_detection.models.bev.lss.utils.network import pack_sequence_dim, unpack_sequence_dim
from tairvision_object_detection.models.bev.lss.utils.geometry import get_geometry
from tairvision_object_detection.models.bev.cpvp.utils.vputils import voxel_pooling_v2
from tairvision_object_detection.models.bev.lss.blocks.heads import FCOSNetAdaptor, DynamicSegmentationHead, StaticSegmentationHead


class EncoderReXnetFpnVP(EncoderReXnetFpn):
    def __init__(self, cfg, D):
        super().__init__(cfg, D)
        self.bevpoolv2 = cfg.USE_BEVPOOLV2

    def forward(self, x):
        feats_2d = self.backbone(x)  # get feature dict of tensor
        x = self.upscale_and_concat_features(feats_2d)  # get feature vector
        x = self.depth_layer(x)  # feature and depth head

        if self.bevpoolv2:
            depth = x[:, : self.D]  # .softmax(dim=1)
            x = x[:, self.D: (self.D + self.C)]
            return x, depth, feats_2d
        elif self.use_depth_distribution:
            depth = x[:, : self.D].softmax(dim=1)
            x = depth.unsqueeze(1) * x[:, self.D: (self.D + self.C)].unsqueeze(
                2)  # outer product depth and features
        else:
            x = x.unsqueeze(2).repeat(1, 1, self.D, 1, 1)

        return x, feats_2d


class CPVP(LiftSplat):
    def __init__(self, cfg):
        self.head_channels = cfg.MODEL.ENCODER.OUT_CHANNELS
        super().__init__(cfg)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.encoder = EncoderReXnetFpnVP(cfg=cfg.MODEL.ENCODER, D=self.depth_channels)

    def _init_heads(self, cfg):

        if cfg.MODEL.USE_HEADDYNAMIC:
            self.dynamic = DynamicSegmentationHead(cfg,
                                                   n_classes=len(cfg.MODEL.HEADDYNAMIC.SEMANTIC_SEG.WEIGHTS),
                                                   in_channels=self.temporal_model.out_channels)
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
            self.head3d = BEVDETAdaptor(cfg=cfg)
        else:
            self.head3d = None
        self.cfg = cfg
        self.use_depth = cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION
        self.use_radar = cfg.USE_RADAR
        assert self.use_depth is True
        self.enc_out_ch = cfg.MODEL.ENCODER.OUT_CHANNELS

        # TODO: DEAL WITH 8 HERE as upDim==8
        enc_out_ch = cfg.MODEL.ENCODER.OUT_CHANNELS

        if self.use_radar:
            self.bev_compressor = torch.nn.Sequential(
                torch.nn.Conv2d(enc_out_ch * 8 + 16 * 8, enc_out_ch, kernel_size=3, padding=1, stride=1,
                                bias=False),
                torch.nn.BatchNorm2d(enc_out_ch),
                torch.nn.ReLU(inplace=True),
            )

    def forward(self, image, intrinsics, extrinsics, view, future_egomotion, pcloud_list=None, is_train=False):
        # Only process features from the past and present

        image, intrinsics, extrinsics, view, future_egomotion, pcloud_list = self.filter_inputs(image, intrinsics, extrinsics,
                                                                                   view, future_egomotion, pcloud_list)

        feats_3d, depth, feats_2d = self.get_features(image)
        # Projecting to bird's-eye view using Reverse Mapping
        feats_pcloud = self.collate_pcloud_features(pcloud_list)
        feats_bev = self.calculate_bev_features(feats_3d, depth.softmax(dim=2), intrinsics, extrinsics, view, feats_pcloud)
        # Temporal model
        states = self.temporal_model(feats_bev)
        # Predict bird's-eye view outputs
        feats_dec = self.decoder(states)
        # Get outputs for available heads using decoder features and 2d features
        output = self.get_head_outputs(feats_dec, feats_2d)

        return output


    def calculate_bev_features(self, x, depth, intrinsics, extrinsics, view, feats_pcloud=None):
        b, s = intrinsics.shape[:2]
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)

        geometry = get_geometry(self.frustum, intrinsics, extrinsics)
        feat_bev = voxel_pooling_v2(x, depth, geometry, view, self.bev_dimension)
        if feats_pcloud is not None:
            feats_radar = feats_pcloud[:, :128]
            feat_bev = torch.cat([feat_bev, feats_radar], dim=1)
        else:
            pass
        feats_bev = self.bev_compressor(feat_bev)
        feats_bev = unpack_sequence_dim(feats_bev, b, s)

        return feats_bev


    def get_features(self, x):
        b, s, n, c, h, w = x.shape
        x = x.view(b * s * n, c, h, w)
        x, depth, feats_2d = self.encoder(x)
        x = x.view(b * s, n, *x.shape[1:])
        depth = depth.view(b * s, n, *depth.shape[1:])

        return x, depth, feats_2d

