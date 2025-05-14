import torch
from tairvision_object_detection.models.bev.lss.lss import LiftSplatLinear
from tairvision_object_detection.models.bev.cprm.blocks.heads import BEVDETAdaptor
from tairvision_object_detection.models.bev.cprm.blocks.reverse_map import ReverseMapping
from tairvision_object_detection.models.bev.lss.blocks.coders import EncoderReXnetFpn
from tairvision_object_detection.models.bev.lss.blocks.heads import FCOSNetAdaptor, DynamicSegmentationHead, StaticSegmentationHead


class CPRM(LiftSplatLinear):
    def __init__(self, cfg):
        self.head_channels = cfg.MODEL.ENCODER.OUT_CHANNELS
        self.use_lidar_head = cfg.USE_LIDAR_HEAD
        super().__init__(cfg)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.use_depth = cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION
        self.use_radar = cfg.USE_RADAR
        assert self.use_depth is False
        self.enc_out_ch = cfg.MODEL.ENCODER.OUT_CHANNELS


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

        # Encoder
        # self.RevMap = ReverseMapping(cfg, self.bev_dimension)

    """
    def forward(self, image, intrinsics, extrinsics, view, future_egomotion=None, pcloud_list=None, inverse_view=None):
        # Only process features from the past and present

        image, intrinsics, extrinsics, view, future_egomotion, pcloud_list = self.filter_inputs(image, intrinsics, extrinsics,
                                                                                   view, future_egomotion, pcloud_list)

        # Getting features from 2d backbone and lifting to 3d
        feats_3d, feats_2d = self.get_features(image)
        B = feats_3d.shape[0]
        # Projecting to bird's-eye view using Reverse Mapping
        feats_pcloud = self.collate_pcloud_features(pcloud_list)
        feats_bev = self.RevMap(feats_3d, intrinsics, extrinsics, view, feats_pcloud[:, 0:128], inverse_view=inverse_view)

        # Temporal model
        states = self.temporal_model(feats_bev)
        # Predict bird's-eye view outputs
        feats_dec = self.decoder(states)
        # Get outputs for available heads using decoder features and 2d features
        output = self.get_head_outputs(feats_dec, feats_2d)

        return output

    
    def get_features(self, x):
        b, s, n, c, h, w = x.shape
        x = x.view(b * s * n, c, h, w)

        x, feats_2d = self.encoder(x)
        x = x.squeeze(2)
        x = x.view(b * s, n, *x.shape[1:])
    

        return x, feats_2d
    """