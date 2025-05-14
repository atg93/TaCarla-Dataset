import torch

import torch.nn as nn

from tairvision_object_detection.models.bev.lss.blocks.coders import EncoderReXnetFpn, DecoderReXnetFpn
from tairvision_object_detection.models.bev.lss.blocks.temporal import TemporalModelIdentity, TemporalModel
from tairvision_object_detection.models.bev.lss.utils.network import (pack_sequence_dim, unpack_sequence_dim, set_bn_momentum,
                                                     remove_past_frames)
from tairvision_object_detection.models.bev.lss.utils.geometry import (create_frustum, get_geometry, projection_to_birds_eye_view,
                                                      create_bev_geometry, get_pixel_locations, map_pixels_to_bev,
                                                      map_pcloud_to_bev)
from tairvision_object_detection.models.bev.common.utils.geometry import (calculate_birds_eye_view_parameters,
                                                         cumulative_warp_features, concat_egomotion)
from tairvision_object_detection.models.bev.lss.blocks.heads import (FCOSNetAdaptor, FCOSBevNetAdaptor, DynamicSegmentationHead,
                                                    StaticSegmentationHead)
from tairvision_object_detection.models.bev.lss.layers.convolutions import SimpleFusion


class LiftSplat(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        try:
            bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(cfg.LIFT.X_BOUND,
                                                                                                    cfg.LIFT.Y_BOUND,
                                                                                                    cfg.LIFT.Z_BOUND
                                                                                                    )
        except:
            bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(cfg["LIFT"]["X_BOUND"],
                                                                                                    cfg["LIFT"]["Y_BOUND"],
                                                                                                    cfg["LIFT"]["Z_BOUND"]
                                                                                                    )

        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        try:
            self.encoder_out_channels = cfg.MODEL.ENCODER.OUT_CHANNELS
        except:
            self.encoder_out_channels = cfg["MODEL"]["ENCODER"]["OUT_CHANNELS"]

        try:
            self.frustum = create_frustum(cfg.IMAGE.FINAL_DIM, cfg.MODEL.ENCODER.DOWNSAMPLE, cfg.LIFT.D_BOUND)
        except:
            self.frustum = create_frustum(cfg["IMAGE"]["FINAL_DIM"], cfg["MODEL"]["ENCODER"]["DOWNSAMPLE"], cfg["LIFT"]["D_BOUND"])

        self.depth_channels, _, _, _ = self.frustum.shape

        # Spatial extent in bird's-eye view, in meters
        try:
            self.spatial_extent = (cfg.LIFT.X_BOUND[1], cfg.LIFT.Y_BOUND[1])
        except:
            self.spatial_extent = (cfg["LIFT"]["X_BOUND"][1], cfg["LIFT"]["Y_BOUND"][1])

        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item(), self.bev_dimension[2].item())

        try:
            # Encoder
            self.encoder = EncoderReXnetFpn(cfg=cfg.MODEL.ENCODER, D=self.depth_channels)
        except:
            self.encoder = EncoderReXnetFpn(cfg=cfg["MODEL"]["ENCODER"], D=self.depth_channels)

        try:
            # Pointcloud Fusion
            self.nb_pcloud_feats = cfg.PCLOUD.N_FEATS
        except:
            # Pointcloud Fusion
            self.nb_pcloud_feats = cfg["PCLOUD"]["N_FEATS"]

        self.fusion = SimpleFusion(in_channels_pixels=self.encoder_out_channels,
                                   in_channels_pcloud=self.nb_pcloud_feats * self.bev_dimension[2].item(),
                                   out_channels=self.encoder_out_channels)

        # Temporal model
        self.temporal_in_channels = self.encoder_out_channels
        try:
            self.receptive_field = cfg.TIME_RECEPTIVE_FIELD
        except:
            self.receptive_field = cfg["TIME_RECEPTIVE_FIELD"]

        self.temporal_model = TemporalModelIdentity(self.temporal_in_channels, self.receptive_field)

        try:
            # Decoder
            self.decoder = DecoderReXnetFpn(cfg.MODEL.DECODER, in_channels=self.temporal_model.out_channels)
        except:
            self.decoder = DecoderReXnetFpn(cfg["MODEL"]["DECODER"], in_channels=self.temporal_model.out_channels)


        self._init_heads(cfg)

        try:
            set_bn_momentum(self, cfg.MODEL.BN_MOMENTUM)
        except:
            set_bn_momentum(self, cfg["MODEL"]["BN_MOMENTUM"])

    def _init_heads(self, cfg):
        try:
            if cfg.MODEL.USE_HEADDYNAMIC:
                self.dynamic = DynamicSegmentationHead(cfg,
                                                       n_classes=len(cfg.MODEL.HEADDYNAMIC.SEMANTIC_SEG.WEIGHTS),
                                                       in_channels=self.temporal_model.out_channels)
            else:
                self.dynamic = None
        except:
            if cfg["MODEL"]["USE_HEADDYNAMIC"]:
                self.dynamic = DynamicSegmentationHead(cfg,
                                                       n_classes=len(cfg["MODEL"].HEADDYNAMIC.SEMANTIC_SEG.WEIGHTS),
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
            self.head3d = FCOSBevNetAdaptor(self.decoder.backbone,
                                            num_classes=cfg.MODEL.HEAD3D.NUM_CLASSES,
                                            regression_out_channels=cfg.MODEL.HEAD3D.REGRESSION_CHANNELS,
                                            regression_functions=cfg.MODEL.HEAD3D.REGRESSION_FUNCTIONS,
                                            in_channels=self.temporal_model.out_channels,
                                            )
        else:
            self.head3d = None

    def forward(self, image, intrinsics, extrinsics, view, future_egomotion=None, pcloud_list=None):
        # Only process features from the past and present
        image, intrinsics, extrinsics, view, _, pcloud_list = self.filter_inputs(image, intrinsics, extrinsics,
                                                                                 view, future_egomotion, pcloud_list)

        # Getting features from 2d backbone and lifting to 3d
        feats_3d, feats_2d = self.get_features(image)
        # Projecting to bird's-eye view
        feats_bev = self.calculate_bev_features(feats_3d, intrinsics, extrinsics, view, pcloud_list)

        # Temporal model
        states = self.temporal_model(feats_bev)

        # Predict bird's-eye view outputs
        feats_dec = self.decoder(states)

        # Get outputs for available heads using decoder features and 2d features
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
        b, s, n, c, h, w = x.shape
        x = x.view(b * s * n, c, h, w)

        x, feats_2d = self.encoder(x)
        x = x.view(b * s, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)
        x = unpack_sequence_dim(x, b, s)

        # take only the present frame for feats_2d
        feats_2d = remove_past_frames(feats_2d, b, s, n)

        return x, feats_2d

    def calculate_bev_features(self, x, intrinsics, extrinsics, view, pcloud_list=None):
        # batch, n_sequences, n_cameras, channels, height, width, depth
        b, s, n, c, h, w, d = x.shape
        # Reshape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)

        geometry = get_geometry(self.frustum, intrinsics, extrinsics)

        x = projection_to_birds_eye_view(x, geometry, view, self.bev_dimension)

        feats_pcloud = self.collate_pcloud_features(pcloud_list)

        x = self.fusion(x, feats_pcloud)
        x = unpack_sequence_dim(x, b, s)

        return x

    def get_pcloud_features(self, pcloud):

        b, s = pcloud.shape[0:2]

        pcloud = pack_sequence_dim(pcloud, pack_n=True)
        pcloud_feats = map_pcloud_to_bev(pcloud, self.bev_size)
        pcloud_feats = pcloud_feats.reshape(b * s, self.bev_size[0], self.bev_size[1], -1)
        pcloud_feats = pcloud_feats.permute(0, 3, 1, 2).contiguous()

        return pcloud_feats

    def collate_pcloud_features(self, pcloud_list, use_prob_train=(1.0, 0.5), use_prob_infer=(1.0, 0.)):

        if pcloud_list is None:
            return None

        feats_pcloud_list = []
        for i_pcloud, pcloud in enumerate(pcloud_list):
            threshold = use_prob_train[i_pcloud] if self.training else use_prob_infer[i_pcloud]

            feats_pcloud = self.get_pcloud_features(pcloud)
            feats_pcloud_coef = torch.rand_like(feats_pcloud[:, 0, 0, 0]).view(-1, 1, 1, 1)
            feats_pcloud_coef = feats_pcloud_coef < threshold
            feats_pcloud *= feats_pcloud_coef

            feats_pcloud_list.append(feats_pcloud)

        return torch.cat(feats_pcloud_list, dim=1)



class LiftSplatLinear(LiftSplat):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._init_mapping(cfg)

    def _init_mapping(self, cfg):
        self.geometry = create_bev_geometry(self.bev_size)
        self.image_final_dim = cfg.IMAGE.FINAL_DIM
        self.encoder_downsample = cfg.MODEL.ENCODER.DOWNSAMPLE

        self.fusion = SimpleFusion(in_channels_pixels=self.encoder_out_channels * self.bev_dimension[2].item(),
                                   in_channels_pcloud=self.nb_pcloud_feats * self.bev_dimension[2].item(),
                                   out_channels=self.encoder_out_channels)

        del self.frustum

        print('Using Linear Mapping')

    def calculate_bev_features(self, x, intrinsics, extrinsics, view, pcloud_list=None):
        # batch, n_sequences, n_cameras, depth, height, width, channels
        b, s, n, d, h, w, c = x.shape

        # Reshape
        x = pack_sequence_dim(x, pack_n=True)
        intrinsics = pack_sequence_dim(intrinsics, pack_n=True)
        extrinsics = pack_sequence_dim(extrinsics, pack_n=True)
        view = pack_sequence_dim(view.repeat(1, 1, n, 1, 1), pack_n=True)

        # get pixel locations and mask for invalid ones
        locations, mask = get_pixel_locations(self.geometry, intrinsics, extrinsics, view,
                                              self.image_final_dim, self.encoder_downsample)

        # map image pixels to bev
        x = map_pixels_to_bev(x, locations, mask, self.bev_size, n)

        # reduce the number of feats using a simple conv2d
        x = x.view(b * s, self.bev_size[0], self.bev_size[1], -1)
        x = x.permute(0, 3, 1, 2).contiguous()

        feats_pcloud = self.collate_pcloud_features(pcloud_list)

        x = self.fusion(x, feats_pcloud)
        x = unpack_sequence_dim(x, b, s)

        return x


class LiftSplatTemporal(LiftSplatLinear, LiftSplat):
    def __init__(self, cfg):
        self.depth_channels = (cfg.LIFT.D_BOUND[1] - cfg.LIFT.D_BOUND[0]) / cfg.LIFT.D_BOUND[2]
        LiftSplat.__init__(self, cfg)
        if self.depth_channels <= 1:
            LiftSplatLinear._init_mapping(self, cfg)

        # Temporal model
        if cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE:
            self.temporal_in_channels += 6

        bev_size = cfg.BEV_SCALED_SIZE if cfg.BEV_SCALE else self.bev_size[0:2]

        self.temporal_model = TemporalModel(self.temporal_in_channels, self.receptive_field,
                                            input_shape=bev_size,
                                            start_out_channels=cfg.MODEL.TEMPORAL_MODEL.START_OUT_CHANNELS,
                                            extra_in_channels=cfg.MODEL.TEMPORAL_MODEL.EXTRA_IN_CHANNELS,
                                            n_spatial_layers_between_temporal_layers=cfg.MODEL.TEMPORAL_MODEL.INBETWEEN_LAYERS,
                                            use_pyramid_pooling=cfg.MODEL.TEMPORAL_MODEL.PYRAMID_POOLING,
                                            )

        # Decoder
        self.decoder = DecoderReXnetFpn(cfg.MODEL.DECODER, in_channels=self.temporal_model.out_channels)

        self._init_heads(cfg)

    def forward(self, image, intrinsics, extrinsics, view, future_egomotion=None, pcloud_list=None):
        # Only process features from the past and present
        image, intrinsics, extrinsics, view, future_egomotion, pcloud_list = self.filter_inputs(image, intrinsics,
                                                                                                extrinsics, view,
                                                                                                future_egomotion,
                                                                                                pcloud_list)

        # Getting features from 2d backbone and lifting to 3d
        feats_3d, feats_2d = self.get_features(image)
        # Projecting to bird's-eye view
        feats_bev = self.calculate_bev_features(feats_3d, intrinsics, extrinsics, view, pcloud_list)
        # Warp past features to the present's reference frame
        feats_bev = cumulative_warp_features(feats_bev.clone(), future_egomotion, mode='bilinear',
                                             spatial_extent=self.spatial_extent)

        if self.cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE:
            feats_bev = concat_egomotion(feats_bev, future_egomotion, self.receptive_field)

        # Temporal model
        states = self.temporal_model(feats_bev)

        # Predict bird's-eye view outputs
        feats_dec = self.decoder(states)

        # Get outputs for available heads using decoder features and 2d features
        output = self.get_head_outputs(feats_dec, feats_2d)

        # compute the fcos3d heads outputs using the bev features
        if self.head3d is not None:
            output['head3d'] = self.head3d.get_head_outputs(feats_dec)

        return output

    def calculate_bev_features(self, x, intrinsics, extrinsics, view, pcloud_list=None):
        if self.depth_channels <= 1:
            return LiftSplatLinear.calculate_bev_features(self, x, intrinsics, extrinsics, view, pcloud_list)
        else:
            return LiftSplat.calculate_bev_features(self, x, intrinsics, extrinsics, view, pcloud_list)
