import torch

from tairvision_object_detection.models.bev.lss.blocks.coders import DecoderReXnetFpn
from tairvision_object_detection.models.bev.lss.blocks.heads import FlowHead
from tairvision_object_detection.models.bev.lss.lss import LiftSplatTemporal
from tairvision_object_detection.models.bev.fiery.blocks.distributions import DistributionModule
from tairvision_object_detection.models.bev.fiery.blocks.prediction import FuturePrediction
from tairvision_object_detection.models.bev.common.utils.geometry import cumulative_warp_features, concat_egomotion
import torch.nn.functional as F
from tairvision_object_detection.models.bev.cprm.blocks.reverse_map import ReverseMapping


class Fiery(LiftSplatTemporal):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Prediction block
        self.n_future = self.cfg.N_FUTURE_FRAMES
        self.latent_dim = self.cfg.MODEL.DISTRIBUTION.LATENT_DIM
        self.scale = self.cfg.BEV_SCALE
        self.scaled_size = self.cfg.BEV_SCALED_SIZE

        assert self.n_future > 0

        # probabilistic sampling
        if self.cfg.PROBABILISTIC.ENABLED:
            assert self.cfg.INSTANCE_FLOW.ENABLED
            # Distribution networks
            self.present_distribution = self.get_distribution_module(self.temporal_model.out_channels)

            future_distribution_in_channels = (self.temporal_model.out_channels +
                                               self.n_future * self.cfg.PROBABILISTIC.FUTURE_DIM)

            self.future_distribution = self.get_distribution_module(future_distribution_in_channels)

        # Future prediction
        self.future_prediction = self.get_future_predictor()

        # Decoder
        self.decoder = DecoderReXnetFpn(cfg.MODEL.DECODER, in_channels=self.temporal_model.out_channels)

        self._init_heads(cfg)

        self.flow = self.get_flow_head(cfg)

        self.RevMap = ReverseMapping(cfg, self.bev_dimension)


    def forward(self, image, intrinsics, extrinsics, view, future_egomotion, future_distribution_inputs=None,
                noise=None, pcloud_list=None):

        # Only process features from the past and present
        image, intrinsics, extrinsics, view, future_egomotion, pcloud_list = self.filter_inputs(image, intrinsics, extrinsics,
                                                                                   view, future_egomotion, pcloud_list)

        # Getting features from 2d backbone and lifting to 3d
        # Then, projecting to bird's-eye view
        if self.cfg.BEV_MODE == 'rev_map':
            feats_3d, feats_2d = self.get_features_revmap(image)
            feats_bev = self.RevMap(feats_3d, intrinsics, extrinsics, radar_data=None, ref_t_mem=None)
        else:
            feats_3d, feats_2d = self.get_features(image)
            feats_bev = self.calculate_bev_features(feats_3d, intrinsics, extrinsics, view, pcloud_list)

        # Warp past features to the present's reference frame
        feats_bev = cumulative_warp_features(feats_bev.clone(), future_egomotion, mode='bilinear', spatial_extent=self.spatial_extent)

        # Scale model input to 120 x 120 if required
        if self.scale:
            feats_bev, future_distribution_inputs = self.scale_features(feats_bev, future_distribution_inputs)

        if self.cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE:
            feats_bev = concat_egomotion(feats_bev, future_egomotion, self.receptive_field)

        # Temporal model
        states = self.temporal_model(feats_bev)

        future_states, output = self.get_future_states(states, future_distribution_inputs, noise)

        # Predict bird's-eye view outputs
        feats_dec = self.decoder(future_states)

        output = self.get_head_outputs(feats_dec, feats_2d, output=output)

        # compute the fcos3d heads outputs using the bev features
        if self.head3d is not None:
            output['head3d'] = self.head3d.get_head_outputs(feats_dec)

        if self.flow is not None:
            output_flow = self.flow.get_head_outputs(feats_dec)
            output.update(output_flow)

        if self.scale:
            output = self.scale_output(output)

        return output

    def get_future_states(self, states, future_distribution_inputs, noise):
        output = {}

        present_state = states[:, :1].contiguous()
        if self.cfg.PROBABILISTIC.ENABLED:
            # Do probabilistic computation
            sample, output_distribution = self.distribution_forward(present_state, future_distribution_inputs, noise)
            output = {**output, **output_distribution}

        # Prepare future prediction input
        b, _, _, h, w = present_state.shape
        hidden_state = present_state[:, 0]

        if self.cfg.PROBABILISTIC.ENABLED:
            future_prediction_input = sample.expand(-1, self.n_future, -1, -1, -1)
        else:
            future_prediction_input = hidden_state.new_zeros(b, self.n_future, self.latent_dim, h, w)

        # Recursively predict future states
        future_states = self.future_prediction(future_prediction_input, hidden_state)

        # Concatenate present state
        future_states = torch.cat([present_state, future_states], dim=1)

        return future_states, output

    def distribution_forward(self, present_features, future_distribution_inputs=None, noise=None):
        """
        Parameters
        ----------
            present_features: 5-D output from dynamics module with shape (b, 1, c, h, w)
            future_distribution_inputs: 5-D tensor containing labels shape (b, s, cfg.PROB_FUTURE_DIM, h, w)
            noise: a sample from a (0, 1) gaussian with shape (b, s, latent_dim). If None, will sample in function

        Returns
        -------
            sample: sample taken from present/future distribution, broadcast to shape (b, s, latent_dim, h, w)
            present_distribution_mu: shape (b, s, latent_dim)
            present_distribution_log_sigma: shape (b, s, latent_dim)
            future_distribution_mu: shape (b, s, latent_dim)
            future_distribution_log_sigma: shape (b, s, latent_dim)
        """
        b, s, _, h, w = present_features.size()
        assert s == 1

        present_mu, present_log_sigma = self.present_distribution(present_features)

        if noise is None:
            if self.training:
                noise = torch.randn_like(present_mu)
            else:
                noise = torch.zeros_like(present_mu)

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            # Concatenate future labels to z_t
            future_features = future_distribution_inputs[:, 1:].contiguous().view(b, 1, -1, h, w)
            future_features = torch.cat([present_features, future_features], dim=2)
            future_mu, future_log_sigma = self.future_distribution(future_features)

        if self.training:
            mu = future_mu
            sigma = torch.exp(future_log_sigma)
        else:
            mu = present_mu
            sigma = torch.exp(present_log_sigma)

        sample = mu + sigma * noise

        # Spatially broadcast sample to the dimensions of present_features
        sample = sample.view(b, s, self.latent_dim, 1, 1).expand(b, s, self.latent_dim, h, w)

        output_distribution = {'present_mu': present_mu,
                               'present_log_sigma': present_log_sigma,
                               'future_mu': future_mu,
                               'future_log_sigma': future_log_sigma,
                               }

        return sample, output_distribution

    def get_features_revmap(self, x):
        b, s, n, c, h, w = x.shape
        x = x.view(b * s * n, c, h, w)

        x, feats_2d = self.encoder(x)
        x = x.squeeze(2)
        x = x.view(b * s, n, *x.shape[1:])
        return x, feats_2d

    def get_distribution_module(self, in_channels):
        dist_module = DistributionModule(in_channels,
                                         self.latent_dim,
                                         min_log_sigma=self.cfg.MODEL.DISTRIBUTION.MIN_LOG_SIGMA,
                                         max_log_sigma=self.cfg.MODEL.DISTRIBUTION.MAX_LOG_SIGMA,
                                         model_name="fiery")
        return dist_module

    def get_future_predictor(self):
        future_prediction = FuturePrediction(self.temporal_model.out_channels,
                                             latent_dim=self.latent_dim,
                                             n_gru_blocks=self.cfg.MODEL.FUTURE_PRED.N_GRU_BLOCKS,
                                             n_res_layers=self.cfg.MODEL.FUTURE_PRED.N_RES_LAYERS)
        return future_prediction

    def get_flow_head(self, cfg):
        if cfg.INSTANCE_FLOW.ENABLED:
            return FlowHead(cfg, self.temporal_model.out_channels)
        else:
            return None

    def scale_features(self, feats_bev, future_distribution_inputs):
        feats_bev = F.interpolate(feats_bev, size=(feats_bev.shape[2], self.scaled_size[0], self.scaled_size[1]),
                                  mode='trilinear')
        future_distribution_inputs = F.interpolate(future_distribution_inputs,
                                                   size=(future_distribution_inputs.shape[2], self.scaled_size[0],
                                                         self.scaled_size[1]), mode='trilinear')
        return  feats_bev, future_distribution_inputs

    def scale_output(self, output):
        output['segm'] = F.interpolate(output['segm'], size=(2, 200, 200), mode='trilinear')
        output['center'] = F.interpolate(output['center'], size=(1, 200, 200), mode='trilinear')
        output['offset'] = F.interpolate(output['offset'], size=(2, 200, 200), mode='trilinear')
        output['zpos'] = F.interpolate(output['zpos'], size=(2, 200, 200), mode='trilinear')
        output['flow'] = F.interpolate(output['flow'], size=(2, 200, 200), mode='trilinear')
        return output

