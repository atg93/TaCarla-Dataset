import torch
from tairvision_object_detection.models.bev.common.utils.geometry import cumulative_warp_features, concat_egomotion
from tairvision_object_detection.models.bev.fiery.fiery import Fiery


class FieryCache(Fiery):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, image, intrinsics, extrinsics, view, future_egomotion, future_distribution_inputs=None,
                noise=None, pcloud_list=None, new_scene=False):
        self.receptive_field = 1
        # Only process features from the past and present
        image, intrinsics, extrinsics, view, future_egomotion, _ = self.filter_inputs(image, intrinsics, extrinsics,
                                                                                   view, future_egomotion, pcloud_list)

        # Getting features from 2d backbone and lifting to 3d
        # Then, projecting to bird's-eye view
        if self.cfg.BEV_MODE == 'rev_map':
            feats_3d, feats_2d = self.get_features_revmap(image)
            feats_bev = self.RevMap(feats_3d, intrinsics, extrinsics, radar_data=None, ref_t_mem=None)
        else:
            feats_3d, feats_2d = self.get_features(image)
            feats_bev = self.calculate_bev_features(feats_3d, intrinsics, extrinsics, view)

        self._init_memory(feats_bev, future_egomotion, new_scene)

        feats_bev = torch.cat([self.memory_bev, feats_bev], dim=1)
        future_egomotion = torch.cat([self.memory_ego, future_egomotion], dim=1)
        self._set_memory(self, feats_bev, future_egomotion)

        # Warp past features to the present's reference frame
        feats_bev = cumulative_warp_features(feats_bev.clone(), future_egomotion, mode='bilinear', spatial_extent=self.spatial_extent)

        # Scale model input to 120 x 120 if required
        if self.scale:
            feats_bev, future_distribution_inputs = self.scale_features(feats_bev, future_distribution_inputs)

        if self.cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE:
            self.receptive_field = 3
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

    def _init_memory(self, feats_bev, future_egomotion, new_scene):
        if new_scene:
            self.memory_bev = torch.cat([feats_bev, feats_bev], dim=1)
            self.memory_ego = torch.cat([future_egomotion, future_egomotion], dim=1)
        else:
            self.memory_bev = self.memory_bev[:, 1:]
            self.memory_ego = self.memory_ego[:, 1:]

    def _set_memory(self, feats_bev, future_egomotion):
        self.memory_bev = feats_bev
        self.memory_ego = future_egomotion