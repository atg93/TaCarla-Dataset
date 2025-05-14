from tairvision.models.bev.lss.lss import LiftSplatLinear, LiftSplatTemporal
from tairvision.models.bev.lss_mask2former.blocks.optimal_transport import OptimalTransportCenterlines, OptimalTransportTrafficElements
from tairvision.models.bev.lss_mask2former.blocks.relation_improver import RelationImprover
from tairvision.models.bev.common.utils.geometry import cumulative_warp_features, concat_egomotion
from tairvision.models.bev.lss_mask2former.blocks.final_centerline_predictor import FinalCenterLinePredictor
from .lss_mask2former import LiftSplatMask2Former
from tairvision.models.bev.lss_mask2former.blocks.heads import Mask2FormerDynamicHead

    
class LiftSplatMask2FormerTopology(LiftSplatMask2Former):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_heads(self, cfg):
        # Centerline head is used to predict the centerlines as masks
        self.centerline_head = self._init_centerline_head(cfg)

        # 2D head is used to predict the traffic elements
        if cfg.MODEL.USE_HEAD2D:
            self.head2d = self._init_2d_head(cfg)
        else:
            self.head2d = None

        # Optimal transport heads aims to improve the relations between the centerlines and the traffic elements
        # and among the centerlines. It is composed of two parts:
        # 1. Optimal transport between centerlines and traffic elements (OptimalTransportTrafficElements)
        # 2. Optimal transport among centerlines (OptimalTransportCenterlines)
        if cfg.MODEL.USE_OPTIMAL_TRANSPORT_HEAD:
            self.optimal_transport_lclc = OptimalTransportCenterlines(
                cfg = cfg,
                matcher = self.centerline_head.matcher
            )
            if cfg.MODEL.USE_HEAD2D:
                self.optimal_transport_lcte = OptimalTransportTrafficElements(
                    cfg = cfg,
                    matcher_te = self.head2d.matcher,
                    matcher_lc = self.centerline_head.matcher
                )
            else:
                self.optimal_transport_lcte = None
        else:
            self.optimal_transport_lclc = None
            self.optimal_transport_lcte = None

        # Relation improver is used to improve the relations between the centerlines and the traffic elements
        # Between the heads of traffic elements and centerlines, there is no relationship block. Therefore, the
        # relation improver is only used to create relations between the centerlines and the traffic elements. 
        if cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_IMPROVED_RELATIONS:
            self.relation_improver = RelationImprover(cfg)
        else:
            self.relation_improver = None
        
        # Separate relationship block for final centerline prediction. It is in the research stage.
        if cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_SEPARATE_IMPROVED_RELATIONS_FOR_FINAL_CENTERLINE:
            self.relation_improver_final_centerline = RelationImprover(cfg)
        else:
            self.relation_improver_final_centerline = None

        # Final centerline predictor is used to predict the final centerlines after the relationship blocks.
        # It is in the research stage.
        if cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_CENTERLINE_PREDICTION_AFTER_RELATION:
            self.centerline_last = FinalCenterLinePredictor(cfg, self.centerline_head.matcher)
        else:
            self.centerline_last = None

    def _init_centerline_head(self, cfg):
        return Mask2FormerDynamicHead(
            cfg,
            n_classes=5, #background, up, down, left, right
            in_channels=self.temporal_model.out_channels,
            in_channels_list=self.decoder.backbone.in_channels_list,
            strides=self.decoder.backbone.strides
        )

    def forward(self, image, intrinsics, extrinsics, view, front_view_image=None, future_egomotion=None, pcloud_list=None):
        # Only process features from the past and present
        image, intrinsics, extrinsics, view, _, pcloud_list = self.filter_inputs(image, intrinsics, extrinsics,
                                                                                 view, future_egomotion, pcloud_list)

        # Getting features from 2d backbone and lifting to 3d
        feats_3d, feats_2d = self.get_features(image)

        # Projecting to bird's-eye view
        feats_bev = self.calculate_bev_features(feats_3d, intrinsics, extrinsics, view, pcloud_list)

        if self.cfg.TIME_RECEPTIVE_FIELD > 1:
            feats_bev = cumulative_warp_features(feats_bev.clone(), future_egomotion, mode='bilinear',
                                            spatial_extent=self.spatial_extent)
            
            if self.cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE:
                feats_bev = concat_egomotion(feats_bev, future_egomotion, self.receptive_field)

        # Temporal model
        states = self.temporal_model(feats_bev)

        # Predict bird's-eye view outputs
        feats_dec = self.decoder(states)

        # Get outputs for available heads using decoder features and 2d features
        output = self.get_head_outputs(feats_dec, feats_2d, images=front_view_image)

        return output
    
    def get_head_outputs(self, feats_dec, feats_2d, images=None, output=None):

        if output is None:
            output = {}

        # Centerline head is used to predict the centerlines as masks
        output_centerline = self.centerline_head.get_head_outputs(feats_dec)
        output.update(output_centerline)

        # 2D head is used to predict the traffic elements
        if self.head2d is not None:
            output['head2d'] = self.head2d.get_head_outputs(images)

        if self.optimal_transport_lclc:
            if self.cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_FILTERING:
                filtered_lc_outputs = self.optimal_transport_lclc.filter_queries(output)
                output["outputs_filtered_lc"] = filtered_lc_outputs
            else:
                outputs_filtered = {}
                for key, value in output.items():
                    outputs_filtered[key] = value
                output["outputs_filtered_lc"] = outputs_filtered
        
        if self.optimal_transport_lcte:
            if self.cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_FILTERING:
                filtered_te_outputs = self.optimal_transport_lcte.filter_queries(output["head2d"])
                output["outputs_filtered_te"] = filtered_te_outputs
            else:
                outputs_filtered = {}
                for key, value in output["head2d"].items():
                    outputs_filtered[key] = value
                output["outputs_filtered_te"] = outputs_filtered
        
        if self.cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_SEPARATE_IMPROVED_RELATIONS_FOR_FINAL_CENTERLINE:
            filtered_lc_outputs_list, filtered_te_outputs_list = self.relation_improver_final_centerline(output)
            if self.cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.IMPROVED_RELATIONS_ENABLE_AUX_LOSS:
                output["outputs_filtered_lc"]["quer_feat_for_final_centerline"] = filtered_lc_outputs_list
                output["outputs_filtered_te"]["quer_feat_for_final_centerline"] = filtered_te_outputs_list
            else:
                output["outputs_filtered_lc"]["quer_feat_for_final_centerline"] = filtered_lc_outputs_list[-1]
                output["outputs_filtered_te"]["quer_feat_for_final_centerline"] = filtered_te_outputs_list[-1]

        if self.cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_IMPROVED_RELATIONS:
            filtered_lc_outputs_list, filtered_te_outputs_list = self.relation_improver(output)
            if self.cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.IMPROVED_RELATIONS_ENABLE_AUX_LOSS:
                output["outputs_filtered_lc"]["quer_feat_relations"] = filtered_lc_outputs_list
                output["outputs_filtered_te"]["quer_feat_relations"] = filtered_te_outputs_list
            else:
                output["outputs_filtered_lc"]["quer_feat_relations"] = filtered_lc_outputs_list[-1]
                output["outputs_filtered_te"]["quer_feat_relations"] = filtered_te_outputs_list[-1]

        if self.optimal_transport_lclc is not None:
            lclc_scores = self.optimal_transport_lclc(output)
            output["lclc_scores"] = lclc_scores

        if self.optimal_transport_lcte is not None:
            lcte_scores = self.optimal_transport_lcte(output)
            output["lcte_scores"] = lcte_scores

        if self.centerline_last:
            output_centerline_after_relation = self.centerline_last(output)
            output["centerline_after_relation"] = output_centerline_after_relation

        return output
    
    
class LiftSplatTemporalMask2FormerTopology(LiftSplatTemporal, LiftSplatMask2FormerTopology):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_heads(self, cfg):
        LiftSplatMask2FormerTopology._init_heads(self, cfg)

    def forward(self, *args, **kwargs):
        return LiftSplatTemporal.forward(self, *args, **kwargs)
    
    def get_head_outputs(self, feats_dec, feats_2d, images=None, output=None):
        return LiftSplatTemporal.get_head_outputs(self, feats_dec, feats_2d, images, output)
    

class LiftSplatLinearMask2FormerTopology(LiftSplatLinear, LiftSplatMask2FormerTopology):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_heads(self, cfg):
        LiftSplatMask2FormerTopology._init_heads(self, cfg)

    def forward(self, *args, **kwargs):
        return LiftSplatTemporal.forward(self, *args, **kwargs)
    
    def get_head_outputs(self, feats_dec, feats_2d, images=None, output=None):
        return LiftSplatTemporal.get_head_outputs(self, feats_dec, feats_2d, images, output)
    

class LiftSplatTemporalMask2FormerTopology(LiftSplatTemporal, LiftSplatMask2FormerTopology):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_heads(self, cfg):
        LiftSplatMask2FormerTopology._init_heads(self, cfg)

    def forward(self, *args, **kwargs):
        return LiftSplatTemporal.forward(self, *args, **kwargs)
    
    def get_head_outputs(self, feats_dec, feats_2d, images=None, output=None):
        return LiftSplatTemporal.get_head_outputs(self, feats_dec, feats_2d, images, output)