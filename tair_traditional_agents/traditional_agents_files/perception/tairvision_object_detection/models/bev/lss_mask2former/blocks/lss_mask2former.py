from tairvision.models.bev.lss_mask2former.blocks.heads import Mask2FormerDynamicHead
from tairvision.models.bev.lss.lss import LiftSplat, LiftSplatLinear, LiftSplatTemporal
from tairvision.models.detection.dab_detr import DeformableDAB, dab_timm_fpn


class LiftSplatMask2Former(LiftSplat):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_dynamic_head(self, cfg):
        return Mask2FormerDynamicHead(
            cfg,
            n_classes=len(cfg.MODEL.HEADDYNAMIC.SEMANTIC_SEG.WEIGHTS),
            in_channels=self.temporal_model.out_channels,
            in_channels_list=self.decoder.backbone.in_channels_list,
            strides=self.decoder.backbone.strides
        )
    
    def _init_2d_head(self, cfg):
        kwargs = {}
        transformer_config = cfg.MODEL.HEAD2D.TRANSFORMER.convert_to_dict()
        transformer_config = {k.lower(): v for k, v in transformer_config.items()}
        kwargs["transformer"] = transformer_config

        dab_params_config = cfg.MODEL.HEAD2D.DAB_PARAMS.convert_to_dict()
        dab_params_config = {k.lower(): v for k, v in dab_params_config.items()}
        kwargs["dab_params"] = dab_params_config

        matcher_config = cfg.MODEL.HEAD2D.MATCHER_CONFIG.convert_to_dict()
        matcher_config = {k.lower(): v for k, v in matcher_config.items()}
        kwargs["matcher"] = {'name': 'HungarianMatcher', 'config': matcher_config}

        loss_weight_config = cfg.MODEL.HEAD2D.LOSS_WEIGHT.convert_to_dict()
        loss_weight_config = {k.lower(): v for k, v in loss_weight_config.items()}
        kwargs["loss_weight"] = loss_weight_config

        kwargs["pretrained"] = cfg.MODEL.HEAD2D.PRETRAINED
        kwargs["pyramid_type"] = cfg.MODEL.HEAD2D.PYRAMID_TYPE

        pe_params_config = cfg.MODEL.HEAD2D.PE_PARAMS.convert_to_dict()
        pe_params_config = {k.lower(): v for k, v in pe_params_config.items()}
        kwargs["pe_params"] = pe_params_config
        
        if cfg.MODEL.HEAD2D.SHARED_ENCODER:    
            head_2d = DeformableDAB(backbone = self.encoder.backbone, num_classes=13, **kwargs)
        else:
            kwargs["dab_params"]["features_selected_keys"] = ['1', '2', '3']
            head_2d = dab_timm_fpn(type=cfg.MODEL.HEAD2D.BACKBONE, num_classes=13, **kwargs)
            
        return head_2d
    
    def _init_heads(self, cfg):
        if cfg.MODEL.USE_HEADDYNAMIC:
            self.dynamic = self._init_dynamic_head(cfg)
        else:
            self.dynamic = None

        if cfg.MODEL.USE_HEADSTATIC:
            raise NotImplementedError
        else:
            self.static = None

        if cfg.MODEL.USE_HEAD2D:
            self.head2d = self._init_2d_head(cfg)
        else:
            self.head2d = None

        if cfg.MODEL.USE_HEAD3D:
            raise NotImplementedError
        else:
            self.head3d = None


class LiftSplatLinearMask2Former(LiftSplatLinear, LiftSplatMask2Former):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_dynamic_head(self, cfg):
        return LiftSplatMask2Former._init_dynamic_head(self, cfg)
    
    def _init_2d_head(self, cfg):
        return LiftSplatMask2Former._init_2d_head(self, cfg)
    
    def _init_heads(self, cfg):
        LiftSplatMask2Former._init_heads(self, cfg)


class LiftSplatTemporalMask2Former(LiftSplatTemporal, LiftSplatMask2Former):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_dynamic_head(self, cfg):
        return LiftSplatMask2Former._init_dynamic_head(self, cfg)
    
    def _init_2d_head(self, cfg):
        return LiftSplatMask2Former._init_2d_head(self, cfg)
    
    def _init_heads(self, cfg):
        LiftSplatMask2Former._init_heads(self, cfg)
