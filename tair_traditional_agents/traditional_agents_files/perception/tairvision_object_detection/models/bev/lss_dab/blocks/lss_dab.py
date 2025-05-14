from tairvision.models.bev.lss_dab.blocks.heads import DABDynamicHead
from tairvision.models.bev.lss_mask2former.blocks.lss_mask2former import LiftSplatMask2Former, LiftSplatLinearMask2Former, LiftSplatTemporalMask2Former


class LiftSplatDAB(LiftSplatMask2Former):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_dynamic_head(self, cfg):
        return DABDynamicHead(
            cfg,
            n_classes=len(cfg.MODEL.HEADDYNAMIC.SEMANTIC_SEG.WEIGHTS),
            in_channels=self.temporal_model.out_channels,
            in_channels_list=self.decoder.backbone.in_channels_list,
            strides=self.decoder.backbone.strides
        )


class LiftSplatLinearDAB(LiftSplatLinearMask2Former, LiftSplatDAB):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_dynamic_head(self, cfg):
        return LiftSplatDAB._init_dynamic_head(self, cfg)
    

class LiftSplatTemporalDAB(LiftSplatTemporalMask2Former, LiftSplatDAB):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_dynamic_head(self, cfg):
        return LiftSplatDAB._init_dynamic_head(self, cfg)
