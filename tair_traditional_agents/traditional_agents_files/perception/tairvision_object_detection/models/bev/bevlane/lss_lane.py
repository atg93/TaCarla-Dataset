from tairvision_object_detection.tairvision_object_detectionmodels.bev.lss.lss import LiftSplatLinear
from tairvision_object_detection.models.bev.bevlane.blocks.static_objects_head import StaticSegmentationHead


class LiftSplatLane(LiftSplatLinear):
    def __init__(self, cfg):
        super(LiftSplatLane, self).__init__(cfg)

    def _init_heads(self, cfg):

        if cfg.MODEL.USE_HEADSTATIC:
            self.static = StaticSegmentationHead(cfg, in_channels=self.temporal_model.out_channels)
        else:
            self.static = None

        self.dynamic = None
        self.head2d = None
        self.head3d = None
