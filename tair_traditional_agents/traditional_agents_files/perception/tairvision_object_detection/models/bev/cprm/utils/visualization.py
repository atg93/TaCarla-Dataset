from tairvision.models.bev.lss.utils.visualization import VisualizationModule


class VisualizationModuleCP(VisualizationModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _import_target_functions(self):
        from tairvision.models.bev.common.utils.instance import get_targets_dynamic
        from tairvision.models.bev.lss.utils.static import get_targets_static
        from tairvision.models.bev.lss.utils.bbox import get_targets2d
        from tairvision.models.bev.cprm.utils.bbox import get_targets3d

        from tairvision.models.bev.cprm.utils.bbox import view_boxes_to_lidar_boxes_3d
        from tairvision.models.bev.cprm.utils.bbox import view_boxes_to_bitmap_3d

        self.get_targets_dynamic = get_targets_dynamic
        self.get_targets_static = get_targets_static
        self.get_targets2d = get_targets2d
        self.get_targets3d = get_targets3d
        self.view_boxes_to_lidar_boxes = view_boxes_to_lidar_boxes_3d
        self.view_boxes_to_bitmap = view_boxes_to_bitmap_3d