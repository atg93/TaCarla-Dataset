from tairvision.models.bev.lss.utils.visualization import VisualizationModule, get_bitmap_with_road
import numpy as np

class VisualizationModuleTransformer(VisualizationModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def plot_segm(dynamic_head, lanes, lines,
                  gt=False,
                  bev_size=(200, 200),
                  nb_z_bins=8,
                  ):

        segm = dynamic_head['segmentation'] if gt else dynamic_head['segm']
        inst = dynamic_head['instance'] if gt else dynamic_head['inst']

        segm = segm[0, :, 0].cpu().numpy()
        inst = inst[0, :, 0].cpu().numpy()

        segm = get_bitmap_with_road(segm, lanes, lines, bev_size=bev_size)
        inst = get_bitmap_with_road(inst, lanes, lines, bev_size=bev_size)

        center = np.zeros_like(segm)
        zpos = np.zeros_like(segm)

        return segm, inst, center, zpos