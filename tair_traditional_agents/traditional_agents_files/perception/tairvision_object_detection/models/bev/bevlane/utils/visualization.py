from tairvision.models.bev.lss.utils.visualization import VisualizationModule, pad_images, get_bitmap
from tairvision.models.bev.common.utils.geometry import points_lidar2view
import numpy as np
import cv2
import torch


class VisualizationModuleLane(VisualizationModule):
    def __init__(self, cfg, score_threshold_2d=0.50, score_threshold_3d=0.40):
        super().__init__(cfg, score_threshold_2d, score_threshold_3d)

    def _import_target_functions(self):
        super()._import_target_functions()
        from tairvision.models.bev.bevlane.utils.line import get_targets_line
        self.get_targets_line = get_targets_line

    def get_bev_maps(self, batch, output, targets, scale=3):
        _, map_pred, map_gt = super().get_bev_maps(batch, output, targets)
        map_pred = cv2.resize(map_pred, (map_pred.shape[1] * scale, map_pred.shape[0] * scale))
        map_gt = cv2.resize(map_gt, (map_gt.shape[1] * scale, map_gt.shape[0] * scale))
        line_map_gt = self.plot_line_instances(targets, batch['view'], bev_size=self.bev_size, bev_scale=scale)
        line_map_pred = self.plot_line_instances(output, batch['view'], bev_size=self.bev_size, bev_scale=scale)
        map_gt = np.concatenate([map_gt, pad_images(line_map_gt, padding_size=2 * scale, ones=True)], axis=1)
        map_pred = np.concatenate([map_pred, pad_images(line_map_pred, padding_size=2 * scale, ones=True)], axis=1)
        map_all = np.concatenate([map_pred, map_gt], axis=0)
        # cv2.imshow('map_all', map_all)
        # cv2.waitKey()
        return map_all, map_pred, map_gt

    def plot_all(self, batch, output, show):

        input_to_show = []
        output_to_show = []
        for camera in range(self.nb_cameras):
            image = batch['images'][0, self.receptive_field - 1, camera].cpu()
            image = self.denormalize(image)
            image = (image * 255).to(torch.uint8)

            self.append_camera_images(input_to_show, image, camera)

        h, w, c = input_to_show[0].shape
        input_to_show = np.stack(input_to_show, axis=1)
        input_to_show = input_to_show.reshape((h, 2, 3, w, c)).swapaxes(0, 1).reshape(h * 2, w * 3, c)
        cv2.imshow("Input", cv2.cvtColor(input_to_show, cv2.COLOR_RGB2BGR))

        # TODO: implement camera projection in future
        # output_to_show = np.stack(output_to_show, axis=1)
        # output_to_show = output_to_show.reshape((h, 2, 3, w, c)).swapaxes(0, 1).reshape(h * 2, w * 3, c)
        # cv2.imshow("Camera projection", cv2.cvtColor(output_to_show, cv2.COLOR_RGB2BGR))

        # visualizing bev maps for segmentation and detection
        targets = self.get_targets_dynamic(batch,
                                           receptive_field=self.receptive_field,
                                           spatial_extent=self.spatial_extent,
                                           )
        targets_line = self.get_targets_line(batch, receptive_field=self.receptive_field)
        targets.update(targets_line)

        bev_map, _, _ = self.get_bev_maps(batch, output, targets)

        nb_maps = bev_map.shape[1] / bev_map.shape[0]
        bev_map = cv2.resize(bev_map, dsize=(int(nb_maps * 600), 600), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("BEV - Predictions vs. GTs", cv2.cvtColor(bev_map, cv2.COLOR_RGB2BGR))

    @staticmethod
    def plot_line_instances(line_head, view,
                  bev_size=(200, 200),
                  bev_scale=1
                  ):
        bev_size = tuple(np.array(bev_size) * bev_scale)
        line_instances = line_head['line_instances'][0][0]
        line_map = get_line_instances_map(line_instances, view[0, 0, 0].cpu().numpy(), bev_size=bev_size, scale=bev_scale)
        line_map = get_bitmap(line_map, bev_size=bev_size)

        return line_map


def get_line_instances_map(line_instances, view, bev_size=(200, 200), thickness=1, scale=1):
    polygons = points_lidar2view(line_instances, view)
    polygons = [p * scale for p in polygons]

    output = np.zeros(bev_size).astype(np.uint8)
    for i, line_instance in enumerate(polygons):
        cv2.polylines(output, [line_instance], False, i + 1, thickness=thickness)
        for point in line_instance:
            output = cv2.circle(output, point, 2, i + 1, -1)

    return output
