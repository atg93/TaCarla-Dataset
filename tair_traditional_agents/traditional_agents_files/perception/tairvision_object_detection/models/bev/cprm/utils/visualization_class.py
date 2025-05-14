import numpy as np
import torch
import matplotlib.pylab
import cv2
from typing import Union, Optional, List, Tuple
from PIL import ImageColor, ImageFont, ImageDraw, Image
from torchvision import transforms as T
from tairvision.models.bev.lss.utils.instance import predict_instance_segmentation_and_trajectories

from tairvision.models.bev.lss.training.process import decide_mean_std
from tairvision.models.bev.lss.utils.network import NormalizeInverse
from tairvision.models.bev.lss.utils.bbox import (box2corners, get_targets3d_xdyd, view_boxes_to_lidar_boxes_xdyd,
                                                  lidar_boxes_to_cam_boxes, view_boxes_to_bitmap_xdyd, get_targets2d)
from tairvision.models.bev.common.utils.instance import instance_to_center_offset_flow
from tairvision.models.bev.lss.utils.static import get_targets_static

DEFAULT_COLORMAP = matplotlib.pylab.cm.hot


def flow_to_image(flow: np.ndarray, autoscale: bool = False) -> np.ndarray:
    """
    Applies colour map to flow which should be a 2 channel image tensor HxWx2. Returns a HxWx3 numpy image
    Code adapted from: https://github.com/liruoteng/FlowNet/blob/master/models/flownet/scripts/flowlib.py
    """
    u = flow[0, :, :]
    v = flow[1, :, :]

    # Convert to polar coordinates
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = np.max(rad)

    # Normalise flow maps
    if autoscale:
        u /= maxrad + np.finfo(float).eps
        v /= maxrad + np.finfo(float).eps

    # Visualise flow with cmap
    return np.uint8(compute_color(u, v) * 255)


def _normalise(image: np.ndarray) -> np.ndarray:
    lower = np.min(image)
    delta = np.max(image) - lower
    if delta == 0:
        delta = 1
    image = (image.astype(np.float32) - lower) / delta
    return image


def apply_colour_map(
        image: np.ndarray, cmap: matplotlib.colors.LinearSegmentedColormap = DEFAULT_COLORMAP, autoscale: bool = False
) -> np.ndarray:
    """
    Applies a colour map to the given 1 or 2 channel numpy image. if 2 channel, must be 2xHxW.
    Returns a HxWx3 numpy image
    """
    if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):
        if image.ndim == 3:
            image = image[0]
        # Grayscale scalar image
        if autoscale:
            image = _normalise(image)
        return cmap(image)[:, :, :3]
    if image.shape[0] == 2:
        # 2 dimensional UV
        return flow_to_image(image, autoscale=autoscale)
    if image.shape[0] == 3:
        # Normalise rgb channels
        if autoscale:
            image = _normalise(image)
        return np.transpose(image, axes=[1, 2, 0])
    raise Exception('Image must be 1, 2 or 3 channel to convert to colour_map (CxHxW)')


def heatmap_image(
        image: np.ndarray, cmap: matplotlib.colors.LinearSegmentedColormap = DEFAULT_COLORMAP, autoscale: bool = True
) -> np.ndarray:
    """Colorize an 1 or 2 channel image with a colourmap."""
    if not issubclass(image.dtype.type, np.floating):
        raise ValueError(f"Expected a ndarray of float type, but got dtype {image.dtype}")
    if not (image.ndim == 2 or (image.ndim == 3 and image.shape[0] in [1, 2])):
        raise ValueError(f"Expected a ndarray of shape [H, W] or [1, H, W] or [2, H, W], but got shape {image.shape}")
    heatmap_np = apply_colour_map(image, cmap=cmap, autoscale=autoscale)
    heatmap_np = np.uint8(heatmap_np * 255)
    return heatmap_np


def compute_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    assert u.shape == v.shape
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nan_mask = np.isnan(u) | np.isnan(v)
    u[nan_mask] = 0
    v[nan_mask] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    f_k = (a + 1) / 2 * (ncols - 1) + 1
    k_0 = np.floor(f_k).astype(int)
    k_1 = k_0 + 1
    k_1[k_1 == ncols + 1] = 1
    f = f_k - k_0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k_0 - 1] / 255
        col1 = tmp[k_1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = col * (1 - nan_mask)

    return img


def make_color_wheel() -> np.ndarray:
    """
    Create colour wheel.
    Code adapted from https://github.com/liruoteng/FlowNet/blob/master/models/flownet/scripts/flowlib.py
    """
    red_yellow = 15
    yellow_green = 6
    green_cyan = 4
    cyan_blue = 11
    blue_magenta = 13
    magenta_red = 6

    ncols = red_yellow + yellow_green + green_cyan + cyan_blue + blue_magenta + magenta_red
    colorwheel = np.zeros([ncols, 3])

    col = 0

    # red_yellow
    colorwheel[0:red_yellow, 0] = 255
    colorwheel[0:red_yellow, 1] = np.transpose(np.floor(255 * np.arange(0, red_yellow) / red_yellow))
    col += red_yellow

    # yellow_green
    colorwheel[col: col + yellow_green, 0] = 255 - np.transpose(
        np.floor(255 * np.arange(0, yellow_green) / yellow_green)
    )
    colorwheel[col: col + yellow_green, 1] = 255
    col += yellow_green

    # green_cyan
    colorwheel[col: col + green_cyan, 1] = 255
    colorwheel[col: col + green_cyan, 2] = np.transpose(np.floor(255 * np.arange(0, green_cyan) / green_cyan))
    col += green_cyan

    # cyan_blue
    colorwheel[col: col + cyan_blue, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, cyan_blue) / cyan_blue))
    colorwheel[col: col + cyan_blue, 2] = 255
    col += cyan_blue

    # blue_magenta
    colorwheel[col: col + blue_magenta, 2] = 255
    colorwheel[col: col + blue_magenta, 0] = np.transpose(np.floor(255 * np.arange(0, blue_magenta) / blue_magenta))
    col += +blue_magenta

    # magenta_red
    colorwheel[col: col + magenta_red, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, magenta_red) / magenta_red))
    colorwheel[col: col + magenta_red, 0] = 255

    return colorwheel


def make_contour(img, colour=[0, 0, 0], double_line=False):
    h, w = img.shape[:2]
    out = img.copy()
    # Vertical lines
    out[np.arange(h), np.repeat(0, h)] = colour
    out[np.arange(h), np.repeat(w - 1, h)] = colour

    # Horizontal lines
    out[np.repeat(0, w), np.arange(w)] = colour
    out[np.repeat(h - 1, w), np.arange(w)] = colour

    if double_line:
        out[np.arange(h), np.repeat(1, h)] = colour
        out[np.arange(h), np.repeat(w - 2, h)] = colour

        # Horizontal lines
        out[np.repeat(1, w), np.arange(w)] = colour
        out[np.repeat(h - 2, w), np.arange(w)] = colour
    return out


def plot_instance_map(instance_image, instance_map, instance_colours=None, bg_image=None):
    if isinstance(instance_image, torch.Tensor):
        instance_image = instance_image.cpu().numpy()
    assert isinstance(instance_image, np.ndarray)
    if instance_colours is None:
        instance_colours = generate_instance_colours(instance_map)
    if len(instance_image.shape) > 2:
        instance_image = instance_image.reshape((instance_image.shape[-2], instance_image.shape[-1]))

    if bg_image is None:
        plot_image = 255 * np.ones((instance_image.shape[0], instance_image.shape[1], 3), dtype=np.uint8)
    else:
        plot_image = bg_image

    for key, value in instance_colours.items():
        plot_image[instance_image == key] = value

    return plot_image


def visualise_output(labels, output):
    semantic_colours = np.array([[255, 255, 255], [0, 0, 0]], dtype=np.uint8)

    consistent_instance_seg = predict_instance_segmentation_and_trajectories(
        output, compute_matched_centers=False
    )

    sequence_length = consistent_instance_seg.shape[1]
    b = 0
    video = []
    for t in range(sequence_length):
        out_t = []
        # Ground truth
        unique_ids = torch.unique(labels['instance'][b, t]).cpu().numpy()[1:]
        instance_map = dict(zip(unique_ids, unique_ids))
        instance_plot = plot_instance_map(labels['instance'][b, t].cpu(), instance_map)[::-1, ::-1]
        instance_plot = make_contour(instance_plot)

        semantic_seg = labels['segmentation'].squeeze(2).cpu().numpy()
        semantic_plot = semantic_colours[semantic_seg[b, t][::-1, ::-1]]
        semantic_plot = make_contour(semantic_plot)

        if 'flow' in labels.keys():
            future_flow_plot = labels['flow'][b, t].cpu().numpy()
            future_flow_plot[:, semantic_seg[b, t] != 1] = 0
            future_flow_plot = flow_to_image(future_flow_plot)[::-1, ::-1]
            future_flow_plot = make_contour(future_flow_plot)
        else:
            future_flow_plot = np.zeros_like(semantic_plot)

        center_plot = heatmap_image(labels['centerness'][b, t, 0].cpu().numpy())[::-1, ::-1]
        center_plot = make_contour(center_plot)

        offset_plot = labels['offset'][b, t].cpu().numpy()
        offset_plot[:, semantic_seg[b, t] != 1] = 0
        offset_plot = flow_to_image(offset_plot)[::-1, ::-1]
        offset_plot = make_contour(offset_plot)

        out_t.append(np.concatenate([instance_plot, future_flow_plot,
                                     semantic_plot, center_plot, offset_plot], axis=0))

        # Predictions
        unique_ids = torch.unique(consistent_instance_seg[b, t]).cpu().numpy()[1:]
        instance_map = dict(zip(unique_ids, unique_ids))
        instance_plot = plot_instance_map(consistent_instance_seg[b, t].cpu(), instance_map)[::-1, ::-1]
        instance_plot = make_contour(instance_plot)

        semantic_seg = output['segm'].argmax(dim=2).detach().cpu().numpy()
        semantic_plot = semantic_colours[semantic_seg[b, t][::-1, ::-1]]
        semantic_plot = make_contour(semantic_plot)

        if 'flow' in labels.keys():
            future_flow_plot = output['flow'][b, t].detach().cpu().numpy()
            future_flow_plot[:, semantic_seg[b, t] != 1] = 0
            future_flow_plot = flow_to_image(future_flow_plot)[::-1, ::-1]
            future_flow_plot = make_contour(future_flow_plot)
        else:
            future_flow_plot = np.zeros_like(semantic_plot)

        center_plot = heatmap_image(output['center'][b, t, 0].detach().cpu().numpy())[::-1, ::-1]
        center_plot = make_contour(center_plot)

        offset_plot = output['offset'][b, t].detach().cpu().numpy()
        offset_plot[:, semantic_seg[b, t] != 1] = 0
        offset_plot = flow_to_image(offset_plot)[::-1, ::-1]
        offset_plot = make_contour(offset_plot)

        out_t.append(np.concatenate([instance_plot, future_flow_plot,
                                     semantic_plot, center_plot, offset_plot], axis=0))
        out_t = np.concatenate(out_t, axis=1)
        # Shape (C, H, W)
        out_t = out_t.transpose((2, 0, 1))

        video.append(out_t)

    # Shape (B, T, C, H, W)
    video = np.stack(video)[None]
    return video


def convert_figure_numpy(figure):
    """ Convert figure to numpy image """
    figure_np = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    figure_np = figure_np.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    return figure_np


INSTANCE_COLOURS = np.asarray([
    [0, 0, 0],
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
])


def generate_instance_colours(instance_map):
    # Most distinct 22 colors (kelly colors from https://stackoverflow.com/questions/470690/how-to-automatically-generate
    # -n-distinct-colors)
    # plus some colours from AD40k

    return {instance_id: INSTANCE_COLOURS[global_instance_id % len(INSTANCE_COLOURS)] for
            instance_id, global_instance_id in instance_map.items()
            }


class VisualizationModule:
    def __init__(self, cfg, filter_classes=None, show=None):
        self.batch = None
        self.cfg = cfg
        self.filter_classes = filter_classes
        self.show = show
        self.bev_size = (int((self.cfg.LIFT.X_BOUND[1] - self.cfg.LIFT.X_BOUND[0]) / self.cfg.LIFT.X_BOUND[2]),
                         int((self.cfg.LIFT.Y_BOUND[1] - self.cfg.LIFT.Y_BOUND[0]) / self.cfg.LIFT.Y_BOUND[2])
                         )
        self.nb_z_bins = (self.cfg.LIFT.Z_BOUND[1] - self.cfg.LIFT.Z_BOUND[0]) / self.cfg.LIFT.Z_BOUND[2]
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD
        self.final_size = (self.cfg.IMAGE.FINAL_DIM[1], self.cfg.IMAGE.FINAL_DIM[0])

        self.bev_conditions = [(0, 100, 0, 120), (5, 190, 0, 100), (100, 200, 0, 120),
                               (0, 100, 80, 200), (5, 190, 100, 200), (100, 200, 80, 200)]

        self.mean, self.std = decide_mean_std(self.cfg.PRETRAINED.LOAD_WEIGHTS)
        self.st_3d, self.st_2d = self.set_score_thresholds()

    @staticmethod
    def set_score_thresholds():
        return 0.40, 0.50

    def plot_preds_and_gts(self, batch, output):
        denormalize = NormalizeInverse(mean=self.mean, std=self.std)
        self.batch = batch
        input_to_show = []
        output_to_show = []
        for camera in range(6):
            image = self.batch['images'][0, self.receptive_field - 1, camera].cpu()
            image = denormalize(image)
            image = (image * 255).to(torch.uint8)

            self.append_camera_images(input_to_show, image, camera)

            if self.show['2d_pred']:
                image = self.plot_2d_boxes_on_image(image, camera,
                                                    output=output,
                                                    score_threshold=self.st_2d)

            if self.show['3d_pred']:
                image = self.plot_view_boxes_on_image(image, camera,
                                                      output=output,
                                                      score_threshold=self.st_3d)

            if self.show['zpos_pred']:
                image = self.plot_zpos_on_image(image, camera,
                                                output=output,
                                                bev_conditions=self.bev_conditions[camera])

            if self.show['gt']:
                image = self.plot_cam_boxes_on_image(image, camera)

                image = self.plot_zpos_on_image(image, camera,
                                                output=None,
                                                bev_conditions=self.bev_conditions[camera])

            if self.show['lanes_gt_on_cam']:
                image = self.plot_lanes_on_image(image, camera)

            self.append_camera_images(output_to_show, image, camera)

        h, w, c = input_to_show[0].shape
        input_to_show = np.stack(input_to_show, axis=1)
        input_to_show = input_to_show.reshape((h, 2, 3, w, c)).swapaxes(0, 1).reshape(h * 2, w * 3, c)
        # cv2.imshow("Input", cv2.cvtColor(input_to_show, cv2.COLOR_RGB2BGR))

        output_to_show = np.stack(output_to_show, axis=1)
        output_to_show = output_to_show.reshape((h, 2, 3, w, c)).swapaxes(0, 1).reshape(h * 2, w * 3, c)
        cv2.imshow("Camera projection", cv2.cvtColor(output_to_show, cv2.COLOR_RGB2BGR))

        # visualizing bev maps for segmentation and detection
        bev_map = self.get_bev_maps(output,
                                    spatial_extent=(self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1]),
                                    ignore_index=self.cfg.DATASET.IGNORE_INDEX)

        nb_maps = bev_map.shape[1] / bev_map.shape[0]
        bev_map = cv2.resize(bev_map, dsize=(int(nb_maps * 600), 600), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("BEV - Predictions vs. GTs", cv2.cvtColor(bev_map, cv2.COLOR_RGB2BGR))

    def plot_view_boxes_on_image(self, input_image, cam_index,
                                 output=None,
                                 score_threshold=0.40):
        if output is not None and 'head3d' in output.keys():
            det3d = output['head3d'][0]
        else:
            targets3d = get_targets3d_xdyd(self.batch, receptive_field=self.receptive_field)[0]
            det3d = targets3d[0]
            print('Visualizing Lidar Boxes using GTs')

        temp_index = self.receptive_field - 1
        intrinsics = self.batch['intrinsics'][0, temp_index, cam_index].cpu()
        extrinsics = self.batch['cams_to_lidar'][0, temp_index, cam_index].cpu()

        lidar_boxes = view_boxes_to_lidar_boxes_xdyd(det3d, self.batch, self.filter_classes,
                                                     score_threshold=score_threshold,
                                                     t=temp_index)
        cam_boxes = lidar_boxes_to_cam_boxes(lidar_boxes, extrinsics)
        boxes, cuboids, categories, _ = box2corners(cam_boxes, intrinsics, self.final_size)
        boxes = torch.from_numpy(np.asarray(boxes))
        output_image = self.draw_cuboids(input_image, boxes, cuboids, categories,
                                         font_size=12,
                                         outline=(255, 128, 0, 50),
                                         bbox_outline=(255, 255, 0, 50))

        return output_image

    def plot_cam_boxes_on_image(self, input_image, cam_index):

        temp_index = self.receptive_field - 1
        intrinsics = self.batch['intrinsics'][0, temp_index, cam_index].cpu()

        boxes = self.batch['boxes'][0][temp_index][cam_index]
        boxes, cuboids, categories, _ = box2corners(boxes, intrinsics, self.final_size)
        boxes = torch.from_numpy(np.asarray(boxes))
        output_image = self.draw_cuboids(input_image, boxes, cuboids, categories, font_size=12)

        return output_image

    def plot_2d_boxes_on_image(self, input_image, cam_index,
                               output=None,
                               score_threshold=0.50
                               ):
        if output is not None and 'head2d' in output.keys():
            det2d = output['head2d'][cam_index]
            score_mask = output['head2d'][cam_index]['scores'].cpu() > score_threshold
        else:
            targets2d = get_targets2d(self.batch, receptive_field=self.receptive_field, image_size=self.final_size)[0]
            det2d = targets2d[cam_index]
            score_mask = torch.tensor([True for _ in det2d['boxes'].cpu()], dtype=torch.bool)
            print('Visualizing 2D Bounding Boxes using GTs')

        boxes2d = det2d['boxes'].cpu()[score_mask]
        labels2d = det2d['labels'].cpu()[score_mask]
        labels2d = [self.filter_classes.classes[i.item()] for i in labels2d]

        output_image = self.draw_cuboids(input_image, boxes2d, [], labels2d,
                                         font_size=12,
                                         bbox_outline=(255, 0, 0, 50))

        return output_image

    def plot_zpos_on_image(self, input_image, cam_index,
                           output=None,
                           bev_conditions=(5, 190, 0, 100),
                           ):

        temp_index = self.receptive_field - 1
        intrinsics = self.batch['intrinsics'][0, temp_index, cam_index].cpu()
        extrinsics = self.batch['cams_to_lidar'][0, temp_index, cam_index].cpu()
        view = self.batch['view'][0, temp_index, 0].cpu()

        if output is not None and 'zpos' in output.keys():
            z_position = output['zpos'][0, temp_index, 0:2].cpu()
            inst_pred = output['inst'][0, temp_index, 0].cpu()
        else:
            z_position = self.batch['z_position'][0, temp_index, 0:2].cpu()
            inst_pred = self.batch['instance'][0, temp_index, 0].cpu()
            print('Visualizing z-positions using GTs')

        output_image = self.get_bev_to_image_points(input_image, extrinsics, intrinsics, view, z_position, inst_pred,
                                                    bev_conditions=bev_conditions,
                                                    )

        return output_image

    def plot_lanes_on_image(self, input_image, cam_index):

        temp_index = self.receptive_field - 1
        lines_cam = self.batch['lines_cam'][0, temp_index, cam_index].cpu()
        lines_cam = (lines_cam * 255).to(torch.uint8).numpy()
        lines_cam = lines_cam[0] + lines_cam[1]
        input_image[:, lines_cam > 0] = torch.ones((3, 1), dtype=torch.uint8) * 255

        return input_image

    def get_bev_maps(self, output,
                     spatial_extent=(50., 50.),
                     ignore_index=255, batch=None
                     ):

        if batch is not None:
            self.batch = batch
        static_gt = get_targets_static(self.batch, self.receptive_field)
        lanes_gt = static_gt['lanes'][0, 0].cpu().numpy()
        lines_gt = static_gt['lines'][0, 0].cpu().numpy()

        if 'lanes' in output.keys():
            lanes = output['lanes'][0, 0].cpu().numpy()
            lines = output['lines'][0, 0].cpu().numpy()
        else:
            lanes = lanes_gt
            lines = lines_gt

        segm_gt, inst_gt, center_gt, zpos_gt = self.plot_segm(lanes_gt, lines_gt,
                                                              gt=True,
                                                              temp_index=self.receptive_field - 1,
                                                              spatial_extent=spatial_extent,
                                                              ignore_index=ignore_index,
                                                              )
        map_gt = [segm_gt, inst_gt, center_gt, zpos_gt]
        map_gt = [self.pad_images(i, padding_size=2, ones=True) for i in map_gt]

        if 'segm' in output.keys():
            segm_pred, inst_pred, center_pred, zpos_pred = self.plot_segm(output, lanes, lines)
            map_pred = [segm_pred, inst_pred, center_pred, zpos_pred]
        else:
            map_pred = [np.zeros_like(segm_gt)] * 4
        map_pred = [self.pad_images(i, padding_size=2, ones=True) for i in map_pred]

        targets3d, _ = get_targets3d_xdyd(self.batch, receptive_field=self.receptive_field, spatial_extent=spatial_extent)
        view_boxes_gt = self.plot_view_boxes(targets3d[0:1], lanes_gt, lines_gt)
        map_gt.append(self.pad_images(view_boxes_gt, padding_size=2, ones=True))

        if 'head3d' in output.keys():
            view_boxes_pred = self.plot_view_boxes(output['head3d'][0:1], lanes, lines, score_threshold=self.st_3d)
        else:
            view_boxes_pred = np.zeros_like(view_boxes_gt)
        map_pred.append(self.pad_images(view_boxes_pred, padding_size=2, ones=True))

        map_pred = np.concatenate(map_pred, axis=1)
        map_gt = np.concatenate(map_gt, axis=1)
        map_all = np.concatenate([map_pred, map_gt], axis=0)

        return map_all

    def plot_segm(self, dynamic_head, lanes, lines,
                  gt=False,
                  temp_index=0,
                  spatial_extent=(50., 50.),
                  ignore_index=255
                  ):

        if gt:
            segm = dynamic_head['segmentation']
            inst = dynamic_head['instance']
            center, _, _ = instance_to_center_offset_flow(dynamic_head['instance'],
                                                          dynamic_head['future_egomotion'],
                                                          ignore_index=ignore_index,
                                                          subtract_egomotion=True,
                                                          spatial_extent=spatial_extent,
                                                          )
            zpos = dynamic_head['z_position']
        else:
            segm = dynamic_head['segm']
            inst = dynamic_head['inst']
            center = dynamic_head['center']
            zpos = dynamic_head['zpos']

        segm = segm[0, temp_index, 0].cpu().numpy()
        inst = inst[0, temp_index, 0].cpu().numpy()
        center = center[0, temp_index, 0].cpu().numpy()
        zpos = zpos[0, temp_index, 0].cpu().numpy() * (256 / self.nb_z_bins)

        center = heatmap_image(center)
        zpos = zpos[:, :, None].repeat(3, 2).astype(np.uint8)

        mask = segm > 0
        segm = self.get_bitmap_with_road(segm, lanes, lines, x_mask=mask)
        inst = self.get_bitmap_with_road(inst, lanes, lines, x_mask=mask)
        center = self.get_bitmap_with_road(center, lanes, lines, x_mask=mask,
                                           color_coef=[1.0, 1.0, 1.0],
                                           )
        zpos = self.get_bitmap_with_road(zpos, lanes, lines, x_mask=mask,
                                         color_coef=[1.0, 0.77, 0.11],
                                         )

        return segm, inst, center, zpos

    def plot_view_boxes(self, det3d, lanes, lines, score_threshold=0.40):

        view_boxes = view_boxes_to_bitmap_xdyd(det3d, self.bev_size, score_threshold=score_threshold)
        view_boxes = self.get_bitmap_with_road(view_boxes, lanes, lines)

        return view_boxes

    def append_camera_images(self, input_list, image, cam_index):

        image_to_append = np.array(T.ToPILImage()(image))
        if cam_index > 2:
            image_to_append = image_to_append[:, ::-1]
        image_to_append = self.pad_images(image_to_append)

        input_list.append(image_to_append)

    def get_bitmap(self, x):
        output = np.zeros((*self.bev_size, 3)).astype(np.uint8)
        mask = x > 0
        colors = INSTANCE_COLOURS[x[mask] % len(INSTANCE_COLOURS)]
        output[mask] = colors  # [255, 172, 28]
        output[95:105, 97:103] = [52, 152, 219]

        return output

    def get_bitmap_with_road(self, x, lanes, lines, x_mask=None, color_coef=None):
        output = np.zeros((*self.bev_size, 3)).astype(np.uint8)

        if x_mask is None:
            x_mask = x > 0
        lane_mask = (lanes > 0).any(0)
        line_mask = (lines > 0).any(0)

        output[lane_mask] = [75, 75, 75]
        output[line_mask] = [255, 255, 255]

        if color_coef is None:
            output[x_mask] = INSTANCE_COLOURS[x[x_mask] % len(INSTANCE_COLOURS)]
        else:
            output[x_mask] = (x[x_mask] * color_coef).astype(np.uint8)
        output[95:105, 97:103] = [52, 152, 219]

        return output

    def plot_future_preds_and_gts(self, batch, output):
        self.batch = batch
        t = self.receptive_field - 1  # Index of the present frame
        targets3d, valid_index = self.get_targets3d_xdyd(self.batch, receptive_field=self.receptive_field)
        targets2d, valid_index2d = self.get_targets2d(self.batch, receptive_field=self.receptive_field,
                                                      image_size=self.final_size)
        valid_indx = 0

        mean, std = decide_mean_std(self.cfg.PRETRAINED.LOAD_WEIGHTS)
        denormalize = NormalizeInverse(mean=mean, std=std)

        input_to_show = []
        box_cuboids_to_show = []
        for camera in range(6):
            image = self.batch['images'][0][t][camera].cpu()
            image = denormalize(image)
            intrinsics = self.batch['intrinsics'][0, t, camera].cpu()
            extrinsics = self.batch['cams_to_lidar'][0, t, camera].cpu()
            view = self.batch['view'][0, t, 0].cpu()
            image = (image.permute(1, 2, 0) * 255).to(torch.uint8)

            if camera < 3:
                input_to_show.append(image.numpy())
            else:
                input_to_show.append(image.numpy()[:, ::-1])

            boxes_gt = self.batch['boxes'][0][t][camera]
            boxes_gt, cuboids_gt, categories_gt, _ = box2corners(boxes_gt, intrinsics, self.final_size)
            boxes_gt = torch.from_numpy(np.asarray(boxes_gt))

            # 2d boxes
            if self.show['2d_pred']:
                if 'head2d' in output.keys():
                    scores = output['head2d'][camera]['scores'].cpu()
                    score_mask = scores > 0.50
                    boxes2d_pred = output['head2d'][camera]['boxes'].cpu()[score_mask]
                    labels2d_pred = output['head2d'][camera]['labels'].cpu()[score_mask]
                else:
                    if valid_index2d[camera]:
                        boxes2d_pred = targets2d[valid_indx]['boxes'].cpu()
                        valid_indx += 1
                    else:
                        boxes2d_pred = []
                    print('Visualizing 2D Bounding Boxes using GTs')
                boxes2d_pred = torch.from_numpy(np.asarray(boxes2d_pred))
                box_cuboids = self.draw_cuboids(image.permute(2, 0, 1), boxes2d_pred, [], None, font_size=12,
                                                bbox_outline=(255, 0, 0, 50))
            else:
                box_cuboids = image.permute(2, 0, 1)

            if self.show['3d_pred']:
                if 'head3d' in output.keys():
                    lidar_boxes = view_boxes_to_lidar_boxes_xdyd(output['head3d'][0], self.batch, self.filter_classes,
                                                                 score_threshold=0.40, t=t)
                else:
                    lidar_boxes = view_boxes_to_lidar_boxes_xdyd(targets3d[0], self.batch, self.filter_classes,
                                                                 score_threshold=0.40, t=t)
                    print('Visualizing Lidar Boxes using GTs')
                cam_boxes = lidar_boxes_to_cam_boxes(lidar_boxes, extrinsics)
                boxes_pred, cuboids_pred, categories_pred, _ = box2corners(cam_boxes, intrinsics, self.final_size)
                boxes_pred = torch.from_numpy(np.asarray(boxes_pred))
                box_cuboids = self.draw_cuboids(box_cuboids, boxes_pred, cuboids_pred, categories_pred, font_size=12,
                                                outline=(255, 128, 0, 50), bbox_outline=(255, 255, 0, 50))

            if self.show['zpos_pred']:
                if 'zpos' in output.keys():
                    z_position = output['zpos'][0, 0, 0:2].cpu()
                    inst_pred, _ = predict_instance_segmentation_and_trajectories(output, compute_matched_centers=True)
                    inst_pred = inst_pred[0, 0, 0].cpu()
                else:
                    z_position = self.batch['z_position'][0, t, 0:2].cpu()
                    inst_pred = self.batch['instance'][0, t, 0].cpu()
                    print('Visualizing z-positions using GTs')

                box_cuboids = self.get_bev_to_image_points(box_cuboids, extrinsics, intrinsics, view, z_position,
                                                           inst_pred, bev_conditions=self.bev_conditions[camera])

            if self.show['gt']:
                box_cuboids = self.draw_cuboids(box_cuboids, boxes_gt, cuboids_gt, categories_gt, font_size=12)
                z_position = self.batch['z_position'][0, t, 0:2].cpu()
                inst_gt = self.batch['instance'][0, t, 0].cpu()
                box_cuboids = self.get_bev_to_image_points(box_cuboids, extrinsics, intrinsics, view, z_position,
                                                           inst_gt, bev_conditions=self.bev_conditions[camera])

            box_cuboids = np.array(T.ToPILImage()(box_cuboids))
            if camera < 3:
                box_cuboids_to_show.append(box_cuboids)
            else:
                box_cuboids_to_show.append(box_cuboids[:, ::-1])

        H, W, C = input_to_show[0].shape
        input_to_show = np.stack(input_to_show, axis=1)
        input_to_show = input_to_show.reshape(H, 2, 3, W, C).swapaxes(0, 1).reshape(H * 2, W * 3, C)
        box_cuboids_to_show = np.stack(box_cuboids_to_show, axis=1)
        box_cuboids_to_show = box_cuboids_to_show.reshape(H, 2, 3, W, C).swapaxes(0, 1).reshape(H * 2, W * 3, 3)

        lanes = self.batch['lanes'][0, t].cpu().numpy()
        lines = self.batch['lines'][0, t].cpu().numpy()

        if 'segm' in output.keys():
            segm_pred = output['segm'].argmax(2, keepdim=True)[0, 0, 0].cpu().numpy()
            pred = output['segm'].argmax(2, keepdim=True)[0, 1:, 0].cpu().numpy()
            segm_pred = self.get_pred_with_road(segm_pred, pred, lanes, lines)

            inst_pred, _ = predict_instance_segmentation_and_trajectories(output, compute_matched_centers=True)
            inst_pred = inst_pred[0, 0, 0].cpu().numpy()
            inst_pred = self.get_bitmap_with_road(inst_pred, lanes, lines)

            center_pred = output['center'][0, 0, 0].cpu().numpy()
            center_pred = (center_pred > 0.10).astype(np.uint8)
            center_pred = self.get_bitmap_with_road(center_pred, lanes, lines)

            zpos_pred = output['zpos'][0, 0, 0].cpu().numpy() * 255 + 128
            zpos_pred = zpos_pred[:, :, None].repeat(3, 2).astype(np.uint8)
            inst_pred_zpos, _ = predict_instance_segmentation_and_trajectories(output, compute_matched_centers=True)
            inst_pred_zpos = inst_pred_zpos[0, 0, 0].cpu().numpy()
            zpos_pred = self.get_zmap_with_road(zpos_pred, inst_pred_zpos, lanes, lines)

            map_pred = np.concatenate([segm_pred, inst_pred, center_pred, zpos_pred], axis=1)
            map_pred = cv2.resize(map_pred, dsize=(1500, 300), interpolation=cv2.INTER_CUBIC)
        else:
            map_pred = None

        segm_gt = self.batch['segmentation'][0, t, 0].cpu().numpy()
        segm_gt = self.get_bitmap_with_road(segm_gt, lanes, lines)

        inst_gt = self.batch['instance'][0, t, 0].cpu().numpy()
        inst_gt = self.get_bitmap_with_road(inst_gt, lanes, lines)

        spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
        center_gt, _, _ = instance_to_center_offset_flow(self.batch['instance'], self.batch['future_egomotion'],
                                                         ignore_index=self.cfg.DATASET.IGNORE_INDEX,
                                                         subtract_egomotion=True,
                                                         spatial_extent=spatial_extent,
                                                         )

        center_gt = center_gt[0, t, 0].cpu().numpy()
        center_gt = (center_gt > 0.10).astype(np.uint8)
        center_gt = self.get_bitmap_with_road(center_gt, lanes, lines)

        zpos_gt = self.batch['z_position'][0, t, 0].cpu().numpy() * 255 + 128
        zpos_gt = zpos_gt[:, :, None].repeat(3, 2).astype(np.uint8)
        inst_gt_zpos = self.batch['instance'][0, t, 0].cpu().numpy()
        zpos_gt = self.get_zmap_with_road(zpos_gt, inst_gt_zpos, lanes, lines)

        view_boxes_gt = view_boxes_to_bitmap_xdyd(targets3d, self.bev_size)
        view_boxes_gt = self.get_bitmap_with_road(view_boxes_gt, lanes, lines)

        map_gt = np.concatenate([segm_gt, inst_gt, center_gt, zpos_gt, view_boxes_gt], axis=1)
        map_gt = cv2.resize(map_gt, dsize=(1500, 300), interpolation=cv2.INTER_CUBIC)

        # cv2.imshow("input", cv2.cvtColor(input_to_show, cv2.COLOR_RGB2BGR))
        cv2.imshow("On Cam", cv2.cvtColor(box_cuboids_to_show, cv2.COLOR_RGB2BGR))
        cv2.imshow("Ground-truths", cv2.cvtColor(map_gt, cv2.COLOR_RGB2BGR))
        if map_pred is not None:
            cv2.imshow("Predictions", cv2.cvtColor(map_pred, cv2.COLOR_RGB2BGR))

    def get_pred_with_road(self, x, pred, lanes, lines):
        output = np.zeros((*self.bev_size, 3)).astype(np.uint8)
        alphas = [0.2, 0.4, 0.6, 0.8]

        x_mask = x > 0
        lane_mask = (lanes > 0).any(0)
        line_mask = (lines > 0).any(0)

        output[lane_mask] = [75, 75, 75]
        output[line_mask] = [255, 255, 255]

        for t in range(pred.shape[0], 0, -1):
            color = INSTANCE_COLOURS[2] * alphas[t - 1]
            output[pred[t - 1] > 0] = color

        colors = INSTANCE_COLOURS[x[x_mask]]
        output[x_mask] = colors  # [255, 172, 28]
        output[95:105, 97:103] = [52, 152, 219]

        return output

    @staticmethod
    @torch.no_grad()
    def draw_cuboids(
            image: torch.Tensor,
            boxes: torch.Tensor,
            cuboids: List[np.array],
            labels: Optional[List[str]] = None,
            keypoints: Optional[List[float]] = None,
            colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
            fill: Optional[bool] = False,
            width: int = 1,
            font: Optional[str] = None,
            font_size: int = 10,
            outline: tuple = (0, 0, 255, 50),
            bbox_outline: tuple = (0, 255, 0, 50),
    ) -> torch.Tensor:
        """
        Draws bounding boxes on given image.
        The values of the input image should be uint8 between 0 and 255.
        If fill is True, Resulting Tensor should be saved as PNG image.

        Args:
            image (Tensor): Tensor of shape (C x H x W) and dtype uint8.
            boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
                the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
                `0 <= ymin < ymax < H`.
            labels (List[str]): List containing the labels of bounding boxes.
            colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of bounding boxes. The colors can
                be represented as `str` or `Tuple[int, int, int]`.
            fill (bool): If `True` fills the bounding box with specified color.
            width (int): Width of bounding box.
            font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
                also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
                `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
            font_size (int): The requested font size in points.
            outline (list): Outline color for cuboids
            bbox_outline (list): Outline color for bounding boxes

        Returns:
            img (Tensor[C, H, W]): Image Tensor of dtype uint8 with bounding boxes plotted.
        """

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Tensor expected, got {type(image)}")
        elif image.dtype != torch.uint8:
            raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
        elif image.dim() != 3:
            raise ValueError("Pass individual images, not batches")

        ndarr = image.permute(1, 2, 0).numpy()
        img_to_draw = Image.fromarray(ndarr)

        img_boxes = boxes.to(torch.int64).tolist()
        img_cuboids = cuboids

        if fill:
            draw = ImageDraw.Draw(img_to_draw, "RGBA")

        else:
            draw = ImageDraw.Draw(img_to_draw)

        txt_font = ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=font_size)

        for i, bbox in enumerate(img_boxes):
            if colors is None:
                color = None
            else:
                color = colors[i]
                if type(color) == list:
                    color = tuple(color)

            if fill:
                if color is None:
                    fill_color = (255, 255, 255, 50)
                elif isinstance(color, str):
                    # This will automatically raise Error if rgb cannot be parsed.
                    fill_color = ImageColor.getrgb(color) + (100,)
                elif isinstance(color, tuple):
                    fill_color = color + (100,)
                draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
                if keypoints is not None:
                    draw.line(keypoints[i].reshape(-1), width=width, fill=fill_color)
            else:

                if keypoints is not None:
                    draw.line(keypoints[i].reshape(-1), width=width, )

                if len(img_cuboids) > 0 and len(img_cuboids[i]) == 8:
                    img_cuboid = [tuple(cuboid) for cuboid in img_cuboids[i][0:4]]
                    draw.polygon(img_cuboid, outline=outline)
                    img_cuboid = [tuple(cuboid) for cuboid in img_cuboids[i][4:]]
                    draw.polygon(img_cuboid, outline=outline)
                    img_cuboid = [tuple(cuboid) for cuboid in [img_cuboids[i][0], img_cuboids[i][1], img_cuboids[i][5],
                                                               img_cuboids[i][4]]]
                    draw.polygon(img_cuboid, outline=outline)
                    img_cuboid = [tuple(cuboid) for cuboid in [img_cuboids[i][2], img_cuboids[i][3], img_cuboids[i][7],
                                                               img_cuboids[i][6]]]
                    draw.polygon(img_cuboid, outline=outline)

                else:
                    draw.rectangle(bbox, width=width, outline=bbox_outline)

            if labels is not None:
                margin = width + 1
                draw.text((bbox[0] + margin, bbox[1] + margin), labels[i], fill=color, font=txt_font)

        return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)

    @staticmethod
    @torch.no_grad()
    def draw_cuboids_only(
            image: torch.Tensor,
            cuboids: List[np.array],
            colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
            fill: Optional[bool] = False,
            width: int = 1,
            font: Optional[str] = None,
            font_size: int = 10
    ) -> torch.Tensor:
        """
        Draws bounding boxes on given image.
        The values of the input image should be uint8 between 0 and 255.
        If fill is True, Resulting Tensor should be saved as PNG image.

        Args:
            image (Tensor): Tensor of shape (C x H x W) and dtype uint8.
            boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
                the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
                `0 <= ymin < ymax < H`.
            labels (List[str]): List containing the labels of bounding boxes.
            colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of bounding boxes. The colors can
                be represented as `str` or `Tuple[int, int, int]`.
            fill (bool): If `True` fills the bounding box with specified color.
            width (int): Width of bounding box.
            font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
                also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
                `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
            font_size (int): The requested font size in points.

        Returns:
            img (Tensor[C, H, W]): Image Tensor of dtype uint8 with bounding boxes plotted.
        """

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Tensor expected, got {type(image)}")
        elif image.dtype != torch.uint8:
            raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
        elif image.dim() != 3:
            raise ValueError("Pass individual images, not batches")

        ndarr = image.permute(1, 2, 0).numpy()
        img_to_draw = Image.fromarray(ndarr)

        img_cuboids = cuboids

        if fill:
            draw = ImageDraw.Draw(img_to_draw, "RGBA")

        else:
            draw = ImageDraw.Draw(img_to_draw)

        txt_font = ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=font_size)

        for i, bbox in enumerate(cuboids):
            if colors is None:
                color = None
            else:
                color = colors[i]
                if type(color) == list:
                    color = tuple(color)

            if fill:
                if color is None:
                    fill_color = (255, 255, 255, 50)
                elif isinstance(color, str):
                    # This will automatically raise Error if rgb cannot be parsed.
                    fill_color = ImageColor.getrgb(color) + (100,)
                elif isinstance(color, tuple):
                    fill_color = color + (100,)
            else:

                if len(img_cuboids) > 0 and len(img_cuboids[i]) == 8:
                    img_cuboid = [tuple(cuboid) for cuboid in img_cuboids[i][0:4]]
                    draw.polygon(img_cuboid, outline=(0, 0, 255, 50))
                    img_cuboid = [tuple(cuboid) for cuboid in img_cuboids[i][4:]]
                    draw.polygon(img_cuboid, outline=(0, 0, 255, 50))
                    img_cuboid = [tuple(cuboid) for cuboid in [img_cuboids[i][0], img_cuboids[i][1], img_cuboids[i][5],
                                                               img_cuboids[i][4]]]
                    draw.polygon(img_cuboid, outline=(0, 0, 255, 50))
                    img_cuboid = [tuple(cuboid) for cuboid in [img_cuboids[i][2], img_cuboids[i][3], img_cuboids[i][7],
                                                               img_cuboids[i][6]]]
                    draw.polygon(img_cuboid, outline=(0, 0, 255, 50))

        return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)

    @staticmethod
    @torch.no_grad()
    def draw_point(
            image: torch.Tensor,
            points: List[np.array],
            radius: Optional[int] = 5,
            fill: Optional[bool] = False,
    ) -> torch.Tensor:

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Tensor expected, got {type(image)}")
        elif image.dtype != torch.uint8:
            raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
        elif image.dim() != 3:
            raise ValueError("Pass individual images, not batches")

        ndarr = image.permute(1, 2, 0).numpy()
        img_to_draw = Image.fromarray(ndarr)

        if fill:
            draw = ImageDraw.Draw(img_to_draw, "RGBA")
        else:
            draw = ImageDraw.Draw(img_to_draw)

        for i, point in enumerate(points):
            x, y = point
            ellipse = (x - radius, y - radius, x + radius, y + radius)

            draw.ellipse(ellipse, outline=(0, 0, 255, 50))

        return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)

    def get_bev_to_image_points(self, image, extrinsic, intrinsic, view, z_positions, instances,
                                bev_conditions=(5, 190, 0, 100)):
        points = []
        nb_instances = instances.numpy().max()
        for i_instance in range(1, nb_instances):
            ys, xs = np.where(instances.numpy() == i_instance)
            min_zs = z_positions[0] - z_positions[1] / 2
            max_zs = z_positions[0] + z_positions[1] / 2
            if len(xs):
                min_x = xs.min()
                max_x = xs.max()
                min_y = ys.min()
                max_y = ys.max()

                point = np.concatenate([
                    np.stack([xs, ys, min_zs[ys, xs], np.ones_like(xs)], axis=-1),
                    np.stack([xs, ys, max_zs[ys, xs], np.ones_like(xs)], axis=-1)], axis=0)
                if min_x > bev_conditions[0] and max_x < bev_conditions[1] and min_y > bev_conditions[2] and max_y < \
                        bev_conditions[3]:
                    points.append(point)

        S1 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]
                       ])

        cam_to_lidar = extrinsic.numpy()
        cam_intr = intrinsic.numpy()
        image = (image.permute(1, 2, 0)).to(torch.uint8)

        if len(points):
            points = np.concatenate(points, axis=0)
            points = np.linalg.inv(cam_to_lidar) @ np.linalg.inv(view) @ points.T
            points = cam_intr @ S1 @ points
            points = points / points[2, :]
            points = points.T[:, 0:2]
            # points_mask = points[:,1] > image.shape[0] // 2
            points_mask = np.logical_and(points[:, 0] < 1600, points[:, 0] > 0)
            points_mask1 = np.logical_and(points[:, 1] < 900, points[:, 1] > 0)
            points_mask = np.logical_and(points_mask, points_mask1)
            points = points[points_mask]
        else:
            points = np.asarray([])

        points = self.draw_point(image.permute(2, 0, 1), points, fill=True, radius=4)

        return points

    @staticmethod
    def pad_images(input_image, padding_size=5, ones=False):
        h, w, c = input_image.shape

        padding_func = np.ones if ones else np.zeros
        vertical_padding = padding_func((padding_size, w, c)).astype(np.uint8) * 255
        horizontal_padding = padding_func((h + padding_size * 2, padding_size, c)).astype(np.uint8) * 255

        output_image = np.concatenate([input_image, vertical_padding], axis=0)
        output_image = np.concatenate([vertical_padding, output_image], axis=0)
        output_image = np.concatenate([output_image, horizontal_padding], axis=1)
        output_image = np.concatenate([horizontal_padding, output_image], axis=1)

        return output_image


class LayoutControl(object):
    def __init__(self, show=None):
        if show is not None:
            self.show = show
        else:
            self.show = {'gt': False,
                         '2d_pred': False,
                         '3d_pred': True,
                         'zpos_pred': True,
                         'lanes_gt_on_cam': False,
                         }
        self.pause = False
        self.next = False

    def __call__(self, ch):
        if ch == 27:
            return True
        elif ch == ord('q') or ch == ord('Q'):
            self.show['gt'] = True
        elif ch == ord('a') or ch == ord('A'):
            self.show['gt'] = False
        elif ch == ord('w') or ch == ord('W'):
            self.show['2d_pred'] = True
        elif ch == ord('s') or ch == ord('S'):
            self.show['2d_pred'] = False
        elif ch == ord('e') or ch == ord('E'):
            self.show['3d_pred'] = True
        elif ch == ord('d') or ch == ord('D'):
            self.show['3d_pred'] = False
        elif ch == ord('r') or ch == ord('R'):
            self.show['zpos_pred'] = True
        elif ch == ord('f') or ch == ord('F'):
            self.show['zpos_pred'] = False
        elif ch == ord('t') or ch == ord('T'):
            self.show['lanes_gt_on_cam'] = True
        elif ch == ord('g') or ch == ord('G'):
            self.show['lanes_gt_on_cam'] = False
        elif ch == 32:
            self.pause = not self.pause
        elif ch == ord('n') or ch == ord('N'):
            self.next = True

    def get_show(self):
        return self.show
