import numpy as np
import torch
import cv2
from typing import Union, Optional, List, Tuple
from PIL import ImageColor, ImageFont, ImageDraw, Image

from tairvision.models.bev.common.nuscenes.process import FilterClasses, decide_mean_std
from tairvision.models.bev.lss.utils.network import NormalizeInverse
from tairvision.models.bev.lss.utils.bbox import box2corners, lidar_boxes_to_cam_boxes
from tairvision.models.bev.common.utils.visualization import heatmap_image, INSTANCE_COLOURS, generate_instance_colours
from torchvision import transforms as T


class VisualizationModule:
    def __init__(self, cfg, score_threshold_2d=0.50, score_threshold_3d=0.40):
        self.bev_size = (int((cfg.LIFT.X_BOUND[1] - cfg.LIFT.X_BOUND[0]) / cfg.LIFT.X_BOUND[2]),
                         int((cfg.LIFT.Y_BOUND[1] - cfg.LIFT.Y_BOUND[0]) / cfg.LIFT.Y_BOUND[2])
                         )
        self.nb_z_bins = (cfg.LIFT.Z_BOUND[1] - cfg.LIFT.Z_BOUND[0]) / cfg.LIFT.Z_BOUND[2]
        self.spatial_extent = (cfg.LIFT.X_BOUND[1], cfg.LIFT.Y_BOUND[1])
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD
        self.final_size = (cfg.IMAGE.FINAL_DIM[1], cfg.IMAGE.FINAL_DIM[0])
        self.nb_cameras = len(cfg.IMAGE.NAMES)

        self.bev_conditions = [(0, 100, 0, 120), (5, 190, 0, 100), (100, 200, 0, 120),
                               (0, 100, 80, 200), (5, 190, 100, 200), (100, 200, 80, 200)]

        mean, std = decide_mean_std(cfg.PRETRAINED.LOAD_WEIGHTS)
        self.denormalize = NormalizeInverse(mean=mean, std=std)

        self.filter_classes = FilterClasses(cfg.DATASET.FILTER_CLASSES, cfg.DATASET.BOX_RESIZING_COEF)
        self.score_threshold_3d = score_threshold_3d
        self.score_threshold_2d = score_threshold_2d
        self.cfg = cfg
        self._import_target_functions()

    def _import_target_functions(self):
        from tairvision.models.bev.common.utils.instance import get_targets_dynamic
        from tairvision.models.bev.lss.utils.static import get_targets_static
        from tairvision.models.bev.lss.utils.bbox import get_targets2d
        from tairvision.models.bev.lss.utils.bbox import view_boxes_to_lidar_boxes_xdyd, view_boxes_to_lidar_boxes_yaw
        from tairvision.models.bev.lss.utils.bbox import view_boxes_to_bitmap_xdyd, view_boxes_to_bitmap_yaw
        from tairvision.models.bev.lss.utils.bbox import get_targets3d_xdyd, get_targets3d_yaw
        self.get_targets_dynamic = get_targets_dynamic
        self.get_targets_static = get_targets_static
        self.get_targets2d = get_targets2d
        if self.cfg.MODEL.HEAD3D.TARGET_TYPE == "yaw":
            self.get_targets3d = get_targets3d_yaw
            self.view_boxes_to_lidar_boxes = view_boxes_to_lidar_boxes_yaw
            self.view_boxes_to_bitmap = view_boxes_to_bitmap_yaw
        else:  # target_type="xdyd"
            self.get_targets3d = get_targets3d_xdyd
            self.view_boxes_to_lidar_boxes = view_boxes_to_lidar_boxes_xdyd
            self.view_boxes_to_bitmap = view_boxes_to_bitmap_xdyd

    def plot_all(self, batch, output, show):

        input_to_show = []
        output_to_show = []
        for camera in range(self.nb_cameras):
            image = batch['images'][0, self.receptive_field - 1, camera].cpu()
            image = self.denormalize(image)
            image = (image * 255).to(torch.uint8)

            self.append_camera_images(input_to_show, image, camera)

            if show['2d_pred']:
                image = self.plot_2d_boxes_on_image(image, batch, camera, output=output)

            if show['3d_pred']:
                image = self.plot_view_boxes_on_image(image, batch, camera, output=output)

            if show['zpos_pred']:
                image = self.plot_zpos_on_image(image, batch, camera,
                                                output=output,
                                                receptive_field=self.receptive_field,
                                                bev_conditions=self.bev_conditions[camera],
                                                )

            if show['inst_pred']:
                image = self.plot_future_inst_on_image(image, batch, camera,
                                                       output=output,
                                                       receptive_field=self.receptive_field,
                                                       bev_conditions=self.bev_conditions[camera])

            if show['gt']:
                image = self.plot_cam_boxes_on_image(image, batch, camera,
                                                     receptive_field=self.receptive_field,
                                                     final_size=self.final_size
                                                     )

                image = self.plot_zpos_on_image(image, batch, camera,
                                                output=None,
                                                receptive_field=self.receptive_field,
                                                bev_conditions=self.bev_conditions[camera],
                                                )

            if show['lanes_gt_on_cam']:
                image = self.plot_lanes_on_image(image, batch, camera,
                                                 receptive_field=self.receptive_field,
                                                 )

            self.append_camera_images(output_to_show, image, camera)

        h, w, c = input_to_show[0].shape
        input_to_show = np.stack(input_to_show, axis=1)
        input_to_show = input_to_show.reshape((h, 2, 3, w, c)).swapaxes(0, 1).reshape(h * 2, w * 3, c)
        # cv2.imshow("Input", cv2.cvtColor(input_to_show, cv2.COLOR_RGB2BGR))

        output_to_show = np.stack(output_to_show, axis=1)
        output_to_show = output_to_show.reshape((h, 2, 3, w, c)).swapaxes(0, 1).reshape(h * 2, w * 3, c)
        cv2.imshow("Camera projection", cv2.cvtColor(output_to_show, cv2.COLOR_RGB2BGR))

        # visualizing bev maps for segmentation and detection
        targets = self.get_targets_dynamic(batch,
                                           receptive_field=self.receptive_field,
                                           spatial_extent=self.spatial_extent,
                                           )
        bev_map, _, _ = self.get_bev_maps(batch, output, targets)

        nb_maps = bev_map.shape[1] / bev_map.shape[0]
        bev_map = cv2.resize(bev_map, dsize=(int(nb_maps * 600), 600), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("BEV - Predictions vs. GTs", cv2.cvtColor(bev_map, cv2.COLOR_RGB2BGR))

    def plot_view_boxes_on_image(self, input_image, batch, cam_index, output=None):
        if output is not None and 'head3d' in output.keys():
            det3d = output['head3d'][0]
        else:
            targets3d = self.get_targets3d(batch, receptive_field=self.receptive_field)[0]
            det3d = targets3d[0]
            print('Visualizing Lidar Boxes using GTs')

        temp_index = self.receptive_field - 1
        intrinsics = batch['intrinsics'][0, temp_index, cam_index].cpu()
        extrinsics = batch['cams_to_lidar'][0, temp_index, cam_index].cpu()

        lidar_boxes = self.view_boxes_to_lidar_boxes(det3d, batch, self.filter_classes,
                                                     score_threshold=self.score_threshold_3d,
                                                     t=temp_index,
                                                     )
        cam_boxes = lidar_boxes_to_cam_boxes(lidar_boxes, extrinsics)
        boxes, cuboids, categories, _ = box2corners(cam_boxes, intrinsics, self.final_size)
        boxes = torch.from_numpy(np.asarray(boxes))
        output_image = draw_cuboids(input_image, boxes, cuboids, categories,
                                    font_size=12,
                                    outline=(255, 128, 0, 50),
                                    bbox_outline=(255, 255, 0, 50),
                                    )

        return output_image

    @staticmethod
    def plot_cam_boxes_on_image(input_image, batch, cam_index,
                                receptive_field=1,
                                final_size=(256, 704),
                                ):

        temp_index = receptive_field - 1
        intrinsics = batch['intrinsics'][0, temp_index, cam_index].cpu()

        boxes = batch['boxes'][0][temp_index][cam_index]
        boxes, cuboids, categories, _ = box2corners(boxes, intrinsics, final_size)
        boxes = torch.from_numpy(np.asarray(boxes))
        output_image = draw_cuboids(input_image, boxes, cuboids, categories, font_size=12)

        return output_image

    def plot_2d_boxes_on_image(self, input_image, batch, cam_index, output=None):
        if output is not None and 'head2d' in output.keys():
            det2d = output['head2d'][cam_index]
            score_mask = output['head2d'][cam_index]['scores'].cpu() > self.score_threshold_2d
        else:
            targets2d = self.get_targets2d(batch, receptive_field=self.receptive_field, image_size=self.final_size)[0]
            det2d = targets2d[cam_index]
            score_mask = torch.tensor([True for _ in det2d['boxes'].cpu()], dtype=torch.bool)
            print('Visualizing 2D Bounding Boxes using GTs')

        boxes2d = det2d['boxes'].cpu()[score_mask]
        labels2d = det2d['labels'].cpu()[score_mask]
        labels2d = [self.filter_classes.classes[i.item()] for i in labels2d]

        output_image = draw_cuboids(input_image, boxes2d, [], labels2d,
                                    font_size=12,
                                    bbox_outline=(255, 0, 0, 50),
                                    )

        return output_image

    @staticmethod
    def plot_zpos_on_image(input_image, batch, cam_index,
                           output=None,
                           receptive_field=1,
                           bev_conditions=(5, 190, 0, 100),
                           ):

        temp_index = receptive_field - 1
        intrinsics = batch['intrinsics'][0, temp_index, cam_index].cpu()
        extrinsics = batch['cams_to_lidar'][0, temp_index, cam_index].cpu()
        view = batch['view'][0, temp_index, 0].cpu()

        if output is not None and 'zpos' in output.keys():
            z_position = output['zpos'][0, temp_index, 0:2].cpu()
            inst_pred = output['inst'][0, temp_index, 0].cpu()
        else:
            z_position = batch['z_position'][0, temp_index, 0:2].cpu()
            inst_pred = batch['instance'][0, temp_index, 0].cpu()
            print('Visualizing z-positions using GTs')

        output_image = get_bev_to_image_points(input_image, extrinsics, intrinsics, view, z_position, inst_pred,
                                               bev_conditions=bev_conditions,
                                               )

        return output_image

    @staticmethod
    def plot_future_inst_on_image(input_image, batch, cam_index,
                                  output=None, 
                                  receptive_field=1,
                                  bev_conditions=(5, 190, 0, 100)
                                  ):
        temp_index = receptive_field - 1
        intrinsics = batch['intrinsics'][0, temp_index:, cam_index].cpu()
        extrinsics = batch['cams_to_lidar'][0, temp_index:, cam_index].cpu()
        view = batch['view'][0, temp_index:, 0].cpu()

        inst_pred = output['inst'][0, :, 0].cpu()
        output_image = get_bev_to_image_trajectories(
            input_image, extrinsics, intrinsics, view, inst_pred,
            bev_conditions=bev_conditions
            )
        return output_image

    @staticmethod
    def plot_lanes_on_image(input_image, batch, cam_index,
                            receptive_field=1,
                            ):

        temp_index = receptive_field - 1
        lines_cam = batch['lines_cam'][0, temp_index, cam_index].cpu()
        lines_cam = (lines_cam * 255).to(torch.uint8).numpy()
        lines_cam = lines_cam[0] + lines_cam[1]
        input_image[:, lines_cam > 0] = torch.ones((3, 1), dtype=torch.uint8) * 255

        return input_image

    def get_bev_maps(self, batch, output, targets):

        static_gt = self.get_targets_static(batch, self.receptive_field)
        lanes_gt = static_gt['lanes'][0, 0].cpu().numpy()
        lines_gt = static_gt['lines'][0, 0].cpu().numpy()

        if 'lanes' in output.keys():
            lanes = output['lanes'][0, 0].cpu().numpy()
            lines = output['lines'][0, 0].cpu().numpy()
        else:
            lanes = lanes_gt
            lines = lines_gt

        segm_gt, inst_gt, center_gt, zpos_gt = self.plot_segm(targets, lanes_gt, lines_gt,
                                                              gt=True,
                                                              bev_size=self.bev_size,
                                                              nb_z_bins=self.nb_z_bins,
                                                              )
        map_gt = [segm_gt, inst_gt, center_gt, zpos_gt]
        map_gt = [pad_images(i, padding_size=2, ones=True) for i in map_gt]

        if 'segm' in output.keys():
            segm_pred, inst_pred, center_pred, zpos_pred = self.plot_segm(output, lanes, lines,
                                                                          bev_size=self.bev_size,
                                                                          nb_z_bins=self.nb_z_bins,
                                                                          )
            map_pred = [segm_pred, inst_pred, center_pred, zpos_pred]
        else:
            map_pred = [np.zeros_like(segm_gt)] * 4
        map_pred = [pad_images(i, padding_size=2, ones=True) for i in map_pred]

        targets3d, _ = self.get_targets3d(batch,
                                          receptive_field=self.receptive_field,
                                          spatial_extent=self.spatial_extent,
                                          )
        view_boxes_gt = self.plot_view_boxes(targets3d[0:1], lanes_gt, lines_gt, bev_size=self.bev_size)
        map_gt.append(pad_images(view_boxes_gt, padding_size=2, ones=True))

        if 'head3d' in output.keys():
            view_boxes_pred = self.plot_view_boxes(output['head3d'][0:1], lanes, lines,
                                                   bev_size=self.bev_size,
                                                   score_threshold=self.score_threshold_3d,
                                                   )
        else:
            view_boxes_pred = np.zeros_like(view_boxes_gt)
        map_pred.append(pad_images(view_boxes_pred, padding_size=2, ones=True))

        map_pred = np.concatenate(map_pred, axis=1)
        map_gt = np.concatenate(map_gt, axis=1)
        map_all = np.concatenate([map_pred, map_gt], axis=0)

        return map_all, map_pred, map_gt

    @staticmethod
    def plot_segm(dynamic_head, lanes, lines,
                  gt=False,
                  bev_size=(200, 200),
                  nb_z_bins=8,
                  ):

        segm = dynamic_head['segmentation'] if gt else dynamic_head['segm']
        inst = dynamic_head['instance'] if gt else dynamic_head['inst']
        center = dynamic_head['centerness'] if gt else dynamic_head['center']
        zpos = dynamic_head['z_position'] if gt else dynamic_head['zpos']
        matched_centers = None if gt else dynamic_head['matched_centers']

        segm = segm[0, :, 0].cpu().numpy()
        inst = inst[0, :, 0].cpu().numpy()
        center = center[0, 0:1, 0].cpu().numpy()
        zpos = zpos[0, 0:1, 0].cpu().numpy() * (256 / nb_z_bins)

        center = [heatmap_image(c) for c in center]
        zpos = zpos[..., None].repeat(3, -1).astype(np.uint8)

        masks = segm > 0
        segm = get_bitmap_with_road(segm, lanes, lines, bev_size=bev_size)
        if matched_centers is None:
            inst = get_bitmap_with_road(inst, lanes, lines, bev_size=bev_size, masks=masks)
        else:
            inst = get_trajectory_with_road(inst, matched_centers, lanes, lines, bev_size=bev_size)
        center = get_bitmap_with_road(center, lanes, lines, bev_size=bev_size, masks=masks,
                                      color_coef=[1.0, 1.0, 1.0],
                                      )
        zpos = get_bitmap_with_road(zpos, lanes, lines, bev_size=bev_size, masks=masks,
                                    color_coef=[1.0, 0.77, 0.11],
                                    )

        return segm, inst, center, zpos

    def plot_view_boxes(self, det3d, lanes, lines, bev_size=(200, 200), score_threshold=0.40, draw_labels=None):

        if draw_labels is None:
            draw_labels = [1, 2, 3]
        view_boxes = self.view_boxes_to_bitmap(det3d, bev_size, score_threshold=score_threshold, draw_labels=draw_labels)
        #view_boxes = get_bitmap_with_road(view_boxes[None, ...], lanes, lines, bev_size=bev_size)

        return view_boxes

    def append_camera_images(self, input_list, image, cam_index):

        image_to_append = np.array(T.ToPILImage()(image))
        if cam_index > self.nb_cameras // 2 - 1:
            image_to_append = image_to_append[:, ::-1]
        image_to_append = pad_images(image_to_append)

        input_list.append(image_to_append)


def get_bitmap(x, bev_size=(200, 200)):
    output = np.zeros((*bev_size, 3)).astype(np.uint8)
    mask = x > 0
    colors = INSTANCE_COLOURS[x[mask] % len(INSTANCE_COLOURS)]
    output[mask] = colors  # [255, 172, 28]
    center_x = bev_size[0] // 2
    center_y = bev_size[1] // 2
    output[center_x - 5:center_x + 5, center_y - 3:center_y + 3] = [52, 152, 219]
    return output


def get_bitmap_with_road(xs, lanes, lines, bev_size=(200, 200), masks=None, color_coef=None):
    output = np.zeros((*bev_size, 3)).astype(np.uint8)

    lane_mask = (lanes > 0).any(0)
    line_mask = (lines > 0).any(0)

    output[lane_mask] = [75, 75, 75]
    output[line_mask] = [255, 255, 255]

    if masks is None:
        masks = [x > 0 for x in xs]

    alpha_step = 1.0 / len(xs)

    for i in range(len(xs)):
        ind = len(xs) - i - 1
        x = xs[ind]
        mask = masks[ind]
        alpha = alpha_step * (i + 1)
        if color_coef is None:
            output[mask] = INSTANCE_COLOURS[x[mask] % len(INSTANCE_COLOURS)] * alpha
        else:
            output[mask] = (x[mask] * color_coef).astype(np.uint8) * alpha
    output[95:105, 97:103] = [52, 152, 219]

    return output


def get_trajectory_with_road(xs, centers, lanes, lines, bev_size=(200, 200), masks=None):
    output = np.zeros((*bev_size, 3)).astype(np.uint8)

    lane_mask = (lanes > 0).any(0)
    line_mask = (lines > 0).any(0)

    output[lane_mask] = [75, 75, 75]
    output[line_mask] = [255, 255, 255]

    if masks is None:
        masks = xs[0] > 0

    output[masks] = INSTANCE_COLOURS[xs[0, masks] % len(INSTANCE_COLOURS)]
    output[95:105, 97:103] = [52, 152, 219]

    xs = torch.from_numpy(xs)
    unique_ids = torch.unique(xs[0]).cpu().long().numpy()[1:]
    instance_map = dict(zip(unique_ids, unique_ids))
    instance_colours = generate_instance_colours(instance_map)

    trajectory_img = np.zeros((200, 200, 3), dtype=np.uint8)

    for instance_id in unique_ids:
        path = centers[instance_id]
        for t in range(len(path) - 1):
            color = instance_colours[instance_id].tolist()
            cv2.line(trajectory_img, (int(path[t][0]), int(path[t][1])), (int(path[t + 1][0]), int(path[t + 1][1])),
                     color, 2)

    temp_img = cv2.addWeighted(output, 0.7, trajectory_img, 0.3, 1.0)
    mask = ~ np.all(trajectory_img == 0, axis=2)
    output[mask] = temp_img[mask]
    return output


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


@torch.no_grad()
def draw_point(
        image: torch.Tensor,
        points: List[np.array],
        radius: Optional[int] = 5,
        fill: Optional[bool] = False,
        color=(0, 0, 255, 50)
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
        ellipse = (x-radius, y-radius, x+radius, y+radius)

        draw.ellipse(ellipse, outline=color)

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


def get_bev_to_image_points(image, extrinsic, intrinsic, view, z_positions, instances, bev_conditions=(5, 190, 0, 100)):
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
            if min_x > bev_conditions[0] and max_x < bev_conditions[1] and min_y > bev_conditions[2] and max_y < bev_conditions[3]:
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

    points = draw_point(image.permute(2, 0, 1), points, fill=True, radius=4)

    return points


def get_bev_to_image_trajectories(image, extrinsic, intrinsic, view, instances, bev_conditions=(5, 190, 0, 100)):
    points = []
    xs, ys = [], []
    nb_instances = instances.max()
    for i_instance in range(1, nb_instances):
        for i in range(0, instances.shape[0]):
            ys, xs = np.where(instances[i].cpu().numpy() == i_instance)
            if len(xs):
                min_x = xs.min()
                max_x = xs.max()
                min_y = ys.min()
                max_y = ys.max()

                point = np.concatenate([
                    np.stack([xs, ys, np.zeros_like(xs), np.ones_like(xs)], axis=-1),
                    np.stack([xs, ys, np.zeros_like(xs), np.ones_like(xs)], axis=-1)], axis=0)
                if min_x > bev_conditions[0] and max_x < bev_conditions[1] and min_y > bev_conditions[2] and max_y < bev_conditions[3]:
                    points.append(point)

    S1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]
                   ])

    ego_index = 0
    cam_to_lidar = extrinsic[ego_index].numpy()
    cam_intr = intrinsic[ego_index].numpy()
    image = (image.permute(1, 2, 0)).to(torch.uint8)

    if len(points):
        points = np.concatenate(points, axis=0)
        points = np.linalg.inv(cam_to_lidar) @ np.linalg.inv(view[ego_index]) @ points.T
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

    points = draw_point(image.permute(2, 0, 1), points, fill=True, radius=4, color=(255, 0, 0, 50))

    return points


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
                         'inst_pred': False
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
            self.show['inst_pred'] = True
        elif ch == ord('g') or ch == ord('G'):
            self.show['inst_pred'] = False
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
