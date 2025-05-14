from typing import Union, Optional, List, Tuple, Text, BinaryIO,Any
from PIL import Image, ImageColor, ImageFont, ImageDraw

import colorsys
import matplotlib as mpl
import torch
import torch.distributed as dist
import numpy as np
from contextlib import contextmanager
from functools import wraps
import logging
import torch.nn.functional as F
import cv2
import matplotlib.colors as mplc

from types import FunctionType

__all__ = ["draw_segmentation_masks", "draw_bounding_boxes", "is_dist_avail_and_initialized", "get_world_size",
           "retry_if_cuda_oom", "PanopticDeepLabTargetGenerator",
           "panoptic_post_process_single_mask", "panoptic_inference_mask2former", "put_thing_names_on_image",
           "panoptic_inference_deeplab", "accuracy"]


@torch.no_grad()
def draw_segmentation_masks(
        image: torch.Tensor,
        masks: torch.Tensor,
        alpha: float = 0.8,
        colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
) -> torch.Tensor:
    """
    Draws segmentation masks on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        colors (list or None): List containing the colors of the masks. The colors can
            be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            When ``masks`` has a single entry of shape (H, W), you can pass a single color instead of a list
            with one element. By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError("masks must be of shape (H, W) or (batch_size, H, W)")
    if masks.dtype != torch.bool:
        raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
    if masks.shape[-2:] != image.shape[-2:]:
        raise ValueError("The image and the masks must have the same height and width")

    num_masks = masks.size()[0]
    if colors is not None and num_masks > len(colors):
        raise ValueError(f"There are more masks ({num_masks}) than colors ({len(colors)})")

    if colors is None:
        colors = _generate_color_palette(num_masks)

    if not isinstance(colors, list):
        colors = [colors]
    if not isinstance(colors[0], (tuple, str)):
        raise ValueError("colors must be a tuple or a string, or a list thereof")
    if isinstance(colors[0], tuple) and len(colors[0]) != 3:
        raise ValueError("It seems that you passed a tuple of colors instead of a list of colors")

    out_dtype = torch.uint8

    colors_ = []
    for color in colors:
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        color = torch.tensor(color, dtype=out_dtype)
        colors_.append(color)

    img_to_draw = image.detach().clone()
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors_):
        img_to_draw[:, mask] = color[:, None]

    out = image * (1 - alpha) + img_to_draw * alpha
    return out.to(out_dtype)


def _generate_color_palette(num_masks):
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    return [tuple((i * palette) % 255) for i in range(num_masks)]


@torch.no_grad()
def draw_bounding_boxes(
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[List[str]] = None,
        keypoints: Optional[List[float]] = None,
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

    img_boxes = boxes.to(torch.int64).tolist()

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
            draw.rectangle(bbox, width=width, outline=color)
            if keypoints is not None:
                draw.line(keypoints[i].reshape(-1), width=width, )

        if labels is not None:
            margin = width + 1
            draw.text((bbox[0] + margin, bbox[1] + margin), labels[i], fill=color, font=txt_font)

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


def _mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x + 0.5 for x in res if len(x) >= 6]
    return res, has_holes


def _draw_polygon(segment, color, linewidth, edge_color=None, alpha=0.5):
    """
    Args:
        linewidth: Detectron library automate it with some function
        segment: numpy array of shape Nx2, containing all the points in the polygon.
        color: color of the polygon. Refer to `matplotlib.colors` for a full list of
            formats that are accepted.
        edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
            full list of formats that are accepted. If not provided, a darker shade
            of the polygon color will be used instead.
        alpha (float): blending efficient. Smaller values lead to more transparent masks.

    Returns:
        output (VisImage): image object with polygon drawn.
    """
    if edge_color is None:
        # make edge color darker than the polygon color
        if alpha > 0.8:
            edge_color = _change_color_brightness(color, brightness_factor=-0.7)
        else:
            edge_color = color
    edge_color = mplc.to_rgb(edge_color) + (1,)

    polygon = mpl.patches.Polygon(
        segment,
        fill=True,
        facecolor=mplc.to_rgb(color) + (alpha,),
        edgecolor=edge_color,
        linewidth=linewidth,
    )

    return polygon


def _change_color_brightness(color, brightness_factor):
    """
    Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
    less or more saturation than the original color.

    Args:
        color: color of the polygon. Refer to `matplotlib.colors` for a full list of
            formats that are accepted.
        brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
            0 will correspond to no change, a factor in [-1.0, 0) range will result in
            a darker color and a factor in (0, 1.0] range will result in a lighter color.

    Returns:
        modified_color (tuple[double]): a tuple containing the RGB values of the
            modified color. Each value in the tuple is in the [0.0, 1.0] range.
    """
    assert brightness_factor >= -1.0 and brightness_factor <= 1.0
    color = mplc.to_rgb(color)
    polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
    modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
    modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
    modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
    modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
    return modified_color


def _jitter(color):
    """
    Randomly modifies given color to produce a slightly different color than the color given.

    Args:
        color (tuple[double]): a tuple of 3 elements, containing the RGB values of the color
            picked. The values in the list are in the [0.0, 1.0] range.

    Returns:
        jittered_color (tuple[double]): a tuple of 3 elements, containing the RGB values of the
            color after being jittered. The values in the list are in the [0.0, 1.0] range.
    """
    color = mplc.to_rgb(color)
    vec = np.random.rand(3)
    # better to do it in another color space
    vec = vec / np.linalg.norm(vec) * 0.5
    res = np.clip(vec + color, 0, 1)
    return tuple(res)


def draw_mask_onto_axis(selected_axs, mask, color, alpha):
    try:
        if len(mask.shape) != 2:
            raise ValueError("invalid mask shape")

        shape2d = (mask.shape[0], mask.shape[1])
        polygons, has_holes = _mask_to_polygons(mask)

        # color = _jitter(color)
        # TODO, what if there is a hole
        if not has_holes:
            # draw polygons for regular masks
            for segment in polygons:
                segment = segment.reshape(-1, 2)
                polygon_patch = _draw_polygon(segment, linewidth=0.5,
                                             color=color, alpha=alpha)
                selected_axs.add_patch(polygon_patch)
        else:
            rgba = np.zeros(shape2d + (4,), dtype="float32")
            rgba[:, :, :3] = color
            rgba[:, :, 3] = (mask == 1).astype("float32") * alpha
            selected_axs.imshow(rgba, extent=(0, shape2d[1], shape2d[0], 0))
    except:
        #TODO, how to handle this
        pass


def panoptic_post_process_single_mask(pred_class, isthing, panoptic_seg, mask, foreground, center_list,
                                      class_id_tracker, stuff_memory_list, segments_info, label_divisor):
    if not isthing:
        if int(pred_class) in stuff_memory_list.keys():
            current_segment_id = stuff_memory_list[int(pred_class)]
        else:
            current_segment_id = label_divisor * pred_class
            if current_segment_id == 0:
                # In order to handle the confusion with void
                current_segment_id = 1
            stuff_memory_list[int(pred_class)] = current_segment_id
    else:
        if pred_class in class_id_tracker:
            new_ins_id = class_id_tracker[pred_class]
        else:
            class_id_tracker[pred_class] = 1
            new_ins_id = 1
        class_id_tracker[pred_class] += 1
        current_segment_id = label_divisor * pred_class + new_ins_id
        foreground[mask] = 1

    mask_filtered = torch.where(mask == True)
    x_center = torch.median(mask_filtered[0])
    y_center = torch.median(mask_filtered[1])
    center = torch.tensor([x_center, y_center])
    center_list.append(center)

    panoptic_seg[mask] = current_segment_id
    return current_segment_id


def put_thing_names_on_image(point_list, semantic, output_image, class_names, thing_list):
    font_scale = 0.6
    thickness = 1
    for point in point_list:
        semantic_class = semantic[0, point[0], point[1]]
        # selected_axs.text(point[1], point[0], str(semantic_class), fontsize=5)
        color = (output_image[point[0], point[1], :] < 125) * 255
        color = (int(color[0]), int(color[1]), int(color[2]))
        if semantic_class >= len(class_names):
            continue
        # if semantic_class not in thing_list:
        #     continue
        test_size = cv2.getTextSize(
            class_names[semantic_class], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, thickness=thickness)

        width_shift = test_size[0][0] // 2
        cv2.putText(img=output_image,
                    text=class_names[semantic_class],
                    org=(point[1] - width_shift, point[0]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA)


def panoptic_inference_mask2former(
        panoptic_mask_function, outputs, number_of_classes, object_mask_threshold,
        ignore_label, thing_list, overlap_threshold, label_divisor
):
    mask_cls = outputs["pred_logits"]
    mask_pred = outputs["pred_masks"]
    scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
    mask_pred = mask_pred.sigmoid()

    keep = labels.ne(number_of_classes) & (scores > object_mask_threshold)
    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_masks = mask_pred[keep]
    # cur_mask_cls = mask_cls[keep]
    # cur_mask_cls = cur_mask_cls[:, :-1]

    cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

    h, w = cur_masks.shape[-2:]
    panoptic_seg = ignore_label * torch.ones((h, w), dtype=torch.int32, device=cur_masks.device)
    foreground = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
    segments_info = []
    center_list = []

    if cur_masks.shape[0] == 0:
        # We didn't detect any mask :(
        pass
    else:
        # take argmax
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list = {}
        class_id_tracker = {}
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = pred_class in thing_list
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < overlap_threshold:
                    continue

                # merge stuff regions
                panoptic_mask_function(
                    pred_class, isthing, panoptic_seg, mask, foreground, center_list, class_id_tracker,
                    stuff_memory_list, segments_info, label_divisor
                )
    if center_list:
        center_points = torch.stack(center_list, 0).unsqueeze(0)
    else:
        center_points = torch.empty([1, 0, 2])

    semantic_from_panoptic = panoptic_seg // 1000

    outputs = {
        "panoptic": panoptic_seg.unsqueeze(0),
        "foreground": foreground.unsqueeze(0),
        "semantic": semantic_from_panoptic.unsqueeze(0),
        "center_points": center_points,
        "segments_info": segments_info
    }
    return outputs


def panoptic_inference_deeplab(panoptic_mask_function, outputs, thing_list,
                               stuff_area, threshold=0.5, nms_kernel=3,
                               top_k=None, ignore_label=0, label_divisor=1000):
    sem = outputs["semantic"]
    ctr_hmp = outputs["center"]
    offsets = outputs["offset"]

    semantic = get_semantic_segmentation(sem)
    h, w = semantic.shape[-2:]

    panoptic_seg = ignore_label * torch.ones((h, w), dtype=torch.int32, device=semantic.device)
    foreground = torch.zeros((h, w), dtype=torch.int32, device=semantic.device)

    segments_info = []
    center_list = []

    instance, center = get_instance_segmentation(semantic, ctr_hmp, offsets, thing_list,
                                                 threshold=threshold, nms_kernel=nms_kernel, top_k=top_k)

    thing_seg = instance > 0
    semantic_thing_seg = torch.zeros_like(semantic)
    for thing_class in thing_list:
        semantic_thing_seg[semantic == thing_class] = 1

    # keep track of instance id for each class
    class_id_tracker = {}
    stuff_memory_list = {}

    # paste thing by majority voting
    instance_ids = torch.unique(instance)
    for ins_id in instance_ids:
        if ins_id == 0:
            continue
        # Make sure only do majority voting within semantic_thing_seg
        thing_mask = (instance == ins_id) & (semantic_thing_seg == 1)
        if torch.nonzero(thing_mask).size(0) == 0:
            continue
        class_id, _ = torch.mode(semantic[thing_mask].view(-1, ))
        thing_mask = thing_mask & (semantic == class_id)
        pred_class = class_id.item()
        panoptic_mask_function(
            pred_class, True, panoptic_seg, thing_mask[0], foreground, center_list, class_id_tracker,
            stuff_memory_list, segments_info, label_divisor
        )

    # paste stuff to unoccupied area
    class_ids = torch.unique(semantic)
    for class_id in class_ids:
        if class_id.item() in thing_list:
            # thing class
            continue
        # calculate stuff area
        stuff_mask = (semantic == class_id) & (~thing_seg)
        area = torch.nonzero(stuff_mask).size(0)
        if area >= stuff_area:
            pred_class = class_id.item()
            panoptic_mask_function(
                pred_class, False, panoptic_seg, stuff_mask[0], foreground, center_list, class_id_tracker,
                stuff_memory_list, segments_info, label_divisor
            )

    if center_list:
        center_points = torch.stack(center_list, 0).unsqueeze(0)
    else:
        center_points = torch.empty([1, 0, 2])

    outputs = {
        "panoptic": panoptic_seg.unsqueeze(0),
        "foreground": foreground.unsqueeze(0),
        "center_points": center_points,
        "semantic": semantic,
        "segments_info": segments_info
    }
    return outputs


class PanopticDeepLabTargetGenerator(object):
    """
    Generates training targets for Panoptic-DeepLab.
    """

    def __init__(
            self,
            ignore_label,
            thing_list,
            sigma=8,
            ignore_stuff_in_offset=False,
            small_instance_area=0,
            small_instance_weight=1,
            ignore_crowd_in_semantic=False,
    ):
        """
        Args:
            ignore_label: Integer, the ignore label for semantic segmentation.
            thing_list: Set, a set of ids from contiguous category ids belonging
                to thing categories.
            sigma: the sigma for Gaussian kernel.
            ignore_stuff_in_offset: Boolean, whether to ignore stuff region when
                training the offset branch.
            small_instance_area: Integer, indicates largest area for small instances.
            small_instance_weight: Integer, indicates semantic loss weights for
                small instances.
            ignore_crowd_in_semantic: Boolean, whether to ignore crowd region in
                semantic segmentation branch, crowd region is ignored in the original
                TensorFlow implementation.
        """
        self.ignore_label = ignore_label
        self.thing_ids = set(thing_list)
        self.ignore_stuff_in_offset = ignore_stuff_in_offset
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight
        self.ignore_crowd_in_semantic = ignore_crowd_in_semantic

        # Generate the default Gaussian image for each center
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, panoptic, segments):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py  # noqa
        reference: https://github.com/facebookresearch/detectron2/blob/main/datasets/prepare_panoptic_fpn.py#L18  # noqa

        Args:
            panoptic: numpy.array, panoptic label, we assume it is already
                converted from rgb image by panopticapi.utils.rgb2id.
            segments_info (list[dict]): see detectron2 documentation of "Use Custom Datasets".

        Returns:
            A dictionary with fields:
                - sem_seg: Tensor, semantic label, shape=(H, W).
                - center: Tensor, center heatmap, shape=(H, W).
                - center_points: List, center coordinates, with tuple
                    (y-coord, x-coord).
                - offset: Tensor, offset, shape=(2, H, W), first dim is
                    (offset_y, offset_x).
                - sem_seg_weights: Tensor, loss weight for semantic prediction,
                    shape=(H, W).
                - center_weights: Tensor, ignore region of center prediction,
                    shape=(H, W), used as weights for center regression 0 is
                    ignore, 1 is has instance. Multiply this mask to loss.
                - offset_weights: Tensor, ignore region of offset prediction,
                    shape=(H, W), used as weights for offset regression 0 is
                    ignore, 1 is has instance. Multiply this mask to loss.
        """
        segments_info = segments
        height, width = panoptic.shape[0], panoptic.shape[1]
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label
        center = np.zeros((height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord, x_coord = np.meshgrid(
            np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij"
        )
        # Generate pixel-wise loss weights
        semantic_weights = np.ones_like(panoptic, dtype=np.uint8)
        # 0: ignore, 1: has instance
        # three conditions for a region to be ignored for instance branches:
        # (1) It is labeled as `ignore_label`
        # (2) It is crowd region (iscrowd=1)
        # (3) (Optional) It is stuff region (for offset branch)
        center_weights = np.zeros_like(panoptic, dtype=np.uint8)
        offset_weights = np.zeros_like(panoptic, dtype=np.uint8)
        for seg in segments_info:
            cat_id = seg["category_id"]
            if not (self.ignore_crowd_in_semantic and seg["iscrowd"]):
                semantic[panoptic == seg["id"]] = cat_id
            if not seg["iscrowd"]:
                # Ignored regions are not in `segments_info`.
                # Handle crowd region.
                center_weights[panoptic == seg["id"]] = 1
                if not self.ignore_stuff_in_offset or cat_id in self.thing_ids:
                    offset_weights[panoptic == seg["id"]] = 1
            if cat_id in self.thing_ids:
                # find instance center
                mask_index = np.where(panoptic == seg["id"])
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[panoptic == seg["id"]] = self.small_instance_weight

                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])

                # generate center heatmap
                y, x = int(round(center_y)), int(round(center_x))
                sigma = self.sigma
                # upper left
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                # bottom right
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                # start and end indices in default Gaussian image
                gaussian_x0, gaussian_x1 = max(0, -ul[0]), min(br[0], width) - ul[0]
                gaussian_y0, gaussian_y1 = max(0, -ul[1]), min(br[1], height) - ul[1]

                # start and end indices in center heatmap image
                center_x0, center_x1 = max(0, ul[0]), min(br[0], width)
                center_y0, center_y1 = max(0, ul[1]), min(br[1], height)
                center[center_y0:center_y1, center_x0:center_x1] = np.maximum(
                    center[center_y0:center_y1, center_x0:center_x1],
                    self.g[gaussian_y0:gaussian_y1, gaussian_x0:gaussian_x1],
                )

                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset[0][mask_index] = center_y - y_coord[mask_index]
                offset[1][mask_index] = center_x - x_coord[mask_index]

        center_weights = center_weights[None]
        offset_weights = offset_weights[None]

        if len(center_pts) == 0:
            center_pts.append([0, 0])

        center = center[None,]
        return dict(
            semantic=semantic,
            center=center,
            center_points=np.array(center_pts),
            offset=offset,
            panoptic=panoptic,
            semantic_weights=semantic_weights,
            center_weights=center_weights,
            offset_weights=offset_weights
        )


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_mean(tensor):
    num_gpus = get_world_size()
    total = reduce_sum(tensor)
    return total.float() / num_gpus


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            pass
        else:
            raise


def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.

    Args:
        func: a stateless callable that takes tensor-like objects as arguments

    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = "cuda" in x.device.type and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cpu")
        else:
            return x

    def maybe_to_cpu_to_call(x):
        if isinstance(x, torch.Tensor):
            return maybe_to_cpu(x)
        elif "out" in x:
            new_returned_dict = {}
            for key, value in x["out"].items():
                new_returned_dict[key] = maybe_to_cpu(value)
            return {"out": new_returned_dict}
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info("Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func)))
        new_args = (maybe_to_cpu_to_call(x) for x in args)
        new_kwargs = {k: maybe_to_cpu_to_call(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped


def get_semantic_segmentation(sem):
    """
    Post-processing for semantic segmentation branch.
    Arguments:
        sem: A Tensor of shape [N, C, H, W], where N is the batch size, for consistent, we only
            support N=1.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    Raises:
        ValueError, if batch size is not 1.
    """
    if sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    sem = sem.squeeze(0)
    return torch.argmax(sem, dim=0, keepdim=True)


# Copyright (c) Facebook, Inc. and its affiliates.
# Reference: https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py  # noqa


def find_instance_center(center_heatmap, threshold=0.1, nms_kernel=3, top_k=None):
    """
    Find the center points from the center heatmap.
    Args:
        center_heatmap: A Tensor of shape [1, H, W] of raw center heatmap output.
        threshold: A float, threshold applied to center heatmap score.
        nms_kernel: An integer, NMS max pooling kernel size.
        top_k: An integer, top k centers to keep.
    Returns:
        A Tensor of shape [K, 2] where K is the number of center points. The
            order of second dim is (y, x).
    """
    # Thresholding, setting values below threshold to -1.
    center_heatmap = F.threshold(center_heatmap, threshold, -1)

    # NMS
    nms_padding = (nms_kernel - 1) // 2
    center_heatmap_max_pooled = F.max_pool2d(
        center_heatmap, kernel_size=nms_kernel, stride=1, padding=nms_padding
    )
    center_heatmap[center_heatmap != center_heatmap_max_pooled] = -1

    # Squeeze first two dimensions.
    center_heatmap = center_heatmap.squeeze()
    assert len(center_heatmap.size()) == 2, "Something is wrong with center heatmap dimension."

    # Find non-zero elements.
    if top_k is None:
        return torch.nonzero(center_heatmap > 0)
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(center_heatmap), top_k)
        return torch.nonzero(center_heatmap > top_k_scores[-1].clamp_(min=0))


def group_pixels(center_points, offsets):
    """
    Gives each pixel in the image an instance id.
    Args:
        center_points: A Tensor of shape [K, 2] where K is the number of center points.
            The order of second dim is (y, x).
        offsets: A Tensor of shape [2, H, W] of raw offset output. The order of
            second dim is (offset_y, offset_x).
    Returns:
        A Tensor of shape [1, H, W] with values in range [1, K], which represents
            the center this pixel belongs to.
    """

    offsets = offsets.squeeze(0)
    height, width = offsets.size()[1:]

    # Generates a coordinate map, where each location is the coordinate of
    # that location.
    y_coord, x_coord = torch.meshgrid(
        torch.arange(height, dtype=offsets.dtype, device=offsets.device),
        torch.arange(width, dtype=offsets.dtype, device=offsets.device),
    )
    coord = torch.cat((y_coord.unsqueeze(0), x_coord.unsqueeze(0)), dim=0)

    center_loc = coord + offsets
    center_loc = center_loc.flatten(1).T.unsqueeze_(0)  # [1, H*W, 2]
    center_points = center_points.unsqueeze(1)  # [K, 1, 2]

    # Distance: [K, H*W].
    distance = torch.norm(center_points - center_loc, dim=-1)

    # Finds center with minimum distance at each location, offset by 1, to
    # reserve id=0 for stuff.
    instance_id = torch.argmin(distance, dim=0).reshape((1, height, width)) + 1
    return instance_id


def get_instance_segmentation(
    sem_seg, center_heatmap, offsets, thing_ids, threshold=0.1, nms_kernel=3, top_k=None
):
    """
    Post-processing for instance segmentation, gets class agnostic instance id.
    Args:
        sem_seg: A Tensor of shape [1, H, W], predicted semantic label.
        center_heatmap: A Tensor of shape [1, H, W] of raw center heatmap output.
        offsets: A Tensor of shape [2, H, W] of raw offset output. The order of
            second dim is (offset_y, offset_x).
        thing_seg: A Tensor of shape [1, H, W], predicted foreground mask,
            if not provided, inference from semantic prediction.
        thing_ids: A set of ids from contiguous category ids belonging
            to thing categories.
        threshold: A float, threshold applied to center heatmap score.
        nms_kernel: An integer, NMS max pooling kernel size.
        top_k: An integer, top k centers to keep.
    Returns:
        A Tensor of shape [1, H, W] with value 0 represent stuff (not instance)
            and other positive values represent different instances.
        A Tensor of shape [1, K, 2] where K is the number of center points.
            The order of second dim is (y, x).
    """

    thing_seg = torch.zeros_like(sem_seg)
    for thing_class in list(thing_ids):
        thing_seg[sem_seg == thing_class] = 1

    center_points = find_instance_center(
        center_heatmap, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k
    )
    if center_points.size(0) == 0:
        return torch.zeros_like(sem_seg), center_points.unsqueeze(0)
    ins_seg = group_pixels(center_points, offsets)
    return thing_seg * ins_seg, center_points.unsqueeze(0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;
    Args:
        obj (class instance or method): an object to extract info from.
    """
    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")


def calc_runtime(func, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    out = func(*args, **kwargs)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    try:
        print(func.__name__ + ': ' + str(start.elapsed_time(end)))
    except:
        print(func.__class__.__name__ + ': ' + str(start.elapsed_time(end)))

    return out


class Camera:

  K = np.zeros([3, 3])
  R = np.zeros([3, 3])
  t = np.zeros([3, 1])
  P = np.zeros([3, 4])

  def setK(self, fx, fy, px, py):
    self.K[0, 0] = fx
    self.K[1, 1] = fy
    self.K[0, 2] = px
    self.K[1, 2] = py
    self.K[2, 2] = 1.0

  def setR(self, y, p, r):

    Rz = np.array([[np.cos(-y), -np.sin(-y), 0.0], [np.sin(-y), np.cos(-y), 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[np.cos(-p), 0.0, np.sin(-p)], [0.0, 1.0, 0.0], [-np.sin(-p), 0.0, np.cos(-p)]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-r), -np.sin(-r)], [0.0, np.sin(-r), np.cos(-r)]])
    Rs = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]) # switch axes (x = -y, y = -z, z = x)
    self.R = Rs.dot(Rz.dot(Ry.dot(Rx)))

  def setT(self, XCam, YCam, ZCam):
    X = np.array([XCam, YCam, ZCam])
    self.t = -self.R.dot(X)

  def updateP(self):
    Rt = np.zeros([3, 4])
    Rt[0:3, 0:3] = self.R
    Rt[0:3, 3] = self.t
    self.P = self.K.dot(Rt)

  def __init__(self, config):
    self.setK(config["fx"], config["fy"], config["px"], config["py"])
    self.setR(np.deg2rad(config["yaw"]), np.deg2rad(config["pitch"]), np.deg2rad(config["roll"]))
    self.setT(config["XCam"], config["YCam"], config["ZCam"])
    self.updateP()

class IPM:
  def __init__(self, fx, fy, u0, v0, x, y, z, yaw, pitch, roll, img_shape):

    self.img_shape = img_shape
    camera_config = dict(fx=fx, fy=fy, px=u0, py=v0, XCam=x, YCam=y, ZCam=z, yaw=yaw, pitch=pitch, roll=roll)

    drone_config = dict(fx=682.578, fy=682.578, px=482.0, py=302.0, XCam=0.0, YCam=0.0, ZCam=18.0)
    cam = Camera(camera_config)

    outputRes = (int(2 * drone_config["py"]), int(2 * drone_config["px"]))
    self.outputRes = outputRes
    dx = outputRes[1] / drone_config["fx"] * drone_config["ZCam"]
    dy = outputRes[0] / drone_config["fy"] * drone_config["ZCam"]
    pxPerM = (outputRes[0] / dy, outputRes[1] / dx)

    # setup mapping from street/top-image plane to world coords
    # shift = (outputRes[0] / 2.0, outputRes[1] / 2.0)
    shift = (outputRes[0] / 2.0, -120.0)

    shift = shift[0] + drone_config["YCam"] * pxPerM[0], shift[1] - drone_config["XCam"] * pxPerM[1]
    M = np.array([[1.0 / pxPerM[1], 0.0, -shift[1] / pxPerM[1]],
                  [0.0, -1.0 / pxPerM[0], shift[0] / pxPerM[0]], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # find IPM as inverse of P*M
    self.IPM = np.linalg.inv(cam.P.dot(M))
    self.inv_IPM = cam.P.dot(M)

  def front_2_bev(self, image):
      warped1 = cv2.warpPerspective(image, self.IPM, (self.outputRes[1], self.outputRes[0]), flags=cv2.INTER_LINEAR)
      return warped1
  def bev_2_front(self, bev):
      front = cv2.warpPerspective(bev, self.inv_IPM, (self.img_shape[1], self.img_shape[0]), flags=cv2.INTER_LINEAR)
      return front
