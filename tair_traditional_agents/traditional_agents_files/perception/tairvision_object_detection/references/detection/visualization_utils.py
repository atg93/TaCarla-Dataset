import numpy as np
import cv2

EPS = 1e-2


def vis_det_bboxes(in_img, bboxes, masks=None, thickness=2, color=None):
    in_img = np.ascontiguousarray(in_img)

    if masks is not None:
        if len(masks) != 0:
            in_img = draw_masks(in_img, masks[:, :, :].cpu().numpy(), with_edge=True, color=color)
    in_img = draw_bboxes(in_img, bboxes.cpu().numpy(), alpha=0.8, thickness=2)

    return in_img


def draw_masks(img, masks, color=None, with_edge=True, alpha=0.8):
    """Draw masks on the image and their edges on the axes.
    Args:
        ax (matplotlib.Axes): The input axes.
        img (ndarray): The image with the shape of (3, h, w).
        masks (ndarray): The masks with the shape of (n, h, w).
        color (ndarray): The colors for each masks with the shape
            of (n, 3).
        with_edge (bool): Whether to draw edges. Default: True.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
    Returns:
        matplotlib.Axes: The result axes.
        ndarray: The result image.
    """
    taken_colors = set([0, 0, 0])
    if color is None:
        random_colors = np.random.randint(0, 255, (masks.shape[0], 3))
        color = [tuple(c) for c in random_colors]
        color = np.array(color, dtype=np.uint8)
    if type(color) == list:
        color = np.array(color).astype('float')

    # polygons = []
    for i, mask in enumerate(masks):
        if with_edge:
            contours, _ = bitmap_to_polygon(mask)
            # polygons += [Polygon(c) for c in contours]

        color_mask = color[i]
        while tuple(color_mask) in taken_colors:
            color_mask = _get_bias_color(color_mask)
        taken_colors.add(tuple(color_mask))

        mask = mask.astype(bool)

        if img.dtype == np.float and color_mask.max() > 1.0:
            color_mask = color_mask / 255.0
        img[mask] = img[mask] * (1 - alpha) + color_mask * alpha

        img = cv2.polylines(img, contours, True, (1, 1, 1), 2)

    return img


def draw_bboxes(in_img, bboxes, color=[0, 255, 0], alpha=0.8, thickness=2):
    """Draw bounding boxes on the axes.
    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 4).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.
    Returns:
        matplotlib.Axes: The result axes.
    """
    # polygons = []
    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        start_point = (bbox_int[0], bbox_int[1])
        end_point = (bbox_int[2], bbox_int[3])
        in_img = cv2.rectangle(in_img, start_point, end_point, color, thickness)

    return in_img


def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.
    Args:
        bitmap (ndarray): masks in bitmap representation.
    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole


def _get_bias_color(base, max_dist=30):
    """Get different colors for each masks.
    Get different colors for each masks by adding a bias
    color to the base category color.
    Args:
        base (ndarray): The base category color with the shape
            of (3, ).
        max_dist (int): The max distance of bias. Default: 30.
    Returns:
        ndarray: The new color for a mask with the shape of (3, ).
    """
    new_color = base + np.random.randint(
        low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)
