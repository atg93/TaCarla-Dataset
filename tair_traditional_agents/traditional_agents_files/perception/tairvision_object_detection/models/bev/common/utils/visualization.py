import numpy as np
import matplotlib.pylab

from tairvision.models.bev.common.nuscenes.process import decide_mean_std
from tairvision.models.bev.lss.utils.network import NormalizeInverse, pack_sequence_dim
from tairvision.models.bev.lss.utils.geometry import create_bev_geometry, get_pixel_locations, map_pixels_to_bev


def image2bev(batch, cfg, bev_dimension):
    mean, std = decide_mean_std(cfg.PRETRAINED.LOAD_WEIGHTS)
    denormalize = NormalizeInverse(mean=mean, std=std)

    bev_size = (bev_dimension[0].item(), bev_dimension[1].item(), bev_dimension[2].item())

    x = batch['images']
    b, s, n, c, h, w = x.shape
    # Reshape
    x = denormalize(pack_sequence_dim(batch['images'], pack_n=True))
    intrinsics = pack_sequence_dim(batch['intrinsics'], pack_n=True)
    extrinsics = pack_sequence_dim(batch['cams_to_lidar'], pack_n=True)
    view = pack_sequence_dim(batch['view'].repeat(1, 1, n, 1, 1), pack_n=True)

    geometry = create_bev_geometry(bev_size).to(x.device)

    # get pixel locations and mask for invalid ones
    image_final_dim = (h, w)
    locations, mask = get_pixel_locations(geometry, intrinsics, extrinsics, view, image_final_dim, downsample_rate=1.00001)

    x = x.permute(0, 2, 3, 1).unsqueeze(1)
    x = map_pixels_to_bev(x, locations, mask, bev_size, n)
    x = x.mean(3)
    x = (x.cpu().numpy() * 255).astype(np.uint8)

    return x

def image2bev_carla(batch):
    mean, std = decide_mean_std(True)
    denormalize = NormalizeInverse(mean=mean, std=std)

    bev_size = (200, 200, 1)

    x = batch['images']
    b, s, n, c, h, w = x.shape

    # Reshape
    x = pack_sequence_dim(batch['images'], pack_n=True)
    intrinsics = pack_sequence_dim(batch['intrinsics'], pack_n=True)
    extrinsics = pack_sequence_dim(batch['cams_to_lidar'], pack_n=True)
    view = pack_sequence_dim(batch['view'].repeat(1, 1, n, 1, 1), pack_n=True)

    geometry = create_bev_geometry(bev_size).to(x.device)

    # get pixel locations and mask for invalid ones
    image_final_dim = (h, w)
    locations, mask = get_pixel_locations(geometry, intrinsics, extrinsics, view, image_final_dim, downsample_rate=1.00001)

    x = x.permute(0, 2, 3, 1).unsqueeze(1)
    x = map_pixels_to_bev(x, locations, mask, bev_size, n)
    x = x.mean(3)
    x = (x.cpu().numpy() * 255).astype(np.uint8)

    return x

DEFAULT_COLORMAP = matplotlib.pylab.cm.hot

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


# TODO: The followings are fiery inherited visualization functions, remove them in the future.
def generate_instance_colours(instance_map):
    # Most distinct 22 colors (kelly colors from https://stackoverflow.com/questions/470690/how-to-automatically-generate
    # -n-distinct-colors)
    # plus some colours from AD40k

    return {instance_id: INSTANCE_COLOURS[global_instance_id % len(INSTANCE_COLOURS)] for
            instance_id, global_instance_id in instance_map.items()
            }


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


def _normalise(image: np.ndarray) -> np.ndarray:
    lower = np.min(image)
    delta = np.max(image) - lower
    if delta == 0:
        delta = 1
    image = (image.astype(np.float32) - lower) / delta
    return image


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
