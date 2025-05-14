import math

import torch
import torchvision
from torch import Tensor
from typing import List, Tuple, Dict

from tairvision_object_detection.models.detection.image_list import ImageList
from tairvision_object_detection.models.detection.roi_heads import paste_masks_in_image

# _onnx_batch_images() is an implementation of
# batch_images() that is supported by ONNX tracing.


@torch.jit.unused
def onnx_batch_images(images: List[Tensor], size_divisible: int = 32) -> Tensor:
    max_size = []
    for i in range(images[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    stride = size_divisible
    max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
    max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # which is not yet supported in onnx
    padded_imgs = []
    for img in images:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

    return torch.stack(padded_imgs)


def max_by_axis(the_list: List[List[int]]) -> List[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def batch_images(images: List[Tensor], size_divisible: int = 32) -> ImageList:
    images = [img for img in images]
    image_sizes = [img.shape[-2:] for img in images]

    # Batch images
    if torchvision._is_tracing():
        # batch_images() does not export well to ONNX
        # call _onnx_batch_images() instead
        batched_imgs = onnx_batch_images(images, size_divisible)
    else:
        max_size = max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    image_sizes_list: List[Tuple[int, int]] = []
    for image_size in image_sizes:
        assert len(image_size) == 2
        image_sizes_list.append((image_size[0], image_size[1]))

    image_list = ImageList(batched_imgs, image_sizes_list)
    return image_list


def resize_keypoints(keypoints: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=keypoints.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=keypoints.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    if torch._C._get_tracing_state():
        resized_data_0 = resized_data[:, :, 0] * ratio_w
        resized_data_1 = resized_data[:, :, 1] * ratio_h
        resized_data = torch.stack((resized_data_0, resized_data_1, resized_data[:, :, 2]), dim=2)
    else:
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
    return resized_data


def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [
              torch.tensor(s, dtype=torch.float32, device=boxes.device) /
              torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
              for s, s_orig in zip(new_size, original_size)]

    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def postprocess(result: List[Dict[str, Tensor]],
                image_shapes: List[Tuple[int, int]],
                original_image_sizes: List[Tuple[int, int]]
                ) -> List[Dict[str, Tensor]]:
    for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
        boxes = pred["boxes"]
        boxes = resize_boxes(boxes, im_s, o_im_s)
        result[i]["boxes"] = boxes
        if "masks" in pred:
            masks = pred["masks"]
            masks = paste_masks_in_image(masks, boxes, o_im_s)
            result[i]["masks"] = masks
        if "keypoints" in pred:
            keypoints = pred["keypoints"]
            keypoints = resize_keypoints(keypoints, im_s, o_im_s)
            result[i]["keypoints"] = keypoints
    return result


def get_original_image_sizes(targets):
    # get the original image sizes
    original_image_sizes: List[Tuple[int, int]] = []
    for target in targets:
        if "original_image_sizes" in target:
            if type(target) is str:
                original_image_sizes.append(targets["original_image_sizes"])
                original_image_sizes = original_image_sizes[0]
            else:
                original_image_sizes.append(target["original_image_sizes"][0])
    return original_image_sizes
