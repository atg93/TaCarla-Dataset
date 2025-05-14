import torch
import torchvision
import numpy as np

from torch import nn, Tensor
from tairvision.transforms import functional as F
from tairvision.transforms.functional import InterpolationMode
from torchvision.transforms import transforms as T
from typing import List, Tuple, Dict, Optional, Union

from PIL import Image
import random


"""
    Do Not Erase the below lines
"""
from .coco_utils import ConvertCocoPolysToMask
from .bdd_utils import ConvertBDD100k
from tairvision.transforms.common_transforms import Compose, RandomSelect, RandomPhotometricDistort, ToTensor
from tairvision.transforms.common_transforms import Normalize, UnNormalize, NormalizeAlsoBoxes


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


def _flip_widerface_facial_keypoints(kps, width):
    flip_inds = [1, 0, 2, 4, 3]
    flipped_data = kps[:, flip_inds]
    mask = flipped_data[..., 0] > -1
    flipped_data[..., 0][mask] = width - flipped_data[..., 0][mask]
    return flipped_data


class CustomCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            # print("transform", t)
            try:
                image, target = t(image, target)
            except:
                image = np.array(image)
                img_transformed = t(image=image)
                image = img_transformed["image"]
                image = Image.fromarray(image)

        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    if not isinstance(target["masks"], torch.Tensor):
                        target["masks"] = torch.from_numpy(target["masks"])
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    if keypoints.shape[1] == 5:
                        keypoints = _flip_widerface_facial_keypoints(keypoints, width)
                    else:
                        keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target


class RandomIoUCrop(nn.Module):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1.0, min_aspect_ratio: float = 0.5,
                 max_aspect_ratio: float = 2.0, sampler_options: Optional[List[float]] = None, trials: int = 40):
        super().__init__()
        # Configuration similar to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L89-L174
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError('image should be 2/3 dimensional. Got {} dimensions.'.format(image.ndimension()))
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_h, orig_w = F.get_dimensions(image)

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return image, target

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    boxes, torch.tensor([[left, top, right, bottom]], dtype=boxes.dtype, device=boxes.device)
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                target["boxes"] = boxes
                target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)

                return image, target


class RandomZoomOut(nn.Module):
    def __init__(
        self, fill: Optional[List[float]] = None, side_range: Tuple[float, float] = (1.0, 4.0), p: float = 0.5
    ):
        super().__init__()
        if fill is None:
            fill = [0.0, 0.0, 0.0]
        self.fill = fill
        self.side_range = side_range
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f"Invalid canvas side range provided {side_range}.")
        self.p = p

    @torch.jit.unused
    def _get_fill_value(self, is_pil):
        # type: (bool) -> int
        # We fake the type to make it work on JIT
        return tuple(int(x) for x in self.fill) if is_pil else 0

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        if torch.rand(1) >= self.p:
            return image, target

        _, orig_h, orig_w = F.get_dimensions(image)

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)

        if torch.jit.is_scripting():
            fill = 0
        else:
            fill = self._get_fill_value(F._is_pil_image(image))

        image = F.pad(image, [left, top, right, bottom], fill=fill)
        if isinstance(image, torch.Tensor):
            # PyTorch's pad supports only integers on fill. So we need to overwrite the colour
            v = torch.tensor(self.fill, device=image.device, dtype=image.dtype).view(-1, 1, 1)
            image[..., :top, :] = image[..., :, :left] = image[..., (top + orig_h) :, :] = image[
                ..., :, (left + orig_w) :
            ] = v

        if target is not None:
            target["boxes"][:, 0::2] += left
            target["boxes"][:, 1::2] += top

        return image, target


class RandomCrop(nn.Module):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1.0):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError('image should be 2/3 dimensional. Got {} dimensions.'.format(image.ndimension()))
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        orig_w, orig_h = F.get_image_size(image)

        no_valid_option = True
        no_valid_count = 0
        while no_valid_option:
            no_valid_count += 1
            r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(1)

            if orig_w < orig_h:
                new_w = int(orig_w * r[0])
                new_h = new_w
            else:
                new_h = int(orig_h * r[0])
                new_w = new_h

            r = np.random.rand(2)
            left = int((orig_w - new_w) * r[0])
            top = int((orig_h - new_h) * r[1])
            right = left + new_w
            bottom = top + new_h

            # check for any valid boxes with centers within the crop area
            cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
            cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
            is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
            if is_within_crop_area.any():
                no_valid_option = False
            elif no_valid_count >= 10:
                return image, target

        # keep only valid boxes and perform cropping
        boxes = target["boxes"][is_within_crop_area]
        labels = target["labels"][is_within_crop_area]
        boxes[:, 0::2] -= left
        boxes[:, 1::2] -= top
        boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=new_w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=new_h)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = target["iscrowd"][is_within_crop_area]

        if "keypoints" in target.keys():
            keypoints = target["keypoints"][is_within_crop_area]
            mask = keypoints[:, :, 0] > -1
            keypoints[:, :, 0][mask] -= left
            keypoints[:, :, 1][mask] -= top

            mask = (keypoints[:, :, 0] < 0) | (keypoints[:, :, 0] > new_w) | (keypoints[:, :, 1] < 0) | (
                    keypoints[:, :, 1] > new_h)
            keypoints[mask] = -1

            target["keypoints"] = keypoints

        if "label_ids" in target.keys():
            target["label_ids"] = target["label_ids"][is_within_crop_area]
        if "categories" in target.keys():
            target["categories"] = [target["categories"][idx] for idx in np.where(is_within_crop_area)[0]]

        image = F.crop(image, top, left, new_h, new_w)

        return image, target


class Resize(nn.Module):
    def __init__(self, size: Tuple[int] = (640, 640)):
        super().__init__()
        self.size = size

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        # if target is None:
        #    raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError('image should be 2/3 dimensional. Got {} dimensions.'.format(image.ndimension()))
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        orig_w, orig_h = F.get_image_size(image)

        ratio_w = self.size[1] / orig_w
        ratio_h = self.size[0] / orig_h

        image = F.resize(image, self.size)

        if target is not None:
            boxes = target["boxes"]
            boxes[:, 0::2] = (boxes[:, 0::2] * ratio_w)
            boxes[:, 1::2] = (boxes[:, 1::2] * ratio_h)

            boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=self.size[1])
            boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=self.size[0])

            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            target["boxes"] = boxes
            target["area"] = area

            if "keypoints" in target.keys():
                mask = target["keypoints"][:, :, 0] > -1
                target["keypoints"][:, :, 0][mask] *= ratio_w
                target["keypoints"][:, :, 1][mask] *= ratio_h

                mask = (target["keypoints"][:, :, 0] < 0) | (target["keypoints"][:, :, 0] > self.size[1]) | (
                        target["keypoints"][:, :, 1] < 0) | (target["keypoints"][:, :, 1] > self.size[0])
                target["keypoints"][mask] = -1

        return image, target


@torch.jit.unused
def get_shape_onnx(image: Tensor) -> Tensor:
    from torch.onnx import operators
    return operators.shape_as_tensor(image)[-2:]

@torch.jit.unused
def fake_cast_onnx(v: Tensor) -> float:
    # ONNX requires a tensor but here we fake its type for JIT.
    return v


class ResizeWithMask(nn.Module):
    def __init__(self, skip_resize=False, min_size=800, max_size=1333, fixed_size: Optional[Tuple[int, int]] = None,
                 is_train=True):
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.skip_resize = skip_resize
        self.min_size = min_size
        self.max_size = max_size
        self.fixed_size = fixed_size
        self.is_train = is_train

    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None) \
            -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]

        if target is None:
            target: Dict[str, Tensor] = {}
        target["original_image_sizes"] = []
        target["original_image_sizes"].append((h, w))

        if self.is_train:
            if self.skip_resize:
                return image, target
            size = float(self.torch_choice(self.min_size))
            # FIXME assume for now that training uses the smallest scale
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])
        image, target = self.resize_image_and_masks(image, size, float(self.max_size), target, self.fixed_size,
                                                    self.is_train)

        # if not self.is_train:
        #     return image, target

        if "boxes" in target:
            if not isinstance(target["boxes"], torch.Tensor):
                bbox = torch.from_numpy(target["boxes"])
            else:
                bbox = target["boxes"]
            bbox = self.resize_boxes(bbox, (h, w), image.shape[-2:])
            target["correctly_resized_boxes"] = bbox
            if self.is_train:
                target["boxes"] = bbox

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = self.resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["correctly_resized_keypoints"] = keypoints
            if self.is_train:
                target["keypoints"] = bbox
        return image, target

    def resize_image_and_masks(self, image: Tensor, self_min_size: float, self_max_size: float,
                                target: Optional[Dict[str, Tensor]] = None,
                                fixed_size: Optional[Tuple[int, int]] = None,
                                is_train=True) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torchvision._is_tracing():
            im_shape = get_shape_onnx(image)
        else:
            im_shape = torch.tensor(image.shape[-2:])

        size: Optional[List[int]] = None
        scale_factor: Optional[float] = None
        recompute_scale_factor: Optional[bool] = None
        if fixed_size is not None:
            size = [fixed_size[1], fixed_size[0]]
        else:
            min_size = torch.min(im_shape).to(dtype=torch.float32)
            max_size = torch.max(im_shape).to(dtype=torch.float32)
            scale = torch.min(self_min_size / min_size, self_max_size / max_size)

            if torchvision._is_tracing():
                scale_factor = fake_cast_onnx(scale)
            else:
                scale_factor = scale.item()
            recompute_scale_factor = True

        image = torch.nn.functional.interpolate(image[None], size=size, scale_factor=scale_factor, mode='bilinear',
                                                recompute_scale_factor=recompute_scale_factor, align_corners=False)[0]

        if "masks" in target:
            mask = target["masks"]
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask.copy())

            mask = torch.nn.functional.interpolate(mask[:, None].float(), size=size, scale_factor=scale_factor,
                                                   recompute_scale_factor=recompute_scale_factor)[:, 0].byte()
            target["correctly_resized_masks"] = mask
            if is_train:
                target["masks"] = mask

        return image, target

    def resize_boxes(self, boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device) /
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    def resize_keypoints(self, keypoints: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
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

    def torch_choice(self, k: List[int]) -> int:
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0.0, float(len(k))).item())
        return k[index]



class ScaleJitter(nn.Module):
    """Randomly resizes the image and its bounding boxes  within the specified scale range.
    The class implements the Scale Jitter augmentation as described in the paper
    `"Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" <https://arxiv.org/abs/2012.07177>`_.
    Args:
        target_size (tuple of ints): The target size for the transform provided in (height, weight) format.
        scale_range (tuple of ints): scaling factor interval, e.g (a, b), then scale is randomly sampled from the
            range a <= scale <= b.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
    """

    def __init__(
            self,
            target_size: Tuple[int, int],
            scale_range: Tuple[float, float] = (0.1, 2.0),
            interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation

    def forward(
            self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_height, orig_width = F.get_dimensions(image)

        scale = self.scale_range[0] + torch.rand(1) * (self.scale_range[1] - self.scale_range[0])
        r = min(self.target_size[1] / orig_height, self.target_size[0] / orig_width) * scale
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                if not isinstance(target["masks"], torch.Tensor):
                    target["masks"] = torch.from_numpy(target["masks"])
                # target["masks"] = Image.fromarray(target["masks"])
                target["masks"] = F.resize(
                    target["masks"], [new_height, new_width], interpolation=InterpolationMode.NEAREST
                )
                # target["masks"] = np.array(target["masks"])
                target["masks"] = target["masks"].numpy()
        return image, target


class FixedSizeCrop(nn.Module):
    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        size = tuple(T._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.crop_height = size[0]
        self.crop_width = size[1]
        self.fill = fill  # TODO: Fill is currently respected only on PIL. Apply tensor patch.
        self.padding_mode = padding_mode

    def _pad(self, img, target, padding):
        # Taken from the functional_tensor.py pad
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        elif len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        else:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        padding = [pad_left, pad_top, pad_right, pad_bottom]
        img = F.pad(img, padding, self.fill, self.padding_mode)
        if target is not None:
            target["boxes"][:, 0::2] += pad_left
            target["boxes"][:, 1::2] += pad_top
            if "masks" in target:
                target["masks"] = torch.from_numpy(target["masks"])
                target["masks"] = F.pad(target["masks"], padding, 0, "constant")
                target["masks"] = target["masks"].numpy()

        return img, target

    def _crop(self, img, target, top, left, height, width):
        img = F.crop(img, top, left, height, width)
        if target is not None:
            boxes = target["boxes"]
            boxes[:, 0::2] -= left
            boxes[:, 1::2] -= top
            boxes[:, 0::2].clip(min=0, max=width)
            boxes[:, 1::2].clip(min=0, max=height)

            is_valid = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])

            target["boxes"] = boxes[is_valid]
            target["labels"] = target["labels"][is_valid]
            if "masks" in target:
                target["masks"] = torch.from_numpy(target["masks"])
                target["masks"] = F.crop(target["masks"][is_valid], top, left, height, width)
                target["masks"] = target["masks"].numpy()

        return img, target

    def forward(self, img, target=None):
        _, height, width = F.get_dimensions(img)
        new_height = min(height, self.crop_height)
        new_width = min(width, self.crop_width)

        if new_height != height or new_width != width:
            offset_height = max(height - self.crop_height, 0)
            offset_width = max(width - self.crop_width, 0)

            r = torch.rand(1)
            top = int(offset_height * r)
            left = int(offset_width * r)

            img, target = self._crop(img, target, top, left, new_height, new_width)

        pad_bottom = max(self.crop_height - new_height, 0)
        pad_right = max(self.crop_width - new_width, 0)
        if pad_bottom != 0 or pad_right != 0:
            img, target = self._pad(img, target, [0, 0, pad_right, pad_bottom])

        return img, target


class RandomShortestSize(nn.Module):
    def __init__(
        self,
        min_size: Union[List[int], Tuple[int], int],
        max_size: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.min_size = [min_size] if isinstance(min_size, int) else list(min_size)
        self.max_size = max_size
        self.interpolation = interpolation

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        _, orig_height, orig_width = F.get_dimensions(image)

        min_size = self.min_size[torch.randint(len(self.min_size), (1,)).item()]
        r = min(min_size / min(orig_height, orig_width), self.max_size / max(orig_height, orig_width))

        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                target["masks"] = torch.tensor(target["masks"])
                target["masks"] = F.resize(
                    target["masks"], [new_height, new_width], interpolation=InterpolationMode.NEAREST
                )
                target["masks"] = np.array(target["masks"])

        return image, target


class PILToTensor(nn.Module):
    """Convert a ``PIL Image`` to a tensor of the same type. This transform does not support torchscript.
    Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
    """

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype(nn.Module):
    """Convert a tensor image to the given ``dtype`` and scale the values accordingly
    This function does not support PIL Image.
    Args:
        dtype (torch.dtype): Desired data type of the output
    .. note::
        When converting from a smaller to a larger integer ``dtype`` the maximum values are **not** mapped exactly.
        If converted back and forth, this mismatch has no effect.
    Raises:
        RuntimeError: When trying to cast :class:`torch.float32` to :class:`torch.int32` or :class:`torch.int64` as
            well as for trying to cast :class:`torch.float64` to :class:`torch.int64`. These conversions might lead to
            overflow errors since the floating point ``dtype`` cannot store consecutive integers over the whole range
            of the integer ``dtype``.
    """

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = np.array([w, h], dtype=np.float32)
        cropped_boxes = boxes - np.array([j, i, j, i])
        cropped_boxes = np.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clip(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")


    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = np.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], 1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)
