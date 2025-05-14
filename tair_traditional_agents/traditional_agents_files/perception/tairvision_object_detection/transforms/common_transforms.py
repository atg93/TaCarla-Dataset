from tairvision_object_detection.transforms import functional as F
from torchvision.transforms import transforms as T
from typing import Tuple, List, Optional, Union, Dict
import numpy as np
from PIL import Image
import torch
from torch import nn, Tensor
import random
from tairvision_object_detection.ops.boxes import box_xyxy_to_cxcywh


class ToTensor:
    def __call__(self, image: Union[Image.Image, torch.Tensor, np.ndarray],
                 target: Union[Image.Image, torch.Tensor, np.ndarray]) -> \
            Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]]:
        if isinstance(image, list):
            image_out = []
            for img in image:
                img = F.to_tensor(img)
                image_out.append(img)
            image = image_out
        elif isinstance(image, dict):
            raise ValueError("not implemented for the time being...")
        elif image is None:
            pass
        elif isinstance(image, torch.Tensor):
            pass
        else:
            image = F.to_tensor(image)

        if isinstance(target, list):
            target_out = []
            for t in target:
                t = torch.as_tensor(np.array(t), dtype=torch.int64)
                target_out.append(t)
            target = target_out
        elif isinstance(target, dict):
            for key, value in target.items():
                if key == "semantic" or key == "foreground" \
                        or key == "panoptic" or key == "mask"\
                        or key == "labels":
                    value = torch.as_tensor(np.array(value), dtype=torch.int64)
                elif key == "image_id":
                    continue
                else:
                    value = torch.as_tensor(np.array(value), dtype=torch.float32)
                target[key] = value
        elif target is None:
            pass
        else:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)

        return image, target

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class RandomPhotometricDistort(nn.Module):
    def __init__(
        self,
        contrast: Tuple[float] = (0.5, 1.5),
        saturation: Tuple[float] = (0.5, 1.5),
        hue: Tuple[float] = (-0.05, 0.05),
        brightness: Tuple[float] = (0.875, 1.125),
        p: float = 0.5,
    ):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        if r[6] < self.p:
            channels, _, _ = F.get_dimensions(image)
            permutation = torch.randperm(channels)

            is_pil = F._is_pil_image(image)
            if is_pil:
                image = F.pil_to_tensor(image)
                image = F.convert_image_dtype(image)
            image = image[..., permutation, :, :]
            if is_pil:
                image = F.to_pil_image(image)

        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = np.asarray(mean)
        self.std = np.asarray(std)

    def __call__(self, image, target):
        image = F.normalize(image, mean=0, std=1/self.std)
        image = F.normalize(image, mean=-self.mean, std=1)
        return image, target


class NormalizeAlsoBoxes(object):
    def __init__(self, mean, std, to_tensor_before=False):
        self.mean = mean
        self.std = std
        self.to_tensor_before = to_tensor_before

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            target["original_boxes"] = boxes
            if not self.to_tensor_before:
                boxes = torch.tensor(boxes)
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            if not self.to_tensor_before:
                boxes = np.array(boxes)
            target["boxes"] = boxes
        return image, target
