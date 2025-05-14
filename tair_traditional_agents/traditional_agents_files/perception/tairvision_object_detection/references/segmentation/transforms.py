import numpy as np
from PIL import Image
import random
import math
import numbers
import torch
from torchvision import transforms as T
from tairvision_object_detection.transforms import functional as F
from typing import Tuple, List, Optional, Union, Dict
from collections.abc import Sequence
from torch import Tensor
from tairvision_object_detection.transforms.functional import InterpolationMode
import warnings
# Do Not Erase the below line
from tairvision_object_detection.transforms.common_transforms import Compose,\
    RandomSelect, RandomPhotometricDistort, ToTensor, Normalize, UnNormalize, NormalizeAlsoBoxes


def pad_if_smaller(img, size, fill=0):
    width, height = F.get_image_size(img)
    size_height = size[0]
    size_width = size[1]
    padding_width = size_width - width if width < size_width else 0
    padding_height = size_height - height if height < size_height else 0
    img = F.pad(img, [0, 0, padding_width, padding_height], fill=fill)
    return img


def pad_if_smaller_with_given_params(img, padding_borders, fill=0):
    img = F.pad(img, padding_borders, fill=fill)
    return img


class RandomResize(object):
    def __init__(self, min_size, max_size=None, limiting_max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size
        self.limiting_max_size = limiting_max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        if isinstance(image, list):
            image_out = []
            for t in image:
                t = F.resize(t, size, max_size=self.limiting_max_size)
                image_out.append(t)
            image = image_out
        else:
            image = F.resize(image, size, max_size=self.limiting_max_size)

        if isinstance(target, list):
            target_out = []
            for t in target:
                t = F.resize(t, size, interpolation=F.InterpolationMode.NEAREST, max_size=self.limiting_max_size)
                target_out.append(t)

            target = target_out
        elif isinstance(target, dict):
            pass
        elif target is None:
            pass
        else:
            target = F.resize(target, size, interpolation=F.InterpolationMode.NEAREST, max_size=self.limiting_max_size)

        return image, target


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, flip_prob=0.5, modify_label=None, number_of_class=None, ignore_label=255, background_class=0):
        super().__init__()
        self.flip_prob = flip_prob
        self.modify_label = modify_label
        self.number_of_class = number_of_class
        self.ignore_label = ignore_label
        self.background_class = background_class
        if modify_label and number_of_class is None:
            raise ValueError("if replace mask, the number of classes should be entered")
        if isinstance(self.modify_label, list):
            assert isinstance(self.number_of_class, list), "number of classes should also be in the list format"
            assert len(self.number_of_class) == len(self.modify_label), "the length of modify label and number of classes" \
                                                                        " should be the same"

    def __call__(self, image, target):
        sampled_random = random.random()
        if sampled_random < self.flip_prob:
            if isinstance(image, list):
                image_out = []
                for img in image:
                    img = F.hflip(img)
                    image_out.append(img)
                image = image_out
            elif isinstance(image, dict):
                raise ValueError("not implemented for the time being")
            else:
                image = F.hflip(image)

        if isinstance(target, list):
            if sampled_random < self.flip_prob:
                if not self.modify_label:
                    target_out = []
                    for t in target:
                        t = F.hflip(t)
                        target_out.append(t)
                    target = target_out
                elif isinstance(self.modify_label, list):
                    target_out = []
                    for t, modify_label, number_of_class in zip(target, self.modify_label, self.number_of_class):
                        if modify_label:
                            t = modify_label_function(t, number_of_class, self.background_class, self.ignore_label)
                        t_out = F.hflip(t)
                        target_out.append(t_out)
                    target = target_out
                elif self.modify_label is True:
                    target_out = []
                    for t in target:
                        t = modify_label_function(t, self.number_of_class, self.background_class, self.ignore_label)
                        t_out = F.hflip(t)
                        target_out.append(t_out)
                    target = target_out

        elif isinstance(target, dict):
            if self.modify_label:
                raise ValueError("Not implemented for the time being")
            if sampled_random < self.flip_prob:

                for key, value in target.items():
                    value = F.hflip(value)
                    target[key] = value

        else:
            if sampled_random < self.flip_prob:
                if self.modify_label:
                    target = modify_label_function(target, self.number_of_class,
                                                   self.background_class, self.ignore_label)
                if target is not None:
                    target = F.hflip(target)

        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '(p={})'.format(self.flip_prob)
        format_string += ', modify_label={0}'.format(self.modify_label)
        format_string += ', number_of_classes={0}'.format(self.number_of_class)


def modify_label_function(target, number_of_class, background_class, ignore_label):
    if isinstance(target, Image.Image):
        target_numpy = np.asarray(target).copy()
        selected = np.logical_and(target_numpy != background_class, target_numpy != ignore_label)
        target_numpy[selected] = number_of_class - target_numpy[selected]
        target = Image.fromarray(np.uint8(target_numpy))
    elif isinstance(target, torch.Tensor):
        selected = torch.logical_and(target != background_class, target != ignore_label)
        target[selected] = number_of_class - target[selected]
    else:
        raise ValueError("Not a suporrted format")
    return target


class RandomCrop(object):
    def __init__(self, size, fill=0):
        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))
        self.fill = fill

    def __call__(self, image, target):

        if not isinstance(image, list):
            image = pad_if_smaller(image, self.size, self.fill)
            crop_params = T.RandomCrop.get_params(image, self.size)
            image = F.crop(image, *crop_params)
        else:
            image_out = []
            for img in image:
                img = pad_if_smaller(img, self.size, self.fill)
                crop_params = T.RandomCrop.get_params(img, self.size)
                img = F.crop(img, *crop_params)
                image_out.append(img)
            image = image_out

        if not isinstance(target, list):
            if target is not None:
                target = pad_if_smaller(target, self.size, fill=255)
                target = F.crop(target, *crop_params)
        else:
            target_out = []
            for t in target:
                t = pad_if_smaller(t, self.size, fill=255)
                t = F.crop(t, *crop_params)
                target_out.append(t)
            target = target_out

        return image, target

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class RandomCropWithRandomPadding(object):
    def __init__(self, size, image_pad_intensity=0, target_pad_intensity=255):
        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))
        self.image_pad_intensity = image_pad_intensity
        self.target_pad_intensity = target_pad_intensity

    def __call__(self, image, target):

        width, height = F.get_image_size(image)
        size_height = self.size[0]
        size_width = self.size[1]
        padding_width = size_width - width if width < size_width else 0
        padding_height = size_height - height if height < size_height else 0

        lower_bound_height = random.randint(0, padding_height)
        upper_bound_height = padding_height - lower_bound_height

        lower_bound_width = random.randint(0, padding_width)
        upper_bound_width = padding_width - lower_bound_width

        padding_borders = [lower_bound_width, lower_bound_height, upper_bound_width, upper_bound_height]

        if self.image_pad_intensity is None:
            image_pad_intensity = (random.random()*255, random.random()*255, random.random()*255)
        else:
            image_pad_intensity = self.image_pad_intensity

        if not isinstance(image, list):
            # image = pad_if_smaller(image, self.size, fill=self.image_pad_intensity)
            image = pad_if_smaller_with_given_params(image, padding_borders, fill=image_pad_intensity)
            crop_params = T.RandomCrop.get_params(image, self.size)
            image = F.crop(image, *crop_params)
        else:
            image_out = []
            for img in image:
                # img = pad_if_smaller(img, self.size, fill=self.image_pad_intensity)
                img = pad_if_smaller_with_given_params(img, padding_borders, fill=image_pad_intensity)
                crop_params = T.RandomCrop.get_params(img, self.size)
                img = F.crop(img, *crop_params)
                image_out.append(img)
            image = image_out

        if not isinstance(target, list):
            # target = pad_if_smaller(target, self.size, fill=self.target_pad_intensity)
            target = pad_if_smaller_with_given_params(target, padding_borders, fill=self.target_pad_intensity)
            target = F.crop(target, *crop_params)
        else:
            target_out = []
            for t in target:
                # t = pad_if_smaller(t, self.size, fill=self.target_pad_intensity)
                t = pad_if_smaller_with_given_params(t, padding_borders, fill=self.target_pad_intensity)
                t = F.crop(t, *crop_params)
                target_out.append(t)
            target = target_out

        return image, target

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class CenterCrop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        if not isinstance(target, list):
            target = F.center_crop(target, self.size)
            return image, target
        else:
            target_out = []
            for t in target:
                t = F.center_crop(t, self.size)
                target_out.append(t)
            return image, target_out


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class RandomResizedCrop(torch.nn.Module):
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.

    """

    def __init__(self, size: List[int],
                 scale: List[float] = [0.08, 1.0],
                 ratio: List[float] = [3. / 4., 4. / 3.],
                 interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio

        self.image_interpolation = interpolation
        self.target_interpolation = InterpolationMode.NEAREST

    @staticmethod
    def get_params(
            img: Union[Image.Image, Tensor], scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F.get_image_size(img)
        area = height * width

        in_ratio = float(width) / float(height)
        ratio = [ratio[0] * in_ratio, ratio[1] * in_ratio]

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img: Union[Image.Image, Tensor],
                target: Union[Image.Image, Tensor]):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            :param img: PIL Image or Tensor: Randomly cropped and resized image.
            :param target: PIL Image or Tensor: Randomly cropped and resized mask.
        """

        if isinstance(img, list):
            i, j, h, w = self.get_params(img[-1], self.scale, self.ratio)
            image_out = []
            for image_sample in img:
                image_resized_cropped = F.resized_crop(image_sample, i, j, h, w, self.size, self.target_interpolation)
                image_out.append(image_resized_cropped)
            image_resized_cropped = image_out
        else:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            image_resized_cropped = F.resized_crop(img, i, j, h, w, self.size, self.image_interpolation)

        if isinstance(target, list):
            target_out = []
            for t in target:
                target_resized_cropped = F.resized_crop(t, i, j, h, w, self.size, self.target_interpolation)
                target_out.append(target_resized_cropped)
            target_resized_cropped = target_out
        else:
            target_resized_cropped = F.resized_crop(target, i, j, h, w, self.size, self.target_interpolation)
        return image_resized_cropped, target_resized_cropped

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        return format_string


class Resize(torch.nn.Module):
    """Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. warning::
        The output image might be different depending on its type: when downsampling, the interpolation of PIL images
        and tensors is slightly different, because PIL applies antialiasing. This may lead to significant differences
        in the performance of a network. Therefore, it is preferable to train and serve a model with the same input
        types.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image: if the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``, then
            the image is resized again so that the longer edge is equal to
            ``max_size``. As a result, ``size`` might be overruled, i.e the
            smaller edge may be shorter than ``size``. This is only supported
            if ``size`` is an int (or a sequence of length 1 in torchscript
            mode).
        antialias (bool, optional): antialias flag. If ``img`` is PIL Image, the flag is ignored and anti-alias
            is always used. If ``img`` is Tensor, the flag is False by default and can be set True for
            ``InterpolationMode.BILINEAR`` only mode.

            .. warning::
                There is no autodiff support for ``antialias=True`` option with input ``img`` as Tensor.

    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size

        self.interpolation_image = interpolation
        self.interpolation_mask = InterpolationMode.NEAREST

    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.
            target (PIL Image or Tensor): Mask to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
            :param target: PIL Image or Tensor: Rescaled mask.
        """
        if isinstance(img, list):
            image_out = []
            for single_img in img:
                single_img_p = F.resize(single_img, self.size, self.interpolation_image)
                image_out.append(single_img_p)
            img = image_out
        elif isinstance(img, dict):
            raise ValueError("Not implemented for the time being...")
        else:
            img = F.resize(img, self.size, self.interpolation_image)

        if isinstance(target, list):
            target_out = []
            for t in target:
                target_p = F.resize(t, self.size, self.interpolation_mask)
                target_out.append(target_p)
            target = target_out
        elif isinstance(target, dict):
            raise ValueError("Not implemented for the time being...")
        elif target is None:
            pass
        else:
            target = F.resize(target, self.size, self.interpolation_mask)

        return img, target

    def __repr__(self):
        interpolate_str = self.interpolation_image.value
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)


class RandomRotation(torch.nn.Module):

    def __init__(self, degrees,
                 image_interpolation=InterpolationMode.BILINEAR,
                 target_interpolation=InterpolationMode.NEAREST,
                 fill=0,
                 expand=False,
                 target_fill=255):
        super().__init__()
        self.degrees = degrees
        self.image_interpolation = image_interpolation
        self.target_interpolation = target_interpolation
        self.expand = expand
        self.fill = fill
        self.target_fill = target_fill
        self.center = None

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def forward(self, img, target):
        """
        Args:
            target:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """

        angle = self.get_params(self.degrees)
        if isinstance(img, list):
            image_out = []
            for single_img in img:
                single_img_p = F.rotate(single_img, angle, self.image_interpolation, self.expand, self.center, self.fill)
                image_out.append(single_img_p)
            image = image_out
        elif isinstance(img, dict):
            raise ValueError("Not implemented for the time being...")
        else:
            image = F.rotate(img, angle, self.image_interpolation, self.expand, self.center, self.fill)

        if isinstance(target, list):
            target_out = []
            for t in target:
                target_p = target = F.rotate(t, angle, self.target_interpolation, self.expand, self.center, self.target_fill)
                target_out.append(target_p)
            target = target_out
        elif isinstance(target, dict):
            raise ValueError("Not implemented for the time being...")
        elif target is None:
            pass
        else:
            target = F.rotate(target, angle, self.target_interpolation, self.expand, self.center, self.target_fill)
        return image, target


class JitterAspectRatio(torch.nn.Module):

    def __init__(self, ratio=None,
                 scale=None,
                 interpolation=InterpolationMode.BILINEAR):
        """

        :type scale: List[float]
        """
        super().__init__()

        if ratio is None:
            ratio = [3. / 4., 4. / 3.]
        if scale is None:
            scale = [1.0, 1.0]

        self.scale = scale
        self.ratio = ratio

        self.image_interpolation = interpolation
        self.target_interpolation = InterpolationMode.NEAREST

    @staticmethod
    def get_params(
            img: Union[Image.Image, Tensor], scale: List[float], ratio: List[float]
    ) -> Tuple[int, int]:

        width, height = F.get_image_size(img)
        area = height * width

        in_ratio = float(width) / float(height)
        ratio = [ratio[0] * in_ratio, ratio[1] * in_ratio]

        log_ratio = torch.log(torch.tensor(ratio))

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        return h, w

    def forward(self, img: Union[Image.Image, Tensor],
                target: Union[Image.Image, Tensor]):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            :param img: PIL Image or Tensor: Randomly cropped and resized image.
            :param target: PIL Image or Tensor: Randomly cropped and resized mask.
        """

        if isinstance(img, list):
            size = self.get_params(img[-1], self.scale, self.ratio)
            image_out = []
            for image_sample in img:
                image_resized_cropped = F.resize(image_sample, size, self.image_interpolation)
                image_out.append(image_resized_cropped)
            image_resized_cropped = image_out
        elif isinstance(img, dict):
            raise ValueError("Not implemented for the time being...")
        else:
            size = self.get_params(img, self.scale, self.ratio)
            image_resized_cropped = F.resize(img, size, self.image_interpolation)

        if isinstance(target, list):
            target_out = []
            for t in target:
                target_resized_cropped = F.resize(t, size, self.target_interpolation)
                target_out.append(target_resized_cropped)
            target_resized_cropped = target_out
        elif isinstance(target, dict):
            raise ValueError("Not implemented for the time being...")
        elif target is None:
            pass
        else:
            target_resized_cropped = F.resize(target, size, self.target_interpolation)

        return image_resized_cropped, target_resized_cropped


class RandomPerspective(torch.nn.Module):
    """Performs a random perspective transformation of the given image with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.
            Default is 0.5.
        p (float): probability of the image being transformed. Default is 0.5.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR,
                 fill=0, target_fill=255, target_interpolation=InterpolationMode.NEAREST):
        super().__init__()
        self.p = p

        self.image_interpolation = interpolation
        self.distortion_scale = distortion_scale
        self.target_interpolation = target_interpolation

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill
        self.target_fill = target_fill

    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be Perspectively transformed.

        Returns:
            PIL Image or Tensor: Randomly transformed image.
        """

        if torch.rand(1) < self.p:
            width, height = F.get_image_size(img)
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)

            if isinstance(img, list):
                image_out = []
                for single_img in img:
                    single_img_p = F.perspective(single_img, startpoints, endpoints, self.image_interpolation, self.fill)
                    image_out.append(single_img_p)
                img = image_out
            elif isinstance(img, dict):
                raise ValueError("Not implemented for the time being...")
            else:
                img = F.perspective(img, startpoints, endpoints, self.image_interpolation, self.fill)

            if isinstance(target, list):
                target_out = []
                for t in target:
                    target_resized_cropped = F.perspective(t, startpoints, endpoints,
                                   self.target_interpolation, self.target_fill)
                    target_out.append(target_resized_cropped)
                target = target_out
            elif isinstance(target, dict):
                raise ValueError("Not implemented for the time being...")
            elif target is None:
                pass
            else:
                target = F.perspective(target, startpoints, endpoints,
                                   self.target_interpolation, self.target_fill)
        return img, target

    @staticmethod
    def get_params(width: int, height: int, distortion_scale: float) -> Tuple[List[List[int]], List[List[int]]]:
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width (int): width of the image.
            height (int): height of the image.
            distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1, )).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1, )).item())
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1, )).item())
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1, )).item())
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1, )).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1, )).item())
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


