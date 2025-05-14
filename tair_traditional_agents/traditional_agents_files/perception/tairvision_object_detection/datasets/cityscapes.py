import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import numpy as np
from .generic_data import GenericSegmentationVisionDataset
from tairvision.utils import PanopticDeepLabTargetGenerator


# For panoptic segmentation, there is a need to utilize cityscapesscripts/preparation/createPanopticImgs.py
class Cityscapes(GenericSegmentationVisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def __init__(
            self,
            prediction_mode: Optional[Any] = False,
            number_of_temporal_frames=None,
            temporal_stride=None,
            small_instance_area=4096,
            small_instance_weight=3,
            **kwargs
    ) -> None:
        super(Cityscapes, self).__init__(**kwargs)
        if "coarse" in self.target_type[0]:
            self.mode = 'gtCoarse'
            if self.split == "train":
                self.split = "train_extra"
        else:
            self.mode = 'gtFine'

        self.root = os.path.join(self.root, "cityscapes")
        self.images_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        self.video_frames_dir = os.path.join(self.root, 'leftImg8bit_sequence', self.split)
        self.targets_dir = os.path.join(self.root, self.mode, self.split)
        self.images = []
        self.targets = []
        self.number_of_temporal_frames = number_of_temporal_frames
        self.temporal_stride = temporal_stride

        if "panoptic" in self.target_type[0]:
            annotations_filename = os.path.join(self.root, self.mode, f"{self.split}.json")
            with open(annotations_filename) as annotations_file:
                annotations_raw = json.load(annotations_file)
            self.panoptic_annotations = {}
            for annotation in annotations_raw['annotations']:
                self.panoptic_annotations[annotation["image_id"]] = annotation['segments_info']
            self.categories = annotations_raw["categories"]

            self.panoptic_json_file = annotations_filename
            self.panoptic_folder_path = os.path.join(self.targets_dir, "panoptic")

            if "panoptic-deeplab" == self.target_type[0]:
                self.panoptic_target_generator = PanopticDeepLabTargetGenerator(
                    ignore_label=self.panoptic_ignore_label,
                    thing_list=self.thing_list,
                    sigma=10,
                    ignore_stuff_in_offset=True,
                    small_instance_area=small_instance_area,
                    small_instance_weight=small_instance_weight,
                    ignore_crowd_in_semantic=False
                )

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            if self.split == 'train_extra':
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))

            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))
            elif self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')
        image_dir_contents = os.listdir(self.images_dir)
        image_dir_contents.sort()
        for city in image_dir_contents:
            img_dir = os.path.join(self.images_dir, city)
            video_dir = os.path.join(self.video_frames_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            image_contents = os.listdir(img_dir)
            image_contents.sort()
            for file_name in image_contents:
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))
                    target_types.append(os.path.join(target_dir, target_name))
                self.targets.append(target_types)

                if self.number_of_temporal_frames is None:
                    self.images.append(os.path.join(img_dir, file_name))
                else:
                    image_list = []
                    get_frame_index_of_target = int(file_name[-18:][:2])
                    if prediction_mode:
                        get_frame_index_of_target = get_frame_index_of_target - self.temporal_stride

                    number_list = list(np.arange(get_frame_index_of_target,
                                                 get_frame_index_of_target - self.temporal_stride * self.number_of_temporal_frames,
                                                 -self.temporal_stride))
                    number_list = number_list[::-1]

                    for frame_index in number_list:
                        if frame_index < 0:
                            new_index = int(file_name[-19:][:3])
                            new_index = new_index - frame_index
                            filename_copy = f"{file_name[:-19]}{new_index:03d}{file_name[-16:]}"
                        else:
                            filename_copy = f"{file_name[:-18]}{frame_index:02d}{file_name[-16:]}"
                        image_list.append(os.path.join(video_dir, filename_copy))
                    self.images.append(image_list)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        if not isinstance(self.images[index], List):
            image = Image.open(self.images[index]).convert('RGB')
        else:
            image = []
            for single_image in self.images[index]:
                image.append(Image.open(single_image).convert('RGB'))

        if not isinstance(self.images[index], List):
            image_id = self.images[index].split('/')[-1][:-16]
            info_dict = {"image_id": image_id,
                         "file_name": image_id + '.png',
                         "image_size": np.asarray(image).shape}
        else:
            # TODO, image id for the temporal dataloader
            pass

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
                # target = np.array(target)
                # target_tensor = torch.from_numpy(target_numpy)

            if "panoptic" in t:
                target_key = self.targets[index][i].split('/')[-1][:-20]
                target_info = self.panoptic_annotations[target_key]
                info_dict.update({"target_info": target_info})
                target = np.array(target)
                target = self.rgb2id(target)
                target = Image.fromarray(target)

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if "panoptic" not in self.target_type[0]:
            target = np.array(target)
            target = self._convert_mask_labels(target)
            target = target.astype('uint8')
            target = Image.fromarray(target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.target_type[0] == "panoptic-deeplab":
            target, target_info = self.panoptic_ground_truth_conversion_to_constant_standard(
                panoptic=target,
                segment_infos=target_info
            )

            target = self.panoptic_target_generator(panoptic=np.array(target), segments=target_info)
            for key, value in target.items():
                if key == "semantic" or key == "foreground" or key == "mask":
                    value = Image.fromarray(value.astype(np.uint8))
                else:
                    pass
                target[key] = value
            _, target = self.ToTensor(None, target)

        elif self.target_type[0] == "panoptic":
            target = self.maskformer_panoptic_format_converter(target_info, target)

        if self.return_dataset_info is True:
            return image, target, info_dict
        else:
            return image, target

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if 'instance' in target_type:
            return '{}_instanceIds.png'.format(mode)
        if 'panoptic' in target_type:
            return '{}_panoptic.png'.format(mode)
        elif 'semantic' in target_type:
            return '{}_labelIds.png'.format(mode)
        elif 'color' in target_type:
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)

    def _valid_splits(self):
        valid_splits = ["train", "val", "test"]
        return valid_splits

    def _valid_target_types(self):
        valid_target_types = ["instance", "semantic", "polygon", "color", "semantic-coarse", "panoptic",
                              "panoptic-deeplab"]
        return valid_target_types

    def _determine_classes(self):
        return self.classes
