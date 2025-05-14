import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from PIL import Image
import numpy as np
from .generic_data import GenericSegmentationVisionDataset


class UrbanUnderstanding(GenericSegmentationVisionDataset):
    # Based on https://github.com/mcordts/cityscapesScripts
    UrbanClass = namedtuple('UrbanClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                           'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        UrbanClass('person', 0, 0, 'human', 0, False, False, (220, 20, 60)),
        UrbanClass('rider', 1, 1, 'human', 0, False, False, (255, 0, 0)),
        UrbanClass('car', 2, 2, 'vehicle', 0, False, False, (0, 0, 142)),
        UrbanClass('truck', 3, 3, 'vehicle', 0, False, False, (0, 0, 70)),
        UrbanClass('bus', 4, 4, 'vehicle', 0, False, False, (0, 60, 100)),
        UrbanClass('train', 5, 5, 'vehicle', 0, False, False, (0, 80, 100)),
        UrbanClass('motorcycle', 6, 6, 'vehicle', 0, False, False, (0, 0, 230)),
        UrbanClass('bicycle', 7, 7, 'vehicle', 1, False, False, (119, 11, 32)),
        UrbanClass('traffic light', 8, 8, 'object', 1, False, False, (250, 170, 30)),
        UrbanClass('traffic sign', 9, 9, 'object', 1, False, False, (220, 220, 0)),
        UrbanClass('road', 10, 10, 'flat', 1, False, False, (128, 64, 128)),
        UrbanClass('sidewalk', 11, 11, 'flat', 1, False, False, (244, 35, 232)),
        UrbanClass('ego vehicle', 13, 12, 'void', 1, False, False, (120, 10, 10)),
        UrbanClass('vegetation', 14, 13, 'nature', 1, False, False, (107, 142, 35)),
        UrbanClass('sky', 15, 14, 'nature', 1, False, False, (70, 130, 180)),
        UrbanClass('building', 16, 15, 'construction', 1, False, False, (70, 70, 70)),
    ]

    def __init__(
            self,
            **kwargs
    ) -> None:
        super(UrbanUnderstanding, self).__init__(**kwargs)

        split_dict = {'train': 'training',
                      'val': 'validation',
                      'test': 'testing'}

        cityscapes = os.path.join(self.root, "cityscapes", "leftImg8bit", self.split)
        bdd100k = os.path.join(self.root, "bdd100k", "images/10k", self.split)
        mapillary = os.path.join(self.root, "mapillary", f'{split_dict[self.split]}/images/')

        list_to_delete = ["3d581db5-2564fb7e.jpg", "52e3fd10-c205dec2.jpg", "78ac84ba-07bd30c2.jpg",
                          "781756b0-61e0a182.jpg", "9342e334-33d167eb.jpg", "80a9e37d-e4548ac1.jpg"]

        target_dir = os.path.join(self.root, "urban_understanding_V1", self.split)

        list_of_dirs = [bdd100k, mapillary]
        for city in os.listdir(cityscapes):
            list_of_dirs.append(os.path.join(cityscapes, city))

        if self.split == "train":
            cityscapes_extra = os.path.join(self.root, "cityscapes", "leftImg8bit", "train_extra")
            for city in os.listdir(cityscapes_extra):
                list_of_dirs.append(os.path.join(cityscapes_extra, city))

        images = []
        for dataset_dir in list_of_dirs:
            files = os.listdir(dataset_dir)
            for file in files:
                if file in list_to_delete:
                    continue
                images.append(os.path.join(dataset_dir, file))

        images.sort()

        self.images = images
        self.targets = []

        for image_file in self.images:
            image_id = image_file.split('/')[-1][:-4]
            self.targets.append([os.path.join(target_dir, f"{image_id}.png")])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        split_info = self.images[index].split('/')
        info_dict = {"image_id": split_info[-1][:-4],
                     "image_size": np.asarray(image).shape,
                     "dataset": split_info[1],
                     "split": split_info[2]}

        targets: Any = []
        for i, t in enumerate(self.target_type):
            target = Image.open(self.targets[index][i])

            target = np.array(target)
            target = self._convert_mask_labels(target)
            target = Image.fromarray(target)

            targets.append(target)

        if self.transforms is not None:
            image, target = self.transforms(image, targets)

        if self.return_dataset_info is True:
            return image, target, info_dict
        else:
            return image, target

    def __len__(self) -> int:
        return len(self.images)

    def _valid_target_types(self):
        valid_target_types = ["semantic"]
        return valid_target_types

    def _determine_classes(self):
        return self.classes
