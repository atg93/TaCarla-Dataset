
from PIL import Image
import numpy as np
from collections import namedtuple
from .generic_data import GenericSegmentationVisionDataset
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import os

class Forddata(GenericSegmentationVisionDataset):

    CulaneClass = namedtuple('CulaneClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CulaneClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        CulaneClass('Line 1', 1, 1, 'line', 0, True, False, (180, 200, 30)),
        CulaneClass('Line 2', 2, 2, 'line', 0, True, False, (200, 100, 1)),
        CulaneClass('Line 3', 3, 3, 'line', 0, True, False, (1, 100, 200)),
        CulaneClass('Line 4', 4, 4, 'line', 0, True, False, (30, 200, 180))
    ]

    def __init__(self,
                 lane_width_radius=None,
                 **kwargs):
        super(Forddata,self).__init__(**kwargs)
        self.root = os.path.join(self.root, "forddata")
        self.data_dir_path = os.path.join(self.root)
        self.valid_modes   = self.valid_splits
        self.images  = []
        self.createIndex()

        self.lane_width_radius = lane_width_radius
        if lane_width_radius is not None:
            self.lane_width_radius = lane_width_radius * 2

        if self.lane_width_radius is not None and self.transforms is not None:
            for transformation in self.transforms.transforms:
                if type(transformation).__name__ == "Resize":
                    width = transformation.size[1]
                    height = transformation.size[0]
                    width_portion = self.lane_width_radius * width / 1640
                    height_portion = self.lane_width_radius * height / 590
                    self.lane_width_radius_for_metric_for_resized = int(np.ceil(np.sqrt(width_portion * height_portion)))
                elif type(transformation).__name__ == "RandomResize":
                    height = transformation.min_size
                    height_portion = self.lane_width_radius * height / 590
                    self.lane_width_radius_for_metric_for_resized = int(np.ceil(height_portion))


    def __len__(self):
        return len(self.images)

    def _valid_splits(self):
        valid_splits = ["train", "val", "test"]
        return valid_splits


    def _valid_target_types(self):
        valid_target_types = ["semantic"]
        return valid_target_types

    # def _determine_classes(self):
    #     return None

    def createIndex(self):
        listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.split))
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                self.images.append(os.path.join(self.data_dir_path,line))


    def __getitem__(self, idx):
        image_name = self.images[idx]
        image = Image.open(image_name).convert('RGB')
        if self.split is self.valid_modes[0] or self.split is self.valid_modes[1]:
            if self.transforms is not None:
                image, _ = self.transforms(image, None)
            return image, None
        else:
            if self.transforms is not None:
                image, _ = self.transforms(image, None)
            return image, image_name