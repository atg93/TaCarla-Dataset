import os
import os.path as osp
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
import numpy as np
import torch
import cv2
from .generic_data import GenericSegmentationVisionDataset
import json
from tairvision.utils import IPM
from tairvision.references.segmentation.lane_utils import draw_line_on_image, simplistic_target, lane_with_radius_settings, obtain_ego_attributes
import warnings


class Llamas(GenericSegmentationVisionDataset):
    LlamasClass = namedtuple('LlamasClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        LlamasClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        LlamasClass('Line 1', 1, 1, 'line', 0, False, False, (255, 255, 255)),
    ]

    culane_classes = [
        LlamasClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        LlamasClass('Line 1', 1, 1, 'line', 0, True, False, (180, 200, 30)),
        LlamasClass('Line 2', 2, 2, 'line', 0, True, False, (200, 100, 1)),
        LlamasClass('Line 3', 3, 3, 'line', 0, True, False, (1, 100, 200)),
        LlamasClass('Line 4', 4, 4, 'line', 0, True, False, (30, 200, 180))
    ]

    image_height = 717
    image_width = 1276

    def __init__(self,
                 lane_width_radius=None,
                 lane_width_radius_for_metric=None,
                 lane_width_radius_for_uncertain=False,
                 include_junctions=False,
                 lane_width_radius_for_binary=None,
                 **kwargs) -> None:
        super(Llamas, self).__init__(**kwargs)

        self.root = os.path.join(self.root, "llamas")
        self.image_set = self.split
        self.valid_modes = self.valid_splits
        self.images = []
        self.exist_list = []
        self.annotations = []

        if self.split == "trainval":
            self.createIndex("train")
            self.createIndex("valid")
        elif self.split == "val":
            self.createIndex("valid")
        else:
            self.createIndex(self.split)

        lane_width_radius_dict = lane_with_radius_settings(lane_width_radius=lane_width_radius,
                                                           lane_width_radius_for_metric=lane_width_radius_for_metric,
                                                           lane_width_radius_for_uncertain=lane_width_radius_for_uncertain,
                                                           lane_width_radius_for_binary=lane_width_radius_for_binary,
                                                           transforms=self.transforms,
                                                           image_height=self.image_height, image_width=self.image_width)

        self.lane_width_radius = lane_width_radius_dict['lane_width_radius']
        self.lane_width_radius_for_metric = lane_width_radius_dict['lane_width_radius_for_metric']
        self.lane_width_radius_for_uncertain = lane_width_radius_dict['lane_width_radius_for_uncertain']
        self.lane_width_radius_for_metric_for_resized = lane_width_radius_dict['lane_width_radius_for_metric_for_resized']
        self.lane_width_radius_binary = lane_width_radius_dict['lane_width_radius_for_binary']

    def createIndex(self, image_set):
        subfolders = os.listdir(os.path.join(self.root, "labels", image_set))
        subfolders.sort()
        for subfolder in subfolders:
            anno_files = os.listdir(os.path.join(self.root, "labels", image_set, subfolder))
            anno_files.sort()
            for anno_file in anno_files:
                image_file = anno_file.replace(".json", "_color_rect.png")
                anno_path = os.path.join(self.root, "labels", image_set, subfolder, anno_file)
                image_path = os.path.join(self.root, "images", image_set, subfolder, image_file)
                self.images.append(image_path)
                self.annotations.append(anno_path)

    def __len__(self):
        return len(self.images)

    def _valid_target_types(self):
        valid_target_types = ["semantic", "semantic_culane", "semantic_culane_lanetrainer", "semantic_culane_manuel"]
        return valid_target_types

    def _valid_splits(self):
        valid_splits = ["train", "val", "trainval"]
        return valid_splits

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        target_path = os.path.join(self.root, self.annotations[idx])

        target_lanes, target_categories = self._load_target(target_path)
        img = Image.open(img_path).convert('RGB')
        target_list = []
        for target_type in self.target_type:
            target_mask = np.zeros_like(np.array(img))
            if target_type == "semantic":
                target = simplistic_target(target_mask.copy(), lanes=target_lanes, lane_categories=[1] * len(target_lanes),
                                                lane_width_radius=6)
                target = Image.fromarray(target)
                target_list.append(target)

            elif target_type == "semantic_culane":
                target = simplistic_target(target_mask.copy(), lanes=target_lanes, lane_categories=target_categories,
                                                lane_width_radius=10)
                target = Image.fromarray(target)
                target_list.append(target)

            elif target_type == "semantic_culane_manuel":
                img_height, img_width = np.array(img).shape[:2]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    categories, target_lanes_in_culane = obtain_ego_attributes(target_lanes, img_height, img_width, length_threshold=90, lane_fit_length=5, merge_lane_pixel_threshold=100)
                target = simplistic_target(target_mask.copy(), lanes=target_lanes_in_culane, lane_categories=categories,
                                                lane_width_radius=10)
                target = Image.fromarray(target)
                target_list.append(target)
            elif target_type == "semantic_culane_lanetrainer":
                img_height, img_width = np.array(img).shape[:2]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    categories, target_lanes_in_culane = obtain_ego_attributes(target_lanes, img_height, img_width, length_threshold=90, lane_fit_length=5, merge_lane_pixel_threshold=100)
                target_orig = simplistic_target(target_mask.copy(), lanes=target_lanes_in_culane, lane_categories=categories,
                                              lane_width_radius=self.lane_width_radius)
                target_validation = simplistic_target(target_mask.copy(), lanes=target_lanes_in_culane, lane_categories=categories,
                                                    lane_width_radius=self.lane_width_radius_for_metric)

                target_orig = Image.fromarray(target_orig)
                target_validation = Image.fromarray(target_validation)

                target_list = [target_validation, target_orig]

        if self.transforms is not None:
            img, target_list = self.transforms(img, target_list)

        if "lanetrainer" in self.target_type[0]:
            target_main_mask = target_list[1]
            target_validation_mask = target_list[0]
            target = {"mask": target_main_mask, "validation_mask": target_validation_mask}
        else:
            target = target_list
        return img, target

    def _load_target(self, path):
        with open(path) as f:
            data = json.load(f)
        line_infos = data['lanes']
        categories = []
        lanes = []
        for line_info in line_infos:
            lane_id = line_info['lane_id']
            if lane_id == "l1":
                categories.append(1)
            elif lane_id == "l0":
                categories.append(2)
            elif lane_id == "r0":
                categories.append(3)
            elif lane_id == "r1":
                categories.append(4)
            else:
                categories.append(None)

            lane_points = []
            for point_info in line_info['markers']:
                x = point_info["pixel_start"]['x']
                y = point_info["pixel_start"]['y']
                lane_points.append((float(x), float(y)))
            lanes.append(lane_points)

        lanes = [list(set(lane)) for lane in lanes]
        lanes = [lane for lane in lanes if len(lane) >= 2]
        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        lanes = [np.array(lane) for lane in lanes]
        # lane_fitted = []
        # for line in lanes:
        #     line_x = line[:, 0]
        #     line_y = line[:, 1]
        #
        #     length = len(line_x)
        #     curve_prev = np.polyfit(line_y, line_x, deg=3)
        #     max_y= max(line_y)
        #     min_y= min(line_y)
        #
        #     unique_y_range = np.linspace(min_y, max_y, length)
        #     x_values = np.poly1d(curve_prev)(unique_y_range)
        #     new_lane = np.stack([x_values, unique_y_range], axis=1)
        #     lane_fitted.append(new_lane)
        # lanes = lane_fitted
        return lanes, categories

    def _determine_classes(self):
        classes_dict = {}

        for target_type in self.target_type:

            if target_type == "semantic":
                classes = self.classes
            elif target_type == "semantic_culane" or target_type == "semantic_culane_manuel" or \
                    target_type == "semantic_culane_lanetrainer":
                classes = self.culane_classes
            else:
                raise ValueError(f"target type {target_type} is not supported for the time being")

            classes_dict.update({target_type: classes})
        if len(self.target_type) == 1:
            return list(classes_dict.values())[0]
        else:
            return classes_dict