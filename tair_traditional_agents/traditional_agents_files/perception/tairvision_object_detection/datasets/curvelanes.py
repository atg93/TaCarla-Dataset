import os
from abc import ABC

from PIL import Image
from collections import namedtuple
import numpy as np
from .generic_data import GenericSegmentationVisionDataset
import os.path as osp
from tairvision.references.segmentation.lane_utils import lane_with_radius_settings, simplistic_target, obtain_ego_attributes
import json
import warnings


class CurveLanes(GenericSegmentationVisionDataset, ABC):
    CurveLaneClass = namedtuple('CurveLaneClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CurveLaneClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        CurveLaneClass('Line', 1, 1, 'line', 0, False, False, (255, 255, 255)),
    ]

    culane_classes = [
        CurveLaneClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        CurveLaneClass('Line 1', 1, 1, 'line', 0, True, False, (180, 200, 30)),
        CurveLaneClass('Line 2', 2, 2, 'line', 0, True, False, (200, 100, 1)),
        CurveLaneClass('Line 3', 3, 3, 'line', 0, True, False, (1, 100, 200)),
        CurveLaneClass('Line 4', 4, 4, 'line', 0, True, False, (30, 200, 180))
    ]

    image_height = 800
    image_width = 2560
    def __init__(self,
                 lane_width_radius=None,
                 lane_width_radius_for_metric=None,
                 lane_width_radius_for_uncertain=None,
                 lane_width_radius_for_binary=None,
                 **kwargs) -> None:
        super(CurveLanes, self).__init__(**kwargs)

        self.root = os.path.join(self.root, "curvelanes")

        self.valid_modes = self.valid_splits
        self.images = []
        self.targets = []
        self.exist_list = []

        if self.split == "trainval":
            self.createIndex("train")
            self.createIndex("val")
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
        if image_set == 'train':
            txt_name = 'train.txt'
        elif image_set == 'val':
            txt_name = 'valid.txt'

        txt_file = os.path.join(self.root, txt_name)
        with open(txt_file) as f:
            for line in f:
                line = line.strip()
                self.images.append(line)
                line_target = line.replace('.jpg', '.lines.json')
                line_target = line_target.replace('images', 'anno')
                self.targets.append(line_target)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        target_path = os.path.join(self.root, self.targets[idx])

        img = Image.open(img_path).convert('RGB')
        target_lanes = self._load_target(target_path)
        target_list = []

        for target_type in self.target_type:
            target_mask = np.zeros_like(np.array(img))
            if target_type == "semantic":
                target = simplistic_target(target_mask, lanes=target_lanes, lane_categories=[1] * len(target_lanes),
                                                lane_width_radius=10)
                target = Image.fromarray(target)
                target_list.append(target)
            elif target_type == "semantic_culane":
                img_height, img_width = np.array(img).shape[:2]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    categories, target_lanes_in_culane = obtain_ego_attributes(target_lanes, img_height, img_width,
                                                                               length_threshold=0, merge_lane_pixel_threshold=400)
                target = simplistic_target(target_mask, lanes=target_lanes_in_culane, lane_categories=categories,
                                                lane_width_radius=10)
                target = Image.fromarray(target)
                target_list.append(target)
            elif target_type == "semantic_culane_lanetrainer":
                img_height, img_width = np.array(img).shape[:2]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    categories, target_lanes_in_culane = obtain_ego_attributes(target_lanes, img_height, img_width,
                                                                               length_threshold=0, merge_lane_pixel_threshold=400)
                ratio = self._get_lane_width_radius_ratio(img)
                target_orig = simplistic_target(target_mask.copy(), lanes=target_lanes_in_culane, lane_categories=categories,
                                              lane_width_radius=int(self.lane_width_radius/ratio))
                target_validation = simplistic_target(target_mask.copy(), lanes=target_lanes_in_culane, lane_categories=categories,
                                                    lane_width_radius=int(self.lane_width_radius_for_metric/ ratio))

                target_orig = Image.fromarray(target_orig)
                target_validation = Image.fromarray(target_validation)

                target_list = [target_validation, target_orig]

        img, target_list = self._shape_image_and_target(img, target_list)
        if self.transforms is not None:
            img, target_list = self.transforms(img, target_list)

        if "lanetrainer" in self.target_type[0]:
            target_main_mask = target_list[1]
            target_validation_mask = target_list[0]
            target = {"mask": target_main_mask, "validation_mask": target_validation_mask}
        else:
            target = target_list

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _shape_image_and_target(img, target_list):
        img_array = np.array(img)
        new_target_list = []
        if img_array.shape == (1440, 2560, 3):
            start_index = 640
            end_index = 0
        elif img_array.shape == (660, 1570, 3):
            start_index = 170
            end_index = 2
        elif img_array.shape == (720, 1280, 3):
            start_index = 320
            end_index = 0

        img_array_new = img_array[start_index:, end_index:, ...]
        for target in target_list:
            target_array = np.array(target)
            target_array = target_array[start_index:, end_index:, ...]
            target = Image.fromarray(target_array)
            new_target_list.append(target)

        img = Image.fromarray(img_array_new)
        return img, new_target_list

    @staticmethod
    def _get_lane_width_radius_ratio(img):
        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]
        if img_height == 1440:
            lane_width_radius_ratio = 800 / 800
        elif img_height == 720:
            lane_width_radius_ratio = 800 / 400
        elif img_height == 660:
            lane_width_radius_ratio = 800 / 490

        return lane_width_radius_ratio


    def _valid_target_types(self):
        valid_target_types = ["semantic", "semantic_culane", "semantic_culane_lanetrainer"]
        return valid_target_types

    def _valid_splits(self):
        valid_splits = ["train", "val", "trainval"]
        return valid_splits

    def _load_target(self, path):
        with open(path) as f:
            data = json.load(f)
        line_infos = data['Lines']
        lanes = []
        for line_info in line_infos:
            lane_points = []
            for point_info in line_info:
                lane_points.append((float(point_info["x"]), float(point_info["y"])))
            lanes.append(lane_points)

        lanes = [list(set(lane)) for lane in lanes]
        lanes = [lane for lane in lanes if len(lane) >= 2]
        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        lanes = [np.array(lane) for lane in lanes]
        return lanes

    def _determine_classes(self):
        classes_dict = {}

        for target_type in self.target_type:

            if target_type == "semantic":
                classes = self.classes
            elif target_type == "semantic_culane" or target_type == "semantic_culane_lanetrainer":
                classes = self.culane_classes
            else:
                raise ValueError(f"target type {target_type} is not supported for the time being")

            classes_dict.update({target_type: classes})
        if len(self.target_type) == 1:
            return list(classes_dict.values())[0]
        else:
            return classes_dict

