import os
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
import numpy as np
import torch
import cv2
from .generic_data import GenericSegmentationVisionDataset
import warnings
from tairvision.references.segmentation.lane_utils import draw_line_on_image, lane_with_radius_settings


class Culane(GenericSegmentationVisionDataset):
    CulaneClass = namedtuple('CulaneClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CulaneClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        CulaneClass('Line 1', 1, 1, 'line', 0, True, False, (180, 200, 30)),
        CulaneClass('Line 2', 2, 2, 'line', 0, True, False, (200, 100, 1)),
        CulaneClass('Line 3', 3, 3, 'line', 0, True, False, (1, 100, 200)),
        CulaneClass('Line 4', 4, 4, 'line', 0, True, False, (30, 200, 180))
    ]

    erosion_kernel = np.ones((7, 7), np.uint8)
    image_height = 590
    image_width = 1640

    def __init__(self, lane_order=None,
                 number_of_points_for_fit=None,
                 lane_fit_position_scaling_factor=None,
                 lane_width_radius=None,
                 lane_width_radius_for_metric=None,
                 lane_width_radius_for_uncertain=None,
                 lane_width_radius_for_binary=None,
                 return_image_name=False,
                 **kwargs) -> None:
        super(Culane, self).__init__(**kwargs)

        self.root = os.path.join(self.root, "culane")
        self.lane_order = lane_order
        self.lane_fit_position_scaling_factor = lane_fit_position_scaling_factor
        self.return_image_name = return_image_name

        self.number_of_points_for_fit = number_of_points_for_fit
        self.data_dir_path = self.root
        self.image_set = self.split
        self.valid_modes = self.valid_splits
        self.images = []
        self.targets = []
        self.exist_list = []

        if self.image_set == "trainval":
            self.createIndex("train")
            self.createIndex("val")
        elif self.image_set == "trainvaltest":
            self.createIndex("train")
            self.createIndex("val")
            self.createIndex("test")
        elif self.image_set == "valtest":
            self.createIndex("val")
            self.createIndex("test")
        else:
            self.createIndex(self.image_set)

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
        if image_set == "test":
            txt_format = "{}.txt"
        else:
            txt_format = "{}_gt.txt"

        listfile = os.path.join(self.data_dir_path, "list", txt_format.format(image_set))
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.images.append(os.path.join(self.data_dir_path, l[0][1:]))  #
                if image_set != "test":
                    self.targets.append(os.path.join(self.data_dir_path, l[1][1:]))
                    self.exist_list.append([int(x) for x in l[2:]])
                else:
                    self.targets.append(None)
                    self.exist_list.append(None)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image = Image.open(image_name).convert('RGB')

        lane_exist = self.exist_list[idx]
        if self.targets[idx] is not None:
            target = Image.open(self.targets[idx])
            target_array = np.array(target)
        else:
            target_array = np.zeros(np.array(image).shape[:2])

        if self.target_type[0] == "semantic_lane_from_txt_with_mask_uncertainty" or self.target_type[0] == "semantic_lanefit" \
                or self.target_type[0] == "semantic_lane_from_txt":
            target_array_validation = target_array
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                target_array_validation =  self.target_from_raw_gt(target_array_validation,
                                                                   image_name, lane_exist,
                                                                   self.lane_width_radius_for_metric)

        if self.target_type[0] == "semantic_lane_from_txt_with_mask_uncertainty" or self.target_type[0] == "semantic_lane_from_txt":
            target_array_original = target_array
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                target_array =  self.target_from_raw_gt(target_array, image_name, lane_exist, self.lane_width_radius)
            if self.target_type[0] == "semantic_lane_from_txt_with_mask_uncertainty" and self.image_set == "train":
                target_array[target_array_original != target_array] = 255

        if self.target_type[0] == self.valid_target_types[1] and self.mask_convert is False:
            for label in [1, 2, 3, 4]:
                self._apply_morphology_operation(target=target_array, kernel=self.erosion_kernel,
                                                 label=label, operation="erosion", null_label=0)
        if self.mask_convert:
            target_array = self._convert_mask_labels(target_array)

        target = Image.fromarray(target_array)

        if self.target_type[0] == "semantic_lane_from_txt_with_mask_uncertainty" or self.target_type[0] == "semantic_lanefit" \
                or self.target_type[0] == "semantic_lane_from_txt":
            target_array_validation = Image.fromarray(target_array_validation)
            target_list = [target, target_array_validation]
        else:
            target_list = target

        if self.transforms is not None:
            image, target_list = self.transforms(image, target_list)

        if self.target_type[0] == "semantic_lanefit":
            target = self.create_lane_target_type(target_list[0])
            target.update({"validation_mask": target_list[1]})
            # mask_lane = create_mask_from_lane_params(target, lane_width_radius,
            #                              self.lane_fit_position_scaling_factor)

            # target["mask_lane"] = mask_lane

        elif self.target_type[0] == "semantic_lane_from_txt_with_mask_uncertainty" or self.target_type[0] == "semantic_lane_from_txt":
            target = {"mask": target_list[0]}
            # if self.image_set is self.valid_modes[1]:
            target.update({"validation_mask": target_list[1]})
            if self.return_image_name:
                target.update({"image_name": image_name})

        else:
            target = target_list

        return image, target


    def __len__(self):
        return len(self.images)

    def _valid_target_types(self):
        valid_target_types = ["semantic", "semantic_7x7_eroded", "semantic_lanefit", "semantic_lane_from_txt_with_mask_uncertainty", "semantic_lane_from_txt"]
        return valid_target_types

    def _valid_splits(self):
        valid_splits = ["train", "val", "test", "trainval", "trainvaltest"]
        return valid_splits

    def _determine_classes(self):
        if self.mask_convert:
            self._modify_training_classes()
        return self.classes

    def _modify_training_classes(self):
        for cls_index, cls in enumerate(self.classes):
            if cls.id != 0 and cls.id != 1:
                cls = cls._replace(train_id=1)
                cls = cls._replace(ignore_in_eval=True)
            self.classes[cls_index] = cls

    #TODO Research the collate function for speedup. This is not utilized for the time being
    @staticmethod
    def collate_test(batch):
        image_list = []
        image_name_list = []
        for image, image_name in batch:
            image_list.append(image)
            image_name_list.append(image_name)
        image_batch = torch.stack(image_list)
        samples = {'img': image_batch,
                   'img_name': image_name_list}

        return samples

    def fit_lane(self, unique_y, mask, label, order=3):
        min_y = np.min(unique_y)
        max_y = np.max(unique_y)

        height = width = self.lane_fit_position_scaling_factor

        x_list_normalized = []
        unique_y_normalized = []
        for y_index in unique_y:
            line_where = np.where(mask[y_index] == label)[0]
            corresponding_x = np.mean(line_where)
            x_list_normalized.append(corresponding_x / width)
            unique_y_normalized.append(y_index / height)

        params = np.polyfit(unique_y_normalized, x_list_normalized, order)
        poly_eqn = np.poly1d(params)

        unique_y_range = np.arange(min_y, max_y) / height
        predicted = poly_eqn(unique_y_range)
        return unique_y_range, predicted, params

    def create_lane_target_type(self, target):
        target_array = np.array(target)

        lane_order = self.lane_order
        lane_param_np = np.empty([0, lane_order + 1])
        border_np = np.empty([0, 2])
        lane_exist = np.zeros([4])
        unique_y_linspace = np.zeros([0, self.number_of_points_for_fit])
        for label in [1, 2, 3, 4]:
            where_tuple = np.where(target_array == label)
            unique_y = np.unique(where_tuple[0])
            if len(unique_y) > 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    unique_y_range, predicted, params = \
                        self.fit_lane(unique_y, target_array, label, order=lane_order)
                    border_np = np.vstack([
                        border_np,
                        np.array([np.min(unique_y_range), np.max(unique_y_range)])
                    ])

                    unique_y_linspace = np.vstack([
                        unique_y_linspace,
                        np.linspace(np.min(unique_y_range), np.max(unique_y_range),
                                    self.number_of_points_for_fit)
                    ])
                lane_param_np = np.vstack([lane_param_np, params])
                lane_exist[label - 1] = 1
            else:
                params = np.zeros([lane_order + 1])
                lane_param_np = np.vstack([lane_param_np, params])
                border_np = np.vstack([border_np, np.array([0, 0])])
                unique_y_linspace = np.vstack([
                    unique_y_linspace,
                    np.zeros([self.number_of_points_for_fit])
                ])

        target_mask = torch.from_numpy(target_array)
        lane_params = torch.from_numpy(lane_param_np)
        lane_exist = torch.from_numpy(lane_exist)
        borders = torch.from_numpy(border_np)
        unique_y_torch = torch.from_numpy(unique_y_linspace)
        target = {"mask": target_mask, "lane_params": lane_params, "lane_exist": lane_exist,
                  "borders": borders, "unique_y": unique_y_torch}

        return target

    def target_from_raw_gt(self, target_mask, image_name, exist, lane_width_radius):

        anno_path = image_name[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt
        with open(anno_path, 'r') as anno_file:
            lane_data = [list(map(float, line.split())) for line in anno_file.readlines()]

        if exist is None:
            if len(lane_data) == 2:
                exist = [0, 1, 1, 0]
            elif len(lane_data) == 4:
                exist = [1, 1, 1, 1]
            elif len(lane_data) == 3:
                if lane_data[1][0] > 820:
                    exist = [0, 1, 1, 1]
                else:
                    exist = [1, 1, 1, 0]
            elif len(lane_data) == 1:
                if lane_data[1][0] > 820:
                    exist = [0, 0, 1, 0]
                else:
                    exist = [0, 1, 0, 0]
            else:
                return target_mask

        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
                 for lane in lane_data]

        lanes = [list(set(lane)) for lane in lanes]

        lane_boolean = [True if len(lane) >= 2 else False for lane in lanes] # remove duplicated points
        lanes = [lane for lane in lanes if len(lane) >= 2]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y

        lane_exist = np.array(exist)

        mask_lane = np.zeros_like(target_mask)

        dummy_mask = np.zeros_like(mask_lane)
        dummy_image = dummy_mask[:, :, None]
        dummy_image = \
            np.concatenate([dummy_image, dummy_image, dummy_image], 2).astype(np.uint8)

        selected_lane_index = 0
        for label, existence in enumerate(lane_exist):

            if existence != 1:
                continue
            if lane_boolean[selected_lane_index] is False:
                continue

            lane_points_raw = lanes[selected_lane_index]

            lane_points = np.array(lane_points_raw)
            # x_coords = lane_points[:, 0]
            # y_coords = lane_points[:, 1]

            # fill_mask_with_circular_points(mask_lane=mask_lane, x_coords=x_coords, y_coords=y_coords,
            #                                lane_width_radius=lane_width_radius, class_label=label + 1)

            lane_array = lane_points[::1, :].round().astype('int').tolist()
            dummy_image = draw_line_on_image(image=dummy_image, points=lane_array, color=(label + 1, 0, 0),
                               thickness=lane_width_radius)

            selected_lane_index = selected_lane_index + 1

        mask_lane = dummy_image[:, :, 0]
        target = mask_lane

        return target


