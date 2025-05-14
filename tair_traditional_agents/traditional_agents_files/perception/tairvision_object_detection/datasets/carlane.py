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
from tairvision.references.segmentation.lane_utils import draw_line_on_image, simplistic_target, lane_with_radius_settings
import warnings

class Carlane(GenericSegmentationVisionDataset):
    CarlaneClass = namedtuple('CarlaneClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CarlaneClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        CarlaneClass('Line 1', 1, 1, 'line', 0, True, False, (180, 200, 30)),
        CarlaneClass('Line 2', 2, 2, 'line', 0, True, False, (200, 100, 1)),
        CarlaneClass('Line 3', 3, 3, 'line', 0, True, False, (1, 100, 200)),
        CarlaneClass('Line 4', 4, 4, 'line', 0, True, False, (30, 200, 180))
    ]

    erosion_kernel = np.ones((7, 7), np.uint8)
    image_height = 732
    image_width = 968

    def __init__(self,
                 lane_width_radius=None,
                 lane_width_radius_for_metric=None,
                 lane_width_radius_for_uncertain=None,
                 lane_width_radius_for_binary=None,
                 include_junctions=False,
                 **kwargs) -> None:
        super(Carlane, self).__init__(**kwargs)

        self.root = os.path.join(self.root, "carlane")
        self.image_set = self.split
        self.include_junctions = include_junctions
        self.valid_modes = self.valid_splits
        self.images = []
        self.targets = []
        self.exist_list = []
        self.annotations = []
        self.camera_params = None
        self.image_shape = None

        self.image_paths = []
        self.exclude_weathers = []
        # self.exclude_weathers = ['clear', 'cloudy', 'hard rain', 'mid rain', 'mid rainy', 'soft rain', 'wet', 'wet cloudy']

        if self.split == "trainval":
            self.createIndex("train")
            self.createIndex("val")
        else:
            self.createIndex(self.split)
        self.ipm = IPM(img_shape=self.image_shape, **self.camera_params)

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

        # Image shape 968 x 732

    def createIndex(self, image_set):
        # ann_path = os.path.join(self.root, "annotations", "{}".format(self.image_set), 'Town10HD.json')
        ann_path = os.path.join(self.root, "annotations", "{}.json".format(image_set))
        data_dir_path = os.path.join(self.root, image_set)
        with open(ann_path) as fp:
            annotations = json.load(fp)
            img_shape = annotations['config']['image_size']
            self.image_shape = [img_shape['height'], img_shape['width']]
            for frame in annotations['frames']:
                path = osp.join(data_dir_path, frame['town'], frame['name'])
                file_exist = osp.exists(path)
                has_exclude_w = all(s not in frame['attributes']['weather'] for s in self.exclude_weathers)

                if self.include_junctions:
                    contains_junction = False
                else:
                    contains_junction = self.__exist_junction(frame['lane_points'])

                if file_exist and has_exclude_w and not contains_junction:
                    self.annotations.append(frame)
                    self.image_paths.append(path)
                    self.camera_params = frame['camera_params']

    @staticmethod
    def __exist_junction(lane_points):
        check_list = [False]
        for l in lane_points:
            if l is not None:
                check_list.append(any([j for _, j in l]))

        return any(check_list)

    def get_number_of_classes(self) -> Union[int, List[int]]:
        if self.target_type[0] == "semantic_bev" or self.target_type[0] == "semantic_front":
            return 4
        elif self.target_type[0] == "semantic_front_lanetrainer":
            return 5

    def __getitem__(self, idx):
        frame = self.annotations[idx]

        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        image = np.array(image)
        target_mask = np.zeros_like(image)

        lane_points = frame['lane_points']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lane_points = [np.array(l) if l is not None else l for l in lane_points]

        if self.target_type[0] == "semantic_bev":
            image = self.ipm.front_2_bev(image)
            # front_image = self.ipm.bev_2_front(bev_image)
            bev_lane_point = self.bev_lane_points(lane_points, image.shape, self.ipm.IPM)
            target_mask = np.zeros_like(image)
            target_mask = self.draw_line_array_on_image(target_mask, bev_lane_point, categories=[1, 2, 3, 4], draw_junction=None)
            target_mask = target_mask[:, :, 0]
            target = Image.fromarray(target_mask)
        elif self.target_type[0] == "semantic_front":
            target_mask = self.draw_line_array_on_image(target_mask, lane_points,
                                                   categories=[1, 2, 3, 4], sort_y=True, draw_junction=False)
            target_mask = target_mask[:, :, 0]
            target = Image.fromarray(target_mask)
        elif self.target_type[0] == "semantic_front_lanetrainer":
            target_mask_org = self.draw_line_array_on_image(target_mask.copy(), lane_points, thickness=self.lane_width_radius,
                                                            categories=[1, 2, 3, 4], sort_y=True, draw_junction=False)
            target_mask_validation = self.draw_line_array_on_image(target_mask.copy(), lane_points, thickness=self.lane_width_radius_for_metric,
                                                        categories=[1, 2, 3, 4], sort_y=True, draw_junction=False)

            target_mask_org = target_mask_org[:, :, 0]
            target_mask_org_pil = Image.fromarray(target_mask_org)

            target_mask_validation = target_mask_validation[:, :, 0]
            target_mask_validation_pil = Image.fromarray(target_mask_validation)

            target = [target_mask_validation_pil, target_mask_org_pil]

        image = Image.fromarray(image)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.target_type[0] == "semantic_front_lanetrainer":
            target_main_mask = target[1]
            target_validation_mask = target[0]
            target = {"mask": target_main_mask, "validation_mask": target_validation_mask}

        return image, target


    def bev_lane_points(self, lane_points, img_shape, IPM):
        lane_points = self._mask_lane_point(lane_points, img_shape)

        ipm_lane_points = [None for _ in lane_points]
        for i, points in enumerate(lane_points):
            if points is not None:
                points_np = np.array(points)
                proj_points = cv2.perspectiveTransform(points_np[:, None, :].astype('float'), IPM)
                ipm_lane_points[i] = proj_points[:, 0, :].round().astype('int')
        return ipm_lane_points

    @staticmethod
    def _mask_lane_point(lane_points, img_shape):
        for i, points in enumerate(lane_points):
            if points is not None:
                lane_points[i] = np.stack([point[0] for point in points])
                mask_x = np.logical_and(lane_points[i][:, 0] > 0, lane_points[i][:, 0] < img_shape[1])
                mask_y = np.logical_and(lane_points[i][:, 1] > 0, lane_points[i][:, 1] < img_shape[0])
                mask = np.logical_and(mask_x, mask_y)
                lane_points[i] = lane_points[i][mask]
                if lane_points[i].shape[0] != 0:
                    lane_points[i] = lane_points[i].tolist()
                else:
                    lane_points[i] = None
        return lane_points

    @staticmethod
    def draw_line_array_on_image(img, points_list, categories=None, thickness=10, sort_y=True,
                                 draw_junction=False):
        if categories is None:
            categories = []
        image = img.copy()
        if len(points_list) == len(categories):
            for point, cat in zip(points_list, categories):
                image = draw_line_on_image(image, point, (cat, 0, 0), thickness=thickness, imgcopy=False,
                                                sort_y=sort_y, draw_junction=draw_junction)
            return image
        else:
            print("Point list and color size not equal")
            return image


    def __len__(self):
        return len(self.annotations)

    def _valid_target_types(self):
        valid_target_types = ["semantic_bev", "semantic_front", "semantic_front_lanetrainer"]
        return valid_target_types

    def _valid_splits(self):
        valid_splits = ["train", "val", "test", "trainval"]
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
