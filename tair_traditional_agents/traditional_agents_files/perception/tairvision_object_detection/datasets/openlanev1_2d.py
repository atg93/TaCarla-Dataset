import copy
import os
import os.path as osp
import warnings

from PIL import Image
from collections import namedtuple
from .generic_data import GenericSegmentationVisionDataset
import json
import numpy as np
from matplotlib import colors
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import glob
import cv2
import heapq
import copy
from tairvision.references.segmentation.lane_utils import simplistic_target, lane_with_radius_settings
from tairvision.references.segmentation.BEV_lane_utils import color_name_2_rgb, prune_3d_lane_by_visibility, \
    homograpthy_g2im_extrinsic, projection_g2im_extrinsic, convert_lanes_3d_to_gflat, \
    prune_3d_lane_by_range, projective_transformation, convert_to_culane_format, bottom_point_extraction_from_3d_world


class Openlane(GenericSegmentationVisionDataset):
    OpenlaneClass = namedtuple('OpenlaneClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        OpenlaneClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        OpenlaneClass('white dash', 1, 1, 'line', 0, True, False, color_name_2_rgb('mediumpurple')),
        OpenlaneClass('white solid', 2, 2, 'line', 0, True, False, color_name_2_rgb('mediumturquoise')),
        OpenlaneClass('double-white dash', 3, 3, 'line', 0, True, False, color_name_2_rgb('mediumorchid')),
        OpenlaneClass('double-white solid', 4, 4, 'line', 0, True, False, color_name_2_rgb('lightskyblue')),
        OpenlaneClass('white-ldash-rsolid', 5, 5, 'line', 0, True, False, color_name_2_rgb('hotpink')),
        OpenlaneClass('white-lsolid-rdash', 6, 6, 'line', 0, True, False, color_name_2_rgb('cornflowerblue')),
        OpenlaneClass('yellow-dash', 7, 7, 'line', 0, True, False, color_name_2_rgb('yellowgreen')),
        OpenlaneClass('yellow-solid', 8, 8, 'line', 0, True, False, color_name_2_rgb('dodgerblue')),
        OpenlaneClass('double-yellow-dash', 9, 9, 'line', 0, True, False, color_name_2_rgb('salmon')),
        OpenlaneClass('double-yellow-solid', 10, 10, 'line', 0, True, False, color_name_2_rgb('lightcoral')),
        OpenlaneClass('yellow-ldash-rsolid', 11, 11, 'line', 0, True, False, color_name_2_rgb('coral')),
        OpenlaneClass('yellow-lsolid-rdash', 12, 12, 'line', 0, True, False, color_name_2_rgb('lightseagreen')),
        # OpenlaneClass('fishbone', 13, 13, 'line', 0, True, False, color_name_2_rgb('royalblue')),
        # OpenlaneClass('others', 14, 14, 'line', 0, True, False, color_name_2_rgb('forestgreen')),
        OpenlaneClass('curb-side-left', 20, 13, 'line', 0, True, False, color_name_2_rgb('gold')),
        OpenlaneClass('curb-side-right', 21, 14, 'line', 0, True, False, color_name_2_rgb('gold'))
    ]

    culane_classes = [
        OpenlaneClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        OpenlaneClass('Line 1', 1, 1, 'line', 0, True, False, (180, 200, 30)),
        OpenlaneClass('Line 2', 2, 2, 'line', 0, True, False, (200, 100, 1)),
        OpenlaneClass('Line 3', 3, 3, 'line', 0, True, False, (1, 100, 200)),
        OpenlaneClass('Line 4', 4, 4, 'line', 0, True, False, (30, 200, 180))
    ]

    simplified_lane_classes = [
        OpenlaneClass('Background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        OpenlaneClass('line', 1, 1, 'line', 0, True, False, (255, 255, 255)),
    ]

    image_height = 1280
    image_width = 1920

    def __init__(self,
                 number_of_points_for_fit=None,
                 lane_fit_position_scaling_factor=None,
                 lane_width_radius=None,
                 lane_width_radius_for_metric=None,
                 uncertain_region_enabled=False,
                 lane_width_radius_for_uncertain=None,
                 lane_width_radius_for_binary=None,
                 **kwargs) -> None:
        super(Openlane, self).__init__(**kwargs)

        self.lane_fit_position_scaling_factor = lane_fit_position_scaling_factor
        self.number_of_points_for_fit = number_of_points_for_fit
        self.uncertain_region_enabled = uncertain_region_enabled

        lane_width_radius_dict = lane_with_radius_settings(lane_width_radius=lane_width_radius,
                                                           lane_width_radius_for_metric=lane_width_radius_for_metric,
                                                           lane_width_radius_for_uncertain=lane_width_radius_for_uncertain,
                                                           lane_width_radius_for_binary=lane_width_radius_for_binary,
                                                           transforms=self.transforms,
                                                           image_height=self.image_height, image_width=self.image_width)

        self.lane_width_radius = lane_width_radius_dict['lane_width_radius']
        self.lane_width_radius_for_metric = lane_width_radius_dict['lane_width_radius_for_metric']
        self.lane_width_radius_for_uncertain = lane_width_radius_dict['lane_width_radius_for_uncertain']
        self.lane_width_radius_for_metric_for_resized = lane_width_radius_dict[
            'lane_width_radius_for_metric_for_resized']
        self.lane_width_radius_binary = lane_width_radius_dict['lane_width_radius_for_binary']

        openlane_version = "lane3d_1000_v12"
        self.root = osp.join(self.root, 'openlane')
        if self.split == 'train':
            self.json_path = osp.join(self.root, openlane_version, 'training')
        elif self.split == 'val':
            self.json_path = osp.join(self.root, openlane_version, "validation")
        elif self.split == 'test':
            self.json_path = osp.join(self.root, openlane_version, 'test')
        elif self.split == 'trainval':
            self.json_path = [osp.join(self.root, openlane_version, 'training'),
                              osp.join(self.root, openlane_version, "validation")]
        elif self.split == 'test_extreme_weather_case':
            self.json_path = osp.join(self.root, openlane_version, 'test', "extreme_weather_case")

        else:
            raise Exception('split has not found')

        self.valid_modes = self.valid_splits

        if isinstance(self.json_path, list):
            self.label_list = []
            for json_path in self.json_path:
                self.label_list.extend(sorted(glob.glob(osp.join(json_path, '**/*.json'), recursive=True)))
        else:
            self.label_list = sorted(glob.glob(osp.join(self.json_path, '**/*.json'), recursive=True))
        # self.label_list = self.label_list[:1000]
        # self.color_palette = self.get_color_palette()[:self.get_number_of_classes()]
        # self.color_palette = self.get_color_palette()[:self.get_number_of_classes()]

    def __len__(self):
        return len(self.label_list)

    def preprocess_data(self, info_dict):

        cam_extrinsics = np.array(info_dict['extrinsic'])
        # Re-calculate extrinsic matrix based on ground coordinate
        R_vg = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]], dtype=float)
        cam_extrinsics[:3, :3] = np.matmul(np.matmul(
            np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
            R_vg), R_gc)
        cam_extrinsics[0:2, 3] = 0.0

        gt_cam_height = cam_extrinsics[2, 3]
        if 'cam_pitch' in info_dict:
            gt_cam_pitch = info_dict['cam_pitch']
        else:
            gt_cam_pitch = 0

        if 'intrinsic' in info_dict:
            cam_intrinsics = info_dict['intrinsic']
            cam_intrinsics = np.array(cam_intrinsics)

        _label_cam_height = gt_cam_height
        _label_cam_pitch = gt_cam_pitch

        attributes = []
        gt_lanes_packed = info_dict['lane_lines']
        gt_lane_pts, gt_lane_visibility, gt_laneline_category = [], [], []
        for i, gt_lane_packed in enumerate(gt_lanes_packed):
            # A GT lane can be either 2D or 3D
            # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
            lane = np.array(gt_lane_packed['xyz'])
            lane_visibility = np.array(gt_lane_packed['visibility'])
            attributes.append(gt_lane_packed["attribute"])
            # Coordinate convertion for openlane_300 data
            lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
            cam_representation = np.linalg.inv(
                np.array([[0, 0, 1, 0],
                          [-1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, 0, 1]], dtype=float))  # transformation from apollo camera to openlane camera
            lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))

            lane = lane[0:3, :].T
            sort_structure = lane[:, 1].argsort()
            lane = lane[sort_structure]
            lane_visibility = lane_visibility[sort_structure]

            gt_lane_pts.append(lane)
            gt_lane_visibility.append(lane_visibility)

            if 'category' in gt_lane_packed:
                lane_cate = gt_lane_packed['category']
                # if lane_cate == 21:  # merge left and right road edge into road edge
                #     lane_cate = 20
                if lane_cate in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21]:
                    gt_laneline_category.append(lane_cate)
                else:
                    gt_laneline_category.append(255)
            else:
                gt_laneline_category.append(255)

        P_g2im = projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)

        H_g2im = homograpthy_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
        H_im2g = np.linalg.inv(H_g2im)
        P_g2gflat = np.matmul(H_im2g, P_g2im)

        img_lanes = []
        bottom_points = []

        min_points = []
        max_points = []
        polynomials_third = []

        gt_lane_pts = [prune_3d_lane_by_visibility(gt_lane, gt_vis) for gt_lane, gt_vis in
                       zip(gt_lane_pts, gt_lane_visibility)]

        gt_lane_pts = [prune_3d_lane_by_range(gt_lane, -30, 30) for gt_lane in gt_lane_pts]
        gt_lane_pts = [lane for lane in gt_lane_pts if lane.shape[0] > 1]

        gt_lane_pts_flat = copy.deepcopy(gt_lane_pts)
        convert_lanes_3d_to_gflat(gt_lane_pts_flat, P_g2gflat)

        bottom_point_extraction_from_3d_world(gt_lane_pts=gt_lane_pts, gt_lane_pts_flat=gt_lane_pts_flat,
                                              attributes=attributes, P_g2im=P_g2im, polynomials_third=polynomials_third,
                                              min_points=min_points, max_points=max_points,
                                              bottom_points=bottom_points, img_lanes=img_lanes)

        return img_lanes, gt_laneline_category, bottom_points, gt_lane_pts, attributes

    def __getitem__(self, idx):
        json_filename = self.label_list[idx]
        with open(json_filename, 'r') as fp:
            info_dict = json.load(fp)
        img_lanes, lane_cats, bottom_points, gt_lane_flat, attributes = self.preprocess_data(info_dict)

        img_path = osp.join(self.root, 'images', info_dict['file_path'])
        image = Image.open(img_path)
        image = np.array(image)
        targets = []

        for t in self.target_type:
            target_mask = np.zeros((image.shape[0:3]))
            target_mask_validation = np.zeros((image.shape[0:3]))

            if t == 'semantic_culane' or t == "semantic_culane_lane_width":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    target_lanes, target_cats = convert_to_culane_format(lanes=img_lanes, lane_categories=lane_cats,
                                                                         bottom_points=bottom_points,
                                                                         gt_flat_lines=gt_lane_flat,
                                                                         middle_point=0,
                                                                         number_of_selected_lanes_per_side=2,
                                                                         attributes=attributes)
            elif t == 'semantic_lane_type':
                target_cats = lane_cats.copy()
                target_lanes = img_lanes.copy()

            elif t == 'semantic_culane_original':
                target_cats = attributes.copy()
                target_lanes = img_lanes.copy()

            elif t == 'semantic_binary':
                target_cats = [1] * len(lane_cats)
                target_lanes = img_lanes.copy()

            else:
                raise ValueError(f"{t} is not a valid target type")

            if t == "semantic_culane_lane_width":
                target_mask = simplistic_target(target_mask, lanes=img_lanes, lane_categories=[255] * len(img_lanes),
                                                lane_width_radius=self.lane_width_radius, return_reduced_mask=False)
                if self.lane_width_radius != self.lane_width_radius_for_metric:
                    target_mask_validation = simplistic_target(target_mask_validation, lanes=img_lanes,
                                                               lane_categories=[255] * len(img_lanes),
                                                               lane_width_radius=self.lane_width_radius_for_metric,
                                                               return_reduced_mask=False)
                else:
                    target_mask_validation = target_mask

            target_mask = simplistic_target(target_mask, lanes=target_lanes, lane_categories=target_cats,
                                            lane_width_radius=self.lane_width_radius)

            if self.lane_width_radius != self.lane_width_radius_for_metric:
                target_mask_validation = simplistic_target(target_mask_validation, lanes=target_lanes,
                                                           lane_categories=target_cats,
                                                           lane_width_radius=self.lane_width_radius_for_metric)
            else:
                target_mask_validation = target_mask

            if t == 'semantic_lane_type':
                target_mask = self._convert_mask_labels(target_mask, t)

            if self.uncertain_region_enabled:
                if self.lane_width_radius_for_metric != self.lane_width_radius_for_uncertain:
                    target_mask_uncertain = np.zeros((image.shape[0:3]))
                    target_mask_uncertain = simplistic_target(target_mask_uncertain, lanes=target_lanes,
                                                              lane_categories=target_cats,
                                                              lane_width_radius=self.lane_width_radius_for_metric)
                else:
                    target_mask_uncertain = target_mask_validation
                target_mask[target_mask != target_mask_uncertain] = 255

            if t == "semantic_culane_lane_width":
                target_mask_validation = Image.fromarray(target_mask_validation)
                targets.append(target_mask_validation)

            target_mask = Image.fromarray(target_mask)
            targets.append(target_mask)

        # target_mask = self.target_from_raw_gt(target_mask, lane_data=img_lanes,
        #                                       lane_categories=lane_cats, lane_width_radius=self.lane_width_radius)

        image = Image.fromarray(image)

        if self.transforms is not None:
            image, targets = self.transforms(image, targets)

        if self.target_type[0] == "semantic_culane_lane_width":
            target_main_mask = targets[1]
            target_validation_mask = targets[0]
            targets = {"mask": target_main_mask, "validation_mask": target_validation_mask}

        return image, targets

    def _determine_classes(self):
        classes_dict = {}

        for target_type in self.target_type:

            if target_type == "semantic_lane_type":
                classes = self.classes
            elif target_type == "semantic_binary":
                classes = self.simplified_lane_classes
            elif target_type == "semantic_culane" or target_type == "semantic_culane_lane_width" \
                    or target_type == 'semantic_culane_original':
                classes = self.culane_classes
            else:
                raise ValueError(f"target type {target_type} is not supported for the time being")

            classes_dict.update({target_type: classes})
        if len(self.target_type) == 1:
            return list(classes_dict.values())[0]
        else:
            return classes_dict

    def _valid_target_types(self):
        valid_target_types = ["semantic_binary", "semantic_lane_type", "semantic_culane", "semantic_culane_lane_width",
                              "semantic_culane_original"]
        return valid_target_types

    def _valid_splits(self):
        valid_splits = ["train", "val", "test", "trainval", "test_extreme_weather_case"]
        return valid_splits
