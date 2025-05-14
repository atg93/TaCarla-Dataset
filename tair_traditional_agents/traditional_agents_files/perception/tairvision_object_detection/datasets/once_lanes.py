import os
from abc import ABC

from PIL import Image
from collections import namedtuple
import numpy as np
from .generic_data import GenericSegmentationVisionDataset
from tairvision.references.segmentation.lane_utils import lane_with_radius_settings, simplistic_target, \
    obtain_ego_attributes
import json
import warnings
from tairvision.references.segmentation.BEV_lane_utils import prune_3d_lane_by_visibility, convert_lanes_3d_to_gflat, \
    prune_3d_lane_by_range, bottom_point_extraction_from_3d_world, projection_g2im, homograpthy_g2im
import copy


class Once3DLanes(GenericSegmentationVisionDataset, ABC):
    OnceLanesClass = namedtuple('OnceLanesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                   'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        OnceLanesClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        OnceLanesClass('Line', 1, 1, 'line', 0, False, False, (255, 255, 255)),
    ]

    culane_classes = [
        OnceLanesClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        OnceLanesClass('Line 1', 1, 1, 'line', 0, True, False, (180, 200, 30)),
        OnceLanesClass('Line 2', 2, 2, 'line', 0, True, False, (200, 100, 1)),
        OnceLanesClass('Line 3', 3, 3, 'line', 0, True, False, (1, 100, 200)),
        OnceLanesClass('Line 4', 4, 4, 'line', 0, True, False, (30, 200, 180))
    ]

    image_height = 1020
    image_width = 1920
    def __init__(self,
                 lane_width_radius=None,
                 lane_width_radius_for_metric=None,
                 lane_width_radius_for_uncertain=None,
                 lane_width_radius_for_binary=None,
                 **kwargs) -> None:
        super(Once3DLanes, self).__init__(**kwargs)

        self.root = os.path.join(self.root, "once3dlanes")
        self.anno_folder = os.path.join(self.root, "labels")

        self.valid_modes = self.valid_splits
        self.images = []
        self.targets = []
        self.exist_list = []

        self.error_json_ids = ["1618776914500", "1619801784499", "1619977228700", "1620681012299"]

        if self.split == "trainval":
            self.createIndex("train")
            self.createIndex("val")
        elif self.split == "trainvaltest":
            self.createIndex("train")
            self.createIndex("val")
            self.createIndex("test")
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
        label_list_txt_file = os.path.join(self.anno_folder, "list", f"{image_set}.txt")
        labels_folder = os.path.join(self.anno_folder, image_set)
        image_folder = os.path.join(self.root, "images")
        with open(label_list_txt_file) as f:
            for line in f:
                line = line.strip()
                line = line[1:]
                id = line.split('/')[-1][:-4]
                if id in self.error_json_ids:
                    continue
                image_file = os.path.join(image_folder, line)
                self.images.append(image_file)
                line_target = line.replace('.jpg', '.json')
                label_file = os.path.join(labels_folder, line_target)
                self.targets.append(label_file)

    def _valid_target_types(self):
        valid_target_types = ["semantic", "semantic_culane", "semantic_culane_lanetrainer"]
        return valid_target_types

    def _valid_splits(self):
        valid_splits = ["train", "val", "test", "trainval", "trainvaltest"]
        return valid_splits

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        target_path = os.path.join(self.root, self.targets[idx])

        img = Image.open(img_path).convert('RGB')
        target_lanes, gt_laneline_category, bottom_points, gt_lane_pts, attributes = self.preprocess_data(target_path)
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
                                                                               length_threshold=0, merge_lane_pixel_threshold=300)
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

    def __len__(self):
        return len(self.images)

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

    def preprocess_data(self, path):
        with open(path) as f:
            info_dict = json.load(f)

        cam_pitch = 0.5 / 180 * np.pi
        cam_height = 1.5
        cam_extrinsics = np.array([[np.cos(cam_pitch), 0, -np.sin(cam_pitch), 0],
                                   [0, 1, 0, 0],
                                   [np.sin(cam_pitch), 0, np.cos(cam_pitch), cam_height],
                                   [0, 0, 0, 1]], dtype=float)
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

        # gt_cam_height = info_dict['cam_height']
        gt_cam_height = cam_extrinsics[2, 3]  # TODO:check the height
        # gt_cam_height = 2.0
        gt_cam_pitch = 0

        cam_intrinsics = info_dict['calibration']
        cam_intrinsics = np.array(cam_intrinsics)
        cam_intrinsics = cam_intrinsics[:, :3]

        _label_cam_height = gt_cam_height
        _label_cam_pitch = gt_cam_pitch

        gt_lanes_packed = info_dict['lanes']
        gt_lane_pts, gt_lane_visibility, gt_laneline_category = [], [], []
        for i, gt_lane_packed in enumerate(gt_lanes_packed):
            # A GT lane can be either 2D or 3D
            # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
            lane = np.array(gt_lane_packed).T
            # lane[2,:]=lane[2,:]/100.0 #TODO: check the unit of z

            # Coordinate convertion for openlane_300 data
            lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
            # cam_representation = np.linalg.inv(
            #                         np.array([[0, 0, 1, 0],
            #                                     [-1, 0, 0, 0],
            #                                     [0, -1, 0, 0],
            #                                     [0, 0, 0, 1]], dtype=float))  # transformation from apollo camera to openlane camera
            # lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))
            lane = np.matmul(cam_extrinsics, lane)

            lane = lane[0:3, :].T
            lane = lane[lane[:, 1].argsort()]  # TODO:make y mono increase
            # lane = np.array(gt_lane_packed)
            gt_lane_pts.append(lane)
            gt_lane_visibility.append(1.0)
            gt_laneline_category.append(1)

        # _label_laneline_org = copy.deepcopy(gt_lane_pts)
        _gt_laneline_category_org = copy.deepcopy(np.array(gt_laneline_category))

        cam_K = cam_intrinsics
        gt_cam_height = _label_cam_height
        gt_cam_pitch = _label_cam_pitch
        P_g2im = projection_g2im(gt_cam_pitch, gt_cam_height, cam_K)
        H_g2im = homograpthy_g2im(gt_cam_pitch, gt_cam_height, cam_K)
        H_im2g = np.linalg.inv(H_g2im)

        P_g2gflat = np.matmul(H_im2g, P_g2im)

        gt_lanes = gt_lane_pts
        gt_visibility = gt_lane_visibility
        gt_category = gt_laneline_category

        # prune gt lanes by visibility labels
        gt_lanes = [prune_3d_lane_by_visibility(gt_lane, gt_visibility[k]).squeeze(0) for k, gt_lane in
                    enumerate(gt_lanes)]
        _label_laneline_org = copy.deepcopy(gt_lanes)

        # prune out-of-range points are necessary before transformation
        gt_lanes = [prune_3d_lane_by_range(gt_lane, -30, 30) for gt_lane in gt_lanes]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        gt_lane_pts = gt_lanes

        img_lanes = []
        bottom_points = []

        min_points = []
        max_points = []
        polynomials_third = []

        gt_lane_pts_flat = copy.deepcopy(gt_lane_pts)
        # convert 3d lanes to flat ground space
        convert_lanes_3d_to_gflat(gt_lane_pts_flat, P_g2gflat)

        attributes = [1] * len(gt_lane_pts)
        bottom_point_extraction_from_3d_world(gt_lane_pts=gt_lane_pts, gt_lane_pts_flat=gt_lane_pts_flat,
                                              attributes=attributes, P_g2im=P_g2im, polynomials_third=polynomials_third,
                                              min_points=min_points, max_points=max_points,
                                              bottom_points=bottom_points, img_lanes=img_lanes)

        return img_lanes, gt_laneline_category, bottom_points, gt_lane_pts, attributes
