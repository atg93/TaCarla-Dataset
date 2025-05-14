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
import pickle
import os.path as osp


class Klane(GenericSegmentationVisionDataset, ABC):
    KlanesClass = namedtuple('KlanesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                   'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        KlanesClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        KlanesClass('Line', 1, 1, 'line', 0, False, False, (180, 200, 30)),
    ]

    culane_classes = [
        KlanesClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        KlanesClass('Line 1', 1, 1, 'line', 0, True, False, (180, 200, 30)),
        KlanesClass('Line 2', 2, 2, 'line', 0, True, False, (200, 100, 1)),
        KlanesClass('Line 3', 3, 3, 'line', 0, True, False, (1, 100, 200)),
        KlanesClass('Line 4', 4, 4, 'line', 0, True, False, (30, 200, 180))
    ]

    # params to visulize on image
    line_thickness = 15

    # params of lidar coordinate
    x_grid = 0.32
    y_grid = 0.16
    z_fix = -1.43  # grounds

    # params to visualize in pointcloud
    z_fix_label = -1.5
    z_fix_conf = -1.1
    z_fix_cls = -1.1

    image_width = 1920
    image_height = 1200

    def __init__(self,
                 lane_width_radius=None,
                 lane_width_radius_for_metric=None,
                 lane_width_radius_for_uncertain=None,
                 lane_width_radius_for_binary=None,
                 **kwargs) -> None:
        super(Klane, self).__init__(**kwargs)

        self.root = os.path.join(self.root, "klane")

        self.valid_modes = self.valid_splits
        self.images = []
        self.targets = []
        self.calib_info_list = []

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
        seq_list = os.listdir(self.root)
        seq_list.sort()
        for seq in seq_list:
            image_path = os.path.join(seq, "frontal_img")
            anno_path = os.path.join(seq, "bev_tensor_label")
            img_list = os.listdir(osp.join(self.root, image_path))
            img_list.sort()
            calib_txt_file = os.path.join(self.root, seq, "calib_seq.txt")
            with open(calib_txt_file, 'r') as calib_file:
                calib_info = [float(x) for x in calib_file.read().split(',') if x != '']

            for img in img_list:
                img_path = os.path.join(image_path, img)
                target_file = img.replace(".jpg", ".pickle")
                target_file = target_file.replace("frontal_img", "bev_tensor_label")
                target_path = os.path.join(anno_path, target_file)
                if osp.isfile(os.path.join(self.root, target_path)):
                    self.images.append(img_path)
                    self.targets.append(target_path)
                    self.calib_info_list.append(calib_info)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        target_path = os.path.join(self.root, self.targets[idx])
        calib = self.calib_info_list[idx]

        img = Image.open(img_path).convert('RGB')
        target_lanes = self.load_target(target_path, calib)
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


    def _valid_target_types(self):
        valid_target_types = ["semantic", "semantic_culane", "semantic_culane_lanetrainer"]
        return valid_target_types

    def _valid_splits(self):
        valid_splits = ["train", "val", "trainval"]
        return valid_splits

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

    def __len__(self):
        return len(self.images)

    def load_target(self, target_path, calib):
        intrinsic = calib[:4]
        extrinsic = calib[4:]

        rot, tra = self.get_rotation_and_translation_from_extrinsic(extrinsic)
        with open(target_path, 'rb') as f:
            bev_tensor_label = pickle.load(f, encoding='latin1')
        pc_label, arr_cls_idx = self.get_point_cloud_from_bev_tensor_label(bev_tensor_label, with_cls=True)
        pc_label = self.get_pointcloud_with_rotation_and_translation(pc_label, rot, tra)
        pixel_label = self.get_pixel_from_point_cloud_in_camera_coordinate(pc_label, intrinsic)

        process_pixel = pixel_label.copy()
        if arr_cls_idx is not None:
            num_points, _ = np.shape(pixel_label)
            process_cls_idx = np.reshape(arr_cls_idx.copy(), (num_points, 1))
            process_pixel = np.concatenate((process_pixel, process_cls_idx), axis=1)
        temp_process_pixel = np.array(list(filter(lambda x: (x[0] >= 0) and (x[0] < self.image_width - 0.5) and \
                                                            (x[1] >= 0) and (x[1] < self.image_height - 0.5), process_pixel)))
        process_pixel = temp_process_pixel[:, :2]
        process_cls_idx = temp_process_pixel[:, 2]

        classes_ids = np.unique(process_cls_idx).tolist()
        # round digits
        process_pixel = np.around(process_pixel)
        lanes = []
        for cls_idx in classes_ids:
            lane = process_pixel[process_cls_idx == cls_idx]
            lanes.append(lane)

        return lanes

    @staticmethod
    def get_rotation_and_translation_from_extrinsic(extrinsic, is_deg=True):
        ext_copy = extrinsic.copy()  # if not copy, will change the parameters permanently
        if is_deg:
            ext_copy[:3] = list(map(lambda x: x * np.pi / 180., extrinsic[:3]))

        roll, pitch, yaw = ext_copy[:3]
        x, y, z = ext_copy[3:]

        ### Roll-Pitch-Yaw Convention
        c_y = np.cos(yaw)
        s_y = np.sin(yaw)
        c_p = np.cos(pitch)
        s_p = np.sin(pitch)
        c_r = np.cos(roll)
        s_r = np.sin(roll)

        R_yaw = np.array([[c_y, -s_y, 0.], [s_y, c_y, 0.], [0., 0., 1.]])
        R_pitch = np.array([[c_p, 0., s_p], [0., 1., 0.], [-s_p, 0., c_p]])
        R_roll = np.array([[1., 0., 0.], [0., c_r, -s_r], [0., s_r, c_r]])

        R = np.dot(np.dot(R_yaw, R_pitch), R_roll)
        trans = np.array([[x], [y], [z]])

        return R, trans

    def get_point_cloud_from_bev_tensor_label(self, bev_label, with_cls=False):
        '''
        * return
        *   n x 3 (x,y,z) [m] in np.array, with_cls == False
        *   n x 3 (x,y,z) [m], n x 1 (cls_idx) in np.array, with_cls == True
        '''
        bev_label_144 = bev_label[:, :144]

        points_arr = []
        cls_arr = []
        for i in range(6):
            points_in_pixel = np.where(bev_label_144 == i)
            _, num_points = np.shape(points_in_pixel)
            for j in range(num_points):
                x_point, y_point = self.get_point_from_pixel_in_m(points_in_pixel[1][j], points_in_pixel[0][j])
                points_arr.append([x_point, y_point, self.z_fix])
                if with_cls:
                    cls_arr.append(i)  # cls

        if with_cls:
            return np.array(points_arr), np.array(cls_arr)
        else:
            return np.array(points_arr)

    def get_point_from_pixel_in_m(self, x_pix, y_pix):
        x_lidar = 144 - (y_pix + 0.5)
        y_lidar = 72 - (x_pix + 0.5)

        return self.x_grid * x_lidar, self.y_grid * y_lidar

    @staticmethod
    def get_pointcloud_with_rotation_and_translation(point_cloud_xyz, rot, tra):
        pc_xyz = point_cloud_xyz.copy()
        num_points = len(pc_xyz)

        for i in range(num_points):
            point_temp = pc_xyz[i, :]
            point_temp = np.reshape(point_temp, (3, 1))

            point_processed = np.dot(rot, point_temp) + tra
            point_processed = np.reshape(point_processed, (3,))

            pc_xyz[i, :] = point_processed

        return pc_xyz

    @staticmethod
    def get_pixel_from_point_cloud_in_camera_coordinate(point_cloud_xyz, intrinsic):
        '''
        * in : pointcloud in np array (nx3)
        * out: projected pixel in np array (nx2)
        '''

        process_pc = point_cloud_xyz.copy()
        if (np.shape(point_cloud_xyz) == 1):
            num_points = 0
        else:
            # Temporary fix for when shape = (0.)
            try:
                num_points, _ = np.shape(point_cloud_xyz)
            except:
                num_points = 0
        fx, fy, px, py = intrinsic

        pixels = []
        for i in range(num_points):
            xc, yc, zc = process_pc[i, :]
            y_pix = py - fy * zc / xc
            x_pix = px - fx * yc / xc

            pixels.append([x_pix, y_pix])
        pixels = np.array(pixels)

        return pixels
