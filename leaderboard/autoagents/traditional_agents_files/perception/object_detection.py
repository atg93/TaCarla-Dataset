import torch
import torch.utils.data

import cv2
import numpy as np
import json
import pickle
import math
import copy
import os
from leaderboard.autoagents.traditional_agents_files.perception.detection_utils.carla_to_lss import Carla_to_Lss_Converter
from leaderboard.autoagents.traditional_agents_files.perception.detection_utils.geometry import scale_and_zoom, get_pose_matrix, euler_to_quaternion

import sys
sys.path.append('/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/tairvision_center_line/')
sys.path.append('/cta/users/tgorgulu/git/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/tairvision_center_line/')
from tairvision.models.bev.lss.training.trainer import TrainingModule
from tairvision.models.bev.lss.utils.visualization import VisualizationModule
from tairvision_utils.bbox import get_center_size_angle, view_boxes_to_lidar_boxes_xdyd, view_boxes_to_lidar_boxes_xdyd_2
#from leaderboard.autoagents.traditional_agents_files.perception.tairvision_object_detection.models.bev.lss.utils.bbox import get_center_size_angle
from tairvision.models.bev.common.nuscenes.process import (ResizeCropRandomFlipNormalize, FilterClasses,
                                                           get_resizing_and_cropping_parameters)
from tairvision.models.bev.common.utils.geometry import cumulative_warp_features, calculate_birds_eye_view_parameters
from collections import deque

import pyquaternion as pyq

import carla

from leaderboard.autoagents.traditional_agents_files.perception.tairvision.models.bev.lss.online_visualize import Run_Detection_with_Carla

from leaderboard.autoagents.traditional_agents_files.perception.tairvision_utils.prepare_lidar_data import Prepare_Lidar_Data

from PIL import Image

class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttributeDict(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Attribute {key} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = AttributeDict(value)
        self[key] = value


class Object_Detection:
    def __init__(self,config):
        self.config = config
        self.intrinsics = {}
        self.extrinsics = {}
        with open('/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/tugrul_json.json') as json_file:
            cfg = json.load(json_file)
        cfg["GPUS"] = '0'
        cfg["BATCHSIZE"] = 1
        a_cfg = AttributeDict(cfg)
        #dummy_model = LiftSplatLinear(a_cfg)
        detection_threshold = config['detection_threshold']
        assert os.path.exists('/workspace/tg22/object_detection/carlaBasicCamLidarDet.ckpt')
        checkpoint = torch.load('/workspace/tg22/object_detection/carlaBasicCamLidarDet.ckpt', map_location='cpu')
        #key_list = list(checkpoint['state_dict'].keys())

        """for key in key_list:
            new_key = ''.join(list(key)[6:])
            dummy_checkpoint[new_key] = checkpoint['state_dict'][key]
            del checkpoint['state_dict'][key]"""

        #with open('/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/module.pickle', 'rb') as pickle_file:
        #    self.module = pickle.load(pickle_file)
        #self.model = dummy_model
        #state_dict = checkpoint['state_dict']
        #self.pcloud_list_collator = PCloudListCollator(cfg["PCLOUD"]["N_FEATS"])

        self.config["log_detection"] = True

        self.module = TrainingModule.load_from_checkpoint('/workspace/tg22/object_detection/carlaBasicCamLidarDet.ckpt', strict=True)
        print(f'Loaded weights from \n {self.config["perception_checkpoint"]}')
        self.module.eval()
        self.module.cuda(2)
        #self.module.eval()
        #self.module.to(torch.device('cuda:0'))
        with open('/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/module_model_cfg.pickle', 'rb') as pickle_file:
            self.module_model_cfg = pickle.load(pickle_file)
        cfg = self.module.cfg
        self.cfg_perception = self.load_perception_cfg_settings(self.module.cfg)
        self.augmentation_parameters = get_resizing_and_cropping_parameters(self.cfg_perception)
        self.transforms_val = ResizeCropRandomFlipNormalize(self.augmentation_parameters,
                                                            enable_random_transforms=False)
        self.visualization_module = VisualizationModule(self.cfg_perception, 0.50, detection_threshold)
        self.ego_queue = deque(maxlen=20)
        self.det_bev_queue_vehicle = deque(maxlen=20)
        self.det_bev_queue_walker = deque(maxlen=20)
        self.previous_ego = None
        self.spatial_extent = (self.cfg_perception.LIFT.X_BOUND[1], self.cfg_perception.LIFT.Y_BOUND[1])
        self.bev_resolution, _, _ = calculate_birds_eye_view_parameters(
            self.cfg_perception.LIFT.X_BOUND, self.cfg_perception.LIFT.Y_BOUND, self.cfg_perception.LIFT.Z_BOUND)
        self.bev_resolution = self.bev_resolution.numpy()
        self.view_array = self.create_view_array()
        self.carla_to_lss = Carla_to_Lss_Converter(self.config["monocular_perception"])

        self.ego_motion_que = deque(maxlen=2)
        self.ego_motion_que_init = True

        """b_res, b_st, b_dim = calculate_birds_eye_view_parameters(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
        self.b_res, self.b_st, self.b_dim = (b_res.numpy(), b_st.numpy(), b_dim.numpy())"""

        self.run_detection = Run_Detection_with_Carla()

        self.prepare_lidar_data = Prepare_Lidar_Data()

        self.cams = ['front_left', 'front', 'front_right', 'back_left', 'back', 'back_right']

        self.get_detection_count = 0
        self.get_detection_threshold = 1

    def get_info(self):
        return self.bev_resolution, self.view_array

    def get_perception_cfg(self):
        return self.cfg_perception

    def load_perception_cfg_settings(self, cfg):
        cfg.GPUS = "[0]"
        cfg.BATCHSIZE = 1
        # cfg.DATASET.DATAROOT = self.args_dict['dataroot']
        # cfg.DATASET.VERSION = self.args_dict['version']
        return cfg

    def create_view_array(self):
        arr = np.zeros((1, 1, 1, 4, 4), dtype=np.float32)
        if self.config["monocular_perception"]:
            arr[0, 0, 0] = [[0., -2., 0., 31.5],
                            [-2., 0., 0., 95.5],
                            [0., 0., 0.8, 4.3],
                            [0., 0., 0., 1.]]
        else:
            arr[0, 0, 0] = [[0, -2, 0, 99.5],
                            [-2, 0, 0, 99.5],
                            [0, 0, 0.8, 4.3],
                            [0, 0, 0, 1]]
        return arr

    def get_variables(self):
        return

    def get_lidar_info(self,sensors):
        x, y, z, yaw, pitch, roll = sensors[1]['x'], sensors[1]['y'], sensors[1]['z'], sensors[1]['yaw'], \
                                    sensors[1]['pitch'], sensors[1]['roll']
        lidar_ext = self.carla_to_lss.find_ext(x, y, z, roll, pitch, yaw)
        return lidar_ext

    def sensors(self):
        sensors = []
        w = 1600#704
        h = 900#396
        # Add cameras
        new_camera = {'type': 'sensor.camera.rgb', 'x': 0.70079118954, 'y': 0.0159456324149, 'z': 1.51095763913,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'width': w, 'height': h, 'fov': 70, 'id': 'front'}
        sensors.append(new_camera)
        self.intrinsics["front"], self.extrinsics["front"] = self.carla_to_lss.find_intrinsics(new_camera['width'], new_camera['height'], new_camera['fov'],
                                                                                                       new_camera['x'],
                                                                                                       new_camera['y'],
                                                                                                       new_camera['z'],
                                                                                                       new_camera['roll'], new_camera['pitch'],
                                                                                                       new_camera['yaw'])



        new_camera = {'type': 'sensor.camera.rgb', 'x': 0.5508477543, 'y': 0.493404796419, 'z': 1.49574800619,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                        'width': w, 'height': h, 'fov': 70, 'id': 'front_right'}
        sensors.append(new_camera)
        self.intrinsics["front_right"], self.extrinsics["front_right"] = self.carla_to_lss.find_intrinsics(new_camera['width'], new_camera['height'], new_camera['fov'],
                                                                                                       new_camera['x'],
                                                                                                       new_camera['y'],
                                                                                                       new_camera['z'],
                                                                                                       new_camera['roll'], new_camera['pitch'],
                                                                                                       new_camera['yaw'])


        new_camera = {'type': 'sensor.camera.rgb', 'x': 0.52387798135, 'y': -0.494631336551, 'z': 1.50932822144,
             'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
             'width': w, 'height': h, 'fov': 70, 'id': 'front_left'}
        sensors.append(new_camera)
        self.intrinsics["front_left"], self.extrinsics["front_left"] = self.carla_to_lss.find_intrinsics(new_camera['width'], new_camera['height'], new_camera['fov'],
                                                                                                       new_camera['x'],
                                                                                                       new_camera['y'],
                                                                                                       new_camera['z'],
                                                                                                       new_camera['roll'], new_camera['pitch'],
                                                                                                       new_camera['yaw'])



        new_camera = {'type': 'sensor.camera.rgb', 'x': -1.5283260309358, 'y': 0.00345136761476, 'z': 1.57910346144,
             'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
             'width': w, 'height': h, 'fov': 110, 'id': 'back'}
        sensors.append(new_camera)
        self.intrinsics["back"], self.extrinsics["back"] = self.carla_to_lss.find_intrinsics(new_camera['width'], new_camera['height'], new_camera['fov'],
                                                                                                       new_camera['x'],
                                                                                                       new_camera['y'],
                                                                                                       new_camera['z'],
                                                                                                       new_camera['roll'], new_camera['pitch'],
                                                                                                       new_camera['yaw'])


        new_camera = {'type': 'sensor.camera.rgb', 'x': -0.53569100218, 'y': -0.484795032713, 'z': 1.59097014818,
             'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
             'width': w, 'height': h, 'fov': 70, 'id': 'back_left'}
        sensors.append(new_camera)
        self.intrinsics["back_left"], self.extrinsics["back_left"] = self.carla_to_lss.find_intrinsics(new_camera['width'], new_camera['height'], new_camera['fov'],
                                                                                                       new_camera['x'],
                                                                                                       new_camera['y'],
                                                                                                       new_camera['z'],
                                                                                                       new_camera['roll'], new_camera['pitch'],
                                                                                                       new_camera['yaw'])



        new_camera = {'type': 'sensor.camera.rgb', 'x': -0.5148780988, 'y': 0.480568219723, 'z': 1.56239545128,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
             'width': w, 'height': h, 'fov': 70, 'id': 'back_right'}
        sensors.append(new_camera)
        self.intrinsics["back_right"], self.extrinsics["back_right"] = self.carla_to_lss.find_intrinsics(new_camera['width'], new_camera['height'], new_camera['fov'],
                                                                                                       new_camera['x'],
                                                                                                       new_camera['y'],
                                                                                                       new_camera['z'],
                                                                                                       new_camera['roll'], new_camera['pitch'],
                                                                                                       new_camera['yaw'])


        #for i in self.intrinsics:
        #    self.intrinsics[i] = self.transforms_val.update_intrinsics_v3(self.intrinsics[i])

        return sensors, self.intrinsics, self.extrinsics

    def calculate_ego_motion(self, input_data, vehicle, current_wp, carla_map):
        _, _, _, diff_location, diff_yaw_degree = self.get_rotation_translation(input_data, vehicle, current_wp, carla_map)

        future_egomotion = self.get_future_egomotion(self.ego_motion_que[0]['rotation'], self.ego_motion_que[0]['translation'],self.ego_motion_que[-1]['rotation'], self.ego_motion_que[-1]['translation'])

        return future_egomotion.unsqueeze(1)


    def get_gnss_data(self, raw_data):
        latitude, longitude, altitude = raw_data[1][0], raw_data[1][1], raw_data[1][2]
        lat_rad = (np.deg2rad(latitude) + np.pi) % (2 * np.pi) - np.pi
        lon_rad = (np.deg2rad(longitude) + np.pi) % (2 * np.pi) - np.pi
        R = 6378135  # Equatorial radius in meters
        x = R * np.sin(lon_rad) * np.cos(lat_rad)  # i0
        y = R * np.sin(-lat_rad)  # i0
        z = altitude
        return carla.Location(x=x,y=y,z=z)

    def convert_compass_to_yaw(self, compass_radians):
        # Convert from North-based to East-based, assuming compass increases clockwise
        yaw_radians = -compass_radians % (math.pi)
        #(math.pi / 2 - compass_radians) % (2 * math.pi)
        # Convert radians to degrees
        yaw_degrees = yaw_radians * (180 / math.pi)

        return yaw_radians, yaw_degrees

    def get_rotation_translation(self, input_data, vehicle, current_wp, carla_map):
        vehicle.get_transform().location, vehicle.get_transform().rotation, input_data['gps'][1], input_data['imu'][1], \

        #self.gps_to_cartesian(carla_map, input_data['gps'][1][0], input_data['gps'][1][1], input_data['gps'][1][2])
        sensor_location = self.get_gnss_data(input_data['gps'])

        ev_rot = vehicle.get_transform().rotation
        ev_loc = vehicle.get_transform().location
        diff_location = ev_loc.distance(sensor_location)
        ev_loc = sensor_location
        compass = input_data['imu'][1][-1]
        # Convert from North-based to East-based, assuming compass increases clockwise

        yaw = (3 * math.pi / 2 + compass) % (2 * math.pi)
        yaw_degree = (yaw * 180)/math.pi

        real_yaw_rad = (ev_rot.yaw * math.pi)/180
        diff_yaw_degree = ev_rot.yaw - yaw_degree
        ego = {}
        ego['rotation'] = euler_to_quaternion(-current_wp.rotation.roll, current_wp.rotation.pitch, -yaw_degree)
        ego['translation'] = ev_loc.x, -ev_loc.y, ev_loc.z
        #ego_pose_matrix = get_pose_matrix(ego, use_flat=False) #input_data['speed'][1]
        if self.ego_motion_que_init:
            self.ego_motion_que_init = False
            for _ in range(self.ego_motion_que.maxlen):
                self.ego_motion_que.append(ego)
        else:
            self.ego_motion_que.append(ego)

        return ego, ego['rotation'], ego['translation'], diff_location, diff_yaw_degree

    def __call__(self, input_data, vehicle, current_wp, carla_map):
        if self.get_detection_count % self.get_detection_threshold == 0:
            ego, _, _, self.diff_location, self.diff_yaw_degree = self.get_rotation_translation(input_data, vehicle, current_wp, carla_map)
            _, _, input_lidar_data = self.prepare_lidar_data(input_data)
            image_dict = {}
            for camera in self.cams:
                image_dict.update({camera:Image.fromarray(input_data[camera][1])})
            current_data_dict = {'image':image_dict, 'lidar':np.vstack(input_lidar_data), 'ego_pose_dict':ego}
            output = self.run_detection(current_data_dict)
            self.plant_boxes, self.bbox_list, self.orientation_angle_list = None, None, None
            self.plant_boxes = view_boxes_to_lidar_boxes_xdyd_2(output, self.run_detection.detection_dataset.view.unsqueeze(0))#self.run_detection.detection_dataset.view

            self.masks = self.plot_boxes(output)[0]
            self.eliminate_bbox(output)

        self.get_detection_count += 1

        return self.masks, self.plant_boxes, self.bbox_list, self.orientation_angle_list, self.diff_location, \
               self.diff_yaw_degree

    

    def eliminate_bbox(self,output):
        new_output = []
        samples = output['head3d'][0][0]
        new_boxes = samples['boxes'][samples['scores'] > 0.4]
        if len(new_boxes) != 0:
            asd = 0

    def dic2matris(self,intrinsics, extrinsics):
        new_instrincsic = []
        for vec in intrinsics.values():
            new_instrincsic.append(vec)
        
        new_instrincsic = torch.stack(new_instrincsic)

        new_extrinsics = []
        for vec in extrinsics.values():
            new_extrinsics.append(vec)

        new_extrinsics = torch.stack(new_extrinsics)

        return new_instrincsic, new_extrinsics

    def get_parameters(self):
        return self.intrinsics, self.extrinsics, self.transforms_val, self.carla_to_lss



    def plot_boxes(self, output):

        lanes = None
        lines = None
        image = np.zeros([200, 200], dtype=np.uint8)

        view_boxes_vehicle_pred = self.visualization_module.plot_view_boxes(output['head3d'][0:1][0], lanes, lines,
                                                                            bev_size=self.visualization_module.bev_size,
                                                                            score_threshold=0.45)

        view_boxes_vehicle_pred *= 255
        #cv2.imwrite("view_boxes_vehicle_pred.png",view_boxes_vehicle_pred)
        view_boxes_walker_pred = self.visualization_module.plot_view_boxes(output['head3d'][0:1][0], lanes, lines,
                                                                           bev_size=self.visualization_module.bev_size,
                                                                           score_threshold=0.45)
        view_boxes_walker_pred *= 255

        return view_boxes_vehicle_pred, view_boxes_walker_pred#np.expand_dims(view_boxes_vehicle_pred, axis=-1), np.expand_dims(view_boxes_walker_pred, axis=-1)

    def get_future_egomotion(self, ego_rotation, ego_translation, ego_rotation_next_rotation, ego_translation_next_translation):

        ego_rot = pyq.Quaternion(np.array(ego_rotation)).rotation_matrix
        ego_trans = np.array(ego_translation)
        ego_pose_matrix = np.eye(4)
        ego_pose_matrix[:3, :3] = ego_rot
        ego_pose_matrix[:3, 3] = ego_trans
        ego_t0 = ego_pose_matrix

        # Identity
        view = torch.tensor([[[[[0, -2, 0, 99.5],
                                 [-2, 0, 0, 99.5],
                                 [0, 0, 0.8, 4.3],
                                 [0, 0, 0, 1]]]]]).squeeze(0).squeeze(0).squeeze(0)
        sh, sw, _ = 1 / self.bev_resolution
        # future_egomotion = np.eye(4, dtype=np.float32)
        view_rot_only = np.eye(4, dtype=np.float32)
        view_rot_only[0, 0:2] = view[0, 0:2] / sw
        view_rot_only[1, 0:2] = view[1, 0:2] / sh

        ego_rot = pyq.Quaternion(np.array(ego_rotation_next_rotation)).rotation_matrix
        ego_trans = np.array(ego_translation_next_translation)
        ego_t1 = np.eye(4)
        ego_t1[:3, :3] = ego_rot
        ego_t1[:3, 3] = ego_trans

        future_egomotion = np.linalg.inv(ego_t1) @ ego_t0
        future_egomotion = view_rot_only @ future_egomotion @ np.linalg.inv(view_rot_only)

        future_egomotion[3, :3] = 0.0
        future_egomotion[3, 3] = 1.0

        future_egomotion = torch.Tensor(future_egomotion).float()

        return future_egomotion.unsqueeze(0)

