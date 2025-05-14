

import os
import json
import time
from pathlib import Path
from tkinter import N
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageOps

import cv2
import torch
import numpy as np
import carla
import torch.nn as nn

#from filterpy.kalman import MerweScaledSigmaPoints
#from filterpy.kalman import UnscentedKalmanFilter as UKF

from tair_traditional_agents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.agent_utils.filter_functions import *
from tair_traditional_agents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.agent_utils.coordinate_utils import \
    preprocess_compass, inverse_conversion_2d

from tair_traditional_agents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.data_agent_boxes import \
    DataAgent
from tair_traditional_agents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.training.PlanT.dataset import \
    generate_batch, split_large_BB
from tair_traditional_agents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.training.PlanT.lit_module import \
    LitHFLM

from tair_traditional_agents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.nav_planner import \
    extrapolate_waypoint_route
from tair_traditional_agents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.nav_planner import \
    RoutePlanner_new as RoutePlanner

import pyproj

import pickle
from omegaconf import OmegaConf

import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_entry_point():
    return 'PlanTAgent'

SAVE_GIF = os.getenv("SAVE_GIF", 'False').lower() in ('true', '1', 't')

class PlanTAgent(DataAgent):

    def save_score(self, name, file_name_without_extension, save_files_name, score_composed, score_route,
                   score_penalty, copy_statistics_manager, weather_state):
        super().save_score(name, file_name_without_extension, save_files_name, score_composed, score_route,
                           score_penalty, copy_statistics_manager, weather_state)

    def update_control(self, control):
        return super().update_control(control)

    def set_info(self, _pixels_per_meter, _world_offset):
        self._pixels_per_meter = _pixels_per_meter
        self._world_offset = _world_offset

    def setup(self, path_to_conf_file, task_name, route_index=None, cfg=None, exec_or_inter=None):
        self.cfg = cfg
        self.exec_or_inter = exec_or_inter

        path_to_conf_file = os.environ['code_path'] + '/leaderboard/autoagents/traditional_agents_files/carla_gym/core/obs_manager/birdview/planning/util'
        assert os.path.exists(path_to_conf_file)
        self.sensors_info = [{'type': 'sensor.opendrive_map', 'reading_frequency': 1e-06, 'id': 'hd_map'},
                             {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'speed'},
                             {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0,
                              'yaw': 0.0, 'sensor_tick': 0.05, 'id': 'imu'},
                             {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0,
                              'yaw': 0.0, 'sensor_tick': 0.01, 'id': 'gps'}]

        # first args than super setup is important!
        args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
        self.args = json.load(args_file)
        #args_file.close()
        self.cfg_agent = OmegaConf.create(self.args)

        super().setup(path_to_conf_file, task_name, route_index, cfg, exec_or_inter)

        print(f'Saving gif: {SAVE_GIF}')

        # Used to set the filter state equal the first measurement
        self.filter_initialized = False
        # Stores the last filtered positions of the ego vehicle.
        # Used to realign.
        self.state_log = deque(maxlen=2)

        LitHFLM_cfg = OmegaConf.load(os.environ.get('code_path') + '/leaderboard/autoagents/traditional_agents_files/carla_gym/core/obs_manager/birdview/planning/util/LitHFLM_config.yaml')

        self.net = LitHFLM(LitHFLM_cfg)


        self.colors = [(0, 255, 0), (192, 192, 192), (128, 128, 128),
                  (255, 255, 255), (128, 128, 0), (0, 128, 0), (128, 0, 128),
                  (0, 128, 128), (0, 0, 128), (255, 165, 0), (255, 20, 147),
                  (75, 0, 130), (240, 230, 140), (220, 20, 60), (47, 79, 79),
                  (255, 215, 0), (218, 112, 214), (199, 21, 133), (0, 100, 0),
                  (255, 69, 0), (128, 128, 255), (139, 69, 19), (244, 164, 96),
                  (64, 224, 208), (0, 0, 139), (238, 130, 238), (135, 206, 250),
                  (152, 251, 152), (106, 90, 205), (0, 250, 154), (85, 107, 47),
                  (72, 61, 139), (0, 191, 255), (250, 128, 114), (34, 139, 34),
                  (255, 99, 71), (210, 105, 30), (32, 178, 170), (255, 140, 0),
                  (255, 222, 173), (153, 50, 204), (255, 160, 122), (70, 130, 180),
                  (0, 255, 127), (216, 191, 216), (238, 232, 170), (152, 251, 152),
                  (255, 228, 181), (250, 250, 210), (127, 255, 212), (216, 191, 216),
                  (255, 228, 225), (245, 245, 220), (255, 235, 205), (255, 228, 196),
                  (255, 239, 213), (255, 245, 238), (255, 250, 205), (255, 255, 224)]

        asd = 0

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def set_vehicle(self, _vehicle):
        super().set_vehicle(_vehicle)

    def _init(self, hd_map=None):
        super()._init(hd_map)
        self.super_class = super()
        self._route_planner = RoutePlanner(7.5, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.save_mask = []
        self.save_topdowns = []
        self.timings_run_step = []
        self.timings_forward_model = []

        self.keep_ids = None

        self.control = carla.VehicleControl()
        self.control.steer = 0.0
        self.control.throttle = 0.0
        self.control.brake = 1.0

        self.initialized = True
        self.initial_gps_value = None

    def sensors(self):
        result = super().sensors()
        return result

    def set_dummy_target_location(self, dummy_target_location):
        self.dummy_target_location = dummy_target_location

    def tick(self, input_data,  lane_guidance):
        assert input_data != None
        result = super().tick(input_data, lane_guidance=lane_guidance)
        plant_input_image, gt_mask_vehicle_image, special_vehicle_image \
            , bike_and_cons_vehicle_image, tl_image, ego_mask, lane_guidance_mask, tl_light_stop = self.draw_label_raw(result['boxes'],
                                                                                                        'detection')
        #print("plant_agent_tick: ",tl_light_stop)
        input_data.update({"tl_light_stop":tl_light_stop})
        gps_input = input_data['gps'][1][:2]

        if type(self.initial_gps_value) == type(None):
            self.initial_gps_value = copy.deepcopy(gps_input)

        pos = self._route_planner.convert_gps_to_carla(gps_input)

        speed = input_data['speed'][1]['speed']
        compass = preprocess_compass(input_data['imu'][1][-1])

        result['gps'] = pos  # filtered_state[0:2]

        waypoint_route = self._route_planner.run_step(pos)

        if len(waypoint_route) > 2:
            target_point, _ = waypoint_route[1]
            next_target_point, _ = waypoint_route[2]
        elif len(waypoint_route) > 1:
            target_point, _ = waypoint_route[1]
            next_target_point, _ = waypoint_route[1]
        else:
            target_point, _ = waypoint_route[0]
            next_target_point, _ = waypoint_route[0]
        # print("target_point, result['gps'], compass:",target_point, result['gps'], compass)
        _ego_target_point = inverse_conversion_2d(target_point, result['gps'], compass)
        result['target_point'] = tuple(_ego_target_point)

        if SAVE_GIF == True and (self.exec_or_inter == 'inter'):
            result['rgb_back'] = input_data['rgb_back']
            result['sem_back'] = input_data['sem_back']



        return result #plant_input_image

    def gps_to_cartesian(self, lat, lon, alt):
        # Define the UTM zone and datum for your coordinates
        utm_zone = 33  # For example, UTM zone 33 for central Europe
        datum = 'WGS84'  # World Geodetic System 1984

        # Create a pyproj transformer for the conversion
        transformer = pyproj.Transformer.from_crs(f'EPSG:4326', f'+proj=utm +zone={utm_zone} +datum={datum}',
                                                  always_xy=True)

        # Convert GPS coordinates (latitude, longitude, altitude) to Cartesian coordinates (X, Y, Z)
        x, y, z = transformer.transform(lon, lat, alt)

        return x, y, z

    def rotate_point(self, x, y, cx, cy, angle_radians):
        # Convert angle to radians
        #angle_radians = np.radians(angle_degrees)

        # Translate point to origin
        translated_x = x - cx
        translated_y = y - cy

        # Apply rotation matrix
        rotated_x = translated_x * np.cos(angle_radians) - translated_y * np.sin(angle_radians)
        rotated_y = translated_x * np.sin(angle_radians) + translated_y * np.cos(angle_radians)

        # Translate back to original position
        final_x = rotated_x + cx
        final_y = rotated_y + cy

        return int(final_x), int(final_y)

    def draw_rectange(self, image,  top_left, bottom_right):              #mask_vehicle, center, top_left, bottom_right
        top_left = tuple(top_left[::-1].astype(np.uint8))
        bottom_right = tuple(bottom_right[::-1].astype(np.uint8))

        top_right = top_left[0], bottom_right[1]
        bottom_left = top_left[1], bottom_right[0]
        #rotated_x = ((rotated_x - cx) * 3) + cx
        #rotated_y = ((rotated_y - cy) * 3) + cy

        # Optionally, draw lines connecting the points
        cv2.line(image, top_left, top_right, (255, 255, 255), 1)
        cv2.line(image, top_right, bottom_right, (255, 255, 255), 1)
        cv2.line(image, bottom_right, bottom_left, (255, 255, 255), 1)
        cv2.line(image, bottom_left, top_left, (255, 255, 255), 1)

        return image

    def draw_vehicle(self, center, orientation_rad, velocity, box_size, bbox, arrow_thick=2):
        """
        Draw a vehicle's bounding box with orientation and velocity on an image using OpenCV.

        Args:
        - image: The image to draw on (numpy array).
        - center (tuple): The (x, y) center of the bounding box.
        - orientation_deg (float): The orientation of the vehicle in degrees.
        - velocity (float): The velocity of the vehicle.
        - box_size (tuple): The size of the bounding box (width, height).
        """
        mask_vehicle = np.zeros((200, 200)).astype(np.uint8)
        mask_arrow = np.zeros((200, 200)).astype(np.uint8)


        # Convert center to integer coordinates for drawing
        width, height = bbox[0] * 3, bbox[1] * 3
        top_left = (int(center[1] - width / 2), int(center[0] - height / 2))
        bottom_right = (int(center[1] + width / 2), int(center[0] + height / 2))


        # Draw the rectangle (bounding box)
        # cv2.rectangle(mask_vehicle, bottom_right, top_left, (255), 2)  # Red box


        # Combine the points into a numpy array with the correct shape
        top_right = (top_left[0], bottom_right[1])
        bottom_left = (bottom_right[0], top_left[1])
        pts = np.array([bottom_right, top_right, top_left, bottom_left], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        # Fill the polygon on the image
        cv2.fillPoly(mask_vehicle, [pts], color=(255))
        #cv2.imwrite('mask_vehicle.png', mask_vehicle)

        center = (int(center[1]), int(center[0]))

        # Calculate the end point of the orientation line (arrow)
        line_length = 10  # box_size[1] // 2
        line_length = min(max(velocity * line_length, line_length), line_length * 2)
        end_point = (int(center[0] - line_length * np.cos(orientation_rad + (np.pi / 2))),
                     int(center[1] - line_length * np.sin(orientation_rad + (np.pi / 2))))

        # Draw the orientation arrow
        cv2.arrowedLine(mask_arrow, center, end_point, (255), arrow_thick)  # Blue arrow

        # Display velocity
        # font = cv2.FONT_HERSHEY_SIMPLEX
        """cv2.putText(image, f'Vel: {velocity} m/s', (center[0], int(center[1] - box_size[1] // 2 - 10)),
                    font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)  # Green text"""

        return mask_vehicle.astype(np.bool), mask_arrow.astype(np.bool)

    def draw_pred_wp(self, plant_input_image, pred_wp, width=1, height=1, thickness=2):
        mask_wp = np.zeros((200, 200)).astype(np.uint8)
        real_mask_wp = np.zeros((200, 200)).astype(np.uint8)
        real_mask_wp_for_rule_based = np.zeros((200, 200)).astype(np.uint8)

        for index, center in enumerate(pred_wp):
            center[0] = center[0] * (-1)
            center = center * 4 + 100

            top_left = (int(center[1] - width / 2), int(center[0] - height / 2))
            bottom_right = (int(center[1] + width / 2), int(center[0] + height / 2))

            # Draw the rectangle (bounding box)
            if index == 0:
                cv2.rectangle(mask_wp, bottom_right, top_left, (255), int(thickness))
            cv2.rectangle(real_mask_wp, bottom_right, top_left, (255), int(thickness))

            if index == 0 or index == 1:
                cv2.rectangle(real_mask_wp_for_rule_based, bottom_right, top_left, (255), int(3))


        plant_input_image[real_mask_wp.astype(np.bool)] = (0, 255, 0)

        return plant_input_image, mask_wp, real_mask_wp_for_rule_based


    def draw_pred_wp_gru(self, plant_input_image, pred_wp, input_cars_info = None, width=1, height=1, thickness=2, scale_1=40, scale_2=90,
                         offset=torch.tensor([73,73]), offset_2=torch.tensor([2,2])):
        #center[0] = center[0] * (-1)
        #center = center * 4 + 100

        for index, center in enumerate(pred_wp):
            mask_wp = np.zeros((200, 200)).astype(np.uint8)

            center = (center - offset) / scale_1
            center = center * (-1)
            center = center * scale_2 + offset  # + np.array([100, 100])

            if type(input_cars_info) != type(None):
                ps_0 = np.array([input_cars_info[0].cpu(), input_cars_info[1].cpu()])
                ps_0[0] = ps_0[0] * (-1)
                ps_0[1] = ps_0[1] * (-1)
                ps_0 = ps_0 * 4 + np.array([100, 100])
                ps_0 = torch.tensor(ps_0)

                if index == 0:
                    diff_offset = center - ps_0
                    new_center = ps_0
                else:
                    new_center = center - diff_offset
            else:
                new_center = center


            top_left = (int(new_center[1] - width / 2), int(new_center[0] - height / 2))
            bottom_right = (int(new_center[1] + width / 2), int(new_center[0] + height / 2))

            # Draw the rectangle (bounding box)
            #if index == 0:
            cv2.rectangle(mask_wp, bottom_right, top_left, (255), int(thickness))

            plant_input_image[mask_wp.astype(np.bool)] = self.colors[index] #(0, 255, 255)

        return plant_input_image


    def draw_centerline(self, plant_input_image, label_raw, pred_wp, width=1, height=1):
        mask_wp = np.zeros((200, 200)).astype(np.uint8)
        last_wp = pred_wp.squeeze(0)[-1].cpu().numpy()
        last_wp[0] = last_wp[0] * (-1)
        last_wp = last_wp * 4 + 100

        for wp in label_raw:
            if wp['class'] == 'Centerline' and wp['distance'] < 50:
                center = copy.deepcopy(np.array([wp['position'][0], wp['position'][1]]))
                center[1] = center[1] * (-1)
                center[0] = center[0] * (-1)
                center = center * 4 + 100

                top_left = (int(center[1] - width / 2), int(center[0] - height / 2))
                bottom_right = (int(center[1] + width / 2), int(center[0] + height / 2))

                # Draw the rectangle (bounding box)
                cv2.rectangle(mask_wp, bottom_right, top_left, (255), 2)

        plant_input_image[mask_wp.astype(np.bool)] = (255, 255, 255)

        return plant_input_image, mask_wp

    def draw_label_raw(self, label_raw, name):
        image = np.zeros((200, 200, 3)).astype(np.uint8)
        mask_vehicle_image = np.zeros((200, 200)).astype(np.uint8)
        special_vehicle_image = np.zeros((200, 200)).astype(np.uint8)
        bike_and_cons_vehicle_image = np.zeros((200, 200)).astype(np.uint8)
        tl_image = np.zeros((200, 200)).astype(np.uint8)
        lane_guidance_mask = np.zeros((200, 200)).astype(np.uint8)
        ego_mask = np.zeros((200, 200)).astype(np.uint8)
        #ego_front_mask = np.zeros((200, 200)).astype(np.uint8)
        tl_light_stop = False
        for index, sample in enumerate(label_raw):
            center = np.array(sample['position'])
            if name == 'detection':
                center = np.array([center[0], center[1]])
            center *= (-1)
            center = center * 4 + 100

            if sample['class'] == 'Route':
                sample['speed'] = 0

            if sample['class'] == 'Car':

                bbox = np.array(sample["extent"])

                mask_vehicle, mask_arrow = self.draw_vehicle(center, sample['yaw'], sample['speed'],
                                                             (bbox[0], bbox[1]), bbox)

                if index == 0:
                    ego_mask = copy.deepcopy(mask_vehicle)
                    ego_front_mask, _ = self.draw_vehicle(np.array([86, 100]), sample['yaw'],
                                      sample['speed'],
                                      (bbox[0] / 2, bbox[1] / 2), bbox / 2)

                    image[mask_vehicle] = (255, 255, 255)
                    image[mask_arrow] = (255, 0, 0)
                    tl_ego_image = copy.deepcopy(mask_vehicle)

                else:
                    image[mask_vehicle] = (0, 0, 255)
                    image[mask_arrow] = (255, 0, 0)
                    _, mask_arrow = self.draw_vehicle(center, sample['yaw'], sample['speed'],
                                                      (bbox[0], bbox[1]), bbox)
                    mask_vehicle_image[mask_vehicle] = (255)
                    mask_vehicle_image[mask_arrow] = (255)

            elif sample['class'] == 'Police' or sample['class'] == 'Firetruck' or sample['class'] == 'Crossbike' or \
                    sample['class'] == 'Construction' or sample['class'] == 'Ambulance' or sample[
                'class'] == 'Walker' or sample['class'] == 'Route':

                bbox = np.array(sample["extent"])

                if sample['class'] == 'Crossbike':
                    bbox = [2.0, 2.0, 2.0]

                mask_vehicle, mask_arrow = self.draw_vehicle(center, sample['yaw'], sample['speed'],
                                                             (bbox[0], bbox[1]), bbox)

                if index == 0:
                    image[mask_vehicle] = (255, 255, 255)
                    image[mask_arrow] = (0, 255, 0)
                    tl_ego_image = copy.deepcopy(mask_vehicle)
                else:
                    image[mask_vehicle] = (0, 255, 0)
                    image[mask_arrow] = (0, 0, 255)
                    mask_vehicle_image[mask_vehicle] = (255)
                    special_vehicle_image[mask_vehicle] = (255)
                    special_vehicle_image[mask_arrow] = (255)
                    if sample['class'] == 'Crossbike' or sample['class'] == 'Construction' or sample[
                        'class'] == 'Walker':
                        bike_and_cons_vehicle_image[mask_vehicle] = (255)
                        bike_and_cons_vehicle_image[mask_arrow] = (255)

                    _, mask_arrow = self.draw_vehicle(center, sample['yaw'], sample['speed'],
                                                      (bbox[0], bbox[1]), bbox)
                    mask_vehicle_image[mask_arrow] = (255)

            elif sample['class'] == 'Radar':
                bbox = np.array(sample["extent"])

                mask_vehicle, mask_arrow = self.draw_vehicle(center, sample['yaw'], sample['speed'],
                                                             (bbox[0], bbox[1]), bbox)

                image[mask_vehicle] = (255, 0, 0)

            elif sample['class'] == 'lane_guidance':
                bbox = np.array(sample["extent"])
                position_center = np.array(sample['position'])
                position_center = position_center * 4 + 100

                mask_lane, mask_arrow_lane = self.draw_vehicle(position_center, sample['yaw'], 0.0,
                                                               (bbox[0], bbox[1]), bbox)

                lane_guidance_mask += mask_lane
                image[mask_lane] = (255, 255, 255)

            elif sample['class'] == "Lane":
                bbox = np.array(sample["extent"])
                #position_center = np.array(sample['position'])
                #position_center = position_center * 4 + 100

                mask_lane, mask_arrow_lane = self.draw_vehicle(center, sample['yaw'], 0.0,
                                                               (bbox[0], bbox[1]), bbox)

                lane_guidance_mask += mask_lane
                image[mask_lane] = (255, 255, 0)


            elif sample['class'] == 'tl_bev_pixel' or sample['class'] == 'Stop_sign':
                bbox = np.array(sample["extent"])

                mask = self.plot_bounding_box_center(center, bbox[0], bbox[1]).astype(np.bool)

                if sample['state'] == 2:
                    image[mask] = (0, 255, 0)
                    if sample['class'] == 'tl_bev_pixel':
                        tl_image = copy.deepcopy(mask)
                elif sample['state'] == 1:
                    image[mask] = (0, 255, 255)
                    if sample['class'] == 'tl_bev_pixel':
                        tl_image = copy.deepcopy(mask)
                elif sample['state'] == 0:
                    image[mask] = (0, 0, 255)

            elif sample['class'] == "Lane_guidance_wp":
                bbox = np.array(sample["extent"])

                mask = self.plot_bounding_box_center(center, bbox[0], bbox[1]).astype(np.bool)

                image[mask] = (255, 255, 255)

            if sample['class'] == 'tl_bev_pixel':

                bbox = np.array(sample["extent"])

                mask = self.plot_bounding_box_center(center, bbox[0], bbox[1]).astype(np.bool)
                if np.linalg.norm(np.array(sample['tl_bev_pixel_coordinate']).mean(0)) < 25:
                    if sample['state'] == 1:
                        tl_light_stop = True

                        image[mask] = (0, 255, 255)
                        if sample['class'] == 'tl_bev_pixel':
                            tl_image = copy.deepcopy(mask)
                    elif sample['state'] == 0:
                        tl_light_stop = True

                        image[mask] = (0, 0, 255)


        tl_image = tl_image * tl_ego_image * ego_front_mask

        return image, mask_vehicle_image,  special_vehicle_image, \
               bike_and_cons_vehicle_image, tl_image, ego_mask, lane_guidance_mask, tl_light_stop

    def plot_bounding_box_center(self, center, width=4, height=8):
        mask = np.zeros((200, 200)).astype(np.uint8)
        # Calculate the top-left corner from the center, width, and height
        top_left = (int(center[1] - width / 2), int(center[0] - height / 2))
        bottom_right = (int(center[1] + width / 2), int(center[0] + height / 2))

        # Draw the rectangle (bounding box)
        cv2.rectangle(mask, bottom_right, top_left, (255), 2)  # Blue box

        return mask

    @torch.no_grad()
    def plant_run_step(self, input_data, timestamp, ego_motion, sensors=None, keep_ids=None, light_hazard=False,
                       bbox=None, plant_boxes=None, tl_list=None, stop_box=None, lane_list=None, plant_input_dict=None):

        self.keep_ids = keep_ids

        # needed for traffic_light_hazard

        if not self.initialized:
            # assert 'hd_map' in input_data.keys()
            if ('hd_map' in input_data.keys()):
                print("*" * 50, "Plant is initialized")
                self._init(input_data['hd_map'])
            else:
                self.control = carla.VehicleControl()
                self.control.steer = 0.0
                self.control.throttle = 0.0
                self.control.brake = 1.0
                print("*" * 50, "not initialized")
                if self.exec_or_inter == 'inter':
                    return [], None
                return self.control, torch.zeros((1, 4, 2)), (0, 0), [], []

        # needed for traffic_light_hazard
        tick_data = self.tick(input_data)

        detected_input_image = np.zeros((200, 200, 3)).astype(np.uint8)

        #label_raw = self.super_class.get_bev_boxes_using_tl_lights(input_data=input_data, pos=tick_data['gps'])
        label_raw = self.super_class.get_bev_boxes(input_data=input_data, pos=tick_data['gps'])

        plant_input_image, gt_mask_vehicle_image, special_vehicle_image \
            , bike_and_cons_vehicle_image, tl_image, ego_mask, lane_guidance_mask, tl_light_stop = self.draw_label_raw(label_raw,
                                                                                                        'detection')

        self.control, pred_wp, keep_vehicle_ids, keep_vehicle_attn, plant_input_image = self.plant_get_control(label_raw, tick_data,
                                                                                            gt_mask_vehicle_image,
                                                                                            plant_input_image,
                                                                                            special_vehicle_image,
                                                                                            bike_and_cons_vehicle_image,
                                                                                            tl_image,
                                                                                            ego_mask,
                                                                                            lane_guidance_mask, lane_list,
                                                                                                               plant_input_dict)

        input_data.update({"tl_light_stop":tl_light_stop})

        return self.control, pred_wp, tick_data[
            'target_point'], keep_vehicle_ids, keep_vehicle_attn, plant_input_image, self.dummy_collision_label, self.pred_task_name


    def plant_get_control(self, label_raw, input_data, mask_vehicle_image, plant_input_image,
                        special_vehicle_image, bike_and_cons_vehicle_image, tl_image,
                          ego_mask, lane_guidance_mask, lane_list, plant_input_dict):
        gt_velocity = torch.FloatTensor([input_data['speed']]).unsqueeze(0)
        input_batch = self.get_input_batch(label_raw, input_data)
        x, y, _, tp, light = input_batch
        x[0], y[0], _, tp, light = x[0].cuda(2), y[0].cuda(2), _, tp.cuda(2), light.cuda(2)

        print("start_measurement")
        t0 = time.time()
        _, _, pred_wp, attn_map, new_other_output, _, input_cars_info, pred_task_name, self.dummy_collision_label, \
        other_vehicle_image, best_lane_reward = self.net(x, y, target_point=tp, light_hazard=light, gt_velocity=gt_velocity,
                                        data_lane=self.data_lane, mcts_data_route=self.mcts_data_route,
                                                         lane_list= lane_list, plant_input_dict=plant_input_dict)

        #_, _, pred_wp, attn_map, new_other_output, _, input_cars_info, pred_task_name
        pred_task_name = torch.argmax(pred_task_name)
        print("pred_task_name:", pred_task_name)
        self.pred_task_name = pred_task_name
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        print("plant_network time:", t1 - t0)


        self.draw_centerline(plant_input_image, label_raw, pred_wp)
        new_other_output = new_other_output.squeeze(0)

        steer, throttle, brake = self.net.model.control_pid(pred_wp[:1], gt_velocity)

        plant_input_image, mask_pred_wp, real_mask_wp = self.draw_pred_wp(plant_input_image,
                                                                          copy.deepcopy(pred_wp).squeeze(0))

        plant_input_image_gru = np.zeros((200, 200, 3)).astype(np.uint8)
        if new_other_output.size(1) != 0:
            new_other_output_list = []
            for gru_index in range(new_other_output.size(1)):
                new_other_output_list.append(new_other_output[:, gru_index].cpu())

            for gru_index, vehicle in enumerate(new_other_output_list):
                plant_input_image = self.draw_pred_wp_gru(plant_input_image,
                                  copy.deepcopy(vehicle), input_cars_info[gru_index])

        #cv2.imwrite("plant_input_image_gru.png", plant_input_image)



        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0

        if brake:
            steer *= self.steer_damping


        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        viz_trigger = ((self.step % 20 == 0) and self.cfg_viz)
        if viz_trigger and self.step > 2:
            create_BEV(label_raw, light, tp, pred_wp)

        exec_or_inter = 'inter'
        attention_score = 'AllLayer'  # self.cfg.attention_score
        topk = 1000  # elf.cfg.topk
        SAVE_GIF = True
        keep_vehicle_ids, keep_vehicle_attn = [], []

        plant_input_image = (plant_input_image, other_vehicle_image, best_lane_reward)

        return control, pred_wp, keep_vehicle_ids, keep_vehicle_attn, plant_input_image

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def get_input_batch(self, label_raw, input_data):
        sample = {'input': [], 'output': [], 'brake': [], 'waypoints': [], 'target_point': [],
                  'light': []}  # tugrul

        if self.cfg_agent.model.training.input_ego:
            data = label_raw
        else:
            data = label_raw[1:]  # remove first element (ego vehicle)

        """data_car = [[
            1., # type indicator for cars
            float(x['position'][0])-float(label_raw[0]['position'][0]),
            float(x['position'][1])-float(label_raw[0]['position'][1]),
            float(x['yaw'] * 180 / 3.14159265359), # in degrees
            float(x['speed'] * 3.6), # in km/h
            float(x['extent'][2]),
            float(x['extent'][1]),
            ] for x in data if x['class'] == 'Car'] """  # and ((self.cfg_agent.model.training.remove_back and float(x['position'][0])-float(label_raw[0]['position'][0]) >= 0) or not self.cfg_agent.model.training.remove_back)]

        data_car = []
        for j, x in enumerate(data):
            type_id = -1  # type_id != -1
            if x["class"] == "Car":
                type_id = 1.0
            elif x["class"] == "Police":
                type_id = 20.0

            elif x["class"] == "Ambulance":
                type_id = 21.0

            elif x["class"] == "Firetruck":
                type_id = 22.0

            elif x["class"] == "Crossbike":
                type_id = 23.0

            elif x["class"] == "Construction":
                type_id = 24.0

            elif x["class"] == "Walker":
                type_id = 25.0

            if type_id != -1:
                new_sample = [type_id,  # type indicator for cars
                              float(x['position'][0]) - float(label_raw[0]['position'][0]),
                              float(x['position'][1]) - float(label_raw[0]['position'][1]),
                              float(x['yaw'] * 180 / 3.14159265359),  # in degrees
                              float(x['speed'] * 3.6),  # in km/h
                              float(x['extent'][2]),
                              float(x['extent'][1]),
                              ]
                data_car.append(new_sample)

        # if we use the far_node as target waypoint we need the route as input
        data_route = [
            [
                2.,  # type indicator for route
                float(x['position'][0]) - float(label_raw[0]['position'][0]),
                float(x['position'][1]) - float(label_raw[0]['position'][1]),
                float(x['yaw'] * 180 / 3.14159265359),  # in degrees
                float(x['id']),
                float(x['extent'][2]),
                float(x['extent'][1]),
            ]
            for j, x in enumerate(data)
            if x['class'] == 'Route'
               and float(x['id']) < self.cfg_agent.model.training.max_NextRouteBBs]

        # we split route segment slonger than 10m into multiple segments
        # improves generalization
        data_route_split = []
        for route in data_route:
            if route[6] > 10:
                routes = split_large_BB(route, len(data_route_split))
                data_route_split.extend(routes)
            else:
                data_route_split.append(route)

        data_route = data_route_split[:self.cfg_agent.model.training.max_NextRouteBBs]

        assert len(data_route) <= self.cfg_agent.model.training.max_NextRouteBBs, 'Too many routes'

        if self.cfg_agent.model.training.get('remove_velocity', 'None') == 'input':
            for i in range(len(data_car)):
                data_car[i][4] = 0.

        if self.cfg_agent.model.training.get('route_only_wp', False) == True:
            for i in range(len(data_route)):
                data_route[i][3] = 0.
                data_route[i][-2] = 0.
                data_route[i][-1] = 0.

        # filter vehicle and route by attention scores
        # only keep entries which are in self.keep_ids
        if self.keep_ids is not None:
            data_car = [x for i, x in enumerate(data_car) if i in self.keep_ids]
            assert len(data_car) <= len(self.keep_ids), f'{len(data_car)} <= {len(self.keep_ids)}'

        # Radar info tugrul
        data_radar = []
        data_radar = [
            [
                10.0,  # type indicator for route
                float(x["position"][0]),
                float(x["position"][1]),
                float(x["yaw"] * 180 / 3.14159265359),  # in degrees
                float(x["id"]),
                float(x["extent"][2]),
                float(x["extent"][1]),
            ]
            for j, x in enumerate(data)
            if x["class"] == "Radar"
        ]

        data_lane = [
            [
                x
            ]
            for j, x in enumerate(data)
            if x["class"] == "Lane"
        ]

        features = data_car + data_radar + data_route

        sample['input'] = features  # np.array(features)

        # dummy data
        sample['output'] = features
        sample['light'] = self.traffic_light_hazard

        local_command_point = np.array([input_data['target_point'][0], input_data['target_point'][1]])
        sample['target_point'] = local_command_point

        batch = [sample]

        input_batch = generate_batch(batch)

        self.data = data
        self.data_car = data_car + data_radar
        self.data_route = data_route
        self.mcts_data_route = data_route_split

        self.data_lane = data_lane

        return input_batch

    def destroy(self):
        super().destroy()
        if self.scenario_logger:
            self.scenario_logger.dump_to_json()
            del self.scenario_logger

        if SAVE_GIF == True and (self.exec_or_inter == 'inter'):
            self.save_path_mask_vid = f'viz_vid/masked'
            self.save_path_org_vid = f'viz_vid/org'
            Path(self.save_path_mask_vid).mkdir(parents=True, exist_ok=True)
            Path(self.save_path_org_vid).mkdir(parents=True, exist_ok=True)
            out_name_mask = f"{self.save_path_mask_vid}/{self.route_index}.mp4"
            out_name_org = f"{self.save_path_org_vid}/{self.route_index}.mp4"
            cmd_mask = f"ffmpeg -r 25 -i {self.save_path_mask}/%d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {out_name_mask}"
            cmd_org = f"ffmpeg -r 25 -i {self.save_path_org}/%d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {out_name_org}"
            print(cmd_mask)
            os.system(cmd_mask)
            print(cmd_org)
            os.system(cmd_org)

            # delete the images
            os.system(f"rm -rf {Path(self.save_path_mask).parent}")

        del self.net

def create_BEV(labels_org, gt_traffic_light_hazard, target_point, pred_wp, pix_per_m=5):
    pred_wp = np.array(pred_wp.squeeze())
    s = 0
    max_d = 30
    size = int(max_d * pix_per_m * 2)
    origin = (size // 2, size // 2)
    PIXELS_PER_METER = pix_per_m

    # color = [(255, 0, 0), (0, 0, 255)]
    color = [(255), (255)]

    # create black image
    image_0 = Image.new('L', (size, size))
    image_1 = Image.new('L', (size, size))
    image_2 = Image.new('L', (size, size))
    vel_array = np.zeros((size, size))
    draw0 = ImageDraw.Draw(image_0)
    draw1 = ImageDraw.Draw(image_1)
    draw2 = ImageDraw.Draw(image_2)

    draws = [draw0, draw1, draw2]
    imgs = [image_0, image_1, image_2]

    for ix, sequence in enumerate([labels_org]):

        # features = rearrange(features, '(vehicle features) -> vehicle features', features=4)
        for ixx, vehicle in enumerate(sequence):
            # draw vehicle
            # if vehicle['class'] != 'Car':
            #     continue

            x = -vehicle['position'][1] * PIXELS_PER_METER + origin[1]
            y = -vehicle['position'][0] * PIXELS_PER_METER + origin[0]
            yaw = vehicle['yaw'] * 180 / 3.14159265359
            extent_x = vehicle['extent'][2] * PIXELS_PER_METER / 2
            extent_y = vehicle['extent'][1] * PIXELS_PER_METER / 2
            origin_v = (x, y)

            if vehicle['class'] == 'Car':
                p1, p2, p3, p4 = get_coords_BB(x, y, yaw - 90, extent_x, extent_y)
                if ixx == 0:
                    for ix in range(3):
                        draws[ix].polygon((p1, p2, p3, p4), outline=color[0])  # , fill=color[ix])
                    ix = 0
                else:
                    draws[ix].polygon((p1, p2, p3, p4), outline=color[ix])  # , fill=color[ix])

                if 'speed' in vehicle:
                    vel = vehicle['speed'] * 3  # /3.6 # in m/s # just for visu
                    endx1, endy1, endx2, endy2 = get_coords(x, y, yaw - 90, vel)
                    draws[ix].line((endx1, endy1, endx2, endy2), fill=color[ix], width=2)

            elif vehicle['class'] == 'Route':
                ix = 1
                image = np.array(imgs[ix])
                point = (int(x), int(y))
                cv2.circle(image, point, radius=3, color=color[0], thickness=3)
                imgs[ix] = Image.fromarray(image)

    for wp in pred_wp:
        x = wp[1] * PIXELS_PER_METER + origin[1]
        y = -wp[0] * PIXELS_PER_METER + origin[0]
        image = np.array(imgs[2])
        point = (int(x), int(y))
        cv2.circle(image, point, radius=2, color=255, thickness=2)
        imgs[2] = Image.fromarray(image)

    image = np.array(imgs[0])
    image1 = np.array(imgs[1])
    image2 = np.array(imgs[2])
    x = target_point[0][1] * PIXELS_PER_METER + origin[1]
    y = -(target_point[0][0]) * PIXELS_PER_METER + origin[0]
    point = (int(x), int(y))
    cv2.circle(image, point, radius=2, color=color[0], thickness=2)
    cv2.circle(image1, point, radius=2, color=color[0], thickness=2)
    cv2.circle(image2, point, radius=2, color=color[0], thickness=2)
    imgs[0] = Image.fromarray(image)
    imgs[1] = Image.fromarray(image1)
    imgs[2] = Image.fromarray(image2)

    images = [np.asarray(img) for img in imgs]
    image = np.stack([images[0], images[2], images[1]], axis=-1)
    BEV = image

    img_final = Image.fromarray(image.astype(np.uint8))
    if gt_traffic_light_hazard:
        color = 'red'
    else:
        color = 'green'
    img_final = ImageOps.expand(img_final, border=5, fill=color)

    ## add rgb image and lidar
    # all_images = np.concatenate((images_lidar, np.array(img_final)), axis=1)
    # all_images = np.concatenate((rgb_imag