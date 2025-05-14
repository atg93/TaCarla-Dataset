import copy

import cv2
import numpy as np
import carla
from gym import spaces
import cv2 as cv
from collections import deque
from pathlib import Path
import h5py

from leaderboard.autoagents.traditional_agents_files.utils.decision_of_overtaking import Decision_of_Overtaking
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.obs_manager import ObsManagerBase
from leaderboard.autoagents.traditional_agents_files.carla_gym.utils.traffic_light import TrafficLightHandler
from leaderboard.autoagents.traditional_agents_files.perception.detection_utils.geometry import scale_and_zoom, get_pose_matrix, euler_to_quaternion
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.agent_utils.coordinate_utils import normalize_angle
from leaderboard.autoagents.traditional_agents_files.utils.destroy_actor import Destroy_Actor
from leaderboard.autoagents.traditional_agents_files.utils.process_radar import Process_Radar
from leaderboard.utils.create_actors import Create_Actors

import torch

import os
import time
import math
from rdp import rdp

import pickle
from leaderboard.autoagents.traditional_agents_files.utils.lss_dataset import ext_data, intr_data
from PIL import Image

from leaderboard.autoagents.traditional_agents_files.perception.detection_utils.carla_to_lss import Carla_to_Lss_Converter

import sys
sys.path.append('/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/')
sys.path.append('/cta/users/tgorgulu/git/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/')

from tairvision_object_detection.models.bev.common.nuscenes.process import (ResizeCropRandomFlipNormalize, FilterClasses,
                                                           get_resizing_and_cropping_parameters)

from pyquaternion import Quaternion

COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)


def tint(color, factor):
    r, g, b = color
    r = int(r + (255-r) * factor)
    g = int(g + (255-g) * factor)
    b = int(b + (255-b) * factor)
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return (r, g, b)

from leaderboard.autoagents.traditional_agents_files.criteria import run_stop_sign
from leaderboard.autoagents.traditional_agents_files.criteria import encounter_light

class ObsManager(ObsManagerBase):
    def __init__(self, obs_configs):
        self._width = int(obs_configs['width_in_pixels'])
        self._pixels_ev_to_bottom = obs_configs['pixels_ev_to_bottom']
        self._pixels_per_meter = obs_configs['pixels_per_meter']
        self._history_idx = obs_configs['history_idx']
        self.plant__history_idx = [-16,-14,-12,-10,-8,-6,-4,-2]
        self._scale_bbox = obs_configs.get('scale_bbox', True)
        self._scale_mask_col = obs_configs.get('scale_mask_col', 1.1)

        self._history_queue = deque(maxlen=20)
        self.ego_queue = deque(maxlen=20)
        self.box_que = deque(maxlen=20)
        self.loc_que = deque(maxlen=20)
        self.transform_que = deque(maxlen=20)
        self.prev_info_que = deque(maxlen=20)
        self.info_transform_que = deque(maxlen=20)
        self.dummy_que = deque(maxlen=20)
        self.dummy_counter = 0

        self._image_channels = 3
        self._masks_channels = 3 + 3*len(self._history_idx)
        self._parent_actor = None
        self._world = None

        self._map_dir = Path(__file__).resolve().parent / 'maps'

        super(ObsManager, self).__init__()

        self.compute_prediction_input = None  # Compute_Prediction_Input(self._width, self._pixels_per_meter,self._pixels_ev_to_bottom, self)


        self._ego_motion_queue = deque(maxlen=3)
        self.sensor = None
        self.sensor_interface = None
        self.control_with = 'plant'
        self.route_plan = None
        self.prev_ev_loc = None
        self.decision_of_overtaking = Decision_of_Overtaking()
        self.process_radar = Process_Radar()
        self.color = "unknown"
        self.save_gt = True
        self.label_list = [carla.CityObjectLabel.Car, carla.CityObjectLabel.Bus, carla.CityObjectLabel.Bicycle, carla.CityObjectLabel.Motorcycle,  carla.CityObjectLabel.Truck, carla.CityObjectLabel.Rider]#carla.CityObjectLabel.Static, #carla.CityObjectLabel.Other,carla.CityObjectLabel.GuardRail,
        self.dynamic_object = [carla.CityObjectLabel.Dynamic]

        self.intrinsics = {}
        self.extrinsics = {}
        self.prev_scenario_instance_name = 'None'

        self.is_there_obstacle = False
        self.dynamic_mask = np.zeros((self._width,self._width))


        """self.cfg_perception = obs_configs['percetion_cfgs']
        self.augmentation_parameters = get_resizing_and_cropping_parameters(self.cfg_perception)
        self.transforms_val = ResizeCropRandomFlipNormalize(self.augmentation_parameters,
                                                            enable_random_transforms=False)

        self.carla_to_lss = Carla_to_Lss_Converter(False)
        self.counter = 0
        self.img_counter = 0

        self.variables = {"time_window":10, "cams":['front_left', 'front', 'front_right', 'back'], "detection_images_path":'/workspace/tg22/leaderboard_data/detection/image_file/'}

        self.view_array = self.create_view_array()
        self.pickle_path = '/workspace/tg22/leaderboard_data/detection/pickle_file'
        
        self.sensors()
        self.destroy_actor_for_data_collection = Destroy_Actor()
        self.start_destroying = False
        self.start_count = 0

        self.zero_velocity_count = 0"""




    def set_path(self, file_name_pickle, file_name_image_front):
        self.file_name_pickle = file_name_pickle
        self.file_name_image_front = file_name_image_front




    def get_counter_value(self):
        print("get_counter_value:", self.img_counter)
        return self.img_counter, self.new_run

    def create_view_array(self):
        arr = np.zeros((1, 1, 1, 4, 4), dtype=np.float32)
        arr[0, 0, 0] = [[0, -2, 0, 99.5],
                        [-2, 0, 0, 99.5],
                        [0, 0, 0.8, 4.3],
                        [0, 0, 0, 1]]
        return arr

    def _define_obs_space(self):
        self.obs_space = spaces.Dict(
            {'rendered': spaces.Box(
                low=0, high=255, shape=(self._width, self._width, self._image_channels),
                dtype=np.uint8),
             'masks': spaces.Box(
                low=0, high=255, shape=(self._masks_channels, self._width, self._width),
                dtype=np.uint8)})

    def attach_ego_vehicle(self, vehicle):
        self.vehicle = vehicle
        self._parent_actor = vehicle
        self._world = vehicle.get_world()

        self.create_actors = None #Create_Actors(self._world, self.vehicle)


        self.criteria_encounter_light = encounter_light.EncounterLight()
        self.criteria_stop = run_stop_sign.RunStopSign(self._world)

        maps_h5_path = ''.join(np.array(list(str(self._map_dir)))[53:]) + '/' + (self._world.get_map().name + '.h5')[11:]
        path_exist = os.path.exists(maps_h5_path)
        if not path_exist:
            maps_h5_path = 'leaderboard' + '/' + maps_h5_path

        print("maps_h5_path:",maps_h5_path)
        assert os.path.exists(maps_h5_path)
        with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
            self._world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
            self._pixels_per_meter = float(hf.attrs['pixels_per_meter'])
            assert np.isclose(self._pixels_per_meter, float(hf.attrs['pixels_per_meter']))
        self._distance_threshold = np.ceil(self._width / self._pixels_per_meter) * 2
        return self._pixels_per_meter, self._world_offset

    @staticmethod
    def _get_stops(criteria_stop, veh_loc):
        stop_sign = criteria_stop._target_stop_sign
        stops = []
        data_stops = []
        bb_list = []
        if (stop_sign is not None) and (not criteria_stop._stop_completed):
            bb_loc = carla.Location(stop_sign.trigger_volume.location)
            bb_ext = carla.Vector3D(stop_sign.trigger_volume.extent)
            bb_ext.x = max(bb_ext.x, bb_ext.y)
            bb_ext.y = max(bb_ext.x, bb_ext.y)
            trans = stop_sign.get_transform()
            stops = [(carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext)]

        if (stop_sign is not None):
            bb_loc = carla.Location(stop_sign.trigger_volume.location)
            bb_ext = carla.Vector3D(stop_sign.trigger_volume.extent)
            bb_ext.x = max(bb_ext.x, bb_ext.y)
            bb_ext.y = max(bb_ext.x, bb_ext.y)
            trans = stop_sign.get_transform()
            data_stops = [(carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext)]
        return stops, data_stops

    def update_route_plant(self,ev_loc, scenario_instance_name, M_warp, nonstop_prediction_label):
        #change_lane = self.check_route_to_change_lane(scenario_instance_name)
        #self.get_left_lane(self.is_there_obstacle, change_lane, scenario_instance_name, M_warp, nonstop_prediction_label)
        try:
            if self.route_plan[0][0].transform.location.distance(ev_loc)<3:
                self.route_plan = self.route_plan[1:]
                self.current_wp = self.route_plan[0]
                print("len(self.route_plan):",len(self.route_plan))
            else:
                self.current_wp = self.route_plan[0]
        except:
            pass

    def check_route_to_change_lane(self, scenario_instance_name):
        scenario_list = ['Accident','AccidentTwoWays','ConstructionObstacle','ConstructionObstacleTwoWays',
                         'HazardAtSideLane','HazardAtSideLaneTwoWays','ParkedObstacle','ParkedObstacleTwoWays',
                         'VehicleOpensDoorTwoWays']
        if self.prev_scenario_instance_name != scenario_instance_name:
            return scenario_instance_name[0] in scenario_list
        else:
            return False

    def get_left_lane(self, is_there_obs, change_lane, scenario_instance_name, M_warp, nonstop_prediction_label):
        try:
            left_wp = self.route_plan[0][0].get_left_lane()
        except:
            left_wp = None

        try:
            right_wp = self.route_plan[0][0].get_right_lane()
        except:
            right_wp = None

        new_wp = left_wp if type(left_wp) != type(None) and str(left_wp.lane_type) == 'Driving' else right_wp

        if type(new_wp) != type(None) and change_lane and is_there_obs and str(new_wp.lane_type) == 'Driving' and not nonstop_prediction_label:
            new_route = new_wp.next_until_lane_end(1.0)

            range_value = min(len(self.route_plan), 30, len(new_route))

            for index in range(range_value):
                try:
                    if index < 4:
                        self.route_plan[index] = new_route[index],'RoadOption.LANECHANGE'
                    else:
                        self.route_plan[index] = new_route[index],'RoadOption.LANEFOLLOW'

                except:
                    pass

            self.prev_scenario_instance_name = scenario_instance_name

    def get_previous_waypoints_in_junction(self, current_waypoint, next_wp):
        # Check if the waypoint is in a junction
        if next_wp == None:
            return []

        # Get previous waypoints
        previous_waypoints = []
        to_explore = [next_wp]
        explored = set()
        
        while to_explore:
            waypoint = to_explore.pop(0)
            if waypoint in explored:
                continue
            explored.add(waypoint)

            # Check each backward connection
            for wp in waypoint.previous(1.0):#len(current_waypoint.previous(1.0)) != 1
                #if wp.is_junction:
                if current_waypoint.transform.location.distance(wp.transform.location) < 6:
                    previous_waypoints.append(wp)
                    to_explore.append(wp)
            if len(previous_waypoints) >= 50:
                break

        return previous_waypoints

    def prediction_stop_label(self, M_warp, scenario_instance_name, next_index=10):
        current_wp = self.route_plan[0][0]
        current_wp_yaw = current_wp.transform.rotation.yaw
        next_route_plan = self.route_plan[0:50]

        next_wp = None
        next_wp_dist_list = []
        next_wp_list = []
        wp_index = 0
        for wp, high_command in next_route_plan:
            if (str(high_command) != 'RoadOption.LANEFOLLOW' and
                                   str(high_command) != 'RoadOption.STRAIGHT'):# and wp.is_junction:
                next_wp_list.append((wp,wp_index, high_command))
                next_wp_dist_list.append(current_wp.transform.location.distance(wp.transform.location))

            wp_index += 1

        if len(next_wp_list) != 0:
            next_wp, selected_wp_index, _ = next_wp_list[np.argmax(next_wp_dist_list)]
            next_route_plan = self.route_plan[selected_wp_index:50]
            wp_index = selected_wp_index
            for wp, high_command in next_route_plan:
                if (str(high_command) == 'RoadOption.LANEFOLLOW' or str(high_command) == 'RoadOption.STRAIGHT'):
                    next_wp = wp
                    break
                wp_index += 1

        prev_list = self.get_previous_waypoints_in_junction(current_wp, next_wp)

        new_prev_list = prev_list


        change_route_maks = np.zeros([self._width, self._width], dtype=np.uint8)
        if len(new_prev_list) != 0:
            change_route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                              for wp in new_prev_list])
            stop_route_warped = cv.transform(change_route_in_pixel, M_warp)
            cv.polylines(change_route_maks, [np.round(stop_route_warped).astype(np.int32)], False, 1, thickness=5)
        change_route_maks = change_route_maks.astype(np.bool)
        dynamic_mask = self.dynamic_mask

        prediction_stoplabel = np.sum(change_route_maks * dynamic_mask) > 0
        try:
            prediction_stoplabel_mask = change_route_maks.astype(np.uint8) * 255+ dynamic_mask.astype(np.bool).astype(np.uint8) * 255
        except:
            prediction_stoplabel_mask = change_route_maks.astype(np.uint8) * 255+ dynamic_mask.astype(np.bool).astype(np.uint8) * 255

        nonstop_prediction_label = False
        if scenario_instance_name == 'None_1':
            prediction_stoplabel
        else:
            nonstop_prediction_label = copy.deepcopy(prediction_stoplabel)
            prediction_stoplabel = False

        return prediction_stoplabel, prediction_stoplabel_mask, nonstop_prediction_label



    def get_ego(self, ev_transform):
        ego = {}
        ego['rotation'] = euler_to_quaternion(-ev_transform.rotation.roll, ev_transform.rotation.pitch,
                                              -ev_transform.rotation.yaw)
        ego['translation'] = ev_transform.location.x, -ev_transform.location.y, ev_transform.location.z
        return ego

    def get_plant_ego(self, ev_transform, prev_ev_transform):
        ego = {}
        ego['rotation'] = euler_to_quaternion(-ev_transform.rotation.roll, ev_transform.rotation.pitch,
                                              -ev_transform.rotation.yaw)
        ego['translation'] = ev_transform.location.x, -ev_transform.location.y, ev_transform.location.z
        dx, dy = ev_transform.location.x, -ev_transform.location.y
        ego_motion = dx, dy, 0, 0, yaw, 0
        return ego

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def get_relative_transform(self, ego_matrix, vehicle_matrix):
        """
        return the relative transform from ego_pose to vehicle pose
        """
        relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
        rot = ego_matrix[:3, :3].T
        relative_pos = rot @ relative_pos

        # transform to right handed system
        relative_pos[1] = - relative_pos[1]

        # transform relative pos to virtual lidar system
        rot = np.eye(3)
        trans = - np.array([1.3, 0.0, 2.5])
        relative_pos = rot @ relative_pos + trans

        return relative_pos

    def get_nearby_object(self, vehicle_position, actor_list, radius):
        nearby_objects = []
        for actor in actor_list:
            trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
            trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x, y=trigger_box_global_pos.y, z=trigger_box_global_pos.z)
            if (trigger_box_global_pos.distance(vehicle_position) < radius):
                nearby_objects.append(actor)
        return nearby_objects
    
    def get_high_level_action(self,input_data,pixel_per_meter=4):
        waypoint_route = self.route_plan
        results = []
        if waypoint_route != None and 'bbox' in input_data.keys():

            self.max_actor_distance = 50.0  # copy from expert
            self.max_light_distance = 15.0  # copy from expert
            self.max_route_distance = 30.0
            self.max_map_element_distance = 30.0
            self.DATAGEN = 1
            self.map_precision = 10.0  # meters per point
            self.rdp_epsilon = 0.5

            ego_location = self.vehicle.get_location()



            pos = np.array([ego_location.x, ego_location.y])
            # pos = self._route_planner.convert_gps_to_carla(pos)

            ego_transform = self.vehicle.get_transform()
            ego_control = self.vehicle.get_control()
            ego_velocity = self.vehicle.get_velocity()
            ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity)  # In m/s
            ego_brake = ego_control.brake
            ego_rotation = ego_transform.rotation
            ego_matrix = np.array(ego_transform.get_matrix())
            ego_extent = self.vehicle.bounding_box.extent
            ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z]) * 2.
            ego_yaw = (ego_rotation.yaw / 180 * np.pi)
            relative_yaw = 0
            relative_pos = self.get_relative_transform(ego_matrix, ego_matrix)

            results = []

            # add ego-vehicle to results list
            # the format is category, extent*3, position*3, yaw, points_in_bbox, distance, id
            # the position is in lidar coordinates
            ego_pixel_loc = np.array([95.5,95.5]).astype(np.float32)
            result = {"class": "Ego",
                      "yaw": int(math.degrees(relative_yaw)%360),
                      "num_points": -1,
                      "distance": -1,
                      "speed": int(ego_speed),
                      "id": int(self.vehicle.id),
                      }
            results.append(result)

            # -----------------------------------------------------------
            # Other vehicles
            # -----------------------------------------------------------

            _actors = self._world.get_actors()

            tlights = _actors.filter('*traffic_light*')
            all_vehicles = _actors.filter('*vehicle*')
            all_walkers = _actors.filter('*pedestrian*')
            objects_list = [all_vehicles, all_walkers]
            name_list = ['Car']
            name = name_list[0]
            for fake_id, corners in enumerate(input_data['bbox']):
                center_of_vec = corners.mean(0)
                relative_vec_pos = center_of_vec - ego_pixel_loc
                distance = math.sqrt(relative_vec_pos[0]**2 + relative_vec_pos[1]**2) / pixel_per_meter
                angle = math.atan2(relative_vec_pos[0], relative_vec_pos[1])
                # Convert the angle from radians to degrees if necessary
                angle_degrees = math.degrees(angle)
                vehicle_speed = input_data['predicted_speed'][fake_id] #get_high_level_action
                color = input_data['color_name_list'][fake_id]

                #distance = np.linalg.norm(relative_pos)
                angle_degrees = (180 - (angle_degrees % 360)) % 360
                if angle_degrees >= 180:
                    angle_degrees = angle_degrees - 360


                result = {
                    "class": name,
                    "yaw": int(angle_degrees),
                    "distance": int(distance),
                    "speed": int(vehicle_speed),
                    "id": str(fake_id),
                    "color": str(color)
                }
                results.append(result)

            _traffic_lights = self.get_nearby_object(ego_location, tlights, self.max_light_distance)
            # _stop_signs = self._actors.filter('*stop*')
            # self.get_nearby_object(ego_location, _stop_signs, self.max_light_distance)
            # print("stop_signs:",_stop_signs[0])
            for light in _traffic_lights:
                if (light.state == carla.libcarla.TrafficLightState.Red):
                    state = "Red"
                elif (light.state == carla.libcarla.TrafficLightState.Yellow):
                    state = "Yellow"
                elif (light.state == carla.libcarla.TrafficLightState.Green):
                    state = "Green"
                else:  # unknown
                    state = "Unknown"

                center_bounding_box = light.get_transform().transform(light.trigger_volume.location)
                center_bounding_box = carla.Location(center_bounding_box.x, center_bounding_box.y, center_bounding_box.z)
                length_bounding_box = carla.Vector3D(light.trigger_volume.extent.x, light.trigger_volume.extent.y,
                                                     light.trigger_volume.extent.z)
                transform = carla.Transform(
                    center_bounding_box)  # can only create a bounding box from a transform.location, not from a location
                bounding_box = carla.BoundingBox(transform.location, length_bounding_box)

                gloabl_rot = light.get_transform().rotation
                bounding_box.rotation = carla.Rotation(pitch=light.trigger_volume.rotation.pitch + gloabl_rot.pitch,
                                                       yaw=light.trigger_volume.rotation.yaw + gloabl_rot.yaw,
                                                       roll=light.trigger_volume.rotation.roll + gloabl_rot.roll)

                light_rotation = transform.rotation
                light_matrix = np.array(transform.get_matrix())

                light_extent = bounding_box.extent
                dx = np.array([light_extent.x, light_extent.y, light_extent.z]) * 2.
                yaw = light_rotation.yaw / 180 * np.pi

                relative_yaw = normalize_angle(yaw - ego_yaw)
                relative_pos = self.get_relative_transform(ego_matrix, light_matrix)

                distance = np.linalg.norm(relative_pos)
                if self.color != "unknown":
                    result = {
                        "class": "Traffic light",
                        "yaw": int(math.degrees(relative_yaw)),
                        "distance": int(distance),
                        "state": state,
                        "id": int(light.id),
                    }
                    results.append(result)
            waypoint_route = np.array([[node[0].transform.location.x, node[0].transform.location.y] for node in waypoint_route])
            max_len = 1
            if len(waypoint_route) < max_len:
                max_len = len(waypoint_route)
            """shortened_route = rdp(waypoint_route[:max_len], epsilon=self.rdp_epsilon)

            # convert points to vectors
            vectors = shortened_route[1:] - shortened_route[:-1]
            midpoints = shortened_route[:-1] + vectors / 2."""

            for i, midpoint in enumerate(waypoint_route):
                if i >= max_len:
                    break
                vector = midpoint-[ego_location.x,ego_location.y]
                # find distance to center of waypoint
                distance = np.linalg.norm(vector, axis=0)
                angles = np.arctan2(vector[1], vector[0])
                relative_yaw = normalize_angle(angles - ego_yaw)

                # visualize subsampled route
                # self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1,
                #                             color=carla.Color(0, 255, 255, 255), life_time=(10.0/self.frame_rate_sim))

                result = {
                    "class": "Route",
                    "yaw": int(math.degrees(relative_yaw)),
                    "distance": int(distance),
                    "id": i,
                }
                results.append(result)

            map = self._world.get_map()
            closest_waypoint = map.get_waypoint(ego_location, project_to_road=True,
                                                lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
            left_lane = False
            right_lane = False
            if closest_waypoint.is_intersection == False and not isinstance(closest_waypoint.get_left_lane(),type(None)) and closest_waypoint.get_left_lane().lane_type == carla.LaneType.Driving:
                left_lane = True
            if closest_waypoint.is_intersection == False and not isinstance(closest_waypoint.get_right_lane(),type(None)) and closest_waypoint.get_right_lane().lane_type == carla.LaneType.Driving:
                right_lane = True
            result = {
                "class": "Lane",
                "right_lane": right_lane,
                "left_lane": left_lane,
            }
            results.append(result)

            _, _, _, _, _, obstacle = self.process_radar.show_radar_output(input_data['front_radar'], compass=None)
            result = {
                "class": "Radar",
                "obstacle": obstacle,
            }
            results.append(result)

        return results

    def _number_and_time(self,actor_list,name):
        start_time = time.time()
        vehicle_actors = [actor for actor in actor_list if actor.type_id.startswith(name)]
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(name + "elapsed_time:", elapsed_time)
        print("len(vehicle_actors):, ",len(vehicle_actors))

    def get_observation(self, input_data, route_plan, current_speed,  detected_masks=None, detected_bbox=None, unwbbox=None, scenario_instance_name=None):
        compass = input_data['imu'][1][-1]
        if self.route_plan == None:
            self.route_plan = route_plan
        else:
            route_plan = self.route_plan

        ev_transform = self.vehicle.get_transform()#.get_light_state()
        ego = self.get_ego(ev_transform)
        self.ego_queue.append(ego)

        #blueprint_library = self._world.get_blueprint_library()
        #vehicle_blueprint = random.choice(blueprint_library.filter('vehicle.*'))

        ev_loc = ev_transform.location
        ev_rot = ev_transform.rotation
        ev_bbox = self.vehicle.bounding_box
        snap_shot = self._world.get_snapshot()
        start_time = time.time()
        #_actors = self._world.get_actors()
        end_time = time.time()

        elapsed_time = end_time - start_time
        print("elapsed_time:", elapsed_time)

        current_speed = self.vehicle.get_velocity().length()
        if self.prev_ev_loc==None:
            self.prev_ev_loc = ev_loc
            self.prev_ev_transform = ev_transform
            self.prev_bb_info = (ev_transform, ev_bbox.location, ev_bbox.extent)
            self.prev_compass = compass
            self.previous_speed = current_speed
            self.previous_yaw = compass

        def is_within_distance(w, scale=1):
            c_distance = abs(ev_loc.x - w.location.x) < (self._distance_threshold * scale) \
                and abs(ev_loc.y - w.location.y) < (self._distance_threshold * scale) \
                and abs(ev_loc.z - w.location.z) < 8.0
            c_ev = abs(ev_loc.x - w.location.x) < 1.0 and abs(ev_loc.y - w.location.y) < 1.0
            return c_distance and (not c_ev)

        actor_list = self._world.get_actors()
        vehicle_actors = [actor for actor in actor_list if actor.type_id.startswith('vehicle.')]
        walker_actors = [actor for actor in actor_list if actor.type_id.startswith('walker.')]


        constructioncone_list = []
        for actor in actor_list:
            if actor.type_id.startswith('static.') and (not actor.type_id.startswith('vehicle.') or not actor.type_id.startswith('sensor.') or not actor.type_id.startswith('walker.') or not actor.type_id.startswith('traffic.')):
                constructioncone_list.append(actor)


        actor_vehicle_bb = []
        walker_bb = []
        for actor in vehicle_actors:
            if actor.type_id == 'static.prop.constructioncone':
                asd = 0

            #if 'police' in actor.type_id:
            bounding_box = actor.bounding_box

            # The bounding box is relative to the vehicle, get the vehicle's location to place the bounding box in the world
            vehicle_location = actor.get_location()
            bounding_box_location = carla.Location(
                bounding_box.location.x + vehicle_location.x,
                bounding_box.location.y + vehicle_location.y,
                bounding_box.location.z + vehicle_location.z
            )
            bounding_box.location = bounding_box_location

            if actor.type_id.startswith('vehicle.') or actor.type_id == 'static.prop.constructioncone':
                # Check if the vehicle is a police car
                actor_vehicle_bb.append((actor, bounding_box, actor.type_id))

            elif actor.type_id.startswith('walker.') or actor.type_id.startswith('pedestrian.'):
                walker_bb.append((actor, bounding_box, actor.type_id))

        vehicle_bbox_list = []
        for new_label in self.label_list:
            vehicle_bbox_list += self._world.get_level_bbs(new_label) #carla.CityObjectLabel.Train

        dynamic_bbox_list = []
        for new_label in self.dynamic_object: #carla.CityObjectLabel self.dynamic_object
            dynamic_bbox_list += self._world.get_level_bbs(new_label)

        police_bbox_list = []
        ambulance_bbox_list = []
        fire_truck_bbox_list = []
        bicycle_bbox_list = []
        all_actor_vehicle_bbox = []
        speed_vehicle_bbox_list = []
        final_vehicle_bbox_list = []
        actors_to_destroy = []
        for vec in vehicle_bbox_list:
            type_list = []
            type_id = []
            actor_list = []
            speed_list = []
            for actor_class, actor, id in actor_vehicle_bb:
                type_list.append(vec.location.distance(actor.location))
                type_id.append(id)
                actor_list.append(actor_class)
                speed_list.append(actor_class.get_velocity().length())

            if type_list[np.argmin(type_list)] < 5.0:
                asd = 0
            else:
                asd = 0 #assert False

            current_type = type_id[np.argmin(type_list)].replace('.', ' ').replace('_', ' ').split()

            if 'police' in current_type:
                police_bbox_list.append((vec, actor_list[np.argmin(type_list)]))
                #vehicle_bbox_list.remove(vec)

            elif 'ambulance' in current_type:
                ambulance_bbox_list.append((vec, actor_list[np.argmin(type_list)]))
                #vehicle_bbox_list.remove(vec)

            elif 'firetruck' in current_type:
                fire_truck_bbox_list.append((vec, actor_list[np.argmin(type_list)]))
                #vehicle_bbox_list.remove(vec)

            elif 'crossbike' in current_type:
                bicycle_bbox_list.append((vec, actor_list[np.argmin(type_list)]))
                #vehicle_bbox_list.remove(vec)

            else:
                final_vehicle_bbox_list.append((vec, actor_list[np.argmin(type_list)]))
                #if self.vehicle.id != actor_list[np.argmin(type_list)].id:
                #    actors_to_destroy.append(actor_list[np.argmin(type_list)])

            if type(self.create_actors) != type(None):
                self.create_actors.set_actor_settings()




            all_actor_vehicle_bbox.append((actor_list[np.argmin(type_list)],vec,actor_list[np.argmin(type_list)].id))

        total_nex_bbox = len(police_bbox_list) + len(ambulance_bbox_list) + len(fire_truck_bbox_list) + len(bicycle_bbox_list) + len(
            final_vehicle_bbox_list)
        assert total_nex_bbox == len(vehicle_bbox_list)

        final_dynamic_bbox_list = []
        for vec in dynamic_bbox_list:
            type_list = []
            type_id = []
            actor_list = []
            for actor_class in constructioncone_list:
                type_list.append(vec.location.distance(actor_class.get_transform().location))
                type_id.append(actor_class.type_id)
                actor_list.append(actor_class)
            if len(type_list) != 0:
                id = type_id[np.argmin(type_list)]#.replace('.', ' ').replace('_', ' ').split()
                if id == 'static.prop.constructioncone' or id == 'static.prop.trafficwarning' or id == 'static.prop.warningconstruction' or id == 'static.prop.dirtdebris02':
                    final_dynamic_bbox_list.append((vec, actor_list[np.argmin(type_list)]))


        walker_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Pedestrians)

        final_walker_bbox_list = []
        for vec in walker_bbox_list:
            type_list = []
            type_id = []
            actor_list = []
            for actor_class in walker_actors:
                type_list.append(vec.location.distance(actor_class.get_transform().location))
                type_id.append(actor_class.type_id)
                actor_list.append(actor_class)
            if len(type_id) != 0:
                id = type_id[np.argmin(type_list)]#.replace('.', ' ').replace('_', ' ').split()
                if 'walker' in id.split('.'):
                    final_walker_bbox_list.append((vec, actor_list[np.argmin(type_list)]))

        if self._scale_bbox:#f_and_l_coeff
            vehicles, vehicles_actors = self._get_surrounding_actors(final_vehicle_bbox_list, is_within_distance, 1.0)
            dynamics_objects, dynamics_objects_actors = self._get_surrounding_actors(final_dynamic_bbox_list, is_within_distance, 1.0)
            police_objects, police_objects_actors = self._get_surrounding_actors(police_bbox_list, is_within_distance, 1.0)
            ambulance_objects, ambulance_objects_actors = self._get_surrounding_actors(ambulance_bbox_list, is_within_distance, 1.0)
            fire_truck_objects, fire_truck_objects_actors = self._get_surrounding_actors(fire_truck_bbox_list, is_within_distance, 1.0)
            bicycle_objects, bicycle_objects_actors = self._get_surrounding_actors(bicycle_bbox_list, is_within_distance, 1.0)
            walkers, walkers_actors = self._get_surrounding_actors(final_walker_bbox_list, is_within_distance, 2.0)
        else:
            vehicles, vehicles_actors = self._get_surrounding_actors(final_vehicle_bbox_list, is_within_distance)
            dynamics_objects, dynamics_objects_actors = self._get_surrounding_actors(final_dynamic_bbox_list, is_within_distance)
            police_objects, police_objects_actors = self._get_surrounding_actors(police_bbox_list, is_within_distance)
            ambulance_objects, ambulance_objects_actors = self._get_surrounding_actors(ambulance_bbox_list, is_within_distance)
            fire_truck_objects, fire_truck_objects_actors = self._get_surrounding_actors(fire_truck_bbox_list, is_within_distance)
            bicycle_objects, bicycle_objects_actors = self._get_surrounding_actors(bicycle_bbox_list, is_within_distance)
            walkers, walkers_actors = self._get_surrounding_actors(final_walker_bbox_list, is_within_distance)



        vehicle_lss_boxes, ego_boxes = self.get_surrounding_objects_vectors(vehicles, vehicles_actors, ev_loc, ev_rot, 1)
        walker_lss_boxes, ego_boxes = self.get_surrounding_objects_vectors(walkers, walkers_actors, ev_loc, ev_rot, 2)
        dynamics_objects_lss_boxes, ego_boxes = self.get_surrounding_objects_vectors(dynamics_objects, dynamics_objects_actors, ev_loc, ev_rot, 3)
        police_lss_boxes, ego_boxes = self.get_surrounding_objects_vectors(police_objects, police_objects_actors, ev_loc, ev_rot, 4)
        ambulance_lss_boxes, ego_boxes = self.get_surrounding_objects_vectors(ambulance_objects, ambulance_objects_actors, ev_loc, ev_rot, 5)
        fire_truck_lss_boxes, ego_boxes = self.get_surrounding_objects_vectors(fire_truck_objects, fire_truck_objects_actors, ev_loc, ev_rot, 6)
        bicycle_lss_boxes, ego_boxes = self.get_surrounding_objects_vectors(bicycle_objects, bicycle_objects_actors, ev_loc, ev_rot, 7)



        tl_green, green_list = TrafficLightHandler.get_stopline_vtx(ev_loc, 0)
        tl_yellow, yel_list = TrafficLightHandler.get_stopline_vtx(ev_loc, 1)
        tl_red, red_list = TrafficLightHandler.get_stopline_vtx(ev_loc, 2)
        stops, data_stops = self._get_stops(self.criteria_stop, ev_loc)
        self.tl_corner_bb_list = green_list + yel_list + red_list
        #self.stops = data_stops

        vehicles += (ambulance_objects + fire_truck_objects + police_objects + bicycle_objects)

        self._history_queue.append((vehicles, walkers, tl_green, tl_yellow, tl_red, stops, data_stops, dynamics_objects))

        tl_data_M_warp = self._get_warp_transform(ev_loc, ev_rot, _pixels_ev_to_bottom=400, _width=800)
        M_warp = self._get_warp_transform(ev_loc, ev_rot)

        prediction_stop_label, prediction_stoplabel_mask, nonstop_prediction_label = self.prediction_stop_label(M_warp, scenario_instance_name)

        self.update_route_plant(ev_loc, scenario_instance_name.split('_'), M_warp, nonstop_prediction_label)


        # objects with history
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks, data_stops_masks, dynamics_objects_masks, vec_list, tl_masks_list \
            = self._get_history_masks(M_warp)

        road_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        lane_mask_all = np.zeros([self._width, self._width], dtype=np.uint8)
        lane_mask_broken = np.zeros([self._width, self._width], dtype=np.uint8)

        # ev_mask
        ev_mask, current_pixel_loc = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location, ev_bbox.extent)], M_warp)


        self.info_transform_que.append((ev_transform, ev_bbox.location, ev_bbox.extent))

        prev_bbox_list, prev_bb_info_list = self.get_item_bbox(detected_bbox, unwbbox)

        prev_pixel_loc_list = []
        for prev_bb_info in prev_bb_info_list:
            _, prev_pixel_loc = self._get_mask_from_actor_list([prev_bb_info], M_warp)
            prev_pixel_loc_list.append(prev_pixel_loc)

        ev_mask_col, _ = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location,
                                                       ev_bbox.extent*self._scale_mask_col)], M_warp)

        # route_mask
        original_route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                   for wp, _ in route_plan[0:80]])
        route_warped = cv.transform(route_in_pixel, M_warp)
        cv.polylines(original_route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)

        fake_tl_masks = tl_green_masks[-1].astype(np.uint8) + tl_yellow_masks[-1].astype(np.uint8) + tl_red_masks[-1].astype(np.uint8)

        #img, close_points_count, mean_alt, mean_vel, is_there_obstacle, route_plan, plant_global_plan_gps, plant_global_plan_world_coord = self.decision_of_overtaking(input_data,fake_tl_masks=fake_tl_masks,ev_mask=ev_mask,original_route_mask=original_route_mask,wp_list=route_plan, global_plan_gps=plant_global_plan_gps, world_coordinate=plant_global_plan_world_coord, high_level_action=high_level_action)
        # route_mask

        _, _, _, _, _, _, _, _, _, tl_masks_list = self._get_history_masks(tl_data_M_warp)
        tl_data_route_mask = np.zeros([800, 800], dtype=np.uint8)
        try:
            route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                       for wp, _ in route_plan[0:320]])
        except:
            route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                       for wp, _ in route_plan[0:-1]])
        route_warped = cv.transform(route_in_pixel, tl_data_M_warp)
        cv.polylines(tl_data_route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=2)#tl_data
        tl_data_route_mask = tl_data_route_mask.astype(np.bool)
        tl_data_route_mask = tl_data_route_mask.astype(np.uint8) * 255

        route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                   for wp, _ in route_plan[0:80]])
        route_warped = cv.transform(route_in_pixel, M_warp)
        cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)
        route_mask = route_mask.astype(np.bool)


        tl_route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        tl_route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                   for wp, _ in route_plan[0:40]])
        tl_route_warped = cv.transform(tl_route_in_pixel, M_warp)
        cv.polylines(tl_route_mask, [np.round(tl_route_warped).astype(np.int32)], False, 1, thickness=16)
        self.tl_route_masks = tl_route_mask.astype(np.bool)

        stop_route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        tl_route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                      for wp, _ in route_plan[0:20]])
        tl_route_warped = cv.transform(tl_route_in_pixel, M_warp)
        cv.polylines(stop_route_mask, [np.round(tl_route_warped).astype(np.int32)], False, 1, thickness=1)
        self.stop_route_mask = stop_route_mask.astype(np.bool)

        stop_obj_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        stop_route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                      for wp, _ in route_plan[0:4]])
        stop_route_warped = cv.transform(stop_route_in_pixel, M_warp)
        cv.polylines(stop_obj_mask, [np.round(stop_route_warped).astype(np.int32)], False, 1, thickness=5)
        stop_route_mask_1 = stop_obj_mask.astype(np.bool)

        change_route_maks = np.zeros([self._width, self._width], dtype=np.uint8)
        change_route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                        for wp, _ in route_plan[0:10]])
        stop_route_warped = cv.transform(change_route_in_pixel, M_warp)
        cv.polylines(change_route_maks, [np.round(stop_route_warped).astype(np.int32)], False, 1, thickness=5)
        change_route_maks = change_route_maks.astype(np.bool)

        planning_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        #route plannig
        control = None
        targetpoint_mask = None
        light_hazard = None
        attention_mask = np.zeros([self._width, self._width, 3], dtype=np.uint8)
        keep_vehicle_ids = []
        keep_vehicle_attn = []
        t1 = time.time()
        if type(self.sensor_interface) != type(None):
            pass

        #t2 = time.time()
        #print("t2-t1:", t2 - t1)
        # render
        image = np.zeros([self._width, self._width, 3], dtype=np.uint8)
        image[road_mask] = COLOR_ALUMINIUM_5
        image[stop_route_mask_1] = COLOR_ALUMINIUM_3
        if type(targetpoint_mask) != type(None):
            image[targetpoint_mask] = COLOR_RED

        image[lane_mask_all] = COLOR_MAGENTA
        image[lane_mask_broken] = COLOR_MAGENTA_2

        h_len = len(self._history_idx)-1
        for i, mask in enumerate(stop_masks):
            image[mask] = tint(COLOR_YELLOW_2, (h_len-i)*0.2)
        for i, mask in enumerate(tl_green_masks):
            image[mask] = tint(COLOR_GREEN, (h_len-i)*0.2)
        for i, mask in enumerate(tl_yellow_masks):
            image[mask] = tint(COLOR_YELLOW, (h_len-i)*0.2)
        for i, mask in enumerate(tl_red_masks):
            image[mask] = tint(COLOR_RED, (h_len-i)*0.2)


        self.set_is_there_list_parameter(data_stops_masks, tl_green_masks, tl_yellow_masks, tl_red_masks)

        image = self.draw_attention_bb(image, keep_vehicle_ids, keep_vehicle_attn, M_warp, vehicle_masks)
        for i, mask in enumerate(vehicle_masks):
            image[mask] = tint(COLOR_BLUE, (h_len - i) * 0.2)
        for i, mask in enumerate(dynamics_objects_masks):
            image[mask] = tint(COLOR_BLUE, (h_len - i) * 0.2)
        for i, mask in enumerate(walker_masks):
            image[mask] = tint(COLOR_CYAN, (h_len - i) * 0.2)

        modified_env_mask = self.get_modified_env_mask()
        image[modified_env_mask] = COLOR_WHITE

        if self.control_with == 'plant':
            image[planning_mask] = COLOR_BLUE

        # image[obstacle_mask] = COLOR_BLUE

        # masks
        c_road = road_mask * 255
        c_route = route_mask * 255
        c_lane = lane_mask_all * 255
        c_lane[lane_mask_broken] = 120

        # masks with history
        c_tl_history = []
        for i in range(len(self._history_idx)):
            c_tl = np.zeros([self._width, self._width], dtype=np.uint8)
            c_tl[tl_green_masks[i]] = 80
            c_tl[tl_yellow_masks[i]] = 170
            c_tl[tl_red_masks[i]] = 255
            c_tl[stop_masks[i]] = 255
            c_tl_history.append(c_tl)

        c_vehicle_history = [m*255 for m in vehicle_masks]
        c_walker_history = [m*255 for m in walker_masks]

        masks = np.stack((c_route, *c_vehicle_history, *c_walker_history, *c_tl_history), axis=2)
        masks = np.transpose(masks, [2, 0, 1])
        ev_loc_in_px = self._world_to_pixel(ev_loc)
        loc_list, transform_list = self.get_prev_loc_and_transform(np.array([ev_loc.x, ev_loc.y]), ev_transform)#self.get_prev_loc_and_transform(ev_loc_in_px, ev_transform)
        prev_ev_loc_in_px = self._world_to_pixel(self.prev_ev_loc)

        plant_motion = [(loc_list[0] - loc_list[1])[0], (loc_list[0] - loc_list[1])[1], 0, 0, (transform_list[0].rotation.yaw-transform_list[1].rotation.yaw), 0]#dx, dy, _, _, yaw, _
        obs_dict = {'rendered': image, 'masks': masks, 'plant_control': control,'light_hazard':light_hazard,'ego_motion':plant_motion}

        M_history_warp_list = []
        for prev_pixel_loc in prev_pixel_loc_list:
            M_history_warp_list.append(cv.getAffineTransform(current_pixel_loc[0].squeeze(1)[:-1],prev_pixel_loc[0].squeeze(1)[:-1])) #self._get_warp_transform(transform_list[1].location,transform_list[1].rotation)
        interval_time = self._world.get_settings().fixed_delta_seconds

        current_loc = np.array([100,100])
        previous_loc = []
        #self.gt_ = prev_pixel_loc[0].squeeze(1)[:-1]

        for dum_cor in current_pixel_loc[0].squeeze(1)[:-1]:
            prev_ego_loc = self.get_detection_warp(compass, interval_time, dum_cor, current_speed)
            previous_loc.append(prev_ego_loc)
        previous_loc = np.stack(previous_loc).astype(np.float32)
        fake_M_history_warp = cv.getAffineTransform(current_pixel_loc[0].squeeze(1)[:-1], previous_loc) #self._get_warp_transform(transform_list[1].location,transform_list[1].rotation)
        self.dead_reckoning = previous_loc

        #fake_M_history_warp = M_history_warp

        self.prev_ev_loc = ev_loc
        self.prev_ev_transform = ev_transform
        self.prev_bb_info = (ev_transform, ev_bbox.location, ev_bbox.extent)
        #prev_bbox = self.box_que[0][1]

        detection_boxes = vehicle_lss_boxes + walker_lss_boxes + dynamics_objects_lss_boxes + police_lss_boxes + ambulance_lss_boxes + fire_truck_lss_boxes + bicycle_lss_boxes

        stop_obj_mask[stop_obj_mask > 0] = 255
        dynamics_objects_m = np.zeros([self._width, self._width], dtype=np.uint8)
        dynamics_objects_m = (vehicle_masks[-1]).astype(np.uint8) # + dynamics_objects_masks[-1] + walker_masks[-1]).astype(np.uint8)
        dynamics_objects_m[dynamics_objects_m > 0] = 255
        #cv2.imwrite('dynamics_objects_m.png', dynamics_objects_m)
        #cv2.imwrite('stop_obj_mask.png', stop_obj_mask)
        self.dynamic_mask = vehicle_masks[-1] + dynamics_objects_masks[-1]
        self.is_there_obstacle = np.sum(change_route_maks * (self.dynamic_mask)) > 0

        number_of_walker, walker_stop = self.get_walker_stop(ev_loc, walkers_actors, fire_truck_objects_actors, ambulance_objects_actors, police_objects_actors, scenario_instance_name)
        stop_label_obj = np.sum((dynamics_objects_m * stop_obj_mask)) > 0
        stop_label = np.sum(tl_red_masks[-1] * modified_env_mask.astype(np.uint8)) > 0
        #obj_masks = cv2.resize(obj_masks, (1034, 586), interpolation=cv2.INTER_CUBIC)


        
        if type(self.create_actors) != type(None):
            if len(actors_to_destroy) != 0:
                self.create_actors.destroy_actors(actors_to_destroy)

            self.create_actors.move_walker()

        return obs_dict, M_history_warp_list, prev_bbox_list, stop_label, stop_label_obj, tl_masks_list, tl_data_route_mask, image, self.current_wp, detection_boxes, ego_boxes, number_of_walker, walker_stop, prediction_stop_label, prediction_stoplabel_mask, nonstop_prediction_label

    def get_modified_env_mask(self, start_point=None):
        # Assuming you want to create a new 200x200 image.
        # For an RGB image, use np.zeros((200, 200, 3)), for grayscale, np.zeros((200, 200))
        modified_env_mask = np.zeros((200, 200), dtype=np.uint8)

        # Define the box dimensions
        box_width, box_height = 3, 20  # Change these values based on your desired box size

        # Calculate the top-left corner of the box to center it
        if type(start_point) == type(None):
            start_point = ((200 - box_width) // 2, (200 - box_height) // 2)

        # Calculate the bottom-right corner of the box
        end_point = (start_point[0] + box_width, start_point[1] + box_height)

        # Box color in BGR (blue, green, red)
        color = (255)  # Here, the box will be blue

        # Draw the rectangle
        cv2.rectangle(modified_env_mask, start_point, end_point, color, -1)

        return np.array(modified_env_mask).astype(np.bool)

    def get_detection_warp(self, compass, time_interval, current_loc, current_speed):
        current_yaw = compass
        item, new_time_interval_list = self.get_item_loc_que(current_speed, current_yaw, time_interval)
        prev_index = 0
        previous_loc = self.estimate_previous_location(current_loc, item[0][0], item[prev_index][0], item[0][1], item[prev_index][1], new_time_interval_list[prev_index])
        self.prev_compass = compass
        self.previous_speed = current_speed
        self.previous_yaw = current_yaw

        return np.array(previous_loc)

    def estimate_previous_location(self, current_loc, current_speed, previous_speed, current_yaw, previous_yaw,
                                   time_interval):
        c_ego_vehicle = (current_yaw-current_yaw)+90
        p_ego_vehicle = (previous_yaw-current_yaw)+90
        # Convert yaw angles from degrees to radians
        current_yaw_rad = math.radians(c_ego_vehicle)
        previous_yaw_rad = math.radians(p_ego_vehicle)

        # Calculate average speed and yaw
        avg_speed = (current_speed + previous_speed) / 2
        avg_yaw_rad = (previous_yaw_rad+current_yaw_rad) / 2

        # Ensure correct angle averaging around 360 degrees
        if abs(current_yaw_rad - previous_yaw_rad) > math.pi:
            avg_yaw_rad += math.pi
            if avg_yaw_rad > math.pi:
                avg_yaw_rad -= 2 * math.pi

        # Calculate distance moved
        distance_moved = (avg_speed * time_interval*2) * self._pixels_per_meter

        # Calculate displacement vector
        dx = (distance_moved * math.cos(avg_yaw_rad))
        dy = (distance_moved * math.sin(avg_yaw_rad))

        # Calculate previous pixel location
        previous_loc = (current_loc[0] - dy, current_loc[1] - dx)

        return previous_loc


    def get_prev_loc_and_transform(self, ev_loc_in_px, ev_transform):
        self.loc_que.append(ev_loc_in_px)
        self.transform_que.append(ev_transform)

        qsize = len(self.loc_que)
        loc_list = []
        transform_list = []
        for idx in self._history_idx:
            idx = max(idx, -1 * qsize)
            loc = self.loc_que[idx]
            loc_list.append(loc)
            transform = self.transform_que[idx]
            transform_list.append(transform)

        return loc_list, transform_list

    def get_item_loc_que(self, current_speed, current_yaw, time_interval):
        self.prev_info_que.append((current_speed, current_yaw))

        qsize = len(self.prev_info_que)
        info_list = []
        interval_list = []
        interval_array = np.arange(0,len(self.prev_info_que))*time_interval
        bbox_list = []
        for index,idx in enumerate(self._history_idx):
            idx = max(idx, -1 * qsize)
            loc = self.prev_info_que[idx]
            info_list.append(loc)
            interval_list.append(interval_array[idx])

        return info_list, interval_list

    def get_item_bbox(self,detected_bbox, unwbbox,selected_time_index=-2):
        self.box_que.append((detected_bbox, unwbbox))
        self.dummy_que.append(self.dummy_counter)
        self.dummy_counter += 1

        qsize = len(self.box_que)
        bbox_list = []
        gt_bbox_list = []
        dummy_list = []
        for index,idx in enumerate(self.plant__history_idx):
            idx = max(idx, -1 * qsize)
            bbox_list.append(self.box_que[idx][1])
            gt_bbox_list.append(self.info_transform_que[idx])
            dummy_list.append(self.dummy_que[idx])

        #return bbox_list[selected_time_index], gt_bbox_list[selected_time_index]
        return bbox_list, gt_bbox_list







    def calculate_angle(self, x1, y1, x2, y2, epsilon=1e-10):
        # Calculate the slope with epsilon to avoid division by zero
        slope = (y2 - y1) / ((x2 - x1) + epsilon)

        # Calculate the angle in radians
        angle_radians = math.atan(slope)

        # Convert the angle to degrees
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees

    def set_info(self,info):
        self.bev_resolution, self.view_array = info

    def get_ego_motion_0(self):
        ego_motions = torch.zeros((1, 3, 4, 4))
        ego_motions[0, :] = torch.eye(4)
        for i, idx in enumerate(self._history_idx):
            idx = max(idx, -1 * len(self.ego_queue))
            if i == 3:
                continue
            next_idx = max(self._history_idx[i + 1], -1 * len(self.ego_queue))
            ego_motions[0, i] = torch.from_numpy(self.get_relative_rt(self.ego_queue[idx], self.ego_queue[next_idx], self.view_array))
            self.box_que
            # ego_motions[0, i] = torch.from_numpy(np.linalg.inv(ego_queue[idx]) @ ego_queue[next_idx])

        return ego_motions

    def get_ego_motion(self):
        ego_motions = torch.zeros((1, 3, 4, 4))
        ego_motions[0, :] = torch.eye(4)
        for i, idx in enumerate(self._history_idx):
            idx = max(idx, -1 * len(self.ego_queue))
            if i == 3:
                continue
            next_idx = max(self._history_idx[i + 1], -1 * len(self.ego_queue))
            ego_motions[0, i] = torch.from_numpy(self.get_relative_rt(self.ego_queue[idx], self.ego_queue[next_idx], self.view_array))
            self.box_que
            # ego_motions[0, i] = torch.from_numpy(np.linalg.inv(ego_queue[idx]) @ ego_queue[next_idx])

        return ego_motions

    def get_relative_rt(self, previous_ego, ego, view):

        lidar_to_world_t0 = get_pose_matrix(previous_ego, use_flat=False)
        lidar_to_world_t1 = get_pose_matrix(ego, use_flat=False)
        future_egomotion = np.linalg.inv(lidar_to_world_t1) @ lidar_to_world_t0

        sh, sw, _ = 1 / self.bev_resolution
        view_rot_only = np.eye(4, dtype=np.float32)
        view_rot_only[0, 0:2] = view[0, 0, 0, 0, 0:2] / sw
        view_rot_only[1, 0:2] = view[0, 0, 0, 1, 0:2] / sh
        future_egomotion = view_rot_only @ future_egomotion @ np.linalg.inv(view_rot_only)
        future_egomotion[3, :3] = 0.0
        future_egomotion[3, 3] = 1.0

        return future_egomotion

    def _get_history_masks(self, M_warp):
        qsize = len(self._history_queue)
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks, data_stops_masks, dynamics_objects_masks = [], [], [], [], [], [], [], []
        vec_list = []
        for idx in self._history_idx:
            idx = max(idx, -1 * qsize)

            vehicles, walkers, tl_green, tl_yellow, tl_red, stops, data_stops, dynamics_objects = self._history_queue[idx]

            obj, vec = self._get_mask_from_actor_list(vehicles, M_warp)
            vec_list.append(vec)
            vehicle_masks.append(obj)

            obj, _ = self._get_mask_from_actor_list(dynamics_objects, M_warp)
            #vec_list.append(vec)
            dynamics_objects_masks.append(obj)

            obj, _ = self._get_mask_from_actor_list(walkers, M_warp)
            walker_masks.append(obj)
            green_masks, green_masks_list = self._get_mask_from_stopline_vtx(tl_green, M_warp)
            tl_green_masks.append(green_masks)
            yellow_masks, yellow_masks_list = self._get_mask_from_stopline_vtx(tl_yellow, M_warp)
            tl_yellow_masks.append(yellow_masks)
            red_masks, red_masks_list = self._get_mask_from_stopline_vtx(tl_red, M_warp)
            tl_red_masks.append(red_masks)
            obj, _ = self._get_mask_from_actor_list(stops, M_warp)
            stop_masks.append(obj)
            obj, _ = self._get_mask_from_actor_list(data_stops, M_warp)
            data_stops_masks.append(obj)
            tl_masks_list = (green_masks_list, yellow_masks_list, red_masks_list)

        return vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks, data_stops_masks, dynamics_objects_masks, vec_list[1], tl_masks_list

    def _get_mask_from_stopline_vtx(self, stopline_vtx, M_warp,):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        mask_list = []
        for sp_locs in stopline_vtx:
            mask_element = np.zeros([800, 800], dtype=np.uint8)
            stopline_in_pixel = np.array([[self._world_to_pixel(x)] for x in sp_locs])
            stopline_warped = cv.transform(stopline_in_pixel, M_warp)
            cv.line(mask, tuple(stopline_warped[0, 0]), tuple(stopline_warped[1, 0]),
                    color=1, thickness=6)
            cv.line(mask_element, tuple(stopline_warped[0, 0]), tuple(stopline_warped[1, 0]),
                    color=1, thickness=6)
            mask_list.append((mask_element, stopline_warped))

        return mask.astype(np.bool), mask_list

    def _get_mask_from_actor_list(self, actor_list, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        corner_list = []
        for actor_transform, bb_loc, bb_ext in actor_list:

            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),
                       #carla.Location(x=bb_ext.x, y=0),
                       carla.Location(x=bb_ext.x, y=bb_ext.y),
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
            corners_warped = cv.transform(corners_in_pixel, M_warp)
            corner_list.append(corners_warped)

            cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
        return mask.astype(np.bool), corner_list

    @staticmethod
    def _get_surrounding_actors(bbox_list, criterium, scale=None, f_and_l=False):
        actors = []
        actor_list = []
        for _bbox in bbox_list:
            try:
                bbox, actor = _bbox
            except:
                bbox = _bbox

            f_and_l_coeff = 1
            if f_and_l:
                f_and_l_coeff = 2

            is_within_distance = criterium(bbox, f_and_l_coeff)
            if is_within_distance:
                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bbox.extent)
                if scale is not None:
                    bb_ext = bb_ext * scale
                    bb_ext.x = max(bb_ext.x, 0.8)
                    bb_ext.y = max(bb_ext.y, 0.8)

                try:
                    actor_list.append(actor)
                except:
                    actor_list.append(None)


                actors.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
        return actors, actor_list

    def _get_warp_transform(self, ev_loc, ev_rot, _pixels_ev_to_bottom=100, _width=200):
        ev_loc_in_px = self._world_to_pixel(ev_loc)
        yaw = np.deg2rad(ev_rot.yaw)

        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw + 0.5*np.pi), np.sin(yaw + 0.5*np.pi)])

        bottom_left = ev_loc_in_px - _pixels_ev_to_bottom * forward_vec - (0.5*_width) * right_vec
        top_left = ev_loc_in_px + (_width-_pixels_ev_to_bottom) * forward_vec - (0.5*_width) * right_vec
        top_right = ev_loc_in_px + (_width-_pixels_ev_to_bottom) * forward_vec + (0.5*_width) * right_vec

        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, _width-1],
                            [0, 0],
                            [_width-1, 0]], dtype=np.float32)
        return cv.getAffineTransform(src_pts, dst_pts)

    def _get_src(self, ev_loc, ev_rot):
        ev_loc_in_px = self._world_to_pixel(ev_loc)
        yaw = np.deg2rad(ev_rot.yaw)

        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw + 0.5*np.pi), np.sin(yaw + 0.5*np.pi)])

        bottom_left = ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5*self._width) * right_vec
        top_left = ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec - (0.5*self._width) * right_vec
        top_right = ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec + (0.5*self._width) * right_vec

        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)

        return src_pts

    def _world_to_pixel(self, location, projective=False):
        """Converts the world coordinates to pixel coordinates"""
        x = self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self._pixels_per_meter * (location.y - self._world_offset[1])

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p

    def _world_to_pixel_width(self, width):
        """Converts the world units to pixel units"""
        return self._pixels_per_meter * width

    def set_is_there_list_parameter(self, stop_masks, tl_green_masks, tl_yellow_masks, tl_red_masks):
        element_1 = np.zeros([self._width, self._width], dtype=np.uint8)
        element_1[(stop_masks[0] + tl_green_masks[0] + tl_yellow_masks[0] + tl_red_masks[0])] = 255

        stop_sign = np.sum((stop_masks[-1] * self.stop_route_mask)).astype(np.bool)
        green_sign = np.sum((tl_green_masks[-1] * self.tl_route_masks)).astype(np.bool)
        yellow_sign = np.sum((tl_yellow_masks[-1] * self.tl_route_masks)).astype(np.bool)
        red_sign = np.sum((tl_red_masks[-1] * self.tl_route_masks)).astype(np.bool)
        #cv2.imwrite("stop_masks.png", stop_masks[-1].astype(np.uint8)*255)
        #cv2.imwrite("tl_route_masks.png", self.tl_route_masks.astype(np.uint8)*255)

        self.color = "unknown"
        if np.sum(stop_sign).astype(np.bool):
            self.color = "stop_sign"
        if np.sum(red_sign).astype(np.bool):
            self.color = "red"
        if np.sum(yellow_sign).astype(np.bool):
            self.color = "yellow"
        if np.sum(green_sign).astype(np.bool):
            self.color = "green"

    def get_is_there_list_parameter(self):
        stop_box = []
        """for actor in self._parent_actor.criteria_stop._list_stop_signs:
            bb = carla.BoundingBox(actor.get_transform().location, actor.bounding_box.extent)
            bb.rotation = actor.get_transform().rotation
            stop_box.append(bb)"""
        return self.color, stop_box, self.tl_corner_bb_list

    def set_sensor(self, sensors):
        # gps = np.array([sensors[0]['lat'], sensors[0]['lon']])#['compass']
        self.sensor = {'gps': np.array([sensors[0]['lat'], sensors[0]['lon']]), 'imu': sensors[1], 'sensor': sensors}


    def set_control_with(self,_state):
        self.control_with = _state

    def draw_attention_bb(self, attention_mask, keep_vehicle_ids, keep_vehicle_attn, M_warp, real_vehicle_masks, scale=1.5):
        actors = self._world.get_actors()
        all_vehicles = actors.filter('*vehicle*')
        all_walkers = actors.filter('*pedestrian*')
        #lights_list = actors.filter("*traffic_light*")
        objects_list = [all_vehicles,all_walkers]
        for objs in objects_list:
            for vehicle in objs:
                # print(vehicle.id)
                if isinstance(vehicle,carla.libcarla.Walker):
                    scale = 5
                else:
                    scale = 2

                if vehicle.id in keep_vehicle_ids:
                    vehicle_mask = np.zeros([self._width, self._width], dtype=np.uint8)
                    index = keep_vehicle_ids.index(vehicle.id)
                    # cmap = plt.get_cmap('YlOrRd')
                    # c = cmap(object[1])
                    # color = carla.Color(*[int(i*255) for i in c])
                    c = self.get_color(keep_vehicle_attn[index])
                    #color = carla.Color(r=int(c[0]), g=int(c[1]), b=int(c[2]))
                    color = int(c[0]), int(c[1]), int(c[2])
                    loc = vehicle.get_location()
                    bb_loc = carla.Location()
                    bb = carla.BoundingBox(loc, vehicle.bounding_box.extent)
                    actor_transform = carla.Transform(bb.location, vehicle.get_transform().rotation)
                    bb_ext = carla.Vector3D(vehicle.bounding_box.extent)
                    if scale is not None:
                        bb_ext = bb_ext * scale
                        bb_ext.x = max(bb_ext.x, 0.8)
                        bb_ext.y = max(bb_ext.y, 0.8)
                    corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                               carla.Location(x=bb_ext.x, y=-bb_ext.y),
                               carla.Location(x=bb_ext.x, y=0),
                               carla.Location(x=bb_ext.x, y=bb_ext.y),
                               carla.Location(x=-bb_ext.x, y=bb_ext.y)]
                    corners = [bb_loc + corner for corner in corners]

                    corners = [actor_transform.transform(corner) for corner in corners]
                    corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
                    corners_warped = cv.transform(corners_in_pixel, M_warp)
                    cv.fillConvexPoly(vehicle_mask, np.round(corners_warped).astype(np.int32), 1)
                    vehicle_mask = vehicle_mask.astype(np.bool)
                    index_mask = vehicle_mask * (1 - real_vehicle_masks[0])
                    attention_mask[index_mask>0] = (int(255*keep_vehicle_attn[index]),0,0) #color #
        #cv2.imwrite("attention_mask.png",attention_mask)
        return attention_mask


    def get_color(self, attention):
        attention = 1
        colors = [
            (255, 255, 255, 255),
            # (220, 228, 180, 255),
            # (190, 225, 150, 255),
            (240, 240, 210, 255),
            # (190, 219, 96, 255),
            (240, 220, 150, 255),
            # (170, 213, 79, 255),
            (240, 210, 110, 255),
            # (155, 206, 62, 255),
            (240, 200, 70, 255),
            # (162, 199, 44, 255),
            (240, 190, 30, 255),
            # (170, 192, 20, 255),
            (240, 185, 0, 255),
            # (177, 185, 0, 255),
            (240, 181, 0, 255),
            # (184, 177, 0, 255),
            (240, 173, 0, 255),
            # (191, 169, 0, 255),
            (240, 165, 0, 255),
            # (198, 160, 0, 255),
            (240, 156, 0, 255),
            # (205, 151, 0, 255),
            (240, 147, 0, 255),
            # (212, 142, 0, 255),
            (240, 137, 0, 255),
            # (218, 131, 0, 255),
            (240, 126, 0, 255),
            # (224, 120, 0, 255),
            (240, 114, 0, 255),
            # (230, 108, 0, 255),
            (240, 102, 0, 255),
            # (235, 95, 0, 255),
            (240, 88, 0, 255),
            # (240, 80, 0, 255),
            (242, 71, 0, 255),
            # (244, 61, 0, 255),
            (246, 49, 0, 255),
            # (247, 34, 0, 255),
            (248, 15, 0, 255),
            (249, 6, 6, 255),
        ]

        ix = int(attention * (len(colors) - 1))
        return colors[ix]
    def clean(self):
        self._parent_actor = None
        self._world = None
        self._history_queue.clear()
    
    def convert_bbox2mask(self,detected_bbox):
        mask = np.zeros((self._width, self._width)).astype(np.uint8)
        for bbox in detected_bbox:
            # Your bounding box parameters
            center_x, center_y = bbox[0], bbox[1]  # center coordinates
            width, height = bbox[4], bbox[5]  # width and height
            angle = bbox[3]

            # Create a rotated rectangle
            rect = ((center_x, center_y), (width, height), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Draw the filled rotated rectangle
            cv2.drawContours(mask, [box], 0, (255), -1)

        return mask.astype(np.bool)

    def save_log_data(self, detection_data=None,boxes=None):
        self.log_saved = False
        if (self.counter - 1) % self.variables["time_window"] == 0:
            data_to_save = {}
            self.log_saved = True

            # data_to_save['images'] = im_data(detection_data[0]).unsqueeze(0)
            for idx, im in enumerate(self.variables["cams"]):
                img = Image.fromarray(detection_data[im][1][:, :, -2::-1])
                filename = f"{self.img_counter:05d}.jpeg"
                img.save(self.file_name_image_front +'/'+ self.variables["cams"][idx] + '/' + filename)


            data_to_save['boxes'] = [boxes]
            data_to_save['intrinsics'] = intr_data(self.intrinsics).unsqueeze(0)
            data_to_save['cams_to_lidar'] = ext_data(self.extrinsics).unsqueeze(0)
            data_to_save['view'] = torch.tensor(self.view_array[0]).unsqueeze(0)


            if os.path.isdir(self.file_name_pickle):
                # Specify the file name with .pickle extension
                file_path = os.path.join(self.file_name_pickle, "info.pickle")

                # Now open the file to append data
                with open(file_path, 'ab') as pickle_file:
                    pickle.dump(data_to_save, pickle_file)

            self.img_counter += 1


        self.counter += 1

    def _syc_pickle_file(self, file_path):
        if self.new_run:
            object_list = []
            with open(file_path, 'rb') as file:
                while True:
                    try:
                        # Attempt to load the next object in the file
                        object_list.append(pickle.load(file))
                        # Process or print the object here
                    except EOFError:
                        # End of file reached
                        break
            new_pickle_file = object_list[0:self.img_counter]
            print("len(new_pickle_file):",len(new_pickle_file))
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(new_pickle_file, pickle_file)
            #new_pickle_file[self.img_counter]
            return new_pickle_file



    def sensors_(self):#'front_left', 'front', 'front_right', 'back_left', 'back', 'back_right'
        sensors = []
        w = 704
        h = 396
        # Add cameras
        new_camera = {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': -0.5, 'z': 2.0,
                      'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                      'width': w, 'height': h, 'fov': 100, 'id': 'front_left'}
        sensors.append(new_camera)
        self.intrinsics["front_left"], self.extrinsics["front_left"] = self.carla_to_lss.find_intrinsics(
            new_camera['width'], new_camera['height'], new_camera['fov'],
            new_camera['x'],
            new_camera['y'],
            new_camera['z'],
            new_camera['roll'], new_camera['pitch'],
            new_camera['yaw'])

        new_camera = {'type': 'sensor.camera.rgb', 'x': 0.5, 'y': 0.0, 'z': 2.0,
                      'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': w, 'height': h, 'fov': 100, 'id': 'front'}
        sensors.append(new_camera)
        self.intrinsics["front"], self.extrinsics["front"] = self.carla_to_lss.find_intrinsics(new_camera['width'],
                                                                                               new_camera['height'],
                                                                                               new_camera['fov'],
                                                                                               new_camera['x'],
                                                                                               new_camera['y'],
                                                                                               new_camera['z'],
                                                                                               new_camera['roll'],
                                                                                               new_camera['pitch'],
                                                                                               new_camera['yaw'])

        new_camera = {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.5, 'z': 2.0,
                      'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0,
                      'width': w, 'height': h, 'fov': 100, 'id': 'front_right'}
        sensors.append(new_camera)
        self.intrinsics["front_right"], self.extrinsics["front_right"] = self.carla_to_lss.find_intrinsics(
            new_camera['width'], new_camera['height'], new_camera['fov'],
            new_camera['x'],
            new_camera['y'],
            new_camera['z'],
            new_camera['roll'], new_camera['pitch'],
            new_camera['yaw'])

        new_camera = {'type': 'sensor.camera.rgb', 'x': -0.5, 'y': 0.0, 'z': 2.0,
                      'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
                      'width': w, 'height': h, 'fov': 100, 'id': 'back'}
        sensors.append(new_camera)
        self.intrinsics["back"], self.extrinsics["back"] = self.carla_to_lss.find_intrinsics(new_camera['width'],
                                                                                             new_camera['height'],
                                                                                             new_camera['fov'],
                                                                                             new_camera['x'],
                                                                                             new_camera['y'],
                                                                                             new_camera['z'],
                                                                                             new_camera['roll'],
                                                                                             new_camera['pitch'],
                                                                                             new_camera['yaw'])

        #for i in self.intrinsics:
        #    self.intrinsics[i] = self.transforms_val.update_intrinsics_v3(self.intrinsics[i])

        radar_config = {
            'type': 'sensor.other.radar',
            'x': 2.0,  # Position in meters on the x-axis relative to the vehicle
            'y': 0.0,  # Position in meters on the y-axis relative to the vehicle
            'z': 1.0,  # Position in meters on the z-axis relative to the vehicle
            'roll': 0.0,  # Rotation around the forward axis
            'pitch': 0.0,  # Rotation around the right axis
            'yaw': 0.0,  # Rotation around the down axis
            'horizontal_fov': 30.0,  # Horizontal field of view in degrees
            'vertical_fov': 10.0,  # Vertical field of view in degrees
            'range': 100.0,  # Maximum detection range in meters
            'points_per_second': 1500,  # Points generated by the radar per second
            'id': 'front_radar'  # Identifier for the radar sensor
        }
        sensors.append(radar_config)

        return sensors

    def get_relative_speed(self,actor_speed,ego_speed):
        relative_speed = np.array([actor_speed.x, actor_speed.y, actor_speed.z]) - np.array([ego_speed.x, ego_speed.y, ego_speed.z])

        return relative_speed

    def get_surrounding_objects_vectors(self, vehicles, actors, ev_loc, ev_rot, label):
        boxes = []
        ego = {}
        ego['rotation'] = euler_to_quaternion(-ev_rot.roll, ev_rot.pitch, -ev_rot.yaw)
        ego['translation'] = ev_loc.x, -ev_loc.y, ev_loc.z
        ego_pose_matrix = get_pose_matrix(ego, use_flat=False)
        index = 0
        for actor_transform, bb_loc, bb_ext in vehicles:
            obj = {}

            obj['rotation'] = euler_to_quaternion(-actor_transform.rotation.roll, actor_transform.rotation.pitch,
                                                  -actor_transform.rotation.yaw)
            obj['translation'] = actor_transform.location.x, -actor_transform.location.y, actor_transform.location.z
            obj_pose_matrix = get_pose_matrix(obj, use_flat=False)
            inside_bev = False  # flag showing the object is inside bev or not
            box = {}
            box['label'] = label
            if label == 1:
                box['name'] = 'vehicle'
            elif label == 2:
                box['name'] = 'pedestrian'
            elif label == 3:
                box['name'] = 'dynamic_object'
            elif label == 4:
                box['name'] = 'police'
            elif label == 5:
                box['name'] = 'ambulance'
            elif label == 6:
                box['name'] = 'firetruck'
            elif label == 7:
                box['name'] = 'crossbike'

            ego_to_obj = np.linalg.inv(ego_pose_matrix) @ obj_pose_matrix
            box['center'] = ego_to_obj[:3, 3]
            box['translation'] = obj['translation']
            box['rotation'] = obj['rotation']
            try:
                box['id'] = actors[index].id
                box['relative_speed'] = self.get_relative_speed(actors[index].get_velocity(),
                                                                self.vehicle.get_velocity())

            except:
                box['id'] = 1
                box['relative_speed'] = 0

            box['rotation_matrix'] = ego_to_obj[:3, :3]
            box['orientation'] = Quaternion(matrix=ego_to_obj)
            box['wlh'] = np.array([bb_ext.y * 2, bb_ext.x * 2, bb_ext.z * 2])

            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),  # bottom_left
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),  # top_left
                       carla.Location(x=bb_ext.x, y=0),  # center
                       carla.Location(x=bb_ext.x, y=bb_ext.y),  # top_right
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]  # bottom_right
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]

            boxes.append(box)

            index += 1

        return boxes, ego

    def get_walker_stop(self,ev_loc, walkers_actors, fire_truck_objects_actors, ambulance_objects_actors, police_objects_actors, scenario_instance_name):
        number_of_walker = len(walkers_actors)
        walker_stop = False
        if number_of_walker != 0:
            walker_stop = walkers_actors[0].get_velocity().length() > 0.0

        if "OppositeVehicleRunningRedLight" in scenario_instance_name or 'OppositeVehicleTakingPriority' in scenario_instance_name:
            if len(fire_truck_objects_actors) != 0 and fire_truck_objects_actors[0].get_transform().location.distance(ev_loc) < 30: # or len(ambulance_objects) != 0 or len(police_objects) != 0:
                walker_stop = fire_truck_objects_actors[0].get_velocity().length() > 0.0 #or ambulance_objects[0].get_velocity().length() > 0.0 or police_objects[0].get_velocity().length() > 0.0

            elif len(ambulance_objects_actors) != 0 and ambulance_objects_actors[0].get_transform().location.distance(ev_loc) < 30:
                walker_stop = ambulance_objects_actors[0].get_velocity().length() > 0.0

            elif len(police_objects_actors) != 0 and police_objects_actors[0].get_transform().location.distance(ev_loc) < 30:
                walker_stop = police_objects_actors[0].get_velocity().length() > 0.0

        walker_stop = False

        return number_of_walker, walker_stop




