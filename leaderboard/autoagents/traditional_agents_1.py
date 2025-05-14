#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import copy

import carla
from agents.navigation.basic_agent import BasicAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from leaderboard.autoagents.traditional_agents_files.carla_gym.plant_sensor import Plant_sensor
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.chauffeurnet_wo_hd import ObsManager
#from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.chauffeurnet import ObsManager

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from leaderboard.autoagents.traditional_agents_files.perception.traffic_lights import Traffic_Lights
#from leaderboard.autoagents.traditional_agents_files.utils.get_speed_yaw import Get_Speed_Yaw
from leaderboard.autoagents.traditional_agents_files.perception.object_detection import Object_Detection

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.planning import Planning

from leaderboard.autoagents.traditional_agents_files.carla_gym.utils.traffic_light import TrafficLightHandler

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.nav_planner import interpolate_trajectory

from leaderboard.utils.lane_guidance import Lane_Guidance
from leaderboard.utils.tl_data_collection import Tl_Data_Collection

from leaderboard.utils.lane import Lane

from leaderboard.utils.rule_based_stop_sign import Rule_Based_Stop_Sign
from leaderboard.utils.four_cameras_and_lidar_data_collection import Four_Cameras_and_Lidar_Data_Collection

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.agent_utils.coordinate_utils import preprocess_compass

from accelerate import infer_auto_device_map, init_empty_weights
import time


from leaderboard.autoagents.traditional_agents_files.utils.manuel_control import Manuel_Control

import json
import numpy as np
import cv2

import math
import os
import glob
from collections import deque



from leaderboard.autoagents.traditional_agents_files.utils.process_radar import Process_Radar

from leaderboard.autoagents.traditional_agents_files.roach.run_roach import Run_Roach

from leaderboard.autoagents.traditional_agents_files.perception.traffic_lights import Traffic_Lights

def get_entry_point():
    return 'TraditionalAgent'

class TraditionalAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False

    def get_name(self):
        return 'TraditionalAgent'

    def setup(self, path_to_conf_file, llm_model):
        """
        Setup the agent parameters
        """
        self.track = Track.MAP

        self._agent = None

        self.plant_sensor = Plant_sensor()

        detection_config = {'detection_threshold': 0.4,
                            'perception_checkpoint': '/home/tg22/lss_models/fine_tuned_epoch=19-step=30160.ckpt',
                            'monocular_perception': False, 'lane_detection': False,
                            'lane_yaml_path': '/home/ad22/leaderboard_carla/tairvision_master/settings/Deeplab/deeplabv3_resnet18_openlane_culane_tusimple_curvelanes_llamas_once_klane_carlane_lanetrainer_640x360_thick_openlane_val_excluded.yml'}
        self.obj_detection = Object_Detection(detection_config)

        obs_config = {'width_in_pixels':200,'pixels_ev_to_bottom':100,'pixels_per_meter':4,'history_idx':[-16,-11,-6,-1],'scale_bbox':True,'scale_mask_col':1.0}#,'percetion_cfgs':self.obj_detection.get_perception_cfg()}

        self.chauffeurnet = ObsManager(obs_config)

        self.ego_vehicle = EgoVehicleHandler(self._client)
        try:
            with open('autoagents/traditional_agents_files/perception/tl_config.json','r') as json_file:
                config = json.load(json_file)
        except:
            with open('leaderboard/autoagents/traditional_agents_files/perception/tl_config.json','r') as json_file:
                config = json.load(json_file)

        config.update({'device':2})
        self.traffic_lights = Traffic_Lights(config)

        #info = self.obj_detection.get_info()
        #self.chauffeurnet.set_info(info)#"""
        #self.get_attribute = Get_Speed_Yaw()
        self.sensors_list = []
        self.count_agent = 0
        self.tl_dataset_count = 0
        self.det_count = 0
        self.counter = 0
        self._carla_map = 'Town12'
        self.draw_count = 0
        self.positive_sample_count = 0
        self.negative_sample_count = 0
        self.previous_positive_sample = True
        self.llm_model = llm_model
        self.high_level_action = 'current_lane'

        self.log_index = 0
        self.plant_dataset_collection = False
        self.plant_inference = False
        self.opendrive_hd_map = None
        self.lane_guidance_init = True

        self.process_radar = Process_Radar()
        self.wp_based_index = 0

        self._global_plan_world_coord_index = 0

        self.lane_class = Lane()
        self.lane_guidance = Lane_Guidance(lane_class=self.lane_class)

        self.rule_based_stop_sign = Rule_Based_Stop_Sign(lane_class=self.lane_class)


        self.prev_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)


        asd = 0

        self.prev_steer_que = deque(maxlen=50)
        self.prev_steer_que.append(0)

        self.weather_change_count = 0
        self.weather_change_index = 0
        pickle_file_path = '/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/weather_class.pickle'
        self.read_weather_class(pickle_file_path)
        self.roach_control_label = True
        self.save_control_label = False

        self.data_collection_for_vision = False
        self.prev_scenario_instance_name = 'None'
        self.scenario_instance_name = 'None_1'


    def get_counter_value(self):
        return self.chauffeurnet.get_counter_value()

    def set_path(self, file_name_pickle, file_name_image_front):
        try:
            self.chauffeurnet.set_path(file_name_pickle, file_name_image_front)
        except:
            pass



    def list_folders(self, directory):
        """List all folders in a given directory."""
        folders = []
        for item in os.listdir(directory):
            # Full path of the item
            item_path = os.path.join(directory, item)
            # Check if the item is a directory
            if os.path.isdir(item_path):
                folders.append(item)
        return folders

    def setup_sensors(self, vehicle, _world):
        self._vehicle = vehicle
        self._world = _world

        self.run_roach = Run_Roach(vehicle)


        pixels_per_meter, _world_offset = self.chauffeurnet.attach_ego_vehicle(vehicle)

        self.planning = Planning(ego_vehicles=vehicle,_world=_world)
        self.planning.set_info(pixels_per_meter, _world_offset)


        # register traffic lights
        TrafficLightHandler.reset(self._world)
        self.initial_ego_location_z = self._vehicle.get_transform().location.z

        self.previous_bb = None

        if self.data_collection_for_vision:
            #self.tl_data_collection = Tl_Data_Collection(self._vehicle, self._world, self.camera, self.current_path)
            self.f_and_l_data_collection = Four_Cameras_and_Lidar_Data_Collection(self._vehicle, self._world, self.current_path)

            try:
                #self.tl_data_collection.close_file()
                self.f_and_l_data_collection.close_file()
            except:
                pass

            #self.tl_data_collection.create_file()
            self.f_and_l_data_collection.create_file()


    def get_sensor_list(self):
        return self.plant_sensor._sensors_list

    def set_client(self, client):
        self._client = client

    def set_path(self,current_path):
        self.current_path = current_path

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]
        """
        desired_scale = 0.5
        fov = 70
        width = 1034
        height = 586

        required_altitude = 40 #(width * desired_scale) / (2 * math.tan(math.radians(fov / 2)))

        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': required_altitude, 'roll': 0.0,
             'pitch': -90.0, 'yaw': 0.0,
             'width': width, 'height': height, 'fov': 70, 'id': 'bev'},
            {'type': 'sensor.lidar.ray_cast', 'x': 0.056287, 'y': 0.0, 'z': 1.84023, 'yaw': 90.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'lidar_0'},
            {'type': 'sensor.lidar.ray_cast', 'x': 0.056287, 'y': 0.0, 'z': 1.84023, 'yaw': -90.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'lidar_1'}, #
        ]
        self.bev_meters_per_pixel_x, self.bev_meters_per_pixel_y = self.find_pom(altitude=required_altitude, fov=fov, image_width=width, image_height=height)

        self.bev_image_x, self.bev_image_y = height * self.bev_meters_per_pixel_x, width*self.bev_meters_per_pixel_y
        print("self.bev_image_x, self.bev_image_y:",self.bev_image_x, self.bev_image_y)
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
            'points_per_second': 20,  # Points generated by the radar per second
            'id': 'front_radar'  # Identifier for the radar sensor
        }


        intrinsics, extrinsics, transforms_val, carla_to_lss = self.obj_detection.get_parameters()

        if len(self.sensors_list)==0:
            object_detection_sensor, intrinsics, extrinsics = self.obj_detection.sensors()#.four_sensors()#
            self.lidar_extrinsic = self.obj_detection.get_lidar_info(sensors)
            with open('object_detection_sensor.json','w') as json_file:
                json.dump(object_detection_sensor,json_file,indent=len(object_detection_sensor))


            self.sensors_list = self.plant_sensor.sensors + sensors + object_detection_sensor #.append(sensors[0])
            if radar_config not in self.sensors_list:
                self.sensors_list.append(radar_config)

            self.image_count = 0#"""

            new_intrinsics_dict = {}
            for _key in intrinsics.keys():
                new_intrinsics_dict.update({_key: intrinsics[_key].cpu().numpy().tolist()})

            new_extrinsics_dict = {}
            for _key in extrinsics.keys():
                new_extrinsics_dict.update({_key: extrinsics[_key].cpu().numpy().tolist()})

            with open('new_intrinsics_dict.json','w') as json_file:
                json.dump(new_intrinsics_dict,json_file,indent=len(new_intrinsics_dict))

            with open('new_extrinsics_dict.json', 'w') as json_file:
                json.dump(new_extrinsics_dict, json_file, indent=len(new_extrinsics_dict))


            self.traffic_lights.set_settings(new_intrinsics_dict, new_extrinsics_dict)

        return self.sensors_list#sensors#

    def set_camera_sensor(self, sensor):
        self.camera = sensor

    def set_global_plan_wp_list(self, wp_list, plant_global_plan_gps, plant_global_plan_world_coord):
        self.wp_list = wp_list
        self.plant_global_plan_gps = plant_global_plan_gps
        self.plant_global_plan_world_coord = plant_global_plan_world_coord
        self.negative_sample_read_count = 0
        self.frame_mean_speed = 0

        #try:
        #    self.tl_data_collection.close_file()
        #except:
        #    pass


        self.previous_bb = None




    def run_planning_model(self, input_data, tl_outputs, plant_boxes,  bbox, ego_motion, bev_image, guidance_masks):

        keep_vehicle_ids, keep_vehicle_attn, light_hazard, control = None, None, False, None
        if self.plant_dataset_collection or self.plant_inference:
            control, pred_wp, target_point, keep_vehicle_ids, keep_vehicle_attn = self.planning(bbox=bbox,input_data=input_data,
                ego_actor=self._vehicle, global_plan_gps=self.plant_global_plan_gps,
                global_plan_world_coord=self.plant_global_plan_world_coord, light_hazard=light_hazard, plant_boxes=plant_boxes, ego_motion=ego_motion, plant_dataset_collection=self.plant_dataset_collection)

        tl_masks, line_color, stop_label = self.planning.draw_attention_bb_in_carla(input_data, tl_outputs, self._world, keep_vehicle_ids, keep_vehicle_attn, ego_vehicle=self._vehicle, ego_location=self._vehicle.get_transform().location, ego_rotation=self._vehicle.get_transform().rotation, plant_inference=self.plant_inference, bev_meters_per_pixel_x=self.bev_meters_per_pixel_x,  bev_meters_per_pixel_y=self.bev_meters_per_pixel_y , bev_image=bev_image, compass= input_data['imu'][1][-1],guidance_masks=guidance_masks)


        return control, tl_masks, line_color, stop_label

    def plant_data_save_score(self, name, file_name_without_extension, save_files_name, score_composed,  score_route,  score_penalty):
        self.planning.save_score(name, file_name_without_extension, save_files_name, score_composed, score_route, score_penalty)

    def plant_warp_pixel_location(self, past_pixel, ego_motion, M_warp):
        """
        Warp the past pixel location based on the ego motion.

        Parameters:
        past_pixel (tuple): The past pixel location of the other vehicle (x, y).
        ego_motion (numpy array): 1x6 vector [x, y, z, pitch, yaw, roll] of the ego vehicle.

        Returns:
        tuple: The warped pixel location.
        """
        # Extract 2D translation and yaw rotation from ego_motion
        dx, dy, _, _, yaw, _ = ego_motion

        # Convert yaw to radians
        yaw_rad = np.radians(yaw)

        # Create a 2D rotation matrix
        rotation_matrix = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad), np.cos(yaw_rad)]
        ])

        # Apply rotation and then translation
        warped_pixel = []
        """for corner in past_pixel:
            cv2.transform(corner, M_warp)
            warped_corner = np.dot(rotation_matrix, corner) + np.array([dx, dy])
            warped_pixel.append(warped_corner)"""
        #warped_pixel = np.stack(warped_pixel,0)
        warped_pixel = cv2.transform(np.expand_dims(past_pixel, 1), M_warp)#cv2.warpAffine(img,M,(cols,rows))

        #print("warped_pixel:",warped_pixel,"past_pixel:",past_pixel,"yaw:",yaw,"dx, dy:",dx, dy)
        return warped_pixel.squeeze(1)

    def get_axis_aligned_bbox(self, points):
        """
        Convert a bounding box represented by four points into an axis-aligned bounding box.

        Parameters:
        points -- a tuple of four points, each point is a tuple (x, y)
        """
        x_coordinates, y_coordinates = zip(*points)
        return (min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates))

    def calculate_iou(self, box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters:
        box1 -- the first bounding box, a tuple of four points (x, y)
        box2 -- the second bounding box, a tuple of four points (x, y)
        """

        # Convert the boxes to axis-aligned bounding boxes
        box1 = self.get_axis_aligned_bbox(box1)
        box2 = self.get_axis_aligned_bbox(box2)

        # Determine the coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The area of intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # The area of both AABBs
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # The area of the union
        union_area = box1_area + box2_area - intersection_area

        # IoU calculation
        iou = intersection_area / union_area

        return iou

    def draw_bounding_box(self, corners):
        # Create a blank image, white background

        mask = np.zeros((200, 200)).astype(np.uint8)

        # Draw the bounding box
        # Assuming corners are in the format [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        for cor in corners:
            cor = np.int0(cor)
            cv2.drawContours(mask, [cor], 0, (255), -1)

        return mask.astype(np.bool)

    def calculate_angle(self, x1, y1, x2, y2):
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    def plant_get_speed(self,bbox, prev_bbox, ego_motion, M_warp, compass):
        image = np.ones((200, 200, 3), np.uint8) * 0
        speed_array = np.zeros(len(bbox)).astype(np.float)
        if not isinstance(prev_bbox, type(None)):
            warped_previous_bb = []
            for index, box in enumerate(prev_bbox):
                warped_previous_bb.append(self.plant_warp_pixel_location(box,ego_motion,M_warp))
            mask_red = self.draw_bounding_box(warped_previous_bb)
            image[mask_red] = (255,0,0)
            mask_blue = self.draw_bounding_box(bbox)
            image[mask_blue] = (0,0,255)

        speed_array = np.zeros(len(bbox)).astype(np.float)
        angle_array = np.zeros(len(bbox)).astype(np.float)
        for index, box in enumerate(bbox):
            print("box:",box)
            if not isinstance(self.previous_bb, type(None)):
                iou_list = []
                for _, prev_bbox in enumerate(warped_previous_bb):
                    iou_list.append(self.calculate_iou(prev_bbox, box))
                if np.sum(iou_list) != 0:
                    max_iou_index = np.argmax(iou_list)
                    prev_box = warped_previous_bb[max_iou_index]
                    prev_center_x, prev_center_y = prev_box.mean(0)[0], prev_box.mean(0)[1]
                    center_x, center_y = box.mean(0)[0], box.mean(0)[1]
                    angle = (self.calculate_angle(prev_center_x, prev_center_y, center_x, center_y)-90)
                    angle += math.degrees(compass)%360
                    speed = ((center_x-prev_center_x)**2+(center_y-prev_center_y)**2)**0.5
                    speed *= 1#4#pixel_per_meter
                    speed_array[index] = speed
                    angle_array[index] = angle
                else:
                    speed_array[index] = 7*4
        print("angle_array:",angle_array,"ego compass:",math.degrees(compass))
        self.previous_bb = bbox
        return speed_array, angle_array, image

    def set_scenario_gt(self, check_current_scenario_name):
        self.check_current_scenario_name = check_current_scenario_name

    def stabilize(self, current_wp, control):
        if (str(current_wp[1]) == 'RoadOption.LANEFOLLOW' or str(current_wp[1]) == 'RoadOption.STRAIGHT') and abs(current_wp[0].transform.rotation.yaw%360 - self._vehicle.get_transform().rotation.yaw%360) < 5: #self._vehicle.get_transform().rotation.yaw
            steer = self.calculate_steering_angle(self._vehicle.get_transform(), current_wp[0].transform) #np.mean(self.prev_steer_que)  # np.clip(control.steer, self.prev_control.steer - 0.1, self.prev_control.steer + 0.1)
            self.stabilize_label = True
            self.prev_steer_que.append(control.steer)
        else:
            steer = control.steer
            self.stabilize_label = False
            self.prev_steer_que.append(0.0)

        return steer

    def calculate_steering_angle(self, vehicle_transform, waypoint_transform):
        """
        Calculate the steering angle to align the vehicle with the waypoint.

        Parameters:
        - vehicle_transform: The transform (location and rotation) of the vehicle.
        - waypoint_transform: The transform (location and rotation) of the waypoint.

        Returns:
        - Steering command as a float between -1.0 and 1.0.
        """
        # Calculate the desired yaw angle towards the waypoint
        dx = waypoint_transform.location.x - vehicle_transform.location.x
        dy = waypoint_transform.location.y - vehicle_transform.location.y
        desired_yaw = math.atan2(dy, dx)

        # Convert angles to degrees and normalize to [-180, 180]
        desired_yaw_deg = math.degrees(desired_yaw) % 360
        if desired_yaw_deg > 180:
            desired_yaw_deg -= 360

        current_yaw_deg = vehicle_transform.rotation.yaw % 360
        if current_yaw_deg > 180:
            current_yaw_deg -= 360

        # Calculate the difference and normalize to [-1, 1] for steering command
        yaw_diff = desired_yaw_deg - current_yaw_deg
        if yaw_diff > 180:
            yaw_diff -= 360
        elif yaw_diff < -180:
            yaw_diff += 360

        # Assuming a simple proportional controller for demonstration
        # This factor controls the sensitivity of the steering, may need tuning
        steering_sensitivity = 1.0 / 90.0  # Adjust this value based on your needs
        steering_command = max(min(yaw_diff * steering_sensitivity, 1.0), -1.0)

        return steering_command


    def run_step(self, world, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        scenario_instance_name = self.check_current_scenario_name(CarlaDataProvider.get_location(self._vehicle))#ego_location = CarlaDataProvider.get_location(self._vehicle)
        self.lane_class(input_data, self._global_plan_world_coord)

        masks, _, _, _, lidar_image, diff_location, diff_yaw_degree = self.obj_detection(input_data, self._vehicle, self.lane_class.current_wp, self._world.get_map())
        tl_output, tl_image, tl_bev_image, tl_state, score_list_lcte, pred_list = self.traffic_lights(input_data)#tl_outputs['metrics']

        if self.prev_scenario_instance_name != scenario_instance_name and scenario_instance_name != 'None_1':
            self.scenario_instance_name = scenario_instance_name

        if self.data_collection_for_vision:
            self.change_weather_and_time()
        stop_label = False
        guidance_masks = self.lane_guidance(input_data, self._global_plan_world_coord)
        self.planning.update_get_transform(self.lane_class.get_transform)

        current_speed = self._vehicle.get_velocity().length()
        obs_dict, _, _, stop_label_tl, stop_label_obj, tl_masks_list, c_route, roach_image, current_wp, detection_boxes, ego_boxes, number_of_walker, walker_stop, prediction_stop_label, prediction_stoplabel_mask, nonstop_prediction_label = self.chauffeurnet.get_observation(input_data, self.wp_list, current_speed, scenario_instance_name=self.scenario_instance_name)

        #------traffic light data collection---------
        if self.data_collection_for_vision:
            #tl_image = self.tl_data_collection(input_data, tl_masks_list, c_route)
            self.f_and_l_data_collection(input_data, detection_boxes, ego_boxes)
        #------traffic light data collection---------


        ego_loc = self._vehicle.get_transform().location#set_transform(
        ego_rot = self._vehicle.get_transform().rotation
        ego_speed = self._vehicle.get_velocity()#set_target_velocity

        #tl_outputs, tl_image = self.traffic_lights(input_data)

        if self.plant_dataset_collection:
            self.log_traffic_manager_tick(ego_loc,self._vehicle.id)
            pass

        self.det_count += 1


        current_speed = 10


        bev_image = input_data['bev'][1][:,:,:3]
        #bev_image[guidance_masks>0] = (255,255,255)

        #control, tl_masks, line_color, stop_label = self.run_planning_model(input_data, tl_outputs=tl_outputs['metrics'], plant_boxes=None, bbox=None,
        #                                  ego_motion=None,bev_image=bev_image,guidance_masks=guidance_masks)

        stop_label, bev_image = self.rule_based_stop_sign(input_data, self._vehicle.get_transform().location, bev_image)

        #stop_label_obj = np.sum(guidance_masks*obj_masks) > 0


        roach_image = cv2.resize(roach_image, (bev_image.shape[1], bev_image.shape[0]))[:,:,:3]#
        prediction_stoplabel_rgb = np.zeros([prediction_stoplabel_mask.shape[0], prediction_stoplabel_mask.shape[1], 3], dtype=np.uint8)
        prediction_stoplabel_rgb[prediction_stoplabel_mask.astype(np.bool)] = (255, 255, 255)

        prediction_stoplabel_rgb = cv2.resize(prediction_stoplabel_rgb, (bev_image.shape[1], bev_image.shape[0]))
        tl_image = cv2.resize(tl_image, (bev_image.shape[1], bev_image.shape[0]))
        masks = cv2.resize(masks, (bev_image.shape[1], bev_image.shape[0]))
        lidar_image = cv2.resize(lidar_image, (bev_image.shape[1], bev_image.shape[0]))

        new_tl_bev_image = self.traffic_lights.get_bev_image(tl_bev_image, tl_state, size_x=bev_image.shape[1], size_y=bev_image.shape[0])
        image = np.concatenate((roach_image, bev_image, tl_image, new_tl_bev_image, masks, lidar_image), axis=1) # ori_render), axis=1)

        #cv2.imwrite("image.png",self.change_r2b(self.image))



        self.image = image #self.change_r2b(image)#input_data['front'][1]#

        self.count_agent += 1

        if (self.plant_dataset_collection and self.get_control != None):
            control = self.get_control()
            self.set_location_of_ego_vehicle()


        if self.roach_control_label:
            control = self.run_roach(obs_dict, prev_control=self.prev_control)
            min_speed_correction = False
            if (not stop_label_obj or not stop_label) and not (control.brake > 0.0) and (str(current_wp[1]) ==
                                                                                         'RoadOption.LANEFOLLOW' or
                                                                                         str(current_wp[1]) ==
                                                                                         'RoadOption.STRAIGHT' or
                                                                                         str(current_wp[1]) ==
                                                                                        'RoadOption.LANECHANGE'):
                min_speed_correction = True
                control = self.min_speed_rule(control)
            #self.write_labels(self.image, 'prediction_stop_label: ' + str(prediction_stop_label), x=50, y=50)
            self.write_labels(self.image, 'score_list_lcte: ' + str(score_list_lcte), x=50, y=50)
            self.write_labels(self.image,'pred_list: ' + str(pred_list),x=50,y=100)
            self.write_labels(self.image,'stop_label_tl: ' + str(stop_label_tl),x=50,y=150)
            #self.write_labels(self.image,'stop_label_obj: ' + str(stop_label_obj),x=50,y=200)
            self.write_labels(self.image, 'nonstop_prediction_label: ' + str(nonstop_prediction_label), x=50, y=250)

            #control = self.rule_based()
            steer = self.stabilize(current_wp, control)

            if stop_label or stop_label_tl or stop_label_obj or walker_stop or prediction_stop_label:
                control = carla.VehicleControl(throttle=0.0, steer=steer, brake=1.0)
            else:
                #throttle = min(control.throttle,0.2)
                control = carla.VehicleControl(throttle=control.throttle, steer=steer, brake=0.0)

            self.write_labels(self.image, 'tl_state: ' + str(tl_state), x=50, y=200)
            self.write_labels(self.image, 'min_speed_correction: ' + str(min_speed_correction), x=50, y=550)
            #self.write_labels(self.image, 'number_of_walker: ' + str(number_of_walker), x=50, y=600)
        else:
            control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
            self.wp_based_control()

        self.write_labels(self.image, 'scenario_instance_name: ' + str(self.scenario_instance_name), x=50, y=300)
        self.write_labels(self.image, 'is_there_obstacle: ' + str(self.chauffeurnet.is_there_obstacle), x=50, y=350)
        self.write_labels(self.image, 'diff_location: ' + str(diff_location), x=50, y=400)
        self.write_labels(self.image, 'diff_yaw_degree: ' + str(diff_yaw_degree), x=50, y=450)


        #self.image = tl_image
        cv2.imwrite(self.current_path+"/image.png", self.image)

        if self.save_control_label and self.save_control != None:
            button_press = 0
            self.save_control(control, ego_loc, ego_rot, ego_speed, button_press, self._vehicle.id)

        self.prev_control = control
        self.prev_scenario_instance_name = scenario_instance_name

        return control

    def set_location_of_ego_vehicle(self,scenario_loc=None,scenario_rot=None):
        # Step 1: Define the new location and rotation
        #self.log_index = 0
        print("loc: ",self.log_ego_loc[self.log_index][0],self.log_ego_loc[self.log_index][1],self.log_ego_loc[self.log_index][2])
        new_location = carla.Location(x=self.log_ego_loc[self.log_index][0], y=self.log_ego_loc[self.log_index][1], z=self.log_ego_loc[self.log_index][2])  # Example coordinates self.log_ego_loc
        new_rotation = carla.Rotation(pitch=self.log_ego_rot[self.log_index][0], yaw=self.log_ego_rot[self.log_index][1], roll=self.log_ego_rot[self.log_index][2])#self._vehicle.get_transform().rotation#carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)  # Example orientation

        # Step 2: Create a new Transform
        new_transform = carla.Transform(new_location, new_rotation)
        new_speed = carla.Vector3D(x=self.log_ego_speed[self.log_index][0], y=self.log_ego_speed[self.log_index][1], z=self.log_ego_speed[self.log_index][2])

        # Step 3: Set the vehicle's Transform
        self._vehicle.set_transform(new_transform)
        self._vehicle.set_target_velocity(new_speed)


        self.log_index += 1

    def set_log_traffic_tick(self, log_traffic_manager_tick):
        self.log_traffic_manager_tick = log_traffic_manager_tick

    def set_save_func(self, save_control,file_name_without_extension):
        self.save_control = save_control
        self.file_name_without_extension = file_name_without_extension

    def save_control_func(self, get_control,ego_loc,ego_rot,ego_speed):
        self.get_control = get_control
        self.log_ego_loc = ego_loc
        self.log_ego_rot = ego_rot
        self.log_ego_speed = ego_speed
        self.log_index = 0


    def change_r2b(self,image):
        new_image = copy.deepcopy(image)
        swap_channel = copy.deepcopy(new_image[:, :, 0])
        new_image[:, :, 0] = new_image[:, :, 2]
        new_image[:, :, 2] = swap_channel
        return new_image

    def im_render(self,render_dict, im_birdview=None):
        if type(im_birdview) == type(None):
            im_birdview = render_dict['im_render']
        else:
            im_birdview = im_birdview
        h, w, c = im_birdview.shape
        im = np.zeros([h, w*4, c], dtype=np.uint8)
        im[:h, :w] = im_birdview

        for i, txt in enumerate(render_dict['info']):
            im = cv2.putText(im, txt, (w, (i+2)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return im

    def wp_based_control(self):
        if self.wp_based_index > 10:
            current_wp, _ = self.wp_list[self.wp_based_index]

            ego_location = self._vehicle.get_transform().location
            new_location = carla.Location(x=current_wp.transform.location.x, y=current_wp.transform.location.y, z=self.initial_ego_location_z)  # Example coordinates self.log_ego_loc
            new_rotation = current_wp.transform.rotation #carla.Rotation(pitch=self.log_ego_rot[self.log_index][0], yaw=self.log_ego_rot[self.log_index][1], roll=self.log_ego_rot[self.log_index][2])#self._vehicle.get_transform().rotation#carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)  # Example orientation

            # Step 2: Create a new Transform
            new_transform = carla.Transform(new_location, new_rotation)

            self._vehicle.set_transform(new_transform)
        else:
            self.initial_ego_location_z = self._vehicle.get_transform().location.z

        self.wp_based_index += 1

    def rule_based(self):
        if not self._agent:

            # Search for the ego actor
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break

            if not hero_actor:
                return carla.VehicleControl()

            # Add an agent that follows the route to the ego
            self._agent = BasicAgent(hero_actor, 30)

            plan = []
            prev_wp = None
            for transform, _ in self._global_plan_world_coord:
                wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                if prev_wp:
                    plan.extend(self._agent.trace_route(prev_wp, wp))
                prev_wp = wp

            self._agent.set_global_plan(plan)

            return carla.VehicleControl()

        else:
            return self._agent.run_step()

    def get_image(self):
        try:
            return self.image#np.zeros((396,704,3))
        except:
            return np.zeros((200,1000,3))#np.zeros((200,400,3))


    def create_render_dict_for_llm(self,message):
        #render_dict['info'] = {'chat gpt: ' + str(high_level_action)}
        sentence = ['Summarize Scene: ',]
        if not message == ['no reply']:
            sentence = self.seperate_lines(sentence, message[0])
        sentence.append(' ')
        sentence.append('Final decision: ')

        if not message == ['no reply']:
            sentence.append(message[1])
        sentence.append(' ')

        sentence.append('Explain decision: ')
        if not message == ['no reply']:
            sentence = self.seperate_lines(sentence, message[2])

        merged_dict = {'info':sentence}
        return merged_dict

    def seperate_lines(self, sentence, message):
        first_message = np.array(list(message))
        for index in range(int(len(first_message) / 100) + 1):
            try:
                sentence.append (''.join(first_message[100 * index:100 * (index + 1)]))
            except:
                sentence.append (''.join(first_message[100 * index:]))
                break

        return sentence

    def min_speed_rule(self, control):
        all_vehicles = CarlaDataProvider.get_all_actors().filter('vehicle*')
        background_vehicles = [v for v in all_vehicles if v.attributes['role_name'] == 'background']
        velocity = CarlaDataProvider.get_velocity(self._vehicle)

        if background_vehicles:
            frame_mean_speed = 0
            for vehicle in background_vehicles:
                frame_mean_speed += CarlaDataProvider.get_velocity(vehicle)
            frame_mean_speed /= len(background_vehicles)

            self.frame_mean_speed = frame_mean_speed + 1

        #print("velocity:",velocity,"self.frame_mean_speed:",self.frame_mean_speed)
        if velocity < self.frame_mean_speed + 1:
            control = carla.VehicleControl(throttle=control.throttle+0.5, steer=control.steer, brake=0)
        else:
            pass

        return control

    def find_pom(self, altitude=40, fov=70, image_width=1600, image_height=1200):

        # Calculate the visible ground area
        # We use the field of view and altitude to calculate this
        # For a camera pointed directly downwards, the width and height of the ground area can be approximated
        # as follows (assuming a flat ground plane):
        visible_width = 2 * (altitude * math.tan(math.radians(fov / 2)))
        visible_height = visible_width * (image_height / image_width)

        # Calculate meters per pixel
        meters_per_pixel_x = visible_width / image_width
        meters_per_pixel_y = visible_height / image_height

        return meters_per_pixel_x, meters_per_pixel_y

    def write_labels(self, image, text, x, y):
        # Specify the text you want to write
        # Position where you want the text to appear (bottom-left corner of the text)
        position = (x, y)  # Change this according to where you want the text

        # Font type (check OpenCV documentation for more types)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Font scale (size of the font)
        font_scale = 1

        # Color of the text in BGR (Blue, Green, Red)
        color = (0, 0, 255)  # This is white

        # Thickness of the text
        thickness = 2

        # Use cv2.putText() method to add text
        cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)



    def change_weather_and_time(self):
        time_of_day = ['daytime_1', 'daytime_2', 'daytime_3', 'daytime_4', 'daytime_5','daytime_6', 'nighttime_0',
                       'nighttime_1', 'nighttime_2', 'nighttime_3', 'nighttime_4', 'nighttime_5', 'nighttime_6']

        self.change_weather(self._world, self.weather_class_list[self.weather_change_index % len(self.weather_class_list)])
        self.change_time_of_day(self._world, time_of_day[self.weather_change_index % len(time_of_day)])
        if self.weather_change_count % 1 == 0:
            self.weather_change_index += 1

        self.weather_change_count += 1
        time.sleep(0.1)

    def change_weather(self, world, weather):
        world.set_weather(weather)


    def change_time_of_day(self, world, time_of_day):
        # Adjust the sun's altitude angle to simulate different times of day
        # Sunrise/sunset ~0.0, Noon ~90.0, Midnight ~-90.0
        if time_of_day == 'daytime_1':
            sun_altitude_angle = 20.0
        elif time_of_day == 'daytime_2':
            sun_altitude_angle = 45.0
        elif time_of_day == 'daytime_3':
            sun_altitude_angle = 90.0
        elif time_of_day == 'daytime_4':
            sun_altitude_angle = 120.0
        elif time_of_day == 'daytime_5':
            sun_altitude_angle = 160.0
        elif time_of_day == 'daytime_6':
            sun_altitude_angle = 160.0
        elif time_of_day == 'nighttime_0':
            sun_altitude_angle = 0.0
        elif time_of_day == 'nighttime_1':
            sun_altitude_angle = -20.0
        elif time_of_day == 'nighttime_2':
            sun_altitude_angle = -45.0
        elif time_of_day == 'nighttime_3':
            sun_altitude_angle = -90.0
        elif time_of_day == 'nighttime_4':
            sun_altitude_angle = -120.0
        elif time_of_day == 'nighttime_5':
            sun_altitude_angle = -160.0
        elif time_of_day == 'nighttime_6':
            sun_altitude_angle = -180.0
        else:
            print("Unknown time of day")
            return

        weather = world.get_weather()
        weather.sun_altitude_angle = sun_altitude_angle
        world.set_weather(weather)


    def read_weather_class(self, pickle_file_path):
        data = []
        import pickle
        with open(pickle_file_path, 'rb') as file:
            while True:
                try:
                    # Attempt to load the next object in the file
                    object = pickle.load(file)
                    data.append(object)
                except EOFError:
                    # End of file reached
                    break

        self.weather_class_list = []

        for weather in data:
            new_weather_class = carla.WeatherParameters(cloudiness=float(weather['cloudiness']),
                                    precipitation=float(weather['precipitation']),
                                    precipitation_deposits=float(weather['precipitation_deposits']),
                                    wind_intensity=float(weather['wind_intensity']),
                                    sun_azimuth_angle=float(weather['sun_azimuth_angle']),
                                    sun_altitude_angle=float(weather['sun_altitude_angle']),
                                    fog_density=float(weather['fog_density']),
                                    wetness=float(weather['wetness']))
            self.weather_class_list.append(new_weather_class)

        asd = 0


