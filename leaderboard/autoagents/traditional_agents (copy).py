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

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
#from leaderboard.autoagents.traditional_agents_files.perception.traffic_lights import Traffic_Lights
#from leaderboard.autoagents.traditional_agents_files.utils.get_speed_yaw import Get_Speed_Yaw
#from leaderboard.autoagents.traditional_agents_files.perception.object_detection import Object_Detection

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.planning import Planning

from leaderboard.autoagents.traditional_agents_files.carla_gym.utils.traffic_light import TrafficLightHandler
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.agent_utils.coordinate_utils import preprocess_compass

from accelerate import infer_auto_device_map, init_empty_weights



from leaderboard.autoagents.traditional_agents_files.utils.manuel_control import Manuel_Control

import json
import numpy as np
import cv2
import datetime
import h5py
import math
import os
import glob




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
        #self.obj_detection = Object_Detection(detection_config)

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
        #self.traffic_lights = Traffic_Lights(config)

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

        self.manuel_control = Manuel_Control()

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

        pixels_per_meter, _world_offset = self.chauffeurnet.attach_ego_vehicle(vehicle)

        self.planning = Planning(ego_vehicles=vehicle,_world=_world)
        self.planning.set_info(pixels_per_meter, _world_offset)

        self.K = self.build_projection_matrix(704, 396, 70)
        # register traffic lights
        TrafficLightHandler.reset(self._world)
        #return self.plant_sensor.setup_sensors(vehicle, _world)


    def get_sensor_list(self):
        return self.plant_sensor._sensors_list

    def set_client(self, client):
        self._client = client

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
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.70079118954, 'y': 0.0159456324149, 'z': 1.51095763913, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1600, 'height': 1200, 'fov': 120, 'id': 'front'},
        ]

        #intrinsics, extrinsics, transforms_val, carla_to_lss = self.obj_detection.get_parameters()

        if len(self.sensors_list)==0:
            """object_detection_sensor = self.obj_detection.four_sensors()#.sensors()
            with open('object_detection_sensor.json','w') as json_file:
                json.dump(object_detection_sensor,json_file,indent=len(object_detection_sensor))"""


            self.sensors_list = self.plant_sensor.sensors + sensors #+ object_detection_sensor #.append(sensors[0])

            self.image_count = 0#"""


        #self.traffic_lights.set_parameters(intrinsics, extrinsics, transforms_val, carla_to_lss)#"""

        return self.sensors_list#sensors#

    def set_camera_sensor(self, sensor):
        self.camera = sensor

    def set_global_plan_wp_list(self, wp_list, plant_global_plan_gps, plant_global_plan_world_coord):
        self.wp_list = wp_list
        self.plant_global_plan_gps = plant_global_plan_gps
        self.plant_global_plan_world_coord = plant_global_plan_world_coord
        self.negative_sample_read_count = 0
        self.frame_mean_speed = 0

        try:
            self.close_file()
        except:
            pass
        self.previous_bb = None

        #self.create_file()


    def run_planning_model(self, input_data, plant_boxes,  bbox, ego_motion):

        light_hazard = False

        control, pred_wp, target_point, keep_vehicle_ids, keep_vehicle_attn = self.planning(bbox=bbox,input_data=input_data,
            ego_actor=self._vehicle, global_plan_gps=self.plant_global_plan_gps,
            global_plan_world_coord=self.plant_global_plan_world_coord, light_hazard=light_hazard, plant_boxes=plant_boxes, ego_motion=ego_motion, plant_dataset_collection=self.plant_dataset_collection)

        return control

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

    def set_scenario_gt(self, scenario_instance_name):
        self.scenario_instance_name = scenario_instance_name

    def run_step(self, world, input_data, timestamp):
        """
        Execute one step of navigation.
        """

        #self.traffic_lights.deneme_traffic_lights()
        #self.image = np.concatenate((input_data['front_left'][1],input_data['front'][1],input_data['front_right'][1], input_data['back_right'][1],input_data['back'][1],input_data['back_left'][1]), axis=1)
        #tl_bbox, front_image = self.traffic_lights(input_data)
        self.det_count += 1

        """if self.image_count % 5 == 0:
            for cam in self.obj_detection.carla_to_lss.cams:
                cv2.imwrite(self.center_line_path+'/'+cam+'_'+str(self.image_count)+".png", input_data[cam][1])
        self.image_count += 1"""

        #_, image = self.traffic_lights(input_data)
        #masks, plant_boxes, bbox, orientation_angle_list = self.obj_detection(input_data)
        current_speed = 10

        #obs_dict, M_warp_list, prev_bbox_list, img, close_points_count, mean_alt, mean_vel, is_there_obstacle, self.plant_global_plan_gps,self.plant_global_plan_world_coord = self.chauffeurnet.get_observation(input_data, self.wp_list,self.plant_global_plan_gps,self.plant_global_plan_world_coord, current_speed, math.degrees(input_data['imu'][1][-1]), self.high_level_action)
        #ori_render = copy.deepcopy(obs_dict['rendered'])

        #speed_array, angle_array, rendered_bev_image, render_dict, color_name_list = self.get_attribute(bbox, prev_bbox_list, obs_dict['ego_motion'],M_warp_list, input_data['imu'][1][-1],obs_dict['rendered'],orientation_angle_list)

        """input_data.update({'predicted_speed': speed_array})
        input_data.update({'predicted_yaw': angle_array})
        input_data.update({'color_name_list': color_name_list})
        input_data.update({'bbox': bbox})"""

        #scene_info = self.chauffeurnet.get_high_level_action(input_data)
        #log_reply_list, self.high_level_action = self.llm_model(scene_info)

        #control = self.run_planning_model(input_data, plant_boxes=None, bbox=None, ego_motion=None)#obs_dict['ego_motion'])

        #render_dict = self.get_attribute.update_render_dict(render_dict, control,close_points_count, mean_alt, mean_vel, is_there_obstacle)
        #render_dict = {'info':object_message}
        #copy_rendered_bev_image = copy.deepcopy(rendered_bev_image)
        #copy_rendered_bev_image[img] = (255,0,0)

        #render_dict = self.create_render_dict_for_llm(log_reply_list)
        #rendered_bev_image = self.im_render(render_dict, im_birdview=ori_render)

        #self.image = np.concatenate((copy_rendered_bev_image, front_image, rendered_bev_image),axis=1) cv2.resize(input_data['front_left'][1], (200, 200))
        self.image = cv2.resize(input_data['front'][1], (200, 200))[:,:,:3]#np.concatenate((cv2.resize(input_data['front'][1], (200, 200))[:,:,:3], cv2.resize(input_data['front_right'][1], (200, 200))[:,:,:3], cv2.resize(input_data['back'][1], (200, 200))[:,:,:3]), axis=1)# ori_render), axis=1)

        #cv2.imwrite("image.png",self.change_r2b(self.image))
        cv2.imwrite("image.png",self.image)
        self.count_agent += 1

        #self.check_traffic_lights(input_data)"""
        """try:
            control = self.get_gt_control()
            control_given_by = "gt"
        except:"""

        speed = self._vehicle.get_velocity().length()

        control = self.manuel_control(input_data, ego_vel=speed, bev_data=None)

        #control = self.rule_based()
        ego_loc = self._vehicle.get_location()
        # control = self.min_speed_rule(control)

        self.save_control(control, ego_loc)
        if self.plant_dataset_collection and self.get_control != None:
            control = self.get_control()

        #print("self.scenario_instance_name: ",self.scenario_instance_name, control_given_by,"control:",control)#self.gt_log_dict['records'][self.scenario_instance_name]['control']['brake']

        print("control: ",control)

        return control

    def set_save_func(self, save_control):
        self.save_control = save_control

    def save_control_func(self, get_control):
        self.get_control = get_control


    def get_gt_control(self):
        scenario_name = ''.join(list(self.scenario_instance_name)[:-2]) if ''.join(list(self.scenario_instance_name)[:-2]) in self.gt_log_dict.keys() else self.scenario_instance_name
        #while not self.start_log and self.gt_log_dict[scenario_name]['records'][self.log_index]['control']['steer'] == 0.0 and self.gt_log_dict[scenario_name]['records'][self.log_index]['control']['throttle'] == 0.0:
        #    self.log_index += 1


        brake = self.gt_log_dict[scenario_name]['records'][self.log_index]['control']['brake']
        steer = self.gt_log_dict[scenario_name]['records'][self.log_index]['control']['steer']
        throttle = self.gt_log_dict[scenario_name]['records'][self.log_index]['control']['throttle']
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

        self.log_index += 1
        return control


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

    def check_traffic_lights(self, input_data,  draw=True):
        color, stop_bounding_boxes, tl_corner_bb = self.chauffeurnet.get_is_there_list_parameter()
        compass = input_data['imu'][1][-1]#preprocess_compass(input_data['imu'][1][-1])
        print("*"*50,"color:",color)
        current_speed = self._vehicle.get_velocity().length()
        vehicle, _ = self._vehicle, self._world


        ego_loc = vehicle.get_location()
        ego_loc = np.array([ego_loc.x, ego_loc.y, ego_loc.z])

        # Remember the edge pairs
        edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

        # Retrieve the first image
        #world.tick()
        self.img_id = datetime.date.today().month,datetime.date.today().day,datetime.datetime.now().time().hour, datetime.datetime.now().time().minute,datetime.datetime.now().time().second, datetime.datetime.now().time().microsecond
        image = input_data['front']
        img = np.reshape(np.copy(image[1]), (image[1].shape[0], image[1].shape[1], 4))

        bounding_box_set = self._world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        # Get the camera matrix
        world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())
        bounding_box_set = bounding_box_set
        bb_list = []
        draw = False
        positive_sample = False
        distance = -1
        for index, bb in enumerate(bounding_box_set):
            points = np.ones(4)*(-1)
            distance = bb.location.distance(vehicle.get_transform().location)

            if 5.5 < distance and distance < 40 and abs(bb.rotation.yaw % 360 - math.degrees(compass)) < 50 and current_speed > 1:
                self.negative_sample_read_count = 0

            # Filter for distance from ego vehicle and abs(bb.rotation.yaw%360-math.degrees(compass))<50
            if 5.5 < distance and distance < 40 and abs(bb.rotation.yaw%360-math.degrees(compass))<50 and current_speed > 1 and color != 'unknown' and color != 'stop_sign':#np.sum(self.previous_loc) != np.sum(ego_loc): #and color != 'unknown' and color != 'stop_sign'
                draw = True
                positive_sample = True
                img = self.save_bb(vehicle, bb, edges, world_2_camera, img, draw, image, distance, color, current_speed,
                            ego_loc)
                cv2.imwrite('tl_dataset/ImageWindowName' + str(self.tl_dataset_count) + '.png', img)
                self.tl_dataset_count += 1

        points = np.ones(4) * (-1)
        if (self.negative_sample_count<30 or self.positive_sample_count > self.negative_sample_count) and not positive_sample and current_speed > 1 and self.negative_sample_read_count > 15:
            draw = True
            org_img = np.reshape(np.copy(image[1]), (image[1].shape[0], image[1].shape[1], 4))
            self.save_arrays(self.img_id, self._carla_map, org_img, points, distance, color, current_speed)
            cv2.imwrite('tl_dataset/ImageWindowName' + str(self.tl_dataset_count) + '.png', img)
            self.tl_dataset_count += 1
            self.negative_sample_count += 1


        self.camera_image = img
        self.previous_loc = ego_loc
        #print("current_speed:",current_speed)

        self.negative_sample_read_count += 1
        if positive_sample:
            self.negative_sample_read_count = 0
            self.positive_sample_count += 1

        return img

    def save_bb(self, vehicle, bb, edges, world_2_camera, img, draw, image, distance, color, current_speed, ego_loc):
        # Calculate the dot product between the forward vector
        # of the vehicle and the vector between the vehicle
        # and the bounding box. We threshold this dot product
        # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
        forward_vec = vehicle.get_transform().get_forward_vector()
        ray = bb.location - vehicle.get_transform().location
        if np.dot(np.array([forward_vec.x, forward_vec.y, forward_vec.z]), np.array([ray.x, ray.y, ray.z])) > 1:
            # Cycle through the vertices
            verts = [v for v in bb.get_world_vertices(carla.Transform())]

            point_list = []
            for edge in edges:
                # Join the vertices into edges
                p1 = self.get_image_point(verts[edge[0]], self.K, world_2_camera)
                p2 = self.get_image_point(verts[edge[1]], self.K, world_2_camera)
                point_list.append(p1)
                point_list.append(p2)
                # Draw the edges into the camera output
            x_min, x_max, y_min, y_max = np.min(np.array(point_list)[:, 0]), np.max(np.array(point_list)[:, 0]), np.min(
                np.array(point_list)[:, 1]), np.max(np.array(point_list)[:, 1])

            if draw:
                cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0, 0, 255, 255), 1)
                cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)
                cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), (0, 0, 255, 255), 1)
                cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)

            points = np.array([x_min, y_min, x_max, y_max])
            org_img = np.reshape(np.copy(image[1]), (image[1].shape[0], image[1].shape[1], 4))
            if not (1-(points>0)).any():
                self.save_arrays(self.img_id, self._carla_map, org_img, points, distance, color, current_speed)
            self.draw_count += 1


        return img

    def save_arrays(self, img_id, _carla_map, image, BB, distance, color, current_speed):
        BB = np.array(BB)
        img_id = np.array(img_id)
        _carla_map = 12#int(''.join(list(_carla_map)[-2:]))

        if color == "stop_sign":
            color = 4
        elif color == 'red':
            color = 3
        elif color == 'yellow':
            color = 2
        elif color == 'green':
            color = 1
        elif color == 'unknown':
            color = 6

        if self.counter == 0:
            self.h5_file.create_dataset("img_id", data=np.expand_dims(img_id, axis=0),
                                        chunks=True, maxshape=(None, np.array(img_id).shape[0]))
            self.h5_file.create_dataset("_carla_map", data=np.expand_dims([_carla_map], axis=0),
                                        chunks=True, maxshape=(None, 1))
            self.h5_file.create_dataset("image", data=np.expand_dims(image, axis=0),
                                        chunks=True, maxshape=(None, image.shape[0], image.shape[1], image.shape[2]))
            self.h5_file.create_dataset("2DBB",
                                        data=np.expand_dims(BB, axis=0),
                                        chunks=True, maxshape=(None, 4))
            self.h5_file.create_dataset("distance",
                                        data=np.expand_dims([distance], axis=0),
                                        chunks=True, maxshape=(None, 1))
            self.h5_file.create_dataset("color",
                                        data=np.expand_dims([color], axis=0),
                                        chunks=True, maxshape=(None, 1))
            self.h5_file.create_dataset("speed",
                                        data=np.expand_dims([current_speed], axis=0),
                                        chunks=True, maxshape=(None, 1))

        else:  # append next arrays to dataset
            self.h5_file["img_id"].resize((self.h5_file["img_id"].shape[0] + 1), axis=0)
            self.h5_file["img_id"][-1] = img_id
            self.h5_file["_carla_map"].resize((self.h5_file["_carla_map"].shape[0] + 1), axis=0)
            self.h5_file["_carla_map"][-1] = _carla_map
            self.h5_file["image"].resize((self.h5_file["image"].shape[0] + 1), axis=0)
            self.h5_file["image"][-1] = image
            self.h5_file["2DBB"].resize((self.h5_file["2DBB"].shape[0] + 1), axis=0)
            self.h5_file["2DBB"][-1] = BB
            self.h5_file["distance"].resize((self.h5_file["distance"].shape[0] + 1), axis=0)
            self.h5_file["distance"][-1] = distance
            self.h5_file["color"].resize((self.h5_file["color"].shape[0] + 1), axis=0)
            self.h5_file["color"][-1] = color
            self.h5_file["speed"].resize((self.h5_file["speed"].shape[0] + 1), axis=0)
            self.h5_file["speed"][-1] = current_speed

        self.counter += 1

    def close_file(self):
        self.h5_file.close()


    def create_file(self):

        self.counter = 0
        self.vehicle_dic_dataset = {}
        self.vehicle_index_dataset = 1

        self.counter = 0
        episode_number = 135
        file = 'episode_' + str(episode_number).zfill(3) + '.h5'
        file_exist = True

        while file_exist:
            episode_number += 1
            file = 'episode_' + str(episode_number).zfill(3) + '.h5'
            file_exist = os.path.exists(file)

        #file = 'image_bb_'+ str(self.file_count) + '_.h5'
        print("file:",file)
        self.h5_file = h5py.File(file, 'a')

    def get_image_point(self, loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

    def build_projection_matrix(self, w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

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

        print("velocity:",velocity,"self.frame_mean_speed:",self.frame_mean_speed)
        if velocity < self.frame_mean_speed:
            control = carla.VehicleControl(throttle=control.throttle+0.5, steer=control.steer, brake=0)
            #control.throttle += 10

        return control





