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

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.agent_utils.filter_functions import *
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.agent_utils.coordinate_utils import preprocess_compass, inverse_conversion_2d
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.agent_utils.explainability_utils import *

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.data_agent_boxes import DataAgent
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.training.PlanT.dataset import generate_batch, split_large_BB
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.training.PlanT.lit_module import LitHFLM

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.nav_planner import extrapolate_waypoint_route
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.nav_planner import RoutePlanner_new as RoutePlanner
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.scenario_logger import ScenarioLogger

import pyproj

import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_entry_point():
    return 'PlanTAgent'

SAVE_GIF = os.getenv("SAVE_GIF", 'False').lower() in ('true', '1', 't')

class PlanTAgent(DataAgent):

    def save_score(self, name, file_name_without_extension, save_files_name, score_composed, score_route, score_penalty):
        super().save_score(name, file_name_without_extension, save_files_name, score_composed, score_route, score_penalty)

    def setup(self, path_to_conf_file, route_index=None, cfg=None, exec_or_inter=None):
        self.cfg = cfg
        self.exec_or_inter = exec_or_inter

        path_to_conf_file = 'autoagents/traditional_agents_files/carla_gym/core/obs_manager/birdview/planning/checkpoints/PlanT/3x/PlanT_medium/log'
        #path_to_conf_file = 'autoagents/traditional_agents_files/carla_gym/core/obs_manager/birdview/planning/checkpoints/PlanT/3x/overfitting/log'
        LOAD_CKPT_PATH = 'autoagents/traditional_agents_files/carla_gym/core/obs_manager/birdview/planning/checkpoints/PlanT/3x/PlanT_medium/checkpoints/epoch=047.ckpt'#tugrul
        #LOAD_CKPT_PATH = 'autoagents/traditional_agents_files/carla_gym/core/obs_manager/birdview/planning/checkpoints/PlanT/3x/overfitting/checkpoints/last.ckpt'#tugrul

        path_exist = os.path.exists(path_to_conf_file)
        if not path_exist:
            path_to_conf_file = 'leaderboard' + '/' + path_to_conf_file
            LOAD_CKPT_PATH = 'leaderboard' + '/' + LOAD_CKPT_PATH

        self.sensors_info = [{'type': 'sensor.opendrive_map', 'reading_frequency': 1e-06, 'id': 'hd_map'}, {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'speed'}, {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.05, 'id': 'imu'}, {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.01, 'id': 'gps'}]


        # first args than super setup is important!
        args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
        self.args = json.load(args_file)
        args_file.close()
        self.cfg_agent = OmegaConf.create(self.args)

        super().setup(path_to_conf_file, route_index, cfg, exec_or_inter)

        print(f'Saving gif: {SAVE_GIF}')

        
        # Filtering
        self.points = MerweScaledSigmaPoints(n=4,
                                            alpha=.00001,
                                            beta=2,
                                            kappa=0,
                                            subtract=residual_state_x)
        self.ukf = UKF(dim_x=4,
                    dim_z=4,
                    fx=bicycle_model_forward,
                    hx=measurement_function_hx,
                    dt=1/self.frame_rate,
                    points=self.points,
                    x_mean_fn=state_mean,
                    z_mean_fn=measurement_mean,
                    residual_x=residual_state_x,
                    residual_z=residual_measurement_h)

        # State noise, same as measurement because we
        # initialize with the first measurement later
        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        # Measurement noise
        self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
        # Used to set the filter state equal the first measurement
        self.filter_initialized = False
        # Stores the last filtered positions of the ego vehicle.
        # Used to realign.
        self.state_log = deque(maxlen=2)


        # exec_or_inter is used for the interpretability metric
        # exec is the model that executes the actions in carla
        # inter is the model that obtains attention scores and a ranking of the vehicles importance
        """if exec_or_inter is not None:
            if exec_or_inter == 'exec':
                LOAD_CKPT_PATH = cfg.exec_model_ckpt_load_path
            elif exec_or_inter == 'inter':
                LOAD_CKPT_PATH = cfg.inter_model_ckpt_load_path
        else:
            LOAD_CKPT_PATH = cfg.model_ckpt_load_path"""

        print(f'Loading model from {LOAD_CKPT_PATH}')
        assert os.path.exists(LOAD_CKPT_PATH)

        if Path(LOAD_CKPT_PATH).suffix == '.ckpt':
            #self.net = LitHFLM.load_from_checkpoint(LOAD_CKPT_PATH)

            try:
                self.net = LitHFLM.load_from_checkpoint(LOAD_CKPT_PATH)
            except:
                with open('pretrained_plant_model.pkl', 'rb') as file:
                    self.net = pickle.load(file)#tugrul !!!!
                LOAD_CKPT_PATH = '/workspace/tg22/plant_pretrained_model/epoch=150.ckpt'
                self.net.load_state_dict(torch.load(LOAD_CKPT_PATH)['state_dict'])
                asd = 0


                #self.net.model.load_state_dict(model.state_dict())

            asd = 0





        else:
            raise Exception(f'Unknown model type: {Path(LOAD_CKPT_PATH).suffix}')
        self.net.cuda(2)
        self.net.eval()
        self.scenario_logger = False
        self.log_path = None
        self.cfg_viz = 0

        if self.log_path is not None:
            self.log_path = Path(self.log_path) / route_index
            Path(self.log_path).mkdir(parents=True, exist_ok=True)   
                 
            self.scenario_logger = ScenarioLogger(
                save_path=self.log_path, 
                route_index=self.route_index,
                logging_freq=self.save_freq,
                log_only=False,
                route_only=False, # with vehicles and lights
                roi = self.detection_radius+10,
            )

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

        if self.scenario_logger:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider # privileged
            self._vehicle = CarlaDataProvider.get_hero_actor()
            self.scenario_logger.ego_vehicle = self._vehicle
            self.scenario_logger.world = self._vehicle.get_world()
            
            vehicle = CarlaDataProvider.get_hero_actor()
            self.scenario_logger.ego_vehicle = vehicle
            self.scenario_logger.world = vehicle.get_world()
        
        if SAVE_GIF == True and (self.exec_or_inter == 'inter'):
            self.save_path_mask = f'viz_img/{self.route_index}/masked'
            self.save_path_org = f'viz_img/{self.route_index}/org'
            Path(self.save_path_mask).mkdir(parents=True, exist_ok=True)
            Path(self.save_path_org).mkdir(parents=True, exist_ok=True)

    def _world_to_pixel(self, location_x, location_y, projective=False):
        """Converts the world coordinates to pixel coordinates"""
        x = self._pixels_per_meter * (location_x - self._world_offset[0])
        y = self._pixels_per_meter * (location_y - self._world_offset[1])

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p

    def set_m_warp(self, M_warp):
        self.M_warp = M_warp

    def set_info(self,_pixels_per_meter,_world_offset):
        self._pixels_per_meter = _pixels_per_meter
        self._world_offset = _world_offset

    def sensors(self):
        result = super().sensors()
        return result

    def set_dummy_target_location(self, dummy_target_location):
        self.dummy_target_location = dummy_target_location


    def tick(self, input_data):
        assert input_data != None
        result = super().tick(input_data)

        pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1][:2])

        speed = input_data['speed'][1]['speed']
        compass = preprocess_compass(input_data['imu'][1][-1])

        if not self.filter_initialized:
            self.ukf.x = np.array([pos[0], pos[1], compass, speed])
            self.filter_initialized = True

        self.ukf.predict(steer=self.control.steer,
                         throttle=self.control.throttle,
                         brake=self.control.brake)
        self.ukf.update(np.array([pos[0], pos[1], compass, speed]))
        filtered_state = self.ukf.x
        self.state_log.append(filtered_state)
        result['gps'] = pos #filtered_state[0:2]

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
        #print("target_point, result['gps'], compass:",target_point, result['gps'], compass)
        _ego_target_point = inverse_conversion_2d(target_point, result['gps'], compass)
        result['target_point'] = tuple(_ego_target_point)

        if SAVE_GIF == True and (self.exec_or_inter == 'inter'):
            result['rgb_back'] = input_data['rgb_back']
            result['sem_back'] = input_data['sem_back']

        if self.scenario_logger:
            waypoint_route = self._waypoint_planner.run_step(filtered_state[0:2])
            waypoint_route = extrapolate_waypoint_route(waypoint_route,
                                                        10)
            route = np.array([[node[0][1], -node[0][0]] for node in waypoint_route]) \
                [:10]
            # Logging
            self.scenario_logger.log_step(route)

        return result

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
        mask_vehicle = np.zeros((200,200)).astype(np.uint8)
        mask_arrow = np.zeros((200,200)).astype(np.uint8)

        # Convert center to integer coordinates for drawing

        width, height = bbox[0]*3, bbox[1]*3
        top_left = (int(center[1] - width / 2), int(center[0] - height / 2))
        bottom_right = (int(center[1] + width / 2), int(center[0] + height / 2))


        # Draw the rectangle (bounding box)
        cv2.rectangle(mask_vehicle,bottom_right, top_left, (255), 2)  # Red box

        center = (int(center[1]), int(center[0]))

        # Calculate the end point of the orientation line (arrow)
        line_length = 10 #box_size[1] // 2
        line_length = min(max(velocity*line_length,line_length),line_length*2)
        end_point = (int(center[0] - line_length * np.cos(orientation_rad+(np.pi/2))),
                    int(center[1] - line_length * np.sin(orientation_rad+(np.pi/2))))

        # Draw the orientation arrow
        cv2.arrowedLine(mask_arrow, center, end_point, (255), arrow_thick)  # Blue arrow


        # Display velocity
        #font = cv2.FONT_HERSHEY_SIMPLEX
        """cv2.putText(image, f'Vel: {velocity} m/s', (center[0], int(center[1] - box_size[1] // 2 - 10)),
                    font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)  # Green text"""

        return mask_vehicle.astype(np.bool), mask_arrow.astype(np.bool)

    def plot_bounding_box_center(self, center, width=4, height=8):
        mask = np.zeros((200,200)).astype(np.uint8)
        # Calculate the top-left corner from the center, width, and height
        top_left = (int(center[1] - width / 2), int(center[0] - height / 2))
        bottom_right = (int(center[1] + width / 2), int(center[0] + height / 2))

        # Draw the rectangle (bounding box)
        cv2.rectangle(mask, bottom_right, top_left, (255), 2)  # Blue box

        return mask

    @torch.no_grad()
    def plant_run_step(self, input_data, timestamp, ego_motion,  sensors=None,  keep_ids=None, light_hazard=False, bbox=None, plant_boxes=None, tl_list=None, stop_box=None):
        
        self.keep_ids = keep_ids

        # needed for traffic_light_hazard

        if not self.initialized:
            #assert 'hd_map' in input_data.keys()
            if ('hd_map' in input_data.keys()):
                print("*"*50,"Plant is initialized")
                self._init(input_data['hd_map'])
            else:
                self.control = carla.VehicleControl()
                self.control.steer = 0.0
                self.control.throttle = 0.0
                self.control.brake = 1.0
                print("*"*50,"not initialized")
                if self.exec_or_inter == 'inter':
                    return [], None
                return self.control, torch.zeros((1,4,2)), (0,0), [], []

        # needed for traffic_light_hazard
        _ = super()._get_brake(vehicle_hazard=0, walker_hazard=0)
        tick_data = self.tick(input_data)

        label_raw = super().get_bev_boxes(input_data=input_data, pos=tick_data['gps'])
        plant_input_image, mask_vehicle_image = self.draw_label_raw(label_raw,'gt')

        detected_input_image = np.zeros((200, 200, 3)).astype(np.uint8)
        if type(input_data['detected_boxes']) != type(None):
            label_raw_1 = super().get_bev_boxes_using_tl_lights(input_data=input_data, pos=tick_data['gps'])
            detected_input_image, detected_mask_vehicle_image = self.draw_label_raw(label_raw_1,'detection')

        if self.exec_or_inter == 'inter':
            keep_vehicle_ids = self.plant_get_control(label_raw, tick_data)
            # print(f'plant: {keep_vehicle_ids}')
            
            return keep_vehicle_ids
        elif self.exec_or_inter == 'exec' or self.exec_or_inter is None:
            self.control, pred_wp, keep_vehicle_ids, keep_vehicle_attn, plant_input_image = self.plant_get_control(
                label_raw, tick_data, mask_vehicle_image, plant_input_image)

        if light_hazard:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
            pred_wp = torch.zeros_like(pred_wp)

        return self.control, pred_wp,  tick_data['target_point'], keep_vehicle_ids, keep_vehicle_attn, plant_input_image, detected_input_image

    def draw_pred_wp(self, plant_input_image, pred_wp, width=1, height=1):
        mask_wp = np.zeros((200, 200)).astype(np.uint8)

        for center in pred_wp:
            center[0] = center[0] * (-1)
            center = center * 4 + 100

            top_left = (int(center[1] - width / 2), int(center[0] - height / 2))
            bottom_right = (int(center[1] + width / 2), int(center[0] + height / 2))

            # Draw the rectangle (bounding box)
            cv2.rectangle(mask_wp, bottom_right, top_left, (255), 2)

        plant_input_image[mask_wp.astype(np.bool)] = (0, 255, 0)

        return plant_input_image, mask_wp

    def draw_lane(self, plant_input_image, label_raw, width=1, height=1):
        mask_wp = np.zeros((200, 200)).astype(np.uint8)

        for wp in label_raw:
            if wp['class'] == 'Lane':
                center = np.array([wp['position'][0], wp['position'][1]])
                #center[0] = center[0] * (-1)
                center = center * 4 + 100

                top_left = (int(center[1] - width / 2), int(center[0] - height / 2))
                bottom_right = (int(center[1] + width / 2), int(center[0] + height / 2))

                # Draw the rectangle (bounding box)
                cv2.rectangle(mask_wp, bottom_right, top_left, (255), 2)

        plant_input_image[mask_wp.astype(np.bool)] = (0, 255, 0)

        return plant_input_image, mask_wp

    def plant_get_control(self, label_raw, input_data, mask_vehicle_image, plant_input_image):
        gt_velocity = torch.FloatTensor([input_data['speed']]).unsqueeze(0)
        input_batch = self.get_input_batch(label_raw, input_data)
        x, y, _, tp, light = input_batch
        x[0], y[0], _, tp, light = x[0].cuda(2), y[0].cuda(2), _, tp.cuda(2), light.cuda(2)
        
        print("start_measurement")
        t0 = time.time()
        _, _, pred_wp, attn_map = self.net(x, y, target_point=tp, light_hazard=light)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        print("plant_network time:",t1-t0)
        plant_input_image, mask_pred_wp = self.draw_pred_wp(plant_input_image, copy.deepcopy(pred_wp).squeeze(0))
        self.draw_lane(plant_input_image, label_raw)
        steer, throttle, brake = self.net.model.control_pid(pred_wp[:1], gt_velocity)



        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0

        if brake:
            steer *= self.steer_damping

        """if np.sum(mask_pred_wp * mask_vehicle_image) > 0:
            brake = 1.0
            steer = 0.0
            throttle = 0.0"""

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
        if exec_or_inter == 'inter':
            attn_vector = get_attn_norm_vehicles(attention_score, self.data_car, attn_map)
            keep_vehicle_ids, attn_indices, keep_vehicle_attn = get_vehicleID_from_attn_scores(self.data, self.data_car, topk, attn_vector)
            #keep_vehicle_attn = self.softmax(keep_vehicle_attn)
            if SAVE_GIF == True and (exec_or_inter == 'inter'):
                draw_attention_bb_in_carla(self._world, keep_vehicle_ids, keep_vehicle_attn, self.frame_rate_sim)
                #if self.step % 1 == 0:
                #    get_masked_viz_3rd_person(self.save_path_org, self.save_path_mask, self.step, input_data)

            #return keep_vehicle_ids, attn_indices

        return control, pred_wp, keep_vehicle_ids, keep_vehicle_attn, plant_input_image

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def get_input_batch(self, label_raw, input_data):
        sample = {'input': [], 'output': [], 'brake': [], 'waypoints': [], 'target_point': [], 'light': []}#tugrul

        if self.cfg_agent.model.training.input_ego:
            data = label_raw
        else:
            data = label_raw[1:] # remove first element (ego vehicle)

        """data_car = [[
            1., # type indicator for cars
            float(x['position'][0])-float(label_raw[0]['position'][0]),
            float(x['position'][1])-float(label_raw[0]['position'][1]),
            float(x['yaw'] * 180 / 3.14159265359), # in degrees
            float(x['speed'] * 3.6), # in km/h
            float(x['extent'][2]),
            float(x['extent'][1]),
            ] for x in data if x['class'] == 'Car'] """# and ((self.cfg_agent.model.training.remove_back and float(x['position'][0])-float(label_raw[0]['position'][0]) >= 0) or not self.cfg_agent.model.training.remove_back)]

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
                2., # type indicator for route
                float(x['position'][0])-float(label_raw[0]['position'][0]),
                float(x['position'][1])-float(label_raw[0]['position'][1]),
                float(x['yaw'] * 180 / 3.14159265359), # in degrees
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

        features = data_car+ data_radar + data_route

        sample['input'] = features #np.array(features)

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
    s=0
    max_d = 30
    size = int(max_d*pix_per_m*2)
    origin = (size//2, size//2)
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
            
            x = -vehicle['position'][1]*PIXELS_PER_METER + origin[1]
            y = -vehicle['position'][0]*PIXELS_PER_METER + origin[0]
            yaw = vehicle['yaw']* 180 / 3.14159265359
            extent_x = vehicle['extent'][2]*PIXELS_PER_METER/2
            extent_y = vehicle['extent'][1]*PIXELS_PER_METER/2
            origin_v = (x, y)
            
            if vehicle['class'] == 'Car':
                p1, p2, p3, p4 = get_coords_BB(x, y, yaw-90, extent_x, extent_y)
                if ixx == 0:
                    for ix in range(3):
                        draws[ix].polygon((p1, p2, p3, p4), outline=color[0]) #, fill=color[ix])
                    ix = 0
                else:                
                    draws[ix].polygon((p1, p2, p3, p4), outline=color[ix]) #, fill=color[ix])
                
                if 'speed' in vehicle:
                    vel = vehicle['speed']*3 #/3.6 # in m/s # just for visu
                    endx1, endy1, endx2, endy2 = get_coords(x, y, yaw-90, vel)
                    draws[ix].line((endx1, endy1, endx2, endy2), fill=color[ix], width=2)

            elif vehicle['class'] == 'Route':
                ix = 1
                image = np.array(imgs[ix])
                point = (int(x), int(y))
                cv2.circle(image, point, radius=3, color=color[0], thickness=3)
                imgs[ix] = Image.fromarray(image)
                
    for wp in pred_wp:
        x = wp[1]*PIXELS_PER_METER + origin[1]
        y = -wp[0]*PIXELS_PER_METER + origin[0]
        image = np.array(imgs[2])
        point = (int(x), int(y))
        cv2.circle(image, point, radius=2, color=255, thickness=2)
        imgs[2] = Image.fromarray(image)
          
    image = np.array(imgs[0])
    image1 = np.array(imgs[1])
    image2 = np.array(imgs[2])
    x = target_point[0][1]*PIXELS_PER_METER + origin[1]
    y = -(target_point[0][0])*PIXELS_PER_METER + origin[0]  
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
    # all_images = np.concatenate((rgb_image, all_images), axis=0)
    all_images = img_final
    
    Path(f'bev_viz').mkdir(parents=True, exist_ok=True)
    all_images.save(f'bev_viz/{time.time()}_{s}.png')

    # return BEV


def get_coords(x, y, angle, vel):
    length = vel
    endx2 = x + length * math.cos(math.radians(angle))
    endy2 = y + length * math.sin(math.radians(angle))

    return x, y, endx2, endy2  

def get_coords_BB(x, y, angle, extent_x, extent_y):
    endx1 = x - extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy1 = y + extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx2 = x + extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy2 = y - extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx3 = x + extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy3 = y - extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    endx4 = x - extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy4 = y + extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    return (endx1, endy1), (endx2, endy2), (endx3, endy3), (endx4, endy4)