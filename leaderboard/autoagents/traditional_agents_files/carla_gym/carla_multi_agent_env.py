import logging
import math

import gym
import numpy as np
import carla
import random
import h5py
import os
import cv2

from .core.zombie_walker.zombie_walker_handler import ZombieWalkerHandler
from .core.zombie_vehicle.zombie_vehicle_handler import ZombieVehicleHandler
from .core.obs_manager.obs_manager_handler import ObsManagerHandler
from .core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from .core.task_actor.scenario_actor.scenario_actor_handler import ScenarioActorHandler
from .utils.traffic_light import TrafficLightHandler
from .utils.dynamic_weather import WeatherHandler
from stable_baselines3.common.utils import set_random_seed
from .utils.zombie_vehicle_control import Zombie_Vehicle_Control
from dataset_collection.dataset_collection import Dataset_Collection

from carla_gym.core.obs_manager.birdview.planning.leaderboard.leaderboard.envs.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid
from carla_gym.core.obs_manager.birdview.planning.leaderboard.leaderboard.envs.sensor_interface import SensorInterface, CallBack

from carla_gym.plant_sensor import Plant_sensor

#from traffic_lights.inference.load import load_model, load_image, preprocess
#from traffic_lights.datasets import reverse_transform_classes
#from traffic_lights.utils import draw_bboxes
import time
import math
import queue
import datetime

logger = logging.getLogger(__name__)


class CarlaMultiAgentEnv(gym.Env):
    def __init__(self, carla_map, host, port, seed, no_rendering,
                 obs_configs, reward_configs, terminal_configs, all_tasks, town_name="Town_name_-1"):
        self._all_tasks = all_tasks
        self._obs_configs = obs_configs
        self._carla_map = carla_map
        self._seed = seed

        self.name = self.__class__.__name__

        self._init_client(carla_map, host, port, seed=seed, no_rendering=no_rendering)

        # define observation spaces exposed to agent
        self._om_handler = ObsManagerHandler(obs_configs)
        self._ev_handler = EgoVehicleHandler(self._client, reward_configs, terminal_configs)
        self._zw_handler = ZombieWalkerHandler(self._client)
        self._zv_handler = ZombieVehicleHandler(self._client, tm_port=self._tm.get_port())
        self._sa_handler = ScenarioActorHandler(self._client)
        self._wt_handler = WeatherHandler(self._world)

        # observation spaces
        self.observation_space = self._om_handler.observation_space
        # define action spaces exposed to agent
        # throttle, steer, brake
        self.action_space = gym.spaces.Dict({ego_vehicle_id: gym.spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32)
            for ego_vehicle_id in obs_configs.keys()})

        self._task_idx = 0
        self._shuffle_task = True
        self._task = self._all_tasks[self._task_idx].copy()

        self.town_name = town_name
        self.trafic_light_disable = False
        self.toggle_tl_in_episode = False

        self.previous_task_id = -1
        self.current_image = None
        self.data_collection_for_imitation = False
        if self.data_collection_for_imitation:
            self.data_collection = Dataset_Collection()

        self.gnss_sensor = None
        self.imu_sensor  = None
        self.gnss_dict = None
        self.imu_dict  = None

        self.control_with = "plant"#"plant" #or "roach"
        self.plant_sensor = None

        self.raw_image_camera_list = []

        #self.predictor = load_model("traffic_lights/pretrained_model/fcos-carla-v01.pth", num_classes=4)
        self.current_raw_image = np.zeros((600,800,3))
        self.camera = None

        self.file_count = 50
        self.previous_loc = np.zeros(3)
        self.none_previous_loc = np.zeros(3)

        self.draw_count = 0





    def set_task_idx(self, task_idx):
        self._task_idx = task_idx
        self._shuffle_task = False
        self._task = self._all_tasks[self._task_idx].copy()
        print("self._task:",self._task)

    @property
    def num_tasks(self):
        return len(self._all_tasks)

    @property
    def task(self):
        return self._task

    def reset(self):

        self.gnss_dict = {}
        if self._shuffle_task:
            self._task_idx = np.random.choice(self.num_tasks)
            self._task = self._all_tasks[self._task_idx].copy()
        self.clean()

        self._wt_handler.reset(self._task['weather'])
        logger.debug("_wt_handler reset done!!")

        ev_spawn_locations = self._ev_handler.reset(self._task['ego_vehicles'])
        logger.debug("_ev_handler reset done!!")

        self._sa_handler.reset(self._task['scenario_actors'], self._ev_handler.ego_vehicles)
        logger.debug("_sa_handler reset done!!")

        self._zw_handler.reset(self._task['num_zombie_walkers'], ev_spawn_locations)
        logger.debug("_zw_handler reset done!!")

        self._zv_handler.reset(self._task['num_zombie_vehicles'], ev_spawn_locations)
        logger.debug("_zv_handler reset done!!")

        self._om_handler.reset(self._ev_handler.ego_vehicles)
        logger.debug("_om_handler reset done!!")

        self._world.tick()

        snap_shot = self._world.get_snapshot()
        self._timestamp = {
            'step': 0,
            'frame': snap_shot.timestamp.frame,
            'relative_wall_time': 0.0,
            'wall_time': snap_shot.timestamp.platform_timestamp,
            'relative_simulation_time': 0.0,
            'simulation_time': snap_shot.timestamp.elapsed_seconds,
            'start_frame': snap_shot.timestamp.frame,
            'start_wall_time': snap_shot.timestamp.platform_timestamp,
            'start_simulation_time': snap_shot.timestamp.elapsed_seconds
        }

        self._ev_handler.set_traffic_light_disable(self.trafic_light_disable)
        _, _, _ = self._ev_handler.tick(self.timestamp)
        #self._om_handler.set_trafic_light_disable(self.trafic_light_disable)
        # get obeservations
        if type(self.gnss_sensor) != type(None):
            self.gnss_sensor.destroy()

        if type(self.imu_sensor) != type(None):
            self.imu_sensor.destroy()

        #self.set_gnss_and_imu_sensor()
        if self.control_with == "plant":
            if type(self.plant_sensor) != type(None):
                self.plant_sensor.cleanup()
            vehicle, _world = self._ev_handler.get_vehicle_and_world()
            self.plant_sensor = Plant_sensor(vehicle, _world)

        obs_dict = self._om_handler.get_observation(self.timestamp)
        self.intersection_label_dic = {}
        self._map = self._world.get_map()
        #self.zombie_vec_control.reset(self._map)
        self.create_actor_for_intersection = False
        self.zombie_vehicle_ids = []
        self._step = 0
        self.toggle_tl = False
        self.prev_action = np.zeros((2)).astype(np.float32)
        obs_dict['hero'].update({'prev_action': self.prev_action})
        self.prev_obs = obs_dict
        if self.data_collection_for_imitation:
            self.data_collection.reset(obs_dict, town_name=self.town_name)
        self._om_handler.set_control_with(self.control_with)

        #self._init_get_raw_image()
        self._init_traffic_light_camera()
        self.create_file()

        return obs_dict

    def step(self, control_dict, model_info={}, lane_action=None):

        if self.control_with == "plant":
            sensor_interface = self.plant_sensor(self._world)
            self.sensor_interface = sensor_interface
            self._om_handler.set_sensors_interface(sensor_interface)
            obs_dict = self._om_handler.get_observation(self.timestamp)
            if type(obs_dict['hero']['birdview']['plant_control']) != type(None):
                control_dict['hero'] = obs_dict['hero']['birdview']['plant_control']
            del obs_dict['hero']['birdview']['plant_control']
        
        self._ev_handler.apply_control(control_dict)
        self._sa_handler.tick()
        # tick world
        self._world.tick()

        if self.trafic_light_disable:
            vehicle_actors = self._world.get_actors().filter('*vehicle*')
            for vec in vehicle_actors:
                self._tm.ignore_lights_percentage(vec, 100)
                self._tm.auto_lane_change(vec, True)

        # update timestamp
        snap_shot = self._world.get_snapshot()
        self._timestamp['step'] = snap_shot.timestamp.frame-self._timestamp['start_frame']
        self._timestamp['frame'] = snap_shot.timestamp.frame
        self._timestamp['wall_time'] = snap_shot.timestamp.platform_timestamp
        self._timestamp['relative_wall_time'] = self._timestamp['wall_time'] - self._timestamp['start_wall_time']
        self._timestamp['simulation_time'] = snap_shot.timestamp.elapsed_seconds
        self._timestamp['relative_simulation_time'] = self._timestamp['simulation_time'] \
            - self._timestamp['start_simulation_time']

        reward_dict, done_dict, info_dict = self._ev_handler.tick(self.timestamp)

        # get observations
        #self._om_handler.set_sensors(self.gnss_dict, self.imu_dict) self._world   self._sensors_list
        #sensor_interface = self.plant_sensor(self._world)
        #self._om_handler.set_sensors_interface(sensor_interface)

        if self.control_with == "roach":
            obs_dict = self._om_handler.get_observation(self.timestamp)

        # update weather
        self._wt_handler.tick(snap_shot.timestamp.delta_seconds)
        if self.toggle_tl_in_episode:
            self.set_toggle_tf_disable()
        #create_new vehicle
        #ego_transform = self._om_handler.get_ego_transform()
        #corresponding_tl_list = TrafficLightHandler.get_traffic_info(ego_transform.location, dist_threshold=20)
        #total_tl_list = TrafficLightHandler.get_traffic_info(ego_transform.location)

        #wp_points = self._om_handler.get_wp_points()
        #self.zombie_vec_control(obs_dict, corresponding_tl_list, total_tl_list, wp_points)

        # num_walkers = len(self._world.get_actors().filter("*walker.pedestrian*"))
        # num_vehicles = len(self._world.get_actors().filter("vehicle*"))
        # logger.debug(f"num_walkers: {num_walkers}, num_vehicles: {num_vehicles}, ")
        self.check_traffic_lights(info_dict)
        self._om_handler.get_is_there_list_parameter()
        #self.detect_traffic_lights()

        if 'carla_map' not in info_dict.keys():
            info_dict['hero'].update({'carla_map':self._carla_map})
            info_dict['hero'].update({'reward_value_env':str(reward_dict['hero'])})
            info_dict['hero'].update({'town_name': self.town_name})
            info_dict['hero'].update({'rendered': obs_dict['hero']['birdview']['rendered']})
            info_dict['hero'].update({'camera': self.camera_image})
            obs_dict['hero'].update({'prev_action': self.prev_action})
            info_dict['hero']['reward_debug']['reward_list'].append('light_hazard: '+str(obs_dict['hero']['birdview']['light_hazard']))
            #info_dict['hero']['reward_debug']['reward_list'].append('color: '+str(color))
            if 'prediction_input' in obs_dict['hero']['birdview'].keys():
                info_dict['hero'].update({'prediction_input': obs_dict['hero']['birdview']['prediction_input']})
                info_dict['hero'].update({'warp_ego_motion': obs_dict['hero']['birdview']['warp_ego_motion']})
            #info_dict['hero'].update({'plant_control': obs_dict['hero']['birdview']['plant_control']})
            #del obs_dict['hero']['birdview']['plant_control']
            del obs_dict['hero']['birdview']['light_hazard']
        

        #self.find_related_object(model_info,obs_dict['hero']['birdview']['masks'][1])

        if self.data_collection_for_imitation:
            self.data_collection.push_data(obs_dict, control_dict, reward_dict['hero'])

        return obs_dict, reward_dict, done_dict, info_dict

    def _init_client(self, carla_map, host, port, seed=2021, no_rendering=False):
        client = None
        while client is None:
            try:
                client = carla.Client(host, port)
                client.set_timeout(60.0)
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    print("Could not connect to Carla server because:", re)
                client = None

        self._client = client
        self._world = client.load_world(carla_map)
        self._tm = client.get_trafficmanager(port+5007)

        self.set_sync_mode(True)
        self.set_no_rendering_mode(self._world, no_rendering)

        # self._tm.set_hybrid_physics_mode(True)

        # self._tm.set_global_distance_to_leading_vehicle(5.0)
        # logger.debug("trafficmanager set_global_distance_to_leading_vehicle")

        set_random_seed(self._seed, using_cuda=True)
        self._tm.set_random_device_seed(self._seed)

        self._world.tick()

        # register traffic lights
        TrafficLightHandler.reset(self._world)

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

    def _init_traffic_light_camera(self):
        try:
            if type(self.camera) != type(None):
                #while self.camera.is_listening:
                self._client.apply_batch([carla.command.DestroyActor(self.camera)])
                self._client.apply_batch([carla.command.DestroyActor(self.imu_sensor)])
                for _ in range(5):
                    self.camera.stop()
                    self.camera.destroy()
                    self.imu_sensor.stop()
                    self.imu_sensor.destroy()

        except:
            pass

        vehicle, world = self._ev_handler.get_vehicle_and_world()
        bp_lib = world.get_blueprint_library()
        carla.IMUMeasurement

        # spawn camera
        self.camera_bp = bp_lib.find('sensor.camera.rgb')
        self.camera_bp.set_attribute("image_size_x", str(704))
        self.camera_bp.set_attribute("image_size_y", str(396))
        self.camera_bp.set_attribute("fov", str(70))
        #camera_init_trans = carla.Transform(carla.Location(z=2))
        camera_init_trans = carla.Transform(carla.Location(x=0.70079118954 , y=0.0159456324149, z=1.51095763913))
        self.camera = world.spawn_actor(self.camera_bp, camera_init_trans, attach_to=vehicle)
        #vehicle.set_autopilot(True)

        # Set up the simulator in synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Create a queue to store and retrieve the sensor data
        self.image_queue = queue.Queue()
        self.camera.listen(self.image_queue.put)

        # Get the attributes from the camera
        image_w = self.camera_bp.get_attribute("image_size_x").as_int()
        image_h = self.camera_bp.get_attribute("image_size_y").as_int()
        fov = self.camera_bp.get_attribute("fov").as_float()


        # Calculate the camera projection matrix to project from 3D -> 2D
        self.K = self.build_projection_matrix(image_w, image_h, fov)

        #
        # Imu sensor
        self.imu_bp = bp_lib.find('sensor.other.imu')

        self.imu_bp.set_attribute('sensor_tick', '0.4')
        self.imu_transform = carla.Transform(carla.Location(x=0.0, z=0.0))
        self.imu_sensor = self._world.spawn_actor(self.imu_bp, self.imu_transform,
                                                  attach_to=vehicle)

        self.imu_sensor.listen(lambda data: self.set_imu(data))

    def set_imu(self, data):
        self.imu_dict = {'accelerometer': data.accelerometer, 'gyroscope': data.gyroscope, 'compass': data.compass}

    def get_traffic_lights(self, world):
        """
        This function returns a list of all the traffic lights in the world.
        """
        return [actor for actor in world.get_actors() if 'traffic_light' in actor.type_id]

    def get_bounding_box(self, actor):
        """
        This function calculates the bounding box of a given actor.
        """
        # Assuming that the actor's shape is approximately a box
        extent = actor.bounding_box.extent
        location = actor.get_location()
        loc = carla.Location(x=location.x, y=location.y, z=location.z)
        #extent = carla.Vector3D(x=location.x + extent.x, y=location.y + extent.y, z=location.z + extent.z)
        bb = carla.BoundingBox(loc, extent)
        return bb

    def draw_bounding_box(self, world, actor):
        """
        Draws a bounding box around the given actor.
        """
        # Get the bounding box of the actor
        bounding_box = actor.bounding_box
        # Get the transform (location + rotation) of the bounding box
        transform = actor.get_transform()
        # Transform the bounding box relative to the actor's location and rotation
        bbox_transform = carla.Transform(bounding_box.location, carla.Rotation())

        # The corners of the box are at:
        # (x +/- extent.x, y +/- extent.y, z +/- extent.z)
        corners = [
            carla.Location(x=-bounding_box.extent.x, y=-bounding_box.extent.y, z=-bounding_box.extent.z),
            carla.Location(x=bounding_box.extent.x, y=-bounding_box.extent.y, z=-bounding_box.extent.z),
            carla.Location(x=bounding_box.extent.x, y=bounding_box.extent.y, z=-bounding_box.extent.z),
            carla.Location(x=-bounding_box.extent.x, y=bounding_box.extent.y, z=-bounding_box.extent.z),
            carla.Location(x=-bounding_box.extent.x, y=-bounding_box.extent.y, z=bounding_box.extent.z),
            carla.Location(x=bounding_box.extent.x, y=-bounding_box.extent.y, z=bounding_box.extent.z),
            carla.Location(x=bounding_box.extent.x, y=bounding_box.extent.y, z=bounding_box.extent.z),
            carla.Location(x=-bounding_box.extent.x, y=bounding_box.extent.y, z=bounding_box.extent.z)
        ]

        # Transform the corners to the world space
        corners = [transform.transform_point(corner) for corner in corners]

        # Draw lines between the corners
        color = carla.Color(255, 0, 0)  # Red color
        thickness = 0.1
        for i in range(4):
            # Draw the bottom square
            world.debug.draw_line(corners[i], corners[(i + 1) % 4], thickness=thickness, color=color, life_time=0.05)
            # Draw the top square
            world.debug.draw_line(corners[i + 4], corners[(i + 1) % 4 + 4], thickness=thickness, color=color,
                                  life_time=0.05)
            # Draw the vertical lines (bottom to top)
            world.debug.draw_line(corners[i], corners[i + 4], thickness=thickness, color=color, life_time=0.05)
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
            org_img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            self.save_arrays(self.img_id, self._carla_map, org_img, points, distance, color, current_speed)
            self.draw_count += 1


        return img

    def save_tl_corner(self,npc,vehicle,img,edges,K, world_2_camera):
        bb = npc.bounding_box
        dist = npc.get_transform().location.distance(vehicle.get_transform().location)

        # Filter for the vehicles within 50m
        if dist < 50:

            # Calculate the dot product between the forward vector
            # of the vehicle and the vector between the vehicle
            # and the other vehicle. We threshold this dot product
            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
            forward_vec = vehicle.get_transform().get_forward_vector()
            ray = npc.get_transform().location - vehicle.get_transform().location

            if forward_vec.dot(ray) > 1:
                p1 = self.get_image_point(bb.location, K, world_2_camera)
                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                for edge in edges:
                    p1 = self.get_image_point(verts[edge[0]], K, world_2_camera)
                    p2 = self.get_image_point(verts[edge[1]], K, world_2_camera)
                    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0, 255), 1)

        return img


    def check_traffic_lights(self, info_dict, draw=True):
        color, stop_bounding_boxes, tl_corner_bb = self._om_handler.get_is_there_list_parameter()
        current_speed = info_dict['hero']['reward_debug']['reward_dict']['r_speed']
        vehicle, _ = self._ev_handler.get_vehicle_and_world()

        ego_loc = vehicle.get_location()
        ego_loc = np.array([ego_loc.x, ego_loc.y, ego_loc.z])
        #self._init_traffic_light_camera()

        # Remember the edge pairs
        edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

        # Retrieve the first image
        #world.tick()
        self.img_id = datetime.date.today().month,datetime.date.today().day,datetime.datetime.now().time().hour, datetime.datetime.now().time().minute,datetime.datetime.now().time().second, datetime.datetime.now().time().microsecond
        image = self.image_queue.get()
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

        bounding_box_set = self._world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        bounding_box_set.extend(self._world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))

        # Get the camera matrix
        world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())
        #bounding_box_set = np.concatenate((stop_bounding_boxes, bounding_box_set), axis=0) [0].bounding_box
        bounding_box_set = stop_bounding_boxes + bounding_box_set# + tl_corner_bb
        bb_list = []
        for index, bb in enumerate(bounding_box_set):
            draw_corner_bb = False
            if isinstance(bb, tuple):
                print("bb:",bb)
                color = bb[1]
                bb = bb[0]
                #self.save_tl_corner(bb,vehicle,img,edges,self.K, world_2_camera)
                #bb = box
            points = np.ones(4)*(-1)
            distance = bb.location.distance(vehicle.get_transform().location)
            """if draw_corner_bb:
                img = self.save_bb(vehicle, bb, edges, world_2_camera, img, draw, image, distance, color, current_speed,
                            ego_loc)"""

            # Filter for distance from ego vehicle
            if 5.5 < distance and distance < 50 and color != 'unknown' and color != 'stop_sign' and abs(bb.rotation.yaw%360-math.degrees(self.imu_dict['compass']))<50 and np.sum(self.previous_loc) != np.sum(ego_loc):
                img = self.save_bb(vehicle, bb, edges, world_2_camera, img, draw, image, distance, color, current_speed,
                            ego_loc)

            if distance < 15 and color == 'stop_sign' and np.sum(self.previous_loc) != np.sum(ego_loc):
                img = self.save_bb(vehicle, bb, edges, world_2_camera, img, draw, image, distance, color, current_speed,
                            ego_loc)



        self.camera_image = img
        self.previous_loc = ego_loc

        if draw:
            # Now draw the image into the OpenCV display window
            #cv2.imwrite('ImageWindowName.png', img)
            # Break the loop if the user presses the Q key
            pass


        return img

    def save_arrays(self, img_id, _carla_map, image, BB, distance, color, current_speed):
        BB = np.array(BB)
        img_id = np.array(img_id)
        _carla_map = int(''.join(list(_carla_map)[-2:]))

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

        file = 'image_bb_'+ str(self.file_count) + '_.h5'

        self.h5_file = h5py.File(file, 'a')
        self.file_count += 1

    def detect_traffic_lights(self):
        # load a model (download from link above - https://drive.google.com/file/d/17mcQ-Ct6bUTS8BEpeDjaZMIFmHS2gptl/view?usp=share_link)
        # load an image
        import torch
        image = self.current_raw_image #load_image("path/to/img.jpg", image_size=480)

        # obtain results
        #preds = self.predictor(image)
        image = preprocess(image)
        preds = self.predictor(image)
        bboxes = preds["predicted_boxes"]
        scores = preds["scores"]
        classes = reverse_transform_classes(preds["pred_classes"], "carla_traffic_lights")

        # optional - visualize predictions
        image = image[0].permute(1, 2, 0).detach().cpu().numpy()
        drawing_image = draw_bboxes(f"./visualized.jpg", image, bboxes[0], scores[0], classes[0])
        return drawing_image

    def set_sync_mode(self, sync):
        settings = self._world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 0.1
        settings.deterministic_ragdolls = True
        self._world.apply_settings(settings)
        self._tm.set_synchronous_mode(sync)

    @staticmethod
    def set_no_rendering_mode(world, no_rendering):
        settings = world.get_settings()
        settings.no_rendering_mode = no_rendering
        world.apply_settings(settings)

    @property
    def timestamp(self):
        return self._timestamp.copy()

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        logger.debug("env __exit__!")

    def close(self):
        self.clean()
        self.set_sync_mode(False)
        self._client = None
        self._world = None
        self._tm = None

    def clean(self):
        self._sa_handler.clean()
        self._zw_handler.clean()
        self._zv_handler.clean()
        self._om_handler.clean()
        self._ev_handler.clean()
        self._wt_handler.clean()
        self._world.tick()

    def set_episode_number(self,episode_number):
        self.episode_number = episode_number

    def get_observation_space(self):
        return self.observation_space

    def get_reward_value(self, intersection_label):
        return self._ev_handler.get_reward_value(timestamp=self.timestamp, intersection_label=intersection_label)

    def set_for_tl_disable(self):
        self._ev_handler.set_traffic_light_disable(self.trafic_light_disable)
        clear_, _, _ = self._ev_handler.tick(self.timestamp)
        self._om_handler.set_trafic_light_disable(self.trafic_light_disable)

    def set_toggle_tf_disable(self):
        ego_transform = self._om_handler.get_ego_transform()
        total_tl_list = TrafficLightHandler.get_traffic_info(ego_transform.location, dist_threshold=30.0)
        if len(total_tl_list) > 0 and self.toggle_tl:
            self.trafic_light_disable = not self.trafic_light_disable
            self.set_for_tl_disable()
            self.toggle_tl = False
        elif len(total_tl_list) == 0:
            self.toggle_tl = True


    def flood_fill(self, image, x, y, target_value, filled):
        height, width = image.shape

        # Create a stack for storing pixel coordinates
        stack = [(x, y)]

        # Perform iterative flood fill
        while stack:
            x, y = stack.pop()

            # Check if the current pixel is within the image boundaries
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            # Check if the current pixel has already been filled or does not match the target value
            if filled[x,y] or image[x, y] != target_value:
                continue

            # Mark the current pixel as filled
            filled[x,y] = True

            # Add neighboring pixels to the stack
            stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

        return filled
    def find_related_object(self, model_info, lane_mask):
        if type(model_info['prediction_output_and_vec']) != type(None):
            prediction_output = model_info['prediction_output_and_vec'].squeeze(0).squeeze(0)
            filled = np.zeros_like(prediction_output, dtype=np.uint8)

            lane_mask[lane_mask > 0] = 1
            mul = (prediction_output * lane_mask)
            mul[mul > 0] = 1
            sum_mul = np.sum(mul)
            if sum_mul > 0:
                coordinates = np.argwhere(mul == 1)

                for coordinate in coordinates:
                    # Perform flood fill to find connected pixels
                    self.flood_fill(prediction_output, coordinate[0], coordinate[1], 255, filled)
                    filled[filled==True]=255
            cv2.imwrite("filled.png",filled.astype(np.uint8))

            mul



    def _init_get_raw_image(self):

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        vehicle, world = self._ev_handler.get_vehicle_and_world()

        # Let's put the vehicle to drive around.
        #vehicle.set_autopilot(True)

        self.raw_image_blueprint_library = world.get_blueprint_library()


        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        self.raw_image_camera_bp = self.raw_image_blueprint_library.find('sensor.camera.rgb')

        self.raw_image_camera_bp.set_attribute('sensor_tick', '0.4')
        #self.camera_bp.set_attribute('image_size_x', '384')
        #self.camera_bp.set_attribute('image_size_y', '192')
        self.raw_image_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.raw_image_camera = world.spawn_actor(self.raw_image_camera_bp, self.raw_image_camera_transform,
                                                       attach_to=vehicle)
        self.raw_image_camera_list.append(self.raw_image_camera)
        self.raw_image_camera.listen(lambda image: self.set_raw_image(image))



    def set_raw_image(self, image):

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.current_raw_image = array
        self.record_image = True
        self.delete_camera = True


    def stop_listening(self):
        while self.raw_image_camera.is_listening:
            self._client.apply_batch([carla.command.DestroyActor(x) for x in self.raw_image_camera_list])
            for camera in self.raw_image_camera_list:
                camera.stop()
                camera.destroy()
        self.raw_image_camera_list = []


