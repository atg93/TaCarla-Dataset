import logging

import cv2
import gym
import numpy as np
import carla

from .core.zombie_walker.zombie_walker_handler import ZombieWalkerHandler
from .core.zombie_vehicle.zombie_vehicle_handler import ZombieVehicleHandler
from .core.obs_manager.obs_manager_handler import ObsManagerHandler
from .core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from .core.task_actor.scenario_actor.scenario_actor_handler import ScenarioActorHandler
from .utils.traffic_light import TrafficLightHandler
from .utils.dynamic_weather import WeatherHandler
from stable_baselines3.common.utils import set_random_seed
#from agents_roach.rl_birdview.lane_model.lane_guidance import Lane_Guidance

logger = logging.getLogger(__name__)

import datetime
import json

class CarlaMultiAgentEnv(gym.Env):
    def __init__(self, carla_map, host, port, seed, no_rendering,
                 obs_configs, reward_configs, terminal_configs, all_tasks, town_name="Town_name_-1"):
        self._all_tasks = all_tasks
        self._obs_configs = obs_configs
        self._carla_map = carla_map
        self._seed = seed

        self.name = self.__class__.__name__

        print("CarlaMultiAgentEnv port:",port)
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
        self.data_collection = True
        self.previous_task_id = -1

        self.lane_guidance = None#Lane_Guidance(self._world)
        self.prev_obs_dict = None
        self.town_name = town_name

        ####record
        self.actor_list = []
        self.image_list = []
        self.current_raw_image = None
        self.record_image = False
        self.ready_to_get_data = False
        self.delete_camera = False
        self.raw_image_camera_list = []
        self.camera_is_listening = False
        self.episode_number = 1



    def set_task_idx(self, task_idx):
        self._task_idx = task_idx
        self._shuffle_task = False
        self._task = self._all_tasks[self._task_idx].copy()

    def set_episode_number(self,episode_number):
        self.episode_number = episode_number

    @property
    def num_tasks(self):
        return len(self._all_tasks)

    @property
    def task(self):
        return self._task

    def reset(self):
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

        _, _, _ = self._ev_handler.tick(self.timestamp)
        # get obeservations
        obs_dict = self._om_handler.get_observation(self.timestamp)
        #print("_init_get_raw_image:")
        #self._init_get_raw_image()
        return obs_dict

    def step(self, control_dict, model_info):
        self._ev_handler.apply_control(control_dict)
        self._sa_handler.tick()
        # tick world
        self._world.tick()

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
        # get_observations

        """obs_dict = self._om_handler.get_observation(self.timestamp,orginal_route_design=False)
        ego_location, wp_list = self._om_handler.get_ego_and_wp_list()
        new_route = self.lane_guidance(current_ego_location = ego_location,
                                       current_wp_list = wp_list, observation = obs_dict,
                                       done = done_dict, info_dic = info_dict,train=True)
        self._om_handler.update_route_plan(new_route)"""

        obs_dict = self._om_handler.get_observation(self.timestamp)

        if self.data_collection:
            self._om_handler.set_collect_data_info(self.data_collection)
            if self.previous_task_id != self._task_idx: #done_dict['hero'] or
                self._om_handler.set_episode_number(self.episode_number)
                self._om_handler.reset_episode(self.timestamp)
                self.previous_task_id = self._task_idx


            _ = self._om_handler.collect_data_for_future_prediction(self.timestamp)

        if self.data_collection and done_dict['hero']:
            self._om_handler.close_h5_file()

        # update weather
        self._wt_handler.tick(snap_shot.timestamp.delta_seconds)

        # num_walkers = len(self._world.get_actors().filter("*walker.pedestrian*"))
        # num_vehicles = len(self._world.get_actors().filter("vehicle*"))
        # logger.debug(f"num_walkers: {num_walkers}, num_vehicles: {num_vehicles}, ")

        if 'carla_map'not in info_dict.keys():
            info_dict['hero'].update({'carla_map':self._carla_map})
            info_dict['hero'].update({'reward_value_env':str(reward_dict['hero'])})
            info_dict['hero'].update({'town_name': self.town_name})
            info_dict['hero'].update({'rendered': obs_dict['hero']['birdview']['rendered']})

        ########deneme
        #self.camera_is_listening = self.raw_image_camera.is_listening
        info_dict['hero'].update({"record_image": self.record_image})
        if self.record_image:
            info_dict['hero'].update({"raw_image": np.array(self.current_raw_image)})

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
