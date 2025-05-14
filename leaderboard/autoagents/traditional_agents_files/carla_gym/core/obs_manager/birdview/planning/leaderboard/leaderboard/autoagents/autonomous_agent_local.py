#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

import copy
from enum import Enum

import carla
from srunner.scenariomanager.timer import GameTime
import numpy as np
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.leaderboard.leaderboard.utils.route_manipulation import downsample_route
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.leaderboard.leaderboard.envs.sensor_interface import SensorInterface


class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'

class AutonomousAgent(object):

    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, path_to_conf_file, task_name, route_index=None, cfg=None, exec_or_inter=None): # CHANGED
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = None #SensorInterface()

        # agent's initialization
        self.setup(path_to_conf_file, task_name, route_index, cfg, exec_or_inter) # CHANGED

        self.wallclock_t0 = None

    def setup(self, path_to_conf_file, route_index=None, cfg=None, exec_or_inter=None):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
        pass

    def sensors(self):  # pylint: disable=no-self-use
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
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def update_sensor_interface(self,input_data):
        self.input_data = input_data

    def __call__(self, sensors=None, light_hazard=None, bbox=None, plant_boxes=None,ego_motion=None,plant_dataset_collection=None, tl_list=None, stop_box=None, lane_list=None, plant_input_dict=None, lane_guidance=None):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.input_data
        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        dummy_collision_label, pred_task_name = None, None
        if plant_dataset_collection:
            control = self.run_step(input_data, timestamp, lane_guidance, sensors)
            tick_data = self.tick(input_data, lane_guidance)
            label_raw = self.super_class.get_bev_boxes(input_data=input_data, pos=tick_data['gps'], lane_guidance=lane_guidance)
            plant_input_image = self.draw_label_raw(label_raw, 'gt')
            pred_wp, target_point, keep_vehicle_ids, keep_vehicle_attn = None, None, None, None
            detected_input_image = np.zeros((200,200,3)).astype(np.uint8)
        else:
            control, pred_wp, target_point, keep_vehicle_ids, keep_vehicle_attn, plant_input_image, dummy_collision_label, pred_task_name = self.plant_run_step(input_data, timestamp, ego_motion, sensors, light_hazard=light_hazard, bbox=bbox, plant_boxes=plant_boxes, tl_list=tl_list, stop_box=stop_box, lane_list=lane_list, plant_input_dict=plant_input_dict)

        control.manual_gear_shift = False

        return control, pred_wp, target_point, keep_vehicle_ids, keep_vehicle_attn, plant_input_image, dummy_collision_label, pred_task_name

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
