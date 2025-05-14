#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Wrapper for autonomous agents required for tracking and checking of used sensors
"""

from __future__ import print_function
import math
import os

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from carla_gym.core.obs_manager.birdview.planning.leaderboard.leaderboard.autoagents.autonomous_agent import Track

from carla_gym.core.obs_manager.birdview.planning.leaderboard.leaderboard.envs.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid
# Use this line instead to run WOR
#from leaderboard1.leaderboard.envs.sensor_interface import (CallBack, StitchCameraReader, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid)

DATAGEN = 1 #int(os.environ.get('DATAGEN'))
MAX_ALLOWED_RADIUS_SENSOR = 10.0

#AGENT_NAME = os.environ.get('AGENT_NAME')
"""if AGENT_NAME == 'Explainability':
    MAX_ALLOWED_RADIUS_SENSOR = 1000.0 #increased for topdown map generation
else:
    MAX_ALLOWED_RADIUS_SENSOR = 10.0"""

SENSORS_LIMITS = {
    'sensor.camera.rgb': 4,
    'sensor.lidar.ray_cast': 1,
    'sensor.other.radar': 2,
    'sensor.other.gnss': 1,
    'sensor.other.imu': 1,
    'sensor.opendrive_map': 1,
    'sensor.speedometer': 1,
    'sensor.stitch_camera.rgb': 1,
    'sensor.camera.depth': 4, # for data generation
    'sensor.camera.semantic_segmentation': 4 # for data generation
}


class AgentError(Exception):
    """
    Exceptions thrown when the agent returns an error during the simulation
    """

    def __init__(self, message):
        super(AgentError, self).__init__(message)


class AgentWrapper(object):

    """
    Wrapper for autonomous agents required for tracking and checking of used sensors
    """

    allowed_sensors = [
        'sensor.opendrive_map',
        'sensor.speedometer',
        'sensor.camera.rgb',
        'sensor.camera',
        'sensor.lidar.ray_cast',
        'sensor.other.radar',
        'sensor.other.gnss',
        'sensor.other.imu',
        'sensor.stitch_camera.rgb', # for World on Rails eval
        'sensor.camera.depth', # for data generation
        'sensor.camera.semantic_segmentation', # for data generation
    ]

    _agent = None
    _sensors_list = []
    sensor_list_names = []

    def __init__(self, agent):
        """
        Set the autonomous agent
        """
        self._agent = agent

    def __call__(self,light_hazard=None):
        """
        Pass the call directly to the agent
        """
        return self._agent(self.sensor_list_names, light_hazard=light_hazard)

    def update_sensor_interface(self,sensor_interface):
        self._agent.update_sensor_interface(sensor_interface)

    def set_world(self, _world=None):
        
        if type(_world) != type(None):
            self._world = _world
        else:
            self._world = CarlaDataProvider.get_world()

    def setup_sensors(self, vehicle, debug_mode=False):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        bp_library = self._world.get_blueprint_library()
        for sensor_spec in self._agent.sensors():
            # These are the pseudosensors (not spawned)
            if sensor_spec['type'].startswith('sensor.opendrive_map'):
                # The HDMap pseudo sensor is created directly here
                self.hd_map_sensor = OpenDriveMapReader(vehicle, sensor_spec['reading_frequency'])
            elif sensor_spec['type'].startswith('sensor.speedometer'):
                delta_time = self._world.get_settings().fixed_delta_seconds
                frame_rate = 1 / delta_time
                self.speed_sensor = SpeedometerReader(vehicle, frame_rate)
            elif sensor_spec['type'].startswith('sensor.stitch_camera'):
                delta_time = self._world.get_settings().fixed_delta_seconds
                frame_rate = 1 / delta_time
                sensor = StitchCameraReader(bp_library, vehicle, sensor_spec, frame_rate)
            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(str(sensor_spec['type']))
                if sensor_spec['type'].startswith('sensor.camera'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    bp.set_attribute('lens_circle_multiplier', str(3.0))
                    bp.set_attribute('lens_circle_falloff', str(3.0))
                    if sensor_spec['type'].startswith('sensor.camera.rgb'):
                        bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                        bp.set_attribute('chromatic_aberration_offset', str(0))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.lidar'):
                    bp.set_attribute('range', str(85))
                    if DATAGEN==1:
                        bp.set_attribute('rotation_frequency', str(sensor_spec['rotation_frequency']))
                        bp.set_attribute('points_per_second', str(sensor_spec['points_per_second']))
                    else:
                        bp.set_attribute('rotation_frequency', str(10))
                        bp.set_attribute('points_per_second', str(600000))
                    bp.set_attribute('channels', str(64))
                    bp.set_attribute('upper_fov', str(10))
                    bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
                    bp.set_attribute('dropoff_general_rate', str(0.45))
                    bp.set_attribute('dropoff_intensity_limit', str(0.8))
                    bp.set_attribute('dropoff_zero_intensity', str(0.4))
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                    # create sensor
                    sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                    sensor = self._world.spawn_actor(bp, sensor_transform, vehicle)
                elif sensor_spec['type'].startswith('sensor.other.radar'):
                    bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('points_per_second', '1500')
                    bp.set_attribute('range', '100')  # meters

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])

                    # create sensor
                    sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                    sensor = self._world.spawn_actor(bp, sensor_transform, vehicle)

                elif sensor_spec['type'].startswith('sensor.other.gnss'):
                    if DATAGEN==0:
                        bp.set_attribute('noise_alt_stddev', str(0.000005))
                        bp.set_attribute('noise_lat_stddev', str(0.000005))
                        bp.set_attribute('noise_lon_stddev', str(0.000005))
                    bp.set_attribute('noise_alt_bias', str(0.0))
                    bp.set_attribute('noise_lat_bias', str(0.0))
                    bp.set_attribute('noise_lon_bias', str(0.0))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation()

                    # create sensor
                    sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                    self.gps_sensor = self._world.spawn_actor(bp, sensor_transform, vehicle)

                elif sensor_spec['type'].startswith('sensor.other.imu'):
                    bp.set_attribute('noise_accel_stddev_x', str(0.001))
                    bp.set_attribute('noise_accel_stddev_y', str(0.001))
                    bp.set_attribute('noise_accel_stddev_z', str(0.015))
                    bp.set_attribute('noise_gyro_stddev_x', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_y', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_z', str(0.001))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                    # create sensor
                    sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                    self.imu_sensor = self._world.spawn_actor(bp, sensor_transform, vehicle)
            """deneme_id = sensor_spec['id']
            deneme_type = sensor_spec['type']
            deneme_sensor = sensor
            deneme_sensor_interface = self._agent.sensor_interface"""
            # setup callback
            print("Plant sensor:",sensor_spec['id'])#hd_map, imu, gps, speed
            if sensor_spec['id'] == 'hd_map':
                self.hd_map_callback = CallBack(sensor_spec['id'], sensor_spec['type'], self.hd_map_sensor, self._agent.sensor_interface)
                self.hd_map_sensor.listen(lambda data: self.hd_map_callback(data))
                sensor = self.hd_map_sensor
            elif sensor_spec['id'] == 'imu':
                self.imu_callback = CallBack(sensor_spec['id'], sensor_spec['type'], self.imu_sensor, self._agent.sensor_interface)
                self.imu_sensor.listen(lambda data: self.imu_callback(data))
                sensor = self.imu_sensor
            elif sensor_spec['id'] == 'gps':
                self.gps_callback = CallBack(sensor_spec['id'], sensor_spec['type'], self.gps_sensor, self._agent.sensor_interface)
                self.gps_sensor.listen(lambda data: self.gps_callback(data))
                sensor = self.gps_sensor
            elif sensor_spec['id'] == 'speed':
                self.speed_callback = CallBack(sensor_spec['id'], sensor_spec['type'], self.speed_sensor, self._agent.sensor_interface)
                self.speed_sensor.listen(lambda data: self.speed_callback(data))
                sensor = self.speed_sensor
            self._sensors_list.append(sensor)
            self.sensor_list_names.append([sensor_spec['id'], sensor])
        asd = 0
        # Tick once to spawn the sensors
        self._world.tick()


    @staticmethod
    def validate_sensor_configuration(sensors, agent_track, selected_track):
        """
        Ensure that the sensor configuration is valid, in case the challenge mode is used
        Returns true on valid configuration, false otherwise
        """
        if Track(selected_track) != agent_track:
            raise SensorConfigurationInvalid("You are submitting to the wrong track [{}]!".format(Track(selected_track)))

        sensor_count = {}
        sensor_ids = []

        for sensor in sensors:

            # Check if the is has been already used
            sensor_id = sensor['id']
            if sensor_id in sensor_ids:
                raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(sensor_id))
            else:
                sensor_ids.append(sensor_id)

            # Check if the sensor is valid
            if agent_track == Track.SENSORS:
                if sensor['type'].startswith('sensor.opendrive_map'):
                    raise SensorConfigurationInvalid("Illegal sensor used for Track [{}]!".format(agent_track))

            # Check the sensors validity
            if sensor['type'] not in AgentWrapper.allowed_sensors:
                raise SensorConfigurationInvalid("Illegal sensor used. {} are not allowed!".format(sensor['type']))

            # Check the extrinsics of the sensor
            if 'x' in sensor and 'y' in sensor and 'z' in sensor:
                if math.sqrt(sensor['x']**2 + sensor['y']**2 + sensor['z']**2) > MAX_ALLOWED_RADIUS_SENSOR:
                    raise SensorConfigurationInvalid(
                        "Illegal sensor extrinsics used for Track [{}]!".format(agent_track))

            # Check the amount of sensors
            if sensor['type'] in sensor_count:
                sensor_count[sensor['type']] += 1
            else:
                sensor_count[sensor['type']] = 1


        for sensor_type, max_instances_allowed in SENSORS_LIMITS.items():
            if sensor_type in sensor_count and sensor_count[sensor_type] > max_instances_allowed:
                raise SensorConfigurationInvalid(
                    "Too many {} used! "
                    "Maximum number allowed is {}, but {} were requested.".format(sensor_type,
                                                                                  max_instances_allowed,
                                                                                  sensor_count[sensor_type]))

    def cleanup(self):
        """
        Remove and destroy all sensors
        """
        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                self._sensors_list[i].stop()
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None
        self._sensors_list.clear()
        self.sensor_list_names.clear()