

import carla
import numpy as np
from leaderboard.autoagents.traditional_agents_files.planning.leaderboard.leaderboard.envs.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid
from leaderboard.autoagents.traditional_agents_files.planning.leaderboard.leaderboard.envs.sensor_interface import SensorInterface, CallBack, GenericMeasurement

class Plant_sensor:
    def __init__(self):
        self.gnss_sensor = None
        self.imu_sensor  = None
        self.gnss_dict = None
        self.imu_dict  = None

        self.sensor_interface = SensorInterface()
        self.sensors = [{
            'type': 'sensor.opendrive_map',
            'reading_frequency': 1e-6,
            'id': 'hd_map'
        },
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'imu'
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'speed'
            }
        ]  # sensor_tugrul
        self._sensors_list = []
        self.sensor_list_names = []
        self._sensors_callback = []

        #self.setup_sensors(vehicle, _world)

    def setup_sensors(self, vehicle, _world):
        return self.setup_sensors(vehicle, _world)

    def __call__(self, _world):
        asd = 0
        GenericMeasurement(self._sensors_list[0](_world),self.frame)
        self._sensors_callback[0](GenericMeasurement(self._sensors_list[0](_world),self.frame))
        self._sensors_callback[1](GenericMeasurement(self._sensors_list[3](),self.frame))
        return self.sensor_interface


    def parse_input(self,data):
        if isinstance(data, carla.libcarla.GnssMeasurement):
            _tag = 'gps'
            self._parse_gnss_cb(data, _tag)
        elif isinstance(data, carla.libcarla.IMUMeasurement):
            _tag = 'imu'
            self._parse_imu_cb(data, _tag)

    def _parse_gnss_cb(self, gnss_data, tag):
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self.sensor_interface.update_sensor(tag, array, gnss_data.frame)

    def _parse_imu_cb(self, imu_data, tag):
        array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                         ], dtype=np.float64)
        self.sensor_interface.update_sensor(tag, array, imu_data.frame)
        self.frame = imu_data.frame

    def setup_sensors(self, vehicle, _world, debug_mode=False, DATAGEN=1):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        bp_library = _world.get_blueprint_library()
        for sensor_spec in self.sensors:
            # These are the pseudosensors (not spawned)
            if sensor_spec['type'].startswith('sensor.opendrive_map'):
                # The HDMap pseudo sensor is created directly here
                self.hd_map_sensor = OpenDriveMapReader(vehicle, sensor_spec['reading_frequency'])
            elif sensor_spec['type'].startswith('sensor.speedometer'):
                delta_time = 0.005 #_world.get_settings().fixed_delta_seconds#tugrul
                frame_rate = 1 / delta_time
                self.speed_sensor = SpeedometerReader(vehicle, frame_rate)
            elif sensor_spec['type'].startswith('sensor.stitch_camera'):
                delta_time = _world.get_settings().fixed_delta_seconds
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
                    if DATAGEN == 1:
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
                    sensor = _world.spawn_actor(bp, sensor_transform, vehicle)
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
                    sensor = _world.spawn_actor(bp, sensor_transform, vehicle)

                elif sensor_spec['type'].startswith('sensor.other.gnss'):
                    if DATAGEN == 0:
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
                    self.gps_sensor = _world.spawn_actor(bp, sensor_transform, vehicle)

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
                    self.imu_sensor = _world.spawn_actor(bp, sensor_transform, vehicle)
            # setup callback
            print("Plant sensor:", sensor_spec['id'])  # hd_map, imu, gps, speed
            if sensor_spec['id'] == 'hd_map':
                self.hd_map_callback = CallBack(sensor_spec['id'], sensor_spec['type'], self.hd_map_sensor,
                                                self.sensor_interface)
                self.hd_map_sensor.listen(self.hd_map_callback)
                sensor = self.hd_map_sensor
                _callback = self.hd_map_callback
                self._sensors_callback.append(_callback)

            elif sensor_spec['id'] == 'imu':
                self.imu_callback = CallBack(sensor_spec['id'], sensor_spec['type'], self.imu_sensor,
                                             self.sensor_interface)
                self.imu_sensor.listen(lambda data: self.parse_input(data))
                sensor = self.imu_sensor
            elif sensor_spec['id'] == 'gps':
                self.gps_callback = CallBack(sensor_spec['id'], sensor_spec['type'], self.gps_sensor,
                                             self.sensor_interface)
                self.gps_sensor.listen(lambda data: self.parse_input(data))
                sensor = self.gps_sensor
            elif sensor_spec['id'] == 'speed':
                self.speed_callback = CallBack(sensor_spec['id'], sensor_spec['type'], self.speed_sensor,
                                               self.sensor_interface)
                _callback = self.speed_callback
                self.speed_sensor.listen(self.speed_callback)
                sensor = self.speed_sensor
                self._sensors_callback.append(_callback)

            self._sensors_list.append(sensor)
            self.sensor_list_names.append(
                [sensor_spec['id'], sensor])  # self._sensors_list[0](self._world), self._sensors_list[3]()
        asd = 0
        # Tick once to spawn the sensors
        _world.tick()


    def set_dummy_data(self, data):
        #print("data:",data)
        pass

    def cleanup(self):
        """
        Remove and destroy all sensors
        """
        self.attempt = 5
        try:
            for _ in range(self.attempt):
                for i, _ in enumerate(self._sensors_list):
                    if self._sensors_list[i] is not None:
                        self._sensors_list[i].stop()
                        self._sensors_list[i].destroy()
                        #self._sensors_list[i] = None
        except:
            pass
        self._sensors_list.clear()
        self.sensor_list_names.clear()