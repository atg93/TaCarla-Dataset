from carla_gym.core.obs_manager.birdview.planning.carla_agent_files.PlanT_agent import PlanTAgent
import carla
import os
import numpy as np
from rdp import rdp

from carla_gym.core.obs_manager.birdview.planning.leaderboard.leaderboard.envs.sensor_interface import SensorInterface, CallBack
from carla_gym.core.obs_manager.birdview.planning.carla_agent_files.nav_planner import RoutePlanner_new as RoutePlanner

from carla_gym.core.obs_manager.birdview.planning.carla_agent_files.nav_planner import PIDController, interpolate_trajectory
from carla_gym.core.obs_manager.birdview.planning.carla_agent_files.agent_utils.coordinate_utils import normalize_angle

from carla_gym.core.obs_manager.birdview.planning.leaderboard.leaderboard.autoagents.agent_wrapper_local import AgentWrapper, AgentError


from srunner.scenariomanager.timer import GameTime

class Planning:
    def __init__(self, ego_vehicles, _world, _debug_mode=False, planning_type='plant'):
        self.planning_type = planning_type
        self.sensor_info = self._sensors()
        path_to_conf_file = '/home/tg22/remote-pycharm/roach/carla-roach/carla_gym/core/obs_manager/birdview/planning/checkpoints/PlanT/3x/PlanT_medium/log'
        self.plant_agent = PlanTAgent(path_to_conf_file=path_to_conf_file)
        self.plant_agent.set_vehicle(ego_vehicles)

        #self.plant_agent._init()
        self.agent_wrapper = AgentWrapper(self.plant_agent)
        self.agent_wrapper.set_world(_world=_world)
        #self.agent_wrapper.setup_sensors(ego_vehicles, _debug_mode)


        #self.sensor_interface = SensorInterface()
        self._route_planner = RoutePlanner(7.5, 50.0)
        #self.plant_agent.set_route_planner(self._route_planner)

        self.map_precision = 10.0  # meters per point
        self.rdp_epsilon = 0.5  # epsilon for route shortening

        # radius in which other actors/map elements are considered
        # distance is from the center of the ego-vehicle and measured in 3D space
        self.max_actor_distance = 50.0  # copy from expert
        self.max_light_distance = 15.0  # copy from expert
        self.max_route_distance = 30.0
        self.max_map_element_distance = 30.0
        self.DATAGEN = 1


    def set_info(self,_pixels_per_meter,_world_offset):
        self.plant_agent.set_info(_pixels_per_meter, _world_offset)

    def save_score(self, score_composed, score_route, score_penalty):
        self.agent_wrapper.save_score(score_composed, score_route, score_penalty)



    def __call__(self, dummy_target_location, world, M_warp, _global_plan, ego_actor, sensor_interface, global_plan_gps, global_plan_world_coord, light_hazard):
        self.plant_agent.traffic_light_hazard = light_hazard
        self.plant_agent.set_dummy_target_location(dummy_target_location)
        timestamp = GameTime.get_time()

        self._world = world
        self._vehicle = ego_actor.vehicle
        self.keep_ids = None
        #self.plant_agent._init(world.get_map())
        self.plant_agent.set_global_plan(global_plan_gps, global_plan_world_coord)
        self.plant_agent.set_m_warp(M_warp)

        self.agent_wrapper.update_sensor_interface(sensor_interface)
        output, pred_wp, target_point,keep_vehicle_ids, keep_vehicle_attn = self.agent_wrapper(light_hazard=light_hazard)

        return output, pred_wp, target_point,keep_vehicle_ids, keep_vehicle_attn

    def preprocess_input(self,input_data):
        new_input = {'gps':None,'speed':0.0,'compass':None,'rgb':None,'rgb_augmented':None,'boxes':None,'lidar':None,'target_point':None}
        for _key in new_input.keys():
            if _key in input_data.keys():
                new_input[_key] = input_data[_key]
        return new_input

    def reorganized_for_plant(self, world_map, _global_plan):
        trajectory = [item[0].transform.location for item in _global_plan]
        _new_global_plan, _ = interpolate_trajectory(world_map, trajectory)

        return _new_global_plan

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


    def _sensors(self):
        result = [{
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
        ]

        return result

    def get_bev_boxes(self, input_data=None, lidar=None, pos=None):

        # -----------------------------------------------------------
        # Ego vehicle
        # -----------------------------------------------------------
        # add vehicle velocity and brake flag
        ego_location = self._vehicle.get_location()
        pos = np.array([ego_location.x, ego_location.y])
        #pos = self._route_planner.convert_gps_to_carla(pos)

        ego_transform = self._vehicle.get_transform()
        ego_control = self._vehicle.get_control()
        ego_velocity = self._vehicle.get_velocity()
        ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity)  # In m/s
        ego_brake = ego_control.brake
        ego_rotation = ego_transform.rotation
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_extent = self._vehicle.bounding_box.extent
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z]) * 2.
        ego_yaw = ego_rotation.yaw / 180 * np.pi
        relative_yaw = 0
        relative_pos = self.get_relative_transform(ego_matrix, ego_matrix)

        results = []

        # add ego-vehicle to results list
        # the format is category, extent*3, position*3, yaw, points_in_bbox, distance, id
        # the position is in lidar coordinates
        result = {"class": "Car",
                  "extent": [ego_dx[2], ego_dx[0], ego_dx[1]],  # TODO:
                  "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                  "yaw": relative_yaw,
                  "num_points": -1,
                  "distance": -1,
                  "speed": ego_speed,
                  "brake": ego_brake,
                  "id": int(self._vehicle.id),
                  }
        results.append(result)

        # -----------------------------------------------------------
        # Other vehicles
        # -----------------------------------------------------------

        self._actors = self._world.get_actors()
        vehicles = self._actors.filter('*vehicle*')
        tlights = self._actors.filter('*traffic_light*')
        for vehicle in vehicles:
            if (vehicle.get_location().distance(ego_location) < self.max_actor_distance):
                if (vehicle.id != self._vehicle.id):
                    vehicle_rotation = vehicle.get_transform().rotation
                    vehicle_matrix = np.array(vehicle.get_transform().get_matrix())

                    vehicle_extent = vehicle.bounding_box.extent
                    dx = np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]) * 2.
                    yaw = vehicle_rotation.yaw / 180 * np.pi

                    relative_yaw = normalize_angle(yaw - ego_yaw)
                    relative_pos = self.get_relative_transform(ego_matrix, vehicle_matrix)

                    vehicle_transform = vehicle.get_transform()
                    vehicle_control = vehicle.get_control()
                    vehicle_velocity = vehicle.get_velocity()
                    vehicle_speed = self._get_forward_speed(transform=vehicle_transform,
                                                            velocity=vehicle_velocity)  # In m/s
                    vehicle_brake = vehicle_control.brake

                    # filter bbox that didn't contains points of contains less points
                    if not lidar is None:
                        num_in_bbox_points = self.get_points_in_bbox(ego_matrix, vehicle_matrix, dx, lidar)
                        # print("num points in bbox", num_in_bbox_points)
                    else:
                        num_in_bbox_points = -1

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": "Car",
                        "extent": [dx[2], dx[0], dx[1]],  # TODO
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "num_points": int(num_in_bbox_points),
                        "distance": distance,
                        "speed": vehicle_speed,
                        "brake": vehicle_brake,
                        "id": int(vehicle.id),
                    }
                    results.append(result)

        # -----------------------------------------------------------
        # Route rdp
        # -----------------------------------------------------------
        #if input_data is not None:
        # pos = self._get_position(input_data['gps'][1][:2])
        # self.gps_buffer.append(pos)
        # pos = np.average(self.gps_buffer, axis=0)  # Denoised position
        #self._route_planner.load()
        waypoint_route = self._route_planner.run_step(pos)
        self.waypoint_route = np.array([[node[0][0], node[0][1]] for node in waypoint_route])
        #self._route_planner.save()

        max_len = 50
        if len(self.waypoint_route) < max_len:
            max_len = len(self.waypoint_route)
        shortened_route = rdp(self.waypoint_route[:max_len], epsilon=self.rdp_epsilon)

        # convert points to vectors
        vectors = shortened_route[1:] - shortened_route[:-1]
        midpoints = shortened_route[:-1] + vectors / 2.
        norms = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        for i, midpoint in enumerate(midpoints):
            # find distance to center of waypoint
            center_bounding_box = carla.Location(midpoint[0], midpoint[1], 0.0)
            transform = carla.Transform(center_bounding_box)
            route_matrix = np.array(transform.get_matrix())
            relative_pos = self.get_relative_transform(ego_matrix, route_matrix)
            distance = np.linalg.norm(relative_pos)

            # find distance to beginning of bounding box
            starting_bounding_box = carla.Location(shortened_route[i][0], shortened_route[i][1], 0.0)
            st_transform = carla.Transform(starting_bounding_box)
            st_route_matrix = np.array(st_transform.get_matrix())
            st_relative_pos = self.get_relative_transform(ego_matrix, st_route_matrix)
            st_distance = np.linalg.norm(st_relative_pos)

            # only store route boxes that are near the ego vehicle
            if i > 0 and st_distance > self.max_route_distance:
                continue

            length_bounding_box = carla.Vector3D(norms[i] / 2., ego_extent.y, ego_extent.z)
            bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
            bounding_box.rotation = carla.Rotation(pitch=0.0,
                                                   yaw=angles[i] * 180 / np.pi,
                                                   roll=0.0)

            route_extent = bounding_box.extent
            dx = np.array([route_extent.x, route_extent.y, route_extent.z]) * 2.
            relative_yaw = normalize_angle(angles[i] - ego_yaw)

            # visualize subsampled route
            # self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1,
            #                             color=carla.Color(0, 255, 255, 255), life_time=(10.0/self.frame_rate_sim))

            result = {
                "class": "Route",
                "extent": [dx[2], dx[0], dx[1]],  # TODO
                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                "yaw": relative_yaw,
                "centre_distance": distance,
                "starting_distance": st_distance,
                "id": i,
            }
            results.append(result)

        if self.DATAGEN:
            # -----------------------------------------------------------
            # Traffic lights
            # -----------------------------------------------------------

            _traffic_lights = self.get_nearby_object(ego_location, tlights, self.max_light_distance)

            for light in _traffic_lights:
                if (light.state == carla.libcarla.TrafficLightState.Red):
                    state = 0
                elif (light.state == carla.libcarla.TrafficLightState.Yellow):
                    state = 1
                elif (light.state == carla.libcarla.TrafficLightState.Green):
                    state = 2
                else:  # unknown
                    state = -1

                center_bounding_box = light.get_transform().transform(light.trigger_volume.location)
                center_bounding_box = carla.Location(center_bounding_box.x, center_bounding_box.y,
                                                     center_bounding_box.z)
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

                result = {
                    "class": "Light",
                    "extent": [dx[2], dx[0], dx[1]],  # TODO
                    "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                    "yaw": relative_yaw,
                    "distance": distance,
                    "state": state,
                    "id": int(light.id),
                }
                results.append(result)

            # -----------------------------------------------------------
            # Map elements
            # -----------------------------------------------------------

            for lane_id, poly in enumerate(self.polygons):
                for point_id, point in enumerate(poly):
                    if (point.location.distance(ego_location) < self.max_map_element_distance):
                        point_matrix = np.array(point.get_matrix())

                        yaw = point.rotation.yaw / 180 * np.pi

                        relative_yaw = yaw - ego_yaw
                        relative_pos = self.get_relative_transform(ego_matrix, point_matrix)
                        distance = np.linalg.norm(relative_pos)

                        result = {
                            "class": "Lane",
                            "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                            "yaw": relative_yaw,
                            "distance": distance,
                            "point_id": int(point_id),
                            "lane_id": int(lane_id),
                        }
                        results.append(result)

        return results, ego_speed

    def get_nearby_object(self, vehicle_position, actor_list, radius):
        nearby_objects = []
        for actor in actor_list:
            trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
            trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x, y=trigger_box_global_pos.y, z=trigger_box_global_pos.z)
            if (trigger_box_global_pos.distance(vehicle_position) < radius):
                nearby_objects.append(actor)
        return nearby_objects

    def set_polygons(self, world_map):
        #super()._init(hd_map)

        # if self.scenario_logger:
        #     from srunner.scenariomanager.carla_data_provider import CarlaDataProvider # privileged
        #     self._vehicle = CarlaDataProvider.get_hero_actor()
        #     self.scenario_logger.ego_vehicle = self._vehicle
        #     self.scenario_logger.world = self._vehicle.get_world()

        topology = [x[0] for x in world_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        self.polygons = []
        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(self.map_precision)[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(self.map_precision)[0]

            left_marking = [self.lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            right_marking = [self.lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
            self.polygons.append(left_marking + [x for x in reversed(right_marking)])

    def lateral_shift(self, transform, shift):
        transform.rotation.yaw += 90
        transform.location += shift * transform.get_forward_vector()
        return transform

    def setup_sensors(self, CarlaDataProvider, vehicle, debug_mode=False):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        bp_library = CarlaDataProvider.get_world().get_blueprint_library()
        for sensor_spec in self._agent.sensors():
            # These are the pseudosensors (not spawned)
            if sensor_spec['type'].startswith('sensor.opendrive_map'):
                # The HDMap pseudo sensor is created directly here
                sensor = OpenDriveMapReader(vehicle, sensor_spec['reading_frequency'])
            elif sensor_spec['type'].startswith('sensor.speedometer'):
                delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
                frame_rate = 1 / delta_time
                sensor = SpeedometerReader(vehicle, frame_rate)
            elif sensor_spec['type'].startswith('sensor.stitch_camera'):
                delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
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
                sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)
            deneme_id = sensor_spec['id']
            deneme_type = sensor_spec['type']
            deneme_sensor = sensor
            deneme_sensor_interface = self._agent.sensor_interface
            # setup callback
            sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self._agent.sensor_interface))
            self._sensors_list.append(sensor)
            self.sensor_list_names.append([sensor_spec['id'], sensor])