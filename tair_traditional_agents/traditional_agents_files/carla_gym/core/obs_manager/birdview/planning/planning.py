from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.PlanT_agent import PlanTAgent
import carla
import os
import numpy as np
from rdp import rdp

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.leaderboard.leaderboard.envs.sensor_interface import SensorInterface, CallBack
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.nav_planner import RoutePlanner_new as RoutePlanner

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.nav_planner import PIDController, interpolate_trajectory
from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.agent_utils.coordinate_utils import normalize_angle

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.leaderboard.leaderboard.autoagents.agent_wrapper_local import AgentWrapper, AgentError

from leaderboard.autoagents.traditional_agents_files.utils.process_radar import Process_Radar

from srunner.scenariomanager.timer import GameTime

import math
import cv2

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


        self.sensor_interface = SensorInterface()
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
        self.actor_type_list = []

        self.process_radar = Process_Radar()


        self._width = 1845
        self.real_height = 586
        self.real_width = 1034

        X_BOUND = [-54.0, 54.0, 0.0542]  # Forward
        Y_BOUND = [-54.0, 54.0, 0.0542]  # Sides
        Z_BOUND = [-10.0, 10.0, 5.0]
        self.project_view = self.calculate_view_matrix(X_BOUND, Y_BOUND, Z_BOUND)
        self.check_before_stop = []


    def project(self, points):
        centerline_points_3d = points

        # point_set = centerline_points_3d

        # attribute = int(centerline_info["is_intersection_or_connector"]) + 1
        centerline_points_4d_dummy = np.zeros((centerline_points_3d.shape[0], 4), dtype=np.float32)
        centerline_points_4d_dummy[:, 0] = centerline_points_3d[:, 0]
        centerline_points_4d_dummy[:, 1] = centerline_points_3d[:, 1]
        centerline_points_4d_dummy[:, 2] = centerline_points_3d[:, 2]
        centerline_points_4d_dummy[:, 3] = 1
        projected_centerlines = self.project_view[0,0].numpy() @ centerline_points_4d_dummy.T
        projected_centerlines = projected_centerlines[0].T
        return projected_centerlines

    def calculate_view_matrix(self, X_BOUND, Y_BOUND, Z_BOUND):
        import sys
        sys.path.append(
            '/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/tairvision_center_line/')

        import torch
        from tairvision.models.bev.common.utils.geometry import calculate_birds_eye_view_parameters
        from tairvision.datasets.nuscenes import get_view_matrix

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            X_BOUND, Y_BOUND, Z_BOUND
        )

        bev_resolution, bev_start_position, bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        lidar_to_view = get_view_matrix(
            bev_dimension,
            bev_resolution,
            bev_start_position=bev_start_position,
            ego_pos='center'
        )

        view = lidar_to_view[None, None, None]
        view_tensor = torch.tensor(view, dtype=torch.float32)

        return view_tensor

    def set_world(self, _world, _vehicle):
        #self.plant_agent.set_world(_world, _vehicle)
        pass

    def set_info(self,_pixels_per_meter,_world_offset):
        self.plant_agent.set_info(_pixels_per_meter, _world_offset)

    def save_score(self, name, file_name_without_extension, save_files_name, score_composed, score_route, score_penalty):
        self.agent_wrapper.save_score(name, file_name_without_extension, save_files_name, score_composed, score_route, score_penalty)



    def __call__(self, input_data,  bbox, ego_actor, global_plan_gps, global_plan_world_coord, light_hazard, plant_boxes, ego_motion, plant_dataset_collection, tl_list=None, stop_box=None):
        self.plant_agent.traffic_light_hazard = light_hazard

        self._vehicle = ego_actor
        self.plant_agent.set_global_plan(global_plan_gps, global_plan_world_coord)
        self.agent_wrapper.update_sensor_interface(input_data)

        output, pred_wp, target_point,keep_vehicle_ids, keep_vehicle_attn, plant_input_image, detected_input_image = self.agent_wrapper(light_hazard=light_hazard,
                                                                                               bbox=bbox,plant_boxes=plant_boxes,
                                                                                               ego_motion=ego_motion,
                                                                                               plant_dataset_collection=plant_dataset_collection,
                                                                                               tl_list=tl_list, stop_box=stop_box)

        return output, pred_wp, target_point, keep_vehicle_ids, keep_vehicle_attn, plant_input_image, detected_input_image

    def rotate_line(self, start_point, end_point, angle_radians):
        angle_radians = math.radians(math.degrees(angle_radians) + 90)
        # Translate line to origin
        translated_x = end_point[0] - 0
        translated_y = end_point[1] - 0

        # Rotate the line
        rotated_x = translated_x * math.cos(angle_radians) - translated_y * math.sin(angle_radians)
        rotated_y = translated_x * math.sin(angle_radians) + translated_y * math.cos(angle_radians)

        # Translate line back
        new_end_x = start_point[0] + rotated_x
        new_end_y = start_point[1] - rotated_y

        # Return the new end point of the line after rotation
        new_location = carla.Location(x=new_end_x,
                       y=new_end_y,
                       z=start_point[2])

        return new_location

    def crop_array_center(self, original_array, new_height, new_width):
        original_height, original_width = original_array.shape

        # Calculate the starting points for the crop
        start_row = (original_height - new_height) // 2
        start_col = (original_width - new_width) // 2

        # Calculate the ending points for the crop
        end_row = start_row + new_height
        end_col = start_col + new_width

        # Crop the array
        cropped_array = original_array[start_row:end_row, start_col:end_col]

        return cropped_array

    def get_tl_color(self, tl_outputs, tl_index):
        line_color = (255, 0, 0)
        tl_state = -1
        if len(tl_outputs['score_list_lcte'][0][tl_index]) != 0 and np.sum(tl_outputs['score_list_lcte'][0][tl_index] > 1.0) > 0:
            tl_number = np.argmax(tl_outputs['score_list_lcte'][0][tl_index])
            tl_state = tl_outputs['pred_list'][0][tl_number]['attribute']

            if tl_state == 1:
                line_color = (0, 0, 255)
            elif tl_state == 2:
                line_color = (0, 255, 0)
            elif tl_state == 3:
                line_color = (0, 255, 255)

        return line_color, tl_state

    def draw_box(self, stop_masks, center_x, center_y, box_width=25, box_height=25):
        # Calculate top-left and bottom-right points from the center
        start_x = center_x - box_width // 2
        start_y = center_y - box_height // 2
        end_x = center_x + box_width // 2
        end_y = center_y + box_height // 2

        # Step 2: Draw the rectangle
        cv2.rectangle(stop_masks, (start_x, start_y), (end_x, end_y), (255, 0, 0), -1)

        return stop_masks

    def world_to_vehicle(self, world_location, vehicle_transform):
        # Retrieve the vehicle's location and rotation (yaw)
        v_location = vehicle_transform.location
        v_rotation = vehicle_transform.rotation

        # Convert degrees to radians for mathematical operations
        yaw_radians = math.radians(v_rotation.yaw)

        # Create a 2D rotation matrix for the yaw
        # Note: CARLA's coordinate system means we need to negate the angle for the rotation matrix
        rotation_matrix = [
            [math.cos(yaw_radians), math.sin(yaw_radians)],
            [-math.sin(yaw_radians), math.cos(yaw_radians)]
        ]

        # Create translation vector
        translation_vector = [v_location.x, v_location.y]

        # Convert world location to a vector (ignoring Z for simplicity)
        world_vector = [world_location.x, world_location.y]

        # Convert the world location to the vehicle-relative location (2D)
        relative_vector = [
            world_vector[0] - translation_vector[0],
            world_vector[1] - translation_vector[1]
        ]

        # Apply the rotation matrix to the vehicle-relative vector
        vehicle_vector = [
            rotation_matrix[0][0] * relative_vector[0] + rotation_matrix[0][1] * relative_vector[1],
            rotation_matrix[1][0] * relative_vector[0] + rotation_matrix[1][1] * relative_vector[1]
        ]

        # Convert back to a CARLA Location (assuming no change in Z)
        vehicle_location = carla.Location(x=vehicle_vector[1]*(-1), y=vehicle_vector[0], z=0.0)#world_location.z)

        return vehicle_location

    def update_get_transform(self,get_transform):
        self.get_transform = get_transform

    def draw_attention_bb_in_carla(self, input_data,  _world, keep_vehicle_ids, keep_vehicle_attn, ego_location, ego_rotation, plant_inference, guidance_masks, ego_vehicle=None, bev_meters_per_pixel_x=None, bev_meters_per_pixel_y=None, bev_image=None, compass=None, frame_rate_sim=20):
        actors = _world.get_actors()
        carla_map = _world.get_map()
        stop_label = False


        all_vehicles = actors.filter('*vehicle*')

        walker = actors.filter('*pedestrian*')

        crosswalks = carla_map.get_crosswalks()

        all_landmarks = carla_map.get_all_landmarks()
        land_list = []
        for el in all_landmarks:
            land_list.append(int(el.type))
        print(set(land_list))
        traffic_lights = [landmark for landmark in all_landmarks if landmark.type == '1000001']
        stop_signs_0 = [landmark for landmark in all_landmarks if landmark.type == '1000011']
        stop_signs_1 = [landmark for landmark in all_landmarks if landmark.type == '205']
        stop_signs_2 = [landmark for landmark in all_landmarks if landmark.type == '206']
        stop_signs_3 = [landmark for landmark in all_landmarks if landmark.type == '274']
        signs = stop_signs_2 #traffic_lights+stop_signs_0 + stop_signs_1 + stop_signs_2 + stop_signs_3



        tl_list = []
        tl_dummy_list = []
        tl_dict ={}
        for tl in signs:
            tl_dummy_list.append(ego_location.distance(tl.transform.location))
            if ego_location.distance(tl.transform.location) < 30 and tl.name != 'Sign_Stop':
                tl_list.append(tl)
                tl_dict.update({tl.id:tl})

        for tl in list(tl_dict.values()):
            if tl.id in self.check_before_stop:
                continue
            stop_masks = np.zeros([2000, 2000], dtype=np.uint8)
            egp_masks = np.zeros([2000, 2000], dtype=np.uint8)

            #new_ln_location = self.world_to_vehicle(tl.transform.location, ego_vehicle.get_transform())
            new_ln_location = self.world_to_vehicle(tl.transform.location, self.get_transform(input_data, tl.transform))
            projected_route_list = self.project(np.array([[new_ln_location.x, new_ln_location.y, new_ln_location.z]]))

            stop_masks = self.draw_box(stop_masks, center_x=int(projected_route_list[0][1]), center_y=int(projected_route_list[0][0]))
            ego_masks = self.draw_box(egp_masks, center_x=1000, center_y=1000, box_width=50, box_height=50)

            stop_masks = self.crop_array_center(stop_masks, self.real_height, self.real_width)
            ego_masks = self.crop_array_center(ego_masks, self.real_height, self.real_width)

            bev_image[stop_masks > 0] = (0,0,255)

            stop_label = np.sum(stop_masks*ego_masks) > 0
            stop_label_plant = np.sum(stop_masks*guidance_masks) > 0
            if stop_label_plant:
                plant_stop_input = projected_route_list

            if stop_label:
                current_speed = input_data['speed'][1]['speed']
                if current_speed <= 0.0:
                    self.check_before_stop.append(tl.id)
                break

        if plant_inference:

            dynamic_bbox_list = [actor for actor in actors if
                                 actor.type_id == 'static.prop.constructioncone' or actor.type_id == 'static.prop.trafficwarning' or actor.type_id == 'static.prop.warningconstruction' or actor.type_id == 'static.prop.dirtdebris02']

            closest_points_info = []#, carla_loc, _, _, _, _, is_there_obstacle = self.process_radar.show_radar_output(input_data['front_radar'],ego_location=ego_location,compass = input_data['imu'][1][-1])
            #closest_points_info, _, _, _, _, _, is_there_obstacle
            results = []
            if len(closest_points_info) != 0:
                for index, radar_p in enumerate(closest_points_info):#sublist, velocity_list[index], dept_list[index], azu_array[index]
                    sublist, _, velocity, dept, azu = radar_p

                    result = {"class": "Radar",
                              "extent": [1.0, 1.0, 1.0], #TODO:
                              "position": [float(sublist[1])*(-1), float(sublist[-1]), float(0)],
                              "yaw": float(azu),
                              "num_points": -1,
                              "distance": float(dept),
                              "speed": float(velocity),
                              "brake": 0,
                              "id": float(index),
                            }
                    results.append(result)


            for _actor_ in actors:
                if _actor_.type_id not in self.actor_type_list:
                    self.actor_type_list.append(_actor_.type_id)

            merged_list = [all_vehicles, walker]
            for _list in merged_list:
                for actor in _list:
                    dynamic_bbox_list.append(actor)
            dynamic_bbox_list = results + dynamic_bbox_list

            ###deneme
            #if ego_vehicle != None and len(results) != 0:


            for dyn_index, dynamic in enumerate(dynamic_bbox_list):
                try:
                    related_id = dynamic.id
                    loc = dynamic.get_location()
                    extent = dynamic.bounding_box.extent
                    bb_loc = dynamic.get_transform().location
                    rot = dynamic.get_transform().rotation

                    if related_id in keep_vehicle_ids:
                        index = keep_vehicle_ids.index(related_id)
                        # cmap = plt.get_cmap('YlOrRd')
                        # c = cmap(object[1])
                        # color = carla.Color(*[int(i*255) for i in c])
                        c = self.get_color(keep_vehicle_attn[index])
                        color = carla.Color(r=int(c[0]), g=int(c[1]), b=int(c[2]))

                        loc.z = extent.z / 2
                        bb = carla.BoundingBox(loc, extent)
                        bb.extent.z = 0.2
                        bb.extent.x += 0.2
                        bb.extent.y += 0.05

                        bb = carla.BoundingBox(bb_loc, extent)
                        _world.debug.draw_box(box=bb, rotation=rot, thickness=0.07, color=color,
                                              life_time=0.1)

                except:
                    related_id = dynamic["id"]
                    loc = carla.Location(x=ego_location.x, y=ego_location.y, z=ego_location.z) #carla_loc[index] #dynamic.get_location()#ego_location
                    extent = carla.Vector3D(x=3.0, y=3.0, z=3.0)
                    bb_loc = loc #dynamic.get_transform().location
                    rot = ego_rotation

                    # print(vehicle.id)
                    if related_id in keep_vehicle_ids:
                        index = keep_vehicle_ids.index(related_id)
                        # cmap = plt.get_cmap('YlOrRd')
                        # c = cmap(object[1])
                        # color = carla.Color(*[int(i*255) for i in c])
                        c = self.get_color(keep_vehicle_attn[index])
                        color = carla.Color(r=int(c[0]), g=int(c[1]), b=int(c[2]))

                        dynamic = ego_vehicle
                        loc = carla.Location(x=carla_loc[0].x, y=carla_loc[0].y,
                                             z=dynamic.get_location().z)  # dynamic.get_location()
                        extent = carla.Vector3D(x=0.2, y=0.2, z=0.2)  # dynamic.bounding_box.extent
                        bb_loc = carla.Location(x=carla_loc[0].x, y=carla_loc[0].y,
                                                z=dynamic.get_location().z)  # dynamic.get_transform().location
                        rot = ego_rotation  # dynamic.get_transform().rotation

                        #color = carla.Color(255, 0, 0)

                        loc.z = extent.z / 2
                        bb = carla.BoundingBox(loc, extent)
                        bb.extent.z = 0.2
                        bb.extent.x += 0.2
                        bb.extent.y += 0.05

                        bb = carla.BoundingBox(bb_loc, extent)
                        _world.debug.draw_box(box=bb, rotation=rot, thickness=0.07, color=color,
                                              life_time=0.1)

        return stop_label



    def get_color(self, attention):
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


