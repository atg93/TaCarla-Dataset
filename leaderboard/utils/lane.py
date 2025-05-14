from tair_traditional_agents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.nav_planner import interpolate_trajectory
import carla
import numpy as np
import math
import copy
import os
import sys
import cv2


class Lane():
    def __init__(self):
        self.opendrive_hd_map = None
        self.lane_guidance_init = False
        self.a = 6378137  # Radius of the Earth in meters
        self.e = 8.1819190842622e-2
        self.real_height = 586
        self.real_width = 1034
        self.close_hd_index = 0

    def get_lane_guidance(self):
        self.lane_guidance = self.transform_list[
                             self._global_plan_world_coord_index:self._global_plan_world_coord_index + 20]

        return self.lane_guidance

    def get_close_wp_list(self):
        return self.close_wp_list

    def get_transform_list(self):
        return self.transform_list[self._global_plan_world_coord_index:self._global_plan_world_coord_index+50]

    def get_related_wp(self):
        return self.transform_list[self._global_plan_world_coord_index]

    def get_world_coord_index(self):
        return self._global_plan_world_coord_index

    def init(self, _global_plan_world_coord):
        trajectory = [item[0] for item in _global_plan_world_coord]
        _, self.transform_list = interpolate_trajectory(self.opendrive_hd_map, trajectory, max_len=1000)
        self.plant_global_plan_world_coord = []
        self.plant_global_plan_gps = []

        for wp, direction in self.transform_list:
            self.plant_global_plan_world_coord.append((wp, direction))
            self.plant_global_plan_gps.append((self.opendrive_hd_map.transform_to_geolocation(wp.location), direction))


        X_BOUND = [-54.0, 54.0, 0.0542]  # Forward
        Y_BOUND = [-54.0, 54.0, 0.0542]  # Sides
        Z_BOUND = [-10.0, 10.0, 5.0]
        self.project_view = self.calculate_view_matrix(X_BOUND, Y_BOUND, Z_BOUND)

        self.real_height = 586
        self.real_width = 1034

        self.waypoints = self.opendrive_hd_map.generate_waypoints(distance=1.0)

    def calculate_view_matrix(self, X_BOUND, Y_BOUND, Z_BOUND):

        import torch
        from lane_utils.geometry import calculate_birds_eye_view_parameters
        from lane_utils.geometry import get_view_matrix

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

    def read_hd_map(self, input_data, _global_plan_world_coord):
        if self.opendrive_hd_map == None:
            self.opendrive_hd_map = carla.Map("RouteMap", input_data['hd_map'][1]['opendrive'])

            self.init(_global_plan_world_coord)

    def update_lane(self, input_data):
        self._global_plan_world_coord_index, self.close_wp_list = self.get_related_wp_index(input_data)


    def __call__(self, input_data, _global_plan_world_coord):
        self.read_hd_map(input_data, _global_plan_world_coord)
        self.update_lane(input_data)
        self.compute_close_hd_wp_list(input_data)

    def compute_close_hd_wp_list(self, input_data):

        if self.close_hd_index % 10 == 0:
            self.close_hd_wp_list = []
            self.close_hd_wp_list_mcts = []
            self.close_hd_wp_center_list_mcts = []

            for index, wp in enumerate(self.waypoints):
                wp_location = wp.transform.location
                tl_gps_location = self.opendrive_hd_map.transform_to_geolocation(wp_location)
                distance = self.haversine(input_data['gps'][1][0], input_data['gps'][1][1], tl_gps_location.latitude,
                                          tl_gps_location.longitude)  # ego_location.distance(tl.transform.location)

                if distance < 500:
                    self.close_hd_wp_list.append(wp)

                if distance < 30 and str(wp.lane_type) == 'Driving':
                    self.close_hd_wp_list_mcts.append((wp.transform, 'dummy'))
                    self.close_hd_wp_center_list_mcts.append(wp)

        self.close_hd_index += 1

        dist_list = []
        for index, wp in enumerate(self.close_hd_wp_list):
            wp_location = wp.transform.location
            tl_gps_location = self.opendrive_hd_map.transform_to_geolocation(wp_location)
            distance = self.haversine(input_data['gps'][1][0], input_data['gps'][1][1], tl_gps_location.latitude,
                                      tl_gps_location.longitude)  # ego_location.distance(tl.transform.location)

            dist_list.append(distance)

        wp_index = np.argmin(dist_list)
        self.hd_curret_wp = self.close_hd_wp_list[wp_index]

        asd = 0



    def get_related_wp_index(self, input_data):
        dist_list = []
        close_wp_list = []
        for index, transform in enumerate(self.transform_list):
            wp_location = transform[0].location
            tl_gps_location = self.opendrive_hd_map.transform_to_geolocation(wp_location)
            distance = self.haversine(input_data['gps'][1][0], input_data['gps'][1][1], tl_gps_location.latitude,
                                      tl_gps_location.longitude)  # ego_location.distance(tl.transform.location)

            dist_list.append(distance)
            if distance < 30:
                close_wp_list.append(transform)
        wp_index = None
        if len(dist_list) != 0:
            wp_index = np.argmin(dist_list)

        self.current_wp = self.transform_list[wp_index][0]

        return wp_index, close_wp_list

    def haversine(self, lat1, lon1, lat2, lon2):
        # Radius of the Earth in kilometers. Use 3956 for miles
        R = 6371.0

        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Difference in coordinates
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine formula
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c

        return distance * 1000

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
        return projected_centerlines, centerline_points_4d_dummy


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


    def get_closest_projected_lane(self, input_data, vehicle):
        closest_wp_list = self.transform_list[self._global_plan_world_coord_index:self._global_plan_world_coord_index+30]#self.get_related_wp_index(input_data)
        #self.close_hd_wp_center_list_mcts, self.hd_curret_wp
        center_project_list, wp_image = self.get_l_c_r_centers(vehicle)
        projected_mask, project_list = self.calculate_mcts_lane(vehicle, closest_wp_list+self.close_hd_wp_list_mcts)#self.close_hd_wp_list_mcts

        return projected_mask, project_list, center_project_list, wp_image #self.close_hd_wp_list_mcts

    def get_wp_on_road(self, wp, resolution=2):
        wp_list = [(wp.transform, 'dummy')]
        wp_org = wp
        for index in range(30):
            try:
                if index < 15:
                    new_wp = wp.next(resolution)[0]
                else:
                    new_wp = wp.previous(resolution)[0]

                if str(new_wp.lane_type) == 'Driving':
                    wp_list.append((new_wp.transform, 'dummy'))#str(wp.lane_type) == 'Driving'

                wp = new_wp
                if index == 16:
                    wp = wp_org
            except:
                pass

        return wp_list

    def get_l_c_r_centers(self, vehicle):

        current_list = self.get_wp_on_road(self.hd_curret_wp)
        if type(None) != type(self.hd_curret_wp.get_left_lane()):
            current_list += self.get_wp_on_road(self.hd_curret_wp.get_left_lane())

        if type(None) != type(self.hd_curret_wp.get_right_lane()):
            current_list += self.get_wp_on_road(self.hd_curret_wp.get_right_lane())

        _, project_list = self.calculate_mcts_lane(vehicle, current_list)

        current_list_0 = self.get_wp_on_road(self.hd_curret_wp)
        _, project_list_0 = self.calculate_mcts_lane(vehicle, current_list_0)
        project_list_1, project_list_2 = [], []

        if type(None) != type(self.hd_curret_wp.get_left_lane()):
            current_list_1 = self.get_wp_on_road(self.hd_curret_wp.get_left_lane())
            _, project_list_1 = self.calculate_mcts_lane(vehicle, current_list_1)

        if type(None) != type(self.hd_curret_wp.get_right_lane()):
            current_list_2 = self.get_wp_on_road(self.hd_curret_wp.get_right_lane())
            _, project_list_2 = self.calculate_mcts_lane(vehicle, current_list_2)


        asd = 0
        wp_image = np.zeros((200, 200), dtype=np.uint8)
        new_por_list = []
        for project_list in [project_list_0, project_list_1, project_list_2]:
            if len(project_list) >= 1:
                for point in project_list[0]:
                    if point[1] < 100:
                        cv2.circle(wp_image, tuple(np.array(point)), 1, (255), 1)
                        new_por_list.append(point)
        cv2.circle(wp_image, tuple(np.array([100,100])), 1, (255), 3)
        #cv2.imwrite("wp_image_get_l_c_r_centers.png", wp_image)
        project_list = (new_por_list, [], [])

        try:
            #project_list = (new_por_list,project_list[1],project_list[2])
            pass
        except:
            project_list = (new_por_list, [], [])


        return project_list, wp_image




    def calculate_mcts_lane(self, vehicle, closest_wp_list):
        ego_yaw = vehicle.get_transform().rotation.yaw / 180 * np.pi
        ego_matrix = np.array(vehicle.get_transform().get_matrix())
        relative_wp_list = []
        closest_wp_loc_array = []
        closest_wp_rot_array = []
        for wp in closest_wp_list:
            world_loc = wp[0].location
            world_rot = wp[0].rotation
            closest_wp_loc_array.append(np.array([world_loc.x, world_loc.y, world_loc.z]))
            closest_wp_rot_array.append(np.array([world_rot.pitch, world_rot.roll, world_rot.yaw]))
            point_matrix = np.array(wp[0].get_matrix())

            yaw = wp[0].rotation.yaw / 180 * np.pi

            # relative_yaw = yaw - ego_yaw
            relative_pos = self.get_relative_transform(ego_matrix, point_matrix)
            relative_wp_list.append(relative_pos)

        projected_mask = np.zeros((200, 200), dtype=np.uint8)
        project_list = []
        for wp in relative_wp_list:
            pred_wp = wp[0:2] * (-1)
            center = pred_wp * 4 + 100
            center = center[::-1]
            project_list.append(copy.deepcopy(center.astype(np.uint8).astype(np.int32)))
            cv2.circle(projected_mask, tuple(center.astype(np.uint8).astype(np.int32)), 5, (255), -1)
        #cv2.imwrite('projected_mask_2.png', projected_mask)


        pts = np.array(project_list).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Draw the rotated rectangle on the image
        projected_mask = cv2.polylines(projected_mask, [pts], isClosed=True, color=(255), thickness=1)
        #cv2.imwrite('projected_mask_dummy.png', projected_mask)

        project_list = (project_list, np.array(closest_wp_loc_array), np.array(closest_wp_rot_array))

        return projected_mask, project_list


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

    def get_transform(self, input_data, ln, real_ego_location=None, wp_index=None):
        yaw = self.convert_heading_to_yaw(math.degrees(input_data['imu'][1][-1]))

        _global_plan_world_coord_index = wp_index#, close_wp_list = self.get_related_wp_index(input_data)

        current_wp, _ = self.transform_list[self._global_plan_world_coord_index]
        self.plant_current_wp = current_wp
        calculated_transform_vector = carla.Transform(
            rotation=carla.Rotation(pitch=ln.rotation.pitch, roll=ln.rotation.roll, yaw=yaw),
            location=current_wp.location)

        return calculated_transform_vector

    def convert_heading_to_yaw(self, heading):
        """
        Convert a geographical heading (degrees clockwise from North) into a simulation yaw angle
        (degrees counterclockwise from East).

        Parameters:
            heading (float): The geographical heading in degrees.

        Returns:
            float: The corresponding yaw angle in the simulation environment.
        """
        # Convert from degrees clockwise from North to degrees counterclockwise from East
        yaw = heading - 90
        # Normalize the yaw angle to the range [0, 360)
        yaw = yaw % 360
        return yaw