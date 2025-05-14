from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.nav_planner import interpolate_trajectory
import carla
import numpy as np
import cv2
import math


class Lane_Guidance():
    def __init__(self):
        self.opendrive_hd_map = None
        self.lane_guidance_init = False
        self.a = 6378137  # Radius of the Earth in meters
        self.e = 8.1819190842622e-2  # Eccentricity


    def init(self, _global_plan_world_coord):
        trajectory = [item[0] for item in _global_plan_world_coord]
        _, self.transform_list = interpolate_trajectory(self.opendrive_hd_map, trajectory, max_len=1000)
        X_BOUND = [-54.0, 54.0, 0.0542]  # Forward
        Y_BOUND = [-54.0, 54.0, 0.0542]  # Sides
        Z_BOUND = [-10.0, 10.0, 5.0]
        self.project_view = self.calculate_view_matrix(X_BOUND, Y_BOUND, Z_BOUND)

        self.real_height = 586
        self.real_width = 1034

    def get_lane_masks(self, input_data):
        waypoints = self.opendrive_hd_map.generate_waypoints(distance=1.0)

        calculated_location = self.gps_to_enu(input_data['gps'][1][1], input_data['gps'][1][0],
                                              input_data['gps'][1][2])  # input_data['gps'][1]

        ego_location = carla.Location(x=calculated_location[0], y=calculated_location[1], z=calculated_location[2])

        # Image dimensions (for example, 100x100 pixels)
        image_height = 2000
        image_width = 2000

        # Waypoint information (example values)
        waypoint_center = (50, 50)  # (x, y) position in pixels
        lane_width = 20  # In pixels

        # Create a blank binary image
        lane_masks = np.zeros([image_height, image_width], dtype=np.uint8)
        import time
        start_time = time.time()
        for waypoint in waypoints:
            lane_type = waypoint.lane_type
            if waypoint.transform.location.distance(ego_location) > 50 or str(waypoint.lane_type) != 'Driving':
                continue

            lane_id = waypoint.lane_id  # ID of the lane
            lane_width = (waypoint.lane_width * 10)/4 # Width of the lane

            new_ln_location = self.world_to_vehicle(waypoint.transform.location, self.get_transform(input_data, waypoint.transform))
            projected_route_list = self.project(np.array([[new_ln_location.x, new_ln_location.y, new_ln_location.z]]))

            waypoint_center = (projected_route_list[0][0], projected_route_list[0][1])
            # Calculate the lane boundaries based on the waypoint center and lane width
            left_lane_boundary = int(waypoint_center[0] - lane_width / 2)
            right_lane_boundary = int(waypoint_center[0] + lane_width / 2)

            # Draw the lane in the binary image
            #lane_masks[left_lane_boundary, int(waypoint_center[1])] = 255
            #lane_masks[right_lane_boundary, int(waypoint_center[1])] = 255
            #lane_masks = self.draw_box(lane_masks, int(waypoint_center[1]), int(waypoint_center[0]))
            lane_masks = self.draw_box(lane_masks, int(waypoint_center[1]), right_lane_boundary)
            lane_masks = self.draw_box(lane_masks, int(waypoint_center[1]), left_lane_boundary)

        cv2.imwrite('lane_masks.png',lane_masks)
        end_time = time.time()
        print("end_time-start_time: ", end_time-start_time)

        return lane_masks

    def draw_box(self, stop_masks, center_x, center_y, box_width=10, box_height=10):
        # Calculate top-left and bottom-right points from the center
        start_x = center_x - box_width // 2
        start_y = center_y - box_height // 2
        end_x = center_x + box_width // 2
        end_y = center_y + box_height // 2

        # Step 2: Draw the rectangle
        cv2.rectangle(stop_masks, (start_x, start_y), (end_x, end_y), (255, 0, 0), -1)

        return stop_masks

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

    def read_hd_map(self, input_data, _global_plan_world_coord):
        if self.opendrive_hd_map == None:
            self.opendrive_hd_map = carla.Map("RouteMap", input_data['hd_map'][1]['opendrive'])

            self.init(_global_plan_world_coord)

    def get_related_wp_index(self, input_data):
        dist_list = []
        for index, transform in enumerate(self.transform_list):
            wp_location = transform[0].location
            tl_gps_location = self.carla_map.transform_to_geolocation(transform.location)
            distance = self.haversine(input_data['gps'][1][0], input_data['gps'][1][1], tl_gps_location.latitude,
                                      tl_gps_location.longitude)  # ego_location.distance(tl.transform.location)

            dist_list.append(distance)
        wp_index = np.argmin(dist_list)

        return wp_index



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
        calculated_location = self.gps_to_enu(input_data['gps'][1][0], input_data['gps'][1][1],
                                              input_data['gps'][1][2])  # input_data['gps'][1]
        ego_location = carla.Location(x=calculated_location[1] * (-1), y=calculated_location[0] * (-1),
                                      z=calculated_location[2])  # calculated_location

        #if type(real_ego_location) != type(None):
        #    ego_location = real_ego_location

        if type(wp_index) == type(None):
            _global_plan_world_coord_index = self.get_related_wp_index(input_data)
        else:
            _global_plan_world_coord_index = wp_index

        current_wp, _ = self.transform_list[_global_plan_world_coord_index]

        calculated_transform_vector = carla.Transform(
            rotation=carla.Rotation(pitch=ln.rotation.pitch, roll=ln.rotation.roll, yaw=yaw),
            location=current_wp.location)

        return calculated_transform_vector





    def __call__(self, input_data, _global_plan_world_coord):
        guidance_masks = np.zeros([2000, 2000], dtype=np.uint8)
        calculated_location = self.gps_to_enu(input_data['gps'][1][1],input_data['gps'][1][0],input_data['gps'][1][2])#input_data['gps'][1]
        #self.ori_ego_location = _vehicle.get_location()

        self.ego_location = carla.Location(x=calculated_location[0], y=calculated_location[1], z=calculated_location[2])# calculated_location

        self.read_hd_map(input_data, _global_plan_world_coord)

        #self.get_lane_masks(input_data)

        self._global_plan_world_coord_index = self.get_related_wp_index(input_data)
        self.lane_guidance = self.transform_list[
                             self._global_plan_world_coord_index:self._global_plan_world_coord_index + 6]
        current_wp, _ = self.transform_list[self._global_plan_world_coord_index]
        lane_list = []
        for ln, _ in self.lane_guidance:
            #transform_vector = _vehicle.get_transform()
            yaw = self.convert_heading_to_yaw(math.degrees(input_data['imu'][1][-1]))
            calculated_transform_vector = carla.Transform(rotation=carla.Rotation(pitch=ln.rotation.pitch,roll=ln.rotation.roll,yaw=yaw),location=current_wp.location)
            new_ln_location = self.world_to_vehicle(ln.location, calculated_transform_vector)
            lane_list.append([new_ln_location.x, new_ln_location.y, new_ln_location.z])
        projected_route_list = self.project(np.array(lane_list))

        for index, gm in enumerate(projected_route_list):
            try:
                y = int(gm[0])
                x = int(gm[1])

                y_1 = int(projected_route_list[index + 1][0])
                x_1 = int(projected_route_list[index + 1][1])

                cv2.line(guidance_masks, (x, y), (x_1, y_1), (255, 255, 255), 30)
            except:
                pass

        guidance_masks = self.crop_array_center(guidance_masks, self.real_height, self.real_width)

        return guidance_masks

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

    # Function to convert GPS to ECEF (Earth-Centered, Earth-Fixed)
    def gps_to_ecef(self, lat, lon, alt):
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        N = self.a / math.sqrt(1 - self.e ** 2 * math.sin(lat_rad) ** 2)

        x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
        y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
        z = ((1 - self.e ** 2) * N + alt) * math.sin(lat_rad)
        return x, y, z

    # Function to convert ECEF to ENU
    def ecef_to_enu(self, x, y, z, lat0, lon0, alt0):
        lat0_rad = math.radians(lat0)
        lon0_rad = math.radians(lon0)
        N0 = self.a / math.sqrt(1 - self.e ** 2 * math.sin(lat0_rad) ** 2)
        x0, y0, z0 = self.gps_to_ecef(lat0, lon0, alt0)

        xd, yd, zd = x - x0, y - y0, z - z0

        t = -math.sin(lon0_rad) * xd + math.cos(lon0_rad) * yd
        e = -math.sin(lat0_rad) * math.cos(lon0_rad) * xd - math.sin(lat0_rad) * math.sin(lon0_rad) * yd + math.cos(
            lat0_rad) * zd
        n = math.cos(lat0_rad) * math.cos(lon0_rad) * xd + math.cos(lat0_rad) * math.sin(lon0_rad) * yd + math.sin(
            lat0_rad) * zd
        return e, t*(-1), n

    # Function to convert from GPS (lat, lon, alt) to ENU coordinates
    def gps_to_enu(self, lat, lon, alt, lat0=0.0, lon0=0.0, alt0=0.0):
        x, y, z = self.gps_to_ecef(lat, lon, alt)
        return self.ecef_to_enu(x, y, z, lat0, lon0, alt0)