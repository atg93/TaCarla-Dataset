import numpy as np
import carla
import math
import cv2

from PIL import Image, ImageDraw, ImageFont

from pyproj import Proj, transform



class Rule_Based_Stop_Sign():
    def __init__(self,lane_class, lane_guidance):
        self.initialize = False
        self.lane_class = lane_class
        self.lane_guidance = lane_guidance

        X_BOUND = [-54.0, 54.0, 0.0542]  # Forward
        Y_BOUND = [-54.0, 54.0, 0.0542]  # Sides
        Z_BOUND = [-10.0, 10.0, 5.0]
        self.project_view = self.calculate_view_matrix(X_BOUND, Y_BOUND, Z_BOUND)

    def calculate_view_matrix(self, X_BOUND, Y_BOUND, Z_BOUND):
        import sys
        #sys.path.append(
        #    '/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/tairvision_center_line/')
        #sys.path.append(
        #    '/cta/users/tgorgulu/git/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/tairvision_center_line/')
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

    def get_corresponding_stop_box(self, input_data, ego_location, guidance_masks):

        plant_stop_input = []
        box_size = (0, 0, 0)
        all_landmarks = self.carla_map.get_all_landmarks()
        land_list = []
        for el in all_landmarks:
            land_list.append(int(el.type))
        print(set(land_list))
        traffic_lights = [landmark for landmark in all_landmarks if landmark.type == '1000001']
        stop_signs_0 = [landmark for landmark in all_landmarks if landmark.type == '1000011']
        stop_signs_1 = [landmark for landmark in all_landmarks if landmark.type == '205']
        stop_signs_2 = [landmark for landmark in all_landmarks if landmark.type == '206']
        stop_signs_3 = [landmark for landmark in all_landmarks if landmark.type == '274']
        signs = stop_signs_2  # traffic_lights+stop_signs_0 + stop_signs_1 + stop_signs_2 + stop_signs_3

        tl_list = []
        tl_dummy_list = []
        tl_dict = {}
        for tl in signs:
            tl_dummy_list.append(ego_location.distance(tl.transform.location))
            if ego_location.distance(tl.transform.location) < 30 and tl.name != 'Sign_Stop':
                tl_list.append(tl)
                tl_dict.update({tl.id: tl})

        plant_stop_input_list = []
        for tl in list(tl_dict.values()):
            if tl.id in self.check_before_stop:
                continue
            stop_masks = np.zeros([2000, 2000], dtype=np.uint8)
            egp_masks = np.zeros([2000, 2000], dtype=np.uint8)

            # new_ln_location = self.world_to_vehicle(tl.transform.location, ego_vehicle.get_transform())
            new_ln_location = self.world_to_vehicle(tl.transform.location, self.lane_class.get_transform(input_data, tl.transform))
            projected_route_list, centerline_points_4d_dummy = self.project(np.array([[new_ln_location.x, new_ln_location.y, new_ln_location.z]]))

            stop_masks = self.draw_box(stop_masks, center_x=int(projected_route_list[0][1]),
                                       center_y=int(projected_route_list[0][0]))
            ego_masks = self.draw_box(egp_masks, center_x=1000, center_y=1000, box_width=50, box_height=50)

            stop_masks = self.crop_array_center(stop_masks, self.real_height, self.real_width)
            ego_masks = self.crop_array_center(ego_masks, self.real_height, self.real_width)


            stop_label = np.sum(stop_masks * ego_masks) > 0
            stop_label_plant = np.sum(stop_masks * guidance_masks) > 0
            if stop_label_plant:
                box_size = (2,5,2)
                plant_stop_input_list.append((1, centerline_points_4d_dummy, box_size))

            if stop_label:
                current_speed = input_data['speed'][1]['speed']
                if current_speed <= 0.0:
                    self.check_before_stop.append(tl.id)
                break

        return plant_stop_input_list


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

    def init(self, carla_map):
        self.carla_map = carla_map
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
        self.signs = stop_signs_2 #traffic_lights+stop_signs_0 + stop_signs_1 + stop_signs_2 + stop_signs_3

        self._width = 1845
        self.real_height = 586
        self.real_width = 1034
        self.a = 6378137  # Radius of the Earth in meters (WGS-84)
        self.e = 8.1819190842622e-2  # Eccentricity

        self.f = 1 / 298.257223563  # Flattening factor WGS-84
        self.b = (1 - self.f) * self.a

        X_BOUND = [-54.0, 54.0, 0.0542]  # Forward
        Y_BOUND = [-54.0, 54.0, 0.0542]  # Sides
        Z_BOUND = [-10.0, 10.0, 5.0]
        self.check_before_stop = []

        self.initialize = True


    def __call__(self, input_data, real_ego_location, guidance_masks, bev_image=None):
        if not self.initialize:
            self.init(self.lane_class.opendrive_hd_map)

        tl_list = []
        tl_dict ={}
        stop_label = False

        for tl in self.signs:
            tl_gps_location = self.lane_class.opendrive_hd_map.transform_to_geolocation(tl.transform.location)
            distance = self.lane_class.haversine(input_data['gps'][1][0], input_data['gps'][1][1], tl_gps_location.latitude, tl_gps_location.longitude) #ego_location.distance(tl.transform.location)
            if distance < 30: # and tl.name == 'Sign_Stop':
                tl_list.append(tl) #tl.id#tl.transform.location
                tl_dict.update({tl.id:tl})

        #close_wp_list = self.lane_class.get_close_wp_list()
        transform_list = self.lane_class.get_transform_list()
        tl_to_wp = {}
        for tl in list(tl_dict.values()):#get_related_wp()
            dist_list = []#tl.id:

            if tl.name == 'Sign_Stop':
                if abs(self.lane_class.get_related_wp()[0].rotation.yaw - tl.transform.rotation.yaw % 360) % 360 > 5:
                    print("yaw:",abs(self.lane_class.get_related_wp()[0].rotation.yaw - tl.transform.rotation.yaw % 360))
                    continue

                stop_sign_wp_list = []
                stop_sign_index = -1
                for ss_index, element in enumerate(transform_list):
                    wp, command = element
                    distance = wp.location.distance(tl.transform.location)

                    if distance < 5 and str(command) != 'RoadOption.LANEFOLLOW':
                        stop_sign_wp_list.append((wp, command, ss_index, distance))
                        dist_list.append(distance)
                        break


                if len(dist_list) != 0:
                    index = np.argmin(dist_list)
                    if dist_list[index] < 5:
                        try:
                            ss_index = ss_index-1
                        except:
                            ss_index = 0
                        tl_to_wp.update({tl.id:transform_list[ss_index]})

            else:
                tl_to_wp.update({tl.id: (tl.transform,0)})



        #self.lane_class.get_related_wp()[0].rotation
        key_list = list(tl_to_wp.keys())
        plant_stop_input_list = []
        for key_index, tl in enumerate(list(tl_to_wp.values())):
            tl_id = key_list[key_index]#tl.id
            if tl_id in self.check_before_stop:#tl[0].rotation.yaw
                continue
            tl_transform, _ = tl #tl.transform
            stop_masks = np.zeros([2000, 2000], dtype=np.uint8)
            ego_masks = np.zeros([2000, 2000], dtype=np.uint8)

            #new_ln_location = self.world_to_vehicle(tl.transform.location, ego_vehicle.get_transform())
            transform = self.lane_class.get_transform(input_data, tl_transform, self.lane_class.get_world_coord_index())
            new_ln_location = self.lane_class.world_to_vehicle(tl_transform.location, transform)
            projected_route_list, centerline_points_4d_dummy = self.lane_class.project(np.array([[new_ln_location.x, new_ln_location.y, new_ln_location.z]]))


            stop_masks = self.draw_box(stop_masks, center_x=int(projected_route_list[0][1]), center_y=int(projected_route_list[0][0]))
            ego_masks = self.draw_box(ego_masks, center_x=1000, center_y=1000, box_width=50, box_height=100)

            stop_masks = self.crop_array_center(stop_masks, self.real_height, self.real_width)
            ego_masks = self.crop_array_center(ego_masks, self.real_height, self.real_width)

            if type(bev_image) != type(None) and False:
                bev_image[stop_masks > 0] = (0,0,255)

            stop_label = np.sum(stop_masks*ego_masks) > 0

            stop_label_plant = abs((self.lane_class.plant_current_wp.rotation.yaw%360) - (tl_transform.rotation.yaw%360))<90 and \
                               self.lane_class.plant_current_wp.location.distance(tl_transform.location) < 15

            if transform_list[0][0].location.distance(tl[0].location) < 10:#if tl in transform_list[0:15]:
                box_size = (2, 5, 2)
                plant_stop_input_list.append((1, centerline_points_4d_dummy, box_size))

            if stop_label:
                current_speed = input_data['speed'][1]['speed']
                if current_speed <= 0.0:
                    self.check_before_stop.append(tl_id)
                #break

        if stop_label and False:
            bev_image = Image.fromarray(bev_image)
            draw = ImageDraw.Draw(bev_image)
            font_path = "arial.ttf"  # Replace with a path to your font
            font = ImageFont.truetype(font_path, 50)
            text = "Stop!"
            position = (50, 50)  # Change to your desired position

            # Specify the color
            color = (0, 0, 255)

            # Add text to image
            draw.text(position, text, fill=color, font=font)

        return stop_label, np.array(bev_image), plant_stop_input_list

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


    def draw_box(self, stop_masks, center_x, center_y, box_width=25, box_height=25):
        # Calculate top-left and bottom-right points from the center
        start_x = center_x - box_width // 2
        start_y = center_y - box_height // 2
        end_x = center_x + box_width // 2
        end_y = center_y + box_height // 2

        # Step 2: Draw the rectangle
        cv2.rectangle(stop_masks, (start_x, start_y), (end_x, end_y), (255, 0, 0), -1)

        return stop_masks

