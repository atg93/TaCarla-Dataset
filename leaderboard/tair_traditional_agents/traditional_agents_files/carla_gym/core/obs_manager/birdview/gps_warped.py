from collections import deque
import math
import numpy as np
import cv2

import pyproj
from geographiclib.geodesic import Geodesic


class Gps_Warped:
    def __init__(self,image_width, pixels_per_meter, _world_offset, _history_idx, que_maxlen=20):
        self.image_width = image_width
        self._pixels_per_meter = pixels_per_meter
        self._history_queue = deque(maxlen=que_maxlen)
        self._world_offset = _world_offset
        self._history_idx = _history_idx
        self.geodesic = pyproj.Geod(ellps='WGS84')
        self.prev_ego_coor = None
        self.update_compass = 10
        self.update_compass_count = 0
        self.dandik_pusula = 0.0

    def add_que(self, bounding_boxes_list, vec_mask_world_coordinate, vec_mask_object_word_cartesian, ego_pixel_coordinate, ego_loc, compass):
        """
        center_pixel, orientation, height_width
        """
        item_list = []
        for index, bb in enumerate(bounding_boxes_list):
            new_cor, bearing, object_compass = self.warp_based_on_pixel_4(vec_mask_world_coordinate[index], np.array([bb]), ego_pixel_coordinate, ego_loc, compass)
            item_list.append((new_cor, bearing, bb[1][0], bb[1][1], object_compass,bb[0], vec_mask_world_coordinate[index],vec_mask_object_word_cartesian[index]))
        self._history_queue.append(item_list)
        asd = 0

    def get_que_item(self, ego_pixel_coordinate, ego_loc, ev_mask_object_word_cartesian, compass):
        qsize = len(self._history_queue)
        warped_boxes_list = []
        mask = np.zeros([self.image_width, self.image_width], dtype=np.uint8)

        for idx in self._history_idx:
            idx = max(idx, -1 * qsize)
            self.ego_coor = self.cartesian_to_gps(ev_mask_object_word_cartesian[0][0],ev_mask_object_word_cartesian[0][1],ev_mask_object_word_cartesian[0][2])
            if type(self.prev_ego_coor) != type(None):
                if np.sum(np.array([self.prev_ego_coor[0], self.prev_ego_coor[1]]) - np.array([self.ego_coor[0], self.ego_coor[1]])) > 0.0:
                    self.dandik_pusula = self.calculate_bearing(self.prev_ego_coor[0], self.prev_ego_coor[1],self.ego_coor[0], self.ego_coor[1])
                    #compass = math.radians(self.dandik_pusula)
                    print("dandik_pusula:",self.dandik_pusula,"simulasyon_pasula:",math.degrees(compass))
            items = self._history_queue[idx]
            print("len(items):",len(items))
            for obj_gps in items:
                center, orientation, height, width, distance = self.get_warped_boxes(obj_gps, ego_pixel_coordinate, ego_loc, ev_mask_object_word_cartesian, compass)
                check = center[0]<self.image_width and center[0]>0 and center[1]<self.image_width and center[1]>0
                warped_boxes_list.append((idx, center, orientation))
                corners_warped = self.calculate_bounding_box_corners_0(center_x=center[0], center_y=center[1], width=7, height=7, angle_degrees=orientation)
                cv2.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
                #cv2.rectangle(mask, (int(center[0]-one_lenght/2),int(center[1]-one_lenght/2)), (int(center[0] + one_lenght/2), int(center[1] + one_lenght/2)), (255), -1)
            if self.update_compass_count % self.update_compass == 0:
                self.prev_ego_coor = self.ego_coor
            self.update_compass_count += 1

        return warped_boxes_list, mask.astype(np.bool)

    def cartesian_to_gps(self, x, y, z):
        # Define the UTM zone and datum for your coordinates
        utm_zone = 33  # For example, UTM zone 33 for central Europe
        datum = 'WGS84'  # World Geodetic System 1984

        # Create a pyproj transformer for the conversion
        transformer = pyproj.Transformer.from_crs(f'+proj=utm +zone={utm_zone} +datum={datum}', 'EPSG:4326',
                                                  always_xy=True)

        # Convert Cartesian coordinates (X, Y, Z) to GPS coordinates (latitude, longitude, altitude)
        lon, lat, alt = transformer.transform(x, y, z)

        return lat, lon, alt

    def draw_line(self, point1, point2, name, image=None):
        if type(image) == type(None):
            image = np.zeros([self.image_width, self.image_width], dtype=np.uint8)
        color = (255, 255, 255)  # Color of the line in BGR format (here, it's green)
        thickness = 2  # Thickness of the line
        point1, point2 = (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1]))
        cv2.line(image, point1, point2, color=255, thickness=2)

        # Save the output image with the line drawn
        #output_path = "output_image.jpg"  # Replace this with the desired output path and filename
        cv2.imwrite(name, image)
        return image

    def calculate_aspect_angle(self, lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Calculate differences in longitude and latitude
        delta_lon = lon2_rad - lon1_rad
        delta_lat = lat2_rad - lat1_rad

        # Calculate aspect angle in radians using arctangent
        aspect_angle_rad = math.atan2(delta_lon, delta_lat)

        # Convert aspect angle from radians to degrees and ensure it's between 0 and 360
        aspect_angle_deg = (math.degrees(aspect_angle_rad) + 360) % 360

        return aspect_angle_deg

    def calculate_slope(self, x1, y1, x2, y2):
        if x2 - x1 == 0:
            return float('inf')  # Vertical line, slope is infinity
        else:
            return (y2 - y1) / (x2 - x1)

    def calculate_angle_between_lines(self, slope1, slope2):
        angle_rad = math.atan(abs((slope2 - slope1) / (1 + slope1 * slope2)))
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def gps_to_cartesian(self, vector):
        lat, lon, alt = vector[0],vector[1],0
        # Define the UTM zone and datum for your coordinates
        utm_zone = 33  # For example, UTM zone 33 for central Europe
        datum = 'WGS84'  # World Geodetic System 1984

        # Create a pyproj transformer for the conversion
        transformer = pyproj.Transformer.from_crs(f'EPSG:4326', f'+proj=utm +zone={utm_zone} +datum={datum}',
                                                  always_xy=True)

        # Convert GPS coordinates (latitude, longitude, altitude) to Cartesian coordinates (X, Y, Z)
        x, y, z = transformer.transform(lon, lat, alt)

        return np.array([x, y, z])

    def get_warped_boxes(self, obj_gps, ego_pixel_coordinate, ego_loc, ev_mask_object_word_cartesian, compass):
        compass = math.degrees(compass)

        obj_lat, obj_lon, bearing, height, width, object_compass, bb, obj_world_loc, obj_world_cartesian = obj_gps[0][0], obj_gps[0][1], obj_gps[1], obj_gps[2], obj_gps[3], obj_gps[4], obj_gps[5], obj_gps[6], obj_gps[7]
        distance_1 = self.haversine(ego_loc[0][0], ego_loc[0][1],obj_lat, obj_lon) * 1000

        reference_point = self.calculate_new_coordinates(ego_loc[0][0], ego_loc[0][1], distance_1/1000, compass)

        ego_catesian = self.gps_to_cartesian(ego_loc[0])
        object_catesian = self.gps_to_cartesian(np.array([obj_world_loc[0], obj_world_loc[1], obj_world_loc[2]]))
        reference_catesian = self.gps_to_cartesian(np.array(reference_point))
        aspect_angle = self.find_clockwise_angle(ego_loc[0], np.array([obj_lat, obj_lon]), ego_loc[0], np.array(reference_point))

        x = (distance_1 * self._pixels_per_meter) * math.sin(math.radians(aspect_angle)) * (-1)
        y = (distance_1 * self._pixels_per_meter) * math.cos(math.radians(aspect_angle)) * (-1)
        current_ego_pixels = ego_pixel_coordinate[0][0]
        pixel_coordinates = np.array([current_ego_pixels + np.array([x, y])])[0]
        real_pixel_coordinate = bb
        box_orientation = (object_compass-compass) + ego_pixel_coordinate[0][2]

        self.prev_gps = ego_loc[0]

        return pixel_coordinates, box_orientation, abs(height), abs(width), distance_1

    def calculate_new_coordinate_for_boxes(self, distance, bearing_angle_degrees, initial_coordinate):
        # Convert bearing angle from degrees to radians
        bearing_angle_radians = math.radians(bearing_angle_degrees)

        # Extract initial coordinates
        x1, y1 = initial_coordinate

        # Calculate new coordinates using trigonometric functions
        x2 = x1 + (distance * math.sin(bearing_angle_radians)*self._pixels_per_meter)
        y2 = y1 + (distance * math.cos(bearing_angle_radians)*self._pixels_per_meter)

        # Return the new coordinates as a tuple
        return x2, y2

    def calculate_vector_from_points(self, point1, point2):
        return (point2[0] - point1[0], point2[1] - point1[1])

    def rotate_vector(self, vector, angle_degrees):
        # Convert angle from degrees to radians
        angle_radians = math.radians(angle_degrees)

        # Calculate rotated components using trigonometric functions
        x = vector[0] * math.cos(angle_radians) - vector[1] * math.sin(angle_radians)
        y = vector[0] * math.sin(angle_radians) + vector[1] * math.cos(angle_radians)

        return x, y

    def find_second_vector(self, angle_degrees, first_vector):
        # Convert angle from degrees to radians
        angle_radians = math.radians(angle_degrees)

        # Extract components of the first vector
        x1, y1 = first_vector

        # Calculate components of the second vector using trigonometric functions
        x2 = x1 * math.cos(angle_radians) - y1 * math.sin(angle_radians)
        y2 = x1 * math.sin(angle_radians) + y1 * math.cos(angle_radians)

        # Return the components of the second vector
        return x2, y2



    def warp_based_on_pixel_4(self, vec_mask_world_coordinate, object_coordinate, ego_pixel_coordinate, ego_loc, compass):
        compass = math.degrees(compass)

        new_angle_1 = compass
        new_lat, new_lon = self.calculate_new_coordinates(ego_loc[0][0], ego_loc[0][1], 100 / 1000, 0)

        distance_1 = self.haversine(ego_loc[0][0], ego_loc[0][1], new_lat, new_lon) * 1000
        #new_angle_1 = 0
        x = distance_1 * math.sin(math.radians(new_angle_1)) * (-1)
        y = distance_1 * math.cos(math.radians(new_angle_1)) * (-1)

        reference_point = np.array(
            [ego_pixel_coordinate[0][0] + np.array([x * self._pixels_per_meter, y * self._pixels_per_meter])])
        aspect_angle = self.find_clockwise_angle(ego_pixel_coordinate[0][0], reference_point[0],
                                                 ego_pixel_coordinate[0][0], object_coordinate[0][0])

        real_bearing = self.calculate_bearing(ego_loc[0][0], ego_loc[0][1] ,vec_mask_world_coordinate[0],vec_mask_world_coordinate[1])
        print("diff_bearing:",real_bearing-aspect_angle)
        bearing = aspect_angle

        object_cor = self._pixel_to_world((object_coordinate[0][0][0], object_coordinate[0][0][1]))
        ego_cor = self._pixel_to_world((ego_pixel_coordinate[0][0][0], ego_pixel_coordinate[0][0][1]))
        new_distance, new_x, new_y = self.calculate_distance(ego_cor[0], ego_cor[1], object_cor[0], object_cor[1])
        new_lat, new_lon = self.calculate_new_coordinates(ego_loc[0][0], ego_loc[0][1], new_distance/1000, (bearing))
        object_orientation = object_coordinate[0][2]

        object_compass = (object_orientation - ego_pixel_coordinate[0][2]) + compass
        
        #return np.array([vec_mask_world_coordinate[0],vec_mask_world_coordinate[1]]), bearing, object_compass
        return np.array([new_lat, new_lon]), bearing, object_compass

    def find_angle_2(self,line1_point1, line1_point2, line2_point1, line2_point2):
        # Calculate direction vectors of the lines
        line1_vector = self.calculate_direction_vector(line1_point1, line1_point2)
        line2_vector = self.calculate_direction_vector(line2_point1, line2_point2)
        dot_product = line1_vector[0] * line2_vector[0] + line1_vector[1] * line2_vector[1]
        #(line1_vector[0])**2 + (line2_vector[0])**2 + (line1_vector[1])**2 + (line2_vector[1])**2
        magnitude = np.sqrt((line1_vector[0])**2+(line2_vector[0])**2) * np.sqrt((line1_vector[1])**2+(line2_vector[1])**2)
        angle = math.degrees(math.acos((dot_product/magnitude)))
        asd = 0


    def calculate_direction_vector(self, point1, point2):
        return (point2[0] - point1[0], point2[1] - point1[1])


    def find_clockwise_angle(self, line1_point1, line1_point2, line2_point1, line2_point2):
        # Calculate direction vectors of the lines
        line1_vector = self.calculate_direction_vector(line1_point1, line1_point2)
        line2_vector = self.calculate_direction_vector(line2_point1, line2_point2)

        # Calculate dot product and cross product of the direction vectors
        dot_product = line1_vector[0] * line2_vector[0] + line1_vector[1] * line2_vector[1]
        cross_product = line1_vector[0] * line2_vector[1] - line1_vector[1] * line2_vector[0]

        # Calculate angle in radians using atan2 function
        angle_radians = math.atan2(cross_product, dot_product)

        # Convert radians to degrees and ensure the result is positive
        angle_degrees = math.degrees(angle_radians) % 360

        # Return the clockwise angle between the two lines
        return angle_degrees

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        # Calculate differences in longitude and latitude
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

        # Calculate bearing angle in radians
        bearing_angle_rad = math.atan2(y, x)

        # Convert bearing angle from radians to degrees
        bearing_angle_deg = math.degrees(bearing_angle_rad)

        # Ensure the angle is between 0 and 360 degrees
        bearing_angle_deg = (bearing_angle_deg + 360) % 360

        return bearing_angle_deg

    def haversine(self, lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        distance = c * r

        return distance
    
    def calculate_new_coordinates(self, lat, lon, distance, angle_degrees):
        # Earth radius in kilometers
        earth_radius = 6371.0

        # Convert latitude and longitude from degrees to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        # Convert distance from kilometers to radians
        distance_rad = distance / earth_radius

        # Convert angle from degrees to radians
        angle_rad = math.radians(angle_degrees)

        # Calculate new latitude and longitude
        new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(distance_rad) +
                                math.cos(lat_rad) * math.sin(distance_rad) * math.cos(angle_rad))
        new_lon_rad = lon_rad + math.atan2(math.sin(angle_rad) * math.sin(distance_rad) * math.cos(lat_rad),
                                           math.cos(distance_rad) - math.sin(lat_rad) * math.sin(new_lat_rad))

        # Convert new latitude and longitude from radians to degrees
        new_lat = math.degrees(new_lat_rad)
        new_lon = math.degrees(new_lon_rad)

        return new_lat, new_lon

    def _pixel_to_world(self, location, projective=False):
        x = location[0]
        y = location[1]

        """Converts the world coordinates to pixel coordinates"""
        x = (x / self._pixels_per_meter) + self._world_offset[0]
        y = (y / self._pixels_per_meter) + self._world_offset[1]

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p

    def calculate_angle(self, x1, y1, x2, y2):
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def calculate_distance(self, x1, y1, x2, y2):
        new_x = x1 - x2
        new_y = y1 - y2
        return np.sqrt((new_x ** 2) + (new_y) ** 2), new_x, new_y

    def calculate_orientation(self, top_left_x, top_left_y, bottom_left_x, bottom_left_y):
        # Calculate the slope
        slope = (bottom_left_y - top_left_y) / (bottom_left_x - top_left_x)

        # Calculate the angle in radians
        angle_radians = math.atan(slope)

        # Convert the angle from radians to degrees
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees

    def calculate_bounding_box_corners(self, center_x, center_y, width, height, angle_degrees):
        # Convert angle from degrees to radians
        angle_radians = math.radians(angle_degrees)
        # Calculate half-width and half-height
        half_width = width / 2
        half_height = height / 2

        # Calculate rotated corners
        corner1_x = center_x + half_width * math.cos(angle_radians) - half_height * math.sin(angle_radians)
        corner1_y = center_y + half_width * math.sin(angle_radians) + half_height * math.cos(angle_radians)

        corner2_x = center_x - half_width * math.cos(angle_radians) - half_height * math.sin(angle_radians)
        corner2_y = center_y - half_width * math.sin(angle_radians) + half_height * math.cos(angle_radians)

        corner3_x = center_x - half_width * math.cos(angle_radians) + half_height * math.sin(angle_radians)
        corner3_y = center_y - half_width * math.sin(angle_radians) - half_height * math.cos(angle_radians)

        corner4_x = center_x + half_width * math.cos(angle_radians) + half_height * math.sin(angle_radians)
        corner4_y = center_y + half_width * math.sin(angle_radians) - half_height * math.cos(angle_radians)

        # Return corner coordinates
        return [(corner1_x, corner1_y), (corner2_x, corner2_y), (center_x,center_y), (corner3_x, corner3_y), (corner4_x, corner4_y)]

    def calculate_bounding_box_corners_0(self, center_x, center_y, width, height, angle_degrees):
        # Calculate half-width and half-height for convenience
        half_width = width / 2
        half_height = height / 2

        # Calculate local coordinates of bounding box corners (before rotation)
        local_corners = [(-half_width, -half_height),
                         (half_width, -half_height),
                         (half_width, half_height),
                         (-half_width, half_height)]

        # Convert orientation from degrees to radians
        orientation_radians = math.radians(angle_degrees)

        # Apply rotation transformation to local coordinates to get global coordinates
        rotated_corners = []
        for corner in local_corners:
            rotated_x = corner[0] * math.cos(orientation_radians) - corner[1] * math.sin(orientation_radians)
            rotated_y = corner[0] * math.sin(orientation_radians) + corner[1] * math.cos(orientation_radians)
            global_x = rotated_x + center_x
            global_y = rotated_y + center_y
            rotated_corners.append((global_x, global_y))

        return rotated_corners

    def calculate_bounding_box_corners_2(self, center_x, center_y, width, height, angle_degrees):
        x = center_x
        y = center_y
        bottom_left = np.array([x - height/2, y - width/2])
        top_left = np.array([x + height/2, y - width/2])
        center = np.array([x, y])
        top_right = np.array([x - height/2, y + width/2])
        bottom_right = np.array([x + height/2, y + width/2])

        return np.array([[bottom_left], [top_left], [center], [top_right], [bottom_right]])