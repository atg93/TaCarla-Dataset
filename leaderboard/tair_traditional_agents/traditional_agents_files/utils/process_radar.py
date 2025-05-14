import copy
import math
import numpy as np
import cv2
import colorsys

import matplotlib.pyplot as plt

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.birdview.planning.carla_agent_files.agent_utils.coordinate_utils import normalize_angle

import io
from PIL import Image

import carla

class Process_Radar:
    def __init__(self):
        self.is_there_obstacle = False

    def polar_to_cartesian(self, r, theta):
        """ Convert polar coordinates to cartesian coordinates. """
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        return int(x), int(y)

    def draw_graph_2(self,polar_points):
        # Sample polar coordinates (r, theta)
        #polar_points = [(100, np.deg2rad(30)), (150, np.deg2rad(60)), (200, np.deg2rad(90)), (250, np.deg2rad(120))]

        # Convert polar to cartesian
        #cartesian_points = [self.polar_to_cartesian(r, theta) for r, theta in polar_points]
        cartesian_points = [(r * math.sin(theta), r * math.cos(theta)) for theta, r in
                            polar_points]
        cartesian_points = (np.array(cartesian_points) * 4).astype(np.int)

        # Create a blank image
        image = np.zeros((200, 200), dtype=np.uint8)

        if np.sum(np.array(cartesian_points)) != 0:
            # Draw the points
            for point in cartesian_points:
                self.radar_pixel_coordinates.append([point[0], point[1]])
                cv2.circle(image, (100 - point[0], 100 - point[1]), 2, (255), -1)

        return image.astype(np.bool)

    def draw_graph(self, a_list, name='graph'):
        plt.figure(figsize=(10, 6))  # Set the size of the plot
        plt.plot(a_list, color='blue')  # Plot x and y, label it, and set the color
        plt.title('Simple Plot')  # Title of the plot
        plt.xlabel('x values')  # Label for the x-axis
        plt.ylabel('sample')  # Label for the y-axis
        # plt.legend()  # Show legend
        plt.grid(True)  # Show grid
        plt.savefig(name + '.png')

    def draw_polar_coordinate(self, points, name='graph'):
        # Convert points to Cartesian coordinates
        cartesian_points = [(r * math.sin(theta), r * math.cos(theta)) for theta, r in
                            points]

        # Unzip points into x and y coordinates
        x_coords, y_coords = zip(*cartesian_points)

        # Plotting
        plt.clf()
        plt.scatter(x_coords, y_coords)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('2D Plot of Points')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True)
        plt.xlim([-3, 3])
        plt.ylim([0, 10])

        plt.savefig(name + '.png')
        img = Image.open(name + '.png')

        return img

    def show_radar_output(self, radar_data, ego_location=None, compass=None, ego_yaw=None):

        points = []
        azu_list = []
        dept_list = []
        altitude_list = []
        velocity_list = []
        carla_loc_list = []
        close_points_count = 0
        self.radar_pixel_coordinates = []
        for detection in radar_data[1]:
            azimuth_angle = detection[1]
            depth = detection[0]
            if depth > 10:
                continue
            if depth < 10:
                close_points_count += 1
                altitude_list.append(detection[2])
                velocity_list.append(detection[3])
                #carla_loc_list.append(self.radar_callback(detection))

            points.append([azimuth_angle, depth])
            azu_list.append(azimuth_angle)
            dept_list.append(detection[2])
        azu_array = np.array(azu_list)
        check_azu = azu_array > 6
        check_azu_number = np.sum(check_azu)
        print("check_azu_number:", check_azu_number)

        #img = self.draw_polar_coordinate(points)

        img = self.draw_graph_2(points)

        carla_loc_list = self.radar_point_loc(ego_location, points, ego_yaw)

        closest_points = self.measure_distance(self.radar_pixel_coordinates, carla_loc_list, velocity_list, dept_list, azu_array)

        mean_alt = np.mean(altitude_list)
        mean_vel = np.mean(velocity_list)

        if mean_vel <= 0.0 and close_points_count >= 5:
            self.is_there_obstacle = True
        else:
            self.is_there_obstacle = False


        return closest_points, carla_loc_list, img, close_points_count, mean_alt, mean_vel, self.is_there_obstacle #cv2.resize(np.array(img)[:, :, 0:3], (200, 200))

    def measure_distance(self, main_list, carla_loc_list, velocity_list, dept_list, azu_array):
        # Initialize an empty list to store the distances of each sublist from the origin
        dis_dict = {}
        distance_list = []
        # Loop through each element (sublist) in the main list
        for index, sublist in enumerate(main_list):
            # Calculate the Euclidean distance from the origin for each sublist
            distance = math.sqrt(sublist[0] ** 2 + sublist[1] ** 2)
            distance_list.append(distance)
            # Append the distance to the distances list
            dis_dict.update({distance:[sublist, carla_loc_list[index], velocity_list[index], dept_list[index], azu_array[index]]})


        distances = sorted(distance_list)
        distances = np.array(distances)[:20]

        closest_points_info = []
        for index in range(len(distances)):
            closest_points_info.append(dis_dict[distances[index]])

        return closest_points_info

    def radar_point_loc(self, ego_location, points, ego_yaw):
        carla_loc_list = []
        if ego_location != None:
            for _p in points:
                yaw_like = (3 * math.pi / 2) - _p[0]
                new_azu = (yaw_like + math.pi) % (2 * math.pi) - math.pi
                relative_yaw = normalize_angle(new_azu - ego_yaw)#*(-1)

                new_x, new_y = self.calculate_new_location(ego_location.x, ego_location.y, relative_yaw, _p[1])
                carla_loc_list.append(carla.Location(x=new_x, y=new_y, z=ego_location.z))

        return carla_loc_list


    def calculate_new_location(self, start_x, start_y, yaw_radians, distance):
        # Convert the yaw angle from degrees to radians
        #yaw_radians = math.radians(yaw_degrees)

        # Calculate the change in the X and Y coordinates
        delta_y = distance * math.cos(yaw_radians)
        delta_x = distance * math.sin(yaw_radians)

        # Calculate the new location
        new_x = start_x + delta_x
        new_y = start_y + delta_y

        return new_x, new_y

    def radar_callback(self, detection):
        # This function will be called each time the radar sensor has new data.

        # Each 'detection' is a Doppler radar detection.

        # Convert from polar coordinates (angle, distance) to Cartesian coordinates.
        azimuth = detection[1] #detection.azimuth
        altitude = detection[2] #detection.altitude
        depth = detection[0] #detection.depth  # distance from the radar

        # Assuming the radar is mounted at the front of the vehicle, pointing forwards.
        # Convert degrees to radians for calculation
        azimuth_radians = math.radians(azimuth)
        altitude_radians = math.radians(altitude)

        # Calculate the position of the detected point in the radar's coordinate frame.
        # Note: You might need to adjust this calculation based on the exact setup and requirements.
        x = depth * math.cos(altitude_radians) * math.sin(azimuth_radians)
        y = depth * math.cos(altitude_radians) * math.cos(azimuth_radians)
        z = depth * math.sin(altitude_radians)

        # Transform this point to the world coordinate system if necessary.
        # Here we are just directly using these as relative coordinates for simplicity.

        # Visualization or further processing goes here.
        location = carla.Location(x=x, y=y, z=z + 1.0)

        return location