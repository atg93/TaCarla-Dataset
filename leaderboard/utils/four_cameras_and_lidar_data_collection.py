import os

import carla
import h5py

import json

import numpy as np
import pickle

import cv2

import math

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return json.JSONEncoder.default(self, obj)

class Four_Cameras_and_Lidar_Data_Collection():
    def __init__(self, _vehicle, _world, current_path):
        self._vehicle = _vehicle
        self._world = _world
        self.opendrive_hd_map = self._world.get_map()
        self.counter = 0
        self.current_path = current_path

    def get_closest_lidar_points(self, input_data, lidar):
        close_point_list = []
        for point in lidar:
            point_location = carla.Location(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            tl_gps_location = self.opendrive_hd_map.transform_to_geolocation(point_location)
            distance = self.haversine(input_data['gps'][1][0], input_data['gps'][1][1], tl_gps_location.latitude+input_data['gps'][1][0],
                                      tl_gps_location.longitude+input_data['gps'][1][1])  # ego_location.distance(tl.transform.location)

            if distance < 50:
                close_point_list.append(point)

        return np.array((lidar[0], np.array(close_point_list)))


    def __call__(self,input_data, detection_boxes, ego_boxes):
        #self.get_relative_speed()
        if self.counter % 5 == 0:#np.flip(x, 1)
            lidar = np.concatenate((input_data['lidar_0'][1], input_data['lidar_1'][1]*(-1)), 0)
            front, front_right, front_left, back, back_left, back_right = input_data['front'], input_data['front_right'],\
                                                                          input_data['front_left'], input_data['back'],\
                                                                          input_data['back_left'], input_data['back_right']
            lidar = self.get_closest_lidar_points(input_data, lidar)
            self.check_lidar(lidar)
            self.save_arrays(front, front_right, front_left, back, back_left, back_right, lidar, detection_boxes, ego_boxes)

        self.counter += 1

        asd = 0


    def save_arrays(self, front, front_right, front_left, back, back_left, back_right, lidar, detection_boxes, ego_boxes):
        cv2.imwrite(self.front_path + '/front_' + str(self.counter) + '_.png', front[1])
        cv2.imwrite(self.front_right_path + '/front_right_' + str(self.counter) + '_.png', front_right[1])
        cv2.imwrite(self.front_left_path + '/front_left_' + str(self.counter) + '_.png', front_left[1])
        cv2.imwrite(self.back_path + '/back_' + str(self.counter) + '_.png', back[1])
        cv2.imwrite(self.back_left_path + '/back_left_' + str(self.counter) + '_.png', back_left[1])
        cv2.imwrite(self.back_right_path + '/back_right_' + str(self.counter) + '_.png', back_right[1])

        np.save(self.lidar_path + '/lidar_' + str(self.counter) + '.npy', lidar[1])

        if len(detection_boxes) == 0:
            detection_boxes = []

        current_weather = self._world.get_weather()
        weather_list = dir(current_weather)[50:-1]
        weather_dict = {}
        for key in weather_list:
            weather_dict.update({key:getattr(current_weather, key)})
        sample = [self.counter,ego_boxes,detection_boxes,weather_dict]
        with open(self.pickle_file_path, 'ab') as file:
            pickle.dump(sample, file)
        asd = 0





    def create_file(self):

        self.vehicle_dic_dataset = {}
        self.vehicle_index_dataset = 1

        self.counter = 0
        episode_number = 0
        file = 'episode_' + str(episode_number) + '.h5'
        file_exist = True
        file = '/workspace/tg22/f_and_l_dataset/' + file
        while file_exist:
            episode_number += 1
            file = '/workspace/tg22/f_and_l_dataset/' + 'episode_' + str(episode_number) #+ '.h5'
            file_exist = os.path.exists(file)

        os.mkdir(file)
        os.mkdir(file+'/'+'front')
        self.front_path = file+'/'+'front'

        os.mkdir(file+'/'+'front_right')
        self.front_right_path = file+'/'+'front_right'

        os.mkdir(file+'/'+'front_left')
        self.front_left_path = file+'/'+'front_left'

        os.mkdir(file+'/'+'back')
        self.back_path = file + '/' + 'back'

        os.mkdir(file+'/'+'back_left')
        self.back_left_path = file + '/' + 'back_left'

        os.mkdir(file+'/'+'back_right')
        self.back_right_path = file + '/' + 'back_right'

        os.mkdir(file+'/'+'lidar')
        self.lidar_path = file + '/' + 'lidar'

        self.folder = file

        print("file:",file)
        #self.h5_file = h5py.File(self.folder + '/episode_' + str(episode_number) + '_labels_' +'.h5', 'a')

        self.pickle_file_path = self.folder + '/episode_' + str(episode_number) + '.pickle'


        asd = 0

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


    def get_relative_speed_plant(self):
        ego_location = self._vehicle.get_location()
        ego_transform = self._vehicle.get_transform()
        ego_control   = self._vehicle.get_control()
        ego_velocity  = self._vehicle.get_velocity()
        ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity) # In m/s

        asd = 0





    def check_lidar(self,lidar,lidar_range=85):
        image_size = 400

        # Convert to 2D by ignoring z (for BEV image)
        points_2d = (lidar[1][:,:2] * (image_size/lidar_range)).astype(int) * 2

        # Create an empty image - 100x100 pixels
        image = np.zeros((image_size, image_size))

        # Scaling factor to fit points in the image dimensions
        new_points_2d = points_2d + 200

        # Populate the image with LiDAR points
        for point in new_points_2d:
            try:
                image[point[1], point[0]] = 1  # Increment to simulate intensity (simplistic approach)
            except:
                pass  # Increment to simulate intensity (simplistic approach)

        # Normalize image to have values between 0 and 255
        #image -= image.min()
        #image = (image / image.max()) * 255
        image[image>0] = 255
        cv2.imwrite(self.current_path+'/lidar.png', image)