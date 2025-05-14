import numpy as np
import torch
import sys

sys.path.append('/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception')

from nuscenes.utils.data_classes import PointCloud

import carla

import math

import cv2

class CarlaLidarPointCloud(PointCloud):

    @staticmethod
    def nbr_dims() -> int:
        return 4

    @classmethod
    def from_file(cls, points) -> 'CarlaLidarPointCloud':
        return cls(np.vstack(points).T)


class Prepare_Lidar_Data:

    def __init__(self, hd_map=None):
        self.extrinsics = {'lidar':torch.Tensor([[0.0, -1.0, 0.0, 0.9443896 - 1],
                            [-1.0, 0.0, 0.0, -0.01839097],
                            [0.0, 0.0, 1.0, 1.83979095],
                            [0., 0., 0., 1.]])}

        self.view = torch.Tensor([[[[ 0.0000, -2.0000,  0.0000, 99.5000],
          [-2.0000,  0.0000,  0.0000, 99.5000],
          [ 0.0000,  0.0000,  0.5000,  2.0000],
          [ 0.0000,  0.0000,  0.0000,  1.0000]]]])

        self.opendrive_hd_map = hd_map

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

    def get_pointcloud_data(self, pcloud_cls, lidar_data, lidar_to_car, min_distance=2.2, view=None):
        """
        Adapted from tairvision.datasets.nuscenes
        """

        points = np.zeros((pcloud_cls.nbr_dims(), 0), dtype=np.float32)
        all_pclouds = pcloud_cls(points)
        all_times = np.zeros((1, 0))

        car_to_view = view[0, 0].numpy()
        max_nb_points = 70000

        curr_pcloud = pcloud_cls.from_file(lidar_data)
        curr_pcloud.remove_close(min_distance)
        curr_pcloud.transform(car_to_view @ np.array(lidar_to_car))

        # Add time vector which can be used as a temporal feature.
        time_lag = 0.0
        times = time_lag * np.ones((1, curr_pcloud.nbr_points()))
        all_times = np.concatenate([all_times, times], axis=1)
        all_pclouds.points = np.concatenate([all_pclouds.points,
                                             curr_pcloud.points], axis=1)

        pcloud_data = np.concatenate([all_pclouds.points, all_times], axis=0)
        pcloud_data = np.pad(pcloud_data, [(0, 0), (0, max_nb_points - pcloud_data.shape[1])], mode='constant')
        pcloud_data = torch.tensor(pcloud_data.T, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        return pcloud_data

    def __call__(self, input_data):

        if self.opendrive_hd_map == None:
            self.opendrive_hd_map = carla.Map("RouteMap", input_data['hd_map'][1]['opendrive'])

        lidar_to_car = self.extrinsics['lidar']
        input_lidar_data = np.concatenate((input_data['lidar_0'][1], input_data['lidar_1'][1] * (-1)), 0)
        input_lidar_data = self.get_closest_lidar_points(input_data, input_lidar_data)
        lidar_data = self.get_pointcloud_data(CarlaLidarPointCloud, input_lidar_data, lidar_to_car, min_distance=2.2, view=self.view).cuda(2)
        lidar_image = self.check_lidar(input_lidar_data[1])
        return lidar_data.unsqueeze(2), lidar_image, input_lidar_data

    def get_closest_lidar_points(self, input_data, lidar):
        close_point_list = []
        for point in lidar:
            point_location = carla.Location(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            tl_gps_location = self.opendrive_hd_map.transform_to_geolocation(point_location)
            distance = self.haversine(input_data['gps'][1][0], input_data['gps'][1][1], tl_gps_location.latitude+input_data['gps'][1][0],
                                      tl_gps_location.longitude+input_data['gps'][1][1])  # ego_location.distance(tl.transform.location)
            if distance < 50:
                close_point_list.append(point)

        return np.array(close_point_list)

    def check_lidar(self,lidar,lidar_range=100):
        #lidar = lidar.squeeze(0).squeeze(0).cpu().numpy()
        image_size = 400

        # Convert to 2D by ignoring z (for BEV image)
        points_2d = (lidar[:2] * (image_size/lidar_range)).astype(int) * 2#lidar[:, :2]

        # Create an empty image - 100x100 pixels
        image = np.zeros((image_size, image_size, 3))
        image_mask = np.zeros((image_size, image_size))

        # Scaling factor to fit points in the image dimensions
        new_points_2d = points_2d + 200

        # Populate the image with LiDAR points
        for point in new_points_2d:
            try:
                image_mask[point[1], point[0]] = 1  # Increment to simulate intensity (simplistic approach)
            except:
                pass  # Increment to simulate intensity (simplistic approach)

        # Normalize image to have values between 0 and 255
        #image -= image.min()
        #image = (image / image.max()) * 255
        image[image_mask>0] = (255,255,255)
        cv2.imwrite('lidar.png', image)
        return image.astype(np.uint8)