import os

import carla
import h5py

import json

import numpy as np
import pickle

import cv2

import math

import numpy as np
from PIL import Image
import os
import laspy
from laspy import ExtraBytesParams

from pathlib import PosixPath

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return json.JSONEncoder.default(self, obj)

class Data_Collection_For_Detection():
    def __init__(self, _vehicle, _world, current_path):
        self._vehicle = _vehicle
        self._world = _world
        self.opendrive_hd_map = self._world.get_map()
        self.counter = 0
        self.current_path = current_path
        """self.depth_image = None
        self.read_dept_camera()
        self.depth_camera.listen(lambda image: self.process_depth_image(image))"""
        asd = 0

    def process_depth_image(self, image):
        self.depth_image = image

    def read_dept_camera(self):
        # Get the blueprint for the depth camera
        blueprint_library = self._world.get_blueprint_library()
        depth_camera_bp = blueprint_library.find('sensor.camera.depth')

        # Set attributes
        depth_camera_bp.set_attribute('image_size_x', '900')
        depth_camera_bp.set_attribute('image_size_y', '1600')
        depth_camera_bp.set_attribute('fov', '90')

        # Attach the camera to the same vehicle
        depth_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))  # Set position

        # Spawn the depth camera
        self.depth_camera = self._world.spawn_actor(depth_camera_bp, depth_camera_transform, attach_to=self._vehicle)

        # Define a callback function to process depth images

        #.convert(carla.ColorConverter.LogarithmicDepth)  # Convert to depth visualization
        #image.save_to_disk('_out/depth/%06d.png' % image.frame)



    def get_closest_lidar_points(self, input_data, lidar):
        close_point_list = []
        for point in lidar:
            point_location = carla.Location(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            distance = input_data["gt_location"].distance(point_location)
            #tl_gps_location = self.opendrive_hd_map.transform_to_geolocation(point_location)
            #distance = self.haversine(input_data['gps'][1][0], input_data['gps'][1][1], tl_gps_location.latitude+input_data['gps'][1][0],
            #                          tl_gps_location.longitude+input_data['gps'][1][1])  # ego_location.distance(tl.transform.location)

            #if distance < 50 or os.getenv("TOWN_NAME")=="Town13":
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

    def convert_rgb_to_dept(self, image):
        #self.depth_image
        """R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]
        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        in_meters = 1000 * normalized"""
        #img = image

        # Convert to depth in meters
        #depth = (img[:, :, 0].astype(np.float64) + img[:, :, 1].astype(np.float64) * 256 + img[:, :, 2].astype(np.float64) * 256 ** 2) / (256 ** 3 - 1) * 1000.0

        # Normalize for visualization
        #depth_visual = depth # (depth / depth.max() * 255).astype(np.uint8)
        return image #cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def save_camera_input(self, path, image):
        image = image[:,:,:-1]
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(image)

        #   PNG (lossless compression)
        #img.save(self.front_path + '/front_' + str(self.counter) + ".png")
        #img.save(self.front_path + '/front_' + str(self.counter) + ".jpeg", quality=95)
        img.save(path, quality=30)


    def save_inputs_only(self, input_data):
        lidar = np.concatenate((input_data['lidar_0'][1], input_data['lidar_1'][1] * (-1)), 0)
        lidar_0 = self.get_closest_lidar_points(input_data, lidar)
        self.check_lidar(lidar_0, self.counter)

        self.save_camera_input(self.bev_image_path + '/' + str(self.counter) + '_.jpg', input_data['bev'][1])

        #rgb_camera
        self.save_camera_input(self.front_path + '/front_' + str(self.counter) + '_.jpg', input_data['front'][1])
        self.save_camera_input(self.front_right_path + '/front_right_' + str(self.counter) + '_.jpg', input_data['front_right'][1])
        self.save_camera_input(self.front_left_path + '/front_left_' + str(self.counter) + '_.jpg', input_data['front_left'][1])
        self.save_camera_input(self.back_path + '/back_' + str(self.counter) + '_.jpg', input_data['back'][1])
        self.save_camera_input(self.back_left_path + '/back_left_' + str(self.counter) + '_.jpg', input_data['back_left'][1])
        self.save_camera_input(self.back_right_path + '/back_right_' + str(self.counter) + '_.jpg', input_data['back_right'][1])

        #semantic_camera
        self.save_camera_input(self.semantic_front_path + '/semantic_front_' + str(self.counter) + '_.jpg', input_data['semantic_front'][1])
        self.save_camera_input(self.semantic_front_right_path + '/semantic_front_right_' + str(self.counter) + '_.jpg', input_data['semantic_front_right'][1])
        self.save_camera_input(self.semantic_front_left_path + '/semantic_front_left_' + str(self.counter) + '_.jpg', input_data['semantic_front_left'][1])
        self.save_camera_input(self.semantic_back_path + '/semantic_back_' + str(self.counter) + '_.jpg', input_data['semantic_back'][1])
        self.save_camera_input(self.semantic_back_left_path + '/semantic_back_left_' + str(self.counter) + '_.jpg', input_data['semantic_back_left'][1])
        self.save_camera_input(self.semantic_back_right_path + '/semantic_back_right_' + str(self.counter) + '_.jpg', input_data['semantic_back_right'][1])

        #semantic_camera
        self.save_camera_input(self.instance_front_path + '/instance_front_' + str(self.counter) + '_.jpg', input_data['instance_front'][1])
        self.save_camera_input(self.instance_front_right_path + '/instance_front_right_' + str(self.counter) + '_.jpg', input_data['instance_front_right'][1])
        self.save_camera_input(self.instance_front_left_path + '/instance_front_left_' + str(self.counter) + '_.jpg', input_data['instance_front_left'][1])
        self.save_camera_input(self.instance_back_path + '/instance_back_' + str(self.counter) + '_.jpg', input_data['instance_back'][1])
        self.save_camera_input(self.instance_back_left_path + '/instance_back_left_' + str(self.counter) + '_.jpg', input_data['instance_back_left'][1])
        self.save_camera_input(self.instance_back_right_path + '/instance_back_right_' + str(self.counter) + '_.jpg', input_data['instance_back_right'][1])

        #depth_camera
        self.save_camera_input(self.depth_front_path + '/depth_front_' + str(self.counter) + '_.jpg', self.convert_rgb_to_dept(input_data['depth_front'][1]))
        self.save_camera_input(self.depth_front_right_path + '/depth_front_right_' + str(self.counter) + '_.jpg', self.convert_rgb_to_dept(input_data['depth_front_right'][1]))
        self.save_camera_input(self.depth_front_left_path + '/depth_front_left_' + str(self.counter) + '_.jpg', self.convert_rgb_to_dept(input_data['depth_front_left'][1]))
        self.save_camera_input(self.depth_back_path + '/depth_back_' + str(self.counter) + '_.jpg', self.convert_rgb_to_dept(input_data['depth_back'][1]))
        self.save_camera_input(self.depth_back_left_path + '/depth_back_left_' + str(self.counter) + '_.jpg', self.convert_rgb_to_dept(input_data['depth_back_left'][1]))
        self.save_camera_input(self.depth_back_right_path + '/depth_back_right_' + str(self.counter) + '_.jpg', self.convert_rgb_to_dept(input_data['depth_back_right'][1]))

        # Create a header; adjust point format and version as needed
        header = laspy.LasHeader(point_format=4, version="1.4")

        # Create a new LAS data object with the header
        las = laspy.LasData(header)

        # Assign the coordinates to the las object
        las.x = lidar_0[1][:, 0]
        las.y = lidar_0[1][:, 1]
        las.z = lidar_0[1][:, 2]
        extra_bytes = ExtraBytesParams(name="float_intensity", type=np.float32)
        las.add_extra_dim(extra_bytes)

        # Assign values
        las["float_intensity"] = lidar_0[1][:, 3]

        # Write the data to a file with a .laz extension (compressed)
        las.write(self.lidar_path + '/lidar_' + str(self.counter) + '.laz')
        #las_0 = laspy.read((self.lidar_path + '/lidar_' + str(self.counter) + '.laz'))


        """np.save(self.radar_front_path + '/radar_' + str(self.counter) + '.npy', input_data['radar_front'][1])
        np.save(self.radar_front_left_path + '/radar_' + str(self.counter) + '.npy', input_data['radar_front_left'][1])
        np.save(self.radar_front_right_path + '/radar_' + str(self.counter) + '.npy', input_data['radar_front_right'][1])
        np.save(self.radar_back_left_path + '/radar_' + str(self.counter) + '.npy', input_data['radar_back_left'][1])
        np.save(self.radar_back_right_path + '/radar_' + str(self.counter) + '.npy', input_data['radar_back_right'][1])"""

        # Open an HDF5 file for writing
        with h5py.File(self.radar_path + '/radar_' + str(self.counter) + '.h5', 'w') as f:
            # Create a dataset with gzip compression. 
            # compression_opts=9 uses maximum compression level (range 1 to 9).
            # Optionally, you can specify a chunk size, which is useful for large datasets.
            f.create_dataset('radar_front', data=input_data['radar_front'][1], compression='gzip', compression_opts=9)
            f.create_dataset('radar_front_left', data=input_data['radar_front_left'][1], compression='gzip', compression_opts=9)
            f.create_dataset('radar_front_right', data=input_data['radar_front_right'][1], compression='gzip', compression_opts=9)
            f.create_dataset('radar_back_left', data=input_data['radar_back_left'][1], compression='gzip', compression_opts=9)
            f.create_dataset('radar_back_right', data=input_data['radar_back_right'][1], compression='gzip', compression_opts=9)

        #with h5py.File(self.radar_path + '/radar_' + str(self.counter) + '.h5', 'r') as f:
        #    # List all dataset names in the file
        #    print("Datasets available:", list(f.keys()))

        #    # Access each dataset
        #    radar_front = f['radar_front'][:]  # Read the entire dataset into memory
        #    radar_front_left = f['radar_front_left'][:]
        #    radar_front_right = f['radar_front_right'][:]
        #    radar_back_left = f['radar_back_left'][:]
        #    radar_back_right = f['radar_back_right'][:]
        self.counter += 1

        # In synchronous mode, tick the world to advance the simulation and get the snapshot:
        #snapshot = self._world.tick()

        # Access the simulation time:
        #self.simulation_time = snapshot.timestamp.elapsed_seconds
        #self.perv_simulation_time = snapshot.timestamp.elapsed_seconds



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

        file = str(self.current_path)
        os.mkdir(file + '/rgb_camera/')
        os.mkdir(file+'/rgb_camera/'+'front')
        self.front_path = file+'/rgb_camera/'+'front'

        os.mkdir(file+'/rgb_camera/'+'front_right')
        self.front_right_path = file+'/rgb_camera/'+'front_right'

        os.mkdir(file+'/rgb_camera/'+'front_left')
        self.front_left_path = file+'/rgb_camera/'+'front_left'

        os.mkdir(file+'/rgb_camera/'+'back')
        self.back_path = file + '/rgb_camera/' + 'back'

        os.mkdir(file+'/rgb_camera/'+'back_left')
        self.back_left_path = file + '/rgb_camera/' + 'back_left'

        os.mkdir(file+'/rgb_camera/'+'back_right')
        self.back_right_path = file + '/rgb_camera/' + 'back_right'

        #semantic_camera
        os.mkdir(file + '/semantic_camera/')
        os.mkdir(file+'/semantic_camera/'+'semantic_front')
        self.semantic_front_path = file+'/semantic_camera/'+'semantic_front'

        os.mkdir(file+'/semantic_camera/'+'semantic_front_right')
        self.semantic_front_right_path = file+'/semantic_camera/'+'semantic_front_right'

        os.mkdir(file+'/semantic_camera/'+'semantic_front_left')
        self.semantic_front_left_path = file+'/semantic_camera/'+'semantic_front_left'

        os.mkdir(file+'/semantic_camera/'+'semantic_back')
        self.semantic_back_path = file + '/semantic_camera/' + 'semantic_back'

        os.mkdir(file+'/semantic_camera/'+'semantic_back_left')
        self.semantic_back_left_path = file + '/semantic_camera/' + 'semantic_back_left'

        os.mkdir(file+'/semantic_camera/'+'semantic_back_right')
        self.semantic_back_right_path = file + '/semantic_camera/' + 'semantic_back_right'

        #instance_camera
        os.mkdir(file + '/instance_camera/')
        os.mkdir(file+'/instance_camera/'+'instance_front')
        self.instance_front_path = file+'/instance_camera/'+'instance_front'

        os.mkdir(file+'/instance_camera/'+'instance_front_right')
        self.instance_front_right_path = file+'/instance_camera/'+'instance_front_right'

        os.mkdir(file+'/instance_camera/'+'instance_front_left')
        self.instance_front_left_path = file+'/instance_camera/'+'instance_front_left'

        os.mkdir(file+'/instance_camera/'+'instance_back')
        self.instance_back_path = file + '/instance_camera/' + 'instance_back'

        os.mkdir(file+'/instance_camera/'+'instance_back_left')
        self.instance_back_left_path = file + '/instance_camera/' + 'instance_back_left'

        os.mkdir(file+'/instance_camera/'+'instance_back_right')
        self.instance_back_right_path = file + '/instance_camera/' + 'instance_back_right'

        #dept_camera
        os.mkdir(file + '/depth_camera/')
        os.mkdir(file+'/depth_camera/'+'depth_front')
        self.depth_front_path = file+'/depth_camera/'+'depth_front'

        os.mkdir(file+'/depth_camera/'+'depth_front_right')
        self.depth_front_right_path = file+'/depth_camera/'+'depth_front_right'

        os.mkdir(file+'/depth_camera/'+'depth_front_left')
        self.depth_front_left_path = file+'/depth_camera/'+'depth_front_left'

        os.mkdir(file+'/depth_camera/'+'depth_back')
        self.depth_back_path = file + '/depth_camera/' + 'depth_back'

        os.mkdir(file+'/depth_camera/'+'depth_back_left')
        self.depth_back_left_path = file + '/depth_camera/' + 'depth_back_left'

        os.mkdir(file+'/depth_camera/'+'depth_back_right')
        self.depth_back_right_path = file + '/depth_camera/' + 'depth_back_right'

        os.mkdir(file+'/'+'lidar')
        self.lidar_path = file + '/' + 'lidar'

        os.mkdir(file + '/' + 'lidar_image')
        self.lidar_image_path = file + '/' + 'lidar_image'

        os.mkdir(file + '/' + 'bev_image')
        self.bev_image_path = file + '/' + 'bev_image'

        os.mkdir(file+'/'+'radar')
        self.radar_path = file + '/' + 'radar'


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





    def check_lidar(self,lidar, index, lidar_range=100):
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
        cv2.imwrite(str(self.lidar_image_path)+'/'+str(index)+'.png', image)