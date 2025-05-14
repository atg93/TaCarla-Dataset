import numpy as np
import datetime
import h5py
import cv2
import math

import carla

import os
import pickle

class Tl_Data_Collection:

    def __init__(self, _vehicle, _world, camera, current_path):
        self._vehicle = _vehicle
        self._world = _world
        self.camera = camera
        self.current_path = current_path
        self.K = self.build_projection_matrix(float(self.camera.attributes['image_size_x']), float(self.camera.attributes['image_size_y']), float(self.camera.attributes['fov']))
        self.negative_sample_count = 0
        self.negative_sample_read_count = 0
        self.positive_sample_count = 0
        self.tl_dataset_count = 0
        self.img_id = 0
        self._carla_map = 12




    def convert_meter(self,points, ego_pix_point, pixel_per_meter):
        ego_pix_point - points.astype(np.int32)

    def set_tl_info(self,state, coordinate, ego_pix_point=400, pixel_per_meter=4):
        self.tl_state = state
        first_point = (ego_pix_point - coordinate[0].astype(np.int32).squeeze(axis=0)) / pixel_per_meter
        second_point = (ego_pix_point - coordinate[1].astype(np.int32).squeeze(axis=0)) / pixel_per_meter
        mid_point = (first_point + second_point) / 2
        self.tl_pixel_coordinate = np.stack([first_point, second_point, mid_point],0)
        print("self.tl_pixel_coordinate: ",self.tl_pixel_coordinate )

    def check_masks(self, mask_list, c_route, state):
        for mask, pixel_coordinates in mask_list:
            if np.sum(mask * c_route) > 1:
                self.set_tl_info(state, pixel_coordinates)

    def get_3d_info(self, tl_masks_list, c_route):
        self.tl_state = 'None'
        self.tl_pixel_coordinate = None
        #green_masks_list, yellow_masks_list, red_masks_list = tl_masks_list
        state_list =['green','yellow','red']
        for index, masks_list in enumerate(tl_masks_list):
            self.check_masks(masks_list, c_route, state=state_list[index])

        return self.tl_state, self.tl_pixel_coordinate

    def __call__(self, input_data, tl_masks_list, c_route):
        self.tl_pixel_coordinate = None
        tl_state, tl_bev_pixel_coordinate = self.get_3d_info(tl_masks_list, c_route)
        img = self.check_traffic_lights(input_data, tl_state, tl_bev_pixel_coordinate)

        asd = 0
        self.img_id += 1

        return img



    def check_traffic_lights(self, input_data, tl_state, tl_bev_pixel_coordinate):
        #color, stop_bounding_boxes, tl_corner_bb = self.chauffeurnet.get_is_there_list_parameter()
        compass = input_data['imu'][1][-1]#preprocess_compass(input_data['imu'][1][-1])
        color = tl_state
        print("*"*50,"color:", color)
        current_speed = self._vehicle.get_velocity().length()
        vehicle, _ = self._vehicle, self._world


        ego_loc = vehicle.get_location()
        ego_loc = np.array([ego_loc.x, ego_loc.y, ego_loc.z])

        # Remember the edge pairs
        edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

        # Retrieve the first image
        image = input_data['front']
        img = np.reshape(np.copy(image[1]), (image[1].shape[0], image[1].shape[1], 4))

        bounding_box_set = self._world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        # Get the camera matrix
        world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())
        bounding_box_set = bounding_box_set
        bb_list = []
        draw = False
        positive_sample = False
        negative_sample = False
        distance = -1

        self.positive_sample_list = []
        self.negative_sample_list = []

        distance_list = []
        bb_list = []
        for index, bb in enumerate(bounding_box_set):
            points = np.ones(4)*(-1)
            distance = bb.location.distance(vehicle.get_transform().location)
            if 5.5 < distance and distance < 50 and abs(abs((bb.rotation.yaw % 360) -
                                                            (((vehicle.get_transform().rotation.yaw + 90) % 360)))) < 50 \
                    and current_speed > 0.1 and type(
                    tl_bev_pixel_coordinate) != type(
                    None) and color != 'None':  # np.sum(self.previous_loc) != np.sum(ego_loc): #and color != 'unknown' and color != 'stop_sign'
                distance_list.append(distance)
                bb_list.append(bb)

        if len(bb_list) != 0:#and abs(abs((bb.rotation.yaw % 360) - ((vehicle.get_transform().rotation.yaw % 360) + 90))) < 50
            index = np.argmin(distance_list)
            distance = distance_list[index]
            bb = bb_list[index]
            # Filter for distance from ego vehicle and abs(bb.rotation.yaw%360-math.degrees(compass))<50
            draw = False #Location(x=1048.991699, y=4990.129395, z=371.430725) #vehicle.get_transform().location
            self.negative_sample_read_count = 0
            img, positive_sample = self.save_bb(vehicle, bb, edges, world_2_camera, img, draw, image, distance, color, current_speed,
                        ego_loc, tl_bev_pixel_coordinate, positive_sample)

        elif (self.negative_sample_count<30 or self.positive_sample_count > self.negative_sample_count) and not positive_sample and current_speed > 0.1 and self.negative_sample_count > 15:
            points = np.ones(4) * (-1)
            color = 'None'
            self._carla_map = self._world.get_map()
            org_img = np.reshape(np.copy(image[1]), (image[1].shape[0], image[1].shape[1], 4))
            self.save_arrays(self.img_id, self._carla_map, org_img, points, distance, color, current_speed)

            negative_sample = True

        if positive_sample or negative_sample:
            cv2.imwrite(self.folder + '/front_' + str(self.tl_dataset_count) + '.png', img)
            if positive_sample:
                positive_sample_list_pickle = list(self.positive_sample_list[0])
                positive_sample_list_pickle[0] = 'front_' + str(self.tl_dataset_count) #self.tl_dataset_count
                with open(self.pickle_file_path, 'ab') as file:
                    pickle.dump(positive_sample_list_pickle, file)
            else:
                negative_sample_list_pickle = list(self.negative_sample_list[0])
                negative_sample_list_pickle[0] = 'front_' + str(self.tl_dataset_count) #self.tl_dataset_count
                with open(self.pickle_file_path, 'ab') as file:
                    pickle.dump(negative_sample_list_pickle, file)

            self.tl_dataset_count += 1


        self.camera_image = img
        self.previous_loc = ego_loc
        #print("current_speed:",current_speed)

        self.negative_sample_read_count += 1
        if positive_sample:
            self.negative_sample_read_count = 0
            self.positive_sample_count += 1
        else:
            self.negative_sample_count += 1

        return img

    def save_bb(self, vehicle, bb, edges, world_2_camera, img, draw, image, distance, color, current_speed, ego_loc, tl_bev_pixel_coordinate, positive_sample):
        # Calculate the dot product between the forward vector
        # of the vehicle and the vector between the vehicle
        # and the bounding box. We threshold this dot product
        # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
        forward_vec = vehicle.get_transform().get_forward_vector()
        ray = bb.location - vehicle.get_transform().location
        if np.dot(np.array([forward_vec.x, forward_vec.y, forward_vec.z]), np.array([ray.x, ray.y, ray.z])) > 1:
            # Cycle through the vertices
            verts = [v for v in bb.get_world_vertices(carla.Transform())]

            point_list = []
            for edge in edges:
                # Join the vertices into edges
                p1 = self.get_image_point(verts[edge[0]], self.K, world_2_camera)
                p2 = self.get_image_point(verts[edge[1]], self.K, world_2_camera)
                point_list.append(p1)
                point_list.append(p2)
                # Draw the edges into the camera output
            x_min, x_max, y_min, y_max = np.min(np.array(point_list)[:, 0]), np.max(np.array(point_list)[:, 0]), np.min(
                np.array(point_list)[:, 1]), np.max(np.array(point_list)[:, 1])

            if draw:
                cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0, 0, 255, 255), 1)
                cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)
                cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), (0, 0, 255, 255), 1)
                cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)

            points = np.array([x_min, y_min, x_max, y_max])
            org_img = np.reshape(np.copy(image[1]), (image[1].shape[0], image[1].shape[1], 4))
            if not (1-(points>0)).any():
                positive_sample = True
                self.save_arrays(self.img_id, self._carla_map, org_img, points, distance, color, current_speed, tl_bev_pixel_coordinate, positive_sample=positive_sample)
                #cv2.imwrite(self.current_path + '/front_' + str(self.tl_dataset_count) + '.png', img)
                print("positive sample saved")
                self.tl_dataset_count += 1


        return img, positive_sample

    def save_arrays(self, img_id, _carla_map, image, BB, distance, color, current_speed, tl_bev_pixel_coordinate=None, positive_sample=False):
        BB = np.array(BB)
        _carla_map = 12#int(''.join(list(_carla_map)[-2:]))
        """if type(tl_bev_pixel_coordinate) == type(None):
            tl_bev_pixel_coordinate = np.zeros((3,2))"""

        if color == "stop_sign":
            color = 4
        elif color == 'red':
            color = 3
        elif color == 'yellow':
            color = 2
        elif color == 'green':
            color = 1
        elif color == 'None':
            color = 6
        else:
            assert False

        current_weather = self._world.get_weather()
        weather_list = dir(current_weather)[50:-1]
        weather_dict = {}
        for key in weather_list:
            weather_dict.update({key:getattr(current_weather, key)})
        if positive_sample:
            self.positive_sample_list.append((img_id, 'Town_12', BB, distance, color, current_speed,
                                              tl_bev_pixel_coordinate,weather_dict))

        else:
            self.negative_sample_list.append(
                (img_id, 'Town_12', BB, distance, color, current_speed, tl_bev_pixel_coordinate,weather_dict))

        self.counter += 1

    def close_file(self):
        self.h5_file.close()


    def create_file(self):

        self.vehicle_dic_dataset = {}
        self.vehicle_index_dataset = 1

        self.counter = 0
        episode_number = 0
        file = 'episode_' + str(episode_number).zfill(3) + '.h5'
        file_exist = True
        file = '/workspace/tg22/tl_dataset/' + file
        while file_exist:
            episode_number += 1
            file = '/workspace/tg22/tl_dataset/' + 'episode_' + str(episode_number).zfill(3) #+ '.h5'
            file_exist = os.path.exists(file)

        os.mkdir(file)
        self.folder = file
        #file = 'image_bb_'+ str(self.file_count) + '_.h5'
        print("file:",file)
        self.h5_file = h5py.File(self.folder + '/episode_' + str(episode_number).zfill(3) + '.h5', 'a')
        self.pickle_file_path = self.folder + '/episode_' + str(episode_number).zfill(3) + '.pickle'

        asd = 0

    def get_image_point(self, loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

    def build_projection_matrix(self, w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K