import numpy as np
import math
import torch
from PIL import Image
import cv2
import time

from collections import deque
import sys

#sys.path.append('/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception')
#sys.path.append('/cta/users/tgorgulu/git/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception')
#git/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/detection_utils/

from leaderboard.autoagents.traditional_agents_files.perception.tairvision_utils.prepare_lidar_data import Prepare_Lidar_Data


class Carla_to_Lss_Converter:
    def __init__(self, monocular,number_of_camera=6, device=2, temporal_range=3):
        self.monocular = monocular
        self.device = device
        self.temporal_range = temporal_range

        self.cams = ['front_left', 'front', 'front_right', 'back_left', 'back', 'back_right']#['front', 'front_right', 'front_left', 'back', 'back_left', 'back_right']
        #self.cams = ['front_left', 'front', 'front_right', 'back']
        #'images': torch.zeros((1, 3, number_of_camera, 3, 512, 1408), dtype=torch.float32, device=self.device)
        # 'intrinsics': torch.zeros((1, 1, number_of_camera, 3, 3), dtype=torch.float32)
        # 'cams_to_lidar': torch.zeros((1, 1, number_of_camera, 4, 4), dtype=torch.float32)
        # 'view': torch.zeros((1, 1, 1, 4, 4), dtype=torch.float32)

        self.batch = {'view': torch.zeros((1, 1, 1, 4, 4), dtype=torch.float32)}

        """self.batch['view'][0, 0, 0] = torch.tensor([[[[[0, -2, 0, 99.5],
                                                    [-2, 0, 0, 99.5],
                                                    [0, 0, 0.8, 4.3],
                                                    [0, 0, 0, 1]]]]])"""

        self.prepare_lidar_data = Prepare_Lidar_Data()

        self.images_que_dict = {}
        self.intrinsics_que_dict = {}
        self.cams_to_lidar_que_dict = {}
        for camera in self.cams:
            self.images_que_dict.update({camera:deque(maxlen=temporal_range)})
            self.intrinsics_que_dict.update({camera:deque(maxlen=temporal_range)})
            self.cams_to_lidar_que_dict.update({camera:deque(maxlen=temporal_range)})

        for camera in self.cams:
            for index in range(temporal_range):
                self.images_que_dict[camera].append(torch.zeros((1, 1, 1, 3, 512, 1408), dtype=torch.float32))
                self.intrinsics_que_dict[camera].append(torch.zeros((1, 1, 1, 3, 3), dtype=torch.float32))
                self.cams_to_lidar_que_dict[camera].append(torch.zeros((1, 1, 1, 4, 4), dtype=torch.float32))


        self.view_que = deque(maxlen=temporal_range)
        for index in range(temporal_range):
            self.view_que.append(
                torch.tensor([[[[[0, -2, 0, 99.5],
                                 [-2, 0, 0, 99.5],
                                 [0, 0, 0.8, 4.3],
                                 [0, 0, 0, 1]]]]]))

        self.batch.update({'view':torch.concatenate(tuple(list(self.view_que)),dim=1).cuda(self.device)})

        self.lidar_que = deque(maxlen=temporal_range)
        self.future_motion_que = deque(maxlen=temporal_range)

        for _ in range(temporal_range):
            self.lidar_que.append(torch.zeros([1, 1, 1,  70000, 5]).cuda(self.device))

            self.future_motion_que.append(torch.zeros([1, 1,  4, 4]).cuda(self.device))

        self.update_dict()


        self.initialization = True

    def update_dict(self):


        image_dict = {}
        intrinsic_dict = {}
        cams_to_lidar_dict = {}
        for camera in self.cams:
            image_dict.update({camera:torch.concatenate(tuple(list(self.images_que_dict[camera])),dim=1).cuda(self.device)})
            intrinsic_dict.update({camera:torch.concatenate(tuple(list(self.intrinsics_que_dict[camera])),dim=1).cuda(self.device)})
            cams_to_lidar_dict.update({camera:torch.concatenate(tuple(list(self.cams_to_lidar_que_dict[camera])),dim=1).cuda(self.device)})

        image_sample = []
        intrinsic_sample = []
        cams_to_lidar_sample = []
        for camera in self.cams:
            image_sample.append(image_dict[camera])
            intrinsic_sample.append(intrinsic_dict[camera])
            cams_to_lidar_sample.append(cams_to_lidar_dict[camera])



        self.batch.update({'images':torch.concatenate(image_sample,dim=2).cuda(self.device)})
        self.batch.update({'intrinsics':torch.concatenate(intrinsic_sample,dim=2).cuda(self.device)})
        self.batch.update({'cams_to_lidar':torch.concatenate(cams_to_lidar_sample,dim=2).cuda(self.device)})

        self.batch.update({'lidar_data':torch.concatenate(tuple(list(self.lidar_que)), dim=1).cuda(self.device)})

        self.future_motion_que.append(torch.eye(4).unsqueeze(0).unsqueeze(0).cuda(self.device))
        self.batch.update({'future_egomotion':torch.concatenate(tuple(list(self.future_motion_que)), dim=1).cuda(self.device)})


        asd = 0


    def find_intrinsics(self, width, height, fov, x, y, z, roll, pitch, yaw, use_nu = True):
        if use_nu:
            roll = -math.radians(roll)
            y = -y
            pitch = math.radians(pitch)
            yaw = -math.radians(yaw)
            nu_matrix = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

        else:
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)
        fx = (width / 2.0) / math.tan(math.radians(fov / 2.0))
        cx = width / 2.0
        cy = height / 2.0

        ## Intrinsic matrix K
        K = np.array([[fx, 0, cx],
                      [0, fx, cy],
                      [0, 0, 1]])

        # Translation vector T
        T = np.array([x, y, z])

        # Rotation matrices for roll, pitch, and yaw
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]])

        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                       [0, 1, 0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])

        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]])

        # Combine the rotation matrices

        R = np.dot(Rz, np.dot(Ry, Rx))
        if use_nu:
            R_nu = R @ nu_matrix
        else:
            R_nu = R

        # Extrinsic matrix RT
        RT = np.zeros((4, 4))
        RT[:3, :3] = R_nu
        RT[:3, 3] = T
        RT[3, 3] = 1

        return torch.tensor(K).to(device=self.device), torch.tensor(RT).to(device=self.device)

    def find_ext(self, x, y, z, roll, pitch, yaw, use_nu = True):
        if use_nu:
            roll = -math.radians(roll)
            y = -y
            pitch = math.radians(pitch)
            yaw = -math.radians(yaw)
            nu_matrix = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

        else:
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)


        # Translation vector T
        T = np.array([x, y, z])

        # Rotation matrices for roll, pitch, and yaw
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]])

        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                       [0, 1, 0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])

        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]])

        # Combine the rotation matrices

        R = np.dot(Rz, np.dot(Ry, Rx))
        if use_nu:
            R_nu = R @ nu_matrix
        else:
            R_nu = R

        # Extrinsic matrix RT
        RT = np.zeros((4, 4))
        RT[:3, :3] = R_nu
        RT[:3, 3] = T
        RT[3, 3] = 1

        return torch.tensor(RT).to(device=self.device)

    def adding_element_to_que(self, input_data_sensor, transforms_val, intrinsics, extrinsics):
        for index, camera in enumerate(self.cams):
            img = Image.fromarray(input_data_sensor[camera][1]).convert('RGB') #Image.fromarray(input_data_sensor[self.cams[camera]][1][:, :, -2::-1])
            img, ints, extr = transforms_val(img, list(intrinsics[camera].cpu().numpy()), list(extrinsics[camera].cpu().numpy()))
            #self.batch['images'][0, 0, index, :, :, :] = img
            #self.batch['intrinsics'][0, 0, index, :, :] = intrinsics[camera]
            #self.batch['cams_to_lidar'][0, 0, index, :, :] = extrinsics[camera]
            self.images_que_dict[camera].append(img.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            self.intrinsics_que_dict[camera].append(ints.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            self.cams_to_lidar_que_dict[camera].append(extr.unsqueeze(0).unsqueeze(0).unsqueeze(0))

    def create_lss_batch(self, input_data_sensor, intrinsics, extrinsics, transforms_val, future_ego_motion):
        if self.initialization:
            self.initialization = False
            lidar_image = np.zeros((200, 200, 3)).astype(np.uint8)

            for index in range(self.temporal_range):
                self.adding_element_to_que(input_data_sensor, transforms_val, intrinsics, extrinsics)

                lidar_data, lidar_image, _ = self.prepare_lidar_data(input_data_sensor)
                self.lidar_que.append(lidar_data)

                self.future_motion_que.append(future_ego_motion.cuda(self.device))

        else:
            self.adding_element_to_que(input_data_sensor, transforms_val, intrinsics, extrinsics)

            lidar_data, lidar_image, _ = self.prepare_lidar_data(input_data_sensor)
            self.lidar_que.append(lidar_data)

            self.future_motion_que.append(future_ego_motion.cuda(self.device))

        self.update_dict()


        return self.batch, lidar_image



