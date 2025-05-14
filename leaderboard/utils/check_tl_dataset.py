import os
import h5py
import numpy as np
import carla
import os
import shutil

# Specify the directory you want to list files from
folder_path = '/path/to/your/folder'

# List all files in the directory
import cv2

import pickle

from gym.wrappers.monitoring.video_recorder import ImageEncoder

class Check_Tl_Dataset():
    def __init__(self):
        self.current_scenario_info = None
        self.episode_number = 0
        self.dataset_folder_name = '/workspace/tg22/tl_dataset'
        #assert os.path.exists(self.dataset_folder_name+'/'+'episode_007/episode_007_1.h5')
        self.read_data(self.dataset_folder_name+'/'+'episode_002/episode_002.pickle')#'episode_071/episode_071.pickle'

    def save_video(self, mask_list):
        video_path = f'{str("esat_dataset")}.mp4'
        encoder = ImageEncoder(video_path, mask_list[0].shape, 30, 30)
        for im in mask_list:
            encoder.capture_frame(im)
        encoder.close()
        encoder = None
        self.episode_number += 1

    def draw(self, start_point, end_point, half_width=50, half_height = 30):
        image = np.zeros((400, 400,3), dtype=np.uint8)
        start_point = np.array([start_point[0]*4, start_point[1]*4]) #+ 100
        end_point = np.array([end_point[0]*4, end_point[1]*4])
        # Calculate the top-left and bottom-right coordinates of the rectangle
        """top_left = (int(center_x - half_width), int(center_y - half_height))
        bottom_right = (int(center_x + half_width), int(center_y + half_height))"""

        # Draw the rectangle on the image
        cv2.rectangle(image, tuple(start_point.astype(np.int32)), tuple(end_point.astype(np.int32)), (255,255,255), thickness=-1)

        return image

    def draw_rgb(self, img, x_min, y_min, x_max, y_max):
        cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0, 0, 255, 255), 1)
        cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)
        cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), (0, 0, 255, 255), 1)
        cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)

        return img

    def read_data(self, file_name):
        data = []
        mask_list = []
        with open(file_name, 'rb') as file:
            index = 0
            while True:
                try:
                    # Attempt to load the next object in the file
                    object = pickle.load(file)
                    if np.sum(object[2]) > 1:
                        data.append(object)
                        path = self.dataset_folder_name+'/'+'episode_002/' + object[0]+ '.png'
                        img = cv2.imread(path)
                        x_min, y_min, x_max, y_max = object[2][0], object[2][1], object[2][2], object[2][3]
                        mask = self.draw_rgb(img, x_min, y_min, x_max, y_max)
                        cv2.imwrite('mask'+str(index)+'.png',mask)
                        mask_list.append(mask)
                        index += 1

                        asd = 0

                except EOFError:
                    # End of file reached
                    break

        self.save_video(mask_list)
        asd = 0


Check_Tl_Dataset()

