import pickle

import numpy as np


class Check_F_and_L_Dataset():
    def __init__(self):
        self.current_scenario_info = None
        self.episode_number = 0
        self.dataset_folder_name = '/workspace/tg22/f_and_l_dataset'
        #assert os.path.exists(self.dataset_folder_name+'/'+'episode_007/episode_007_1.h5')
        self.read_data(self.dataset_folder_name+'/'+'episode_3/lidar/lidar_0.npy')

    def read_data(self, file_name):
        data= np.load(file_name)

        asd = 0


Check_F_and_L_Dataset()