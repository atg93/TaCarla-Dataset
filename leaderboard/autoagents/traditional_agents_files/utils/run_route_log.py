import os
import h5py
import numpy as np
import carla

class Run_Route_Log():

    def __init__(self,current_task='Accident_3'):
        self.current_task = current_task
        self.current_scenario_info = None
        self.episode_number = 0
        self.dataset_folder_name = '/workspace/tg22/route_dataset_log_correction/'+'Accident_3_roach_1' +'/final_data'#'ParkingExit_1_2'
        self.files_name = self.get_all_files_name()
        self.sort_files()
        self.file_number = 0
        self.control_data_index = 0
        print("self.scenario_array:",self.scenario_array,"scenario_index:",self.scenario_array[self.file_number])
        print("self.scenario_dict:",self.scenario_dict[self.scenario_array[self.file_number]])
        self.read_control(self.scenario_dict[self.scenario_array[self.file_number]])
        self.save_files_name = self.scenario_dict[self.scenario_array[self.file_number]]

        asd = 0

    def save_world(self, world):
        self.world = world

    def sort_files(self):
        self.scenario_dict = {}

        for index, filename in enumerate(self.files_name):
            self.get_scenario_name(index, self.dataset_folder_name+'/'+filename)
        self.scenario_array = np.sort(list(self.scenario_dict.keys()))#[2:]

        asd = 0


    def get_scenario_name(self, index, filename):
        try:
            print("filename: ",filename)
            with h5py.File(filename, 'r') as file:
                parts = file['current_scenario_name'][0][0].decode('utf-8').split('_')
                integer_value = int(parts[1])
                self.scenario_dict.update({integer_value:filename})
        except:
            print(filename,"is NOT read")



    def reset(self):
        self.control_data_index = 0
        self.file_number += 1
        print("self.scenario_array:",self.scenario_array)

        self.read_control(self.scenario_dict[self.scenario_array[self.file_number]])
        self.save_files_name = self.scenario_dict[self.scenario_array[self.file_number]]

    def get_all_files_name(self):
        files = [f for f in os.listdir(self.dataset_folder_name) if os.path.isfile(os.path.join(self.dataset_folder_name, f))]
        return files

    def read_control(self, filename):
        self.control_data = []

        print("filename:",filename)
        self.filename = filename
        with h5py.File(filename, 'r') as file:
            # Access a dataset within the file
            # Replace 'dataset_name' with the name of the dataset you want to access
            #self.control_data.append(file['control'][:])#file['score_route'][0] #file['file_name'][0]
            file_name_decode = file['file_name'][0][0].decode('utf-8')
            if 'shrinked' in file['file_name'][0][0].decode('utf-8').split('_'):
                file_name_decode = '_'.join(file['file_name'][0][0].decode('utf-8').split('_')[1:])
            if file_name_decode == self.current_task:# and file['score_composed'][0][0] >= 85 and file['score_route'][0][0]==100:
                print(filename, "file['score_composed'][0][0]:",file['score_composed'][0][0])
                self.current_scenario_name = file['current_scenario_name'][0][0].decode('utf-8')
                print("self.current_scenario_name: ",self.current_scenario_name)
                self.control_data.append(file['control'][:])
                self.loc_array = np.array(file['ego_loc'])
                try:
                    self.rot_array = np.array(file['ego_rot'])
                    self.speed_array = np.array(file['ego_speed'])
                except:
                    pass




    def get_control(self):
        control = self.control_data[0][self.control_data_index]
        carla_control = carla.VehicleControl(
            throttle=control[0],
            steer=control[1],
            brake=control[2]
        )

        #if self.control_data_index > self.control_threshold:
        #    asd += 1

        self.control_data_index += 1

        return carla_control