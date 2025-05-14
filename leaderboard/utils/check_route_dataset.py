import os
import h5py
import numpy as np
import carla
import os
import shutil

# Specify the directory you want to list files from
folder_path = '/path/to/your/folder'

# List all files in the directory


class Check_Il_Data():
    def __init__(self):
        self.current_scenario_info = None
        self.episode_number = 0
        self.dataset_folder_name = '/workspace/tg22/route_dataset_log/'+ 'ParkingExit_1_new'
        self.files_name = self.get_all_files_name()
        self.control_data = []
        self.copy_file_name = []
        for filename in self.files_name:
            copy_file = self.read_control(self.dataset_folder_name+'/'+filename)

            if copy_file:
                current_folder_path = self.dataset_folder_name + '/' + filename
                new_folder_path = self.dataset_folder_name + '/final_data' + '/' + filename
                shutil.move(current_folder_path, new_folder_path)




        asd = 0

    def get_all_files_name(self):
        files = [f for f in os.listdir(self.dataset_folder_name) if os.path.isfile(os.path.join(self.dataset_folder_name, f))]
        return files


    def read_control(self, filename):
        try:
            copy_file = False
            with h5py.File(filename, 'r') as file:
                # Access a dataset within the file
                # Replace 'dataset_name' with the name of the dataset you want to access
                #print(filename, "file['score_penalty'][0]:", file['score_penalty'][0])
                """print(file['score_composed'][0])
                #print(file.keys())
                print(filename, file['file_name'][0], file['current_scenario_name'][0], "file['score_composed'][0]:",
                      file['score_composed'][0], file['score_penalty'][0])"""
                self.control_data.append(file['control'][:])
                print(filename, file['file_name'][0], file['current_scenario_name'][0])
                for file_key in list(file.keys()):
                    if 'state' in file_key.split('-'):
                        asd = 0

                if file['score_composed'][0] >= 75.0:# and file['score_penalty'][0] >= 0.9:
                    print(filename, file['file_name'][0],file['current_scenario_name'][0],"file['score_composed'][0]:",file['score_composed'][0])
                    copy_file = True
            return copy_file
        except:
            return False

if __name__ == '__main__':
    check = Check_Il_Data()
