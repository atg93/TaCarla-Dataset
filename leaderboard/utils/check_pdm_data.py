import os
from pathlib import Path

import shutil

def find_files(path):
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if 'txt' in filename.split('.'):
                files.append(os.path.join(root, filename))
    return files

def read_txt_file(path):

    # Specify the path to the file
    file_path = Path(path)

    data = []
    # Read the file line by line
    with file_path.open('r') as file:
        for line in file:
            data.append(line.strip().split(' '))

    return data

class Check_Pdm_Data:

    def __init__(self):

        try:
            data_path = '/workspace/tg22/leaderboard_plant_pdm/'
            destination_path = '/workspace/tg22/leaderboard_plant_pdm_final_data/'
            assert os.path.exists(data_path)
        except:
            data_path = "/cta/users/tgorgulu/workspace/tgorgulu/leaderboard_plant_pdm/"
            destination_path = '/cta/users/tgorgulu/workspace/tgorgulu/leaderboard_plant_pdm/leaderboard_plant_pdm_final_data/'
            assert os.path.exists(data_path)

        self.existing_route_dict = {}

        files = find_files(data_path)

        for file in files:
            data = read_txt_file(file)

            if float(data[0][4]) > 70.0 and data[0][0]+'_'+data[0][1] not in self.existing_route_dict.keys():
                self.existing_route_dict.update({data[0][0]+'_'+data[0][1]:data[0][4]})
                copy_file_path = '/'.join(file.split('/')[:-1])
                shutil.move(copy_file_path, destination_path)


        print("existing route dict:", len(self.existing_route_dict), self.existing_route_dict)
        asd = 0





    def __call__(self, new_scenario):
        if new_scenario in list(self.existing_route_dict.keys()):
            return True
        else:
            return False

if __name__ == '__main__':
    pdm_data = Check_Pdm_Data()