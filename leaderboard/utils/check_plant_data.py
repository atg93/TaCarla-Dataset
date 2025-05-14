#data_path = '/workspace/tg22/leaderboard_plant/'
data_path = '/workspace/tg22/lead_plant_leaderboard_1_deneme/'
new_data_path = '/workspace/tg22/filtered_leaderboard_1_data/'

import os
import shutil

import numpy as np
import json

assert os.path.exists(data_path)

def list_folders(path):
    folders = [f.name for f in os.scandir(path) if f.is_dir()]
    return folders

def list_all_files(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file_name in files:
            all_files.append(os.path.join(root, file_name))
    return all_files

def read_txt_file(path):
    with open(path, 'r') as file:
        # Read the entire content of the file
        content = file.read()

    return content.split(' ')

folders = list_folders(data_path)

for fol in folders:
    name = fol.split('_')
    current_folder_path = data_path + fol
    assert os.path.exists(current_folder_path)
    score_file_path = [file for file in list_all_files(current_folder_path) if 'txt' in file.split('.')]
    if len(score_file_path) != 0:
        content = read_txt_file(score_file_path[0])
        name[2] = content[0].split('_')[0]
        name[3] = str(content[0].split('_')[1])
        name.append(str(content[4]))
        new_name = '_'.join(name)

        if float(content[4]) > 70 or ('ParkingExit'== content[0].split('_')[0] and float(content[-1])>=1.0):
            new_accepted_data_path = new_data_path + 'accepted/' + new_name + '/'
            current_data_path = data_path + fol + '/'
            shutil.copytree(current_data_path, new_accepted_data_path)
        else:
            new_accepted_data_path = new_data_path + 'declined/' + new_name + '/'
            current_data_path = data_path + fol + '/'
            shutil.copytree(current_data_path, new_accepted_data_path)


        asd = 0

asd= 0