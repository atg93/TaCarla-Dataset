import os

import shutil

def find_files(path):
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if 'txt' in filename.split('.'):
                files.append(os.path.join(root, filename))
    return files

data_path = '/workspace/tg22/plant_dataset_0/Routes_Town12_Scenario0_Seed2010_trainable_big_data_1'
assert os.path.exists(data_path)

existing_route_dict = {}

files = find_files(data_path)

for fi in files:
    original_folder = '/'.join(fi.split('/')[:-1]) + '/boxes'
    new_folder = '/'.join(fi.split('/')[:-1]) + '/detected_tl_boxes'

    shutil.copytree(original_folder, new_folder)

    asd = 0