import os
import h5py
import numpy as np

def get_info_from_h5(filename):
    pass_scenario = False
    print("filename: ",filename)

    with h5py.File(filename, 'r') as file:
        # Access a dataset within the file
        # Replace 'dataset_name' with the name of the dataset you want to access
        # print(filename, "file['score_penalty'][0]:", file['score_penalty'][0])
        route_scenario = file['current_scenario_name'][0][0]
        score_composed = file['score_composed'][0][0]
        task_name = file['file_name'][0][0]
        if score_composed > 85.0:
            pass_scenario = True
            print("task_name: ",task_name,"score_composed: ",score_composed,"route_scenario: ", route_scenario)

        asd = 0

    return pass_scenario, score_composed, route_scenario, task_name

def take_current_logs(path_files, excel_file):
    for h5_file in path_files:
        pass_scenario, score_composed, route_scenario, task_name = get_info_from_h5(h5_file)
        if pass_scenario:
            excel_file.append((task_name.decode('utf-8'), route_scenario.decode('utf-8')))

    return excel_file

directory = '/workspace/tg22/route_dataset_log/'

files = os.listdir(directory)

out_of_scope = ['undeterministic','eski']
print('files: ', files)

dummy_files = files# ['NonSignalizedJunctionLeftTurn_2', 'ParkingCrossingPedestrian_3_roach']

folder_inside_folder = ['final_data']

excel_file = []
for file in dummy_files:
    path = directory + file



    path_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


    new_path_files = []
    #for fpf in path_files:
    #    new_path_files.append(path+'/'+ fpf)

    final_path = directory + file + '/final_data'

    if os.path.exists(final_path):
        final_path_files = [f for f in os.listdir(final_path) if os.path.isfile(os.path.join(final_path, f))]
        for fpf in final_path_files:
            new_path_files.append(final_path+'/'+ fpf)

    excel_file = take_current_logs(new_path_files, excel_file)

organized_file_list = []
organized_file_dict = {}

for fl in excel_file:
    split_fl = fl[0].split('_')
    if 'shrinked' in split_fl:
        task_name = split_fl[1] + '_' + split_fl[2]
    else:
        task_name = fl[0]
    route_name = fl[1]

    organized_file_list.append((task_name,route_name))
    if task_name in organized_file_dict.keys():
        organized_file_dict[task_name].append(route_name)
    else:
        organized_file_dict.update({task_name:[route_name]})

import json

with open('organized_route.json','w') as json_file:
    json.dump(organized_file_dict, json_file)


import pandas as pd

# Your list of tuples
data = excel_file

# Decode bytes to string if necessary and create a DataFrame
df = pd.DataFrame(data, columns=['Column', 'Row'])

# Convert bytes to strings if your tuples are in bytes
df['Column'] = df['Column'].apply(lambda x: x)
df['Row'] = df['Row'].apply(lambda x: x)

# Create a pivot table. The values can be a count or a specific marker (e.g., 1)
pivot_table = df.pivot_table(index='Row', columns='Column', aggfunc=len, fill_value=0)

# Write to an Excel file
pivot_table.to_excel('output.xlsx', engine='openpyxl')

print("Excel file has been created successfully with specified structure.")

asd = 0

