import os
import json

def get_save_path():
    if os.getenv("TOWN_NAME") == "Town12":
        try:
            path = "/media/hdd/carla_data_collection/leaderboard_plant_pdm_Town12/"
            assert os.path.exists(path)
        except:
            path = "/cta/users/tgorgulu/workspace/tgorgulu/leaderboard_plant_pdm_Town12/"
            assert os.path.exists(path)
    elif os.getenv("TOWN_NAME") == "Town13":
        try:
            path = "/media/hdd/carla_data_collection/leaderboard_plant_pdm_Town13/"
            assert os.path.exists(path)
        except:
            path = "/cta/users/tgorgulu/workspace/tgorgulu/leaderboard_plant_pdm_Town13/"
            assert os.path.exists(path)

    return path

import pickle
import json
import sys

def get_folder_size(folder_path):
    total_size = 0
    # Walk through all subdirectories and files
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Optionally, skip symbolic links
            if not os.path.islink(file_path):
                try:
                    total_size += os.path.getsize(file_path)
                except OSError:
                    # File might have been deleted or is inaccessible
                    pass
    return total_size / (1024 ** 3)


def get_file_number():
    os.environ['TOWN_NAME'] = "Town12"
    try:
        path = "/media/hdd/carla_data_collection/leaderboard_plant_pdm_Town12/"
        assert os.path.exists(path)
    except:
        path = "/cta/users/tgorgulu/workspace/tgorgulu/leaderboard_plant_pdm_Town12/"
        assert os.path.exists(path)

    path = get_save_path()

    scenario_dict = {}
    print("path: ",path)
    file_list = os.listdir(path)
    sum = 0
    memory_sum = 0
    print("len:",len(file_list))
    for file_name in file_list:
        if not("Town12" in file_name.split('_') or "Town13" in file_name.split('_')):
            asd = 0
            continue
        new_path = path + file_name + "/detection/rgb_camera/front"
        try:
            frame_list = os.listdir(new_path)
        except:
            continue

        sum += len(frame_list)
        memory_sum += get_folder_size(path + file_name)

        score_txt_path = '/'.join(new_path.split('/')[:6]) + '/score.txt'

        # Open and read the file 'example.txt'
        try:
            with open(score_txt_path, 'r') as file:
                content = file.read().split(' ')

            #if float(content[4]) < 70.0:
            #    continue

            if content[0] not in scenario_dict.keys():
                scenario_dict.update({content[0] +"_"+content[1]:float(content[4])})
            else:
                scenario_dict[content[0] +"_"+content[1]].append(float(content[4]))

            #print(content)
        except FileNotFoundError:
            #print("The file 'example.txt' was not found.")
            pass

    return scenario_dict, sum, memory_sum

def run_main():
    scenario_dict, sum, memory_sum = get_file_number()
    print(scenario_dict)
    print("sample number:", sum)
    print("memory_sum:", memory_sum)
    asd = 0


#if __name__ == '__main__':
run_main()


