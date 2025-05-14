import paramiko
import json
#from count_number_of_frame import *
import ast
import os
import gzip
import select


def minpeed_causes(simulation_results):
    infractions = simulation_results['infractions']

    other_causes = False
    for key in infractions.keys():
        if len(infractions[key]) and key != 'min_speed_infractions' and key != 'route_dev':
            other_causes = True

    if other_causes == False and len(infractions['min_speed_infractions'])>0:
        return True
    else:
        return False

def episode_limit(simulation_results, boxes_len):
    infractions = simulation_results['infractions']

    other_causes = False
    for key in infractions.keys():
        if len(infractions[key]) and key != 'min_speed_infractions' and key != 'route_dev':
            other_causes = True

    if other_causes == False and boxes_len == 400:
        return True
    else:
        return False


hostname_list  = ["10.93.16.131", "10.93.16.69", "172.16.0.103"] #"10.93.16.69",
password_list  = ["Hataraporu1.", "Hataraporu1.", "smn8Eicx"] #"10.93.16.69",
username_list  = ["tg22", "tg22", "tgorgulu"] #"10.93.16.69",
main_path_list = ["//media/hdd/carla_data_collection", "//media/hdd/carla_data_collection", "workspace/tgorgulu"]

port = 22  # default SSH port
username = "tg22"
os.environ['TOWN_NAME'] = "Town12"

scenario_list = []
scenario_list_planing = []
# Create an SSH client instance
client = paramiko.SSHClient()
# Automatically add host keys from the local HostKeys object (not recommended for production)
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
total_sample = 0
total_memory = 0
global_boxes_number = 0
two_d_light_p_count = 0
object_count_dict = {'Walker':0, 'Car':0, 'Police':0, 'Ambulance':0, 'Firetruck':0, 'Crossbike':0, 'Construction':0}

new_scenario_dict = {}
sng_cmd = "singularity exec --nv --bind /cta/share/tair/,/cta/users/tgorgulu/workspace //cta/users/tgorgulu/containers/leaderboard.sif "
for index, hostname in enumerate(hostname_list):
    hostname = hostname_list[index]
    password = password_list[index]
    username = username_list[index]
    asd = 0
    try:
        # Connect to the server
        client.connect(hostname, port=port, username=username, password=password)
        if os.getenv("TOWN_NAME") == "Town12":
            town_name = "Town12"
        elif os.getenv("TOWN_NAME") == "Town13":
            town_name = "Town13"

        # Execute a command on the remote server
        data_dict = {}
        stdin, stdout, stderr = client.exec_command(
            "cd " + main_path_list[index] + " && cd leaderboard_plant_pdm_" + town_name + " && ls")
        stdout_0 = stdout.read().decode()
        stderr_0 = stderr.read().decode()
        hpc_file_list = stdout_0.split("\n")
        for file_index, current_file in enumerate(hpc_file_list):
            stdin, stdout, stderr = client.exec_command(
                "cd " + main_path_list[index] + " && cd leaderboard_plant_pdm_" + town_name + " && cd " + current_file + "&& ls")
            stdout_0 = stdout.read().decode()
            stderr_0 = stderr.read().decode()
            score_path = stdout_0.split("\n")
            print("town_name: ",town_name,"hostname:", hostname,"percentage: ",file_index*100/len(hpc_file_list), "global_boxes_number: ",global_boxes_number, "two_d_light_p_count:",two_d_light_p_count)
            if "score.txt" in score_path:

                stdin, stdout, stderr = client.exec_command(
                    "cd " + main_path_list[index]+ " && cd leaderboard_plant_pdm_" + town_name + " && cd " + current_file + "&& cd boxes && find . -maxdepth 1 -type f | wc -l")
                stdout_0 = stdout.read().decode()
                stderr_0 = stderr.read().decode()
                boxes_number = stdout_0.split("\n")[0]

                stdin, stdout, stderr = client.exec_command(
                    "cd " + main_path_list[index]+ " && cd leaderboard_plant_pdm_" + town_name + " && cd " + current_file + "&& cat score.txt")
                stdout_0 = stdout.read().decode()
                stderr_0 = stderr.read().decode()
                text_content = stdout_0.split(" ")
                scenario_list.append(text_content[0])

                stdin, stdout, stderr = client.exec_command(
                    "cd " + main_path_list[
                        index] + " && cd leaderboard_plant_pdm_" + town_name + " && cd " + current_file + "&& cat simulation_results.json")
                stdout_0 = stdout.read().decode()
                stderr_0 = stderr.read().decode()
                simulation_results = json.loads(stdout_0)
                if (float(text_content[4]) > 70.0) or (float(text_content[4]) < 70.0 and minpeed_causes(simulation_results)) or (float(text_content[4]) < 70.0 and episode_limit(simulation_results, boxes_number)):
                    scenario_list_planing.append(text_content[0])
                    global_boxes_number += int(boxes_number)


    except Exception as e:
        print(f"Connection failed: {e, hostname}")

import pickle

with open("scenario_list_"+os.getenv("TOWN_NAME")+".pkl","wb") as pickle_file:
    pickle.dump(scenario_list, pickle_file)

with open("scenario_list_planing_"+os.getenv("TOWN_NAME")+".pkl","wb") as pickle_file:
    pickle.dump(scenario_list_planing, pickle_file)

print("final global_boxes_number: ",global_boxes_number)
#print("final two_d_light_p_count: ",two_d_light_p_count)
#print("object_count_dict:",object_count_dict)