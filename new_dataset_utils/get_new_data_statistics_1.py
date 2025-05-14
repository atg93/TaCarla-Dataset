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

data_path = "//media/hdd/carla_data_collection/leaderboard_plant_pdm_" + os.getenv("TOWN_NAME") +"/"
assert os.path.exists(data_path)

episode_list = os.listdir(data_path)

new_scenario_dict = {}
sng_cmd = "singularity exec --nv --bind /cta/share/tair/,/cta/users/tgorgulu/workspace //cta/users/tgorgulu/containers/leaderboard.sif "

two_d_light_p_count = 0
object_count_dict = {'Walker':0, 'Car':0, 'Police':0, 'Ambulance':0, 'Firetruck':0, 'Crossbike':0, 'Construction':0}

for index, episode in enumerate(episode_list):
    print(index*100/len(episode_list))
    boxes_path = data_path + episode +"/boxes/"
    boxes_list = os.listdir(boxes_path)
    for boxes in boxes_list:
        with gzip.open(boxes_path + boxes, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        for content in data:
            if content['class'] == "two_d_light_p":
                two_d_light_p_count += 1

            if content['class'] in object_count_dict.keys():
                object_count_dict[content['class']] += 1

            if content['class'] == "Lane":
                break
        asd = 0
        asd = 0
    asd = 0

print("two_d_light_p_count: ",two_d_light_p_count)
print("object_count_dict: ",object_count_dict)

