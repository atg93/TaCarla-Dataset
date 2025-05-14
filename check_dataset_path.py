import paramiko
import json
#from count_number_of_frame import *
import ast
import os
import pandas as pd
import numpy as np

def log_dictionary(name, new_dict):
    # Create a DataFrame
    # Create a DataFrame with rows of data
    df = pd.DataFrame(list(new_dict.items()), columns=["Town", "Percentage"])

    # Create a wandb.Table from the DataFrame
    table = wandb.Table(dataframe=df)

    # Log the table
    wandb.log({name: table})


def merge_dictionary(new_scenario_dict, candidate_dict):
    for sample in candidate_dict.keys():
        if sample not in new_scenario_dict.keys():
            new_scenario_dict.update({sample:candidate_dict[sample]})
            asd = 0
        else:
            new_scenario_dict.update({sample: new_scenario_dict[sample] + candidate_dict[sample]})
            asd = 0
        asd = 0

    return new_scenario_dict


def get_name_list_dataset():
    # Define connection parameters
    ip2name = {"10.93.16.131":"idea","10.93.16.69":"helix","172.16.0.103":"hpc_datasets"}
    hostname_list  = ["172.16.0.103"] #"10.93.16.69",
    password_list  = ["smn8Eicx"] #"10.93.16.69",
    username_list  = ["tgorgulu"] #"10.93.16.69",
    port = 22  # default SSH port
    name2datalist = {}

    # Create an SSH client instance
    client = paramiko.SSHClient()
    # Automatically add host keys from the local HostKeys object (not recommended for production)
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    data_list = []
    town_list = ["Town12", "Town13"]

    for town_name in town_list:

        for index, hostname in enumerate(hostname_list):
            password = password_list[index]
            username = username_list[index]
            name = ip2name[hostname]

            # Connect to the server
            client.connect(hostname, port=port, username=username, password=password) #/cta/users/tgorgulu/datasets/carla_data_collection

            # Execute a command on the remote server
            if hostname == "172.16.0.103": # .13.19.28": #leaderboard_plant_pdm_Town12
                stdin, stdout, stderr = client.exec_command(
                    "cd datasets && cd carla_data_collection && cd leaderboard_plant_pdm_" + town_name + " && ls")
                stdout_0 = stdout.read().decode()
                stderr_0 = stderr.read().decode()
                hpc_file_list = stdout_0.split("\n")
                data_list = data_list + hpc_file_list
                asd = 0

            name2datalist.update({name + "_" + town_name: hpc_file_list})

    return name2datalist

def get_name_list():
    # Define connection parameters
    ip2name = {"10.93.16.131":"idea","10.93.16.69":"helix","172.16.0.103":"hpc_workspace"}
    hostname_list  = ["172.16.0.103", "10.93.16.131", "10.93.16.69"] #"10.93.16.69",
    password_list  = ["smn8Eicx","Hataraporu1.", "Hataraporu1."] #"10.93.16.69",
    username_list  = ["tgorgulu", "tg22", "tg22"] #"10.93.16.69",
    port = 22  # default SSH port
    name2datalist = {}

    # Create an SSH client instance
    client = paramiko.SSHClient()
    # Automatically add host keys from the local HostKeys object (not recommended for production)
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    data_list = []
    town_list = ["Town12", "Town13"]

    path = "//media/hdd/carla_data_collection/"
    for town_name in town_list:

        for index, hostname in enumerate(hostname_list):
            password = password_list[index]
            username = username_list[index]
            name = ip2name[hostname]

            # Connect to the server
            client.connect(hostname, port=port, username=username, password=password)

            # Execute a command on the remote server
            if hostname == "172.16.0.103": # .13.19.28": #leaderboard_plant_pdm_Town12
                data_dict = {}
                stdin, stdout, stderr = client.exec_command(
                    "cd workspace && cd tgorgulu && cd leaderboard_plant_pdm_" + town_name + " && ls")
                stdout_0 = stdout.read().decode()
                stderr_0 = stderr.read().decode()
                hpc_file_list = stdout_0.split("\n")
                data_list = data_list + hpc_file_list
                asd = 0
            else:
                data_dict = {}
                stdin, stdout, stderr = client.exec_command(
                    "cd " + path + " && cd leaderboard_plant_pdm_" + town_name + " && ls")
                stdout_0 = stdout.read().decode()
                stderr_0 = stderr.read().decode()
                hpc_file_list = stdout_0.split("\n")
                data_list = data_list + hpc_file_list
                asd = 0


            name2datalist.update({name+"_"+town_name:hpc_file_list})
            asd = 0



    client.close()

    return name2datalist

import wandb

if __name__ == '__main__':
    os.environ['TOWN_NAME'] = "Town12"
    dataset_dict = get_name_list_dataset()
    workspace_dict = get_name_list()

    wandb.init(project="Carla_data_transfer")
    # wandb_logger = WandbLogger(project="Carla_data_collection_project", name="collection")
    wandb.login(key="e7371c92ce98bb428b5aa96cf33589cb39abf955")

    #dataset_dict =
    datasets_total_town12 = len(dataset_dict['hpc_datasets_Town12'])
    datasets_total_town13 = len(dataset_dict['hpc_datasets_Town13'])
    #total_Town12 = []
    #total_Town13 = []
    total_Town12_number = 0
    total_Town13_number = 0
    for sample in workspace_dict.keys():
        if sample.split("_")[-1] == "Town12":
            total_Town12_number += len(workspace_dict[sample])
            total_Town12 = (datasets_total_town12/total_Town12_number)
            asd = 0
        elif sample.split("_")[-1] == "Town13":
            total_Town13_number += len(workspace_dict[sample])
            total_Town13 = (datasets_total_town13/total_Town13_number)
            asd = 0

    total_datasets_dict = {"Town12":int(total_Town12*100),"Town13":int(total_Town13*100)}
    log_dictionary("total_datasets_percentage", total_datasets_dict)



    pc_details_dict = {}
    for key in workspace_dict.keys():
        pc_details_dict.update({key:[]})

    for key in workspace_dict.keys():
        current_town = key.split("_")[-1]
        for sample in workspace_dict[key]:
            if len(sample.split("_")) > 5 and sample in dataset_dict[list(dataset_dict.keys())[0][0:13] + current_town]:
                pc_details_dict[key].append(1)


    final_pc_details_dict = {}
    for key in pc_details_dict.keys():
        final_pc_details_dict.update({key:np.sum(pc_details_dict[key])*100/len(workspace_dict[key])})
        asd = 0

    log_dictionary("pc_based", final_pc_details_dict)
    asd = 0
asd = 0
