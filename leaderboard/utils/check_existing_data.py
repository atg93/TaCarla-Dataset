import os
from pathlib import Path
#from ssh_connection import *
import time

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

class Check_Existing_Data:

    def __init__(self):
        if os.getenv("TOWN_NAME") == "Town12":
            data_path = os.environ['DATASAVEPATH'] + "/leaderboard_plant_pdm_Town12/"
            if not os.path.exists(data_path):
                os.makedirs(data_path)
        elif os.getenv("TOWN_NAME") == "Town13":
            data_path = os.environ['DATASAVEPATH'] + "/leaderboard_plant_pdm_Town13/"
            if not os.path.exists(data_path):
                os.makedirs(data_path)
        self.data_path =data_path
        self.first_check = True
        self.current_time = time.time()
        self.last_time = time.time()

        self.check_existing_data(self.data_path)
        asd = 0



    def check_existing_data(self, data_path):
        """self.existing_route_dict = {}

        files = find_files(data_path)

        for file in files:
            data = read_txt_file(file)

            if float(data[0][4]) > 70.0 and data[0][0] + '_' + data[0][1] not in self.existing_route_dict.keys():
                self.existing_route_dict.update({data[0][0] + '_' + data[0][1]: data[0][4]})"""

        #print("existing route dict:", len(self.existing_route_dict), self.existing_route_dict)
        self.existing_route_dict = {}
        #if type(os.getenv("PYCHARM")) == type(None):
        #    self.existing_route_dict = get_scenario_dict()[0]
        asd = 0





    def __call__(self, new_scenario):

        # If 5 minutes (300 seconds) have passed
        if self.first_check or (self.current_time - self.last_time >= 60*5):
            self.check_existing_data(self.data_path)
            self.first_check = False
            self.last_time = time.time()
        self.current_time = time.time()
        if new_scenario in list(self.existing_route_dict.keys()):
            return True
        else:
            return False
