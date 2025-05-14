import xml.etree.ElementTree as ET
import os
import csv


def write_as_csv(data,name):
    # Sort the dictionary by values in descending order
    sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=True)

    # File path for the CSV
    csv_file = name + "_data_information.csv"

    # Write to a CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Scenario", "Score"])
        # Write the data
        writer.writerows(sorted_data)

    print(f"Data has been saved to {csv_file}.")



# Directory containing XML files
data_type = "training"
if data_type == "validation":
    directory_path = "/workspace/tg22/remote-pycharm/leaderboard2.0/shrinked_dummy_val/"  # Update this path to your XML files directory
elif data_type == "training":
    directory_path = "/workspace/tg22/remote-pycharm/leaderboard2.0/shrinked_split_trigger_data/"  # Update this path to your XML files directory
assert os.path.exists(directory_path)

keyword_list = list(set([key.split('.')[0].split('_')[1] for key in os.listdir(directory_path)]))


#os.makedirs("/workspace/tg22/remote-pycharm/leaderboard2.0/single_scenario_" + data_type + "_xml/", exist_ok=True)

number_dict = {}

directory = "/workspace/tg22/leaderboard_plant_pdm/a_validation_data/"
directory = "/workspace/tg22/plant_dataset_0/data/PlanT/PlanT_data_lead1/" #a_validation_data/"
#files = os.listdir(directory_path)
assert os.path.exists(directory)
data_files = os.listdir(directory)


for keyword in keyword_list:
    # Keyword to filter XML files
    file_specific_list = []
    for d_f in data_files:
        files = [file for file in os.listdir(directory+d_f) if keyword in file.split('_')]
        file_specific_list += files
        asd = 0
    number_dict.update({keyword:len(file_specific_list)})
    asd = 0

write_as_csv(number_dict, data_type)
print("current path: ",os.getcwd())
asd = 0
