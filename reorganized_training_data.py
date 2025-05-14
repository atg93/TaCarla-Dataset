import os
import shutil
import json

def move_scenario_folders(base_folder, new_folder1, new_folder2):
    # Create the destination folders if they don't exist
    os.makedirs(new_folder1, exist_ok=True)
    os.makedirs(new_folder2, exist_ok=True)

    for root, dirs, files in os.walk(base_folder):
        # Check if "detected_boxes" exists in current path
        if "detected_tl_boxes" in dirs:
            planning_scenario_folder = os.path.abspath(root)
            detected_boxes_path = os.path.join(planning_scenario_folder, "detected_tl_boxes")

            # Check for JSON files in the detected_boxes folder
            contains_bev_tl = False
            number_of_bev = 0
            for file in os.listdir(detected_boxes_path):
                if file.endswith(".json"):
                    json_file_path = os.path.join(detected_boxes_path, file)
                    with open(json_file_path, "r") as json_file:
                        try:
                            data = json.load(json_file)
                            if "bev_light" in str(data):  # Check for 'bev_tl' in the file content
                                number_of_bev += 1

                        except json.JSONDecodeError:
                            print(f"Error decoding JSON file: {json_file_path}")


            if number_of_bev > 3:
                contains_bev_tl = True

            # If any JSON file contains 'bev_tl', move the folder to the second new folder
            if contains_bev_tl:
                #shutil.move(os.path.join(new_folder1, os.path.basename(planning_scenario_folder)), new_folder2)
                shutil.move(planning_scenario_folder, new_folder2)
                print(f"Moved to {new_folder2}: {planning_scenario_folder}")
            else:
                # Move the planning scenario folder to the first new folder
                shutil.move(planning_scenario_folder, new_folder1)
                print(f"Moved to {new_folder1}: {planning_scenario_folder}")

# Define your paths
#base_folder = "/workspace/tg22/plant_dataset_0/data/PlanT/PlanT_data_lead2/Routes_Town12_Scenarioı0_Seed2010_Leaderboard2/"
base_folder = "/workspace/tg22/plant_dataset_0/data/PlanT/PlanT_data_1/Routes_Town12_Scenario0_Seed2010/"
new_folder1 = "/workspace/tg22/plant_dataset_0/data/PlanT/PlanT_data_1/filtered_folder_1"
new_folder2 = "/workspace/tg22/plant_dataset_0/data/PlanT/PlanT_data_1/filtered_folder_2"

listdir = os.listdir(base_folder)
for folder in listdir:
    # Run the function
    move_scenario_folders(base_folder+folder, new_folder1, new_folder2)
