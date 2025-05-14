import os


def extract_scenario_names(base_path):
    scenario_names = []

    # Iterate through all folders in the base path
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        # Check if it is a directory
        if os.path.isdir(folder_path):
            score_file_path = os.path.join(folder_path, 'score.txt')

            # Check if the score.txt file exists
            if os.path.isfile(score_file_path):
                with open(score_file_path, 'r') as file:
                    for line in file:
                        line = line.split(' ')
                        scenario_names.append((line[0],line[1]))
                        break  # Stop reading after the first match in the file

    return scenario_names


# Usage example
base_directory = '/workspace/tg22/plant_dataset_0/data/PlanT/PlanT_data_1/wo/need2fix/'  # Replace with your folder path
scenarios = extract_scenario_names(base_directory)

# Print or save the list of scenario names
print("Extracted Scenario Names:")
scenario_dict = {}
for scenario in scenarios:
    if scenario[0] not in scenario_dict.keys():
        scenario_dict.update({scenario[0]:[scenario[1]]})
    else:
        scenario_dict[scenario[0]].append(scenario[1])

print(scenario_dict)
