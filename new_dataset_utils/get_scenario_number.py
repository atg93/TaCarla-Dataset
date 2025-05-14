import pickle

def get_scenario_dict(dummy_scenario_list):
    scenario_dict = {}

    for scenario in dummy_scenario_list:
        scenario = scenario.split("_")[0]
        if scenario not in scenario_dict.keys():
            scenario_dict.update({scenario: [1]})
        else:
            scenario_dict[scenario].append(1)
        asd = 0

    return scenario_dict

town_name = "Town12"

with open("scenario_list_"+town_name+".pkl","rb") as pickle_file:
    scenario_list = pickle.load(pickle_file)

with open("scenario_list_planing_"+town_name+".pkl","rb") as pickle_file:
    scenario_list_planing = pickle.load(pickle_file)

scenario_dict = get_scenario_dict(scenario_list_planing)

for scenario_name in scenario_dict.keys():
    print(scenario_name, len(scenario_dict[scenario_name]), "\\\\")

asd = 0