import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import xml


def get_all_type_name(xml_file_path):
    """
    Removes scenarios from an XML file where the scenario type is not 'ParkingExit'
    and saves the modified XML to a new file.

    Parameters:
    - xml_file_path: The path to the input XML file.
    - output_file_path: The path where the modified XML file should be saved.
    """
    # Load the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    type_set = set()

    # Iterate through all routes in the document
    for route in root.findall('.//route'):
        # Iterate through all scenarios under each route
        scenarios = route.findall('scenarios')[0]
        # Iterate through all scenarios under each route
        type_exists = False
        trigger_point_list = []
        for scenario in scenarios.findall('scenario'):
            type_set.update({scenario.get('name')})

    return type_set

def parse_scenario(scenario):
    scenario_dict = {'name': scenario.attrib['name'], 'type': scenario.attrib['type']}
    for child in scenario:
        if 'value' in child.attrib:
            scenario_dict[child.tag] = child.attrib['value']
        else:
            scenario_dict[child.tag] = {k: v for k, v in child.attrib.items()}
    return scenario_dict

def create_route_xml(routes, route_id, town, weathers, waypoints, scenario):

    route = ET.SubElement(routes, "route", id=str(route_id), town=town)

    # Adding weather elements
    weathers_elem = ET.SubElement(route, "weathers")
    for weather in weathers:
        # Convert all values to strings
        weather_attributes = {k: str(v) for k, v in weather.items()}
        ET.SubElement(weathers_elem, "weather", **weather_attributes)

    # Adding waypoint elements
    waypoints_elem = ET.SubElement(route, "waypoints")
    for waypoint in waypoints:
        # Convert all values to strings
        waypoint_attributes = {k: str(v) for k, v in waypoint.items()}
        ET.SubElement(waypoints_elem, "position", **waypoint_attributes)

    # Adding scenario elements
    scenarios_elem = ET.SubElement(route, "scenarios")
    scenario_dict = parse_scenario(scenario)
    scenario_elem = ET.SubElement(scenarios_elem, "scenario", name=scenario.attrib["name"], type=scenario.attrib["type"])
    trigger_point_attributes = {k: str(v) for k, v in scenario_dict["trigger_point"].items()}
    ET.SubElement(scenario_elem, "trigger_point", **trigger_point_attributes)

    if type(scenario.find('frequency')) != type(None) and len(scenario.find('frequency').attrib.keys()) > 0:
        frequency = scenario.find('frequency')
        ET.SubElement(scenario_elem, "frequency", **frequency.attrib)

    if type(scenario.find('start_actor_flow')) != type(None) and len(scenario.find('start_actor_flow').attrib.keys()) > 0:
        frequency = scenario.find('start_actor_flow')
        ET.SubElement(scenario_elem, "start_actor_flow", **frequency.attrib)

    if type(scenario.find('end_actor_flow')) != type(None) and len(scenario.find('end_actor_flow').attrib.keys()) > 0:
        frequency = scenario.find('end_actor_flow')
        ET.SubElement(scenario_elem, "end_actor_flow", **frequency.attrib)

    if type(scenario.find('source_dist_interval')) != type(None) and len(scenario.find('source_dist_interval').attrib.keys()) > 0:
        frequency = scenario.find('source_dist_interval')
        ET.SubElement(scenario_elem, "source_dist_interval", **frequency.attrib)

    if type(scenario.find('other_actor_location')) != type(None) and len(scenario.find('other_actor_location').attrib.keys()) > 0:
        frequency = scenario.find('other_actor_location')
        ET.SubElement(scenario_elem, "other_actor_location", **frequency.attrib)

    for key, value in scenario_dict.items():
        if key not in ["name", "type", "trigger_point", "frequency", "start_actor_flow", "end_actor_flow",
                       "source_dist_interval", "other_actor_location"]:
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    subelement = ET.SubElement(scenario_elem, key)
                    subelement.set(subkey, str(subvalue))
            else:
                ET.SubElement(scenario_elem, key, value=str(value))


def get_trigger_point(scenario):
    trigger_point = scenario.findall('trigger_point')
    return trigger_point[0].attrib


def remove_scenarios_not_parking_exit(xml_file_path, output_file_path, key_type="ParkingExit"):
    """
    Removes scenarios from an XML file where the scenario type is not 'ParkingExit'
    and saves the modified XML to a new file.

    Parameters:
    - xml_file_path: The path to the input XML file.
    - output_file_path: The path where the modified XML file should be saved.
    """
    # Load the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    routes = ET.Element("routes")
    # Iterate through all routes in the document
    for route in root.findall('.//route'):
        # Iterate through all scenarios under each route
        scenarios = route.findall('scenarios')[0]
        weathers = route.findall('weathers')[0]
        id = route.attrib['id']
        # Iterate through all scenarios under each route
        type_exists = False
        for scenario_index, scenario in enumerate(scenarios.findall('scenario')):
            # If the scenario type is not "ParkingExit", remove it
            if scenario.get('name') == key_type:
                new_id = id+str(scenario_index)
                current_trigger_point = get_trigger_point(scenario)
                try:
                    next_trigger_point = get_trigger_point(scenarios.findall('scenario')[scenario_index+1])
                except:
                    next_trigger_point = route.findall('waypoints')[0].findall('position')[-1]

                waypoints = [
                    current_trigger_point,
                    next_trigger_point,
                    # Add more waypoints as needed
                ]
                create_route_xml(routes=routes, route_id=new_id, town='Town13', weathers=weathers, waypoints=waypoints, scenario=scenario)
                asd = 0

        # Convert the ElementTree to a string
        xml_str = ET.tostring(routes, encoding='unicode')

        # Use minidom to pretty-print the XML
        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml_str = dom.toprettyxml(indent="  ")

    # Create the tree and write to file
    """tree = ET.ElementTree(routes)
    tree.write(output_file_path)"""
    # Write the pretty-printed XML to a file
    with open(output_file_path, "w") as f:
        f.write(pretty_xml_str)


def find_the_element(lower_bound):
    # Calculate distances from the trigger_point to each position
    distances = [np.linalg.norm(pos - trigger_point) for pos in list(lower_bound)]
    # Sort the distances and get the indexes of the two closest positions
    sorted_indexes = np.argsort(distances)
    element = lower_bound[sorted_indexes[:1]]

    return element

def find_ind(positions, trigger_point):
    mask = positions > trigger_point
    upper_bound = positions[mask]
    lower_bound = positions[~mask]

    upper_element = find_the_element(upper_bound)
    lower_element = find_the_element(lower_bound)
    upper_index = np.where(upper_element == positions)
    lower_index = np.where(lower_element == positions)

    return upper_index[0][0], lower_index[0][0]




positions = np.array([
    np.array([516.3]),
    np.array([355.1]),
    np.array([285.9])
])

trigger_point = np.array([427.2])
#trigger_point = np.array([427.2, 6086.8, 359])

upper_ind, lower_ind = find_ind(positions, trigger_point)
print(upper_ind)
print(lower_ind)



key_type = "ParkingExit"
# Example usage
#file_path = Path('/home/tg22/remote-pycharm/leaderboard2.0/modified_data/routes_training.xml')
file_path = Path('/home/tg22/remote-pycharm/leaderboard2.0/data/routes_validation.xml')
all_types = get_all_type_name(file_path)
for key_type in list(all_types):
    print("key_type:", key_type)
    #if not "CrossingBicycleFlow" in key_type.split('_'):
    #    continue
    output_file_path = '/home/tg22/remote-pycharm/leaderboard2.0/val_data_trigger_point/' + key_type +'.xml'
    output_file_path = Path(output_file_path)
    assert file_path.exists()
    xml_file_path = file_path #'modified_data/routes_training.xml'
    #output_file_path = 'path_to_your_output_file.xml'
    remove_scenarios_not_parking_exit(xml_file_path, output_file_path, key_type=key_type)
print("len(list(all_types)): ",len(list(all_types)))
