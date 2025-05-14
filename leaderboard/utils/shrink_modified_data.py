import pickle
import xml.etree.ElementTree as ET
import re
import os


def remove_duplicate_routes(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    seen_positions = set()
    routes_to_remove = []

    for route in root.findall('route'):
        positions = tuple((pos.get('x'), pos.get('y'), pos.get('z')) for pos in route.findall('.//position'))
        if positions in seen_positions:
            routes_to_remove.append(route)
        else:
            seen_positions.add(positions)

    for route in routes_to_remove:
        root.remove(route)

    return tree


# Specify the path to your XML file
directory = '/home/tg22/remote-pycharm/leaderboard2.0/dummy_val'
new_directory = '/home/tg22/remote-pycharm/leaderboard2.0/shrinked_dummy_val'


assert os.path.exists(new_directory)
assert os.path.exists(directory)

files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

for file in files:
    path = directory+'/'+file
    new_tree = remove_duplicate_routes(path)
    new_tree.write(new_directory+'/shrinked_'+file)