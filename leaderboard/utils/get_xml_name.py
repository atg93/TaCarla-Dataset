import os
import json


def find_files(path):
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if 'xml' in filename.split('.'):
                files.append(os.path.join(root, filename))
    return files

path = '/home/tg22/remote-pycharm/'+ 'leaderboard2.0/modified_data/'
files = find_files(path)

name_list = []
for fi in files:
    name = fi.split('/')[-1].split('_')[0]
    if not name == 'routes' and not name == 'different' and not name == 'tl' and not name == 'validation.xml':
        name_list.append(name)

name_set = sorted(set(name_list))

name_dict = {}
for index, element in enumerate(name_set):
    name_dict.update({element:index})

file_path = 'name_dict.json'
# Write the dictionary to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(name_dict, json_file)

asd = 0