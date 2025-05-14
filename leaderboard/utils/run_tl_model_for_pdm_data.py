
import os
from pathlib import Path
import json
import sys
import numpy as np

from PIL import Image

current_path = '/home/tg22/remote-pycharm/leaderboard2.0/'#os.getcwd()
sys.path.append(current_path + '/autoagents/traditional_agents_files/perception/tairvision_center_line/')

sys.path.append(current_path +'/leaderboard'+'/autoagents/traditional_agents_files/perception/tairvision_center_line/')

from leaderboard.autoagents.traditional_agents_files.perception.traffic_lights import Traffic_Lights
from leaderboard.autoagents.traditional_agents_files.perception.object_detection import Object_Detection

import cv2

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

class Run_Tl_Model:

    def __init__(self,data_path='/workspace/tg22/leaderboard_plant_pdm_final_data/',
                 destination_path='/workspace/tg22/leaderboard_plant_pdm/final_data/'):
        data_path = '/workspace/tg22/plant_dataset_0/data/PlanT/PlanT_data_1'
        assert os.path.exists(data_path)

        self.existing_route_dict = {}

        self.files = find_files(data_path)

        try:
            with open('autoagents/traditional_agents_files/perception/tl_config.json','r') as json_file:
                config = json.load(json_file)
        except:
            with open(current_path + 'leaderboard/autoagents/traditional_agents_files/perception/tl_config.json','r') as json_file:
                config = json.load(json_file)

        config.update({'device':2})

        detection_config = {'detection_threshold': 0.4,
                            'perception_checkpoint': '/home/tg22/lss_models/fine_tuned_epoch=19-step=30160.ckpt',
                            'monocular_perception': False, 'lane_detection': False,
                            'lane_yaml_path': '/home/ad22/leaderboard_carla/tairvision_master/settings/Deeplab/deeplabv3_resnet18_openlane_culane_tusimple_curvelanes_llamas_once_klane_carlane_lanetrainer_640x360_thick_openlane_val_excluded.yml'}
        self.obj_detection = Object_Detection(detection_config)

        object_detection_sensor, intrinsics, extrinsics = self.obj_detection.sensors()
        with open('object_detection_sensor.json', 'w') as json_file:
            json.dump(object_detection_sensor, json_file, indent=len(object_detection_sensor))

        new_intrinsics_dict = {}
        for _key in intrinsics.keys():
            new_intrinsics_dict.update({_key: intrinsics[_key].cpu().numpy().tolist()})

        new_extrinsics_dict = {}
        for _key in extrinsics.keys():
            new_extrinsics_dict.update({_key: extrinsics[_key].cpu().numpy().tolist()})

        self.tl_model = Traffic_Lights(config)

        if type(self.tl_model) != type(None):
            self.tl_model.set_settings(new_intrinsics_dict, new_extrinsics_dict)

        asd = 0







    def __call__(self):
        for file in self.files:
            data = read_txt_file(file)
            self.images = []


            if True: #float(data[0][4]) > 70.0 and data[0][0] + '_' + data[0][1] not in self.existing_route_dict.keys():
                self.existing_route_dict.update({data[0][0] + '_' + data[0][1]: data[0][4]})
                copy_file_path = '/'.join(file.split('/')[:-1]) + '/detection/front/'
                boxes_file_path = '/'.join(file.split('/')[:-1]) + '/boxes/'
                directory_path = '/'.join(file.split('/')[:-1]) + '/detected_tl_boxes/'
                try:
                    os.makedirs(directory_path, exist_ok=False)
                except:
                    continue

                print("copy_file_path: ", copy_file_path)
                for filename in os.listdir(copy_file_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        # Construct the full path to the image file
                        file_path = os.path.join(filename)
                        number = str(file_path.split('_')[1]).zfill(4)
                        boxes_file_name = number + '.json'
                        with open(boxes_file_path+boxes_file_name, 'r') as json_file:
                            original_boxes_results = json.load(json_file)
                        # Open the image file
                        img = Image.open(copy_file_path + file_path)
                        # Append the image to the list
                        self.images.append({'front': (int(file_path.split('_')[1]), np.array(img))})
                        tl_output, tl_image, tl_bev_image, tl_state, score_list_lcte, pred_list, tl_bbox, box_size = \
                            self.tl_model({'front': (int(file_path.split('_')[1]), np.array(img))})

                        twod_bbox = None
                        print("score_list_lcte:",score_list_lcte.flatten())

                        if len(score_list_lcte.flatten()) != 0 and max(score_list_lcte.flatten()) > 0.5:
                            twod_bbox = tl_output['metrics']['pred_list']
                            bev_bbox = tl_output['metrics']['lane_centerline_list_pred'][0][0]['points']

                            bev_position, bev_extent = np.mean(bev_bbox, 0), np.abs(bev_bbox[0] - bev_bbox[1])
                            distance = np.linalg.norm(bev_position)
                            bev_result = {
                                "class": "bev_light",
                                "extent": [float(bev_extent[2]), float(bev_extent[0]), float(bev_extent[1])],  # TODO
                                "position": [float(bev_position[0]), float(bev_position[1]), float(bev_position[2])],
                                "yaw": 0,
                                "distance": float(distance),
                                "state": float(twod_bbox[0][0]['attribute']), #1 red, 2 green, 3 yellow
                                "id": int(0),
                            }
                            original_boxes_results.append(bev_result)

                            asd = 0
                            twod_position = np.mean(twod_bbox[0][0]['points'],0)
                            twod_extent = np.abs(twod_bbox[0][0]['points'][0] - twod_bbox[0][0]['points'][1])
                            twod_distance = np.linalg.norm(twod_extent)

                            two_result = {
                                "class": "twod_light",
                                "extent": [float(twod_extent[0]), float(twod_extent[1]), 0.0],  # TODO
                                "position": [float(twod_position[0]), float(twod_position[1]), 0.0],
                                "yaw": 0,
                                "distance": float(twod_distance),
                                "state": float(twod_bbox[0][0]['attribute']),  # 1 red, 2 green, 3 yellow
                                "id": int(0),
                            }

                            original_boxes_results.append(two_result)



                            asd = 0
                        try:
                            with open(directory_path+boxes_file_name, 'w') as f:
                                json.dump(original_boxes_results, f, indent=4)
                        except:
                            asd = 0

                        print("twod_bbox:",twod_bbox)
                        cv2.imwrite('tl_image.png', tl_image)
                        cv2.imwrite('tl_bev_image.png', tl_bev_image)

                        asd = 0

                asd = 0

if __name__ == '__main__':
    tl_model = Run_Tl_Model()
    tl_model()