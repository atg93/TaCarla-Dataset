import torch
import torch.utils.data

import cv2
import numpy as np

import copy
import os
import json
import warnings
import torch.nn.functional as F

import sys

current_path = os.getcwd()
sys.path.append(current_path + '/autoagents/traditional_agents_files/perception/tairvision_center_line/')

sys.path.append(current_path +'/leaderboard'+'/autoagents/traditional_agents_files/perception/tairvision_center_line/')


#leaderboard.autoagents.traditional_agents_files.perception.
from tairvision.models.bev.lss_mask2former.evaluate_openlaneV2 import EvaluationInterfaceMask2FormerOpenLaneV2
from tairvision.models.bev.lss.utils.visualization import LayoutControl

from leaderboard.autoagents.traditional_agents_files.perception.detection_utils.carla_to_lss import Carla_to_Lss_Converter
import torchvision.transforms as transforms

from tairvision.models.bev.lss_mask2former.inference import OpenLaneV2InferenceInterface
from PIL import Image

import carla

class Traffic_Lights:
    def __init__(self,config):
        self.device = torch.device(config['device'])

        with open('/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/center_line_config.json','r') as json_file:
            args_dict = json.load(json_file)
        self.tl_inference = OpenLaneV2InferenceInterface(visualize=False)

        layout_control = LayoutControl()
        self.layout_control = layout_control

        self._width = 1845
        self.real_height = 200 #586
        self.real_width = 200#1034

        X_BOUND = [-54.0, 54.0, 0.0542]  # Forward
        Y_BOUND = [-54.0, 54.0, 0.0542]  # Sides
        Z_BOUND = [-10.0, 10.0, 5.0]
        self.project_view = self.calculate_view_matrix(X_BOUND, Y_BOUND, Z_BOUND)

    def set_parameters(self,intrinsics, extrinsics, transforms_val, carla_to_lss):
        self.intrinsics, self.extrinsics, self.transforms_val, self.carla_to_lss = intrinsics, extrinsics, transforms_val, carla_to_lss


    def __call__(self, input_data):
        #batch = self.carla_to_lss.create_lss_batch(input_data, self.intrinsics, self.extrinsics,
        #                                           self.transforms_val)


        #frame = Image.fromarray(frame[:, :, ::-1])
        img = Image.fromarray(input_data['front'][1][:, :, -2::-1])

        frame_dict = {"CAM_FRONT": img}
        intrinsic_dict = {"CAM_FRONT": self._intrinsic["front"]}
        cam_to_lidar_dict = {"CAM_FRONT": self._cam_to_lidar["front"]}

        outputs, image = self.tl_inference.predict(frame_dict, intrinsic_dict, cam_to_lidar_dict)

        tl_bev_image, tl_state, score_list_lcte, pred_list, tl_bbox, box_size = self.map2bev(outputs['metrics'])
        #image = input_data['front']

        return outputs, image, tl_bev_image, tl_state, score_list_lcte, pred_list, tl_bbox, box_size


    def get_tl_color(self, tl_outputs, tl_index):
        line_color = (255, 0, 0)
        tl_state = -1
        #if len(tl_outputs['score_list_lcte'][0][tl_index]) != 0 and np.sum(tl_outputs['score_list_lcte'][0][tl_index] > 1.0) > 0:
        print("tl_outputs['score_list_lcte'][0][tl_index]:",tl_outputs['score_list_lcte'][0],tl_outputs['pred_list'][0])
        score_list_lcte, pred_list =  tl_outputs['score_list_lcte'][0], tl_outputs['pred_list'][0]
        if len(tl_outputs['pred_list'][0]) != 0:
            tl_number = np.argmax(tl_outputs['score_list_lcte'][0][tl_index])
            tl_state = tl_outputs['pred_list'][0][tl_number]['attribute']

            if tl_state == 1:
                line_color = (0, 0, 255)
            elif tl_state == 2:
                line_color = (0, 255, 0)
            elif tl_state == 3:
                line_color = (0, 255, 255)

        return line_color, tl_state, score_list_lcte, pred_list


    def map2bev(self, tl_outputs):
        centerlines = tl_outputs['lane_centerline_list_pred']  # compass
        bev_image = np.zeros((586,1034)).astype(np.uint8)

        if len(tl_outputs['pred_list'][0]) != 0:
            tl_outputs['pred_list'], tl_outputs['lane_centerline_list_pred'], tl_outputs[
                'score_list_lcte'], np.argmax(tl_outputs['score_list_lcte'][0])
        color = carla.Color(r=0, g=0, b=255, a=0)
        thickness = 0.5
        life_time = 0.2
        stop_label = False

        tl_lights_masks = np.zeros([self._width, self._width], dtype=np.uint8)
        mid_point = int(self._width / 2)
        tl_lights_masks = self.draw_from_center(tl_lights_masks, np.array([mid_point, mid_point]), 2, 2)
        tl_bbox = []
        box_size = (0,0,0)
        for tl_index, cl in enumerate(centerlines[0]):
            line_thickness = 2
            stop_masks = np.zeros([self._width, self._width], dtype=np.uint8)
            line_color, tl_state, score_list_lcte, pred_list = self.get_tl_color(tl_outputs, tl_index)


            if tl_state != -1:
                tl_box_mid_point = np.array([cl['points'][:, 0].mean(), cl['points'][:, 1].mean(), cl['points'][:, 2].mean()])
                tl_bbox = tl_box_mid_point
                new_tl_box_mid_point = tl_box_mid_point + mid_point
                box_size = abs((cl['points'][0] - cl['points'][-1]).astype(np.int)) + 2
                tl_lights_masks = self.draw_from_center(tl_lights_masks, new_tl_box_mid_point,box_size[0],box_size[1])


            tl_lights_masks = self.crop_array_center(tl_lights_masks, self.real_height,
                                                     self.real_width)  # cv2.resize(tl_lights_masks, (self.real_width, self.real_height))#


            stop_masks = self.crop_array_center(stop_masks, self.real_height,
                                                self.real_width)  # cv2.resize(tl_lights_masks, (self.real_width, self.real_height))#


        return tl_lights_masks, tl_state, score_list_lcte, pred_list, tl_bbox, box_size

    def draw_from_center(self,image, center,  width=5,height=2):
        # Create a blank 300x300 black image

        center_x, center_y = center[0], center[1]
        half_width = width // 2
        half_height = height // 2

        # Calculate top-left and bottom-right points based on the center point
        top_left = (int(center_x - half_width), int(center_y - half_height))
        bottom_right = (int(center_x + half_width), int(center_y + half_height))

        # Draw the rectangle
        cv2.rectangle(image, top_left, bottom_right, (255), thickness=-1)
        cv2.imwrite("image_tl_lights_masks_1.png", image)

        return image

    def crop_array_center(self, original_array, new_height, new_width):
        original_height, original_width = original_array.shape

        # Calculate the starting points for the crop
        start_row = (original_height - new_height) // 2
        start_col = (original_width - new_width) // 2

        # Calculate the ending points for the crop
        end_row = start_row + new_height
        end_col = start_col + new_width

        # Crop the array
        cropped_array = original_array[start_row:end_row, start_col:end_col]

        return cropped_array

    def calculate_view_matrix(self, X_BOUND, Y_BOUND, Z_BOUND):
        import sys
        sys.path.append(
            '/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/tairvision_center_line/')

        import torch
        from tairvision.models.bev.common.utils.geometry import calculate_birds_eye_view_parameters
        from tairvision.datasets.nuscenes import get_view_matrix

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            X_BOUND, Y_BOUND, Z_BOUND
        )

        bev_resolution, bev_start_position, bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        lidar_to_view = get_view_matrix(
            bev_dimension,
            bev_resolution,
            bev_start_position=bev_start_position,
            ego_pos='center'
        )

        view = lidar_to_view[None, None, None]
        view_tensor = torch.tensor(view, dtype=torch.float32)

        return view_tensor

    def get_bev_image(self, tl_bev_image, tl_state, size_x, size_y):#bev_image.shape[1], bev_image.shape[0]
        tl_bev_image = cv2.resize(tl_bev_image, (size_x, size_y))

        new_tl_bev_image = cv2.cvtColor(tl_bev_image, cv2.COLOR_GRAY2RGB)  # new_tl_bev_image[tl_bev_image] = (255,255,255)

        if tl_state == 1:
            new_tl_bev_image[tl_bev_image>0] = (255,0,0)
        elif tl_state == 2:
            new_tl_bev_image[tl_bev_image>0] = (0, 255, 0)
        elif tl_state == 3:
            new_tl_bev_image[tl_bev_image>0] = (255, 255, 0)
        else:
            new_tl_bev_image[tl_bev_image>0] = (0, 0, 255)

        return new_tl_bev_image



    def set_settings(self, intrinsic, cam_to_lidar):
        self._intrinsic = intrinsic
        self._cam_to_lidar = cam_to_lidar


    def draw_detected_output(self, images, output):
        res_img = np.array(images.reshape(396, 704, 3).numpy())
        box_above_score =[]

        scores = output[0]['scores'].cpu()
        boxes = output[0]['boxes'].cpu()
        labels = output[0]['labels'].clamp_(1).cpu().numpy()
        for index, bb in enumerate(boxes):
            if scores[index].item() > self.threshold:
                print("scores[index].item() > self.threshold:",scores[index].item(), self.threshold)
                box_above_score.append(bb)
                res_img = self.draw_boxes(res_img, bb[0], bb[1], bb[2], bb[3], labels[0])

        return res_img.astype(np.uint8), box_above_score

    def deneme_traffic_lights(self):
        for index, data in enumerate(self.data_loader_test):
            images, targets = data
            output, _ = self.__call__(images)

            pre_img = np.array(images[0].reshape(396, 704, 3).numpy())

            boxes = output[0]['boxes'].cpu()
            scores = output[0]['scores'].cpu()
            # labels = output[0]['labels'].cpu().numpy()
            labels = output[0]['labels'].clamp_(1).cpu().numpy()
            # speed = output[0]['speed'].clamp_(1).cpu().numpy()-1
            masks = output[0]['masks'].cpu() if 'masks' in output[0].keys() else None

            threshold_mask = scores > float(self.threshold)
            boxes = boxes[threshold_mask.numpy()]
            labels = labels[threshold_mask.numpy()]
            if masks is not None:
                masks = masks[threshold_mask.numpy()]

            boxes_gt = torch.tensor(targets[0]['boxes'])
            masks_gt = targets[0]['masks'] if 'masks' in targets[0].keys() else None
            labels_gt = targets[0]['labels']
            # speed_gt = targets[0]['speed']
            print("*" * 50, "labels:", labels, "labels_gt:",labels_gt)

            colors = np.array([get_label_color(label) for label in labels])
            colors_gt = np.array([get_label_color(label) for label in labels_gt])

            if masks is not None:
                res_masks = masks[:, 0, :, :]
            else:
                res_masks = masks
                masks_gt = None

            if len(boxes) != 0:
                res_img = copy.deepcopy(pre_img)
                for bb in boxes:
                    res_img = self.draw_boxes(res_img, bb[0], bb[1], bb[2], bb[3], labels[0])
            else:
                res_img = copy.deepcopy(pre_img)

            if len(boxes_gt) != 0:
                gt_img = copy.deepcopy(pre_img)
                for bb in boxes_gt:
                    gt_img = self.draw_boxes(gt_img, bb[0], bb[1], bb[2], bb[3], labels_gt[0])
            else:
                gt_img = copy.deepcopy(pre_img)

            vis = np.concatenate((res_img, gt_img), axis=1)
            cv2.imwrite("fcos_resim/vis" + str(index) + ".png", vis)
            print(os.getcwd())


        cv2.destroyAllWindows()
        assert False


    def draw_boxes(self, img, x_min, y_min, x_max, y_max, color):
        if color == 3:
            color = (0, 0, 255)
        elif color == 2:
            color = (0, 255, 255)
        elif color == 1:
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)

        cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), color, 1)
        cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), color, 1)
        cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), color, 1)
        cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), color, 1)

        return img
