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
sys.path.append('/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/tairvision_center_line/')
sys.path.append('/cta/users/tgorgulu/git/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/tairvision_center_line/')

#leaderboard.autoagents.traditional_agents_files.perception.
from tairvision.models.bev.lss_mask2former.evaluate_openlaneV2 import EvaluationInterfaceMask2FormerOpenLaneV2
from tairvision.models.bev.lss.utils.visualization import LayoutControl

from leaderboard.autoagents.traditional_agents_files.perception.detection_utils.carla_to_lss import Carla_to_Lss_Converter
import torchvision.transforms as transforms

class Traffic_Lights:
    def __init__(self,config):
        self.device = torch.device(config['device'])

        with open('/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/center_line_config.json','r') as json_file:
            args_dict = json.load(json_file)

        self.evaluation_interface = EvaluationInterfaceMask2FormerOpenLaneV2(**args_dict)

        #self.visualization_module = self._import_visualization_module()(self.cfg)

        layout_control = LayoutControl()
        self.layout_control = layout_control

        asd = 0


    def set_parameters(self,intrinsics, extrinsics, transforms_val, carla_to_lss):
        self.intrinsics, self.extrinsics, self.transforms_val, self.carla_to_lss = intrinsics, extrinsics, transforms_val, carla_to_lss


    def __call__(self, input_data):
        batch = self.carla_to_lss.create_lss_batch(input_data, self.intrinsics, self.extrinsics,
                                                   self.transforms_val)

        resized_image = F.interpolate(batch['images'][:, :, 0, :, :][0], size=(800, 1422), mode='bilinear', align_corners=False)
        #front_view = 255*batch['images'][:, :, 0, :, :][0].resize(256, 704, 3).cpu().numpy()
        #cv2.imwrite("front_view.png",front_view)
        #batch['images'] = batch['images'][:, :, 0, :, :].unsqueeze(2) #torch.zeros(1,1, 1, 3, 256, 704).cuda(resized_image.device)
        batch['front_view_images'] = [[resized_image.squeeze(0)]] #[[torch.zeros(3,800,1422).cuda(resized_image.device)]] #resized_image
        #batch['intrinsics'] = batch['intrinsics'][:, :, 0, :, :].unsqueeze(2)
        #batch['cams_to_lidar'] = batch['cams_to_lidar'][:, :, 0, :, :].unsqueeze(2)
        outputs = self.evaluation_interface.predict_step(batch)

        #self.visualization_module.return_necessary_plots(batch, outputs, None)
        image = self.evaluation_interface.visualization_module.plot_all(batch, outputs, None, None)

        return outputs, image


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