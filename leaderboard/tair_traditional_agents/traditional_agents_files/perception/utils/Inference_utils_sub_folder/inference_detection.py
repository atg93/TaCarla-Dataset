import cv2
import numpy as np
import torch
from tairvision.models import detection as det
from typing import Tuple, List, Optional, Union

import os

import matplotlib
from .instance_class import Instance
from .inference_sub_utils import compute_overlap, draw_keypoints
from .inference_sub_utils import to_numpy

try:
    from apex import amp
except:
    print("apex is not imported")

from .inference_main import Inference


class InferenceDetection(Inference):
    def __init__(self, yaml_name: str, **kwargs):
        super(InferenceDetection, self).__init__(yaml_name, **kwargs)
        self.output_names = ['cls_logits', 'bbox_regression']
        if self.dataset_name == "widerface":
            self.number_of_classes = 2
        elif self.dataset_name == "coco":
            self.number_of_classes = 91

        self.torch_model = self.load_torch_model()
        anchor = self.torch_model.anchor
        self.number_of_boxes = len(anchor)
        self.num_anchors_per_level = self.torch_model.number_of_anchors_per_level
        anchors = 1 * [anchor.to(self.device)]
        self.split_anchors = [list(a.split(self.num_anchors_per_level)) for a in anchors]

        self.model_path = self.path_for_model_load()
        self.load_model()
        self.color_chart = self.create_color_space2()

        self.edges = [
            (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
            (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
            (12, 14), (14, 16), (5, 6)
        ]

    def load_torch_model(self):
        model_dict = det.__dict__
        if self.load_coco:
            model = model_dict[self.model_name](size=self.size,
                                                pretrained=self.load_coco,
                                                deployment_mode=True)
        else:
            model = model_dict[self.model_name](num_classes=self.number_of_classes,
                                                size=self.size,
                                                deployment_mode=True)
        return model

    def visualize(self, output, frame_resized):
        boxes = output[0]['boxes'].cpu().numpy().astype(int)
        labels = output[0]['labels'].cpu().numpy()
        scores = output[0]['scores'].cpu().numpy()
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.4:
                cv2.rectangle(frame_resized, (box[0], box[1]), (box[2], box[3]),
                              (int(self.color_chart[label][0]),
                               int(self.color_chart[label][1]),
                               int(self.color_chart[label][2])), 2)

        if "keypoints" in output[0].keys():
            frame_resized = draw_keypoints(output, frame_resized)
        return frame_resized

    def create_color_space(self):
        vector = np.linspace(0, 255, int(self.number_of_classes ** (1 / 3)) + 1).astype(int)
        red, green, blue = np.meshgrid(vector, vector, vector)
        red = red.reshape(-1, 1)
        green = green.reshape(-1, 1)
        blue = blue.reshape(-1, 1)
        color_chart = np.concatenate([blue, green, red], 1)
        color_chart = color_chart.astype(int)
        return color_chart

    def create_color_space2(self):
        color_chart = []
        for class_id in range(self.number_of_classes):
            # get different colors for the edges
            rgb = matplotlib.colors.hsv_to_rgb([
                class_id / self.number_of_classes, 1.0, 1.0
            ])
            rgb = rgb * 255
            color_chart.append(rgb)
        return color_chart


class InferenceDetectionTorch(InferenceDetection):
    def __init__(self, yaml_name: str, **kwargs):
        super(InferenceDetectionTorch, self).__init__(yaml_name, **kwargs)

    def load_model(self):
        if not self.load_coco:
            weights = torch.load(self.model_path, map_location="cpu")
            self.torch_model.load_state_dict(weights['model_state_dict'])
        self.torch_model.to(self.device)
        self.torch_model.eval()

        if self.mixed_precision_training:
            amp.initialize(self.torch_model, opt_level="O3", keep_batchnorm_fp32=True)

    def path_for_model_load(self):
        torch_file = os.path.join(self.weights_main_path, self.yaml_name, "best_eval.pth")
        return torch_file

    def model_feedforward(self, image):
        image_tensor = torch.tensor(image)
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.torch_model(image_tensor)

        return outputs

    def post_process_outputs(self, outputs):
        if "retina" in self.yaml_name:
            split_head_outputs = {}
            for k in outputs:
                split_head_outputs[k] = list(
                    torch.tensor(outputs[k], device=self.device).split(
                        self.num_anchors_per_level, dim=1))

            outputs = self.torch_model.postprocess_detections(split_head_outputs,
                                                              self.split_anchors,
                                                              1 * [self.size])
        return outputs


class InferenceTrackedDetectionTorch(InferenceDetectionTorch):
    def __init__(self, yaml_name: str, **kwargs):
        super(InferenceTrackedDetectionTorch, self).__init__(yaml_name, **kwargs)

        self.maximum_number_of_instances: int = 99
        self.instanceIDList: List[int] = []

        self.old_instances_list: List[Instance] = []
        self.current_instances_list: List[Instance] = []

        self.objectness_threshold = 0.6

        self.iou_overlap_threshold: float = 0.5

    def track_detections(self, outputs):
        objectified_instances_list = self.objectify_instances(outputs)
        if len(self.old_instances_list) == 0:
            self.current_instances_list = objectified_instances_list
            return

        if len(objectified_instances_list) == 0:
            self.current_instances_list = []
            return

        current_boxes_array = np.concatenate(
            [current_instance.box[None, :] for current_instance in objectified_instances_list], 0)
        old_boxes_array = np.concatenate(
            [old_instance.box[None, :] for old_instance in self.old_instances_list], 0)
        iou_overlap_matrix = compute_overlap(current_boxes_array, old_boxes_array)

        iou_vector = np.max(iou_overlap_matrix, axis=0)
        old_instances_list_temp = []
        for i in range(len(iou_vector)):
            if iou_vector[i] < self.iou_overlap_threshold:
                continue
            else:
                current_instance_maximum_overlapping_index = np.argmax(iou_overlap_matrix[:, i])
                old_instance_maximum_overlapping_index = np.argmax(
                    iou_overlap_matrix[current_instance_maximum_overlapping_index, :])
                if old_instance_maximum_overlapping_index == i:
                    current_instance: Instance = objectified_instances_list[current_instance_maximum_overlapping_index]
                    old_instance: Instance = self.old_instances_list[i]
                    current_instance.update(old_instance.create_instance_dict())

        self.current_instances_list = objectified_instances_list

    def _pick_an_empty_id(self) -> int:
        id = 0
        while id in self.instanceIDList:
            id += 1
        self.instanceIDList.append(id)
        return id

    def objectify_instances(self, outputs) -> List[Instance]:
        boxes = outputs[0]['boxes']
        labels = outputs[0]['labels']
        scores = outputs[0]['scores']
        objectified_instances_list = []
        self.instanceIDList = [instance.instanceID for instance in self.old_instances_list]
        for index, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            if score < self.objectness_threshold:
                continue
            if 'keypoints' in outputs[0].keys():
                keypoint = to_numpy(outputs[0]['keypoints'][index])
            else:
                keypoint = None

            instanceID = self._pick_an_empty_id()
            detected_box = to_numpy(box)
            instance_dict = {'box': detected_box,
                             'classID': label.item(),
                             'score': score.item(),
                             'instanceID': instanceID,
                             'keypoint': keypoint}

            new_instance = Instance(instance_dict)
            objectified_instances_list.append(new_instance)
        return objectified_instances_list

    def post_process_in_main_loop(self, outputs):
        self.current_instances_list = []
        self.track_detections(outputs)
        self.old_instances_list = self.current_instances_list

    def visualize(self, output, frame_resized):
        for object_instance in self.current_instances_list:
            if object_instance.score > self.objectness_threshold:
                object_instance.visualize(frame_resized)

        return frame_resized

