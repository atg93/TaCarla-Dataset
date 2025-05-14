import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional, Union
from .inference_sub_utils import post_process_outputs_single, visualize_single, to_numpy

from .inference_segmentation_and_panoptic import InferenceSegmentationTorch
import heapq
from .inference_sub_utils import PolynomialRegression
from sklearn.linear_model import RANSACRegressor
import warnings
from tairvision.references.segmentation.lane_utils import extract_points_from_mask, create_mask_from_points


class InferenceSegmentationLane(InferenceSegmentationTorch):
    def __init__(self, yaml_name: str, **kwargs):
        super(InferenceSegmentationLane, self).__init__(yaml_name, **kwargs)

        self.probability_threshold = 0.7
        self.filter_threshold = 0.5
        self.background_probability = 0.9
        self.lane_width_radius = 4
        self.class_indices = [1, 2, 3, 4]
        if not isinstance(self.color_palette, List):
            self.color_palette = [self.color_palette]
        self.color_palette = 2 * self.color_palette

    def post_process_outputs(self, outputs):
        output_prob = outputs["out"]
        output_prob = output_prob.detach()
        probabilities = output_prob.softmax(1)

        output_mask = torch.argmax(output_prob, 1)
        output_mask_numpy = to_numpy(output_mask)

        foreground_mask = probabilities[:, 0, ...] < self.background_probability
        foreground_mask_numpy = to_numpy(foreground_mask)

        batch_lane_list = extract_points_from_mask(output_mask, probabilities, self.class_indices,
                                                   selection_probability = self.probability_threshold,
                                                   filter_probability = self.filter_threshold)

        mask_from_points = create_mask_from_points(batch_lane_list, output_mask_numpy, self.lane_width_radius)

        mask_from_points = mask_from_points * foreground_mask_numpy

        output_mask_list = [output_mask_numpy, mask_from_points]

        return output_mask_list, batch_lane_list