from .inference_segmentation_and_panoptic import InferenceSegmentationTorch
import numpy as np
import cv2
from .inference_sub_utils import post_process_outputs_single, to_numpy
import torch


class InferenceSegmentationLaneFit(InferenceSegmentationTorch):
    def __init__(self, yaml_name: str, **kwargs):
        super(InferenceSegmentationLaneFit, self).__init__(yaml_name, **kwargs)
        self.color_palette = [self.color_palette, self.color_palette]
        self.lane_fit_position_scaling_factor = self.dataset_kwargs['lane_fit_position_scaling_factor']

    def post_process_outputs(self, outputs):
        output_mask_list = []

        output_prob = outputs["out"]
        output_mask_segm = post_process_outputs_single(output_prob)
        outputs["mask"] = output_mask_segm

        lane_exist_prob = torch.nn.functional.sigmoid(outputs["lane_exist"])
        outputs["lane_exist"] = lane_exist_prob
        mask_lane = self._create_mask_from_lane_params(outputs)

        output_mask_list.append(mask_lane)
        output_mask_list.append(output_mask_segm)

        return output_mask_list

    def _create_mask_from_lane_params(self, output):
        params = output["lane_params"]
        lane_exists = to_numpy(output["lane_exist"])
        params = to_numpy(params)
        borders = to_numpy(output["borders"])

        mask_list = []
        for (param_batch, lane_exist, border) in zip(params, lane_exists, borders):
            mask_lane = np.zeros_like(to_numpy(output["mask"]))[0]
            height = width = self.lane_fit_position_scaling_factor

            for index, param in enumerate(param_batch):
                border_per_lane = border[index]
                if border_per_lane[0] > border_per_lane[1]:
                    continue

                if lane_exist[index] < 0.5:
                    continue

                dummy_mask = np.zeros_like(mask_lane)
                dummy_image = dummy_mask[:, :, None]
                dummy_image = \
                    np.concatenate([dummy_image, dummy_image, dummy_image], 2).astype(np.uint8)

                poly_eqn = np.poly1d(param)
                unique_y_range = np.arange(border_per_lane[0] * height, border_per_lane[1] * height)
                unique_y_range_normalized = unique_y_range / height
                predicted_normalized = poly_eqn(unique_y_range_normalized)
                predicted = predicted_normalized * width

                for y, out in zip(unique_y_range, predicted):
                    x_int = int(out)
                    y = int(y)
                    cv2.circle(dummy_image, (x_int, y),
                               4, (255, 255, 255), -1)

                dummy_image = np.mean(dummy_image, 2)
                mask_lane[dummy_image == 255] = index + 1
            mask_list.append(mask_lane[None, :])
        mask_lane = np.vstack(mask_list)
        return mask_lane