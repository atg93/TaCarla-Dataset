from typing import Tuple, List, Optional, Union, Dict
import numpy as np
import cv2
from tairvision.references.segmentation.demo import to_numpy
import torch
import heapq
from matplotlib import colors
import warnings


def create_mask_from_points(batch_lane_list: List[Dict], output_mask, lane_width_radius_for_metric: int):
    mask_list = []
    for lane_dict in batch_lane_list:
        mask_lane = np.zeros_like(output_mask)[0]

        dummy_mask = np.zeros_like(mask_lane)
        dummy_image = dummy_mask[:, :, None]
        dummy_image = \
            np.concatenate([dummy_image, dummy_image, dummy_image], 2).astype(np.uint8)

        for class_index, numpy_out in lane_dict.items():

            x_coords = numpy_out[:, 0]
            y_coords = numpy_out[:, 1]

            # fill_mask_with_circular_points(mask_lane=mask_lane, x_coords=x_coords, y_coords=y_coords,
            #                                lane_width_radius=lane_width_radius_for_metric, class_label=class_index)

            lane_array = numpy_out[::1, :].round().astype('int').tolist()
            dummy_image = draw_line_on_image(image=dummy_image, points=lane_array, color=(class_index, 0, 0),
                               thickness=lane_width_radius_for_metric)

        mask_lane = dummy_image[:, :, 0]
        mask_list.append(mask_lane[None, :])
    mask_lane = np.vstack(mask_list)

    return mask_lane


def fill_mask_with_circular_points(mask_lane, x_coords, y_coords, lane_width_radius, class_label):
    dummy_mask = np.zeros_like(mask_lane)
    dummy_image = dummy_mask[:, :, None]
    dummy_image = \
        np.concatenate([dummy_image, dummy_image, dummy_image], 2).astype(np.uint8)

    for y, out in zip(y_coords, x_coords):
        x_int = int(out)
        y = int(y)
        cv2.circle(dummy_image, (x_int, y),
                   lane_width_radius,
                   (255, 255, 255), -1)

    dummy_image = np.mean(dummy_image, 2)
    mask_lane[dummy_image == 255] = class_label


def draw_line_on_image(image, points, color=(255, 0, 0), thickness=10, imgcopy=True, sort_y=True, draw_junction=None):
    width, height, _ = image.shape

    if imgcopy:
        image = image.copy()

    if points is None or len(points) == 0:
        return image

    if draw_junction is not None:
        if len(points) > 2:
            if sort_y:
                points = sorted(points, key=lambda x: x[0][1])
            else:
                points = sorted(points, key=lambda x: x[0])

        _points = [i[0] for i in points]
        _juntion = [i[1] for i in points]
    else:
        _points = points

    for i in range(len(_points) - 1):
        x1 = int(_points[i][0])
        y1 = int(_points[i][1])
        x2 = int(_points[i + 1][0])
        y2 = int(_points[i + 1][1])
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            continue
        if draw_junction is not None and not draw_junction and _juntion[i]:
            continue
        start_point = (x1, y1)
        end_point = (x2, y2)
        image = cv2.line(image, start_point, end_point, color=color, thickness=thickness)

    return image


def create_mask_from_lane_params(output, lane_width_radius_for_metric, lane_fit_position_scaling_factor):
    params = output["lane_params"]
    lane_exists = to_numpy(output["lane_exist"])
    params = to_numpy(params)
    borders = to_numpy(output["borders"])

    mask_list = []
    for (param_batch, lane_exist, border) in zip(params, lane_exists, borders):
        mask_lane = np.zeros_like(to_numpy(output["mask"]))[0]

        height = width = lane_fit_position_scaling_factor

        dummy_mask = np.zeros_like(mask_lane)
        dummy_image = dummy_mask[:, :, None]
        dummy_image = \
            np.concatenate([dummy_image, dummy_image, dummy_image], 2).astype(np.uint8)

        for index, param in enumerate(param_batch):
            border_per_lane = border[index]
            if border_per_lane[0] > border_per_lane[1]:
                continue

            if lane_exist[index] < 0.1:
                continue

            poly_eqn = np.poly1d(param)
            unique_y_range = np.arange(border_per_lane[0] * height, border_per_lane[1] * height)
            unique_y_range_normalized = unique_y_range / height
            predicted_normalized = poly_eqn(unique_y_range_normalized)
            predicted = predicted_normalized * width

            numpy_out = np.concatenate([predicted[:, None], unique_y_range[:, None]], 1)
            lane_array = numpy_out[::1, :].round().astype('int').tolist()
            dummy_image = draw_line_on_image(image=dummy_image, points=lane_array, color=(index + 1, 0, 0),
                               thickness=lane_width_radius_for_metric)

        mask_lane = dummy_image[:, :, 0]
        mask_list.append(mask_lane[None, :])
    mask_lane = np.vstack(mask_list)
    return mask_lane


avg_pool = torch.nn.AvgPool2d(5, stride=1, padding=2)
def extract_points_from_mask(output_mask, probabilities, class_indices, selection_probability = 0.5, filter_probability = 0.3):
    output_mask_onehot = torch.zeros_like(probabilities).scatter_(1, output_mask.unsqueeze(1), 1.)
    probabilities = probabilities * output_mask_onehot
    # probabilities[probabilities < filter_probability] = 0
    probabilities_sum = probabilities.sum(-1)
    width = probabilities.shape[-1]
    x_position_anchors = torch.arange(width, device=probabilities.device).view(1, 1, 1, width)
    x_positions_based_probabilities = x_position_anchors * probabilities
    anchor_selection = probabilities.max(-1)[0]
    anchor_selection[anchor_selection < selection_probability] = 0
    estimated_x_positions = x_positions_based_probabilities.sum(-1) / probabilities_sum

    batch_lane_list = []
    background_mask_list = []
    for batch_index in range(probabilities.shape[0]):
        lane_dict = {}
        for cls in class_indices:
            class_specific_estimated_x_positions = estimated_x_positions[batch_index, cls, :]
            class_specific_anchors = anchor_selection[batch_index, cls, :]

            # max_based_anchors_unfiltered = torch.sum(output_mask[0] == cls, 1)
            # max_based_anchors_unfiltered[max_based_anchors_unfiltered > 0] = 1
            # class_specific_anchors = class_specific_anchors * max_based_anchors_unfiltered

            class_specific_y_coordinates = torch.where(class_specific_anchors > 0)[0]
            class_specific_x_coordinates = class_specific_estimated_x_positions[class_specific_anchors > 0]
            coordinates = torch.cat([class_specific_x_coordinates[:, None], class_specific_y_coordinates[:, None]], dim=1)

            selected_coordinates = \
                probabilities[batch_index, cls, class_specific_y_coordinates.long(), class_specific_x_coordinates.long()] > filter_probability
            coordinates = coordinates[selected_coordinates]

            lane_dict.update({cls: coordinates.detach().cpu().numpy()})

        batch_lane_list.append(lane_dict)
    return batch_lane_list


def simplistic_target(mask_cat, lanes, lane_categories, lane_width_radius, return_reduced_mask=True):
    mask_lane = np.zeros(mask_cat.shape[:2])

    for lane, cat in zip(lanes, lane_categories):
        if cat is None:
            continue
        # x_coords = lane[:, 0]
        # y_coords = lane[:, 1]
        #
        # params = np.polyfit(y_coords, x_coords, 2)
        # poly_eqn = np.poly1d(params)
        #
        # min_y = np.min(y_coords)
        # max_y = np.max(y_coords)
        #
        # y_coords = np.arange(min_y, 1280)
        # x_coords = poly_eqn(y_coords)
        #
        # fill_mask_with_circular_points(mask_lane=mask_lane, x_coords=x_coords, y_coords=y_coords,
        #                                lane_width_radius=lane_width_radius, class_label=cat)

        lane_array = lane[::1, :].round().astype('int').tolist()
        cat = int(cat)
        mask_cat = draw_line_on_image(mask_cat, lane_array, thickness=lane_width_radius,
                                      color=(cat, 0, 0), imgcopy=False)

    # return mask_lane
    if return_reduced_mask:
        return mask_cat[:, :, 0]
    else:
        return mask_cat

def _determine_lane_width_radius_for_metric(lane_width_radius_for_metric, transforms, image_height, image_width):
    lane_width_radius_for_metric_for_resized = None
    if lane_width_radius_for_metric is not None and transforms is not None:
        for transformation in transforms.transforms:
            if type(transformation).__name__ == "Resize":
                width = transformation.size[1]
                height = transformation.size[0]
                width_portion = lane_width_radius_for_metric * width / image_width
                height_portion = lane_width_radius_for_metric * height / image_height
                lane_width_radius_for_metric_for_resized = int(np.ceil(np.sqrt(width_portion * height_portion)))
            elif type(transformation).__name__ == "RandomResize":
                height = transformation.min_size
                height_portion = lane_width_radius_for_metric * height / image_height
                lane_width_radius_for_metric_for_resized = int(np.ceil(height_portion))

    return lane_width_radius_for_metric_for_resized


def lane_with_radius_settings(lane_width_radius, lane_width_radius_for_metric,
                              lane_width_radius_for_uncertain, lane_width_radius_for_binary,
                              transforms, image_height, image_width):
    if lane_width_radius is not None:
        lane_width_radius = lane_width_radius * 2
    else:
        lane_width_radius_dict = {"lane_width_radius": None,
                                  "lane_width_radius_for_metric": None,
                                  "lane_width_radius_for_uncertain": None,
                                  "lane_width_radius_for_metric_for_resized": None,
                                  "lane_width_radius_for_binary": None}
        return lane_width_radius_dict

    if lane_width_radius_for_metric is None:
        lane_width_radius_for_metric = lane_width_radius
    else:
        lane_width_radius_for_metric = lane_width_radius_for_metric * 2

    if lane_width_radius_for_uncertain is None:
        lane_width_radius_for_uncertain = lane_width_radius_for_metric
    else:
        lane_width_radius_for_uncertain = lane_width_radius_for_uncertain * 2

    if lane_width_radius_for_binary is None:
        lane_width_radius_for_binary = lane_width_radius
    else:
        lane_width_radius_for_binary = lane_width_radius_for_binary * 2

    lane_width_radius_for_metric_for_resized = _determine_lane_width_radius_for_metric(
        lane_width_radius_for_metric,transforms, image_height, image_width
    )

    lane_width_radius_dict = {"lane_width_radius": lane_width_radius,
                              "lane_width_radius_for_metric": lane_width_radius_for_metric,
                              "lane_width_radius_for_uncertain": lane_width_radius_for_uncertain,
                              "lane_width_radius_for_metric_for_resized": lane_width_radius_for_metric_for_resized,
                              "lane_width_radius_for_binary": lane_width_radius_for_binary}

    return lane_width_radius_dict

def obtain_ego_attributes(lines, height, width, length_threshold=90, merge_lane_pixel_threshold=0, number_of_selected_lanes_per_side=2, lane_fit_length=2):
    """
    Calculate the ego attributes from the lane lines and return the filtered lane list
    """
    categories = []
    filtered_lanes = []

    left_lanes = []
    right_lanes = []

    result_list = []
    left_labels = list(range(number_of_selected_lanes_per_side, 0, -1))
    right_labels = list(range(number_of_selected_lanes_per_side + 1, number_of_selected_lanes_per_side * 2 + 1, 1))

    for line in lines:
        line_x = line[:, 0]
        line_y = line[:, 1]

        length = np.sqrt((line_x[0] - line_x[-1]) ** 2 + (line_y[0] - line_y[-1]) ** 2)
        if length < length_threshold:
            continue

        curve = np.polyfit(line_x[-lane_fit_length:], line_y[-lane_fit_length:], deg=1)
        rad = np.arctan(curve[0])
        curve1 = np.polyfit(line_y[-lane_fit_length:], line_x[-lane_fit_length:], deg=1)

        if rad < 0:
            # result = np.poly1d(curve1)(height)
            y = np.poly1d(curve)(0)
            if y > height:
                result = np.poly1d(curve1)(height)
            else:
                result = -(height - y)
            if result not in result_list:
                left_lanes.append((result, line))
                result_list.append(result)
        elif rad > 0:
            # result = np.poly1d(curve1)(height)
            y = np.poly1d(curve)(width)
            if y > height:
                result = np.poly1d(curve1)(height)
            else:
                result = width + (height - y)
            if result not in result_list:
                right_lanes.append((result, line))
                result_list.append(result)

    heapq.heapify(left_lanes)
    heapq.heapify(right_lanes)

    left_lanes = merge_lanes(left_lanes, height, reverse_enabled=True, skip_heapification=True, pixel_threshold=merge_lane_pixel_threshold)
    right_lanes = merge_lanes(right_lanes, height, reverse_enabled=False, skip_heapification=True, pixel_threshold=merge_lane_pixel_threshold)

    left_lanes_selected = heapq.nlargest(number_of_selected_lanes_per_side, left_lanes)
    right_lanes_selected = heapq.nsmallest(number_of_selected_lanes_per_side, right_lanes)

    left_labels_filtered = left_labels[:len(left_lanes_selected)]
    right_labels_filtered = right_labels[:len(right_lanes_selected)]

    for line_tuple, label in zip(left_lanes_selected, left_labels_filtered):
        categories.append(label)
        filtered_lanes.append(line_tuple[1])

    for line_tuple, label in zip(right_lanes_selected, right_labels_filtered):
        categories.append(label)
        filtered_lanes.append(line_tuple[1])

    return categories, filtered_lanes

def merge_lanes(lines, height, pixel_threshold=100, reverse_enabled=False, skip_heapification=False):
    if not skip_heapification:
        lane_filtered_tuples = []
        for line in lines:
            line_x = line[:, 0]
            line_y = line[:, 1]

            try:
                curve1 = np.polyfit(line_y[-2:], line_x[-2:], deg=1)
            except Exception:
                curve1 = np.polyfit(line_y[-3:], line_x[-3:], deg=1)

            result = np.poly1d(curve1)(height)
            lane_filtered_tuples.append((result, line))

        heapq.heapify(lane_filtered_tuples)
    else:
        lane_filtered_tuples = lines

    lane_filtered_tuples.sort()
    if reverse_enabled:
        lane_filtered_tuples.reverse()
    eliminate_next = False
    lane_filtered = []
    for lane_index in range(len(lane_filtered_tuples)):
        if eliminate_next:
            eliminate_next = False
            continue

        if lane_index == len(lane_filtered_tuples) - 1:
            lane_filtered.append(lane_filtered_tuples[lane_index])
            continue

        line_prev_result = lane_filtered_tuples[lane_index][0]
        line_next_result = lane_filtered_tuples[lane_index+1][0]
        if abs(line_prev_result - line_next_result) < pixel_threshold:
            eliminate_next = True
            if abs(line_prev_result - line_next_result) > 20:
                lane_filtered.append(lane_filtered_tuples[lane_index])
                continue

            line_prev = lane_filtered_tuples[lane_index][1]
            line_next = lane_filtered_tuples[lane_index+1][1]

            line_prev_x = line_prev[:, 0]
            line_prev_y = line_prev[:, 1]

            line_next_x = line_next[:, 0]
            line_next_y = line_next[:, 1]

            length = max(len(line_prev), len(line_next))
            curve_prev = np.polyfit(line_prev_y, line_prev_x, deg=3)
            curve_next = np.polyfit(line_next_y, line_next_x, deg=3)

            max_y= min(max(line_next_y), max(line_prev_y))
            min_y= max(min(line_next_y), min(line_prev_y))

            unique_y_range = np.linspace(min_y, max_y, length)

            x_values_prev = np.poly1d(curve_prev)(unique_y_range)
            x_values_next = np.poly1d(curve_next)(unique_y_range)

            next_x_mean = np.mean(x_values_next)
            prev_x_mean = np.mean(x_values_prev)

            if next_x_mean > prev_x_mean:
                if reverse_enabled:
                    lane_filtered.append(lane_filtered_tuples[lane_index + 1])
                else:
                    lane_filtered.append(lane_filtered_tuples[lane_index])
            else:
                if reverse_enabled:
                    lane_filtered.append(lane_filtered_tuples[lane_index])
                else:
                    lane_filtered.append(lane_filtered_tuples[lane_index + 1])
        else:
            lane_filtered.append(lane_filtered_tuples[lane_index])
    return lane_filtered







