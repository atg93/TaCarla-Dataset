from typing import Tuple, List, Optional, Union, Dict
import numpy as np
import heapq
from matplotlib import colors
import warnings


# These part related with original openlane repo functions including pruning or 3D camera matrices
def color_name_2_rgb(color_name):
    rgb_code = colors.to_rgb(color_name)
    rgb_code = np.array(rgb_code) * 255
    rgb_code = tuple(rgb_code.astype('int'))
    return rgb_code

def prune_3d_lane_by_visibility(lane_3d, visibility):
    lane_3d = lane_3d[visibility > 0, ...]
    return lane_3d

def projection_g2im(cam_pitch, cam_height, K):
    P_g2c = np.array([[1,                             0,                              0,          0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                      [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0]])
    P_g2im = np.matmul(K, P_g2c)
    return P_g2im

def homograpthy_g2im(cam_pitch, cam_height, K):
    # transform top-view region to original image region
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                      [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
    H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))
    return H_g2im

def homograpthy_g2im_extrinsic(E, K):
    """E: extrinsic matrix, 4*4"""
    E_inv = np.linalg.inv(E)[0:3, :]
    H_g2c = E_inv[:, [0,1,3]]
    H_g2im = np.matmul(K, H_g2c)
    return H_g2im

def projection_g2im_extrinsic(E, K):
    E_inv = np.linalg.inv(E)[0:3, :]
    P_g2im = np.matmul(K, E_inv)
    return P_g2im

def convert_lanes_3d_to_gflat(lanes, P_g2gflat):
    """
        Convert a set of lanes from 3D ground coordinates [X, Y, Z], to IPM-based
        flat ground coordinates [x_gflat, y_gflat, Z]
    :param lanes: a list of N x 3 numpy arrays recording a set of 3d lanes
    :param P_g2gflat: projection matrix from 3D ground coordinates to flat ground coordinates
    :return:
    """
    # TODO: this function can be simplified with the derived formula
    for lane in lanes:
        # convert gt label to anchor label
        lane_gflat_x, lane_gflat_y = projective_transformation(P_g2gflat, lane[:, 0], lane[:, 1], lane[:, 2])
        lane[:, 0] = lane_gflat_x
        lane[:, 1] = lane_gflat_y

def prune_3d_lane_by_range(lane_3d, x_min, x_max, y_min=0, y_max=200):
    # TODO: solve hard coded range later
    # remove points with y out of range
    # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
    # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > y_min, lane_3d[:, 1] < y_max), ...]

    # remove lane points out of x range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                     lane_3d[:, 0] < x_max), ...]
    return lane_3d

def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
    """
    ones = np.ones((1, len(z)))
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals

### From here these are related with OpenLane dataset, maybe these fucntions can also be utilized for other 3D lane datasets
def _extract_important_lane_features(labels_filtered, ordered_lane_features, culane_category_dict,
                                    filtered_lanes, filtered_categories, original_categories,
                                    culane_categories_of_selected_lanes, lane_parameter_list,
                                    min_y_list, max_y_list):
    for label, lane_tuple in zip(labels_filtered, ordered_lane_features):
        # if label in culane_category_dict:
        #     selected_lane = culane_category_dict[label]
        # else:
        #     selected_lane = lane_tuple[1]
        #
        # if selected_lane[3] > 0 and selected_lane[3] != label:
        #     continue

        selected_lane = lane_tuple[1]

        filtered_lanes.append(selected_lane[0])
        filtered_categories.append(label)
        original_categories.append(selected_lane[1])
        culane_categories_of_selected_lanes.append(selected_lane[3])

        gt_flat_line_points = selected_lane[2]
        p = np.polyfit(gt_flat_line_points[:, 1], gt_flat_line_points[:, 0], 3)
        lane_parameter_list.append(p)

        min_y = np.min(gt_flat_line_points[:, 1])
        max_y = np.max(gt_flat_line_points[:, 1])
        min_y_list.append(min_y)
        max_y_list.append(max_y)


def convert_to_culane_format(lanes, lane_categories, bottom_points, gt_flat_lines,
                             middle_point, number_of_selected_lanes_per_side, attributes):
    join_lane_list_negative = []
    join_lane_list_positive = []

    negative_labels = list(range(number_of_selected_lanes_per_side, 0, -1))
    positive_labels = list(range(number_of_selected_lanes_per_side + 1, number_of_selected_lanes_per_side * 2 + 1, 1))

    filtered_lanes = []
    filtered_categories = []
    lane_parameter_list = []
    min_y_list = []
    max_y_list = []

    culane_category_dict = {}
    culane_categories_of_selected_lanes = []

    original_categories = []
    joint_lane_list = []
    for lane, category, bottom_point, gt_flat_line, attribute in zip(lanes, lane_categories, bottom_points, gt_flat_lines, attributes):
        if attribute > 0:
            culane_category_dict.update({attribute: [lane, category, gt_flat_line, attribute]})

    for lane, category, bottom_point, gt_flat_line, attribute in zip(lanes, lane_categories, bottom_points, gt_flat_lines, attributes):
        # max_y_point = np.max(lane[:, 1])
        # if max_y_point < y_threshold:
        #     continue

        joint_lane_list.append((None, [lane, category, gt_flat_line, attribute]))

        if bottom_point is None:
            continue

        if bottom_point < middle_point:
            new_tuple = (middle_point - bottom_point, [lane, category, gt_flat_line, attribute])
            join_lane_list_negative.append(new_tuple)
        else:
            new_tuple = (bottom_point - middle_point, [lane, category, gt_flat_line, attribute])
            join_lane_list_positive.append(new_tuple)

    heapq.heapify(join_lane_list_negative)
    heapq.heapify(join_lane_list_positive)

    negative_ordered = heapq.nsmallest(number_of_selected_lanes_per_side, join_lane_list_negative)
    positive_ordered = heapq.nsmallest(number_of_selected_lanes_per_side, join_lane_list_positive)

    negative_labels_filtered = negative_labels[:len(negative_ordered)]
    positive_labels_filtered = positive_labels[:len(positive_ordered)]

    negative_labels_filtered.reverse()
    negative_ordered.reverse()

    for lane_label, lane in culane_category_dict.items():
        if lane_label not in (negative_labels_filtered + positive_labels_filtered):
            if lane_label == 1:
                negative_labels_filtered = [lane_label] + negative_labels_filtered
                negative_ordered = [(None, lane)] + negative_ordered
            elif lane_label == 2:
                negative_labels_filtered = negative_labels_filtered + [lane_label]
                negative_ordered = negative_ordered + [(None, lane)]
            elif lane_label == 3:
                positive_labels_filtered = [lane_label] + positive_labels_filtered
                positive_ordered = [(None, lane)]  + positive_ordered
            elif lane_label == 4:
                positive_labels_filtered = positive_labels_filtered + [lane_label]
                positive_ordered = positive_ordered + [(None, lane)]

    # if len(negative_ordered) == 0 or len(positive_ordered) == 0:
    #     return filtered_lanes, filtered_categories

    _extract_important_lane_features(labels_filtered=negative_labels_filtered, ordered_lane_features=negative_ordered, culane_category_dict=culane_category_dict,
                                    filtered_lanes=filtered_lanes, filtered_categories=filtered_categories, original_categories=original_categories,
                                    culane_categories_of_selected_lanes=culane_categories_of_selected_lanes, lane_parameter_list=lane_parameter_list,
                                    min_y_list=min_y_list, max_y_list=max_y_list)

    _extract_important_lane_features(labels_filtered=positive_labels_filtered, ordered_lane_features=positive_ordered, culane_category_dict=culane_category_dict,
                                    filtered_lanes=filtered_lanes, filtered_categories=filtered_categories, original_categories=original_categories,
                                    culane_categories_of_selected_lanes=culane_categories_of_selected_lanes, lane_parameter_list=lane_parameter_list,
                                    min_y_list=min_y_list, max_y_list=max_y_list)

    _add_road_edge_as_attribute(joint_lane_list=joint_lane_list, culane_category_dict=culane_category_dict)

    problematic_indices, minor_problematic_indices = _check_lane_correctness(lane_parameter_list=lane_parameter_list, min_y_list=min_y_list, max_y_list=max_y_list,
                           filtered_lanes=filtered_lanes, filtered_categories=filtered_categories,
                           original_categories=original_categories, culane_categories_of_selected_lanes=culane_categories_of_selected_lanes)

    if len(minor_problematic_indices) == 1:
        del filtered_lanes[minor_problematic_indices[0]]
        del filtered_categories[minor_problematic_indices[0]]
        del original_categories[minor_problematic_indices[0]]

    if len(problematic_indices) > 0:
        filtered_lanes = [lane_feat[0] for lane_feat in list(culane_category_dict.values())]
        filtered_categories = list(culane_category_dict.keys())

    if len(filtered_categories) == 0:
        filtered_lanes = [lane_feat[0] for lane_feat in list(culane_category_dict.values())]
        filtered_categories = list(culane_category_dict.keys())

    elif len(filtered_categories) == 1:
        if original_categories[0] in [20, 21]:
            filtered_lanes = [lane_feat[0] for lane_feat in list(culane_category_dict.values())]
            filtered_categories = list(culane_category_dict.keys())

    elif len(filtered_categories) == 2:
        if original_categories[0] in [20, 21] and \
                original_categories[1] in [20, 21]:
            filtered_lanes = [lane_feat[0] for lane_feat in list(culane_category_dict.values())]
            filtered_categories = list(culane_category_dict.keys())

    return filtered_lanes, filtered_categories


def _check_lane_correctness(lane_parameter_list, min_y_list, max_y_list, filtered_lanes, filtered_categories, original_categories, culane_categories_of_selected_lanes):
    problematic_indices = []
    minor_problematic_indices = []
    for i in range(len(lane_parameter_list) - 1):
        # if i in attributes:
        #     continue
        prev_lane_parameters = lane_parameter_list[i]
        next_lane_parameters = lane_parameter_list[i+1]
        prev_min_y = min_y_list[i]
        next_min_y = min_y_list[i+1]
        prev_max_y = max_y_list[i]
        next_max_y = max_y_list[i+1]

        common_max = np.minimum(prev_max_y, next_max_y)
        common_min = np.maximum(prev_min_y, next_min_y)

        common_max = np.minimum(common_max, 80)

        if common_min > common_max:
            if prev_min_y > next_min_y:
                if culane_categories_of_selected_lanes[i] < 1:
                    problematic_indices.append(i)
            else:
                if culane_categories_of_selected_lanes[i+1] < 1:
                    problematic_indices.append(i+1)
            continue

        intervals = np.linspace(common_min, common_max, 10)

        previous_lane_values = np.polyval(prev_lane_parameters, intervals)
        next_lane_values = np.polyval(next_lane_parameters, intervals)

        differences = next_lane_values - previous_lane_values
        # print(differences)

        if np.sum(differences < 0) == differences.size:
            problematic_indices.append(i)
            problematic_indices.append(i + 1)
            continue

        differences = np.abs(differences)
        calculated_diff = np.min(differences)
        calculated_diff_max = np.max(differences)
        if calculated_diff > 5.5:
            if filtered_categories[i] == 2:
                if culane_categories_of_selected_lanes[i] != filtered_categories[i]:
                    problematic_indices.append(i)
                if culane_categories_of_selected_lanes[i+1] != filtered_categories[i+1]:
                    problematic_indices.append(i+1)
            elif filtered_categories[i] == 1:
                if culane_categories_of_selected_lanes[i] != filtered_categories[i]:
                    problematic_indices.append(i)
            elif filtered_categories[i] == 3:
                if culane_categories_of_selected_lanes[i+1] != filtered_categories[i+1]:
                    problematic_indices.append(i + 1)


        if 2 > calculated_diff_max:
            if filtered_categories[i] == 1 and original_categories[i] in [20, 21]:
                minor_problematic_indices.append(i)
            if filtered_categories[i] == 1 and 1 > calculated_diff_max:
                minor_problematic_indices.append(i)
            if filtered_categories[i] == 3 and original_categories[i+1] in [20, 21]:
                minor_problematic_indices.append(i + 1)
            if filtered_categories[i] == 3 and 1 > calculated_diff_max:
                minor_problematic_indices.append(i + 1)

    problematic_indices = list(dict.fromkeys(problematic_indices))
    minor_problematic_indices = list(dict.fromkeys(minor_problematic_indices))
    # print(f"problematic_indices: {problematic_indices}")
    return problematic_indices, minor_problematic_indices


def _add_road_edge_as_attribute(joint_lane_list, culane_category_dict):
    min_y_list = []
    max_y_list = []
    lane_parameter_list = []
    culane_categories_of_selected_lanes = []
    original_categories = []
    filtered_lanes = []
    filtered_categories = []
    labels_filtered = [-1] * len(joint_lane_list)

    _extract_important_lane_features(labels_filtered=labels_filtered, ordered_lane_features=joint_lane_list, culane_category_dict=culane_category_dict,
                                    filtered_lanes=filtered_lanes, filtered_categories=filtered_categories, original_categories=original_categories,
                                    culane_categories_of_selected_lanes=culane_categories_of_selected_lanes, lane_parameter_list=lane_parameter_list,
                                    min_y_list=min_y_list, max_y_list=max_y_list)

    culane_classes = list(culane_category_dict.keys())
    if len(culane_classes) == 4 or len(culane_classes) == 0:
        return

    culane_classes.sort()
    leftest_index = culane_classes[0]
    rightest_index = culane_classes[-1]
    if leftest_index != 1:
        _select_additional_lane_to_gt(culane_category_index=leftest_index, lane_parameter_list=lane_parameter_list,
                                      min_y_list=min_y_list, max_y_list=max_y_list,
                                      culane_categories_of_selected_lanes=culane_categories_of_selected_lanes, culane_category_dict=culane_category_dict,
                                      filtered_lanes=filtered_lanes, original_categories=original_categories, mode="left")

    if rightest_index != 4:
        _select_additional_lane_to_gt(culane_category_index=rightest_index, lane_parameter_list=lane_parameter_list,
                                      min_y_list=min_y_list, max_y_list=max_y_list,
                                      culane_categories_of_selected_lanes=culane_categories_of_selected_lanes, culane_category_dict=culane_category_dict,
                                      filtered_lanes=filtered_lanes, original_categories=original_categories, mode="right")


def _select_additional_lane_to_gt(culane_category_index, lane_parameter_list, min_y_list, max_y_list, culane_categories_of_selected_lanes, culane_category_dict, filtered_lanes, original_categories, mode):
    lane_parameter_list_index = culane_categories_of_selected_lanes.index(culane_category_index)

    lane_parameters_current = lane_parameter_list[lane_parameter_list_index]
    target_min_y = min_y_list[lane_parameter_list_index]
    target_max_y = max_y_list[lane_parameter_list_index]

    closest = 1000
    for selected_lane_index in range(len(lane_parameter_list)):
        if selected_lane_index == lane_parameter_list_index:
            continue

        selected_lane_parameters = lane_parameter_list[selected_lane_index]
        selected_min_y = min_y_list[selected_lane_index]
        selected_max_y = max_y_list[selected_lane_index]

        common_max = np.minimum(selected_max_y, target_max_y)
        common_min = np.maximum(selected_min_y, target_min_y)

        if common_min > common_max:
            continue

        intervals = np.linspace(common_min, common_max, 10)

        target_lane_values = np.polyval(lane_parameters_current, intervals)
        selected_lane_values = np.polyval(selected_lane_parameters, intervals)

        differences = target_lane_values - selected_lane_values
        if mode == "left" and np.mean(differences) < 0:
            continue
        if mode == "right" and np.mean(differences) > 0:
            continue

        differences = np.abs(differences)
        calculated_diff = np.min(differences)
        calculated_diff_max = np.max(differences)
        if calculated_diff < 6 and calculated_diff_max > 2:
            if np.mean(differences) < closest:
                if mode == "left":
                    update_index = culane_category_index - 1
                elif mode == "right":
                    update_index = culane_category_index + 1

                culane_category_dict.update({update_index: [
                    filtered_lanes[selected_lane_index], original_categories[selected_lane_index], None, update_index]})
                closest = np.mean(differences)

def dummy_lane_merge_code(img_lanes, min_points, max_points, gt_lane_pts, polynomials_third, bottom_points, P_g2im):
    for index_main in range(len(img_lanes)):
        min_point = min_points[index_main]
        if min_point < 30:
            continue

        for index_possible_merge in range(len(img_lanes)):
            if min_point < max_points[index_possible_merge]:
                continue

            if len(img_lanes[index_possible_merge]) < 100:
                continue

            intervals = np.linspace(min_points[index_possible_merge],
                                    max_points[index_possible_merge], 40)

            gt_lane_flat_merged = np.vstack([gt_lane_pts[index_possible_merge], gt_lane_pts[index_main]])
            p3_merged = np.polyfit(gt_lane_flat_merged[:, 1], gt_lane_flat_merged[:, 0], 3)
            p3_orig = polynomials_third[index_possible_merge]

            sampled_x_merged = np.polyval(p3_merged, intervals)
            sampled_x_orig = np.polyval(p3_orig, intervals)

            difference = np.mean(np.abs(sampled_x_orig - sampled_x_merged))
            if difference < 0.05:
                new_interval = np.linspace(min_points[index_possible_merge],
                                    max_points[index_main], 300)
                new_interval_x = np.polyval(p3_merged, new_interval)
                gt_lane_flat_merged = np.vstack([new_interval_x, new_interval, np.zeros_like(new_interval)]).T
                gt_lane_pts[index_possible_merge] = gt_lane_flat_merged
                x_vals, y_vals = projective_transformation(P_g2im, gt_lane_flat_merged[:, 0],
                                                           gt_lane_flat_merged[:, 1], gt_lane_flat_merged[:, 2])
                gt_laneline_im_oneline = np.array([x_vals, y_vals]).T
                img_lanes[index_possible_merge] = gt_laneline_im_oneline
                bottom_points[index_main] = None


def bottom_point_extraction_from_3d_world(gt_lane_pts, gt_lane_pts_flat, attributes, P_g2im, polynomials_third, min_points, max_points, bottom_points, img_lanes):
    ordering_polynomials = []
    for gt_lane, gt_lane_flat, attribute in zip(gt_lane_pts, gt_lane_pts_flat, attributes):
        x_vals, y_vals = projective_transformation(P_g2im, gt_lane[:, 0], gt_lane[:, 1], gt_lane[:, 2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # p = np.polyfit(y_vals, x_vals, 1)
            # x_bottom = np.polyval(p, np.array([1080]))

            if gt_lane[:, 1].shape[0] < 35:
                p = np.polyfit(gt_lane[:, 1], gt_lane[:, 0], 1)
            else:
                p = np.polyfit(gt_lane[:, 1], gt_lane[:, 0], 2)

            p3 = np.polyfit(gt_lane[:, 1], gt_lane[:, 0], 3)
            polynomials_third.append(p3)
            ordering_polynomials.append(p)

            x_bottom = (np.polyval(p, 5) + np.polyval(p, 7) + np.polyval(p, 10)) / 3

            min_point = np.min(gt_lane[:, 1])
            max_point = np.max(gt_lane[:, 1])

            min_points.append(min_point)
            max_points.append(max_point)

            if min_point > 30 and attribute < 1:
                x_bottom = None

            bottom_points.append(x_bottom)
        gt_laneline_im_oneline = np.array([x_vals, y_vals]).T
        gt_laneline_im_oneline = gt_laneline_im_oneline[gt_laneline_im_oneline[:, 1].argsort()]
        img_lanes.append(gt_laneline_im_oneline)

    if len(min_points) > 0:
        selected_part_of_min = np.array(min_points)[np.array(bottom_points) != None]
        if selected_part_of_min.size > 0:
            selected_min = np.min(np.array(min_points)[np.array(bottom_points) != None])
            selected_max = np.max(np.array(max_points)[np.array(bottom_points) != None])
        else:
            selected_min = None
            selected_max = None

    for index, bottom_point in enumerate(bottom_points):
        if bottom_point is None and selected_min is not None:
            if selected_min < min_points[index] < selected_max and img_lanes[index].shape[0] > 35:
                p = ordering_polynomials[index]
                bottom_points[index] = (np.polyval(p, 5) + np.polyval(p, 7) + np.polyval(p, 10)) / 3

    for index, bottom_point in enumerate(bottom_points):
        if bottom_point is None:
            continue
        bottom_points_to_operate = np.array(bottom_points)
        bottom_points_to_operate[index] = 1000
        bottom_points_to_operate[bottom_points_to_operate == None] = 1000
        difference = np.abs(bottom_point - np.array(bottom_points_to_operate))
        least_dist_index = np.argmin(difference)
        if difference[least_dist_index] < 1:

            prev_lane_parameters = polynomials_third[index]
            next_lane_parameters = polynomials_third[least_dist_index]

            prev_min_y = min_points[index]
            next_min_y = min_points[least_dist_index]

            prev_max_y = max_points[index]
            next_max_y = max_points[least_dist_index]

            common_max = np.minimum(prev_max_y, next_max_y)
            common_min = np.maximum(prev_min_y, next_min_y)

            common_max = np.minimum(common_max, 80)

            if common_min > common_max:
                if min_points[index] > min_points[least_dist_index]:
                    bottom_points[index] = None
                    # if difference[least_dist_index] < 0.1:
                    #     img_lanes[least_dist_index] = np.vstack([img_lanes[least_dist_index], img_lanes[index]])
                else:
                    bottom_points[least_dist_index] = None
                    # if difference[least_dist_index] < 0.1:
                    #     img_lanes[index] = np.vstack([img_lanes[index], img_lanes[least_dist_index]])

            else:
                intervals = np.linspace(common_min, common_max, 10)
                previous_lane_values = np.polyval(prev_lane_parameters, intervals)
                next_lane_values = np.polyval(next_lane_parameters, intervals)

                differences = np.abs(next_lane_values - previous_lane_values)
                if np.max(differences) < 0.2:
                    if min_points[index] > min_points[least_dist_index]:
                        bottom_points[index] = None
                    else:
                        bottom_points[least_dist_index] = None

def switch_lanes(min_y_list, max_y_list, lane_parameter_list, filtered_lanes, filtered_categories,
                 original_categories, culane_categories_of_selected_lanes, differences, index_prev):
    i = index_prev
    prev_min_y = min_y_list[i]
    next_min_y = min_y_list[i + 1]
    min_y_list[i] = next_min_y
    min_y_list[i + 1] = prev_min_y

    prev_max_y = max_y_list[i]
    next_max_y = max_y_list[i + 1]
    max_y_list[i] = next_max_y
    max_y_list[i + 1] = prev_max_y

    prev_lane_parameters = lane_parameter_list[i]
    next_lane_parameters = lane_parameter_list[i + 1]
    lane_parameter_list[i] = next_lane_parameters
    lane_parameter_list[i + 1] = prev_lane_parameters

    prev_lane = filtered_lanes[i]
    next_lane = filtered_lanes[i + 1]
    filtered_lanes[i] = next_lane
    filtered_lanes[i + 1] = prev_lane

    prev_category = filtered_categories[i]
    next_category = filtered_categories[i + 1]
    filtered_categories[i] = next_category
    filtered_categories[i + 1] = prev_category

    prev_orig_category = original_categories[i]
    next_orig_category = original_categories[i + 1]
    original_categories[i] = next_orig_category
    original_categories[i + 1] = prev_orig_category

    prev_culane_category = culane_categories_of_selected_lanes[i]
    next_culane_category = culane_categories_of_selected_lanes[i + 1]
    culane_categories_of_selected_lanes[i] = next_culane_category
    culane_categories_of_selected_lanes[i + 1] = prev_culane_category

    differences = -differences
    return differences


