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


class InferenceSegmentationFourHead(InferenceSegmentationTorch):
    def __init__(self, yaml_name: str, **kwargs):
        super(InferenceSegmentationFourHead, self).__init__(yaml_name, **kwargs)
        # self.color_palette.append(self.color_palette[-1])
        # self.color_palette.append(self.color_palette[-1])
        # self.color_palette.append(self.color_palette[-1])
        # self.color_palette.append(self.color_palette[0])
        # self.color_palette.append(self.color_palette[-2])
        # self.color_palette.append(self.color_palette[-1])

        self.color_palette.append(self.color_palette[-1])
        self.port = 2002
        # self.color_palette = [self.color_palette[-1]]

    def select_indices(self, ordered_x):
        indices = None
        if len(ordered_x) == 4:
            if ordered_x[0][0] < ordered_x[1][0] < (self.size[1] / 2) < ordered_x[2][0] < ordered_x[3][0]:
                indices = [1, 2, 3, 4]
            elif ordered_x[0][0] < ordered_x[1][0] < ordered_x[2][0] < (self.size[1] / 2):
                indices, ordered_x = self.select_indices(ordered_x[1:])
            elif ordered_x[3][0] > ordered_x[2][0] > ordered_x[1][0] > (self.size[1] / 2):
                indices, ordered_x = self.select_indices(ordered_x[:-1])
        elif len(ordered_x) == 3:
            if ordered_x[0][0] < ordered_x[1][0] < (self.size[1] / 2) < ordered_x[2][0]:
                indices = [1, 2, 3]
            elif ordered_x[0][0] < (self.size[1] / 2) < ordered_x[1][0] < ordered_x[2][0]:
                indices = [2, 3, 4]
        elif len(ordered_x) == 2:
            if ordered_x[0][0] < (self.size[1] / 2) < ordered_x[1][0]:
                indices = [2, 3]
            elif ordered_x[0][0] < ordered_x[1][0] < (self.size[1] / 2):
                indices = [1, 2]
            elif (self.size[1] / 2) < ordered_x[0][0] < ordered_x[1][0]:
                indices = [3, 4]
        elif len(ordered_x) == 1:
            if ordered_x[0][0] < (self.size[1] / 2):
                indices = [2]
            elif (self.size[1] / 2) < ordered_x[0][0]:
                indices = [3]
        return indices, ordered_x

    def fit_lane(self, unique_y, mask, label, prob_mask, order=3):
        min_y = np.min(unique_y)
        max_y = np.max(unique_y)

        x_list = []
        for y_index in unique_y:
            line_where = np.where(mask[y_index] == label)[0]
            weight = prob_mask[y_index, line_where]
            weight = weight / weight.sum()
            x_list.append((line_where * weight).sum())

        params = np.polyfit(unique_y, x_list, order)
        poly_eqn = np.poly1d(params)

        unique_y_range = np.arange(min_y, max_y)
        predicted = poly_eqn(unique_y_range)
        return unique_y_range, predicted, params

    def calculate_prob(self, unique_y, params, prob_tensor):
        min_y = np.min(unique_y)
        max_y = np.max(unique_y)
        unique_y_range = np.arange(min_y, max_y)

        poly_eqn = np.poly1d(params)
        predicted = poly_eqn(unique_y_range)

        prob_list = []
        for y, out in zip(unique_y_range, predicted):
            x_int = int(out)
            min_x_int = max(0, x_int - 4)
            max_x_int = min(self.size[1], x_int + 5)
            if min_x_int >= max_x_int:
                continue

            prob = prob_tensor[y, min_x_int:max_x_int].mean()
            prob_list.append(prob)

        predicted_prob = np.mean(prob_list)
        return predicted_prob, unique_y_range, predicted

    def find_culane_from_bdd(self, mask_bdd, prob_bdd_raw):
        mask_bdd_1 = mask_bdd.copy()

        prob_bdd_raw = prob_bdd_raw.softmax(dim=1)[0]
        prob_bdd_raw_lane = prob_bdd_raw[1] + prob_bdd_raw[2]
        prob_bdd = to_numpy(prob_bdd_raw_lane)
        # mask_bdd_2 = mask_bdd.copy()
        #
        # mask_bdd_1[mask_bdd == 0] = 0
        # mask_bdd_1[mask_bdd == 2] = 255
        # mask_bdd_1[mask_bdd == 1] = 0
        #
        # mask_bdd_2[mask_bdd == 0] = 0
        # mask_bdd_2[mask_bdd == 2] = 0
        # mask_bdd_2[mask_bdd == 1] = 255
        #
        # mask_bdd_1 = mask_bdd_1.squeeze(0).astype(np.uint8)
        # mask_bdd_2 = mask_bdd_2.squeeze(0).astype(np.uint8)
        #
        # outputs_1 = cv2.connectedComponentsWithStats(mask_bdd_1, 4, cv2.CV_32S)
        # outputs_2 = cv2.connectedComponentsWithStats(mask_bdd_2, 4, cv2.CV_32S)
        #
        # mask2 = outputs_2[1].copy()
        # mask2[mask2 != 0] += (outputs_1[0] - 1)
        #
        # label_outputs_1 = outputs_1[1].copy() + mask2
        # label_outputs = np.zeros_like(label_outputs_1)
        # label_features = np.concatenate([outputs_1[2], outputs_2[2][1:]], 0)

        mask_bdd_1[mask_bdd == 0] = 0
        mask_bdd_1[mask_bdd == 2] = 255
        mask_bdd_1[mask_bdd == 1] = 255

        mask_bdd_1 = mask_bdd_1.squeeze(0).astype(np.uint8)

        # define the kernel
        # kernel = np.ones((10, 10), np.uint8)
        #
        # # opening the image
        # mask_bdd_1 = cv2.morphologyEx(mask_bdd_1, cv2.MORPH_OPEN,
        #                            kernel, iterations=1)

        outputs_1 = cv2.connectedComponentsWithStats(mask_bdd_1, 4, cv2.CV_32S)

        label_outputs_1 = outputs_1[1]
        label_outputs = np.zeros_like(label_outputs_1)
        label_features = outputs_1[2]

        x_center_list = []
        feature_tuple_list = []
        prob_dict = {}
        lane_dict = {}
        param_dict = {}
        param_half_dict = {}

        threshold = self.size[0] * self.size[1] / 400
        for i in range(label_features.shape[0]):
            if i == 0:
                continue
            if label_features[i][4] > threshold:  # Set this as hyper parameter
                feature_tuple_list.append((label_features[i][4], i))
                # label_outputs[label_outputs_1 == i] = label
                # label += 1

        heapq.heapify(feature_tuple_list)
        n_largest_list = heapq.nlargest(4, feature_tuple_list)
        for _, label in n_largest_list:
            where_tuple = np.where(label_outputs_1 == label)
            unique_y = np.unique(where_tuple[0])

            _, _, params = self.fit_lane(unique_y, label_outputs_1, label, prob_bdd)
            _, _, params1d = self.fit_lane(unique_y, label_outputs_1, label, prob_bdd, order=1)

            # _, _, params_half = self.fit_lane(unique_y_filtered, label_outputs_1, label, prob_bdd)

            poly_eqn = np.poly1d(params1d)
            y_hat = poly_eqn(self.size[0])
            param_dict.update({label: params})
            # param_half_dict.update({label: params_half})
            # min_index = np.argmax(where_tuple[0])
            # lowest_left = where_tuple[1][min_index]
            # x_center_list.append((lowest_left, label))
            x_center_list.append((y_hat, label))

        heapq.heapify(x_center_list)
        ordered_x = heapq.nsmallest(4, x_center_list)

        indices, ordered_x = self.select_indices(ordered_x)
        if indices is not None:
            for (index, (_, label)) in zip(indices, ordered_x):
                prob_list = []
                dummy_mask = np.zeros_like(mask_bdd.squeeze(0))
                dummy_image = dummy_mask[:, :, None]
                dummy_image = \
                    np.concatenate([dummy_image, dummy_image, dummy_image], 2).astype(np.uint8)

                where_tuple = np.where(label_outputs_1 == label)
                unique_y = np.unique(where_tuple[0])
                params = param_dict[label]
                # params_alternative = param_half_dict[label]

                predicted_prob, unique_y_range, predicted = self.calculate_prob(unique_y, params, prob_bdd)
                # if predicted_prob < 0.5:
                #     mean_y = np.mean(unique_y)
                #     unique_y_filtered = unique_y[unique_y > mean_y]
                #     predicted_prob_alternative, unique_y_range, predicted_alternative = \
                #         self.calculate_prob(unique_y_filtered, params_alternative, prob_bdd)
                #     if predicted_prob_alternative > predicted_prob:
                #         predicted = predicted_alternative
                #         predicted_prob = predicted_prob_alternative
                #         params = params_alternative
                #         unique_y_range = unique_y_filtered

                min_y = np.min(unique_y_range)
                max_y = np.max(unique_y_range)

                for y, out in zip(unique_y_range, predicted):
                    x_int = int(out)
                    cv2.circle(dummy_image, (x_int, y),
                               2, (255, 255, 255), -1)

                prob_dict.update({index: predicted_prob})

                lane_dict.update({index: {"prob": predicted_prob,
                                          "params": params,
                                          "trusted_region": (min_y, max_y)}})

                dummy_image = np.mean(dummy_image, 2)
                label_outputs[dummy_image == 255] = index
                # label_outputs[label_outputs_1 == label] = index

        label_outputs = label_outputs[None,]
        print(f"bdd scores: {prob_dict}")
        return label_outputs, lane_dict

    def culane_lane_fit(self, mask_culane, prob_culane_raw):
        mask_culane_1 = mask_culane.squeeze(0)
        label_outputs_culane = np.zeros_like(mask_culane_1)
        prob_culane = to_numpy(prob_culane_raw.softmax(dim=1)[0])

        prob_dict = {}
        lane_dict = {}
        for label in range(1, 5):
            where_tuple = np.where(mask_culane_1 == label)
            unique_y = np.unique(where_tuple[0])
            x_list = []
            prob_list = []
            if len(unique_y) > 0:
                unique_y_range, predicted, params = self.fit_lane(unique_y, mask_culane_1, label, prob_culane[label])

                dummy_mask = np.zeros_like(mask_culane.squeeze(0))
                dummy_image = dummy_mask[:, :, None]
                dummy_image = \
                    np.concatenate([dummy_image, dummy_image, dummy_image], 2).astype(np.uint8)

                min_y = np.min(unique_y)
                max_y = np.max(unique_y)

                for y, out in zip(unique_y_range, predicted):
                    x_int = int(out)
                    min_x_int = max(0, x_int - 4)
                    max_x_int = min(self.size[1], x_int + 5)
                    if min_x_int > max_x_int:
                        continue
                    cv2.circle(dummy_image, (x_int, y),
                               2, (255, 255, 255), -1)
                    prob = prob_culane[label, y, min_x_int:max_x_int].mean()
                    prob_list.append(prob)
                prob_dict.update({label: np.mean(prob_list)})

                lane_dict.update({label: {"prob": np.mean(prob_list),
                                          "params": params,
                                          "trusted_region": (min_y, max_y)}})

                dummy_image = np.mean(dummy_image, 2)
                label_outputs_culane[dummy_image == 255] = label

        label_outputs_culane = label_outputs_culane[None,]
        print(f"culane scores: {prob_dict}")
        return label_outputs_culane, lane_dict

    def find_culane_from_drivable(self, mask_drivable, prob_drivable_raw):
        mask_drivable_1 = mask_drivable.squeeze(0)
        mask_drivable_1_copy = mask_drivable_1.copy().astype(np.uint8)
        mask_drivable_1_copy[mask_drivable_1 == 0] = 255
        mask_drivable_1_copy[mask_drivable_1 > 0] = 0

        prob_drivable_raw = prob_drivable_raw.softmax(dim=1)[0]
        prob_drivable = to_numpy(prob_drivable_raw[0])

        mask_drivable_1 = np.ones_like(mask_drivable_1) * 2

        outputs_connected_drivable = cv2.connectedComponentsWithStats(mask_drivable_1_copy, 4, cv2.CV_32S)

        label_connected_outputs_drivable = outputs_connected_drivable[1]
        label_features_drivable = outputs_connected_drivable[2]
        feature_tuple_list_drivable = []

        label_outputs_drivable = np.zeros_like(mask_drivable_1)
        label_outputs_image = label_outputs_drivable[:, :, None]
        label_outputs_image = \
            np.concatenate([label_outputs_image, label_outputs_image, label_outputs_image], 2).astype(np.uint8)

        threshold = self.size[0] * self.size[1] / 400
        for i in range(label_features_drivable.shape[0]):
            if i == 0:
                continue
            if label_features_drivable[i][4] > threshold:
                feature_tuple_list_drivable.append((label_features_drivable[i][4], i))

        prob_dict = {}
        lane_dict = {}
        if len(feature_tuple_list_drivable) > 0:
            largest_list = heapq.nlargest(1, feature_tuple_list_drivable)
            drivable_connected_label = largest_list[0][1]

            mask_drivable_1[label_connected_outputs_drivable == drivable_connected_label] = 0

            ego_lane_where = np.where(mask_drivable_1 == 0)

            min_y = np.min(ego_lane_where[0])
            max_y = np.max(ego_lane_where[0])
            left_lane_list = []
            right_lane_list = []
            y_list = []
            # upper_thres = int((max_y - min_y) / 10)
            upper_thres = 3
            lower_thres = 3
            if max_y - upper_thres > min_y + lower_thres:
                for y_index in range(min_y + lower_thres, max_y - upper_thres, 1):
                    line_where = np.where(mask_drivable_1[y_index] == 0)[0]
                    if len(line_where) == 0:
                        continue
                    left_lane_list.append(line_where[0])
                    right_lane_list.append(line_where[-1])
                    y_list.append(y_index)

                residual_threshold = 5
                ransac1 = RANSACRegressor(PolynomialRegression(),
                                          residual_threshold=residual_threshold,
                                          random_state=0)

                ransac2 = RANSACRegressor(PolynomialRegression(),
                                          residual_threshold=residual_threshold,
                                          random_state=0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ransac1.fit(np.expand_dims(np.array(y_list), axis=1), left_lane_list)
                    ransac2.fit(np.expand_dims(np.array(y_list), axis=1), right_lane_list)
                fitness1 = np.sum(ransac1.inlier_mask_) / ransac1.inlier_mask_.shape[0]
                fitness2 = np.sum(ransac2.inlier_mask_) / ransac2.inlier_mask_.shape[0]

                # params1 = np.polyfit(np.array(y_list), left_lane_list, 2)
                # params2 = np.polyfit(np.array(y_list), right_lane_list, 2)
                #
                # poly_eqn = np.poly1d(params1)
                # out1_pure = poly_eqn(np.array(y_list))
                #
                # poly_eqn = np.poly1d(params2)
                # out2_pure = poly_eqn(np.array(y_list))

                # y = np.arange(min_y, max_y, 1)
                # for y, out in zip(y, out1_pure):
                #     cv2.circle(label_outputs_image, (int(out), y),
                #                2, (20, 20, 20), -1)
                #
                # y = np.arange(min_y, max_y, 1)
                # for y, out in zip(y, out2_pure):
                #     cv2.circle(label_outputs_image, (int(out), y),
                #                2, (180, 180, 180), -1)

                prob_list_1 = []
                prob_list_2 = []
                # if fitness1 > 0.6:
                y = np.arange(min_y, max_y, 1)
                # out_1 = ransac1.predict(np.expand_dims(y, axis=1))
                poly_eqn = np.poly1d(ransac1.estimator_.coeffs)
                out_1 = poly_eqn(y)
                for y, out in zip(y, out_1):
                    x_int = int(out)
                    max_x_int = min(self.size[1], x_int + 5)
                    cv2.circle(label_outputs_image, (int(out), y),
                               2, (255, 255, 255), -1)

                    prob = prob_drivable[y, x_int:max_x_int].mean()
                    prob_list_1.append(prob)

                # if fitness2 > 0.6:
                y = np.arange(min_y, max_y, 1)
                # out_2 = ransac2.predict(np.expand_dims(y, axis=1))
                poly_eqn = np.poly1d(ransac2.estimator_.coeffs)
                out_2 = poly_eqn(y)

                for y, out in zip(y, out_2):
                    x_int = int(out)
                    min_x_int = max(0, x_int - 4)

                    cv2.circle(label_outputs_image, (int(out), y),
                               2, (125, 125, 125), -1)

                    prob = prob_drivable[y, min_x_int:x_int].mean()
                    prob_list_2.append(prob)

                label_outputs_image = np.mean(label_outputs_image, 2)
                label_outputs_drivable[label_outputs_image == 255] = 2
                label_outputs_drivable[label_outputs_image == 125] = 3
                label_outputs_drivable[label_outputs_image == 20] = 1
                label_outputs_drivable[label_outputs_image == 180] = 4

                if len(prob_list_1) > 0:
                    prob_2 = np.mean(prob_list_1) * fitness1
                    lane_dict.update({2: {"prob": prob_2,
                                          "params": ransac1.estimator_.coeffs,
                                          "trusted_region": (min_y, max_y)}})
                    prob_dict.update({2: prob_2})
                    prob_dict.update({"2_pure": np.mean(prob_list_1)})

                if len(prob_list_2) > 0:
                    prob_3 = np.mean(prob_list_2) * fitness2
                    lane_dict.update({3: {"prob": prob_3,
                                          "params": ransac2.estimator_.coeffs,
                                          "trusted_region": (min_y, max_y)}})
                    prob_dict.update({3: prob_3})
                    prob_dict.update({"3_pure": np.mean(prob_list_2)})

                print(f"Drivable: {prob_dict}, fitness1: {fitness1}, fitness2: {fitness2}")

        label_outputs_culane_from_drivable = label_outputs_drivable[None,]
        mask_drivable_modified = mask_drivable_1[None,]

        return mask_drivable_modified, label_outputs_culane_from_drivable, lane_dict

    def _merge_lanes(self, lane_dict_bdd, lane_dict_culane, lane_dict_drivable, lane_dict_bdd_cropped):
        merged_dict = {}
        threshold = 0.3
        lane_dict_list = [lane_dict_bdd, lane_dict_culane, lane_dict_drivable, lane_dict_bdd_cropped]

        for lane_dict in lane_dict_list:
            if 2 in lane_dict:
                if 3 in lane_dict:
                    lane_3_prob = lane_dict[3]["prob"]
                else:
                    lane_3_prob = 0
                lane_dict[2]["prob"] = 2 / 3 * lane_dict[2]["prob"] + 1 / 3 * lane_3_prob

            if 3 in lane_dict:
                if 2 in lane_dict:
                    lane_2_prob = lane_dict[2]["prob"]
                else:
                    lane_2_prob = 0
                lane_dict[3]["prob"] = 2 / 3 * lane_dict[3]["prob"] + 1 / 3 * lane_2_prob

        for label in range(1, 5):
            prob_list = []
            for lane_dict in lane_dict_list:
                if label in lane_dict:
                    prob_list.append(lane_dict[label]["prob"])
                else:
                    prob_list.append(0)

            prob_array = np.array(prob_list) - np.array([0.0, 0, 0.0, 0.1])
            max_prob_index = np.argmax(prob_array)
            max_prob = prob_array[max_prob_index]
            if max_prob > threshold:
                lane_dict_selected = lane_dict_list[max_prob_index][label]
                merged_dict.update({label: lane_dict_selected})

        label_outputs_merged = np.zeros([self.size[0], self.size[1]])
        for label, lane_dict_selected in merged_dict.items():
            label_outputs_image = np.zeros([self.size[0], self.size[1], 1])
            dummy_image = \
                np.concatenate([label_outputs_image, label_outputs_image, label_outputs_image], 2).astype(np.uint8)

            min_y, max_y = lane_dict_selected["trusted_region"]
            unique_y_range = np.arange(min_y, max_y, 1)
            poly_eqn = np.poly1d(lane_dict_selected["params"])
            predicted = poly_eqn(unique_y_range)

            for y, out in zip(unique_y_range, predicted):
                x_int = int(out)
                cv2.circle(dummy_image, (x_int, y),
                           2, (255, 255, 255), -1)

            dummy_image = np.mean(dummy_image, 2)
            label_outputs_merged[dummy_image == 255] = label

        label_outputs_merged = label_outputs_merged[None,]

        return label_outputs_merged

    def post_process_outputs(self, outputs):
        mask_drivable = post_process_outputs_single(outputs["out"])
        mask_bdd = post_process_outputs_single(outputs["out_2"])
        mask_culane = post_process_outputs_single(outputs["out_3"])
        prob_culane = outputs["out_3"]
        prob_bdd = outputs["out_2"]
        prob_drivable = outputs["out"]
        # mask_mapillary = post_process_outputs_single(outputs["out_4"])

        # mask_bdd_binary = mask_bdd.copy()
        # mask_bdd_binary[mask_bdd > 0] = 255
        #
        # mask_culane_binary = mask_culane.copy()
        # mask_culane_binary[mask_culane > 0] = 255
        #
        # mask_bdd = mask_bdd & mask_culane_binary
        # mask_culane = mask_culane & mask_bdd_binary
        mask_bdd_copy = mask_bdd.copy()
        unique_y = np.where(mask_bdd > 0)[1]
        if len(unique_y) > 0:
            min_y = np.min(unique_y)
            max_y = np.max(unique_y)
            middle_y = int((min_y + (max_y - min_y) * 1 / 4))
            mask_bdd_copy[0, 0:middle_y, :] = 0

        label_outputs_culane_from_bdd, lane_dict_bdd = self.find_culane_from_bdd(mask_bdd, prob_bdd)

        label_outputs_culane_from_bdd_cropped, lane_dict_bdd_cropped = self.find_culane_from_bdd(mask_bdd_copy,
                                                                                                 prob_bdd)

        label_outputs_culane, lane_dict_culane = self.culane_lane_fit(mask_culane, prob_culane)

        mask_drivable_modified, label_outputs_culane_from_drivable, lane_dict_drivable = \
            self.find_culane_from_drivable(mask_drivable, prob_drivable)

        label_outputs_merged = self._merge_lanes(lane_dict_bdd, lane_dict_culane, lane_dict_drivable,
                                                 lane_dict_bdd_cropped)

        # output_mask_list = [mask_drivable, mask_bdd, mask_culane, label_outputs_culane_from_bdd,
        #                     label_outputs_culane_from_drivable, label_outputs_culane,
        #                     mask_drivable_modified, label_outputs_culane_from_bdd_cropped,
        #                     label_outputs_merged]

        output_mask_list = [mask_drivable, mask_bdd, mask_culane, label_outputs_merged]

        return output_mask_list

    def visualize(self, outputs: List[Union[np.ndarray, torch.Tensor]], frame_resized):
        output_frame_list = []
        frame_merged = frame_resized.copy()
        for out, color_palette in zip(outputs, self.color_palette):
            output_frame = visualize_single(out,
                                            frame_resized.copy(),
                                            color_palette,
                                            self.segmentation_mask_visualization_weights)
            output_frame_list.append(output_frame)

        # visualize_single(outputs[0],
        #                  frame_merged,
        #                  self.color_palette[0], 0.6)
        #
        # # visualize_single(outputs[3],
        # #                  frame_merged,
        # #                  self.color_palette[3], 0.85)
        #
        # visualize_single(outputs[2],
        #                  frame_merged,
        #                  self.color_palette[2], 0.7)

        return output_frame_list, frame_merged
