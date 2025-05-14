from ._utils import _SimpleSegmentationModel
from tairvision_object_detection.ops.misc import MLP
from tairvision_object_detection.nn import select_loss_function
from torch import nn
from collections import OrderedDict
import torch
from torch.nn import functional as F


class LaneFitSegmentationModel(_SimpleSegmentationModel):
    def __init__(self, *args, **kwargs):
        super(LaneFitSegmentationModel, self).__init__(*args, **kwargs)

        self.loss_coefficients = kwargs["loss_coefficients"]
        
        self.segmentation_mask_guided_feature_extraction = \
            kwargs.get("segmentation_mask_guided_feature_extraction", False)

        self.number_of_points_for_fit = kwargs["number_of_points_for_fit"]

        self.lane_order = kwargs["lane_order"] + 1
        self.number_of_lanes = kwargs["num_classes"] - 1
        self.number_of_lane_params = self.number_of_lanes * self.lane_order
        self.number_of_limits = self.number_of_lanes * 2

        if self.segmentation_mask_guided_feature_extraction:
            self.last_layer_num_channel = kwargs['number_of_channel_levels'][-1] * 6
        else:
            self.last_layer_num_channel = kwargs['number_of_channel_levels'][-1]

        self.lanefit_enabled = False
        if "lanefit_loss" in self.loss_coefficients and "border_loss" in self.loss_coefficients:
            self.lanefit_enabled = True
            self.hidden_channels_lane_fit = kwargs["hidden_channels"] + \
                                            [self.number_of_lane_params + self.number_of_limits + self.number_of_lanes]
            assert self.segmentation_mask_guided_feature_extraction is False, "this is only enabled for lane fit concept"

        else:
            self.hidden_channels_lane_fit = kwargs["hidden_channels"] + \
                                            [self.number_of_lanes]

        self.mlp = MLP(in_channels=self.last_layer_num_channel, hidden_channels=self.hidden_channels_lane_fit)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        segmentation_loss_function_dict = kwargs["segmentation_loss_function"]
        self.segmentation_loss_function = select_loss_function(segmentation_loss_function_dict)
        self.lane_params_loss_function = select_loss_function(
            {
                "name": "MSE",
                "config": {"reduction": "none"}
             }
        )
        self.lane_exist_loss_function = torch.nn.BCEWithLogitsLoss()

        self.relu = torch.nn.ReLU()

    def forward(self, image, ground_truth=None):
        features = self.backbone(image)
        result = OrderedDict()
        self._get_result_from_classifiers(features, result, self.classifier_keys, self.classifier_list, image)
        if self.segmentation_mask_guided_feature_extraction:
            features_to_be_out = features["out"][:, :, None]
            size = features_to_be_out.shape[-2:]
            x = F.interpolate(result["out"], size=size, mode='bilinear', align_corners=False)[:, None]
            x = x.detach()  # TODO, is it meaningful, decide it from the experiments
            normalization_coeff = x.view(*x.shape[:-2], -1).sum(-1).view(*x.shape[:-2], 1, 1)
            out_prob_tensort = features_to_be_out * x / normalization_coeff
            pooled_lane_feature = self.avgpool(out_prob_tensort)
            pooled_lane_feature = pooled_lane_feature.view(pooled_lane_feature.shape[0], -1)
            feature_avg_pool = self.avgpool(features["out"]).view(pooled_lane_feature.shape[0], -1)
            lane_fit_feature = torch.cat([feature_avg_pool, pooled_lane_feature], 1)
        else:
            lane_fit_feature = self.avgpool(features["out"]).view(-1, self.last_layer_num_channel)
        output = self.mlp(lane_fit_feature)
        # lane_exist = self.mlp2(feature_last_avg_pooled)
        if self.lanefit_enabled:
            lane_fit_params = output[:, :self.number_of_lane_params]
            lane_fit_params = lane_fit_params.view(-1, self.number_of_lanes, self.lane_order)
            result["lane_params"] = lane_fit_params
            borders = output[:, self.number_of_lane_params:self.number_of_lane_params + self.number_of_limits]
            borders = borders.view(-1, self.number_of_lanes, 2)
            borders = self.relu(borders)

            result["borders"] = borders

        result["lane_exist"] = output[:, -self.number_of_lanes:]

        if ground_truth is not None:
            loss_dict = self.calculate_loss(result, ground_truth)
            return result, loss_dict

        return result

    def calculate_loss(self, result, ground_truth):
        loss_dict = {}
        gt_lane_exist_float = ground_truth["lane_exist"].float()
        main_loss = self.segmentation_loss_function(result["out"], ground_truth["mask"])
        loss_dict.update({"main_loss": main_loss})

        if self.lanefit_enabled:
            gt_unique_y = ground_truth["unique_y"].float()
            gt_lane_params = ground_truth["lane_params"].float()

            gt_borders = ground_truth["borders"].float()
            predicted_lane_params = result["lane_params"] * gt_lane_exist_float[:, :, None]

            gt_x = self._calculate_power(gt_unique_y, gt_lane_params)
            predicted_x = self._calculate_power(gt_unique_y, predicted_lane_params)
            lanefit_loss_no_reduction = self.lane_params_loss_function(gt_x, predicted_x)

            gt_lane_exist_zero_handled = gt_lane_exist_float.sum(1)
            gt_lane_exist_zero_handled[gt_lane_exist_zero_handled == 0] = 1

            lanefit_loss_scaled = lanefit_loss_no_reduction.view(
                -1, self.number_of_points_for_fit * self.number_of_lanes).mean(1) / gt_lane_exist_zero_handled * 4

            lanefit_loss = lanefit_loss_scaled.mean()

            predicted_borders = result["borders"] * gt_lane_exist_float[:, :, None]
            border_loss_no_reduction = self.lane_params_loss_function(gt_borders, predicted_borders)

            border_loss_scaled = border_loss_no_reduction.view(
                -1, 2 * self.number_of_lanes).mean(1) / gt_lane_exist_zero_handled * 4

            border_loss = border_loss_scaled.mean()
            loss_dict.update({"lanefit_loss": lanefit_loss, "border_loss": border_loss})
        # lanefit_loss = self.lane_params_loss_function(
        #     result["lane_params"] * ground_truth["lane_exist"][:, :, None].float(),
        #     ground_truth["lane_params"].float()
        # )

        exist_loss = self.lane_exist_loss_function(result["lane_exist"], ground_truth["lane_exist"])

        loss_dict.update({"exist_loss": exist_loss})

        for key, value in self.loss_coefficients.items():
            loss_dict[key] = loss_dict[key] * value

        return loss_dict

    def _calculate_power(self, unique_y, gt_lane_params):
        degree = gt_lane_params.shape[-1] - 1
        result = torch.zeros_like(unique_y)
        for i in range(degree + 1):
            power = degree - i
            coeff = gt_lane_params[:, :, i][:, :, None]
            result += coeff * torch.pow(unique_y, power)
        return result


class GridBasedLaneFitSegmentationModel(LaneFitSegmentationModel):
    def __init__(self, *args, **kwargs):
        super(GridBasedLaneFitSegmentationModel, self).__init__(*args, **kwargs)

        self.lane_order = kwargs["lane_order"] + 1
        self.number_of_lanes = kwargs["num_classes"] - 1

        self.number_of_lane_params = self.lane_order
        self.number_of_limits = 2

        self.hidden_channels_lane_fit = kwargs["hidden_channels"] + \
                                        [self.number_of_lane_params + self.number_of_limits + self.number_of_lanes]

        self.last_layer_num_channel = kwargs['number_of_channel_levels'][-1]

        self.number_of_points_for_fit = kwargs["number_of_points_for_fit"]

        self.mlp = MLP(in_channels=self.last_layer_num_channel, hidden_channels=self.hidden_channels_lane_fit)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        segmentation_loss_function_dict = kwargs["segmentation_loss_function"]
        self.segmentation_loss_function = select_loss_function(segmentation_loss_function_dict)
        self.lane_params_loss_function = select_loss_function(
            {
                "name": "MSE",
                "config": {"reduction": "none"}
             }
        )
        self.lane_exist_loss_function = torch.nn.BCEWithLogitsLoss()

        self.loss_coefficients = kwargs["loss_coefficients"]

        self.relu = torch.nn.ReLU()

    def forward(self, image, ground_truth=None):
        features = self.backbone(image)
        result = OrderedDict()
        self._get_result_from_classifiers(features, result, self.classifier_keys, self.classifier_list, image)

        output = self.mlp(features["out"])

        # TODO, fill the keywords properly
        # result["lane_params"] =
        # result["borders"] =
        # result["lane_exist"] =

        if ground_truth is not None:
            loss_dict = self.calculate_loss(result, ground_truth)
            return result, loss_dict

        return result

