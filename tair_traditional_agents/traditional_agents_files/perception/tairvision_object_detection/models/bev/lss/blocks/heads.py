import torch
import torch.nn as nn

from tairvision.models.detection.fcos import FCOSNet
from tairvision.models.detection.fcos_bev import FCOSBevNet
from tairvision.models.bev.lss.training.losses import SpatialRegressionLoss, SegmentationLoss
from tairvision.models.bev.lss.utils.instance import predict_instance_segmentation_and_trajectories


class FCOSNetAdaptor(FCOSNet):
    def __init__(self, backbone, num_classes):
        super().__init__(backbone, num_classes)
        self.locations_strides_sois = None, None, None
        self.cls_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.ctr_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.bbox_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def get_head_outputs(self, features):
        features = list(features.values())
        # Compute the fcos heads outputs using the features
        head_outputs = self.head(features)
        self.locations_strides_sois = self.compute_locations(features)

        return head_outputs

    def get_loss(self, head_outputs, targets):
        targets = self.compute_relative_targets(targets, *self.locations_strides_sois)
        loss = self.head.compute_loss(targets, head_outputs)

        cls_factor = 1 / torch.exp(self.cls_weight)
        ctr_factor = 1 / (2 * torch.exp(self.ctr_weight))
        bbox_factor = 1 / torch.exp(self.bbox_weight)

        loss = {
            'loss_2d_cls': cls_factor * loss['cls_loss'],
            'loss_2d_ctr': ctr_factor * loss['ctr_loss'],
            'loss_2d_bbox': bbox_factor * loss['bbox_loss'],
            'uncertainty_2d_cls': 0.5 * self.cls_weight,
            'uncertainty_2d_ctr': 0.5 * self.ctr_weight,
            'uncertainty_2d_bbox': 0.5 * self.bbox_weight,
        }

        factor = {
            '2d_cls': cls_factor,
            '2d_ctr': ctr_factor,
            '2d_bbox': bbox_factor
        }

        return loss, factor

    def get_detections(self, head_outputs):
        locations, strides, _ = self.locations_strides_sois
        detections = self.predict_proposals(head_outputs, locations, strides, top_feats=None)

        return detections


class FCOSBevNetAdaptor(FCOSBevNet):
    def __init__(self, backbone, num_classes, regression_out_channels, regression_functions=None, **kwargs):
        super().__init__(backbone, num_classes, regression_out_channels,
                         regression_functions=regression_functions, **kwargs)
        self.locations_strides_sois = None, None, None
        self.cls_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.ctr_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.bbox_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.other_weights = nn.ParameterList()
        for _ in regression_out_channels:
            self.other_weights.append(nn.Parameter(torch.tensor(0.0), requires_grad=True))

    def get_head_outputs(self, feats_dec):
        features = []
        feature = feats_dec['y']
        b, s, c, h, w = feature.shape
        feature = feature.view(b * s, c, h, w)
        features.append(feature)

        # Compute the fcos heads outputs using the features
        head_outputs = self.head(features)
        self.locations_strides_sois = self.compute_locations(features)

        return head_outputs

    def get_loss(self, head_outputs, targets):
        targets = self.compute_relative_targets(targets, *self.locations_strides_sois)
        loss = self.head.compute_loss(targets, head_outputs)

        cls_factor = 1 / torch.exp(self.cls_weight)
        ctr_factor = 1 / (2 * torch.exp(self.ctr_weight))
        bbox_factor = 1 / torch.exp(self.bbox_weight)

        loss_dict = {
            'loss_3d_cls': cls_factor * loss['cls_loss'],
            'loss_3d_ctr': ctr_factor * loss['ctr_loss'],
            'loss_3d_bbox': bbox_factor * loss['bbox_loss'],
            'uncertainty_3d_cls': 0.5 * self.cls_weight,
            'uncertainty_3d_ctr': 0.5 * self.ctr_weight,
            'uncertainty_3d_bbox': 0.5 * self.bbox_weight,
        }

        factor = {
            '3d_cls': cls_factor,
            '3d_ctr': ctr_factor,
            '3d_bbox': bbox_factor,
        }

        for i_other, other_weight in enumerate(self.other_weights):
            idx_str = str(i_other)
            other_factor = 1 / (2 * torch.exp(other_weight))

            loss_dict['loss_3d_other_' + idx_str] = other_factor * loss['other_losses'][i_other]
            loss_dict['uncertainty_3d_other_' + idx_str] = 0.5 * other_weight
            factor['3d_other_' + idx_str] = other_factor

        return loss_dict, factor

    def get_detections(self, head_outputs):
        locations, strides, _ = self.locations_strides_sois
        detections = self.predict_proposals(head_outputs, locations, strides, top_feats=None)

        return detections


class DynamicSegmentationHead(nn.Module):
    def __init__(self, cfg, n_classes, in_channels):
        super().__init__()

        self.segm_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.zpos_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.segm_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1, padding=0),
        )

        self.center_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        self.zpos_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            )

        self.loss_fn = nn.ModuleDict()
        self.loss_fn['segm'] = SegmentationLoss(class_weights=torch.Tensor(cfg.MODEL.HEADDYNAMIC.SEMANTIC_SEG.WEIGHTS),
                                                use_top_k=cfg.MODEL.HEADDYNAMIC.SEMANTIC_SEG.USE_TOP_K,
                                                top_k_ratio=cfg.MODEL.HEADDYNAMIC.SEMANTIC_SEG.TOP_K_RATIO,
                                                future_discount=cfg.FUTURE_DISCOUNT
                                                )
        self.loss_fn['center'] = SpatialRegressionLoss(norm=2, future_discount=cfg.FUTURE_DISCOUNT)
        self.loss_fn['offset'] = SpatialRegressionLoss(norm=1, future_discount=cfg.FUTURE_DISCOUNT,
                                                       ignore_index=cfg.DATASET.IGNORE_INDEX
                                                       )
        self.loss_fn['zpos'] = SpatialRegressionLoss(norm=3, future_discount=cfg.FUTURE_DISCOUNT,
                                                     ignore_index=cfg.DATASET.IGNORE_INDEX
                                                     )

    def get_head_outputs(self, feats_dec):
        x = feats_dec['y']

        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)

        segm = self.segm_head(x)
        center = self.center_head(x)
        offset = self.offset_head(x)
        zpos = self.zpos_head(x)

        output = {
            'segm': segm.view(b, s, *segm.shape[1:]),
            'center': center.view(b, s, *center.shape[1:]),
            'offset': offset.view(b, s, *offset.shape[1:]),
            'zpos': zpos.view(b, s, *zpos.shape[1:])
        }

        return output

    def get_loss(self, head_outputs, targets):
        segm_factor = 1 / torch.exp(self.segm_weight)
        center_factor = 1 / (2 * torch.exp(self.center_weight))
        offset_factor = 1 / (2 * torch.exp(self.offset_weight))
        zpos_factor = 1 / (2 * torch.exp(self.zpos_weight))

        loss_segm = self.loss_fn['segm'](head_outputs['segm'], targets['segmentation'])
        loss_center = self.loss_fn['center'](head_outputs['center'], targets['centerness'])
        loss_offset = self.loss_fn['offset'](head_outputs['offset'], targets['offset'])
        loss_zpos = self.loss_fn['zpos'](head_outputs['zpos'], targets['z_position'])

        loss = {
            'loss_dyn_segm': segm_factor * loss_segm,
            'loss_dyn_center': center_factor * loss_center,
            'loss_dyn_offset': offset_factor * loss_offset,
            'loss_dyn_zpos': zpos_factor * loss_zpos,
            'uncertainty_dyn_segm': 0.5 * self.segm_weight,
            'uncertainty_dyn_center': 0.5 * self.center_weight,
            'uncertainty_dyn_offset': 0.5 * self.offset_weight,
            'uncertainty_dyn_zpos': 0.5 * self.zpos_weight
        }

        factor = {
            'dyn_segm': segm_factor,
            'dyn_center': center_factor,
            'dyn_offset': offset_factor,
            'dyn_zpos': zpos_factor
        }

        return loss, factor

    @staticmethod
    def post_process(output):
        compute_matched_centers = output['segm'].shape[1] > 1
        inst, matched_centers = predict_instance_segmentation_and_trajectories(
            output, compute_matched_centers=compute_matched_centers)
        post_output = {
            'inst': inst,
            'matched_centers': matched_centers,
            'segm': torch.argmax(output['segm'], dim=2, keepdims=True)
        }

        return post_output


class StaticSegmentationHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()

        self.lanes_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.lines_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.lanes_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1, padding=0),
        )

        self.lines_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1, padding=0),
        )

        self.loss_fn = nn.ModuleDict()
        self.loss_fn['lanes'] = SegmentationLoss(class_weights=torch.Tensor(cfg.MODEL.HEADSTATIC.LANES.WEIGHTS),
                                                 use_top_k=cfg.MODEL.HEADSTATIC.LANES.USE_TOP_K,
                                                 top_k_ratio=cfg.MODEL.HEADSTATIC.LANES.TOP_K_RATIO,
                                                 future_discount=cfg.FUTURE_DISCOUNT
                                                 )

        self.loss_fn['lines'] = SegmentationLoss(class_weights=torch.Tensor(cfg.MODEL.HEADSTATIC.LINES.WEIGHTS),
                                                 use_top_k=cfg.MODEL.HEADSTATIC.LINES.USE_TOP_K,
                                                 top_k_ratio=cfg.MODEL.HEADSTATIC.LINES.TOP_K_RATIO,
                                                 future_discount=cfg.FUTURE_DISCOUNT
                                                 )

    def get_head_outputs(self, feats_dec):
        x = feats_dec['y']

        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)

        lanes = self.lanes_head(x)
        lines = self.lines_head(x)

        output = {
            'lanes': lanes.view(b, s, *lanes.shape[1:]),
            'lines': lines.view(b, s, *lines.shape[1:]),
        }

        return output

    def get_loss(self, head_outputs, targets):
        lanes_factor = 1 / torch.exp(self.lanes_weight)
        lines_factor = 1 / torch.exp(self.lines_weight)

        loss_lanes = self.loss_fn['lanes'](head_outputs['lanes'], targets['lanes'])
        loss_lines = self.loss_fn['lines'](head_outputs['lines'], targets['lines'])

        loss = {
            'loss_lanes': lanes_factor * loss_lanes,
            'loss_lines': lines_factor * loss_lines,
            'uncertainty_lanes': 0.5 * self.lanes_weight,
            'uncertainty_lines': 0.5 * self.lines_weight
        }

        factor = {
            'lanes': lanes_factor,
            'lines': lines_factor
        }

        return loss, factor

    @staticmethod
    def post_process(output):
        post_output = {
            'lanes': output['lanes'].argmax(2, keepdim=True),
            'lines': output['lines'].argmax(2, keepdim=True)
        }

        return post_output


class FlowHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()

        self.flow_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.flow_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1, padding=0),
        )

        self.loss_fn = SpatialRegressionLoss(norm=1, future_discount=cfg.FUTURE_DISCOUNT,
                                             ignore_index=cfg.DATASET.IGNORE_INDEX)

    def get_head_outputs(self, feats_dec):
        x = feats_dec['y']

        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)

        flow = self.flow_head(x)

        output = {
            'flow': flow.view(b, s, *flow.shape[1:])
        }

        return output

    def get_loss(self, head_outputs, targets):
        flow_factor = 1 / (2 * torch.exp(self.flow_weight))

        loss_flow = self.loss_fn(head_outputs['flow'], targets['flow'])

        loss = {
            'loss_flow': flow_factor * loss_flow,
            'uncertainty_flow': 0.5 * self.flow_weight,
        }

        factor = {
            'flow': flow_factor
        }

        return loss, factor
