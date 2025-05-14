import math
from collections import OrderedDict
import warnings
import copy
import numpy as np
from functools import partial
import cv2

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Dict, List, Tuple, Optional, Union

from .process import batch_images, get_original_image_sizes
from .backbone_utils import resnet_fpn_backbone, _validate_trainable_layers, regnet_fpn_backbone, convnext_fpn_backbone
from tairvision_object_detection.ops.feature_pyramid_network import LastLevelP6, LastLevelP6P7, LastLevelMaxPool
from tairvision_object_detection.ops import sigmoid_focal_loss
import torch.nn.functional as F

from tairvision_object_detection.models.detection.context import IndependentContextNetwork, SharedContextNetwork

from tairvision_object_detection.models.detection.fcos import eager_outputs, compute_ious


__all__ = [
    "SOLOv2", "SOLOv2Head", "solov2_regnet_fpn"
]

INF = 100000000

def solov2_regnet_fpn(type="regnet_y_1_6gf",num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None,
                      extra_before=False, pyramid_type="bifpn", depthwise=True, repeats=3,
                      fusion_type='fastnormed', bifpn_norm_layer=None,
                      **kwargs):

    # kwargs = adDict(kwargs)
    trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

    # returned_layers = kwargs.backbone.returned_layers
    returned_layers = kwargs['backbone']['returned_layers']

    extra_blocks_dict = dict(LastLevelP6=LastLevelP6, LastLevelP6P7=LastLevelP6P7, LastLevelMaxPool=LastLevelMaxPool)

    # extra_block_type = kwargs.neck.extra_block_type
    extra_block_type = kwargs['neck']['extra_block_type']
    extra_blocks = extra_blocks_dict[extra_block_type] if extra_block_type in extra_blocks_dict else None

    backbone = regnet_fpn_backbone(type, pretrained_backbone, returned_layers=returned_layers,
                                   extra_blocks=extra_blocks, trainable_layers=trainable_backbone_layers,
                                   extra_before=extra_before,
                                   pyramid_type=pyramid_type, depthwise=depthwise, repeats=repeats,
                                   fusion_type=fusion_type, bifpn_norm_layer=bifpn_norm_layer)

    del kwargs['backbone']
    del kwargs['neck']
    model = SOLOv2(backbone, num_classes, **kwargs)
    return model


class SOLOv2(nn.Module):

    def __init__(self, backbone, num_classes, num_keypoints=0,
                 # transform parameters
                 min_size=800, max_size=1333,
                 context_module=None,
                 mask_head=None, # mask head cfg
                 test_cfg=None,
                 **kwargs
                 ):
        super().__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.backbone = backbone

        if context_module is not None:
            context_module, module_sharing = context_module.split('-')
            if module_sharing == 'shared':
                self.context = SharedContextNetwork(backbone.out_channels, context_module)
            elif module_sharing == 'independent':
                self.context = IndependentContextNetwork(backbone.out_channels, context_module, num_feature_layers=5)
            else:
                raise ValueError("context module sharing should be either -shared or -independent")
        else:
            self.context = None

        self.mask_head = SOLOv2Head(num_classes, backbone.out_channels, test_cfg=test_cfg, **mask_head)

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        # used only on torchscript mode
        self._has_warned = False

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None, calc_val_loss: bool = False) \
            -> Union[Dict[str, Tensor], List[Dict[str, Tensor]], Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]]:
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
            calc_val_loss (bool): enable val_loss calculation

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                            boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = get_original_image_sizes(targets) if targets is not None else []

        if not self.training:
            targets = None

        # transform the input
        images = batch_images(images)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        if self.context is not None:
            features = self.context(features)

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the fcos heads outputs using the features
        # head_outputs = self.head(features[self.box_feat_range])

        cls_preds, kernel_preds, bbox_preds, ins_preds = self.mask_head(features)

        loss, detections = None, None
        if self.training or calc_val_loss:
            assert targets is not None
            gt_bboxes = [g['boxes'] for g in targets]
            gt_labels = [g['labels'] for g in targets]

            gt_masks = []
            for g in targets:
                ins_masks = torch.zeros((len(g['labels']), *images.tensors.shape[2:]), device=images.tensors.device)
                ins_masks[:, :g['masks'].shape[1], :g['masks'].shape[2]] = g['masks']
                gt_masks.append(ins_masks)

            loss = self.mask_head.loss(cls_preds, kernel_preds, bbox_preds,
                                       ins_preds, gt_bboxes, gt_labels, gt_masks)

        if not self.training or calc_val_loss:
            detections = self.mask_head.get_results(cls_preds, kernel_preds, bbox_preds, ins_preds,
                                                    images.image_sizes, original_image_sizes)

        return eager_outputs(loss, detections)


class SOLOv2Head(nn.Module):
    """SOLO mask head used in `SOLO: Segmenting Objects by Locations.

    <https://arxiv.org/abs/1912.04488>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
            Default: 256.
        stacked_convs (int): Number of stacking convs of the head.
            Default: 4.
        strides (tuple): Downsample factor of each feature map.
        scale_ranges (tuple[tuple[int, int]]): Area range of multiple
            level masks, in the format [(min1, max1), (min2, max2), ...].
            A range of (16, 64) means the area range between (16, 64).
        pos_scale (float): Constant scale factor to control the center region.
        num_grids (list[int]): Divided image into a uniform grids, each
            feature map has a different grid value. The number of output
            channels is grid ** 2. Default: [40, 36, 24, 16, 12].
        cls_down_index (int): The index of downsample operation in
            classification branch. Default: 0.
        loss_mask (dict): Config of mask loss.
        loss_cls (dict): Config of classification loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
                                   requires_grad=True).
        train_cfg (dict): Training config of head.
        test_cfg (dict): Testing config of head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        kernel_branch=dict(
            feat_channels=512,
            num_grids=[40, 36, 24, 16, 12],
            out_channels=256,
            stacked_cons=4,
            switch_head_ratio=3,
            switch_head_start=500,
        ), # kernel branch cfg
        feature_branch=dict(
            feat_channels=128,
            in_layers=[0, 1, 2, 3],
            out_channels=256
        ),
        in_layers=[0, 1, 2, 3, 4],
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        strides=(8, 8, 16, 32, 32),
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        loss_mask=dict(weight=3.0),
        loss_cls=dict(
            focal_alpha=0.25,
            focal_gamm=2.0,
            weight=1.0
        ),
        test_cfg=dict(
            filter_thr=0.5,
            kernel='gaussian',
            mask_thr=0.5,
            max_per_img=100,
            nms_pre=500,
            score_thr=0.1,
            sigma=2.0
        ),
    ):
        super(SOLOv2Head, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.sigma = sigma
        self.num_grids = num_grids
        self.strides = strides
        self.scale_ranges = scale_ranges
        self.in_layers = in_layers

        self.kernel_head = KernelBranch(self.num_classes, self.in_channels, **kernel_branch)
        self.feature_head = FeatureBranch(self.in_channels, **feature_branch)

        self.ins_loss_weight = loss_mask['weight']

        self.focal_alpha = loss_cls['focal_alpha']
        self.focal_gamma = loss_cls['focal_gamma']

        self.mask_dot_product = MaskDotProduct(out_planes=feature_branch['out_channels'])

        self.hit_indices = torch.zeros(5)
        self.obj_locs = [[] for _ in range(5)]
        self.itr_indx = 0

        self.test_cfg = test_cfg

    def forward(self, feats):

        feats = [feats[i] for i in self.in_layers]
        cate_pred, kernel_pred, bbox_pred = self.kernel_head(feats)
        mask_feat_pred = self.feature_head(feats)

        return cate_pred, kernel_pred, bbox_pred, mask_feat_pred

    def loss(self,
             cls_preds,
             kernel_preds,
             bbox_preds,
             ins_pred,
             gt_bboxes,
             gt_labels,
             gt_masks):

        mask_feat_size = ins_pred.size()[-2:]
        ins_label_list, bbox_label_list, cate_label_list, ins_ind_label_list, grid_order_list = multi_apply(
            self.solov2_target_single,
            gt_bboxes,
            gt_labels,
            gt_masks,
            mask_feat_size=mask_feat_size)

        for i in range(len(cls_preds)):
            assert cate_label_list[0][i].shape == cls_preds[i].shape[2:]
        # ins

        ins_labels = [torch.cat([ins_labels_level_img for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]
        bbox_labels = [torch.cat([bbox_labels_level_img for bbox_labels_level_img in bbox_labels_level], 0)
                       for bbox_labels_level in zip(*bbox_label_list)]

        query_kernels = [kernel_pred[0] for kernel_pred in kernel_preds]
        query_kernels = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(query_kernels, zip(*grid_order_list))]


        bbox_preds = [[bbox_preds_level_img.view(bbox_preds_level_img.shape[0], -1)[:, grid_orders_level_img].T
                       for bbox_preds_level_img, grid_orders_level_img in zip(bbox_preds_level, grid_orders_level)]
                       for bbox_preds_level, grid_orders_level in zip(bbox_preds, zip(*grid_order_list))]


        # generate masks
        ins_pred = ins_pred
        ins_pred_list = []
        for b_query_kernel in query_kernels:
            b_mask_pred = []
            for idx, query_kernel in enumerate(b_query_kernel):

                if query_kernel.size()[-1] == 0:
                    continue

                cur_ins_pred = ins_pred[idx, ...]
                N, I = query_kernel.shape

                query_kernel = query_kernel.permute(1, 0).reshape(I, -1, 1, 1)

                cur_ins_pred = self.mask_dot_product(cur_ins_pred[None, :], query_kernel)
                b_mask_pred.append(cur_ins_pred)

            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        bbox_pred_list = [torch.cat(bbox_pred, 0) if torch.cat(bbox_pred, 0).shape[0] != 0  else None
                          for bbox_pred in bbox_preds]
        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        loss_ins = []
        for input, target in zip(ins_pred_list, ins_labels):
            if input is None:
                continue
            loss_ins.append(dice_loss(input, target))
        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        loss_bbox = []
        for input, target in zip(bbox_pred_list, bbox_labels):
            if input is not None:
                ious, gious = compute_ious(input, target)
                loss_bbox.append(1 - gious)

        loss_bbox = torch.cat(loss_bbox).mean()
        loss_bbox = loss_bbox * 1.5
        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cls_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cate_pred in cls_preds
        ]
        flatten_cate_preds = torch.cat(cls_preds)

        pos_inds = torch.nonzero(flatten_cate_labels != self.num_classes).squeeze(1)
        cls_target = torch.zeros_like(flatten_cate_preds)
        cls_target[pos_inds, flatten_cate_labels[pos_inds]] = 1


        loss_cate = sigmoid_focal_loss(flatten_cate_preds, cls_target,
                                       alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="sum")
        loss_cate = loss_cate / (num_ins + 1)

        return dict(
            loss_ins=loss_ins,
            loss_cls=loss_cate,
            loss_bbox=loss_bbox)

    def solov2_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               mask_feat_size):

        def empty_results(mask_feat_size, ins_label_list, bbox_label_list, cate_label_list,
                          ins_ind_label_list, grid_order_list):
            ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            bbox_label = torch.zeros([0, 4], dtype=torch.uint8, device=device)
            ins_label_list.append(ins_label)
            bbox_label_list.append(bbox_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append([])

        output_stride = 4
        device = gt_labels_raw[0].device
        upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * \
                              (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        # gt_areas_ = gt_masks_raw.sum((1, 2)).sqrt()

        ins_label_list = []
        bbox_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []

        for level_index, ((lower_bound, upper_bound), stride, num_grid) \
                in enumerate(zip(self.scale_ranges, self.strides, self.num_grids)):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)
            self.hit_indices[level_index] += num_ins

            ins_label = []
            bbox_label = []
            grid_order = []

            num_grid_y = round(upsampled_size[0] / num_grid) * 2
            num_grid_x = round(upsampled_size[1] / num_grid) * 2

            grid_size_h = upsampled_size[0] / num_grid_y
            grid_size_w = upsampled_size[1] / num_grid_x

            grid_y_centers = torch.arange(grid_size_h / 2, upsampled_size[0] - 1, grid_size_h)
            grid_x_centers = torch.arange(grid_size_w / 2, upsampled_size[1] - 1, grid_size_w)

            assert len(grid_y_centers) == num_grid_y
            assert len(grid_x_centers) == num_grid_x

            map_y = torch.zeros(upsampled_size)
            map_x = torch.zeros(upsampled_size)

            map_y[grid_y_centers.long(), :] = 1
            map_x[:, grid_x_centers.long()] = 1

            map = torch.logical_and(map_x, map_y)
            grid_centers = torch.stack(torch.meshgrid(grid_y_centers, grid_x_centers), 2)

            # map = map.to(device)

            cate_label = torch.zeros([num_grid_y, num_grid_x], dtype=torch.int64, device=device) + self.num_classes
            ins_ind_label = torch.zeros([num_grid_y * num_grid_x], dtype=torch.bool, device=device)
            ins_dist_matrix = torch.ones([num_grid_y * num_grid_x], device=device) * 10e10

            if num_ins == 0:
                empty_results(mask_feat_size, ins_label_list, bbox_label_list, cate_label_list,
                              ins_ind_label_list, grid_order_list)
                continue

            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices, ...]

            grid_masks = []
            filtered_indices = []
            for i, seg_mask in enumerate(gt_masks):
                seg_cv = seg_mask.cpu().numpy()
                seg_cv = seg_cv.astype('uint8')
                grid_mask = cv2.resize(seg_cv, (len(grid_x_centers), len(grid_y_centers)),
                                       interpolation=cv2.INTER_AREA)
                if (grid_mask > 0).any():
                    grid_masks.append(grid_mask)
                    filtered_indices.append(i)

            if len(filtered_indices) == 0:
                empty_results(mask_feat_size, ins_label_list, bbox_label_list, cate_label_list,
                              ins_ind_label_list, grid_order_list)
                continue

            gt_bboxes = gt_bboxes[filtered_indices]
            gt_labels = gt_labels[filtered_indices]
            gt_masks = gt_masks[filtered_indices, ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            gt_masks_pt = gt_masks.to(device=device)
            center_ws, center_hs = center_of_mass(gt_masks_pt)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

            for seg_mask, grid_mask, gt_bbox, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in\
                    zip(gt_masks, grid_masks, gt_bboxes, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue

                # assign_points = self.grid_mask_assign(grid_mask, upsampled_size, visualize=False)
                assign_points = self.adaptive_location_selection(grid_mask, grid_centers, center_h, center_w,
                                                                 upsampled_size, device)
                # assign_points = self.adaptive_location_selection_maskiou(grid_mask, seg_mask, grid_centers,
                #                                                          center_h, center_w, upsampled_size, device)
                # assign_points = self.adaptive_locations_border(seg_mask, grid_mask, grid_centers, center_h, center_w,
                #                                                num_grid / 2, upsampled_size, device)
                self.obj_locs[level_index].append(len(assign_points))

                # seg_cv = seg_mask.cpu().numpy()
                # self.visualize_assign_points(seg_cv, assign_points, grid_x_centers, grid_y_centers, num_grid)

                h, w = seg_mask.shape[:2]
                seg_mask = cv2.resize(seg_mask.detach().cpu().numpy().astype('uint8'),
                                      (int(w/output_stride), int(h/output_stride)), interpolation=cv2.INTER_LINEAR)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)

                for (i, j) in assign_points:
                    label = int(i * num_grid_x + j)
                    dist = torch.sqrt((grid_x_centers[j] - center_w) ** 2 + (grid_y_centers[i] - center_h) ** 2)
                    if dist < ins_dist_matrix[label]:
                        cate_label[i, j] = gt_label
                        cur_ins_label = torch.zeros([mask_feat_size[0],mask_feat_size[1]],
                                                    dtype=torch.uint8, device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask

                        if ins_ind_label[label]:
                            pre_index = grid_order.index(label)
                            grid_order.pop(pre_index)
                            ins_label.pop(pre_index)
                            bbox_label.pop(pre_index)

                        relative_bbox = ((grid_x_centers[j] - gt_bbox[0])/grid_size_w,
                                         (grid_y_centers[i] - gt_bbox[1])/grid_size_h,
                                         (gt_bbox[2] - grid_x_centers[j])/grid_size_w,
                                         (gt_bbox[3] - grid_y_centers[i])/grid_size_h)

                        relative_bbox = torch.tensor(relative_bbox, device=device)
                        assert (relative_bbox > 0).any()
                        ins_label.append(cur_ins_label)
                        bbox_label.append(relative_bbox)
                        ins_dist_matrix[label] = dist
                        ins_ind_label[label] = True
                        grid_order.append(label)

            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                bbox_label = torch.zeros([0, 4], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
                bbox_label = torch.stack(bbox_label, 0)
            ins_label_list.append(ins_label)
            bbox_label_list.append(bbox_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
            # num_act_grids.append(torch.stack(level_act_grids))

        return ins_label_list, bbox_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def adaptive_location_selection_maskiou(self, grid_mask, seg_mask, grid_centers,
                                            center_h, center_w, upsampled_size, device):
        p = np.nonzero(grid_mask > 0.6)
        if len(p[0]) == 0:
            p = np.nonzero(grid_mask == grid_mask.max())

        locations, _ = self.level_locations_strides(upsampled_size, grid_mask.shape, device)
        locations = locations.reshape(*grid_mask.shape[:2], 2).cpu().int()

        points = torch.from_numpy(np.vstack((p[0], p[1])).transpose())
        point_locations = [locations[point[0], point[1]] for point in points]
        # mask_cv = seg_mask.cpu().numpy()
        mask_ious = []
        for point_location in point_locations:
            mask_iou = self.move_mask_iou(point_location, (center_w, center_h), seg_mask)
            mask_ious.append(mask_iou)

        mask_ious = torch.tensor(mask_ious)
        # highest_ious = mask_ious.sort(descending=True)[0][:int(len(mask_ious) / 2)]
        highest_ious = mask_ious.sort(descending=True)[0][:16]
        iou_thresh = highest_ious.mean() + highest_ious.std()
        # indices = torch.tensor(mask_ious).sort(descending=True)[1][:5]

        assign_points = points[mask_ious > iou_thresh]

        # dists = torch.sqrt(((torch.stack(point_locations) - torch.tensor([center_w, center_h])) ** 2).sum(dim=1))
        all_dists = ((grid_centers - torch.tensor([center_h, center_w])[None, None, :]) ** 2).sum(2).sqrt()
        # close_dists = all_dists.view(-1).sort()[0][:16]
        close_indices = all_dists.view(-1).sort()[1][:16]
        close_grids = grid_centers.reshape(-1, 2)[close_indices]

        mask_ious_ = []
        for point_location in close_grids:
            mask_iou = self.move_mask_iou(torch.stack((point_location[1], point_location[0])), (center_w, center_h), seg_mask)
            mask_ious_.append(mask_iou)

        mask_ious_ = torch.tensor(mask_ious_)
        iou_thresh = mask_ious_.mean() + mask_ious_.std()
        iou_thresh = max(0.30, iou_thresh)
        assign_points = points[mask_ious > iou_thresh]

        # radius = close_dists.mean() - close_dists.std().sqrt()

        # assign_points = points[dists < radius]

        if len(assign_points) == 0:
            # p = np.nonzero(grid_mask == grid_mask.max())
            # assign_points = torch.from_numpy(np.vstack((p[0], p[1])).transpose())
            assign_points = points[mask_ious.argmax()][None, :]
        return assign_points
        return assign_points

    def move_mask_iou(self, point, center, mask):
        # orig_mask = copy.deepcopy(mask)
        # moved_mask = copy.deepcopy(mask)
        orig_mask = mask.clone()
        moved_mask = mask.clone()


        movement = (torch.tensor(center) - point).int()
        if movement[1] > 0:
            moved_mask = F.pad(moved_mask, (0, abs(int(movement[1])), 0, 0))
            orig_mask = F.pad(orig_mask, (abs(int(movement[1])), 0, 0, 0))
            # moved_mask = cv2.copyMakeBorder(moved_mask, 0, abs(int(movement[1])), 0, 0, cv2.BORDER_CONSTANT)
            # orig_mask = cv2.copyMakeBorder(orig_mask, abs(int(movement[1])), 0, 0, 0, cv2.BORDER_CONSTANT)

        else:
            moved_mask = F.pad(moved_mask, (abs(int(movement[1])), 0, 0, 0))
            orig_mask = F.pad(orig_mask, (0, abs(int(movement[1])), 0, 0))
            # moved_mask = cv2.copyMakeBorder(moved_mask, abs(int(movement[1])), 0, 0, 0, cv2.BORDER_CONSTANT)
            # orig_mask = cv2.copyMakeBorder(orig_mask, 0, abs(int(movement[1])), 0, 0, cv2.BORDER_CONSTANT)

        if movement[0] > 0:
            moved_mask = F.pad(moved_mask, (0, 0, 0, abs(int(movement[0]))))
            orig_mask = F.pad(orig_mask, (0, 0, abs(int(movement[0])), 0))
            # moved_mask = cv2.copyMakeBorder(moved_mask, 0, 0, 0, abs(int(movement[0])), cv2.BORDER_CONSTANT)
            # orig_mask = cv2.copyMakeBorder(orig_mask, 0, 0, abs(int(movement[0])), 0, cv2.BORDER_CONSTANT)
        else:
            moved_mask = F.pad(moved_mask, (0, 0, abs(int(movement[0])), 0))
            orig_mask = F.pad(orig_mask, (0, 0, 0, abs(int(movement[0]))))
            # moved_mask = cv2.copyMakeBorder(moved_mask, 0, 0, abs(int(movement[0])), 0, cv2.BORDER_CONSTANT)
            # orig_mask = cv2.copyMakeBorder(orig_mask, 0, 0, 0, abs(int(movement[0])), cv2.BORDER_CONSTANT)

        mask_iou = torch.logical_and(moved_mask, orig_mask).sum() / torch.logical_or(moved_mask, orig_mask).sum()
        return mask_iou

    def adaptive_location_selection(self, grid_mask, grid_centers, center_h, center_w, upsampled_size, device):
        p = np.nonzero(grid_mask > 0.6)
        if len(p[0]) == 0:
            p = np.nonzero(grid_mask == grid_mask.max())

        locations, _ = self.level_locations_strides(upsampled_size, grid_mask.shape, device)
        locations = locations.reshape(*grid_mask.shape[:2], 2).cpu().int()

        points = torch.from_numpy(np.vstack((p[0], p[1])).transpose())
        point_locations = [locations[point[0], point[1]] for point in points]
        dists = torch.sqrt(((torch.stack(point_locations) - torch.tensor([center_w, center_h])) ** 2).sum(dim=1))
        all_dists = ((grid_centers - torch.tensor([center_h, center_w])[None, None, :]) ** 2).sum(2).sqrt()
        close_dists = all_dists.view(-1).sort()[0][:16]
        radius = close_dists.mean() - close_dists.std().sqrt()

        assign_points = points[dists < radius]

        if len(assign_points) == 0:
            p = np.nonzero(grid_mask == grid_mask.max())
            assign_points = torch.from_numpy(np.vstack((p[0], p[1])).transpose())
        return assign_points

    @staticmethod
    def get_border(seg_cv, upsize, visualize=False):
        h, w = seg_cv.shape

        new_mask = cv2.copyMakeBorder(seg_cv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        kernel = np.ones((3, 3), np.uint8)
        seg_cv_erode = cv2.erode(new_mask, kernel, iterations=1)
        border = new_mask - seg_cv_erode

        border = border[1: h + 1, 1: w + 1]

        if visualize:
            border_ = cv2.resize(border, (upsize[1], upsize[0]), cv2.INTER_NEAREST)

            cv2.imshow('border', border_.astype('float'))
            cv2.waitKey()

        return border

    def adaptive_locations_border(self, seg_mask, grid_mask, grid_centers, center_h, center_w, grid_size,
                                  upsampled_size, device):
        p = np.nonzero(grid_mask > 0.6)

        locations, _ = self.level_locations_strides(upsampled_size, grid_mask.shape, device)
        locations = locations.reshape(*grid_mask.shape[:2], 2).cpu().int()

        points = torch.from_numpy(np.vstack((p[0], p[1])).transpose())
        point_locations = [locations[point[0], point[1]] for point in points]
        point_locations = torch.stack(point_locations)
        point_locations = torch.index_select(point_locations, 1, torch.LongTensor([1, 0]))

        seg_cv = seg_mask.cpu().numpy()
        border = self.get_border(seg_cv, upsampled_size)
        b = border.nonzero()
        border_locations = np.stack((b[0], b[1])).transpose()
        border_locations = torch.from_numpy(border_locations)
        dist_table = torch.cdist(point_locations.float(), border_locations.float())

        closest_border = dist_table.min(dim=1)
        thresh = closest_border[0].mean() + closest_border[0].std() + math.sqrt(grid_size)

        assign_points = points[closest_border[0] > thresh]

        if len(assign_points) == 0:
            center_dist = torch.cdist(point_locations.float(), torch.tensor([center_h, center_w])[None, :])
            close_idx = center_dist.argmin()
            assign_points = points[close_idx][None, :]

        return assign_points

    def grid_mask_assign(self, grid_mask, upsize, visualize=False):
        grid_mask = grid_mask.astype('uint8')

        if grid_mask.any():
            border = self.get_border(grid_mask, upsize, visualize)
            erode_mask = center_mask_np(grid_mask, border.sum())
        else:
            erode_mask = grid_mask

        if visualize:
            grid_mask_ = cv2.resize(grid_mask, (upsize[1], upsize[0]), cv2.INTER_NEAREST)
            cv2.imshow('grid_mask_up', grid_mask_*255)

            erode_mask_ = cv2.resize(erode_mask, (upsize[1], upsize[0]), cv2.INTER_NEAREST)
            cv2.imshow('erode_mask_up', erode_mask_*255)
            cv2.waitKey()

        p = np.nonzero(erode_mask)
        if len(p[0]) == 0:
            center = center_of_mass(torch.from_numpy(grid_mask)[None, :])
            p = np.array([[int(center[1].round())], [int(center[0].round())]])
            # p = np.nonzero(grid_mask == grid_mask.max())
        points = torch.from_numpy(np.vstack((p[0], p[1])).transpose())

        return points

    @staticmethod
    def visualize_assign_points(seg_cv, assign_points, grid_x_centers, grid_y_centers, num_grid):
        seg_cv = copy.deepcopy(seg_cv).astype('float')
        assigned_locations = [torch.tensor([grid_y_centers[point[0]], grid_x_centers[point[1]]]) for point in assign_points]
        if len(assign_points) != 0:
            assigned_locations = torch.stack(assigned_locations).int()
            #
            for loc in assigned_locations:
                seg_cv[loc[0]-3:loc[0]+3, loc[1]-3:loc[1]+3] = 0.5

        seg_cv = cv2.putText(seg_cv, '{}'.format(num_grid), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (1, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('assigned', seg_cv)
        cv2.waitKey()


    def get_results(self, cate_preds, kernel_preds, bbox_preds, seg_pred, img_shapes, ori_shapes):
        num_levels = len(cate_preds)
        featmap_size = seg_pred.size()[-2:]

        result_list = []
        for img_id in range(len(img_shapes)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.num_classes).detach() for i in range(num_levels)
            ]
            seg_pred_list = seg_pred[img_id, ...].unsqueeze(0)

            query_kernel_list = [
                kernel_preds[i][0][img_id].permute(1, 2, 0).view(-1, self.kernel_head.kernel_out_channels).detach()
                                for i in range(num_levels)
            ]

            bbox_pred_list = [
                bbox_preds[i][img_id].permute(1, 2, 0).view(-1, 4).detach() for i in range(num_levels)]

            img_shape = img_shapes[img_id]
            # scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = ori_shapes[img_id]

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            query_kernel_list = torch.cat(query_kernel_list, dim=0)
            bbox_pred_list = torch.cat(bbox_pred_list, dim=0)

            cate_preds_img = [cate_pred[img_id] for cate_pred in cate_preds]
            result = self.get_seg_single(cate_pred_list, cate_preds_img, seg_pred_list, query_kernel_list, bbox_pred_list,
                                         featmap_size, img_shape, ori_shape,
                                         self.test_cfg)
            result_list.append(result)
        return result_list

    def get_seg_single(self, cate_preds, cate_preds_orig, seg_preds, query_kernels, bbox_preds,
                       featmap_size, img_shape, ori_shape, cfg):

        def empty_results(results, cls_scores, ori_shape):
            """Generate a empty results."""
            results['scores'] = cls_scores.new_ones(0)
            results['masks'] = cls_scores.new_zeros(0, *ori_shape[:2])
            results['labels'] = cls_scores.new_ones(0)
            results['boxes'] = cls_scores.new_ones(0, 4)
            results['locations'] = cls_scores.new_ones(0, 2)
            return results

        assert len(cate_preds) == len(query_kernels) == len(bbox_preds)
        results = dict()

        # overall info.
        h, w, = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > cfg['score_thr'])
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return empty_results(results, cate_scores, ori_shape)

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        # attn_kernels = attn_kernels[inds[:, 0]]
        query_kernels = query_kernels[inds[:, 0]]
        bbox_preds = bbox_preds[inds[:, 0]]

        # up_kernel_preds = up_kernel_preds[inds[:, 0]]
        locations, bbox_strides = self.get_locations_strides(upsampled_size_out, cate_preds_orig)
        locations = torch.cat(locations, dim=0)[inds[:, 0]]
        bbox_strides = torch.cat(bbox_strides, dim=0)[inds[:, 0]]

        bboxes = [locations[:, 0] - bbox_preds[:, 0] * bbox_strides[:, 0],
                  locations[:, 1] - bbox_preds[:, 1] * bbox_strides[:, 1],
                  locations[:, 0] + bbox_preds[:, 2] * bbox_strides[:, 0],
                  locations[:, 1] + bbox_preds[:, 3] * bbox_strides[:, 1]]

        bboxes = torch.stack(bboxes).T

        # trans vector.
        size_trans = cate_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = query_kernels.new_ones(size_trans[-1])

        n_stage = len(self.num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_-1]:size_trans[ind_]] *= self.strides[ind_]
        # strides = strides[inds]
        # strides = strides[inds[:, 0]]
        strides = (bbox_strides[:, 0] * bbox_strides[:, 1]).sqrt()

        # mask encoding.
        I, N = query_kernels.shape
        query_kernels = query_kernels.reshape(I, -1, 1, 1)
        seg_preds = self.mask_dot_product(seg_preds, query_kernels)

        seg_masks = seg_preds > cfg['mask_thr']
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return empty_results(results, cate_scores, ori_shape)

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]
        bboxes = bboxes[keep]

        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg['nms_pre']:
            sort_inds = sort_inds[:cfg['nms_pre']]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]
        bboxes = bboxes[sort_inds]

        cate_scores, cate_labels, _, keep_inds = mask_matrix_nms(
            seg_masks,
            cate_labels,
            cate_scores,
            mask_area=sum_masks,
            kernel=cfg['kernel'],
            sigma=cfg['sigma'])

        seg_preds = seg_preds[keep_inds, :, :]
        bboxes = bboxes[keep_inds]

        keep = cate_scores >= cfg['filter_thr']
        if keep.sum() == 0:
            return empty_results(results, cate_scores, ori_shape)

        seg_preds = seg_preds[keep, :, :]
        bboxes = bboxes[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg['max_per_img']:
            sort_inds = sort_inds[:cfg['max_per_img']]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]
        bboxes = bboxes[sort_inds]

        # temp_seg = torch.zeros_like(seg_preds)
        # bboxes_ = (bboxes / 4).int()
        # for i in range(len(seg_preds)):
        #     temp_seg[i][bboxes_[i][1]:bboxes_[i][3], bboxes_[i][0]:bboxes_[i][2]] = \
        #         seg_preds[i][bboxes_[i][1]:bboxes_[i][3], bboxes_[i][0]:bboxes_[i][2]]
        #
        # seg_preds = temp_seg

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                    size=upsampled_size_out,
                                    mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                               size=ori_shape[:2],
                               mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg['mask_thr']

        seg_boxes = torch.zeros((seg_masks.shape[0], 4), device=seg_masks.device)
        for i, seg_mask in enumerate(seg_masks):
            if seg_mask.nonzero().shape[0] != 0:
                seg_boxes[i][0] = seg_mask.nonzero()[:, 1].min()
                seg_boxes[i][1] = seg_mask.nonzero()[:, 0].min()
                seg_boxes[i][2] = seg_mask.nonzero()[:, 1].max()
                seg_boxes[i][3] = seg_mask.nonzero()[:, 0].max()

        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * ori_shape[0] / h
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * ori_shape[0] / h

        results['masks'] = seg_masks[:, None, :, :]
        results['boxes'] = bboxes
        results['labels'] = cate_labels
        results['scores'] = cate_scores
        results['locations'] = torch.stack(((bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2)).T

        self.itr_indx += 1

        return results

    def get_locations_strides(self, upsampled_size, cate_preds_orig):
        strides = []
        locations = []
        for cate_pred in cate_preds_orig:
            locations_per_level, level_stride = self.level_locations_strides(upsampled_size, cate_pred.shape,
                                                                             cate_pred.device)
            strides.append(level_stride)
            locations.append(locations_per_level)

        return locations, strides


    def level_locations_strides(self, upsampled_size, pred_shape, device=None):
        grid_size_h = upsampled_size[0] / pred_shape[0]
        grid_size_w = upsampled_size[1] / pred_shape[1]

        grid_y_centers = torch.arange(grid_size_h / 2, upsampled_size[0] - 1, grid_size_h)
        grid_x_centers = torch.arange(grid_size_w / 2, upsampled_size[1] - 1, grid_size_w)

        grid_y_centers, grid_x_centers = torch.meshgrid(grid_y_centers, grid_x_centers)
        locations_per_level = torch.stack([grid_x_centers.reshape(-1), grid_y_centers.reshape(-1)], dim=1)
        if device is not None:
            locations_per_level = locations_per_level.to(device)
            level_stride = torch.tensor([grid_size_w, grid_size_h], device=device)
        level_stride = level_stride[None, :].repeat(pred_shape[0] * pred_shape[1], 1)
        return locations_per_level, level_stride


class KernelBranch(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=512,
                 out_channels=256,
                 stacked_convs=4,
                 num_grids=[40, 36, 24, 16, 12],
                 ):
        super(KernelBranch, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = self.num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = out_channels

        self.num_grids = num_grids

        self.num_levels = len(num_grids)

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        kernel_convs = nn.ModuleList()
        cls_convs = nn.ModuleList()
        bbox_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            norm_layer = nn.GroupNorm(num_groups=32, num_channels=self.feat_channels)
            kernel_convs.append(
                ConvNormAct(chn, self.feat_channels, 3, stride=1, padding=1, norm=norm_layer))

            bbox_convs.append(
                ConvNormAct(chn, self.feat_channels, 3, stride=1, padding=1, norm=norm_layer))

            chn = self.in_channels if i == 0 else self.feat_channels
            norm_layer = nn.GroupNorm(num_groups=32, num_channels=self.feat_channels)
            cls_convs.append(
                ConvNormAct(chn, self.feat_channels, 3, stride=1, padding=1, norm=norm_layer))
            if i == self.stacked_convs // 2:
                cls_convs.append(nn.Upsample(scale_factor=2))
                bbox_convs.append(nn.Upsample(scale_factor=2))
                kernel_convs.append(nn.Upsample(scale_factor=2))

        solo_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

        solo_kernel = nn.Conv2d(
            self.feat_channels, self.kernel_out_channels, 3, padding=1)

        solo_bbox = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

        self.cls_branch = nn.Sequential(*cls_convs, solo_cls)
        self.ins_branch = nn.Sequential(*kernel_convs[:2])

        # self.kernel_head = solo_kernel
        # self.bbox_head = solo_bbox
        self.kernel_branch = nn.Sequential(*kernel_convs[2:], solo_kernel)
        self.bbox_branch = nn.Sequential(*bbox_convs[2:], solo_bbox)

        # self.kernel_branch = nn.Sequential(*kernel_convs, solo_kernel)
        # self.bbox_branch = nn.Sequential(*kernel_convs, solo_bbox)


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, ConvNormAct):
                normal_init(m.conv, std=0.01)

        # for m in self.cls_branch[:-1]:
        #     normal_init(m.conv, std=0.01)
        # for m in self.kernel_branch[:-1]:
        #     normal_init(m.conv, std=0.01)
        # for m in self.bbox_branch[:-1]:
        #     normal_init(m.conv, std=0.01)

        bias_cate = float(-np.log((1 - 0.01) / 0.01))
        normal_init(self.cls_branch[-1], std=0.01, bias=bias_cate)
        normal_init(self.bbox_branch[-1], std=0.01, bias=0.)
        normal_init(self.kernel_branch[-1], std=0.01)
        # normal_init(self.solo_up_kernel, std=0.01)


    def forward(self, feats, eval=False):

        assert len(feats) == self.num_levels
        # new_feats = self.resize_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in feats]
        upsampled_size = (featmap_sizes[0][0], featmap_sizes[0][1])
        cate_pred, kernel_pred, bbox_pred = multi_apply(self.forward_single, feats,
                                                       list(range(len(self.num_grids))),
                                                       eval=eval, upsampled_size=upsampled_size)
        return cate_pred, kernel_pred, bbox_pred

    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        eval = not self.training
        ins_kernel_feat = x
        # ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

        # kernel branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.num_grids[idx]

        grid_y = round(upsampled_size[0]*4 / seg_num_grid)
        grid_x = round(upsampled_size[1]*4 / seg_num_grid)

        kernel_feat = F.interpolate(kernel_feat, size=(grid_y, grid_x), mode='bilinear')

        cate_feat = kernel_feat[:, :-2, :, :]

        # cate branch
        cate_feat = cate_feat.contiguous()
        cate_pred = self.cls_branch(cate_feat)

        kernel_feat = kernel_feat.contiguous()
        ins_feat = self.ins_branch(kernel_feat)

        kernel_pred = self.kernel_branch(ins_feat)
        bbox_pred = self.bbox_branch(ins_feat).relu()

        # bbox_pred = self.bbox_head(ins_feat).relu()
        # kernel_pred = self.kernel_head(ins_feat)

        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_pred, [kernel_pred], bbox_pred

    @staticmethod
    def resize_feats(feats):
        """Downsample the first feat and upsample last feat in feats."""
        out = []
        for i in range(len(feats)):
            if i == 0:
                out.append(
                    F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'))
            elif i == len(feats) - 1:
                out.append(
                    F.interpolate(
                        feats[i],
                        size=feats[i - 1].shape[-2:],
                        mode='bilinear'))
            else:
                out.append(feats[i])
        return out


class FeatureBranch(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=256,
                 feat_channels=128,
                 in_layers=[0, 1, 2, 3]):
        super(FeatureBranch, self).__init__()

        self.mask_num_classes = out_channels
        self.mask_out_channels = feat_channels
        self.in_channels = in_channels
        self.in_layers = in_layers

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        # norm_layer = nn.GroupNorm(num_groups=32, requires_grad=True)
        self.mask_convs_all_levels = nn.ModuleList()
        for i in self.in_layers:
            convs_per_level = nn.Sequential()
            if i == 0:
                norm_layer = nn.GroupNorm(num_groups=32, num_channels=self.mask_out_channels)
                one_conv = ConvNormAct(self.in_channels, self.mask_out_channels, 3, padding=1, norm=norm_layer)
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.mask_convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.in_channels+2 if i==3 else self.in_channels
                    norm_layer = nn.GroupNorm(num_groups=32, num_channels=self.mask_out_channels)
                    one_conv = ConvNormAct(chn, self.mask_out_channels, 3, padding=1, norm=norm_layer)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), one_upsample)
                    continue

                norm_layer = nn.GroupNorm(num_groups=32, num_channels=self.mask_out_channels)
                one_conv = ConvNormAct(self.mask_out_channels, self.mask_out_channels, 3, padding=1, norm=norm_layer)
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)

            self.mask_convs_all_levels.append(convs_per_level)

        norm_layer = nn.GroupNorm(num_groups=32, num_channels=self.mask_num_classes)
        self.mask_conv_pred = nn.Sequential(
            ConvNormAct(self.mask_out_channels, self.mask_num_classes, 1, padding=0, norm=norm_layer),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, ConvNormAct):
                normal_init(m.conv, std=0.01)


    def forward(self, inputs):

        inputs =  [inputs[i] for i in self.in_layers]

        feature_add_all_level = self.mask_convs_all_levels[0](inputs[0])
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == 3:
                input_feat = input_p
                x_range = torch.linspace(-1, 1, input_feat.shape[-1], device=input_feat.device)
                y_range = torch.linspace(-1, 1, input_feat.shape[-2], device=input_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([input_feat.shape[0], 1, -1, -1])
                x = x.expand([input_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                input_p = torch.cat([input_p, coord_feat], 1)

            feature_add_all_level = feature_add_all_level + self.mask_convs_all_levels[i](input_p)

        feature_pred = self.mask_conv_pred(feature_add_all_level)
        return feature_pred

class MaskDotProduct(nn.Module):
    def __init__(self, out_planes):
        super().__init__()
        self.query_norm = nn.LayerNorm(out_planes)

    def forward(self, x, query_kernel):
        bs_, in_planels, h, w = x.shape
        bs = query_kernel.shape[0]
        x = x.view(1, -1, h, w)
        query_kernel = self.query_norm(query_kernel[:, :, 0, 0])[:, :, None, None]
        seg_pred = F.conv2d(x, query_kernel)[0]

        seg_pred = seg_pred.sigmoid()
        return seg_pred


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding, norm, stride=1, act='relu', bias=False):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=bias)
        self.norm = copy.deepcopy(norm)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def mask_matrix_nms(masks,
                    labels,
                    scores,
                    filter_thr=-1,
                    nms_pre=-1,
                    max_num=-1,
                    kernel='gaussian',
                    sigma=2.0,
                    mask_area=None):
    """Matrix NMS for multi-class masks.

    Args:
        masks (Tensor): Has shape (num_instances, h, w)
        labels (Tensor): Labels of corresponding masks,
            has shape (num_instances,).
        scores (Tensor): Mask scores of corresponding masks,
            has shape (num_instances).
        filter_thr (float): Score threshold to filter the masks
            after matrix nms. Default: -1, which means do not
            use filter_thr.
        nms_pre (int): The max number of instances to do the matrix nms.
            Default: -1, which means do not use nms_pre.
        max_num (int, optional): If there are more than max_num masks after
            matrix, only top max_num will be kept. Default: -1, which means
            do not use max_num.
        kernel (str): 'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
        mask_area (Tensor): The sum of seg_masks.

    Returns:
        tuple(Tensor): Processed mask results.

            - scores (Tensor): Updated scores, has shape (n,).
            - labels (Tensor): Remained labels, has shape (n,).
            - masks (Tensor): Remained masks, has shape (n, w, h).
            - keep_inds (Tensor): The indices number of
                the remaining mask in the input mask, has shape (n,).
    """
    assert len(labels) == len(masks) == len(scores)
    if len(labels) == 0:
        return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
            0, *masks.shape[-2:]), labels.new_zeros(0)
    if mask_area is None:
        mask_area = masks.sum((1, 2)).float()
    else:
        assert len(masks) == len(mask_area)

    # sort and keep top nms_pre
    scores, sort_inds = torch.sort(scores, descending=True)

    keep_inds = sort_inds
    if nms_pre > 0 and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    flatten_masks = masks.reshape(num_masks, -1).float()
    # inter.
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    # Upper triangle iou matrix.
    iou_matrix = (inter_matrix /
                  (expanded_mask_area + expanded_mask_area.transpose(1, 0) -
                   inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    expanded_labels = labels.expand(num_masks, num_masks)
    # Upper triangle label matrix.
    label_matrix = (expanded_labels == expanded_labels.transpose(
        1, 0)).triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num_masks,
                                           num_masks).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # Calculate the decay_coefficient
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(
            f'{kernel} kernel is not supported in matrix nms!')
    # update the score.
    scores = scores * decay_coefficient

    if filter_thr > 0:
        keep = scores >= filter_thr
        keep_inds = keep_inds[keep]
        if not keep.any():
            return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
                0, *masks.shape[-2:]), labels.new_zeros(0)
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    # sort and keep top max_num
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if max_num > 0 and len(sort_inds) > max_num:
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()

    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y

def center_mask_np(bitmask, border_size):
    h, w = bitmask.shape

    ys = np.arange(0, h).astype('float')
    xs = np.arange(0, w).astype('float')

    x_r = bitmask.sum(axis=0)
    y_r = bitmask.sum(axis=1)

    mask_radius_x = x_r[x_r != 0].mean() / 2
    mask_radius_y = y_r[y_r != 0].mean() / 2

    new_mask = cv2.copyMakeBorder(bitmask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    kernel = np.ones((3, 3), np.uint8)
    num_iter = bitmask.sum() / border_size
    radius = min(max(mask_radius_y/2, 1), max(mask_radius_x/2, 1))
    # seg_cv_erode = cv2.erode(new_mask, kernel, iterations=int(round(radius)))
    seg_cv_erode = cv2.erode(new_mask, kernel, iterations=round(num_iter))

    seg_cv_erode = seg_cv_erode[1: h + 1, 1: w + 1]

    return seg_cv_erode

def center_mask(bitmask):
    h, w = bitmask.shape


    ys = torch.arange(0, h, dtype=torch.float32, device=bitmask.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmask.device)

    x_r = bitmask.sum(dim=0)
    y_r = bitmask.sum(dim=1)

    mask_radius_x = x_r[x_r != 0].mean() / 2
    mask_radius_y = y_r[y_r != 0].mean() / 2

    seg_cv = bitmask.cpu().numpy().astype('uint8')

    # img_diag = np.sqrt(h ** 2 + w ** 2)
    # dilation = int(round(0.02 * img_diag))
    # if dilation < 1:
    #     dilation = 1

    # kernel = np.ones((int(mask_radius_y * 0.8), int(mask_radius_x * 0.8)), np.uint8)
    # kernel = np.ones((int(6 * (mask_radius_y / (mask_radius_x + mask_radius_y))),
    #                   int(6 * (mask_radius_x / (mask_radius_x + mask_radius_y)))), np.uint8)
    # seg_cv_erode = cv2.erode(seg_cv, kernel, iterations=int(((mask_radius_x + mask_radius_y)/2) * 0.5))
    kernel = np.ones((3, 3), np.uint8)
    seg_cv_erode = cv2.erode(seg_cv, kernel, iterations=5)
    # seg_cv_erode = cv2.erode(seg_cv, kernel, iterations=1)

    x_c = (bitmask * xs).mean(dim=0)
    y_c = (bitmask * ys[:, None]).mean(dim=1)

    return seg_cv_erode


def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d

def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
