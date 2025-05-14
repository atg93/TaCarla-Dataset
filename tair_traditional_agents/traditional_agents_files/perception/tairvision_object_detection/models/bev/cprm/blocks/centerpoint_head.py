import copy
import torch
import numpy as np
from tairvision.models.bev.cprm.training.losses import GaussianFocalLoss, L1Loss
import sys

try:
    sys.path.insert(0, "/usr/lib/python3.8/site-packages/tairbackend-1.0.0-py3.8-linux-aarch64.egg")
except:
    pass

try:
    import tairbackend._ext
except:
    print("install tairbackend._ext")


class CenterHead(torch.nn.Module):
    """CenterHead for CenterPoint.
    """

    def __init__(self,
                 cfg=None,
                 num_heatmap_convs=2,
                 task_specific=True,
                 loss_prefix='loss_'):
        super(CenterHead, self).__init__()
        self.use_lidar_head = cfg.USE_LIDAR_HEAD
        use_velo = cfg.MODEL.CPHEAD.USE_VELOCITY

        # BEV SPACE CONFIGURATION
        xb = cfg.LIFT.X_BOUND
        yb = cfg.LIFT.Y_BOUND
        zb = cfg.LIFT.Z_BOUND
        bev_front = (xb[1] - xb[0]) / xb[2]
        bev_side = (yb[1] - yb[0]) / yb[2]
        point_cloud_range = [xb[0], yb[0], zb[0], xb[1], yb[1], zb[1]]  #[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        voxel_size = [xb[2], yb[2], zb[2]]  # zb wont be used anyways  #[0.1, 0.1, 0.2]
        loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # , 1.0, 1.0] # Uncomment when velocity arrives.

        # HEAD STRUCTURE CONFIGURATION
        nms_rescale_factor = cfg.MODEL.CPHEAD.NMS_RESCALE_FACTOR
        nms_type = cfg.MODEL.CPHEAD.NMS_TYPE
        nms_thr = cfg.MODEL.CPHEAD.NMS_THR
        min_radius = cfg.MODEL.CPHEAD.MIN_RADIUS
        assert len(nms_type) == len(nms_thr) == len(min_radius) == len(nms_rescale_factor)
        # This is to adapt configurable head structure with existing code, class names play no role
        tasks = [{"num_class": 1, "class_names": ['class0']} if isinstance(x, float)
                else {"num_class": len(x), "class_names": [f"class{i}" for i in range(len(x))]}
                for x in nms_rescale_factor]

        # VELOCITY CONFIGURATION FOR FUTURE USAGE
        if use_velo:
            code_size = 9
            common_heads = dict(
                reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2) , vel=(2, 2))
        else:
            code_size = 7
            common_heads = dict(
                reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2))  # , vel=(2, 2))
        share_conv_in_channel = cfg.MODEL.ENCODER.OUT_CHANNELS
        share_conv_out_channel = 64
        bbox_coder = dict(
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            voxel_size=voxel_size,
            code_size=code_size)
        separate_head = dict(init_bias=-2.19, final_kernel=3)
        loss_cls = dict(reduction='mean')
        norm_bbox = True
        train_cfg = dict(
            point_cloud_range=point_cloud_range,
            grid_size=[bev_front, bev_side],  # , 40],
            voxel_size=voxel_size,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=loss_weights)
        test_cfg = dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,  # not used
            max_pool_nms=False,
            min_radius=min_radius,
            score_threshold=0.1,
            voxel_size=voxel_size,
            pre_max_size=1000,
            post_max_size=83,
            nms_type=nms_type,
            nms_thr=nms_thr,
            nms_rescale_factor=nms_rescale_factor
        )

        self.use_velo = use_velo
        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox
        self.loss_cls = GaussianFocalLoss(**loss_cls)
        self.loss_bbox = L1Loss(reduction='sum', loss_weight=0.25)  # TODO: ablate this
        self.bbox_coder = CenterPointBBoxCoder(**bbox_coder)
        self.num_anchor_per_locs = [n for n in num_classes]
        self.fp16_enabled = False

        # a shared convolution
        requires_grad = True
        conv = [torch.nn.Conv2d(share_conv_in_channel, share_conv_out_channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)]
        bnlayer = torch.nn.BatchNorm2d(share_conv_out_channel, eps=1e-05)
        for param in bnlayer.parameters():
            param.requires_grad = requires_grad
        conv.append(bnlayer)
        conv.append(torch.nn.ReLU(inplace=True))
        self.shared_conv = torch.nn.Sequential(*conv)
        for layer in self.shared_conv.children():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, 1)
                torch.nn.init.constant_(layer.bias, 0)

        self.task_heads = torch.nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            pre_dict = dict(in_channels=share_conv_out_channel, heads=heads, num_cls=num_cls)
            separate_head = {**separate_head, **pre_dict}
            self.task_heads.append(SeparateHead(**separate_head))

        self.task_specific = task_specific
        self.loss_prefix = loss_prefix
        self.enterCount = 0

    def forward(self, feats):
        """Forward pass.
        """
        ret_dicts = []
        feats = self.shared_conv(feats)

        for task in self.task_heads:
            ret_dicts.append(task(feats))

        return ret_dicts

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_targets(self, gts):
        """Generate targets.
        """

        all_heatmaps, all_anno_boxes, all_inds, all_masks = [], [], [], []
        for target in gts:
            gt_bbox_3d = target['boxes_3d']
            gt_label_3d = target['labels_3d']
            device = gt_label_3d.device
            gt_bbox_3d = gt_bbox_3d.to(device)
            max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
            #grid_size = torch.tensor(self.train_cfg['grid_size'])
            grid_size = self.train_cfg['grid_size']
            grid_size[0] = int(grid_size[0])
            grid_size[1] = int(grid_size[1])
            feature_map_size = grid_size[:2]

            # reorganize the gt_dict by tasks
            task_masks = []
            flag = 1
            for class_name in self.class_names:
                task_masks.append([torch.where(gt_label_3d == class_name.index(i) + flag) for i in class_name])
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                for m in mask:
                    task_box.append(gt_bbox_3d[m])
                    task_class.append(gt_label_3d[m] - flag2)
                task_boxes.append(torch.cat(task_box, axis=0).to(device))
                task_classes.append(torch.cat(task_class).long().to(device))
                flag2 += len(mask)
            draw_gaussian = draw_heatmap_gaussian
            heatmaps, anno_boxes, inds, masks = [], [], [], []

            for idx, task_head in enumerate(self.task_heads):
                heatmap = gt_bbox_3d.new_zeros((len(self.class_names[idx]), feature_map_size[1], feature_map_size[0]))
                if self.use_velo:
                    anno_box = gt_bbox_3d.new_zeros((max_objs, 10), dtype=torch.float32)
                else:
                    anno_box = gt_bbox_3d.new_zeros((max_objs, 8), dtype=torch.float32)

                ind = gt_label_3d.new_zeros((max_objs), dtype=torch.int64)
                mask = gt_bbox_3d.new_zeros((max_objs), dtype=torch.uint8)

                num_objs = min(task_boxes[idx].shape[0], max_objs)

                for k in range(num_objs):
                    cls_id = task_classes[idx][k] - 1

                    width = task_boxes[idx][k][3]
                    length = task_boxes[idx][k][4]
                    # Don't do these while using view matrix

                    if width > 0 and length > 0:
                        radius = gaussian_radius(
                            (length, width),
                            min_overlap=self.train_cfg['gaussian_overlap'])
                        radius = max(self.train_cfg['min_radius'], int(radius))

                        # be really careful for the coordinate system of
                        # your box annotation.
                        x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                            1], task_boxes[idx][k][2]

                        # THIS IS INTENTIONAL COORD SWITCH
                        # Don't do this while using view matrix
                        coor_x = x
                        coor_y = y

                        #Do this while using view matrix


                        center = torch.tensor([coor_x, coor_y],
                                              dtype=torch.float32,
                                              device=device)
                        center_int = center.to(torch.int32)

                        # throw out not in range objects to avoid out of array
                        # area when creating the heatmap
                        if not (0 <= center_int[0] < feature_map_size[0]
                                and 0 <= center_int[1] < feature_map_size[1]):
                            continue

                        draw_gaussian(heatmap[cls_id], center_int, radius)

                        new_idx = k
                        x, y = center_int[0], center_int[1]

                        assert (y * feature_map_size[0] + x < feature_map_size[0] * feature_map_size[1])

                        ind[new_idx] = y * feature_map_size[0] + x
                        mask[new_idx] = 1
                        rot = task_boxes[idx][k][6]
                        box_dim = task_boxes[idx][k][3:6]
                        if self.norm_bbox:
                            box_dim = box_dim.log()
                        if self.use_velo:
                            vx, vy = task_boxes[idx][k][7:]
                            anno_box[new_idx] = torch.cat([
                                center - torch.tensor([x, y], device=device),
                                z.unsqueeze(0), box_dim,
                                torch.sin(rot).unsqueeze(0),
                                torch.cos(rot).unsqueeze(0),
                                vx.unsqueeze(0),
                                vy.unsqueeze(0)
                            ])
                        else:
                            anno_box[new_idx] = torch.cat([
                                center - torch.tensor([x, y], device=device),
                                z.unsqueeze(0), box_dim,
                                torch.sin(rot).unsqueeze(0),
                                torch.cos(rot).unsqueeze(0)
                            ])

                # A list for each task head, normally len 6
                heatmaps.append(heatmap)
                anno_boxes.append(anno_box)
                masks.append(mask)
                inds.append(ind)
            # A list for each batch element, probably 8
            all_heatmaps.append(heatmaps)
            all_anno_boxes.append(anno_boxes)
            all_masks.append(masks)
            all_inds.append(inds)

        # Interchange inner list of per task head with outer list of batch
        # Create a list of task heads outs for entire batch
        # Then stack batch results, resulting in a list of task heads with tensors with dim Batch x bla x bla
        # Transpose heatmaps
        all_heatmaps = list(map(list, zip(*all_heatmaps)))
        all_heatmaps = [torch.stack(hms_) for hms_ in all_heatmaps]
        # Transpose anno_boxes
        all_anno_boxes = list(map(list, zip(*all_anno_boxes)))
        all_anno_boxes = [torch.stack(all_anno_boxes_) for all_anno_boxes_ in all_anno_boxes]
        # Transpose inds
        all_inds = list(map(list, zip(*all_inds)))
        all_inds = [torch.stack(all_inds_) for all_inds_ in all_inds]
        # Transpose masks
        all_masks = list(map(list, zip(*all_masks)))
        all_masks = [torch.stack(all_masks_) for all_masks_ in all_masks]
        return all_heatmaps, all_anno_boxes, all_inds, all_masks

    def loss(self, gts, preds_dicts, **kwargs):
        """Loss function for CenterHead.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(gts)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['heatmap'] = torch.clamp(preds_dict['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(preds_dict['heatmap'], heatmaps[task_id], avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            if self.use_velo:
                preds_dict['anno_box'] = torch.cat(
                    (preds_dict['reg'], preds_dict['height'],
                     preds_dict['dim'], preds_dict['rot'], preds_dict['vel']),
                    dim=1)
            else:
                preds_dict['anno_box'] = torch.cat(
                    (preds_dict['reg'], preds_dict['height'],
                     preds_dict['dim'], preds_dict['rot']),  # , preds_dict['vel']),
                    dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan
            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            if self.task_specific:
                if self.use_velo:
                    name_list = ['xy', 'z', 'whl', 'yaw', 'vel']
                    clip_index = [0, 2, 3, 6, 8, 10]
                else:
                    name_list = ['xy', 'z', 'whl', 'yaw']  # , 'vel']
                    clip_index = [0, 2, 3, 6, 8]  # , 10]
                for reg_task_id in range(len(name_list)):
                    pred_tmp = pred[..., clip_index[reg_task_id]:clip_index[reg_task_id + 1]]
                    target_box_tmp = target_box[..., clip_index[reg_task_id]:clip_index[reg_task_id + 1]]
                    bbox_weights_tmp = bbox_weights[..., clip_index[reg_task_id]:clip_index[reg_task_id + 1]]
                    loss_bbox_tmp = self.loss_bbox(pred_tmp, target_box_tmp, bbox_weights_tmp, avg_factor=(num + 1e-4))
                    loss_dict[f'%stask{task_id}.loss_%s' % (self.loss_prefix, name_list[reg_task_id])] = loss_bbox_tmp
            else:
                loss_bbox = self.loss_bbox(pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
                loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            loss_dict[f'%stask{task_id}.loss_heatmap' % (self.loss_prefix)] = loss_heatmap

        return loss_dict

    def get_bboxes(self, preds_dicts):
        """Generate bboxes from bbox head predictions.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict['heatmap'].shape[0]
            batch_heatmap = preds_dict['heatmap'].sigmoid()

            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict['dim'])
            else:
                batch_dim = preds_dict['dim']

            batch_rots = preds_dict['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            nms_type = self.test_cfg.get('nms_type')
            if isinstance(nms_type, list):
                nms_type = nms_type[task_id]
            if nms_type == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, task_id))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
            bbox_list = [dict(boxes_3d=bboxes.to('cpu').detach(), scores_3d=scores.cpu().detach(),
                              labels_3d=labels.cpu().detach())
                         for bboxes, scores, labels in ret_list]

        return bbox_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, task_id):
        """Rotate nms for each task.
        """
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(post_center_range, dtype=batch_reg_preds[0].dtype,
                                             device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            nms_rescale_factor = self.test_cfg.get('nms_rescale_factor', [1.0 for _ in range(len(self.task_heads))])[
                task_id]
            if isinstance(nms_rescale_factor, list):
                for cid in range(len(nms_rescale_factor)):
                    box_preds[cls_labels == cid, 3:6] = box_preds[cls_labels == cid, 3:6] * nms_rescale_factor[cid]
            else:
                box_preds[:, 3:6] = box_preds[:, 3:6] * nms_rescale_factor

            # Apply NMS in birdeye view
            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(cls_preds.shape[0], device=cls_preds.device, dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg['score_threshold']],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                box_bev = box_preds[:, [0, 1, 3, 4, 6]]
                boxes_for_nms = xywhr2xyxyr(box_bev)
                # the nms in 3d detection just remove overlap boxes.
                if isinstance(self.test_cfg['nms_thr'], list):
                    nms_thresh = self.test_cfg['nms_thr'][task_id]
                else:
                    nms_thresh = self.test_cfg['nms_thr'][0][task_id]
                selected = nms_gpu(boxes_for_nms, top_scores, thresh=nms_thresh,
                                   pre_maxsize=self.test_cfg['pre_max_size'],
                                   post_max_size=self.test_cfg['post_max_size'])
            else:
                selected = []

            if isinstance(nms_rescale_factor, list):
                for cid in range(len(nms_rescale_factor)):
                    box_preds[top_labels == cid, 3:6] = box_preds[top_labels == cid, 3:6] / nms_rescale_factor[cid]
            else:
                box_preds[:, 3:6] = box_preds[:, 3:6] / nms_rescale_factor

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                predictions_dict = dict(bboxes=final_box_preds, scores=final_scores, labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size], dtype=dtype, device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0], dtype=top_labels.dtype, device=device))

            predictions_dicts.append(predictions_dict)

        return predictions_dicts




class SeparateHead(torch.nn.Module):
    """SeparateHead for CenterHead."""

    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(SeparateHead, self).__init__()
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                requires_grad = True
                conv_layers = [torch.nn.Conv2d(c_in, 64, kernel_size=final_kernel, stride=1,
                                               padding=final_kernel // 2, dilation=1, groups=1, bias=False)]
                c_in = head_conv
                bnlayer = torch.nn.BatchNorm2d(64, eps=1e-05)
                for param in bnlayer.parameters():
                    param.requires_grad = requires_grad
                conv_layers.append(bnlayer)
                conv_layers.append(torch.nn.ReLU(inplace=True))

            conv_layers.append(torch.nn.Conv2d(c_in, classes, kernel_size=final_kernel, stride=1,
                                               padding=final_kernel // 2, dilation=1, groups=1, bias=True))
            conv_layers = torch.nn.Sequential(*conv_layers)

            for layer in conv_layers.children():
                if isinstance(layer, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_out', nonlinearity='relu')
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        if 'heatmap' in head:
                            bias = self.init_bias
                        else:
                            bias = 0
                        torch.nn.init.constant_(layer.bias, bias)
                elif isinstance(layer, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)

            self.__setattr__(head, conv_layers)

    def forward(self, x):
        """Forward function for SepHead."""

        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


class CenterPointBBoxCoder:
    """Bbox coder for CenterPoint.
    """

    def __init__(self,
                 pc_range,
                 voxel_size,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 code_size=9):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.code_size = code_size

        # New with view matrix usage_
        self.post_center_range[0:2] = [(self.post_center_range[0] - self.pc_range[0]) / self.voxel_size[0],
                                       (self.post_center_range[1] - self.pc_range[1]) / self.voxel_size[1]]
        self.post_center_range[3:5] = [(self.post_center_range[3] - self.pc_range[0]) / self.voxel_size[0],
                                       (self.post_center_range[4] - self.pc_range[1]) / self.voxel_size[1]]

    def _gather_feat(self, feats, inds, feat_masks=None):
        """Given feats and indexes, returns the gathered feats.
        """
        dim = feats.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _topk(self, scores, K=80):
        """Get indexes based on scores.
        """
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds.float() /
                   torch.tensor(width, dtype=torch.float)).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1),
                                      topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1),
                                    topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1),
                                    topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(self, feat, ind):
        """Given feats and indexes, returns the transposed and gathered feats.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def encode(self):
        pass

    def decode(self,
               heat,
               rot_sine,
               rot_cosine,
               hei,
               dim,
               vel,
               reg=None,
               task_id=-1):
        """Decode bboxes.
        """
        batch, cat, _, _ = heat.size()

        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)

        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, self.max_num, 2)
            xs = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5

        # rotation value and direction label
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.view(batch, self.max_num, 1)

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)

        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.max_num, 1)

        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.max_num, 3)

        # class label
        clses = clses.view(batch, self.max_num).float()
        scores = scores.view(batch, self.max_num)

        #  NOTE: We do not do resizing and shifting here, instead it'll be done via inverse view matrix.
        xs = xs.view(batch, self.max_num, 1)
        ys = ys.view(batch, self.max_num, 1)


        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:  # exist velocity, nuscene format
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, self.max_num, 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)

        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=heat.device)
            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap."""
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
        radius - left:radius + right]).to(heatmap.device,
                                          torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian."""
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def circle_nms(dets, thresh, post_max_size=83):
    """Circular NMS.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[
            i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1
    return keep[:post_max_size]


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.
    """
    order = scores.sort(0, descending=True)[1]

    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()
    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = tairbackend._ext.nms_gpu(boxes, keep, thresh, boxes.device.index)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def gaussian_2d(shape, sigma=1):
    """Generate gaussian map."""

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format"""

    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes