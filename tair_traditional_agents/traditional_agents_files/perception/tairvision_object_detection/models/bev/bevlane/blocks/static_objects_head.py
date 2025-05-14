import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
import networkx as nx
from einops import rearrange

from line_nms import line_nms
from point_nms import point_nms
from scipy.interpolate import interp1d
from tairvision.ops import sigmoid_focal_loss
from abc import ABC, abstractmethod

from tairvision.models.bev.bevlane.blocks.optimal_transport import OptimalTransport
from tairvision.models.bev.bevlane.utils.transformer_utils import MultiHeadAttention, PositionEmbeddingSine
from tairvision.models.bev.common.utils.geometry import points_lidar2view, sample_polyline_points
from tairvision.models.bev.bevlane.utils.assigner import Assigner


class StaticSegmentationHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(StaticSegmentationHead, self).__init__()
        self.hidden_dim = 256
        self.num_classes = 2

        self.cls_thresh = 0.1
        self.adj_thresh = 0.05
        self.lane_thresh = 0.01

        x_bound = cfg['LIFT']['X_BOUND']
        y_bound = cfg['LIFT']['Y_BOUND']
        # Fiery accepts x forward, y side, but we assume y forward x side, for this reason we swap x and y in init
        self.anchor_points, self.anchor_y_points, self.anchor_x_points = self.create_anchors(y_bound, x_bound)
        self.x_bound, self.y_bound = y_bound, x_bound

        self.reduce_dim = ConvNormAct(in_channels, self.hidden_dim, kernel_size=1, padding=0,
                                      batch_norm=True, bias=False)

        self.prior_head = ConvClsRegDirHead(in_channels=self.hidden_dim, num_classes=2, num_layers=1)
        self.final_head = FcClsRegDirHead(in_channels=self.hidden_dim, num_classes=2, num_layers=1)

        self.attn_head = LocalAttention(5, self.hidden_dim)
        self.optimal_transport = OptimalTransport(x_slide=self.x_bound[2], y_slide=self.y_bound[2])

        self.assigner = Assigner(cls_alpha=0.25, cls_gamma=2.0, x_thresh=self.x_bound[2] * 2,
                                 y_thresh=self.y_bound[2] * 2)

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

        self.cls_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.reg_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.cls_final_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.direction_final_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.direction_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.adj_final_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.reg_final_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    @staticmethod
    def create_anchors(x_bound, y_bound):
        anchor_x_points = torch.arange(x_bound[0] + x_bound[2] / 2, x_bound[1] - x_bound[2] / 2 + 1e-9, x_bound[2])
        anchor_y_points = torch.arange(y_bound[0] + y_bound[2] / 2, y_bound[1] - y_bound[2] / 2 + 1e-9, y_bound[2])
        anchor_y_points = anchor_y_points.flip(dims=(0,))
        anchor_x_points = anchor_x_points.flip(dims=(0,))
        anchor_points = torch.stack(torch.meshgrid(anchor_y_points, anchor_x_points), 2)
        return anchor_points, anchor_y_points, anchor_x_points

    def get_head_outputs(self, feats_dec):
        x = feats_dec['y'][:, 0, ...]
        x = self.reduce_dim(x)

        cls_probs, xy_offsets, direction_probs = self.prior_head(x)

        direction_pred = direction_probs.argmax(dim=1)
        node_positions = self.offsets_2_positions(xy_offsets, direction_pred)

        exist_probs = cls_probs.max(dim=1)[0]
        pos_pre_nodes = self.point_nms(exist_probs, node_positions, overlap_x=self.x_bound[2],
                                       overlap_y=self.y_bound[2] / 2)

        x_att, active_nodes = self.attn_head(x, pos_pre_nodes)
        cls_probs_final, xy_offsets_final, direction_probs_final, edge_feats_final = \
            self.final_head(x_att, active_points=active_nodes)

        direction_pred_final = direction_probs_final.argmax(dim=1)

        exist_probs_final = cls_probs_final.max(dim=1)[0]
        exist_probs_final[active_nodes.logical_not()] = inv_sigmoid(torch.zeros(1).to(exist_probs_final.device))

        node_positions_final = self.offsets_2_positions(xy_offsets_final, direction_pred_final)

        pos_post_nodes = self.point_nms(exist_probs_final, node_positions_final,
                                        overlap_x=self.x_bound[2] - 1e-5, overlap_y=self.y_bound[2] - 1e-5)

        output = dict(
            cls_probs=cls_probs,
            xy_offsets=xy_offsets,
            cls_probs_final=cls_probs_final,
            direction_probs_final=direction_probs_final,
            direction_probs=direction_probs,
            edge_feats_final=edge_feats_final,
            xy_offsets_final=xy_offsets_final,
            active_nodes=active_nodes,
            pos_post_nodes=pos_post_nodes,
            pos_pre_nodes=pos_pre_nodes
        )

        return output

    def get_loss(self, head_outputs, targets):
        lines = targets['line_instances']
        categories = targets['line_classes']
        view = targets['view']
        device = head_outputs['cls_probs'].device
        batch_size = len(head_outputs['cls_probs'])
        target_cls, target_xy_offset, target_id, single_samples, target_direction, node_order = \
            self.generate_targets(lines, categories, view, device)

        pos_indices = (target_cls == 1).any(dim=3)
        cls_loss = sigmoid_focal_loss(head_outputs['cls_probs'], target_cls.permute(0, 3, 1, 2))
        cls_loss = cls_loss.sum() / (target_cls.sum() + 1)

        pn = pos_indices.nonzero()
        dir_loss = self.cross_entropy_loss(head_outputs['direction_probs'][pn[:, 0], :, pn[:, 1], pn[:, 2]],
                                           target_direction.argmax(dim=3)[pn[:, 0], pn[:, 1], pn[:, 2]])
        if pos_indices.sum() == 0:
            dir_loss.fill_(0)

        y_directed = target_direction[..., 1] == 1
        x_directed = target_direction[..., 0] == 1
        reg_x_loss = self.l1_loss(head_outputs['xy_offsets'][:, 0][y_directed], target_xy_offset[y_directed][:, 0])
        if y_directed.sum() == 0:
            reg_x_loss.fill_(0)
        reg_y_loss = self.l1_loss(head_outputs['xy_offsets'][:, 1][x_directed], target_xy_offset[x_directed][:, 1])
        if x_directed.sum() == 0:
            reg_y_loss.fill_(0)
        reg_loss = reg_x_loss + reg_y_loss

        direction_pred = head_outputs['direction_probs_final'].argmax(dim=1)

        pred_nodes = self.offsets_2_positions(head_outputs['xy_offsets_final'], direction_pred)
        target_dir = target_direction.argmax(dim=3)
        gt_nodes = self.offsets_2_positions(target_xy_offset.permute(0, 3, 1, 2), target_dir)

        single_samples = single_samples != 0
        gt_classes = [target_cls[b_idx][single_samples[b_idx]].argmax(dim=1) for b_idx in range(batch_size)]
        gt_points = [gt_nodes[b_idx][single_samples[b_idx]] for b_idx in range(batch_size)]
        gt_dirs = [target_direction[b_idx][single_samples[b_idx]].argmax(dim=1) for b_idx in range(batch_size)]
        gt_indices = [single_samples[b_idx].nonzero() for b_idx in range(batch_size)]

        match_list = self.assigner.assign(head_outputs['cls_probs_final'], pred_nodes,
                                          head_outputs['direction_probs_final'],
                                          gt_points, gt_classes, gt_dirs, head_outputs['pos_post_nodes'])

        match_list = [torch.stack((m[0], gt_indices[b_idx][m[1]]), dim=1) if m is not None else None for b_idx, m in
                      enumerate(match_list)]
        target_id[single_samples.logical_not()] = -1
        adj_final_loss, _ = self.optimal_transport(head_outputs['edge_feats_final'],
                                                   head_outputs['pos_post_nodes'],
                                                   pred_nodes, target_id, node_order, match_list)

        cls_pairs = [torch.zeros((self.num_classes, 0), device=device),
                     torch.zeros((self.num_classes, 0), device=device)]
        position_pairs = [torch.zeros((0, 1), device=device),
                          torch.zeros((0, 1), device=device)]
        direction_pairs = [torch.zeros((2, 0), device=device),
                           torch.zeros(0, device=device, dtype=torch.long)]

        for batch_idx, img_matches in enumerate(match_list):
            if img_matches is None:
                continue
            targets = torch.zeros_like(target_cls[batch_idx])

            targets[img_matches[:, 0, 0], img_matches[:, 0, 1]] = \
                target_cls[batch_idx][img_matches[:, 1, 0], img_matches[:, 1, 1]]

            m = head_outputs['active_nodes'][batch_idx]
            cls_pairs[0] = torch.cat((cls_pairs[0], head_outputs['cls_probs_final'][batch_idx][:, m]), dim=1)
            cls_pairs[1] = torch.cat((cls_pairs[1], targets[m].T), dim=1)

            x_diff = self.anchor_points[img_matches[:, 1, 0], img_matches[:, 1, 1]][:, 1] - \
                     self.anchor_points[img_matches[:, 0, 0], img_matches[:, 0, 1]][:, 1]
            x_diff = x_diff / self.x_bound[2]
            offset_gt_x = target_xy_offset[batch_idx][img_matches[:, 1, 0], img_matches[:, 1, 1]][:, 0]
            offset_gt_x = offset_gt_x + x_diff.to(device)

            y_diff = self.anchor_points[img_matches[:, 1, 0], img_matches[:, 1, 1]][:, 0] - \
                     self.anchor_points[img_matches[:, 0, 0], img_matches[:, 0, 1]][:, 0]
            y_diff = y_diff / self.y_bound[2]
            offset_gt_y = target_xy_offset[batch_idx][img_matches[:, 1, 0], img_matches[:, 1, 1]][:, 1]
            offset_gt_y = offset_gt_y + y_diff.to(device)

            offset_gt = torch.stack((offset_gt_x, offset_gt_y)).T
            offset_pred = head_outputs['xy_offsets_final'][batch_idx][:, img_matches[:, 0, 0], img_matches[:, 0, 1]]

            gt_directions = target_direction[batch_idx, img_matches[:, 1, 0], img_matches[:, 1, 1]].argmax(dim=1)
            pred_direction = head_outputs['direction_probs_final'][batch_idx, :, img_matches[:, 0, 0],
                             img_matches[:, 0, 1]]
            direction_pairs[0] = torch.cat((direction_pairs[0], pred_direction), dim=1)
            direction_pairs[1] = torch.cat((direction_pairs[1], gt_directions))

            offset_pred = offset_pred[(1 - gt_directions).abs(), torch.arange(0, len(gt_directions))]
            offset_gt = offset_gt[torch.arange(0, len(gt_directions)), (1 - gt_directions).abs()]

            position_pairs[0] = torch.cat((position_pairs[0], offset_pred[:, None]))
            position_pairs[1] = torch.cat((position_pairs[1], offset_gt[:, None]))

        final_cls_loss = sigmoid_focal_loss(cls_pairs[0], cls_pairs[1])
        final_cls_loss = final_cls_loss.sum() / (cls_pairs[1].sum() + 1)

        final_reg_loss = self.l1_loss(position_pairs[0], position_pairs[1])
        final_dir_loss = self.cross_entropy_loss(direction_pairs[0][None, :], direction_pairs[1][None, :])
        if len(position_pairs[0]) == 0:
            final_reg_loss.fill_(0)
            final_dir_loss.fill_(0)

        cls_factor = 1 / torch.exp(self.cls_weight)
        reg_factor = 1 / torch.exp(self.reg_weight)
        dir_factor = 1 / torch.exp(self.direction_weight)

        cls_final_factor = 1 / torch.exp(self.cls_final_weight)
        dir_final_factor = 1 / torch.exp(self.direction_final_weight)
        reg_final_factor = 1 / torch.exp(self.reg_final_weight)
        adj_final_factor = 1 / torch.exp(self.adj_final_weight)

        loss = dict(loss_cls=cls_factor * cls_loss,
                    loss_reg=reg_factor * reg_loss,
                    loss_dir=dir_factor * dir_loss,
                    loss_cls_final=cls_final_factor * final_cls_loss,
                    loss_reg_final=reg_final_factor * final_reg_loss,
                    loss_dir_final=dir_final_factor * final_dir_loss,
                    loss_adj_final=adj_final_loss * adj_final_factor,
                    uncertainty_cls=0.5 * self.cls_weight,
                    uncertainty_reg=0.5 * self.reg_weight,
                    uncertainty_direction=0.5 * self.direction_weight,
                    uncertainty_cls_final=0.5 * self.cls_final_weight,
                    uncertainty_direction_final=0.5 * self.direction_final_weight,
                    uncertainty_reg_final=0.5 * self.reg_final_weight,
                    uncertainty_adj_final=0.5 * self.adj_final_weight)

        factor = dict(cls=cls_factor,
                      reg=reg_factor,
                      dir=dir_factor,
                      cls_final=cls_final_factor,
                      reg_final=reg_final_factor,
                      adj_final=adj_final_factor,
                      dir_final=dir_final_factor)

        return loss, factor

    def post_process(self, head_outputs):

        adjacent_feats = head_outputs['edge_feats_final']
        pos_post_nodes = head_outputs['pos_post_nodes']
        direction_probs_final = head_outputs['direction_probs_final']
        direction_pred_final = direction_probs_final.argmax(dim=1)
        cls_probs_final = head_outputs['cls_probs_final']

        filter_positions = pos_post_nodes.logical_not().nonzero()
        cls_probs_final[filter_positions[:, 0], :, filter_positions[:, 1], filter_positions[:, 2]] = \
            inv_sigmoid(torch.zeros(self.num_classes).cuda())

        pred_points = self.offsets_2_positions(head_outputs['xy_offsets_final'], direction_pred_final)

        batch_size = head_outputs['cls_probs'].shape[0]
        output_lanes = []
        for batch_idx in range(batch_size):
            active_nodes_high = (cls_probs_final[batch_idx].sigmoid() >= self.cls_thresh).any(dim=0)

            _, similarity_matrix = self.optimal_transport(adjacent_feats[batch_idx, None],
                                                          pos_post_nodes[batch_idx, None],
                                                          pred_points[batch_idx, None])

            img_lanes = []
            if similarity_matrix[0] is None:
                output_lanes.append(img_lanes)
                continue

            similarity_matrix = similarity_matrix[0][0, :-1, :-1]
            similarity_matrix.fill_diagonal_(0)

            subset = active_nodes_high[pos_post_nodes[batch_idx]]
            if subset.sum() == 0:
                output_lanes.append(img_lanes)
                continue

            lane_graph = similarity_matrix[subset][:, subset]
            active_indices = torch.logical_and(active_nodes_high,
                                               pos_post_nodes[batch_idx]).flatten().nonzero()[:, 0]

            lane_graph[lane_graph < self.adj_thresh] = 0
            img_lanes = self.extract_lanes(head_outputs['cls_probs_final'][batch_idx].sigmoid(), lane_graph,
                                           active_indices, pred_points[batch_idx])

            output_lanes.append(img_lanes)

        out_lanes, _ = self.nms_bev(output_lanes, device=head_outputs['cls_probs'].device, overlap=0.0)
        results = self.organize_output(out_lanes)
        return results

    def organize_output(self, output_lanes):
        results = dict(line_instances=[], line_classes=[], line_probs=[])
        for im_idx, im_lanes in enumerate(output_lanes):
            yx_lanes = []
            lane_probs = []
            lane_cats = []
            for lane in im_lanes:
                if lane['score'] > self.lane_thresh:
                    x_pos = lane['x_positions']
                    y_pos = lane['y_positions']
                    yx_pos = np.stack((y_pos, x_pos)).transpose()
                    yx_lanes.append(yx_pos)
                    lane_probs.append(float((lane['score']).clip(0.0, 1.0)[None]))
                    lane_cats.append(int(lane['category']))

            results['line_instances'].append([yx_lanes])
            results['line_probs'].append([lane_probs])
            results['line_classes'].append([lane_cats])
        return results

    def nms_bev(self, output_lanes, device, overlap=1.0):
        keep_lanes = []
        nms_y_stride = (self.y_bound[1] - self.y_bound[0]) / 72
        nms_y_anchors = torch.arange(self.y_bound[0] + nms_y_stride / 2, self.y_bound[1] - 1e-9, nms_y_stride)
        keep_ids = []
        for _, im_lanes in enumerate(output_lanes):
            keep = []
            if len(im_lanes) != 0:
                im_lanes_nms = torch.zeros((len(im_lanes), 2 + 3 + len(nms_y_anchors)), device=device)
                scores = torch.zeros(len(im_lanes), device=device)
                for i, lane in enumerate(im_lanes):
                    input_lane = torch.stack((torch.from_numpy(lane['x_positions']),
                                              torch.from_numpy(lane['y_positions']))).T[
                                 lane['start_y']: lane['end_y'] + 1]
                    x_values, y_start, y_end = resample_lane_in_y_axis(input_lane, nms_y_anchors)
                    if x_values is None:
                        continue
                    im_lanes_nms[i][2] = int(y_start) / (len(nms_y_anchors) - 1)
                    im_lanes_nms[i][4] = int(y_end) - int(y_start) + 1
                    im_lanes_nms[i][5:] = torch.from_numpy(x_values).to(device)
                    scores[i] = lane['score']
                # nms requires less than 72 points, we should sample 72 points from y axis
                keep, num_to_keep, _ = line_nms(im_lanes_nms, scores, overlap=overlap, top_k=20)
                keep = keep[:num_to_keep]
                im_lanes = [l for i, l in enumerate(im_lanes) if i in keep]
            keep_lanes.append(im_lanes)
            keep_ids.append(keep)
        return keep_lanes, keep_ids

    def extract_lanes(self, pred_cls, lane_graph, active_indices, pred_points):

        lanes = []
        active_nodes = (pred_cls >= self.cls_thresh).any(dim=0)

        num_connections = torch.logical_and((lane_graph != 0).T, (lane_graph != 0)).sum(dim=0)
        start_points = torch.logical_and(num_connections < 2, num_connections > 0).nonzero()[:, 0]
        prob_starts = pred_cls.max(dim=0)[0].flatten()[active_indices[start_points]]
        start_points = start_points[prob_starts.argsort(descending=True)]

        # lane_graph = 1 - lane_graph.cpu().numpy()
        # lane_graph[lane_graph > 0] = 1 - lane_graph[lane_graph > 0]
        lane_graph = lane_graph.cpu().numpy() > 0
        graph_nx = nx.from_numpy_array(lane_graph, create_using=nx.DiGraph())

        active_cls_probs = pred_cls.max(dim=0)[0][active_nodes]
        start_points = start_points.tolist()

        while len(start_points) != 0:
            start_point = start_points[0]
            start_points = start_points[1:]
            if start_point in graph_nx.nodes:
                # lengths, paths = nx.single_source_bellman_ford(graph_nx, start_point, target=None, weight='weight')
                lengths, paths = nx.single_source_dijkstra(graph_nx, int(start_point))
                end_points_ = list(set(paths.keys()) & set(start_points))

                candidate_paths = [(paths[idx], (lengths[idx] / len(paths[idx]))) for i, idx in enumerate(end_points_)]

                if len(candidate_paths) != 0:
                    # best_path = torch.tensor(
                    #     [(active_cls_probs[c].mean() + l) / 2 for c, l in candidate_paths]).argmin()
                    best_path = torch.tensor(
                        [active_cls_probs[c].mean() for c, l in candidate_paths]).argmin()

                    path_nodes, length = candidate_paths[best_path]
                    node_indices = active_indices[path_nodes]
                    num_classes = pred_cls.shape[0]
                    path_cls_probs = pred_cls.view(num_classes, -1)[:, node_indices]
                    path_cls = path_cls_probs.mean(dim=1).argmax()

                    # path_score = (path_cls_probs.max(dim=0)[0].mean() + length) / 2
                    path_score = path_cls_probs.max(dim=0)[0].mean()

                    if len(node_indices) > 5:
                        lane = dict(
                            x_positions=pred_points.flatten(0, 1)[node_indices, 1].cpu().numpy(),
                            y_positions=pred_points.flatten(0, 1)[node_indices, 0].cpu().numpy(),
                            score=path_score,
                            category=path_cls,
                            category_probs=path_cls_probs.mean(dim=1),
                            start_y=0,
                            end_y=len(pred_points.flatten(0, 1)[node_indices, 1]),
                        )
                        lanes.append(lane)

        return lanes

    def offsets_2_positions(self, xy_offsets, directions_xy):
        device = xy_offsets.device
        batch_size = len(xy_offsets)
        anchor_points = self.anchor_points.to(device).clone()
        node_positions = anchor_points[None, :].repeat(batch_size, 1, 1, 1)
        x_dir = directions_xy == 0
        y_dir = directions_xy == 1
        node_positions[..., 0][x_dir] = node_positions[..., 0][x_dir] + xy_offsets[:, 1][x_dir] * self.y_bound[
            2]  # y values
        node_positions[..., 1][y_dir] = node_positions[..., 1][y_dir] + xy_offsets[:, 0][y_dir] * self.x_bound[
            2]  # x values
        return node_positions

    def generate_targets(self, lines, categories, view, device):
        batch_size = len(lines)

        target_id = torch.ones((batch_size, *self.anchor_points.shape[:2]), device=device).long() * -1
        target_offset = torch.zeros((batch_size, *self.anchor_points.shape[:2], 2), device=device)
        direction_map = torch.zeros((batch_size, *self.anchor_points.shape[:2], 2), device=device)
        # dist_map = torch.ones((batch_size, *self.anchor_points.shape[:2]), device=device) * 100
        target_cls = torch.zeros((batch_size, *self.anchor_points.shape[:2], self.num_classes), device=device)
        single_samples = torch.zeros((batch_size, *self.anchor_points.shape[:2]), device=device)
        node_order = torch.zeros((batch_size, *self.anchor_points.shape[:2]), device=device, dtype=torch.long)

        for batch_idx, (img_lines, img_cats) in enumerate(zip(lines, categories)):
            for lane_id, (line, cat) in enumerate(zip(img_lines[0], img_cats[0])):
                if len(line) < 2:
                    continue

                line_map = torch.zeros(*self.anchor_points.shape[:2], device=device)
                x_y_offsets = torch.zeros(*self.anchor_points.shape[:2], 2, device=device)

                selected_indices, selected_xy, offsets, y_directed = self.find_active_points(line, view[batch_idx])
                x_directed = torch.logical_not(y_directed)
                line_map.flatten()[selected_indices] = 1

                single_points, _, _, _ = self.find_active_points(line, view[batch_idx], single_point=True)

                node_order[batch_idx].flatten()[single_points] = torch.arange(len(single_points), device=device) + 1
                line_map_single = torch.zeros_like(line_map)
                line_map_single.flatten()[single_points] = 1

                x_y_offsets.view(-1, 2)[selected_indices[[x_directed]], 1] = (offsets[x_directed] / self.y_bound[2]).to(
                    device)
                x_y_offsets.view(-1, 2)[selected_indices[y_directed], 0] = (offsets[y_directed] / self.x_bound[2]).to(
                    device)

                direction_map[batch_idx, selected_xy[x_directed, 1], selected_xy[x_directed, 0], 0] = True
                direction_map[batch_idx, selected_xy[y_directed, 1], selected_xy[y_directed, 0], 1] = True

                # pos_indices = line_map.nonzero()[x_y_offsets[line_map == 1][:, 0].abs() < dist_map[batch_idx][line_map == 1].abs()]
                pos_indices = line_map.nonzero()  # TODO: to get preliminary result, the elimination has been disabled, remove it
                line_map = torch.zeros_like(line_map)
                x_z_offsets_temp = x_y_offsets
                x_y_offsets = torch.zeros_like(x_y_offsets)
                line_map[pos_indices[:, 0], pos_indices[:, 1]] = 1
                line_map_single = torch.logical_and(line_map_single, line_map).float()
                x_y_offsets[pos_indices[:, 0], pos_indices[:, 1]] = x_z_offsets_temp[
                    pos_indices[:, 0], pos_indices[:, 1]]

                target_offset[batch_idx][line_map == 1] = x_z_offsets_temp[line_map == 1]
                # dist_map[batch_idx][line_map == 1] = x_y_offsets[line_map == 1][:, 0]

                cls_part = torch.zeros((int(line_map.sum()), self.num_classes), device=device)
                cls_part[:, int(cat)] = 1

                target_cls[batch_idx][line_map == 1] = cls_part
                target_id[batch_idx][line_map_single == 1] = lane_id
                single_samples[batch_idx][line_map_single == 1] = 1

        return target_cls, target_offset, target_id, single_samples, direction_map, node_order

    def find_active_points(self, line, view, single_point=False):
        line_dense = []
        line_bev = []
        y_directed = []

        segment_starts = line[:-1]
        segment_ends = line[1:]
        for seg_start, seg_end in zip(segment_starts, segment_ends):
            height = abs(seg_start[0] - seg_end[0])
            width = abs(seg_start[1] - seg_end[1])
            y_direct = height > width

            dense_points = sample_polyline_points([[seg_start, seg_end]], sample_distance=0.5)[0]
            dense_bev = points_lidar2view([dense_points], view[0, 0].cpu().numpy())[0]  # output x,y format

            line_dense += dense_points.tolist()
            line_bev += dense_bev.tolist()
            y_directed += [y_direct] * len(dense_bev)

        y_directed = torch.tensor(y_directed)
        line_bev = torch.tensor(line_bev)
        line_dense = torch.tensor(line_dense)
        point_idx = torch.arange(0, len(y_directed))

        y_dir_points_bev = line_bev[y_directed]
        y_mask = first_unique_in_seq(y_dir_points_bev[:, 1])
        y_point_idx = point_idx[y_directed][y_mask]
        y_dir_points_lidar = line_dense[y_directed][y_mask]

        x_directed = torch.logical_not(y_directed)
        x_dir_points_bev = line_bev[x_directed]
        x_mask = first_unique_in_seq(x_dir_points_bev[:, 0])
        x_point_idx = point_idx[x_directed][x_mask]
        x_dir_points_lidar = line_dense[x_directed][x_mask]

        point_idx = torch.cat((y_point_idx, x_point_idx))
        line_dense = torch.cat((y_dir_points_lidar, x_dir_points_lidar))[point_idx.argsort()]
        line_bev = torch.cat((y_dir_points_bev[y_mask], x_dir_points_bev[x_mask]))[point_idx.argsort()]
        y_directed = torch.zeros_like(point_idx, dtype=torch.bool)
        y_directed[:len(y_point_idx)] = True
        y_directed = y_directed[point_idx.argsort()]
        x_directed = torch.logical_not(y_directed)
        point_idx = point_idx.sort()[0]

        # line_map = np.zeros((64, 32))
        # line_map[line_bev[:, 1][y_directed], line_bev[:, 0][y_directed]] = 1
        # line_map[line_bev[:, 1][x_directed], line_bev[:, 0][x_directed]] = 2

        x_dist = (line_dense[:, 1, None][y_directed] - self.anchor_x_points[None, :])
        if not single_point:
            active_y_dir = x_dist.abs() < 1.5
        else:
            active_y_dir = torch.zeros_like(x_dist, dtype=torch.bool)
            active_y_dir[torch.arange(0, len(x_dist)), x_dist.abs().argmin(dim=1)] = True

        selected_y_dir = active_y_dir.nonzero()
        selected_x_offset = x_dist[selected_y_dir[:, 0], selected_y_dir[:, 1]]
        selected_y_idx = point_idx[y_directed][selected_y_dir[:, 0]]
        selected_y_dir[:, 0] = line_bev[:, 1][y_directed][selected_y_dir[:, 0]]

        y_dist = (line_dense[:, 0][None, :][:, x_directed] - self.anchor_y_points[:, None])
        if not single_point:
            active_x_dir = y_dist.abs() < 1.5
        else:
            active_x_dir = torch.zeros_like(y_dist, dtype=torch.bool)
            active_x_dir[y_dist.abs().argmin(dim=0), torch.arange(0, y_dist.shape[1])] = True
        selected_x_dir = active_x_dir.nonzero()
        selected_y_offset = y_dist[selected_x_dir[:, 0], selected_x_dir[:, 1]]
        selected_x_idx = point_idx[x_directed][selected_x_dir[:, 1]]
        selected_x_dir[:, 1] = line_bev[:, 0][x_directed][selected_x_dir[:, 1]]

        selected_points = torch.cat((selected_y_dir, selected_x_dir))
        selected_idx = torch.cat((selected_y_idx, selected_x_idx))
        offsets = torch.cat((selected_x_offset, selected_y_offset))
        y_directed = torch.zeros_like(offsets, dtype=torch.bool)
        y_directed[:len(selected_x_offset)] = True

        selected_points = selected_points[selected_idx.argsort()]
        offsets = offsets[selected_idx.argsort()]
        y_directed = y_directed[selected_idx.argsort()]

        selected_indices = selected_points[:, 0] * len(self.anchor_x_points) + selected_points[:, 1]
        return selected_indices, selected_points[:, [1, 0]], offsets, y_directed

    def point_nms(self, exist_probs, points, overlap_x, overlap_y):

        b, ipm_h, ipm_w = exist_probs.shape
        device = exist_probs.device
        exist_map = exist_probs.sigmoid() > 0.01
        nms_map = torch.zeros_like(exist_map, dtype=torch.bool)
        for batch_idx in range(b):
            img_points = points[batch_idx][exist_map[batch_idx]]
            scores = exist_probs[batch_idx][exist_map[batch_idx]].sigmoid()
            keep, num_to_keep, _ = point_nms(img_points, scores, overlap_x=overlap_x, overlap_y=overlap_y, top_k=800)
            keep_positions = exist_map[batch_idx].nonzero()[keep[:num_to_keep]]
            nms_map[batch_idx, keep_positions[:, 0], keep_positions[:, 1]] = True

        return nms_map

    @staticmethod
    def point_nms_func_v2(exist_probs, points, overlap_x, overlap_y):

        b, ipm_h, ipm_w = exist_probs.shape
        device = exist_probs.device
        exist_probs = exist_probs.sigmoid()
        exist_map = exist_probs > 1e-5
        pre_nms_map = torch.zeros_like(exist_map, dtype=torch.bool)
        nms_map = torch.zeros_like(exist_probs, dtype=torch.bool)
        points = points.contiguous()
        for batch_idx in range(b):
            for row, row_pts in enumerate(points[batch_idx]):
                row_pts = row_pts[..., [1, 0]][exist_map[batch_idx, row]]
                row_scores = exist_probs[batch_idx, row][exist_map[batch_idx, row]]
                if len(row_pts) == 0:
                    continue
                keep, num_to_keep, _ = point_nms(row_pts, row_scores, overlap_x=overlap_x, overlap_y=overlap_y,
                                                 top_k=400)
                keep_pts = exist_map[batch_idx, row].nonzero()[keep[:num_to_keep]][:, 0]
                pre_nms_map[batch_idx, row, keep_pts] = True

            scores = exist_probs[batch_idx][pre_nms_map[batch_idx]]
            _, keep = torch.topk(scores, k=min(400, len(scores)))
            keep_positions = pre_nms_map[batch_idx].nonzero()[keep]
            mask = exist_probs[batch_idx][keep_positions[:, 0], keep_positions[:, 1]] > 1e-5
            keep_positions = keep_positions[mask]
            nms_map[batch_idx, keep_positions[:, 0], keep_positions[:, 1]] = True

        return nms_map


class ClsRegDirHead(nn.Module, ABC):
    # Classification Regression and Direction Head
    def __init__(self, in_channels, num_classes, hidden_dim=256, num_layers=1):
        super(ClsRegDirHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.cls_head = self.create_head(num_classes)
        self.reg_head = self.create_head(2)
        self.dir_head = self.create_head(2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)

        prior_probability = 0.01
        torch.nn.init.normal_(self.cls_head[-1].weight, std=0.01)
        torch.nn.init.constant_(self.cls_head[-1].bias, -math.log((1 - prior_probability) / prior_probability))

        torch.nn.init.normal_(self.reg_head[-1].weight, std=0.01)
        torch.nn.init.zeros_(self.reg_head[-1].bias)

    @abstractmethod
    def create_head(self, out_size, final_bias=True):
        pass

    @abstractmethod
    def forward(self, x, **kwargs):
        pass


class FcClsRegDirHead(ClsRegDirHead):
    def __init__(self, in_channels, num_classes, hidden_dim=256, num_layers=1):
        super(FcClsRegDirHead, self).__init__(in_channels, num_classes, hidden_dim=hidden_dim, num_layers=num_layers)
        self.feat_head = self.create_head(self.hidden_dim, final_bias=False)

    def create_head(self, out_size, final_bias=True):
        layers = []
        first_layer = FCNormAct(self.in_channels, self.hidden_dim, layer_norm=True, bias=False)
        layers.append(first_layer)
        for _ in range(self.num_layers - 1):
            layer = FCNormAct(self.hidden_dim, self.hidden_dim, layer_norm=True, bias=False)
            layers.append(layer)

        out_layer = nn.Linear(self.hidden_dim, out_size, bias=final_bias)
        layers.append(out_layer)

        return nn.ModuleList(layers)

    def forward(self, x, **kwargs):
        input_shape = x.shape
        dtype = x.dtype
        device = x.device
        active_points = kwargs['active_points']
        # x = x.permute(0, 2, 3, 1)
        x = x[active_points]

        x_cls = x
        for layer in self.cls_head:
            x_cls = layer(x_cls)

        x_reg = x
        for layer in self.reg_head:
            x_reg = layer(x_reg)

        x_dir = x
        for layer in self.dir_head:
            x_dir = layer(x_dir)

        x_feat = x
        for layer in self.feat_head:
            x_feat = layer(x_feat)

        out_cls = torch.zeros((*input_shape[0:3], self.num_classes), dtype=dtype,
                              device=device)
        out_reg = torch.zeros((*input_shape[0:3], 2), dtype=dtype,
                              device=device)
        out_dir = torch.zeros((*input_shape[0:3], 2), dtype=dtype,
                              device=device)
        out_feat = torch.zeros((*input_shape[0:3], self.hidden_dim), dtype=dtype,
                               device=device)

        out_cls[active_points] = x_cls
        out_reg[active_points] = x_reg
        out_dir[active_points] = x_dir
        out_feat[active_points] = x_feat

        out_cls = out_cls.permute(0, 3, 1, 2)
        out_reg = out_reg.permute(0, 3, 1, 2)
        out_dir = out_dir.permute(0, 3, 1, 2)
        out_feat = out_feat.permute(0, 3, 1, 2)

        return out_cls, out_reg, out_dir, out_feat


class ConvClsRegDirHead(ClsRegDirHead):
    def __init__(self, in_channels, num_classes, hidden_dim=256, num_layers=1):
        super(ConvClsRegDirHead, self).__init__(in_channels, num_classes, hidden_dim=hidden_dim, num_layers=num_layers)

    def create_head(self, out_size, final_bias=True):
        layers = []
        first_layer = ConvNormAct(self.in_channels, self.hidden_dim, kernel_size=3, padding=1,
                                  batch_norm=True, bias=False)
        layers.append(first_layer)
        for _ in range(self.num_layers - 1):
            layer = ConvNormAct(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1,
                                batch_norm=True, bias=False)
            layers.append(layer)

        out_layer = nn.Conv2d(self.hidden_dim, out_size, kernel_size=1, padding=0, bias=final_bias)
        layers.append(out_layer)

        return nn.ModuleList(layers)

    def forward(self, x, **kwargs):
        x_cls = x
        for layer in self.cls_head:
            x_cls = layer(x_cls)

        x_reg = x
        for layer in self.reg_head:
            x_reg = layer(x_reg)

        x_dir = x
        for layer in self.dir_head:
            x_dir = layer(x_dir)

        return x_cls, x_reg, x_dir


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False,
                 bias=True, inplace=True):
        super(ConvNormAct, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, stride=stride, bias=bias)
        if batch_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None

        self.act = nn.ReLU(inplace)

        if self.norm is not None:
            self.layer = nn.Sequential(self.conv2d, self.norm, self.act)
        else:
            self.layer = nn.Sequential(self.conv2d, self.act)

    def forward(self, x):
        return self.layer(x)


class FCNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, layer_norm=False, bias=True, inplace=True):
        super(FCNormAct, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = None

        self.act = nn.ReLU(inplace)

        if self.norm is not None:
            self.layer = nn.Sequential(self.fc, self.norm, self.act)
        else:
            self.layer = nn.Sequential(self.fc, self.act)

    def forward(self, x):
        return self.layer(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim, dropout, local_attn=False):
        super(EncoderLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(dim, 8)
        self.local_attn = local_attn
        if local_attn:
            # self.patch_attn = LocalPatchAttention(dim, patch_size)
            self.patch_attn = MultiHeadAttention(dim, 8)
            self.norm3 = nn.LayerNorm(dim)
            self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, cross, memory=None, mask=None):
        src_, attention = self.attn(src, cross, cross, mask=mask)
        src = self.norm1(src + self.dropout1(src_))
        if self.local_attn:
            src_, attention = self.patch_attn(src, memory, memory, mask=None)
            src = self.norm3(src + self.dropout3(src_))
        src = self.norm2(src + self.__ff_block(src))
        return src, attention

    def __ff_block(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class LocalAttentionLayer(nn.Module):
    def __init__(self, patch_edge, dim, dropout=0.1, local_attn=False):
        super(LocalAttentionLayer, self).__init__()
        self.patch_edge = patch_edge
        self.encoder = EncoderLayer(dim, dropout, local_attn)

    def forward(self, x, memory, active_nodes):
        out_tensor = torch.zeros_like(x)

        pad = self.patch_edge // 2
        center = self.patch_edge // 2
        active_nodes = active_nodes[..., None]
        act_patches = split_tensor_bhwc(F.pad(active_nodes, (0, 0, pad, pad, pad, pad, 0, 0)),
                                        self.patch_edge, 1)
        act_patches[:, center, center, :] = False
        active_nodes_wo_center = act_patches.flatten(3, 4).any(dim=3)
        active_nodes = torch.logical_and(active_nodes_wo_center, active_nodes)

        act_patches = split_tensor_bhwc(F.pad(active_nodes, (0, 0, pad, pad, pad, pad, 0, 0)), self.patch_edge, 1)
        x_patches = split_tensor_bhwc(F.pad(x, (0, 0, pad, pad, pad, pad, 0, 0)), self.patch_edge, 1)
        x_patches = act_patches * x_patches
        x_patches = rearrange(x_patches, 'b h w lh lw c -> b h w (lh lw) c')

        memory_patches = split_tensor_bhwc(F.pad(memory, (0, 0, pad, pad, pad, pad, 0, 0)), self.patch_edge, 1)
        memory_patches = rearrange(memory_patches, 'b h w lh lw c -> b  h w (lh lw) c')
        memory_patches = memory_patches[active_nodes[..., 0]]

        x_patches = x_patches[active_nodes[..., 0]]
        center_point = (self.patch_edge ** 2) // 2
        queries = x_patches[:, [center_point], :]
        mask = act_patches[..., 0][active_nodes[..., 0]].flatten(1)
        queries, attention = self.encoder(queries, x_patches, memory_patches, mask=mask[:, None, None, :])
        out_tensor[active_nodes[..., 0]] = queries[:, 0]
        return out_tensor, active_nodes[..., 0], attention, memory


class LocalAttention(nn.Module):
    def __init__(self, patch_size, dim, dropout=0.1, num_layers=3):
        super(LocalAttention, self).__init__()
        self.patch_size = patch_size
        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            attention_layer = LocalAttentionLayer(patch_size, dim, dropout)
            self.attention_layers.append(attention_layer)
        self.bev_pos = PositionEmbeddingSine(dim // 2, normalize=True, temperature=10000)

    def forward(self, x, active_nodes):
        bev_pos = self.bev_pos(x[:, 0, ...])
        x = x + bev_pos
        x = x.permute(0, 2, 3, 1)
        memory = x
        for attention_layer in self.attention_layers:
            x, active_nodes, _, memory = attention_layer(x, memory, active_nodes)
        return x, active_nodes


def split_tensor_bhwc(input_tensor, patch_size, stride):
    # Get the input tensor shape
    batch_size, height, width, channels = input_tensor.size()

    # Calculate the number of patches in each dimension
    num_patches_height = ((height - patch_size) // stride) + 1
    num_patches_width = ((width - patch_size) // stride) + 1

    # Calculate the indices of the starting positions of each patch
    patch_starts_height = torch.arange(0, num_patches_height, device=input_tensor.device) * stride
    patch_starts_width = torch.arange(0, num_patches_width, device=input_tensor.device) * stride

    # Create a 4D tensor of patch indices
    patch_indices = torch.meshgrid(patch_starts_height, patch_starts_width)
    patch_indices = torch.stack(patch_indices, dim=-1)
    patch_indices = patch_indices.view(-1, 2)

    # Calculate the indices of the pixels in each patch
    pixel_indices = torch.arange(0, patch_size, device=input_tensor.device).view(1, -1, 1) + \
                    patch_indices[:, 0].view(-1, 1, 1), \
                    torch.arange(0, patch_size, device=input_tensor.device).view(1, 1, -1) + \
                    patch_indices[:, 1].view(-1, 1, 1)

    # Extract the patches using the pixel indices
    patches = input_tensor[:, pixel_indices[0], pixel_indices[1], :]
    patches = patches.view(batch_size, num_patches_height, num_patches_width, patch_size, patch_size, channels)

    # Return the patches
    return patches


def split_tensor_bchw(input_tensor, patch_size, stride):
    # Get the input tensor shape
    batch_size, channels, height, width = input_tensor.size()

    # Calculate the number of patches in each dimension
    num_patches_height = ((height - patch_size) // stride) + 1
    num_patches_width = ((width - patch_size) // stride) + 1

    # Calculate the indices of the starting positions of each patch
    patch_starts_height = torch.arange(0, num_patches_height, device=input_tensor.device) * stride
    patch_starts_width = torch.arange(0, num_patches_width, device=input_tensor.device) * stride

    # Create a 4D tensor of patch indices
    patch_indices = torch.meshgrid(patch_starts_height, patch_starts_width)
    patch_indices = torch.stack(patch_indices, dim=-1)
    patch_indices = patch_indices.view(-1, 2)

    # Calculate the indices of the pixels in each patch
    pixel_indices = torch.arange(0, patch_size, device=input_tensor.device).view(1, -1, 1) + \
                    patch_indices[:, 0].view(1, -1, 1, 1), \
                    torch.arange(0, patch_size, device=input_tensor.device).view(1, 1, -1) + \
                    patch_indices[:, 1].view(1, -1, 1, 1)

    # Extract the patches using the pixel indices
    patches = input_tensor[:, :, pixel_indices[0], pixel_indices[1]]
    patches = patches.view(batch_size, channels, num_patches_height, num_patches_width, patch_size, patch_size)

    # Return the patches
    return patches


def resample_lane_in_y_axis(input_lane, y_steps):
    # y_steps = y_steps.cpu().numpy()
    y_stride = abs(y_steps[1] - y_steps[0])
    # input_lane = input_lane.cpu().numpy()
    y_max = input_lane[:, 1].max() + y_stride
    y_min = input_lane[:, 1].min() - y_stride
    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    x_values = f_x(y_steps)
    mask = np.logical_and(y_steps > y_min, y_steps < y_max)
    y_start = mask.nonzero()[0].min()
    y_end = mask.nonzero()[0].max()
    return x_values, y_start, y_end


def inv_sigmoid(input):
    out = torch.log(input.clip(1e-5, 1 - 1e-5) / (1 - input.clip(1e-5, 1 - 1e-5)))
    return out


def transform_lane_gflat2g(h_cam, X_gflat, Y_gflat, Z_g):
    """
        Given X coordinates in flat ground space, Y coordinates in flat ground space, and Z coordinates in real 3D ground space
        with projection matrix from 3D ground to flat ground, compute real 3D coordinates X, Y in 3D ground space.

    :param P_g2gflat: a 3 X 4 matrix transforms lane form 3d ground x,y,z to flat ground x, y
    :param X_gflat: X coordinates in flat ground space
    :param Y_gflat: Y coordinates in flat ground space
    :param Z_g: Z coordinates in real 3D ground space
    :return:
    """

    X_g = X_gflat - X_gflat * Z_g / h_cam
    Y_g = Y_gflat - Y_gflat * Z_g / h_cam

    return X_g, Y_g


def first_unique_in_seq(input_tensor):
    if len(input_tensor) == 0:
        return []
    inverse_indices = input_tensor.unique_consecutive(dim=0, return_inverse=True)[1]
    shifted_tensor = torch.cat([inverse_indices[0].unsqueeze(0), inverse_indices[:-1]])
    mask = inverse_indices != shifted_tensor
    mask[0] = True
    return mask
