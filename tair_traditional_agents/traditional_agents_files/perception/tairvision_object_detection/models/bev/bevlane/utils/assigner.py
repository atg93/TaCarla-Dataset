from scipy.optimize import linear_sum_assignment
import torch


class Assigner:
    def __init__(self, cls_alpha, cls_gamma, x_thresh, y_thresh):
        self.cls_apha = cls_alpha
        self.cls_gamma = cls_gamma
        self.x_thresh = x_thresh
        self.y_thresh = y_thresh

    def assign(self, cls_pred, pred_points, pred_dir, gt_points, gt_labels, gt_dir, active_points):
        batch_size, ipm_h, ipm_w = active_points.shape
        device = active_points.device
        match_points = []

        for batch_idx in range(batch_size):
            img_pred_points = pred_points[batch_idx][active_points[batch_idx]]
            img_cls_pred = cls_pred[batch_idx][:, active_points[batch_idx]].sigmoid().permute(1, 0)
            img_dir_pred = pred_dir[batch_idx][:, active_points[batch_idx]].permute(1, 0)
            gt_classes = gt_labels[batch_idx].to(device)
            img_gt_dirs = gt_dir[batch_idx].to(device)
            img_gt_points = gt_points[batch_idx].to(device)
            if len(img_gt_points) == 0:
                img_gt_points = torch.zeros((0, img_pred_points.shape[-1]), device=device)
            pred_index, gt_index = self.assign_core(img_pred_points, img_cls_pred, img_dir_pred,
                                                    img_gt_points, gt_classes, img_gt_dirs)

            if len(pred_index) > 0:
                pred_indices = active_points[batch_idx].nonzero()[pred_index]
                match_points.append((pred_indices, gt_index))
            else:
                match_points.append(None)
        return match_points

    def assign_core(self, pred_points, pred_cls, pred_dir, gt_points, gt_classes, gt_dir):
        inf_value = 2**12
        x_cost_mat = torch.cdist(pred_points[:, [1]], gt_points[:, [1]], p=1)
        y_cost_mat = torch.cdist(pred_points[:, [0]], gt_points[:, [0]], p=1)

        if len(gt_classes) == 0:
            cls_cost_mat = torch.zeros_like(x_cost_mat)
        else:
            cls_cost_mat = self.calc_cls_cost(pred_cls, gt_classes)

        if len(gt_dir) == 0:
            dir_cost_mat = torch.zeros_like(x_cost_mat)
        else:
            dir_cost_mat = self.calc_dir_cost(pred_dir, gt_dir)

        cost_mat = cls_cost_mat * 2  + dir_cost_mat * 2 + (x_cost_mat ** 2 + y_cost_mat ** 2).sqrt() * 5

        cost_mask = torch.logical_or(x_cost_mat > self.x_thresh, y_cost_mat > self.y_thresh)
        cost_mat[cost_mask] = inf_value

        pred_index, gt_index = linear_sum_assignment(cost_mat.cpu().detach().numpy())
        mask = cost_mat[pred_index, gt_index] < inf_value
        mask = mask.cpu().numpy()
        return pred_index[mask], gt_index[mask]

    def calc_cls_cost(self, pred_cls, gt_classes):
        neg_cost = -(1 - pred_cls + 1e-12).log() * (1 - self.cls_apha) * pred_cls.pow(self.cls_gamma)
        pos_cost = -(pred_cls + 1e-12).log() * self.cls_apha * (1 - pred_cls).pow(self.cls_gamma)
        cls_cost_mat = pos_cost[:, gt_classes] - neg_cost[:, gt_classes]
        return cls_cost_mat

    def calc_dir_cost(self, pred_cls, gt_classes):
        pred_scores = pred_cls.softmax(-1)
        cls_cost = -pred_scores[:, gt_classes]
        return cls_cost
