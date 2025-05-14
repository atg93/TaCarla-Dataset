import numpy as np
import torch
from functools import partial
import time
import datetime

from multiprocessing import Pool
from torchmetrics.metric import Metric
from tairvision.eval_utils.openlanev2_evaluation.distance import chamfer_distance, pairwise
from tairvision.models.bev.common.utils.geometry import sample_polyline_points


class ChamferDistanceAP(Metric):
    def __init__(self, n_classes, distance_thresholds=[0.5, 1.0, 1.5], sampling_dist=0.3):
        super().__init__()

        self.distance_thresholds = distance_thresholds
        self.sampling_dist = sampling_dist

        self.class_pred_lines = {cls_id: [] for cls_id in range(n_classes)}
        self.class_pred_scores = {cls_id: [] for cls_id in range(n_classes)}
        self.class_gt_lines = {cls_id: [] for cls_id in range(n_classes)}

        self.class_num_gts = {cls_id: 0 for cls_id in range(n_classes)}
        self.class_num_pred = {cls_id: 0 for cls_id in range(n_classes)}

        self.n_classes = n_classes
        self.n_workers = 32

    def compute_tp_fp(self, pred_lines, gt_lines, pred_scores):

        num_pred = len(pred_lines)
        num_gt = len(gt_lines)

        tp_fp_list = []
        tp = np.zeros(num_pred, dtype=np.float32)
        fp = np.zeros(num_pred, dtype=np.float32)

        if num_gt == 0:
            tp_fp_score_by_thr = {}
            fp[...] = 1
            for thr in self.distance_thresholds:
                tp_fp_score = np.hstack([tp.copy()[:, None], fp.copy()[:, None], pred_scores[:, None]])
                tp_fp_score_by_thr[thr] = tp_fp_score
            return tp_fp_score_by_thr

        if num_pred == 0:
            tp_fp_score_by_thr = {}
            for thr in self.distance_thresholds:
                tp_fp_score = np.hstack([tp.copy()[:, None], fp.copy()[:, None], pred_scores[:, None]])
                tp_fp_score_by_thr[thr] = tp_fp_score

            return tp_fp_score_by_thr

        pred_lines = sample_polyline_points(pred_lines, sample_distance=self.sampling_dist)
        gt_lines = sample_polyline_points(gt_lines, sample_distance=self.sampling_dist)
        dist_mat = pairwise(pred_lines,
                            gt_lines,
                            chamfer_distance, relax=True)

        matrix_min = dist_mat.min(axis=1)
        matrix_argmin = dist_mat.argmin(axis=1)

        sort_inds = np.argsort(-pred_scores)

        tp_fp_score_by_thr = {}

        for thr in self.distance_thresholds:
            tp = np.zeros(num_pred, dtype=np.float32)
            fp = np.zeros(num_pred, dtype=np.float32)
            gt_covered = np.zeros(num_gt, dtype=bool)

            # tp = 0 and fp = 0 means ignore this detected bbox,
            for i in sort_inds:
                if matrix_min[i] <= thr:
                    matched_gt = matrix_argmin[i]
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[i] = 1
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1

            tp_fp_score = np.hstack([tp[:, None], fp[:, None], pred_scores[:, None]])
            tp_fp_score_by_thr[thr] = tp_fp_score

            tp_fp_list.append((tp, fp))
        return tp_fp_score_by_thr

    def update(self, prediction, prediction_probs, prediction_classes, gt_lines, gt_classes):
        batch_size = len(prediction)
        for batch_idx in range(batch_size):

            img_pred_scores = np.array(prediction_probs[batch_idx][0])
            img_pred_classes = np.array(prediction_classes[batch_idx][0])
            img_gt_classes = np.array(gt_classes[batch_idx][0])
            img_pred_lines = prediction[batch_idx][0]
            img_gt_lines = gt_lines[batch_idx][0]

            for class_id in range(self.n_classes):

                pred_mask = np.array(img_pred_classes) == class_id
                gt_mask = np.array(img_gt_classes) == class_id

                self.class_num_gts[class_id] += int(gt_mask.sum())
                self.class_num_pred[class_id] += int(pred_mask.sum())

                pred_lines = [img_pred_lines[i] for i in pred_mask.nonzero()[0]]
                pred_scores = img_pred_scores[pred_mask]

                gt_lines_ = [img_gt_lines[i] for i in gt_mask.nonzero()[0]]

                self.class_gt_lines[class_id].append(gt_lines_)
                self.class_pred_lines[class_id].append(pred_lines)
                self.class_pred_scores[class_id].append(pred_scores)

    def collect_outputs(self):
        result_dict = dict()
        result_dict['num_gts'] = self.class_num_gts
        result_dict['num_preds'] = self.class_num_pred

        result_dict['pred_lines'] = self.class_pred_lines
        result_dict['pred_scores'] = self.class_pred_scores
        result_dict['gt_lines'] = self.class_gt_lines

        # result_dict['tp_fp'] = self.class_tp_fp_score

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                local_rank = torch.distributed.get_rank()

                obj_list = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(obj_list, result_dict)

                if local_rank == 0:
                    for rank in range(torch.distributed.get_world_size()):
                        if rank == 0:
                            continue

                        result_dict = merge_dicts(result_dict, obj_list[rank])

        return result_dict

    def compute(self):
        output_dict = self.collect_outputs()
        result_dict = dict()
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return None

        start_time = time.time()
        pool = Pool(self.n_workers)
        sum_mAP = 0
        for cls_id in range(self.n_classes):

            fn = partial(self.compute_tp_fp)
            samples = list(zip(self.class_pred_lines[cls_id], self.class_gt_lines[cls_id], self.class_pred_scores[cls_id]))
            tp_fp_score_list = pool.starmap(fn, samples)

            result_dict[cls_id] = dict()
            sum_AP = 0

            for thr in self.distance_thresholds:
                tp_fp_score = [i[thr] for i in tp_fp_score_list]
                tp_fp_score = np.vstack(tp_fp_score)  # (num_dets, 3)
                sort_inds = np.argsort(-tp_fp_score[:, -1])
                tp = tp_fp_score[sort_inds, 0]  # (num_dets,)
                fp = tp_fp_score[sort_inds, 1]  # (num_dets,)
                tp = np.cumsum(tp, axis=0)
                fp = np.cumsum(fp, axis=0)
                eps = np.finfo(np.float32).eps
                recalls = tp / np.maximum(output_dict['num_gts'][cls_id], eps)
                precisions = tp / np.maximum((tp + fp), eps)

                AP = average_precision(recalls, precisions)
                sum_AP += AP
                result_dict[cls_id].update({f'AP@{thr}': AP})

            AP = sum_AP / len(self.distance_thresholds)
            sum_mAP += AP
            result_dict[cls_id].update({f'AP': AP})

        mAP = sum_mAP / self.n_classes
        result_dict['mAP'] = mAP
        pool.close()

        print(f"finished in {datetime.timedelta(seconds=time.time() - start_time)}")

        return result_dict

    def reset(self):
        self.class_pred_lines = {cls_id: [] for cls_id in range(self.n_classes)}
        self.class_pred_scores = {cls_id: [] for cls_id in range(self.n_classes)}
        self.class_gt_lines = {cls_id: [] for cls_id in range(self.n_classes)}

        self.class_num_gts = {cls_id: 0 for cls_id in range(self.n_classes)}
        self.class_num_pred = {cls_id: 0 for cls_id in range(self.n_classes)}


def merge_dicts(source_dict, new_dict):
    for key, value in source_dict.items():
        if type(value) == list:
            source_dict[key] += new_dict[key]
        elif type(value) == int:
            source_dict[key] += new_dict[key]
        elif type(value) == float:
            source_dict[key] += new_dict[key]
        elif type(value) == dict:
            source_dict[key] = merge_dicts(source_dict[key], new_dict[key])
        elif type(value) == np.ndarray:
            source_dict[key] = np.concatenate(source_dict[key], new_dict[key])
        else:
            raise Exception

    return source_dict


# TODO: check existence in tairvision, if exist, remove and use the code
def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision.

    Args:
        recalls (ndarray): shape (num_dets, )
        precisions (ndarray): shape (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float: calculated average precision
    """

    recalls = recalls[np.newaxis, :]
    precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = 0.

    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])

        ind = np.where(mrec[0, 1:] != mrec[0, :-1])[0]
        ap = np.sum(
            (mrec[0, ind + 1] - mrec[0, ind]) * mpre[0, ind + 1])

    elif mode == '11points':
        for thr in np.arange(0, 1 + 1e-3, 0.1):
            precs = precisions[0, recalls[i, :] >= thr]
            prec = precs.max() if precs.size > 0 else 0
            ap += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')

    return ap
