import torch
import numpy as np

from tairvision.eval_utils.openlanev2_evaluation.evaluate import pairwise, chamfer_distance, frechet_distance, evaluate
from tairvision.eval_utils.openlanev2_evaluation.utils import format_metric
from math import factorial
from scipy.spatial.distance import cdist
import cv2
from tairvision.models.bev.lss.utils.visualization import INSTANCE_COLOURS


traffic_element_list = [{
            'id': i,
            'category': 0,
            'attribute': i,
            'points': np.asarray([
                [1, 2],
                [3, 4],
            ]),
            'confidence': 0.0,
        } for i in range(13)]


def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def calculate_and_print_results(trainer, results_dict_openlanev2, metric_ap, panoptic_metrics, iou_metrics, evaluation_ranges):

    results_openlanev2 = evaluate(
        ground_truth=f'{trainer.cfg.DATASET.DATAROOT}/{trainer.cfg.DATASET.VERSION}_val.pkl', 
        predictions=results_dict_openlanev2,
    )

    format_metric(results_openlanev2)

    if trainer.model.head2d is not None:
        print('2D AP')
        result_dict = metric_ap.compute()
        for key, score in result_dict.items():
            print(f'{key}: {100 * score: .1f}')

    results = {}
    for key, grid in evaluation_ranges.items():
        panoptic_scores = panoptic_metrics[key].compute()
        for panoptic_key, value in panoptic_scores.items():
            results[f'{panoptic_key}'] = results.get(f'{panoptic_key}', []) + [100 * value[1].item()]

        iou_scores = iou_metrics[key].compute()
        results['iou'] = results.get('iou', []) + [100 * iou_scores["iou"][1].item()]

    for panoptic_key in ['iou', 'pq', 'sq', 'rq']:
        print(panoptic_key)
        print(' & '.join([f'{x:.1f}' for x in results[panoptic_key]]))

    return results_openlanev2


def predict_2d(trainer, output, pred_list_batch, batch_size,
            orig_target_sizes, filtered_te_implementaion, threshold_2d, 
            enable_relationship_queries, enable_topk_selection, num_topk, 
            ):
    output['head2d'].update({"orig_target_sizes": orig_target_sizes.repeat(batch_size, 1)})
    
    if enable_relationship_queries:
        input_predict2d = {}
        for key, value in output["outputs_filtered_te"].items():
            input_predict2d[key] = value

        input_predict2d["quer_feat"] = input_predict2d["quer_feat_relations"][-1]
        input_predict2d["orig_target_sizes"] = output['head2d']["orig_target_sizes"]
        input_predict2d["target_sizes"] = output['head2d']["target_sizes"]
    else:
        input_predict2d = {}
        for key, value in output["head2d"].items():
            input_predict2d[key] = value

    keep_2d = output["head2d"]["pred_logits"].sigmoid().max(dim=-1)[0] > threshold_2d

    # if filtered_te_implementaion:
    #     num_queries = trainer.cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.NUMBER_OF_SELECTED_QUERIES_TE
    # else:
    #     num_queries = 100

    num_queries = num_topk

    # if trainer.cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_IMPROVED_RELATIONS:
    #     output['outputs_filtered_te'].update({"orig_target_sizes": orig_target_sizes.repeat(batch_size, 1)})
    #     output['outputs_filtered_te'].update({"target_sizes": output['head2d']["target_sizes"]})
    #     num_queries = trainer.cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.NUMBER_OF_SELECTED_QUERIES_TE
    #     output['outputs_filtered_te']["quer_feat"] = output['outputs_filtered_te']["quer_feat"][-1]
    #     input_predict2d = output['outputs_filtered_te']

    post_outputs_2d = trainer.model.head2d.postprocess(
        input_predict2d,
        num_queries=num_queries, 
        select_topk=enable_topk_selection
    )

    if enable_topk_selection:
        keep_2d = torch.ones((batch_size, num_queries), dtype=torch.bool, device=output['head2d']["pred_logits"].device)

    output.update({"post_outputs_2d": post_outputs_2d})
    output.update({"keep_2d": keep_2d})
    for i in range(len(post_outputs_2d)):
        pred_list = []
        post_output = post_outputs_2d[i]
        pred_labels = post_output['labels'][keep_2d[i]].detach().cpu().numpy()
        pred_scores = post_output['scores'][keep_2d[i]].detach().cpu().numpy()
        pred_boxes = post_output['boxes'][keep_2d[i]].detach().cpu().numpy()
        for pred_index, (label, score, box) in enumerate(zip(pred_labels, pred_scores, pred_boxes)):
            if label in [1, 2, 3]:
                category_label = 1
            else:
                category_label = 0

            pred_sample = {
                'id': pred_index,
                'category': category_label,
                'attribute': label.astype(np.int8),
                'points': box.reshape(2, 2).astype(np.int16),
                'confidence': score.astype(np.float16),
            } 
            
            pred_list.append(pred_sample)
        pred_list_batch.append(pred_list)

def update_result_dictionary(results_dict_openlanev2, result_key_list, lane_centerline_list_pred_batch,
                              pred_list_batch, score_list_lclc_batch, score_list_lcte_batch):
    
    assert len(result_key_list) == len(lane_centerline_list_pred_batch), "result_key_list and lane_centerline_list_all_frames should have the same length"
    for index, result_key in enumerate(result_key_list):
        
        lane_centerline_list_pred = lane_centerline_list_pred_batch[index]
        
        if len(pred_list_batch) > 0:
            traffic_element_list_from_batch_list = pred_list_batch[index]
        else:
            traffic_element_list_from_batch_list = traffic_element_list
        number_of_lc = len(lane_centerline_list_pred)
        number_of_te = len(traffic_element_list_from_batch_list)

        if len(score_list_lclc_batch) > 0:
            topology_lclc = score_list_lclc_batch[index]
        else:
            topology_lclc = np.zeros([number_of_lc, number_of_lc])

        if len(score_list_lcte_batch) > 0:
            topology_lcte = score_list_lcte_batch[index]
        else:
            topology_lcte = np.zeros([number_of_lc, number_of_te])

        results_dict_openlanev2['results'][result_key] = {
        'predictions': {
            'lane_centerline': lane_centerline_list_pred,
            'traffic_element': traffic_element_list_from_batch_list,
            'topology_lclc': topology_lclc,
            'topology_lcte': topology_lcte,
            }
        }

def update_other_metrics(trainer, output, batch, metric_ap, panoptic_metrics, iou_metrics, device, evaluation_ranges):
    post_out = output["post_out_lc"]
    targets2d_raw = batch["targets2d"]
    targets2d_metrics = []
    for i in range(len(targets2d_raw)):
        targets2d_metrics_dict = {
            "boxes": targets2d_raw[i][0]["correctly_resized_boxes"].to(device),
            "labels": targets2d_raw[i][0]["labels"].to(device), 
            "areas": targets2d_raw[i][0]["areas"].to(device),
        }
        targets2d_metrics.append(targets2d_metrics_dict)

    labels = trainer.prepare_targets(batch)

    if trainer.model.head2d is not None:
        metric_ap(output["post_outputs_2d"], targets2d_metrics)

    pred_consistent_instance_seg = post_out["instance"]
    segmentation_pred = post_out["semantic"]

    for key, grid in evaluation_ranges.items():
        limits1 = slice(grid[0], grid[1])
        limits2 = slice(grid[2], grid[3])
        panoptic_metrics[key](pred_consistent_instance_seg[..., limits1, limits2].contiguous().detach(),
                            labels['instance'][..., limits1, limits2].contiguous()
                            )

        iou_metrics[key](segmentation_pred[..., limits1, limits2].contiguous(),
                        labels['segmentation'][..., limits1, limits2].contiguous()
                        )

def predict_relations(trainer, output, score_list_lclc_batch, score_list_lcte_batch, 
                      lclc_sinkhorn_threshold, lcte_sinkhorn_threshold, 
                      enable_after_centerline_pred, enable_relationship_queries, enable_distance_prior):
    post_out = output["post_out_lc"]
    keep2d = output["keep_2d"]
    for batch_idx, keep in enumerate(post_out["keep"]):
        keep_of_keep = post_out["keep_of_keep"][batch_idx]

        # if enable_relationship_queries:
        #     quer_feat = output["outputs_filtered_lc"]["quer_feat_relations"][-1][batch_idx][keep][keep_of_keep][None, :]
        # else:
        #     quer_feat = output["quer_feat"][batch_idx][keep][keep_of_keep][None, :]

        # if enable_after_centerline_pred:
        #     pred_masks = output["centerline_after_relation"]["pred_masks"][batch_idx][keep][keep_of_keep][None, :]
        #     pred_logits = output["centerline_after_relation"]["pred_logits"][batch_idx][keep][keep_of_keep][None, :]
        # else:
        #     pred_masks = output["pred_masks"][batch_idx][keep][keep_of_keep][None, :]
        #     pred_logits = output["pred_logits"][batch_idx][keep][keep_of_keep][None, :]

        quer_feat = post_out["quer_feat"][batch_idx][None, ]
        pred_masks = post_out["pred_masks"][batch_idx][None, ]
        pred_logits = post_out["pred_logits"][batch_idx][None, ]
        labels = post_out["labels"][batch_idx][None, :]
        labels_numpy = labels[0].cpu().numpy()

        # labels_lc = post_out["labels"][batch_idx][None, :] - 1
        # labels_lc = output["labels"][batch_idx][keep][keep_of_keep][None, :]
        outputs_filtered = {
            "quer_feat": quer_feat,
            "pred_logits": pred_logits, 
            "pred_masks": pred_masks,
            # "labels": labels_lc
        }
        # count1 = output["quer_feat"][batch_idx][keep][keep_of_keep][None, :].shape[1]
        # count2 = len(post_out["attribute"][batch_idx].values())
        # if count1 != count2:
        #     print("asdasd")


        start_points = post_out["result_set_numpy_concat"][batch_idx][:, 0]
        end_points = post_out["result_set_numpy_concat"][batch_idx][:, -1]
        distances = cdist(end_points, start_points, p=1)

        new_output = {}
        new_output["outputs_filtered_lc"] = outputs_filtered

        scores = trainer.model.optimal_transport_lclc(new_output)
        if trainer.model.optimal_transport_lclc.sinkhorn_activated:
            scores_shaped = scores[0, :-1, :-1]
            score_filtered_base = scores_shaped > lclc_sinkhorn_threshold

        else:
            scores_shaped = scores[0]
            score_filtered_base = scores_shaped > 0.05

        score_filtered_base = score_filtered_base.detach().cpu().numpy()
        # score_filtered[distances < 1] = 0.45
        # score_filtered[distances < 2] = 0.40
        # score_filtered[distances < 3] = 0.35
        score_filtered = score_filtered_base * 1.0

        # score_filtered = np.zeros_like(score_filtered_base)
        # score_filtered[score_filtered_base] = 0.50

        scores_shaped_numpy = scores_shaped.detach().cpu().numpy()
        score_filtered = score_filtered + scores_shaped_numpy
        # score_filtered[distances > 3] = score_filtered[distances > 3] - 1.0

        # score_filtered[distances < 3] = score_filtered[distances < 3] + np.exp(-distances * 0.2)[distances < 3]
        if enable_distance_prior:
            score_filtered[distances < 1] = score_filtered[distances < 1] + 1.0
            score_filtered[distances < 2] = score_filtered[distances < 2] + 0.9
            score_filtered[distances < 3] = score_filtered[distances < 3] + 0.8
            score_filtered[distances < 4] = score_filtered[distances < 4] + 0.7

        score_list_lclc_batch.append(score_filtered.astype(np.float16))

        if trainer.model.optimal_transport_lcte is not None:
            keep2d_batch = keep2d[batch_idx]
            # head2d_out_quer_feat = {"quer_feat": output["head_2d"]["quer_feat"][batch_idx][None, :]}
            head2d_out_quer_feat = {"quer_feat": output["post_outputs_2d"][batch_idx]["quer_feat"][keep2d_batch][None, :]}
            new_output["outputs_filtered_te"] = head2d_out_quer_feat
            scores = trainer.model.optimal_transport_lcte(new_output)

            if trainer.model.optimal_transport_lcte.sinkhorn_activated:
                scores_shaped = scores[0, :-1, :-1]
                score_filtered = scores_shaped > lcte_sinkhorn_threshold
            else:
                scores_shaped = scores[0]
                score_filtered = scores_shaped > 0.05

            score_filtered = score_filtered.detach().cpu().numpy()
            score_filtered[end_points[:, 0] < -8] = False
            score_filtered[end_points[:, 0] > 60] = False
            score_filtered[end_points[:, 1] > 15] = False
            score_filtered[end_points[:, 1] < -15] = False
            score_filtered[labels_numpy != 1] = False
            score_filtered = score_filtered * 1.0
            score_filtered = score_filtered + scores_shaped.detach().cpu().numpy()
            # score_filtered[labels_numpy != 1] = score_filtered[labels_numpy != 1] - 2.0
            score_list_lcte_batch.append(score_filtered.astype(np.float16))

def predict_relations_sinkhorn_fixed(trainer, output, score_list_lclc_batch, score_list_lcte_batch, lclc_sinkhorn_threshold, lcte_sinkhorn_threshold):
    post_out = output["post_out_lc"]
    keep2d = output["keep_2d"]
    for batch_idx, keep in enumerate(post_out["keep"]):
        keep_of_keep = post_out["keep_of_keep"][batch_idx]

        lc_out = output["lclc_scores"][-1][batch_idx, :-1, :-1]
        lc_out1 = lc_out[keep][keep_of_keep]
        scores = lc_out1[:, keep][:, keep_of_keep]

        start_points = post_out["result_set_numpy_concat"][batch_idx][:, 0]
        end_points = post_out["result_set_numpy_concat"][batch_idx][:, -1]
        distances = cdist(end_points, start_points, p=1)

        if trainer.model.optimal_transport_lclc.sinkhorn_activated:
            score_filtered = scores > lclc_sinkhorn_threshold
            score_filtered = score_filtered.detach().cpu().numpy()
            score_filtered = score_filtered * 0.50
            score_filtered[distances > 6] = 0
            score_filtered = score_filtered + scores.detach().cpu().numpy()
            score_list_lclc_batch.append(score_filtered)
        else:
            scores_sigmoid = scores.sigmoid().detach().cpu().numpy()
            score_filtered = scores_sigmoid > 0.5
            score_filtered = score_filtered * 1
            score_list_lclc_batch.append(score_filtered[0])

        if trainer.model.optimal_transport_lcte is not None:
            te_out = output["lcte_scores"][-1][batch_idx, :-1, :-1]
            keep2d_batch = keep2d[batch_idx]
            te_out1 = te_out[keep][keep_of_keep]
            scores = te_out1[:, keep2d_batch]

            if trainer.model.optimal_transport_lcte.sinkhorn_activated:
                score_filtered = scores > lcte_sinkhorn_threshold
                score_filtered = score_filtered.detach().cpu().numpy()
                score_filtered = score_filtered * 0.5
                score_filtered = score_filtered + scores.detach().cpu().numpy()
                score_list_lcte_batch.append(score_filtered)
            else:
                scores_sigmoid = scores.sigmoid().detach().cpu().numpy()
                score_filtered = scores_sigmoid > 0.5
                score_filtered = score_filtered * 1
                score_list_lcte_batch.append(score_filtered[0])

def predict_centerlines_bezier(output, batch, lane_centerline_list_pred_batch):
    # print("bezier_pred")
    post_out = output["post_out_lc"]
    point_dict_list = post_out["point_dict"]
    confidence_dict_list = post_out["confidence"]
    attribute_dict_list = post_out["attribute"]

    result_set_numpy_batch_list = []

    batch_size = batch['view'].shape[0]
    for batch_idx in range(batch_size):
        keep = post_out["keep"][batch_idx]
        keep_of_keep = post_out["keep_of_keep"][batch_idx]
        # bezier_regressions = output["pred_regressions"][batch_idx][keep][keep_of_keep][None, :]
        bezier_regressions = post_out["pred_regressions"][batch_idx][None, :]
        lanes = bezier_regressions.reshape(-1, bezier_regressions.shape[-1] // 3, 3)

        result_set_numpy_list = []
        
        n_points = 11
        n_control = lanes.shape[1]
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        bezier_A = torch.tensor(A, dtype=torch.float32, device=lanes.device)
        lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
        lanes = lanes.cpu().numpy()

        confidence_dict = confidence_dict_list[batch_idx]
        lane_centerline_list = []
        view_inv = batch['view'][batch_idx, 0, 0].inverse()
        view_inv = view_inv.cpu().numpy()

        for lane_index, lane in enumerate(lanes):
            confidence = confidence_dict[lane_index + 1]
            lane_unprojected = np.concatenate([lane, np.ones((lane.shape[0], 1))], axis=1)
            unprojected_centerlines = view_inv @ lane_unprojected.T
            result_set_numpy_bezier = unprojected_centerlines.T[:, :3]

            # result_set_numpy_bezier[:, 2] = 0

            lane_centerline_sample_result_dict = {
                    'id': (batch_idx + 1) * 1000 + lane_index,
                    'points': result_set_numpy_bezier,
                    'confidence': confidence,
            }
            result_set_numpy_list.append(result_set_numpy_bezier[None, :])
            lane_centerline_list.append(lane_centerline_sample_result_dict)

        lane_centerline_list_pred_batch.append(lane_centerline_list)
        result_set_numpy_batch_list.append(np.concatenate(result_set_numpy_list, axis=0))

    output["post_out_lc"]["result_set_numpy_concat"] = result_set_numpy_batch_list

def predict_centerlines(output, batch, lane_centerline_list_pred_batch):
    post_out = output["post_out_lc"]
    # print("mask_pred")
    point_dict_list = post_out["point_dict"]
    confidence_dict_list = post_out["confidence"]
    attribute_dict_list = post_out["attribute"]

    result_set_numpy_batch_list = []
    
    batch_size = batch['view'].shape[0]
    for batch_idx in range(batch_size):
        lane_centerline_list = []
        view_inv = batch['view'][batch_idx, 0, 0].inverse()

        # z_info = batch["z_info"][i][0][0].detach()
        point_dict = point_dict_list[batch_idx]
        confidence_dict = confidence_dict_list[batch_idx]
        attribute_dict = attribute_dict_list[batch_idx]
        result_set_numpy_list = []

        for id, point_set in point_dict.items():                            
            confidence = confidence_dict[id]
            attribute = attribute_dict[id]
            indices = point_set.long()
            # z_to_cat = z_info[indices[1, :], indices[0, :]]
            point_set = torch.cat([point_set, torch.zeros((1, point_set.shape[1])).to(point_set.device)], dim=0)
            point_set = torch.cat([point_set, torch.ones((1, point_set.shape[1])).to(point_set.device)], dim=0)
            result_set = view_inv @ point_set
            result_set[0:3, :] = result_set[0:3, :] / result_set[3, :]
            result_set_numpy = result_set[0:3, :].cpu().numpy().transpose()
            
            # indices = result_set_numpy.astype(np.int64)
            # z_numpy = z_to_cat.cpu().numpy()
            # result_set_numpy[:, 2] = z_numpy

            # result_set_numpy = result_set_numpy[result_set_numpy[:, 0].argsort()[::-1]]
            number_of_points = 11
            if attribute == 1 or attribute == 2:
                params = np.polyfit(result_set_numpy[:, 0], result_set_numpy[:, 1], 2)
                poly_eqn = np.poly1d(params)

                unique_range = np.linspace(result_set_numpy[:, 0].min(), result_set_numpy[:, 0].max(), number_of_points)
                predicted = poly_eqn(unique_range)
                unique_range = unique_range.reshape(-1, 1)
                predicted = predicted.reshape(-1, 1)
                zeros = np.zeros_like(unique_range)
                new_out = np.hstack([unique_range, predicted, zeros])
                result_set_numpy = new_out
            else:
                params = np.polyfit(result_set_numpy[:, 1], result_set_numpy[:, 0], 2)
                poly_eqn = np.poly1d(params)

                unique_range = np.linspace(result_set_numpy[:, 1].min(), result_set_numpy[:, 1].max(), number_of_points)
                predicted = poly_eqn(unique_range)
                unique_range = unique_range.reshape(-1, 1)
                predicted = predicted.reshape(-1, 1)
                zeros = np.zeros_like(unique_range)
                new_out = np.hstack([predicted, unique_range, zeros])
                result_set_numpy = new_out

            if attribute == 1:
                result_set_numpy = result_set_numpy[result_set_numpy[:, 0].argsort()]
            elif attribute == 2:
                result_set_numpy = result_set_numpy[result_set_numpy[:, 0].argsort()[::-1]]
            elif attribute == 3:
                result_set_numpy = result_set_numpy[result_set_numpy[:, 1].argsort()]
            elif attribute == 4:
                result_set_numpy = result_set_numpy[result_set_numpy[:, 1].argsort()[::-1]]

            # result_set_numpy = result_set_numpy[::20]
            lane_centerline_sample_result_dict = {
                    'id': (batch_idx + 1) * 1000 + id,
                    'points': result_set_numpy.astype(np.float16),
                    'confidence': confidence,
                }
            
            result_set_numpy_list.append(result_set_numpy[None, :])
            lane_centerline_list.append(lane_centerline_sample_result_dict)
            
        lane_centerline_list_pred_batch.append(lane_centerline_list)
        result_set_numpy_batch_list.append(np.concatenate(result_set_numpy_list, axis=0))

    output["post_out_lc"]["result_set_numpy_concat"] = result_set_numpy_batch_list


def predict_centerlines_bezier_mask_fusion(output, batch, lane_centerline_list_pred_batch):
    # print("mask_with_bezier_pred_range_from_bezier")
    post_out = output["post_out_lc"]
    point_dict_list = post_out["point_dict"]
    confidence_dict_list = post_out["confidence"]
    attribute_dict_list = post_out["attribute"]

    result_set_numpy_batch_list = []

    batch_size = batch['view'].shape[0]
    for batch_idx in range(batch_size):
        lane_centerline_list = []
        view_inv = batch['view'][batch_idx, 0, 0].inverse()

        # z_info = batch["z_info"][i][0][0].detach()
        point_dict = point_dict_list[batch_idx]
        confidence_dict = confidence_dict_list[batch_idx]
        attribute_dict = attribute_dict_list[batch_idx]

        keep = post_out["keep"][batch_idx]
        keep_of_keep = post_out["keep_of_keep"][batch_idx]
        # bezier_regressions = output["pred_regressions"][batch_idx][keep][keep_of_keep][None, :]
        bezier_regressions = post_out["pred_regressions"][batch_idx][None, :]
        lanes = bezier_regressions.reshape(-1, bezier_regressions.shape[-1] // 3, 3)
        
        n_points = 11
        n_control = lanes.shape[1]
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        bezier_A = torch.tensor(A, dtype=torch.float32, device=lanes.device)
        lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
        lanes = lanes.cpu().numpy()

        confidence_dict = confidence_dict_list[batch_idx]
        lane_centerline_list = []
        view_inv = batch['view'][batch_idx, 0, 0].inverse()
        view_inv_npy = view_inv.cpu().numpy()

        result_set_numpy_list = []

        for id, point_set in point_dict.items():                            
            confidence = confidence_dict[id]
            attribute = attribute_dict[id]
            indices = point_set.long()
            # z_to_cat = z_info[indices[1, :], indices[0, :]]
            point_set = torch.cat([point_set, torch.zeros((1, point_set.shape[1])).to(point_set.device)], dim=0)
            point_set = torch.cat([point_set, torch.ones((1, point_set.shape[1])).to(point_set.device)], dim=0)
            result_set = view_inv @ point_set
            result_set[0:3, :] = result_set[0:3, :] / result_set[3, :]
            result_set_numpy = result_set[0:3, :].cpu().numpy().transpose()

            lane = lanes[id - 1]
            lane_unprojected = np.concatenate([lane, np.ones((lane.shape[0], 1))], axis=1)
            unprojected_centerlines = view_inv_npy @ lane_unprojected.T
            result_set_numpy_bezier = unprojected_centerlines.T[:, :3]

            # Utilize this code part for checking the monotonicity of the bezier curve
            # x_monotonicity_check_1 = np.sum(result_set_numpy_bezier[:, 0][:-1] - result_set_numpy_bezier[:, 0][1:] < 0)
            # x_monotonicity_check_2 = np.sum(result_set_numpy_bezier[:, 0][:-1] - result_set_numpy_bezier[:, 0][1:] > 0)
            # y_monotonicity_check_1 = np.sum(result_set_numpy_bezier[:, 1][:-1] - result_set_numpy_bezier[:, 1][1:] < 0)
            # y_monotonicity_check_2 = np.sum(result_set_numpy_bezier[:, 1][:-1] - result_set_numpy_bezier[:, 1][1:] > 0)
            # x_monotonicity_check = np.min([x_monotonicity_check_1, x_monotonicity_check_2])
            # y_monotonicity_check = np.min([y_monotonicity_check_1, y_monotonicity_check_2])
            # monotonicity_check_two_directions = np.min([x_monotonicity_check, y_monotonicity_check])
            # if monotonicity_check_two_directions < 1:

            number_of_points = 11
            if attribute == 1 or attribute == 2:
                params = np.polyfit(result_set_numpy[:, 0], result_set_numpy[:, 1], 2)
                poly_eqn = np.poly1d(params)

                unique_range = np.linspace(result_set_numpy[:, 0].min(), result_set_numpy[:, 0].max(), number_of_points)
                # unique_range = result_set_numpy_bezier[:, 0]

                predicted = poly_eqn(unique_range)
                unique_range = unique_range.reshape(-1, 1)
                predicted = predicted.reshape(-1, 1)
                zeros = np.zeros_like(unique_range)
                new_out = np.hstack([unique_range, predicted, zeros])
                result_set_numpy = new_out
            else:
                params = np.polyfit(result_set_numpy[:, 1], result_set_numpy[:, 0], 2)
                poly_eqn = np.poly1d(params)

                unique_range = np.linspace(result_set_numpy[:, 1].min(), result_set_numpy[:, 1].max(), number_of_points)
                # unique_range = result_set_numpy_bezier[:, 1]

                predicted = poly_eqn(unique_range)
                unique_range = unique_range.reshape(-1, 1)
                predicted = predicted.reshape(-1, 1)
                zeros = np.zeros_like(unique_range)
                new_out = np.hstack([predicted, unique_range, zeros])
                result_set_numpy = new_out

            if attribute == 1:
                result_set_numpy = result_set_numpy[result_set_numpy[:, 0].argsort()]
            elif attribute == 2:
                result_set_numpy = result_set_numpy[result_set_numpy[:, 0].argsort()[::-1]]
            elif attribute == 3:
                result_set_numpy = result_set_numpy_bezier
            elif attribute == 4:
                result_set_numpy = result_set_numpy_bezier

            # result_set_numpy[:, 2] = result_set_numpy_bezier[:, 2]
            # result_set_numpy = result_set_numpy / 2 + result_set_numpy_bezier / 2

            lane_centerline_sample_result_dict = {
                    'id': (batch_idx + 1) * 1000 + id,
                    'points': result_set_numpy,
                    'confidence': confidence,
                }
            
            lane_centerline_list.append(lane_centerline_sample_result_dict)
            result_set_numpy_list.append(result_set_numpy[None, :])

        lane_centerline_list_pred_batch.append(lane_centerline_list)
        result_set_numpy_batch_list.append(np.concatenate(result_set_numpy_list, axis=0))

    output["post_out_lc"]["result_set_numpy_concat"] = result_set_numpy_batch_list


def test_gt_performance(batch, lane_centerline_list_pred_batch):

    batch_size = batch['view'].shape[0]
    centerlines_gt = batch['centerlines_list']
    for i in range(batch_size):
        lane_centerline_list = []

        # For giving Ground Truth as prediction
        gt_points_list = centerlines_gt[i][0]
        gt_point_dict = {gt_point_dict["id"]: gt_point_dict["points"] for gt_point_dict in gt_points_list}
        for id, point_set in gt_point_dict.items():   
            # Ground Truth rule check 
            x_monotonicity_check_1 = np.sum(point_set[:, 0][:-1] - point_set[:, 0][1:] < 0)
            x_monotonicity_check_2 = np.sum(point_set[:, 0][:-1] - point_set[:, 0][1:] > 0)
            y_monotonicity_check_1 = np.sum(point_set[:, 1][:-1] - point_set[:, 1][1:] < 0)
            y_monotonicity_check_2 = np.sum(point_set[:, 1][:-1] - point_set[:, 1][1:] > 0)
            x_monotonicity_check = np.min([x_monotonicity_check_1, x_monotonicity_check_2])
            y_monotonicity_check = np.min([y_monotonicity_check_1, y_monotonicity_check_2])
            if x_monotonicity_check < y_monotonicity_check:
                if point_set[:, 0][0] - point_set[:, 0][-1] <= 0:
                    attribute = 1
                else:
                    attribute = 2

                if x_monotonicity_check > 0:
                    print(f"x_whoaa: {x_monotonicity_check}")
            elif x_monotonicity_check > y_monotonicity_check:
                if point_set[:, 1][0] - point_set[:, 1][-1] <= 0:
                    attribute = 3
                else:
                    attribute = 4
                if y_monotonicity_check > 0:
                    print(f"y_whoaa: {y_monotonicity_check}")
            else:
                x_movement = np.abs(point_set[:, 0][0] - point_set[:, 0][-1])
                y_movement = np.abs(point_set[:, 1][0] - point_set[:, 1][-1])
                if x_movement > y_movement:
                    if point_set[:, 0][0] - point_set[:, 0][-1] <= 0:
                        attribute = 1
                    else:
                        attribute = 2
                else:
                    if point_set[:, 1][0] - point_set[:, 1][-1] <= 0:
                        attribute = 3
                    else:
                        attribute = 4

            if attribute == 1:
                point_set = point_set[point_set[:, 0].argsort()]
            elif attribute == 2:
                point_set = point_set[point_set[:, 0].argsort()[::-1]]
            elif attribute == 3:
                point_set = point_set[point_set[:, 1].argsort()]
            elif attribute == 4:
                point_set = point_set[point_set[:, 1].argsort()[::-1]]
                
            point_set[:, 2] = 0
            lane_centerline_sample_result_dict = {
                    'id': id,
                    'points': point_set,
                    'confidence': 1,
                }
            
            lane_centerline_list.append(lane_centerline_sample_result_dict)
        lane_centerline_list_pred_batch.append(lane_centerline_list)


def visualize_pred_gt(prediction_points, gt_points, view):
    multiplier = 3
    for batch_idx in range(len(gt_points)):
        gt_map = np.zeros((400*multiplier, 208*multiplier))
        for gt_idx, gt_lane in  enumerate(gt_points[batch_idx][0]):
            gt_lane = gt_lane.astype('int') * multiplier
            for start_point, end_point in zip(gt_lane[0, :-1, :2], gt_lane[0, 1:, :2]):
                gt_map = cv2.arrowedLine(gt_map, start_point, end_point,
                                        gt_idx, 1, tipLength=0.5)

        pred_map = np.zeros((400 * multiplier, 208 * multiplier))
        for pred_idx, pred_lane in enumerate(prediction_points[batch_idx]):
            pred_lane = np.pad(pred_lane, ((0, 0), (0, 1)), constant_values=1)
            pred_lane = view[batch_idx, 0, 0].cpu().numpy() @ pred_lane.T
            pred_lane = pred_lane[:3, :].T.astype('int') * multiplier
            for start_point, end_point in zip(pred_lane[:-1, :2], pred_lane[1:, :2]):
                pred_map = cv2.arrowedLine(pred_map, start_point, end_point,
                                        pred_idx, 1, tipLength=0.5)

        out_gt_map = get_bitmap(gt_map.astype('int'), gt_map.shape)
        cv2.namedWindow('gt_map', cv2.WINDOW_NORMAL)
        cv2.imshow('gt_map', out_gt_map)

        out_pred_map = get_bitmap(pred_map.astype('int'), pred_map.shape)
        cv2.namedWindow('pred_map', cv2.WINDOW_NORMAL)
        cv2.imshow('pred_map', out_pred_map)
        cv2.waitKey()

def get_bitmap(self, x, bev_size=(200, 200)):
    output = np.zeros((*bev_size, 3)).astype(np.uint8)

    x_mask = (x > 0) * (x < 70)

    colors = INSTANCE_COLOURS[x[x_mask]]
    output[x_mask] = colors  # [255, 172, 28]
    center_x = bev_size[0] // 2
    center_y = bev_size[1] // 2
    output[center_x-5:center_x+5, center_y-3:center_y+3] = [52, 152, 219]

    return output

def apply_chamfer_nms(output, batch_size, chamfer_nms_distance, lane_centerline_list_pred_batch):
    for batch_idx in range(batch_size):
        lanes = [elem for elem in output["post_out_lc"]["result_set_numpy_concat"][batch_idx]]
        result_pairwise = pairwise(
            lanes, lanes,
            frechet_distance,
            relax=True,
        )

        remove_indices = []
        accept_indices = []
        confidences = output["post_out_lc"]["confidence"][batch_idx]
        number_of_lanes = len(lanes)
        if number_of_lanes > 0:
            for i in range(number_of_lanes):
                for j in range(i+1, number_of_lanes):
                    if result_pairwise[i][j] < chamfer_nms_distance:
                        if confidences[i+1] > confidences[j+1]:
                            remove_indices.append(j)
                        else:
                            remove_indices.append(i)

        for i in range(number_of_lanes):
            if i not in remove_indices:
                accept_indices.append(i)

        lane_list = [lanes[i][None, ] for i in accept_indices]
        nms_lanes = np.concatenate(lane_list, axis=0)
        output["post_out_lc"]["result_set_numpy_concat"][batch_idx] = nms_lanes
        lane_centerline_list_pred_batch[batch_idx] = [
            lane_centerline_list_pred_batch[batch_idx][i] for i in accept_indices]
        
        keep_positions = np.array(output['post_out_lc']['keep_of_keep'][batch_idx]).nonzero()[0]
        for k in remove_indices:
            output['post_out_lc']['keep_of_keep'][batch_idx][keep_positions[k]] = False

