import torch
import torch.nn.functional as F


def post_process_softmax(outputs, object_mask_threshold, overlap_threshold, number_of_classes):
    """
    Post process function for the original mask2former model code. 
    However, the code can be applied to any transformer-based segmentation model. 
    It also includes some study to predict z and yaw angle prediction to adapt the segmentation model to the 3D detection task. 

    Args:
        outputs (dict): the outputs that will be post processed. 
        object_mask_threshold (float): The score threshold for the object mask
        overlap_threshold (float): The overlap threshold concept that is special to this post processing. 
        This concept aims to obtain confident masks. It tries to filter masks with lower threshold even if it has the maximum score.
        number_of_classes (int): the number of classes
    """
    mask_cls_batch = outputs["pred_logits"].detach()
    mask_pred_batch = outputs["pred_masks"].detach()

    # This line assumes that the training is implemented with softmax function
    soft_batch_outs = F.softmax(mask_cls_batch, dim=-1)

    scores_batch, labels_batch = soft_batch_outs.max(-1)
    # Last class is no object class therefore can be ignored. 
    score_segm_outs = soft_batch_outs[..., :-1]

    # Mask loss is still implemented with sigmoid function
    mask_pred_batch = mask_pred_batch.sigmoid()

    h, w = mask_pred_batch.shape[-2:]
    batch_size = mask_pred_batch.shape[0]
    instance_seg_batch = torch.zeros(batch_size, 1, number_of_classes - 1, h, w, 
                                     dtype=torch.int64, device=mask_pred_batch.device, requires_grad=False)
    
    # zhs study aims to predict the z as mask but this concept is open to modification and not completed
    if "pred_zhs" in outputs:
        zhs_batch = outputs["pred_zhs"].detach()
        yaws_batch = outputs["pred_yaws"].detach()
        zhs_map_batch = torch.zeros(batch_size, 1, 2, h, w, dtype=torch.float32, device=mask_pred_batch.device, requires_grad=False)

    # Multiply query probability with mask probability to get the final probability
    output_prob = torch.einsum("bqc,bqhw->bchw", score_segm_outs, mask_pred_batch)

    # This part gets the argmax of the output probability, yields semantic segmentation. 
    semantic_seg_batch = torch.argmax(output_prob, dim=1, keepdim=True)[:, None]
    for i in range(batch_size):
        labels = labels_batch[i]
        scores = scores_batch[i]
        mask_pred = mask_pred_batch[i]

        # There is also no object class in the query classification therefore negation exists
        keep = labels.ne(number_of_classes) & (scores > object_mask_threshold)

        # keep keyword is used to filter the masks with lower threshold even and no object class
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        if "pred_zhs" in outputs:
            cur_zhs = zhs_batch[i][keep]
            cur_yaws = yaws_batch[i][keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        # Instance segmentation is filled with instance IDs
        instance_seg = torch.zeros((number_of_classes - 1, h, w), dtype=torch.int32, device=cur_masks.device)
        if "pred_zhs" in outputs:
            map_zhs = torch.zeros((2, h, w), dtype=torch.float32, device=cur_masks.device)

        current_segment_id = 1
        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                if pred_class == 0 or pred_class > number_of_classes - 1:
                    continue
                mask_area = (cur_mask_ids == k).sum().item()

                # The original code utilizes 0.5, therefore it is also kept
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue

                    instance_seg[pred_class - 1, mask] = current_segment_id
                    current_segment_id += 1
                    if "pred_zhs" in outputs:
                        map_zhs[:, mask] = cur_zhs[k][:, None].repeat([1, mask.sum()])

        for class_index in range(number_of_classes - 1):
            instance_seg_batch[i, 0, class_index] = instance_seg[class_index]
    output_post = {"inst": instance_seg_batch, "segm": semantic_seg_batch}

    if "pred_zhs" in outputs:
        zhs_map_batch[i, 0] = map_zhs
        output_post.update({"zhs_map": zhs_map_batch})

    return output_post


def post_process_softmax_lane(
        outputs, object_mask_threshold, overlap_threshold, number_of_classes, 
        threshold1 = 0.5, threshold2 = 0.2, overlap_threshold2 = 0.0
    ):
    """
    This function is the post processing function for the centerline prediction task. 
    Conceptually, it can be also applied to lane divider concept. 

    This concept assumes that there are 4 direction label which are up, down, left, right. 
    Threshold1 and overplay_threshold are utilized for up and down directions
    Threshold2 and overlap_threshold2 are utilized for left and right directions. 

    In the experimental details, overlap_thresholds are found to be not useful for frechet-based mAP metrics. 

    The function also converts the masks to point set because centerline is defined or meaningful with points. 
    Masks are the middle step towards the point set.

    """

    mask_cls_batch = outputs["pred_logits"].detach()
    mask_pred_batch_unprocessed = outputs["pred_masks"].detach()
    quer_features_batch = outputs["quer_feat"].detach()

    # pred regressions are for the output formats such as Bezier Curves. 
    if outputs["pred_regressions"] is not None:
        pred_regressions_batch = outputs["pred_regressions"].detach()
    soft_batch_outs = F.softmax(mask_cls_batch, dim=-1)

    # This part is added in order to increase the number of predictions for centerline predictions. 
    # This part is added in order to take the second best class prediction into account.
    scores_batch, labels_batch = soft_batch_outs.max(-1)
    bool_of_max = soft_batch_outs == scores_batch[:, :, None]

    # Set the max value to 0, so that take the second max. 
    soft_batch_outs[bool_of_max] = 0
    scores_second_batch, labels_second_batch = soft_batch_outs.max(-1)

    # Last class is for no object class, therefore it is not taken into account.
    score_segm_outs = soft_batch_outs[..., :-1]

    # sigmoid function is applied to the mask prediction. 
    mask_pred_batch = mask_pred_batch_unprocessed.sigmoid()

    h, w = mask_pred_batch.shape[-2:]
    batch_size = mask_pred_batch.shape[0]
    instance_seg_batch = torch.zeros(batch_size, 1, number_of_classes - 1, h, w, 
                                     dtype=torch.int64, device=mask_pred_batch.device, requires_grad=False)
    
    output_prob = torch.einsum("bqc,bqhw->bchw", score_segm_outs, mask_pred_batch)
    semantic_seg_batch = torch.argmax(output_prob, dim=1, keepdim=True)[:, None]

    # Centerline concept has lots of complexities compared to standard segmentation task. Therefore, this function returns additional lists. 
    point_dict_list = []
    confidence_list = []
    attributes_list = []
    keep_list = []
    keep_of_keep_list = []
    labels_list = []

    pred_masks_batch_list = []
    pred_logits_batch_list = []
    pred_queries_batch_list = []
    pred_regressions_batch_list = []

    # Every batch is processed separately.. 
    for i in range(batch_size):
        # TODO, when utilized for deployment or in product, second best concept might not be beneficial. 
        # Therefore, this part is commented for further development purposes. 
        # labels = labels_batch[i]
        # scores = scores_batch[i]
        # mask_pred = mask_pred_batch[i]
        # mask_pred_unprocessed = mask_pred_batch_unprocessed[i]
        # mask_cls = mask_cls_batch[i]
        # quer_features = quer_features_batch[i]
        # pred_regressions = pred_regressions_batch[i]

        # Second best prediction handling. 
        labels = torch.cat([labels_batch[i], labels_second_batch[i]], dim=0)
        scores = torch.cat([scores_batch[i], scores_second_batch[i]], dim=0)
        mask_pred = torch.cat([mask_pred_batch[i], mask_pred_batch[i]], dim=0)
        mask_pred_unprocessed = torch.cat([mask_pred_batch_unprocessed[i], mask_pred_batch_unprocessed[i]], dim=0)
        mask_cls = torch.cat([mask_cls_batch[i], mask_cls_batch[i]], dim=0)
        quer_features = torch.cat([quer_features_batch[i], quer_features_batch[i]], dim=0)
        
        if outputs["pred_regressions"] is not None:
            pred_regressions = torch.cat([pred_regressions_batch[i], pred_regressions_batch[i]], dim=0)

        keep = labels.ne(number_of_classes) & (scores > object_mask_threshold)
        keep_list.append(keep)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_masks_unprocessed = mask_pred_unprocessed[keep]
        cur_logits = mask_cls[keep]
        cur_queries = quer_features[keep]

        if outputs["pred_regressions"] is not None:
            cur_regressions = pred_regressions[keep]

        # The preserved list is will be saved for the additional utilization of the outputs. 
        keep_of_keep = []
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        instance_seg = torch.zeros((number_of_classes - 1, h, w), dtype=torch.int32, device=cur_masks.device)
        current_segment_id = 1

        # These are specific to single batch. 
        point_dict = {}
        confidence = {}
        attributes = {}
        pred_masks = []
        pred_logits = []
        pred_queries = []
        pred_regressions = []
        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                if pred_class == 0 or pred_class > number_of_classes - 1:
                    keep_of_keep.append(False)
                    continue

                mask_area = (cur_mask_ids == k).sum().item()

                original_area = (cur_masks[k] >= threshold1).sum().item()
                mask = (cur_masks[k] >= threshold1)

                original_area2 = (cur_masks[k] >= threshold1).sum().item()
                mask2 = (cur_masks[k] >= threshold2)

                probabilities = cur_prob_masks[k]
                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if (mask_area / original_area < overlap_threshold) and (pred_class == 1 or pred_class == 2):
                        keep_of_keep.append(False)
                        continue

                    if (mask_area / original_area2 < overlap_threshold2) and (pred_class == 3 or pred_class == 4):
                        keep_of_keep.append(False)
                        continue
                    
                    # This part extracts the point sets from the given masks. Up/Down and Left/Right directions are treated separately. 
                    if (pred_class == 1 or pred_class == 2):
                        point_set_union, score = lane_extract_points_from_bev(mask, probabilities, pred_class, probability_threshold=threshold1)
                        probability_filtering_threshold = threshold1
                    else:
                        point_set_union, score = lane_extract_points_from_bev(mask2, probabilities, pred_class, probability_threshold=threshold2)
                        probability_filtering_threshold = threshold2

                    prob_of_selected_points = probabilities[point_set_union[1].round().long(), point_set_union[0].round().long()]

                    if torch.sum(prob_of_selected_points > probability_filtering_threshold).item() == 0:
                        keep_of_keep.append(False)
                        continue

                    # Extracted points are filtered according to the probability threshold.
                    point_set_union = point_set_union[:, prob_of_selected_points > probability_filtering_threshold]
                    
                    point_dict[current_segment_id] = point_set_union
                    confidence[current_segment_id] = cur_scores[k].item() + score
                    attributes[current_segment_id] = pred_class
                    instance_seg[pred_class - 1, mask] = current_segment_id
                    pred_masks.append(cur_masks_unprocessed[k][None, ])
                    pred_logits.append(cur_logits[k][None, ])
                    pred_queries.append(cur_queries[k][None, ])
                    
                    if outputs["pred_regressions"] is not None:
                        pred_regressions.append(cur_regressions[k][None, ])

                    current_segment_id += 1
                    keep_of_keep.append(True)
                else:
                    keep_of_keep.append(False)

        # Batch outputs are collected in this step. 
        point_dict_list.append(point_dict)
        confidence_list.append(confidence)
        attributes_list.append(attributes)

        pred_masks_batch_list.append(torch.cat(pred_masks, dim=0))
        pred_logits_batch_list.append(torch.cat(pred_logits, dim=0))
        pred_queries_batch_list.append(torch.cat(pred_queries, dim=0))
        if outputs["pred_regressions"] is not None:
            pred_regressions_batch_list.append(torch.cat(pred_regressions, dim=0))

        keep_of_keep_list.append(keep_of_keep)
        labels_list.append(cur_classes[keep_of_keep])
        for class_index in range(number_of_classes - 1):
            instance_seg_batch[i, 0, class_index] = instance_seg[class_index]

    output_post = {
        "instance": instance_seg_batch, "semantic": semantic_seg_batch,
        "point_dict": point_dict_list, "confidence": confidence_list, 
        "attribute": attributes_list, "keep": keep_list, "keep_of_keep": keep_of_keep_list, 
        "labels": labels_list, 
        "pred_masks": pred_masks_batch_list, "pred_logits": pred_logits_batch_list,
        "quer_feat": pred_queries_batch_list, "pred_regressions": pred_regressions_batch_list
    }

    return output_post


def lane_extract_points_from_bev(mask, probabilities, pred_class, probability_threshold):
    """
    This function extracts the point sets from the given mask.
    The function works with the concept of row-wise expectation and column-wise expectation. 

    args:
        mask: The mask of the lane.
        probabilities: The probabilities mask of the segmentation output for every instance. 
        pred_class: The class of the lane which denotes the direction information. 
        probability_threshold: The probability threshold for the point selection.
    """
    probabilities = probabilities * mask
    anchor_selection_x = mask.sum(1)
    anchor_selection_y = mask.sum(0)
    width = probabilities.shape[1]
    height = probabilities.shape[0]
    probabilities_sum_across_x = probabilities.sum(1)
    probabilities_sum_across_y = probabilities.sum(0)
    
    # Row-wise expectation. Expected to work for centerlines whose directions are up/down.
    x_position_anchors = torch.arange(width, device=probabilities.device)[None, :]
    x_positions_based_probabilities = x_position_anchors * probabilities
    estimated_x_positions_raw = x_positions_based_probabilities.sum(-1) / probabilities_sum_across_x
    estimated_x_positions = estimated_x_positions_raw[anchor_selection_x > 0]
    corresponding_y_positions_to_estimated_x_positions = torch.where(anchor_selection_x > 0)[0]
    point_set_1 = torch.cat([estimated_x_positions[None, :], corresponding_y_positions_to_estimated_x_positions[None, :]], 0)

    # Column-wise expectation. Expected to work for centerlines whose directions are left/right.
    y_position_anchors = torch.arange(height, device=probabilities.device)[:, None]
    y_positions_based_probabilities = y_position_anchors * probabilities
    estimated_y_positions_raw = y_positions_based_probabilities.sum(-2) / probabilities_sum_across_y
    estimated_y_positions = estimated_y_positions_raw[anchor_selection_y > 0]
    corresponding_x_positions_to_estimated_y_positions = torch.where(anchor_selection_y > 0)[0]
    point_set_2 = torch.cat([corresponding_x_positions_to_estimated_y_positions[None, :], estimated_y_positions[None, :]], 0)

    if pred_class == 1 or pred_class == 2:
        point_set_union = point_set_1
    elif pred_class == 3 or pred_class == 4:
        point_set_union = point_set_2

    # This line is commented but not removed because ablations are required. Instead of label, we can directly union the two point sets.. 
    # point_set_union = torch.cat([point_set_1, point_set_2], 1)

    point_set_union_index = point_set_union.long()
    final_prob = probabilities[point_set_union_index[1, :], point_set_union_index[0, :]]

    # Points are filtered if the expectation-based locations have lower probabilities than the pre-determined threshold.
    filter = final_prob > probability_threshold
    if filter.sum() > 0:
        point_set_union = point_set_union[:, filter]
        final_prob = final_prob[filter]

    score = torch.mean(final_prob).item()

    return point_set_union, score


def post_process_sigmoid(outputs, object_mask_threshold, overlap_threshold, number_of_classes, topk=200):
    """
    This function modifies the post_process_softmax function for sigmoid based training. 
    The dab code utilizes sigmoid for classification training. 
    It is observed that softmax is better choice but still there is detailed ablations. 
    Args:
        outputs (dict): the output of the model
        object_mask_threshold (float): the threshold for the object mask
        overlap_threshold (float): the threshold for the overlap
        number_of_classes (int): the number of classes
        topk (int): the number of topk
    """

    mask_cls_batch = outputs["pred_logits"].detach()
    mask_pred_batch = outputs["pred_masks"].detach()
    # The critical difference lies in that line with softmax version of it. 
    soft_batch_outs = torch.sigmoid(mask_cls_batch)
    scores_batch, labels_batch = soft_batch_outs.max(-1)
    score_segm_outs = soft_batch_outs
    mask_pred_batch_sigmoid = mask_pred_batch.sigmoid()

    topk_values, topk_indexes = torch.topk(soft_batch_outs.view(mask_cls_batch.shape[0], -1), topk, dim=1)
    scores_batch = topk_values
    topk_boxes = topk_indexes // mask_cls_batch.shape[2]
    labels_batch = topk_indexes % mask_cls_batch.shape[2]
    B, R, H, W = mask_pred_batch_sigmoid.shape
    mask_pred_batch = mask_pred_batch_sigmoid.view(B, R, H * W)
    mask_pred_batch = torch.gather(mask_pred_batch, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, H * W))
    mask_pred_batch = mask_pred_batch.view(B, topk, H, W)

    h, w = mask_pred_batch.shape[-2:]
    batch_size = mask_pred_batch.shape[0]
    instance_seg_batch = torch.zeros(batch_size, 1, 1, h, w, dtype=torch.int64, device=mask_pred_batch.device, requires_grad=False)

    output_prob = torch.einsum("bqc,bqhw->bchw", score_segm_outs, mask_pred_batch_sigmoid)
    semantic_seg_batch = torch.argmax(output_prob, dim=1, keepdim=True)[:, None]
    for i in range(batch_size):
        labels = labels_batch[i]
        scores = scores_batch[i]
        mask_pred = mask_pred_batch[i]

        keep = labels.ne(number_of_classes) & (scores > object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        # This line is added for class imbalance problem in sigmoid training. Still inferior to softmax version. 
        cur_scores[cur_classes == 0] *= 0.2
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        instance_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        current_segment_id = 1
        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                if pred_class == 0:
                    continue
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_masks[k] >= 0.5)
                # mask = (cur_mask_ids == k)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue

                    instance_seg[mask] = current_segment_id
                    current_segment_id += 1
        instance_seg_batch[i, 0, 0] = instance_seg
    output_post = {"inst": instance_seg_batch, "segm": semantic_seg_batch}
    return output_post