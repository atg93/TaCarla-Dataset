import torch.nn as nn
import torch
import numpy as np
from tairvision.models.transormer_utils import gen_sineembed_for_position, MLP
import torch.nn.functional as F
from tairvision.ops import sigmoid_focal_loss
import math


class OptimalTransportCenterlines(nn.Module):
    def __init__(self, cfg, matcher, sinkhorn_threshold = 0.07, sigmoid_threshold = 0.05):
        super(OptimalTransportCenterlines, self).__init__()
        """
        This class is used for the optimal transport head for the centerlines.
        It creates relationship (connectivity) among the centerlines. 
        
        The centerline relationship can be represented as a directed graph. 
        The concept is the part of road topology.

        Args:
            cfg (CfgNode): config.
            matcher (Matcher): matcher which is needed for matching with the GT.
            sinkhorn_threshold (float): threshold for the sinkhorn algorithm.
            sigmoid_threshold (float): threshold for the sigmoid function (sigmoid focal loss).
        """
        self.sinkhorn_activated = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.SINKHORN_ACTIVATED
        self.matcher = matcher
        self.sinkhorn_threshold = sinkhorn_threshold
        self.sigmoid_threshold = sigmoid_threshold

        self.normalized_positions = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.NORMALIZED_POSITIONS

        self.methodology = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.METHODOLOGY
        # The directed graph might require the ablation of shared projection.
        self.shared_projection = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.SHARED_PROJECTION
        num_layers = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.NUMBER_OF_MLP_LAYERS

        # Filtering concept eliminates some of the queries of centerlines.
        if not cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_FILTERING:
            self.number_of_selected_queries = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MASKED_TRANSFORMER_DECODER_CONFIG.NUM_QUERIES
        else:
            self.number_of_selected_queries = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.NUMBER_OF_SELECTED_QUERIES_LC

        self.loss_coefficient = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.LOSS_COEFFICIENT_LCLC
        d_model = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MASKED_TRANSFORMER_DECODER_CONFIG.HIDDEN_DIM
        self.d_model = d_model
        self.sinusoidal_dimension = self.d_model // 2
        
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, num_layers)
        if not self.shared_projection:
            self.ref_point_head2 = MLP(2 * d_model, d_model, d_model, num_layers)
        
        # The queries of the centerlines can be concatenated with each other. 
        if self.methodology == "concatenation_based":
            self.classification_head = MLP(2 * d_model, d_model, 1, num_layers)
            prior_probability = 0.01
            torch.nn.init.normal_(self.classification_head.layers[-1].weight, std=0.01)
            torch.nn.init.constant_(self.classification_head.layers[-1].bias, -math.log((1 - prior_probability) / prior_probability))

        if self.sinkhorn_activated:
            self.bin_score = torch.nn.Parameter(torch.tensor(1.))
            self.sinkhorn_iterations = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.NUMBER_OF_SINKHORN_ITERATIONS

    def forward(self, outputs):
        outputs_filtered_lc = outputs["outputs_filtered_lc"]
        positions = self.get_positions(outputs_filtered_lc)
        positional_embeddigs = gen_sineembed_for_position(positions, self.sinusoidal_dimension)
        positional_embeddigs_start = positional_embeddigs[:, :, :self.d_model]
        positional_embeddigs_end = positional_embeddigs[:, :, self.d_model:]
        
        # If the relationship improver is utilized, the improved queries should be utilized. 
        # Otherwise, the original queries should be utilized.
        if "quer_feat_relations" in outputs_filtered_lc:
            quer_feat_lc = outputs_filtered_lc["quer_feat_relations"]
        else:
            quer_feat_lc = outputs_filtered_lc["quer_feat"]

        # For auxiliary loss, the relationship head is applied onto the outputs of other layers
        if isinstance(quer_feat_lc, list):
            score_list = []
            for quer_feat_lc_single in quer_feat_lc:
                scores = self.forward_single(
                    quer_feat_lc_single, 
                    positional_embeddigs_start, 
                    positional_embeddigs_end
                )
                score_list.append(scores)
            scores = score_list
        else:
            scores = self.forward_single(
                quer_feat_lc, 
                positional_embeddigs_start, 
                positional_embeddigs_end
            )
        return scores
    
    def forward_single(self, quer_feat_lc, positional_embeddigs_start, positional_embeddigs_end):
        
        # Positional embeddings are concatenated with the queries of the centerlines.
        features1 = torch.concat([quer_feat_lc, positional_embeddigs_end], dim=2)
        features2 = torch.concat([quer_feat_lc, positional_embeddigs_start], dim=2)

        features1 = self.ref_point_head(features1)
        # Shared projection enables the same head to be applied onto the queries of the centerlines.
        if self.shared_projection:
            features2 = self.ref_point_head(features2)
        else:
            features2 = self.ref_point_head2(features2)

        # similarity-based methodology is the dot product between the queries of the centerlines.
        if self.methodology == "similarity_based":
            scores = torch.einsum('bnc,bmc->bnm', features1, features2)
            scores = scores / self.d_model ** .5

        # The other methodology is concatenation-based in which the queries of the centerlines are concatenated with each other.
        else:
            number_of_selected_queries = features1.shape[1]
            adj = torch.cat([
                features1.unsqueeze(2).repeat(1, 1, number_of_selected_queries, 1),
                features2.unsqueeze(1).repeat(1, number_of_selected_queries, 1, 1),
            ], dim=-1)
            scores = self.classification_head(adj).squeeze(-1)
            scores = scores.sigmoid()

        if self.sinkhorn_activated:
            diag_indices = torch.arange(features1.shape[1], device=scores.device)
            scores[:, diag_indices, diag_indices] = float('-inf')
            scores = log_optimal_transport(scores, self.bin_score, self.sinkhorn_iterations)
            scores = scores.exp()

        return scores
    
    # filter query option is used for eliminating some of the queries of the centerlines.
    def filter_queries(self, outputs):
        outputs_filtered = filter_queries(
            outputs=outputs, 
            number_of_selected_queries=self.number_of_selected_queries, 
            topk_start_index=1,
            function = "softmax"
        )
        return outputs_filtered

    def get_loss(self, head_outputs, targets):

        target_for_loss, targets_list, src_location_tuples_list = self.create_lclc_target_for_loss(targets, head_outputs)
        lclc_scores = head_outputs["lclc_scores"]

        if isinstance(lclc_scores, list):
            loss_list = []
            for lclc_score in lclc_scores:
                loss, metrics = self.get_loss_single(lclc_score, target_for_loss, targets_list, src_location_tuples_list)
                loss_list.append(loss)
            loss = loss_list
        else:
            loss, metrics = self.get_loss_single(lclc_scores, target_for_loss, targets_list, src_location_tuples_list)
        
        return loss, metrics
    
    def create_lclc_target_for_loss(self, targets, head_outputs):
        """
        This function creates the target for the relationship head. 
        It utilizes matcher for creating the target.
        """
        new_target_list = targets["dab_ordered"]
        # TODO, matcher might also need some thresholding. Try it in the future for research.
        matching_result = self.matcher(head_outputs["outputs_filtered_lc"], new_target_list)
        lclc_matrices = targets["lclc_list"]
        new_lclc_matrix_list = []
        targets_list = []
        src_location_tuples_list = []
        for batch_size, lclc_matrix in enumerate(lclc_matrices):
            lclc_matrix = lclc_matrix[0]
            target_ids = matching_result[batch_size][1]
            src_ids = matching_result[batch_size][0]
            target_location_tuples = torch.meshgrid(target_ids, target_ids)
            src_location_tuples = torch.meshgrid(src_ids, src_ids)

            new_lclc_matrix = torch.zeros(
                self.number_of_selected_queries + 1, 
                self.number_of_selected_queries + 1, 
                dtype=lclc_matrix.dtype, 
                device=lclc_matrix.device
            )

            new_lclc_matrix[src_location_tuples] = lclc_matrix[target_location_tuples]
            new_lclc_matrix_list.append(new_lclc_matrix[None, :])
            targets_list.append(lclc_matrix[target_location_tuples][None, :])
            src_location_tuples_list.append(src_location_tuples)
            
        new_lclc_matrix_batch = torch.concat(new_lclc_matrix_list, dim=0)
        return new_lclc_matrix_batch, targets_list, src_location_tuples_list
    
    def get_loss_single(self, lclc_scores, target_for_loss, targets_list, src_location_tuples_list):
        score_list = []
        for batch_size, src_location_tuples in enumerate(src_location_tuples_list):            
            scores_batch = lclc_scores[batch_size]
            if self.sinkhorn_activated:
                scores_batch = (scores_batch > self.sinkhorn_threshold) * 1
                lclc_scores_for_metric = (lclc_scores > self.sinkhorn_threshold) * 1
            else:
                scores_batch = (scores_batch > self.sigmoid_threshold) * 1
                lclc_scores_for_metric = (lclc_scores > self.sigmoid_threshold) * 1
            scores_batch_for_only_target = scores_batch[src_location_tuples]
            score_list.append(scores_batch_for_only_target[None, :])

        metrics = {"predictions": score_list, "targets": targets_list}
        metrics.update({"targets_all_queries": target_for_loss[:, :-1, :-1]})
        if self.sinkhorn_activated:
            target_for_loss[:, -1, :-1] = (target_for_loss[:, :-1, :-1] == 0).all(dim=1)
            target_for_loss[:, :-1, -1] = (target_for_loss[:, :-1, :-1] == 0).all(dim=2)
            target_for_loss[:, -1, -1] = 1
            loss = -torch.log(lclc_scores[target_for_loss == 1]).mean()
            metrics.update({"predictions_all_queries": lclc_scores_for_metric[:, :-1, :-1]})
        else:
            loss = sigmoid_focal_loss(lclc_scores, target_for_loss[:, :-1, :-1].float(), alpha=0.25, gamma=2.0, reduction="sum", sigmoid_input=True)
            loss = loss / target_for_loss[:, :-1, :-1].sum().clip(min=1)
            metrics.update({"predictions_all_queries": lclc_scores_for_metric})
        loss = loss * self.loss_coefficient
        return loss, metrics

    def get_positions(self, outputs_filtered):
        """
        This funtion aims to get the positions of the centerlines from the mask predictions.
        The positions are the start and end points of the centerlines.
        The predicted labels are also utilized for determining the end and start points of the centerlines.
        """        
        mask_pred_batch_prob = outputs_filtered["pred_masks"].sigmoid()
        mask_pred_batch_binary = (mask_pred_batch_prob > 0.5)
        B, Q, H, W = mask_pred_batch_binary.shape
        start_end_points = torch.zeros([B, Q, 4], device=mask_pred_batch_binary.device, dtype=torch.float32)

        for b in range(B):
            for q in range(Q):
                if "label" in outputs_filtered:
                    label = outputs_filtered["labels"][b, q].item()
                else:
                    label = torch.argmax(outputs_filtered["pred_logits"][b, q]) - 1
                non_binary_indices = torch.where(mask_pred_batch_binary[b][q])

                if non_binary_indices[0].nelement() == 0:
                    continue

                topmost_index = non_binary_indices[0].argmin()
                bottommost_index = non_binary_indices[0].argmax()
                leftmost_index = non_binary_indices[1].argmin()
                rightmost_index = non_binary_indices[1].argmax()

                if label == 0:
                    positions = torch.hstack([
                        non_binary_indices[0][bottommost_index],
                        non_binary_indices[1][bottommost_index], 
                        non_binary_indices[0][topmost_index],
                        non_binary_indices[1][topmost_index]
                    ])

                elif label == 1:
                    positions = torch.hstack([
                        non_binary_indices[0][topmost_index],
                        non_binary_indices[1][topmost_index], 
                        non_binary_indices[0][bottommost_index],
                        non_binary_indices[1][bottommost_index]
                    ])

                elif label == 2:
                    positions = torch.hstack([
                        non_binary_indices[0][rightmost_index],
                        non_binary_indices[1][rightmost_index], 
                        non_binary_indices[0][leftmost_index],
                        non_binary_indices[1][leftmost_index]
                    ])

                elif label == 3:
                    positions = torch.hstack([
                        non_binary_indices[0][leftmost_index],
                        non_binary_indices[1][leftmost_index], 
                        non_binary_indices[0][rightmost_index],
                        non_binary_indices[1][rightmost_index]
                    ])

                else:
                    positions = torch.tensor([0, 0, 0, 0]).to(mask_pred_batch_binary.device)

                start_end_points[b, q, ] = positions
        if self.normalized_positions:
            height, width = mask_pred_batch_binary[b][q].shape
            normalization_tensor = torch.tensor([height, width, height, width], device=mask_pred_batch_binary.device, dtype=torch.float32)
            start_end_points = start_end_points / normalization_tensor
        return start_end_points


class OptimalTransportTrafficElements(nn.Module):
    def __init__(self, cfg, matcher_lc, matcher_te, sinkhorn_threshold=0.01, sigmoid_threshold=0.05):
        super(OptimalTransportTrafficElements, self).__init__()
        """
        This class is used for the relationship head between the traffic elements and centerlines.
        It creates relationship (connectivity) among the traffic elements and centerlines.

        The traffic element-centerline relationship can be represented as an undirected graph.
        The concept is the part of road topology.

        Args:
            cfg (CfgNode): config.
            matcher_lc (Matcher): matcher which is needed for matching with the GT for the centerlines.
            matcher_te (Matcher): matcher which is needed for matching with the GT for the traffic elements.
            sinkhorn_threshold (float): threshold for the sinkhorn algorithm.
            sigmoid_threshold (float): threshold for the sigmoid function (sigmoid focal loss).

        """
        self.sinkhorn_activated = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.SINKHORN_ACTIVATED_LCTE
        self.cfg = cfg
        self.matcher_lc = matcher_lc
        self.matcher_te = matcher_te
        self.sinkhorn_threshold = sinkhorn_threshold
        self.sigmoid_threshold = sigmoid_threshold

        if not cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_FILTERING:
            self.number_of_selected_te_queries = cfg.MODEL.HEAD2D.DAB_PARAMS.NUM_QUERIES
        else:
            self.number_of_selected_te_queries = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.NUMBER_OF_SELECTED_QUERIES_TE

        if not cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_FILTERING:
            self.number_of_selected_queries = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MASKED_TRANSFORMER_DECODER_CONFIG.NUM_QUERIES
        else:
            self.number_of_selected_queries = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.NUMBER_OF_SELECTED_QUERIES_LC

        self.loss_coefficient = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.LOSS_COEFFICIENT_LCTE
        d_model = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MASKED_TRANSFORMER_DECODER_CONFIG.HIDDEN_DIM
        self.d_model = d_model
        self.d_model_te = cfg.MODEL.HEAD2D.TRANSFORMER.D_MODEL
        self.lc_head = MLP(d_model, d_model, d_model, 2)
        self.te_head = MLP(self.d_model_te, d_model, d_model, 2)

        self.methodology = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.METHODOLOGY_LCTE
        num_layers = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.NUMBER_OF_MLP_LAYERS

        if self.sinkhorn_activated:
            self.bin_score = torch.nn.Parameter(torch.tensor(1.))
            self.sinkhorn_iterations = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.NUMBER_OF_SINKHORN_ITERATIONS

        if self.methodology == "concatenation_based":
            self.classification_head = MLP(2 * d_model, d_model, 1, num_layers)
            prior_probability = 0.01
            torch.nn.init.normal_(self.classification_head.layers[-1].weight, std=0.01)
            torch.nn.init.constant_(self.classification_head.layers[-1].bias, -math.log((1 - prior_probability) / prior_probability))


    def forward(self, outputs):
        outputs_filtered_lc = outputs["outputs_filtered_lc"]
        outputs_filtered_te = outputs["outputs_filtered_te"]

        if "quer_feat_relations" in outputs_filtered_lc:
            quer_feat_lc = outputs_filtered_lc["quer_feat_relations"]
        else:
            quer_feat_lc = outputs_filtered_lc["quer_feat"]

        if "quer_feat_relations" in outputs_filtered_te:
            quer_feat_te = outputs_filtered_te["quer_feat_relations"]
        else:
            quer_feat_te = outputs_filtered_te["quer_feat"]

        if isinstance(quer_feat_lc, list):
            scores_list = []
            for quer_feat_lc_single, quer_feat_te_single in zip(quer_feat_lc, quer_feat_te):
                scores_single = self.forward_single(quer_feat_lc_single, quer_feat_te_single)
                scores_list.append(scores_single)
            scores = scores_list
        else:
            scores = self.forward_single(quer_feat_lc, quer_feat_te)

        return scores
    
    def forward_single(self, quer_feat_lc, quer_feat_te):
        features1 = self.lc_head(quer_feat_lc)
        features2 = self.te_head(quer_feat_te)
        
        if self.methodology == "similarity_based":
            scores = torch.einsum('bnc,bmc->bnm', features1, features2)
            scores = scores / self.d_model ** .5

        else:
            number_of_selected_queries_lc = features1.shape[1]
            number_of_selected_queries_te = features2.shape[1]
            adj = torch.cat([
                features1.unsqueeze(2).repeat(1, 1, number_of_selected_queries_te, 1),
                features2.unsqueeze(1).repeat(1, number_of_selected_queries_lc, 1, 1),
            ], dim=-1)
            scores = self.classification_head(adj).squeeze(-1)
            scores = scores.sigmoid()


        if self.sinkhorn_activated:
            scores = log_optimal_transport(scores, self.bin_score, self.sinkhorn_iterations)
            scores = scores.exp()

        return scores

    def filter_queries(self, outputs):
        outputs_filtered = filter_queries(outputs=outputs,
            number_of_selected_queries=self.number_of_selected_te_queries,
            topk_start_index=0,
            function = "sigmoid"
        )
            
        return outputs_filtered

    def get_loss(self, head_outputs, targets):
        target_for_loss, targets_list, src_location_tuples_list = self.create_lcte_target_for_loss(targets, head_outputs)
        lcte_scores = head_outputs["lcte_scores"]

        if isinstance(lcte_scores, list):
            loss_list = []
            for lcte_score in lcte_scores:
                loss, metrics = self.get_loss_single(lcte_score, target_for_loss, targets_list, src_location_tuples_list)
                loss_list.append(loss)
            loss = loss_list
        else:
            loss, metrics = self.get_loss_single(lcte_scores, target_for_loss, targets_list, src_location_tuples_list)

        return loss, metrics
    
    def get_loss_single(self, lcte_scores, target_for_loss, targets_list, src_location_tuples_list):
        score_list = []
        for batch_size, src_location_tuples in enumerate(src_location_tuples_list):            
            scores_batch = lcte_scores[batch_size]
            if self.sinkhorn_activated:
                scores_batch = scores_batch > self.sinkhorn_threshold
                lcte_scores_for_metric = lcte_scores > self.sinkhorn_threshold
            else:
                scores_batch = scores_batch > self.sigmoid_threshold
                lcte_scores_for_metric = lcte_scores > self.sigmoid_threshold
            scores_batch_for_only_target = scores_batch[src_location_tuples]
            score_list.append(scores_batch_for_only_target[None, :])

        metrics = {"predictions": score_list, "targets": targets_list}
        metrics.update({"targets_all_queries": target_for_loss[:, :-1, :-1]})
        
        if self.sinkhorn_activated:
            target_for_loss[:, -1, :-1] = (target_for_loss[:, :-1, :-1] == 0).all(dim=1)
            target_for_loss[:, :-1, -1] = (target_for_loss[:, :-1, :-1] == 0).all(dim=2)
            target_for_loss[:, -1, -1] = 1
            loss = -torch.log(lcte_scores[target_for_loss == 1]).mean()
            metrics.update({"predictions_all_queries": lcte_scores_for_metric[:, :-1, :-1]})
        else:
            loss = sigmoid_focal_loss(lcte_scores, target_for_loss[:, :-1, :-1].float(), alpha=0.25, gamma=2.0,
                                      reduction="sum", sigmoid_input=True)
            loss = loss / target_for_loss[:, :-1, :-1].sum().clip(min=1)
            metrics.update({"predictions_all_queries": lcte_scores_for_metric})
        loss = loss * self.loss_coefficient

        return loss, metrics
    
    def create_lcte_target_for_loss(self, targets, head_outputs):
        new_target_list = targets["dab_ordered"]

        matching_result_te = self.matcher_te(head_outputs["outputs_filtered_te"], targets["targets2d_loss"])
        matching_result_lc = self.matcher_lc(head_outputs["outputs_filtered_lc"], new_target_list)

        lcte_matrices = targets["lcte_list"]

        new_lcte_matrix_list = []
        targets_list = []
        src_location_tuples_list = []
        for batch_size, lcte_matrix in enumerate(lcte_matrices):
            lcte_matrix = lcte_matrix[0]

            target_ids_lc = matching_result_lc[batch_size][1]
            src_ids_lc = matching_result_lc[batch_size][0]

            target_ids_te = matching_result_te[batch_size][1]
            src_ids_te = matching_result_te[batch_size][0]

            target_location_tuples = torch.meshgrid(target_ids_lc, target_ids_te)
            src_location_tuples = torch.meshgrid(src_ids_lc, src_ids_te)

            new_lcte_matrix = torch.zeros(
                self.number_of_selected_queries + 1,
                self.number_of_selected_te_queries + 1,
                dtype=lcte_matrix.dtype, device=lcte_matrix.device
            )
            new_lcte_matrix[src_location_tuples] = lcte_matrix[target_location_tuples]
            new_lcte_matrix_list.append(new_lcte_matrix[None, :])
            targets_list.append(lcte_matrix[target_location_tuples][None, :])
            src_location_tuples_list.append(src_location_tuples)

        new_lcte_matrix_batch = torch.concat(new_lcte_matrix_list, dim=0)
        return new_lcte_matrix_batch, targets_list, src_location_tuples_list



def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def filter_queries(outputs, number_of_selected_queries, topk_start_index=1, function = "softmax"):
    mask_cls_batch = outputs["pred_logits"]
    mask_query_feat_batch = outputs["quer_feat"]
    channel_dim = mask_query_feat_batch.shape[2]
    B = mask_query_feat_batch.shape[0]

    if function == "softmax":
        soft_batch_outs = F.softmax(mask_cls_batch, dim=-1)
        probabilities = soft_batch_outs[..., topk_start_index:-1].contiguous()
    elif function == "sigmoid":
        probabilities = mask_cls_batch.sigmoid()[..., topk_start_index:].contiguous()

    number_of_classes = probabilities.shape[2]
    logits_dim = mask_cls_batch.shape[2]
    
    topk_values, topk_indexes = torch.topk(probabilities.view(B, -1), number_of_selected_queries, dim=1)
    topk_queries = topk_indexes // number_of_classes
    labels = topk_indexes % number_of_classes

    mask_cls_batch_filtered = torch.gather(mask_cls_batch, 1, topk_queries.unsqueeze(-1).repeat(1, 1, logits_dim))
    mask_query_feat_batch_filtered = torch.gather(mask_query_feat_batch, 1, topk_queries.unsqueeze(-1).repeat(1, 1, channel_dim))

    outputs_filtered = {
        "pred_logits": mask_cls_batch_filtered,
        "quer_feat": mask_query_feat_batch_filtered, 
        "labels": labels,
    }

    if "pred_masks" in outputs:
        mask_pred_batch = outputs["pred_masks"]
        B, R, H, W = mask_pred_batch.shape
        mask_pred_batch = mask_pred_batch.view(B, R, H * W)
        mask_pred_batch_filtered = torch.gather(mask_pred_batch, 1, topk_queries.unsqueeze(-1).repeat(1, 1, H * W))
        mask_pred_batch_filtered = mask_pred_batch_filtered.view(B, number_of_selected_queries, H, W)
        outputs_filtered["pred_masks"] = mask_pred_batch_filtered

    if "pred_boxes" in outputs:
        mask_pred_boxes_batch = outputs["pred_boxes"]
        mask_pred_boxes_batch_filtered = torch.gather(mask_pred_boxes_batch, 1, topk_queries.unsqueeze(-1).repeat(1, 1, 4))
        outputs_filtered["pred_boxes"] = mask_pred_boxes_batch_filtered

    if "pred_regressions" in outputs:
        if outputs["pred_regressions"] is not None:
            regressions_pred_regressions_batch = outputs["pred_regressions"]
            code_size = regressions_pred_regressions_batch.shape[-1]
            regressions_pred_regressions_batch_filtered = torch.gather(regressions_pred_regressions_batch, 1, topk_queries.unsqueeze(-1).repeat(1, 1, code_size))
            outputs_filtered["pred_regressions"] = regressions_pred_regressions_batch_filtered
        else:
            outputs_filtered["pred_regressions"] = None
            
    return outputs_filtered


