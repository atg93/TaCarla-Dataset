# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import os
import torch
import torch.nn.functional as F
from torch import nn
import math

from tairvision.ops.misc import (NestedTensor, nested_tensor_from_tensor_list,
                                 inverse_sigmoid, create_masks_from_feature_dict)
from tairvision.models.transormer_utils import MLP, PositionEmbeddingSine
import copy
from tairvision.models.transormer_utils import MHAttentionMap, MaskHeadSmallConv
import tairvision.models.detection.dabdetr_sub.segmentation as segm_heads

from tairvision.models.detection.dabdetr_sub.dn_components import prepare_for_dn, dn_post_process, compute_dn_loss
from tairvision.models.segmentation.mask2former_sub import ShapeSpec
from typing import Callable, Dict, List, Optional, Tuple, Union

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DABDeformableDETR(nn.Module):
    """ This is the DAB-Deformable-DETR for object detection """
    def __init__(self, input_shape: ShapeSpec, transformer, positional_embedding,
                 num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=True, two_stage=False,
                 use_dab=True, 
                 num_patterns=0,
                 random_refpoints_xy=False,
                 segmentation=False,
                 freeze_detr=False,
                 dn_training=False,
                 transformer_in_features: List[str] = ['1', '2', '3'],
                 extra_regression_target=False,
                 extra_regression_target_length=0,
                 **kwargs
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            use_dab: using dynamic anchor boxes formulation
            num_patterns: number of pattern embeddings
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """
        super().__init__()
        self.segmentation = segmentation
        self.freeze_detr = freeze_detr
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.extra_regression_target = extra_regression_target
        if extra_regression_target:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4 + extra_regression_target_length, 3)
        else:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_dab = use_dab
        self.num_patterns = num_patterns
        self.random_refpoints_xy = random_refpoints_xy
        self.num_classes = num_classes

        self.p2_key = kwargs.get('segmentation_head', {}).get('config', {}).get('mask_feature_key', '0')

        self.dn_training = dn_training
        # dn label enc
        if self.dn_training:
            self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator

        if not two_stage:
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
            else:
                if self.dn_training:
                    self.tgt_embed = nn.Embedding(num_queries, hidden_dim - 1)
                else:
                    self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

                if extra_regression_target:
                    self.refpoint_embed = nn.Embedding(num_queries, 4 + extra_regression_target_length)
                else:
                    self.refpoint_embed = nn.Embedding(num_queries, 4)

                if random_refpoints_xy:
                    self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
                    self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                    self.refpoint_embed.weight.data[:, :2].requires_grad = False

        # TODO, better handling of these two lines, with arguments or etc.

        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]

        strides = self.transformer_feature_strides
        num_channels = transformer_in_channels

        if self.num_patterns > 0:
            self.patterns_embed = nn.Embedding(self.num_patterns, hidden_dim)

        self.pe_layer = positional_embedding

        # TODO, Normally, this should be implemented in neck and be removed from here
        if num_feature_levels > 1:
            num_backbone_outs = len(strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value  # Why class embedding
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        if segmentation:
            if freeze_detr:
                print('Training with freezing detection branch of deformable detr.')
                for p in self.parameters():
                    p.requires_grad_(False)

            self.segm_head = segm_heads.__dict__[kwargs["segmentation_head"]["name"]](
                **kwargs["segmentation_head"]["config"])

    def forward(self, features_raw, masks_raw, dn_args=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        features = {key: features for key, features in features_raw.items() if key in self.transformer_in_features}
        feature_p2 = {key: features for key, features in features_raw.items() if key == str(self.p2_key)}

        masks = create_masks_from_feature_dict(features, masks_raw)
        mask_p2 = create_masks_from_feature_dict(feature_p2, masks_raw)[str(self.p2_key)]

        features = list(features.values())
        masks = list(masks.values())
        srcs = []
        pos = []
        for index, (feat, mask) in enumerate(zip(features, masks)):
            src = feat
            srcs.append(self.input_proj[index](src))
            pos.append(self.pe_layer(src[:, 0, ...], mask))

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1])
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks_raw
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.pe_layer(src[:, 0, ...], mask)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.two_stage:
            query_embeds = None
        elif self.use_dab:
            if self.num_patterns == 0:
                tgt_all_embed = tgt_embed = self.tgt_embed.weight           # nq, 256
                refanchor = self.refpoint_embed.weight      # nq, 4
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
            else:
                # multi patterns
                tgt_embed = self.tgt_embed.weight           # nq, 256
                pat_embed = self.patterns_embed.weight      # num_pat, 256
                tgt_embed = tgt_embed.repeat(self.num_patterns, 1) # nq*num_pat, 256
                pat_embed = pat_embed[:, None, :].repeat(1, self.num_queries, 1).flatten(0, 1) # nq*num_pat, 256
                tgt_all_embed = tgt_embed + pat_embed
                refanchor = self.refpoint_embed.weight.repeat(self.num_patterns, 1)      # nq*num_pat, 4
                query_embeds = torch.cat((tgt_all_embed, refanchor), dim=1)
        else:
            query_embeds = self.query_embed.weight

        attn_mask = None
        mask_dict = None
        if self.dn_training:
            # prepare for dn
            input_query_label, input_query_bbox, attn_mask, mask_dict = \
                prepare_for_dn(**dn_args, tgt_weight=tgt_all_embed, embedweight=refanchor,
                               batch_size=src.size(0), training=self.training, num_queries=self.num_queries,
                               num_classes=self.num_classes, hidden_dim=self.hidden_dim, label_enc=self.label_enc)
            query_embeds = torch.cat((input_query_label, input_query_bbox), dim=2)
            query_embeds = query_embeds[0]

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, memory, level_start_index = \
            self.transformer(
                srcs, masks, pos, query_embeds, return_memory=True, attn_mask=attn_mask
            )


        outputs_classes = []
        outputs_coords = []
        outputs_regression = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
                outputs_coord = tmp.sigmoid()
            elif reference.shape[-1] > 4:
                tmp += reference
                full_pred = tmp.sigmoid()
                outputs_coord = full_pred[..., :4]
                pred_regression = full_pred[..., 4:]
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            if self.extra_regression_target:
                outputs_regression.append(pred_regression)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        if self.extra_regression_target:
            outputs_regression = torch.stack(outputs_regression)

        outputs_class, outputs_coord = dn_post_process(outputs_class, outputs_coord, mask_dict)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.extra_regression_target:
            out.update({'pred_regressions': outputs_regression[-1]})
            
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, 
                outputs_coord, 
                outputs_regression if self.extra_regression_target else None
            )

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        out['quer_feat'] = hs[-1]
        if self.segmentation:
            outputs_seg_masks_list = self.segm_head(srcs, masks, memory, level_start_index, hs, features_raw, mask_p2)
            out["pred_masks"] = outputs_seg_masks_list[-1]
            if len(outputs_seg_masks_list) > 1:
                for i, pred_masks in enumerate(outputs_seg_masks_list[:-1]):
                    out["aux_outputs"][i].update({"pred_masks": pred_masks})

        return out, mask_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_regression):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.extra_regression_target:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_regressions': c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_regression[:-1])]
        else:
            return [{'pred_logits': a, 'pred_boxes': b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]



