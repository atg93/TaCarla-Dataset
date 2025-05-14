from torch import nn
from tairvision.models.transormer_utils import MHAttentionMap, MaskHeadSmallConv
import torch
from tairvision.models.transormer_utils import MLP
from tairvision.ops.misc import c2_xavier_fill
import torch.nn.functional as F
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDetrStandardSegmHead(nn.Module):
    def __init__(self, d_model, nhead, num_queries, use_p2=False):
        super().__init__()
        hidden_dim, nheads = d_model, nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0)
        # Original was [1024, 512, 256], I have increased to the four levels
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256, 128], hidden_dim, use_p2=use_p2)
        input_proj_list = []
        for in_channels in [1024, 512, 256, 128]:
            # The original code was using stride=2, but I have changed it to stride=1
            input_proj_list.append(nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, stride=1, padding=1))
        self.seg_input_proj = nn.ModuleList(input_proj_list)
        self.num_queries = num_queries

    def forward(self, srcs, masks, memory, level_start_index, hs, features_raw, mask_p2):
        bs, c, h, w = srcs[-1].shape
        flattened_masks_list = []
        for mask in masks:
            mask_flattened = mask.flatten(1)
            flattened_masks_list.append(mask_flattened)

        flattened_masks = torch.cat(flattened_masks_list, 1)
        # FIXME h_boxes takes the last one computed, keep this in mind
        seg_memory, seg_mask = memory[:, level_start_index[-1]:, :], flattened_masks[:, level_start_index[-1]:]
        seg_memory = seg_memory.permute(0, 2, 1).view(bs, c, h, w)
        seg_mask = seg_mask.view(bs, h, w)
        bbox_mask = self.bbox_attention(hs[-1], seg_memory, mask=seg_mask)
        
        # The original code does not use '0' feature. 
        seg_masks = self.mask_head(srcs[-1], bbox_mask,
                                   [self.seg_input_proj[0](features_raw['3']),
                                    self.seg_input_proj[1](features_raw['2']),
                                    self.seg_input_proj[2](features_raw['1']),
                                    self.seg_input_proj[3](features_raw['0']),
                                    ])

        outputs_seg_masks = seg_masks.view(bs, self.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        outputs_seg_masks_list = [outputs_seg_masks]

        return outputs_seg_masks_list


class MaskFormerLikeHead(nn.Module):
    def __init__(self, hidden_dim: int,
                 conv_dim: int,
                 mask_dim: int,
                 in_channels,
                 aux_loss: bool,
                 multi_scale: bool = False,
                 num_scale: int = 4,
                 use_mask: bool = False,
                 different_weight: bool = False,
                 mum_pred: int = 6,
                 mask_feature_key = '0',
                 **kwargs):
        super().__init__()
        self.mask_feature_key = mask_feature_key
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        if different_weight:
            self.mask_embed = _get_clones(self.mask_embed, mum_pred)

        if different_weight:
            assert aux_loss == different_weight, "different weight cannot be applied wo aux loss"
        self.aux_loss = aux_loss
        use_bias = False
        self.multi_scale = multi_scale
        self.hidden_dim = hidden_dim
        self.mask_dim = mask_dim
        self.use_mask = use_mask
        self.different_weight = different_weight
        self.mum_pred = mum_pred

        lateral_norm = nn.GroupNorm(32, conv_dim)
        output_norm = nn.GroupNorm(32, conv_dim)

        in_channels = in_channels
        self.lateral_conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_dim, kernel_size=1, bias=use_bias),
            lateral_norm
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            output_norm,
            nn.ReLU(inplace=True)
        )

        if self.multi_scale:
            projection_norm = nn.GroupNorm(32, conv_dim)
            self.projection_conv = nn.Sequential(
                nn.Conv2d(conv_dim * num_scale, conv_dim, kernel_size=1, bias=use_bias),
                projection_norm
            )
            c2_xavier_fill(self.projection_conv[0])

        c2_xavier_fill(self.lateral_conv[0])
        c2_xavier_fill(self.output_conv[0])

    def forward(self, srcs, masks, memory, level_start_index, hs, features_raw, mask_p2):
        mask_features = self.create_mask_features(srcs, memory, level_start_index, features_raw)

        outputs_seg_masks_list = []
        mask = None
        if self.use_mask:
            mask = mask_p2

        if not self.aux_loss:
            outputs_seg_masks = self.forward_prediction_heads(hs[-1], mask_features, mask=mask)
            outputs_seg_masks_list.append(outputs_seg_masks)
        else:
            for index, query in enumerate(hs):
                outputs_seg_masks = self.forward_prediction_heads(query, mask_features, mask=mask, index=index)
                outputs_seg_masks_list.append(outputs_seg_masks)

        return outputs_seg_masks_list

    def forward_prediction_heads(self, output, mask_features, mask = None, index = None):
        #TODO, is there a way to insert mask utilization
        decoder_output = self.decoder_norm(output)
        if self.different_weight and index is not None:
            mask_embed = self.mask_embed[index](decoder_output)
        else:
            mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        if mask is not None:
            outputs_mask.masked_fill_(mask.unsqueeze(1), float("-inf"))

        return outputs_mask

    def create_mask_features(self, srcs, memory, level_start_index, features_raw):
        if not self.multi_scale:
            bs, c, h, w = srcs[0].shape
            seg_memory_stride8 = memory[:, level_start_index[0]:level_start_index[1], :]
            seg_memory_stride8 = seg_memory_stride8.permute(0, 2, 1).view(bs, c, h, w)
        else:
            seg_memory_levels = []
            for level, src in enumerate(srcs):
                bs, c, h, w = src.shape
                if level + 1 == len(srcs):
                    seg_memory_level = memory[:, level_start_index[level]:, :]
                else:
                    seg_memory_level = memory[:, level_start_index[level]:level_start_index[level + 1], :]
                # interesting ddp error if we do not add contiguous here??
                seg_memory_level = seg_memory_level.permute(0, 2, 1).contiguous().view(bs, c, h, w)
                seg_memory_level = F.interpolate(seg_memory_level, size=srcs[0].shape[-2:],
                                                 mode="bilinear", align_corners=False)
                seg_memory_levels.append(seg_memory_level)
            seg_memory_levels_cat = torch.cat(seg_memory_levels, 1)
            seg_memory_stride8 = self.projection_conv(seg_memory_levels_cat)

        x = features_raw[str(self.mask_feature_key)].float()
        cur_fpn = self.lateral_conv(x)
        # Following FPN implementation, we use nearest upsampling here
        y = cur_fpn + F.interpolate(seg_memory_stride8, size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
        mask_features = self.output_conv(y)

        return mask_features


class MaskFormerLikeHeadIterative(MaskFormerLikeHead):
    def __init__(self, add_to_query_content: bool = False,
                 is_detach: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not add_to_query_content:
            self.mask_embed_iterative = MLP(self.hidden_dim, self.hidden_dim, self.mask_dim, 3)
        else:
            self.mask_embed_iterative = MLP(self.hidden_dim * 2, self.hidden_dim, self.mask_dim, 3)
            self.downsample = nn.AvgPool2d(4)
            self.mask_query_norm = nn.LayerNorm(self.hidden_dim)

        if self.different_weight:
            self.mask_embed = MLP(self.hidden_dim, self.hidden_dim, self.mask_dim, 3)
            self.mask_embed_iterative = _get_clones(self.mask_embed_iterative, self.mum_pred - 1)

        self.add_to_query_content = add_to_query_content
        self.is_detach = is_detach

    def forward(self, srcs, masks, memory, level_start_index, hs, features_raw, mask_p2):
        mask_features = self.create_mask_features(srcs, memory, level_start_index, features_raw)

        outputs_seg_masks_list = []

        mask = None
        if self.use_mask:
            mask = mask_p2

        outputs_seg_mask = self.forward_prediction_heads(hs[0], mask_features, mask=mask)
        outputs_seg_masks_list.append(outputs_seg_mask)

        for index, query in enumerate(hs[1:]):
            mask_query = None
            if self.add_to_query_content:
                outputs_seg_mask_sigm = self.downsample(outputs_seg_mask)
                outputs_seg_mask_sigm = outputs_seg_mask_sigm.sigmoid()
                outputs_seg_mask_sigm = outputs_seg_mask_sigm.flatten(2,3)[:, :, None]
                outputs_seg_mask_sigm = outputs_seg_mask_sigm.detach()
                mask_features_modified = self.downsample(mask_features).flatten(2,3)[:, None]
                weighted_feature = mask_features_modified * outputs_seg_mask_sigm
                mask_query = weighted_feature.sum(-1) / outputs_seg_mask_sigm.sum(-1)
            outputs_seg_masks_diff = self.forward_prediction_heads_iterative(
                query, mask_features, mask_query, mask=mask, index = index)
            new_outputs_seg_mask = outputs_seg_mask + outputs_seg_masks_diff
            if self.is_detach:
                outputs_seg_mask = new_outputs_seg_mask.detach()
            else:
                outputs_seg_mask = new_outputs_seg_mask
            outputs_seg_masks_list.append(outputs_seg_mask)

        return outputs_seg_masks_list

    def forward_prediction_heads_iterative(self, output, mask_features, mask_query = None, mask = None, index = None):
        decoder_output = self.decoder_norm(output)
        if mask_query is not None:
            mask_query_output = self.mask_query_norm(mask_query)
            decoder_output = torch.cat([decoder_output, mask_query_output], 2)
        if self.different_weight and index is not None:
            mask_embed = self.mask_embed_iterative[index](decoder_output)
        else:
            mask_embed = self.mask_embed_iterative(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        if mask is not None:
            outputs_mask.masked_fill_(mask.unsqueeze(1), float("-inf"))

        return outputs_mask


class PanopticSegFormerLikeHead(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 aux_loss: bool,
                 num_queries: int,
                 num_scale: int = 4,
                 use_p2: bool = False,
                 **kwargs
                 ):
        super().__init__()
        hidden_dim, nheads = d_model, n_head
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0, apply_softmax=False)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.aux_loss = aux_loss
        self.num_queries = num_queries
        self.use_p2 = use_p2


        dimensions = [num_scale * n_head, num_scale * n_head // 4, num_scale * n_head // 16]

        self.lay1 = nn.Sequential(
            nn.Conv2d(dimensions[0], dimensions[1], 1, padding=0),
            nn.ReLU(),
        )
        # self.lay2 = torch.nn.Conv2d(dimensions[1], dimensions[2], 3, padding=1)
        # self.gn2 = torch.nn.GroupNorm(n_head, dimensions[2])

        self.out_lay = torch.nn.Conv2d(dimensions[1], 1, 1, padding=0)

        if self.use_p2:
            self.multi_scale = kwargs["multi_scale"]
            conv_dim = kwargs["conv_dim"]
            lateral_norm = nn.GroupNorm(32, conv_dim)
            output_norm = nn.GroupNorm(32, conv_dim)

            in_channels = kwargs["in_channels"]
            use_bias = False
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, conv_dim, kernel_size=1, bias=use_bias),
                lateral_norm
            )

            self.output_conv = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
                output_norm,
                nn.ReLU(inplace=True)
            )

            if self.multi_scale:
                projection_norm = nn.GroupNorm(32, conv_dim)
                self.projection_conv = nn.Sequential(
                    # num-scale - 1 due to addition of p2
                    nn.Conv2d(conv_dim * (num_scale - 1), conv_dim, kernel_size=1, bias=use_bias),
                    projection_norm
                )
                c2_xavier_fill(self.projection_conv[0])

            c2_xavier_fill(self.lateral_conv[0])
            c2_xavier_fill(self.output_conv[0])


    def forward(self, srcs, masks, memory, level_start_index, hs, features_raw, mask_p2):
        flattened_masks_list = []
        for mask in masks:
            mask_flattened = mask.flatten(1)
            flattened_masks_list.append(mask_flattened)
        flattened_masks = torch.cat(flattened_masks_list, 1)

        seg_memory_levels = []
        seg_mask_levels = []
        for level, src in enumerate(srcs):
            bs, c, h, w = src.shape
            if level + 1 == len(srcs):
                seg_memory_level = memory[:, level_start_index[level]:, :]
                seg_mask_level = flattened_masks[:, level_start_index[level]:]
            else:
                seg_memory_level = memory[:, level_start_index[level]:level_start_index[level + 1], :]
                seg_mask_level = flattened_masks[:, level_start_index[level]:level_start_index[level + 1]]

            # interesting ddp error if we do not add contiguous here??
            seg_memory_level = seg_memory_level.permute(0, 2, 1).contiguous().view(bs, c, h, w)
            seg_mask_level = seg_mask_level.view(bs, h, w)

            seg_memory_levels.append(seg_memory_level)
            seg_mask_levels.append(seg_mask_level)

        if self.use_p2:
            if not self.multi_scale:
                seg_memory_stride8 = seg_memory_levels[0]
            else:
                seg_memory_levels_for_p2 = []
                for level, seg_memory_level in enumerate(seg_memory_levels):
                    seg_memory_level = F.interpolate(seg_memory_level, size=srcs[0].shape[-2:],
                                                     mode="bilinear", align_corners=False)
                    seg_memory_levels_for_p2.append(seg_memory_level)
                seg_memory_levels_cat = torch.cat(seg_memory_levels_for_p2, 1)
                seg_memory_stride8 = self.projection_conv(seg_memory_levels_cat)

            x = features_raw['0'].float()
            cur_fpn = self.lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(seg_memory_stride8, size=cur_fpn.shape[-2:], mode="bilinear",
                                        align_corners=False)
            mask_features = self.output_conv(y)

            seg_memory_levels.insert(0, mask_features)
            seg_mask_levels.insert(0, mask_p2)


        outputs_seg_masks_list = []
        if not self.aux_loss:
            outputs_seg_masks = self.forward_prediction_heads(hs[-1], seg_memory_levels, seg_mask_levels)
            outputs_seg_masks_list.append(outputs_seg_masks)
        else:
            for query in hs:
                outputs_seg_masks = self.forward_prediction_heads(query, seg_memory_levels, seg_mask_levels)
                outputs_seg_masks_list.append(outputs_seg_masks)

        return outputs_seg_masks_list

    def forward_prediction_heads(self, query_output, seg_memory_levels, seg_mask_levels):
        query = self.decoder_norm(query_output)
        attn_mask_levels = []
        bs, _, h_target, w_target = seg_memory_levels[0].shape

        for level_index, (seg_memory_level, seg_mask_level) in enumerate(zip(seg_memory_levels, seg_mask_levels)):
            attn_mask_level = self.bbox_attention(query, seg_memory_level, mask=seg_mask_level)
            bs, q, n, h, w = attn_mask_level.shape
            attn_mask_level = attn_mask_level.contiguous().view(-1, n, h, w)
            attn_mask_level = F.interpolate(attn_mask_level, size=(h_target, w_target),
                                            mode="bilinear", align_corners=False)
            attn_mask_level = attn_mask_level.view(bs, q, n, h_target, w_target)
            attn_mask_levels.append(attn_mask_level)

        attn_mask_levels_cat = torch.cat(attn_mask_levels, 2).flatten(0, 1)

        # attn_mask_level = self.bbox_attention(query, seg_memory_levels[0], mask=seg_mask_levels[0])
        #
        # attn_mask_levels_cat = attn_mask_level.flatten(0, 1)
        x = attn_mask_levels_cat
        x = self.lay1(x)
        # x = self.lay2(x)
        x = self.out_lay(x)
        output_mask = x
        output_mask = output_mask.view(bs, self.num_queries, h_target, w_target)
        return output_mask
