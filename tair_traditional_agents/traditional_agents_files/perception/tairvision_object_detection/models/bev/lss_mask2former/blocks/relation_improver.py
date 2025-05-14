import torch.nn as nn
from tairvision.models.segmentation.mask2former_sub.multi_scale_decoder import SelfAttentionLayer, CrossAttentionLayer, FFNLayer


class RelationImprover(nn.Module):
    def __init__(self, cfg):
        super(RelationImprover, self).__init__()
        num_layers = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.IMPROVED_RELATIONS_NUM_LAYERS
        self.num_layers = num_layers
        nheads = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.IMPROVED_RELATIONS_NHEADS
        self.transformer_self_attention_layers_lclc = nn.ModuleList()
        self.transformer_self_attention_layers_tete = nn.ModuleList()
        self.transformer_cross_attention_layers_lcte = nn.ModuleList()
        self.transformer_cross_attention_layers_telc = nn.ModuleList()
        self.transformer_ffn_layers_lclc = nn.ModuleList()
        self.transformer_ffn_layers_tete = nn.ModuleList()
        pre_norm = False
        hidden_dim = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MASKED_TRANSFORMER_DECODER_CONFIG.HIDDEN_DIM

        if not cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_FILTERING:
            number_of_te_queries = cfg.MODEL.HEAD2D.DAB_PARAMS.NUM_QUERIES
        else:
            number_of_te_queries = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.NUMBER_OF_SELECTED_QUERIES_TE

        if not cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_FILTERING:
            number_of_lc_queries = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MASKED_TRANSFORMER_DECODER_CONFIG.NUM_QUERIES
        else:
            number_of_lc_queries = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.NUMBER_OF_SELECTED_QUERIES_LC

        self.traffic_element_pe = nn.Embedding(number_of_te_queries, hidden_dim)
        self.centerline_pe = nn.Embedding(number_of_lc_queries, hidden_dim)

        for _ in range(num_layers):
            self.transformer_self_attention_layers_lclc.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_self_attention_layers_tete.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers_lcte.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers_telc.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers_lclc.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=2 * hidden_dim,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers_tete.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=2 * hidden_dim,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )


    def forward(self, output):
        filtered_lc_outputs = output["outputs_filtered_lc"]["quer_feat"]
        filtered_te_outputs = output["outputs_filtered_te"]["quer_feat"]
        bs = filtered_lc_outputs.shape[0]
                # QxNxC
        traffic_element_pe = self.traffic_element_pe.weight.unsqueeze(1).repeat(1, bs, 1)
        centerline_pe = self.centerline_pe.weight.unsqueeze(1).repeat(1, bs, 1)

        filtered_lc_outputs_list = []
        filtered_te_outputs_list = []
        filtered_lc_outputs_list.append(filtered_lc_outputs)
        filtered_te_outputs_list.append(filtered_te_outputs)

        filtered_lc_outputs = filtered_lc_outputs.permute(1, 0, 2)
        filtered_te_outputs = filtered_te_outputs.permute(1, 0, 2)


        for i in range(self.num_layers):
            filtered_te_outputs = self.transformer_cross_attention_layers_lcte[i](
                filtered_te_outputs, filtered_lc_outputs,
                memory_mask=None,
                memory_key_padding_mask=None, 
                pos=centerline_pe, query_pos=traffic_element_pe
            )

            filtered_lc_outputs = self.transformer_cross_attention_layers_telc[i](
                filtered_lc_outputs, filtered_te_outputs,
                memory_mask=None,
                memory_key_padding_mask=None, 
                pos=traffic_element_pe, query_pos=centerline_pe
            )

            filtered_lc_outputs = self.transformer_self_attention_layers_lclc[i](
                filtered_lc_outputs, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=centerline_pe
            )

            filtered_te_outputs = self.transformer_self_attention_layers_tete[i](
                filtered_te_outputs, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=traffic_element_pe
            )

            # FFN
            filtered_lc_outputs = self.transformer_ffn_layers_lclc[i](
                filtered_lc_outputs
            )

            # FFN
            filtered_te_outputs = self.transformer_ffn_layers_tete[i](
                filtered_te_outputs
            )

            filtered_lc_outputs_list.append(filtered_lc_outputs.permute(1, 0, 2).contiguous())
            filtered_te_outputs_list.append(filtered_te_outputs.permute(1, 0, 2).contiguous())

        return filtered_lc_outputs_list, filtered_te_outputs_list