import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from ..transormer_utils import DeformableTransformerEncoder, DeformableTransformerEncoderLayer, PositionEmbeddingSine
from ..transormer_utils import DeformableTransformerDecoder, DeformableTransformerDecoderLayer, MHAttentionMap
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, NamedTuple

from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from ._utils import _SimpleSegmentationModel, preprocess_temporal_info

__all__ = ["TemporalEncoder", "TemporalDeformableEncoder", "TemporalDeformableTransformer"]


class TemporalEncoder(_SimpleSegmentationModel):
    def __init__(self, *args, **kwargs):
        super(TemporalEncoder, self).__init__(*args, **kwargs)
        self.number_of_temporal_frames = None
        self.feature_dim = None
        self.d_ffn = None
        self.n_points = None
        self.number_of_layers = None
        self.aspp_number_of_input_channels = None
        self.number_of_query = None
        self.number_of_heads = None

        # TODO Only for ResNet18 now, the channel number is directly assumed as 512
        self.get_kwargs_arguments(kwargs)
        self.assert_check()
        self.reducer = self.create_reducer()

    def forward(self, image):
        image, batch_size = preprocess_temporal_info(image)
        features = self.backbone(image)
        reduced_features = self.reducer(features["out"])
        encoder_output, spatial_shapes, level_start_index, valid_ratios = self.get_encoded_features(reduced_features, batch_size)
        decoded_output = self.get_decoded_features(encoder_output, spatial_shapes, level_start_index, valid_ratios, batch_size)
        result = self.feedforward_head(features, decoded_output, image)

        return result

    def get_kwargs_arguments(self, kwargs):
        self.number_of_temporal_frames = kwargs.get("number_of_temporal_frames", None)
        self.feature_dim = kwargs.get("hidden_dimension_of_transformer", None)
        self.d_ffn = kwargs.get("expansion_dimension_of_transformer", None)
        self.n_points = kwargs.get("number_of_attention_points", None)
        self.number_of_layers = kwargs.get("number_of_layers", None)
        self.aspp_number_of_input_channels = kwargs.get("aspp_number_of_input_channels", None)
        self.number_of_query = kwargs.get("number_of_query", None)
        self.number_of_heads = kwargs.get("number_of_heads", None)

    def create_reducer(self):
        reducer = nn.Sequential(
            nn.Conv2d(512, 512 // self.number_of_temporal_frames, (1, 1), bias=False),
            nn.BatchNorm2d(512 // self.number_of_temporal_frames),
            nn.ReLU(inplace=True),
        )
        return reducer

    def assert_check(self):
        pass

    def get_encoded_features(self, reduced_features, batch_size):
        features_out_shape = reduced_features.shape
        encoder_output = \
            reduced_features.view(-1, self.number_of_temporal_frames * features_out_shape[1],
                                  features_out_shape[2], features_out_shape[3])
        return encoder_output, None, None, None

    def get_decoded_features(self, encoder_output, spatial_shapes, level_start_index, valid_ratios, batch_size):
        return encoder_output

    def feedforward_head(self, features, encoder_output, image):
        # TODO only for simple deeplab head, only out keyword has been changed
        features["out"] = encoder_output
        result = OrderedDict()
        self._get_result_from_classifiers(features, result, self.classifier_keys, self.classifier_list, image)

        if self.aux_classifier_list is not None:
            self._get_result_from_classifiers(features, result, self.aux_classifier_keys, self.aux_classifier_list,
                                              image)

        return result


class TemporalDeformableEncoder(TemporalEncoder):
    def __init__(self, *args, **kwargs):
        super(TemporalDeformableEncoder, self).__init__(*args, **kwargs)
        encoder_layer = DeformableTransformerEncoderLayer(d_model=self.feature_dim,
                                                          d_ffn=self.d_ffn,
                                                          n_levels=self.number_of_temporal_frames,
                                                          n_points=self.n_points,
                                                          n_heads=self.number_of_heads)
        self.encoder: DeformableTransformerEncoder = DeformableTransformerEncoder(encoder_layer, num_layers=self.number_of_layers)
        self.level_embed = nn.Parameter(torch.Tensor(self.number_of_temporal_frames, self.feature_dim))
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=self.feature_dim // 2)
        normal_(self.level_embed)

    def assert_check(self):
        assert self.aspp_number_of_input_channels == self.feature_dim * self.number_of_temporal_frames, \
            "Number of input channels of aspp should be equal to output of deformable encoder"

    def create_reducer(self):
        reducer = nn.Sequential(
            nn.Conv2d(512, self.feature_dim, (1, 1), bias=False),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
        )
        return reducer

    def get_encoded_features(self, reduced_features, batch_size):
        features_out_shape = reduced_features.shape
        spatial_shapes = [(features_out_shape[-2], features_out_shape[-1])] * self.number_of_temporal_frames
        feature_out_new = \
            reduced_features.view(batch_size, self.number_of_temporal_frames, features_out_shape[1],
                                  features_out_shape[2], features_out_shape[3])
        pos = self.positional_encoding(feature_out_new[:, 0, 0, ...])[:, None, ...].repeat(1, self.number_of_temporal_frames, 1, 1, 1)
        pos = pos + self.level_embed[None, :, :, None, None]
        feature_out_new = feature_out_new.permute(0, 2, 1, 3, 4).flatten(2).transpose(1, 2)
        pos = pos.permute(0, 2, 1, 3, 4).flatten(2).transpose(1, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feature_out_new.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.ones([batch_size, self.number_of_temporal_frames, 2], dtype=torch.float32,
                                  device=feature_out_new.device)

        encoder_output = self.encoder(src=feature_out_new,
                                      spatial_shapes=spatial_shapes,
                                      level_start_index=level_start_index,
                                      valid_ratios=valid_ratios,
                                      pos=pos)

        return encoder_output, spatial_shapes, level_start_index, valid_ratios

    def get_decoded_features(self, encoder_output, spatial_shapes, level_start_index, valid_ratios, batch_size):
        encoder_output = encoder_output.transpose(1, 2).reshape(
            batch_size, -1,
            spatial_shapes[0][-2],
            spatial_shapes[0][-1])

        return encoder_output


class TemporalDeformableTransformer(TemporalDeformableEncoder):
    def __init__(self, *args, **kwargs):
        super(TemporalDeformableTransformer, self).__init__(*args, **kwargs)
        self.reference_points = nn.Linear(self.feature_dim, 2)
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)

        self.query_embed = nn.Embedding(self.number_of_query, self.feature_dim * 2)

        decoder_layer = DeformableTransformerDecoderLayer(d_model=self.feature_dim,
                                                          d_ffn=self.d_ffn,
                                                          n_levels=self.number_of_temporal_frames,
                                                          n_points=self.n_points,
                                                          n_heads=self.number_of_heads)

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_layers=self.number_of_layers)

        self.mask_attention = MHAttentionMap(self.feature_dim, self.feature_dim, num_heads=8, dropout=0)

    def assert_check(self):
        assert self.aspp_number_of_input_channels == self.number_of_query * self.number_of_heads, \
            "Number of input channels of aspp should be equal to output of deformable decoder"

    def get_decoded_features(self, encoder_output, spatial_shapes, level_start_index, valid_ratios, batch_size):
        query_embeds = self.query_embed.weight

        query_embed, tgt = torch.split(query_embeds, self.feature_dim, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        tgt = tgt.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()

        hs, inter_references = self.decoder(tgt, reference_points, encoder_output,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed)

        encoder_output = encoder_output.transpose(1, 2).view(
            batch_size, self.feature_dim, self.number_of_temporal_frames,
            spatial_shapes[0][-2],
            spatial_shapes[0][-1])

        attention_mask = self.mask_attention(hs, encoder_output[:, :, -1, ...])
        attention_mask = attention_mask.view(batch_size, -1, spatial_shapes[0][-2], spatial_shapes[0][-1])
        return attention_mask
