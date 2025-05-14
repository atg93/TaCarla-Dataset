from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Any

from torch import nn, Tensor

try:
    from ..models.transormer_utils import DeformableTransformer, PositionEmbeddingSine
    from tairvision.models.segmentation.mask2former_sub.pixel_decoder import MSDeformAttnPixelDecoder, ShapeSpec
except:
    print("MSDA is missing")

from .feature_pyramid_network import ExtraFPNBlock


class DeformableEncoderNeck(nn.Module):
    def __init__(
            self,
            in_channels_list: List[int],
            out_channels: int,
            strides: List[int],
            extra_blocks: Optional[ExtraFPNBlock] = None
    ):
        super(DeformableEncoderNeck, self).__init__()

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks
        if self.extra_blocks is not None and not self.extra_blocks.before_pyramid:
            in_channels_list = in_channels_list[:-self.extra_blocks.num_levels]

        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=1024//2, normalize=True)

        input_shape = {}
        for i in range(3):
            level_shape = ShapeSpec(
                channels=in_channels_list[i],
                height=None,
                width=None,
                stride=strides[i+1]
            )
            input_shape.update({f"p{i + 3}": level_shape})

        out_shape = ShapeSpec(channels=out_channels, height=None, width=None, stride=strides[-1])
        input_shape.update({"out": out_shape})

        # TODO: Add below params into config.yaml?
        self.pixel_encoder = MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=256,
            common_stride=8,
            transformer_in_features=["p5", "p4", "p3"]  # TODO: what if Use_P2
        )

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the deformable encoder neck for a set of feature maps.
        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after deformable encoder neck.
        """
        names = list(x.keys())
        x_values = list(x.values())

        if self.extra_blocks is not None and self.extra_blocks.before_pyramid:
            x_values, names = self.extra_blocks(x_values, x_values, names)

        # Rename OrderedDict keys
        new_keys = []
        for i in range(len(names)):
            new_keys.append("p" + str(i+3))
        temp_dict = dict(zip(names, new_keys))
        renamed_dict = {temp_dict[old_key]: value for old_key, value in x.items()}

        mask_features, out, multi_scale_features = self.pixel_encoder.forward_features(renamed_dict)

        if self.extra_blocks is not None and not self.extra_blocks.before_pyramid:
            multi_scale_features, names = self.extra_blocks(multi_scale_features, list(renamed_dict.values()), new_keys)

        # Make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, multi_scale_features)])
        return out