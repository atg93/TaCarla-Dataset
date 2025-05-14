import torch
import torch.nn as nn
from tairvision.models.segmentation.mask2former_sub import MSDeformAttnPixelDecoder, ShapeSpec, MultiScaleMaskedTransformerDecoder
from tairvision.models.segmentation.mask2former_sub.matcher import HungarianMatcher
from tairvision.models.detection.dabdetr_sub.criterion import SetCriterion as SetCriterionDAB
import torch.nn.functional as F
from tairvision.models.bev.lss_dab.lss_transformer_utils.postprocess import post_process_softmax

class Mask2FormerDynamicHead(nn.Module):
    def __init__(self, cfg, n_classes, strides, **kwargs):
        super(Mask2FormerDynamicHead, self).__init__()

        self.number_of_classes = n_classes
        self.batch_size = cfg.BATCHSIZE

        input_shape = {}
        level_shape = ShapeSpec(
            channels=cfg.MODEL.ENCODER.OUT_CHANNELS,
            height=None,
            width=None,
            stride=1
        )
        input_shape.update({"-1": level_shape})

        for i in range(4):
            level_shape = ShapeSpec(
                channels=cfg.MODEL.DECODER.BACKBONE.PYRAMID_CHANNELS,
                height=None,
                width=None,
                stride=strides[i]
            )
            input_shape.update({f"{i}": level_shape})

        self.enable_pixel_encoder = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.ENABLE_PIXEL_ENCODER
        if self.enable_pixel_encoder:
            pixel_encoder_config = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.PIXEL_ENCODER_CONFIG.convert_to_dict()
            pixel_encoder_config = {k.lower(): v for k, v in pixel_encoder_config.items()}
            self.pixel_encoder = MSDeformAttnPixelDecoder(
                input_shape=input_shape,
                **pixel_encoder_config
            )
        else:
            self.mask_features = nn.Conv2d(
                cfg.MODEL.ENCODER.OUT_CHANNELS,
                cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MASKED_TRANSFORMER_DECODER_CONFIG.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        masked_transformer_decoder_config = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MASKED_TRANSFORMER_DECODER_CONFIG.convert_to_dict()
        masked_transformer_decoder_config = {k.lower(): v for k, v in masked_transformer_decoder_config.items()}
        self.transformer_decoder = MultiScaleMaskedTransformerDecoder(
            num_classes=n_classes,
            **masked_transformer_decoder_config
        )

        matcher_config = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MATCHER_CONFIG.convert_to_dict()
        matcher_config = {k.lower(): v for k, v in matcher_config.items()}
        matcher = HungarianMatcher(
            **matcher_config
        )
        self.matcher = matcher

        dec_layers = masked_transformer_decoder_config["dec_layers"]
        self.dec_layers = dec_layers

        self.class_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.mask_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.dice_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        if cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MASKED_TRANSFORMER_DECODER_CONFIG.INCLUDE_REGRESSION:
            self.l1_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            self.l1_weight = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        weight_dict = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.LOSS_WEIGHT_DICT.convert_to_dict()
        weight_dict = {k.lower(): v for k, v in weight_dict.items()}

        aux_weight_dict = {}
        for i in range(dec_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        loss_config = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.LOSS_CONFIG.convert_to_dict()
        loss_config = {k.lower(): v for k, v in loss_config.items()}

        self.loss_function = SetCriterionDAB(
            num_classes=self.number_of_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            **loss_config
        )

        self.object_mask_threshold = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.POST_PROCESSING_CONFIG.OBJECT_MASK_THRESHOLD
        self.overlap_threshold = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.POST_PROCESSING_CONFIG.OVERLAP_THRESHOLD

    def get_head_outputs(self, feats_dec):
        features = {}
        for key, feature in feats_dec.items():
            b, s, c, h, w = feature.shape
            feature = feature.view(b * s, c, h, w)
            if key == "y":
                features.update({"-1": feature})
            else:
                features.update({key: feature})

        if self.enable_pixel_encoder:
            mask_features, out, multi_scale_features = self.pixel_encoder.forward_features(features)
        else:
            multi_scale_features = []
            multi_scale_features.append(features["3"])
            multi_scale_features.append(features["2"])
            multi_scale_features.append(features["1"])
            mask_features = self.mask_features(features["-1"])
        out = self.transformer_decoder(x=multi_scale_features, mask_features=mask_features)
        return out

    def get_loss(self, head_outputs, targets):

        class_factor = 1 / torch.exp(self.class_weight)
        mask_factor = 1 / torch.exp(self.mask_weight)
        dice_factor = 1 / torch.exp(self.dice_weight)
        l1_factor = 1 / torch.exp(self.l1_weight)

        loss = self.loss_function(head_outputs, targets["dab"])

        weight_dict_learnable = {"loss_ce": class_factor, "loss_mask": mask_factor, "loss_dice": dice_factor, "loss_l1": l1_factor}
        aux_weight_dict = {}
        for i in range(self.dec_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict_learnable.items()})
        weight_dict_learnable.update(aux_weight_dict)

        for k in list(loss.keys()):
            if k in weight_dict_learnable:
                loss[k] *= weight_dict_learnable[k]

        loss.update(
            {
                'uncertainty_dyn_class': 0.5 * self.class_weight,
                'uncertainty_dyn_mask': 0.5 * self.mask_weight,
                'uncertainty_dyn_dice': 0.5 * self.dice_weight,
                'uncertainty_dyn_l1': 0.5 * self.l1_weight
            }
        )

        factor = {
            'dyn_class': class_factor,
            'dyn_mask': mask_factor,
            'dyn_dice': dice_factor, 
            'dyn_l1': l1_factor
        }

        return loss, factor

    def post_process(self, outputs):
        output_post = post_process_softmax(
            outputs=outputs, 
            object_mask_threshold=self.object_mask_threshold,
            overlap_threshold=self.overlap_threshold,
            number_of_classes=self.number_of_classes
        )
        return output_post