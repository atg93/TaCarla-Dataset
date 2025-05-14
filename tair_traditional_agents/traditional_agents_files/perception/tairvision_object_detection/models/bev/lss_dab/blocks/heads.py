import torch
import torch.nn as nn
from tairvision.models.segmentation.mask2former_sub import ShapeSpec
from tairvision.models.segmentation.mask2former_sub.matcher import HungarianMatcher
# from tairvision.models.bev.lss_dab.blocks.dabdetr_sub import DABDeformableDETR
from tairvision.models.detection.dab_detr import DABDeformableDETR
import torch.nn.functional as F
from tairvision.models.transormer_utils import DeformableTransformer
from tairvision.models.transormer_utils import PositionEmbeddingSine
from tairvision.models.detection.dabdetr_sub.criterion import SetCriterion as SetCriterionDAB
from tairvision.models.bev.lss_dab.lss_transformer_utils.postprocess import post_process_softmax, post_process_sigmoid


class DABDynamicHead(nn.Module):
    def __init__(self, cfg, n_classes, strides, **kwargs):
        super(DABDynamicHead, self).__init__()

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

        transformer_config = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.TRANSFORMER.convert_to_dict()
        transformer_config = {k.lower(): v for k, v in transformer_config.items()}
        transformer = DeformableTransformer(**transformer_config)


        hidden_dim = transformer.d_model
        N_steps = hidden_dim // 2
        pe_params = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.POSITIONAL_ENCODING.convert_to_dict()
        pe_params = {k.lower(): v for k, v in pe_params.items()}
        positional_embedding = PositionEmbeddingSine(N_steps, **pe_params)

        dab_config = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.DAB_CONFIG.convert_to_dict()
        dab_config = {k.lower(): v for k, v in dab_config.items()}

        segmentation_config = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.SEGMENTATION_HEAD.convert_to_dict()

        dab_config["segmentation_head"] = {}
        dab_config["segmentation_head"]["name"] = segmentation_config["NAME"]

        segmentation_head_config = {k.lower(): v for k, v in segmentation_config["CONFIG"].items()}
        dab_config["segmentation_head"]["config"] = segmentation_head_config

        matcher_config = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MATCHER_CONFIG.convert_to_dict()
        matcher_config = {k.lower(): v for k, v in matcher_config.items()}
        matcher = HungarianMatcher(
            **matcher_config
        )

        dec_layers = transformer_config["num_decoder_layers"]
        self.dec_layers = dec_layers

        self.class_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.mask_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.dice_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.bbox_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.giou_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        if cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.DAB_CONFIG.EXTRA_REGRESSION_TARGET:
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

        if "labels_ce" in self.loss_function.losses:
            n_classes_dab = n_classes + 1
        else:
            n_classes_dab = n_classes

        self.model = DABDeformableDETR(
            input_shape,
            transformer,
            positional_embedding,
            num_classes=n_classes_dab,
            **dab_config,
            dn_training=False
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

        mask_raw = torch.zeros_like(features["-1"][:, 0]).bool()
        out, _ = self.model(features, mask_raw)
        return out

    def get_loss(self, head_outputs, targets):

        class_factor = 1 / torch.exp(self.class_weight)
        mask_factor = 1 / torch.exp(self.mask_weight)
        dice_factor = 1 / torch.exp(self.dice_weight)
        bbox_factor = 1 / torch.exp(self.bbox_weight)
        giou_factor = 1 / torch.exp(self.giou_weight)
        l1_factor = 1 / torch.exp(self.l1_weight)

        loss = self.loss_function(head_outputs, targets["dab"])

        weight_dict_learnable = {"loss_ce": class_factor, "loss_mask": mask_factor, "loss_dice": dice_factor, "loss_bbox": bbox_factor, "loss_giou": giou_factor, "loss_l1": l1_factor}
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
                'uncertainty_dyn_bbox': 0.5 * self.bbox_weight,
                'uncertainty_dyn_giou': 0.5 * self.giou_weight, 
                'uncertainty_dyn_l1': 0.5 * self.l1_weight,
            }
        )

        factor = {
            'dyn_class': class_factor,
            'dyn_mask': mask_factor,
            'dyn_dice': dice_factor,
            'dyn_bbox': bbox_factor,
            'dyn_giou': giou_factor, 
            'dyn_l1': l1_factor,
        }

        return loss, factor

    def post_process(self, outputs):
        if "labels_ce" in self.loss_function.losses:
            output_post = post_process_softmax(
                outputs=outputs, 
                object_mask_threshold=self.object_mask_threshold,
                overlap_threshold=self.overlap_threshold,
                number_of_classes=self.number_of_classes
            )
        elif "labels" in self.loss_function.losses:
            output_post = post_process_sigmoid(
                outputs=outputs, 
                object_mask_threshold=self.object_mask_threshold,
                overlap_threshold=self.overlap_threshold,
                number_of_classes=self.number_of_classes
            )
        else:
            raise NotImplementedError
        return output_post