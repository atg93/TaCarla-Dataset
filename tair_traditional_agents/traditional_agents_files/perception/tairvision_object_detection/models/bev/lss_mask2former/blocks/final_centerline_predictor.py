import torch.nn as nn
import torch
from tairvision.models.transormer_utils import MLP
import torch.nn.functional as F
from tairvision.models.bev.lss_dab.lss_transformer_utils.postprocess import post_process_softmax
from tairvision.models.detection.dabdetr_sub.criterion import SetCriterion as SetCriterionDAB


class FinalCenterLinePredictor(nn.Module):
    def __init__(self, cfg, matcher):
        super().__init__()

        hidden_dim = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MASKED_TRANSFORMER_DECODER_CONFIG.hidden_dim
        mask_dim = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MASKED_TRANSFORMER_DECODER_CONFIG.mask_dim 
        
        dec_layers = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.IMPROVED_RELATIONS_NUM_LAYERS
        self.num_layers = dec_layers

        if cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.ENABLE_SEPARATE_IMPROVED_RELATIONS_FOR_FINAL_CENTERLINE:
            self.enable_separate_improved_relations = True
        else:
            self.enable_separate_improved_relations = False

        num_classes=len(cfg.MODEL.HEADDYNAMIC.SEMANTIC_SEG.WEIGHTS)
        self.num_classes = num_classes
        
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        weight_dict = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.LOSS_WEIGHT_DICT.convert_to_dict()
        aux_weight_dict = {}
        for i in range(dec_layers + 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        loss_config = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.LOSS_CONFIG.convert_to_dict()

        self.loss_function = SetCriterionDAB(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            **loss_config
        )

        self.object_mask_threshold = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.POST_PROCESSING_CONFIG.object_mask_threshold
        self.overlap_threshold = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.POST_PROCESSING_CONFIG.overlap_threshold
        
        code_size = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MASKED_TRANSFORMER_DECODER_CONFIG.code_size
        include_regression = cfg.MODEL.DYNAMIC_TRANSFORMER_DECODER.MASKED_TRANSFORMER_DECODER_CONFIG.include_regression 
        self.include_regression = include_regression

        self.regress_embed = None
        if include_regression:
            self.regress_embed = MLP(hidden_dim, hidden_dim, code_size, 3)

        reproject_mask_features = cfg.MODEL.OPTIMAL_TRANSPORT_HEAD.REPROJECT_MASK_FEATURES_IN_FINAL_PREDICTOR
        self.reproject_mask_features = reproject_mask_features
        if reproject_mask_features:
            output_norm1 = nn.GroupNorm(32, mask_dim)
            output_norm2 = nn.GroupNorm(32, mask_dim)

            output_conv = nn.Sequential(
                nn.Conv2d(mask_dim, mask_dim, kernel_size=1, stride=1, padding=0, bias=False),
                output_norm1,
                nn.ReLU(inplace=True),
                nn.Conv2d(mask_dim, mask_dim, kernel_size=1, stride=1, padding=0, bias=False),
                output_norm2,
            )
            self.reproject_mask_features_conv = output_conv

    def forward(self, outputs):
        predictions_class = []
        predictions_mask = []
        predictions_regression = []
        
        if self.enable_separate_improved_relations:
            quer_feats = outputs["outputs_filtered_lc"]["quer_feat_for_final_centerline"]
        else:
            if "quer_feat_relations" in outputs["outputs_filtered_lc"]:
                quer_feats = outputs["outputs_filtered_lc"]["quer_feat_relations"]
            else:
                quer_feats = outputs["outputs_filtered_lc"]["quer_feat"]

        mask_features = outputs["mask_features"]
        if self.reproject_mask_features:
            mask_features = self.reproject_mask_features_conv(mask_features)
        regresssion_out = outputs["outputs_filtered_lc"]["pred_regressions"]

        if not isinstance(quer_feats, list):
            pass
        else:
            for quer_feat in quer_feats:
                outputs_class, outputs_mask, regresssion_out = self.forward_prediction_heads(
                    quer_feat, mask_features, regresssion_out
                )

                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_regression.append(regresssion_out)

            assert len(predictions_class) == self.num_layers + 1

            out = {
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(
                    predictions_class, 
                    predictions_mask, 
                    predictions_regression
                ), 
                "quer_feat": quer_feat, 
                "pred_regressions": predictions_regression[-1]
            }

        return out

    def forward_prediction_heads(self, output, mask_features, regresssion_out):
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)

        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        if self.regress_embed is not None:
            regresssion_diff = self.regress_embed(decoder_output)
            regresssion_out = regresssion_out + regresssion_diff

        return outputs_class, outputs_mask, regresssion_out
    
    def get_loss(self, head_outputs, targets):
        loss = self.loss_function(head_outputs["centerline_after_relation"], targets["dab"])
        return loss
    
    def post_process(self, outputs):
        output_post = post_process_softmax(
            outputs=outputs, 
            object_mask_threshold=self.object_mask_threshold,
            overlap_threshold=self.overlap_threshold,
            number_of_classes=self.num_classes
        )
        return output_post
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_regression):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.include_regression:
            return [{"pred_logits": a, "pred_masks": b, "pred_regressions": c}
                    for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_regression[:-1])]
        else:
            return [{"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]