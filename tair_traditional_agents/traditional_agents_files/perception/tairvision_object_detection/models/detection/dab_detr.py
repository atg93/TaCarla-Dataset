import torch.nn as nn
from .dabdetr_sub import DABDeformableDETR
from tairvision_object_detection.models.segmentation.mask2former_sub import ShapeSpec
from tairvision_object_detection.models.transormer_utils import DeformableTransformer
from .backbone_utils import resnet_fpn_backbone, _validate_trainable_layers, swin_fpn_backbone, timm_generic_fpn_backbone
import tairvision_object_detection.models.detection.dabdetr_sub.matcher as matchers
from .dabdetr_sub.criterion import SetCriterion
import torch
from tairvision_object_detection.ops import boxes as box_ops
from tairvision_object_detection.models.transormer_utils import PositionEmbeddingSine
import torch.nn.functional as F
from tairvision_object_detection.ops.misc import NestedTensor, nested_tensor_from_tensor_list, create_masks_from_feature_dict


class DeformableDAB(nn.Module):
    def __init__(self, backbone, num_classes, trainer_side=False,
                 *args, **kwargs):
        super().__init__()

        self.backbone = backbone

        input_shape = {}
        for i in range(4):
            level_shape = ShapeSpec(
                channels=backbone.num_channels[i],
                height=None,
                width=None,
                stride=backbone.strides[i]
            )
            input_shape.update({f"{i}": level_shape})

        transformer = DeformableTransformer(**kwargs["transformer"])
        self.training_side = trainer_side

        hidden_dim = transformer.d_model
        N_steps = hidden_dim // 2
        positional_embedding = PositionEmbeddingSine(N_steps, **kwargs["pe_params"])

        dn_training = False
        if "dn_params" in kwargs:
            dn_training = True
            self.dn_args = kwargs["dn_params"]
        self.dn_training = dn_training

        self.model = DABDeformableDETR(
            input_shape,
            transformer,
            positional_embedding,
            num_classes=num_classes,
            **kwargs["dab_params"],
            dn_training=dn_training
        )

        weight_dict = kwargs["loss_weight"]

        if kwargs["dab_params"]["aux_loss"]:
            aux_weight_dict = {}
            for i in range(kwargs["transformer"]["num_decoder_layers"]):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        self.weight_dict = weight_dict

        self.segmentation = False
        if self.model.segmentation:
            self.segmentation = True

        if "losses_labels" in kwargs:
            losses = kwargs["losses_labels"]
        else:
            losses = ['labels', 'boxes']

        matchers_dict = matchers.__dict__
        matcher_name = kwargs["matcher"]["name"]
        matcher_config = kwargs["matcher"]["config"]
        self.matcher = matchers_dict[matcher_name](**matcher_config)

        self.criterion = SetCriterion(num_classes=num_classes,
                                      matcher=self.matcher,
                                      weight_dict=weight_dict,
                                      losses=losses,
                                      num_points=12544,  # TODO, from the yaml file
                                      oversample_ratio=3.0,
                                      importance_sample_ratio=0.75
                                      )

        self.postprocess = PostProcess(segmentation=self.model.segmentation)

    def forward(self, image, target=None, calc_val_loss=False):
        features_raw, masks_raw = self.image_inference(image)
        if self.dn_training:
            self.dn_args["targets"] = target
            out, mask_dict = self.model(features_raw, masks_raw, dn_args=self.dn_args)
        else:
            out, mask_dict = self.model(features_raw, masks_raw)

        target_sizes = torch.stack(
            [torch.tensor([img.shape[1], img.shape[2]], device=img.device)
             for img in image], dim=0)
        out["target_sizes"] = target_sizes

        if target is not None:
            orig_target_sizes = torch.stack([trgt['orig_target_sizes'] for trgt in target], dim=0)
            out["orig_target_sizes"] = orig_target_sizes

        if self.training_side:
            self.mask_dict = mask_dict
            return out

        if self.training:
            loss = self.criterion(out, target)
            return loss
        else:
            outputs = self.postprocess(out)
            if calc_val_loss:
                loss = self.criterion(out, target)
                return loss, outputs
            else:
                return outputs
            
    def get_head_outputs(self, image):
        features_raw, masks_raw = self.image_inference(image)
        out, mask_dict = self.model(features_raw, masks_raw)
        target_sizes = torch.stack(
            [torch.tensor([img.shape[1], img.shape[2]], device=img.device)
             for img in image], dim=0)
        out["target_sizes"] = target_sizes
        return out 
    
    def get_loss(self, output, target):
        loss = self.criterion(output, target)
        return loss
    
    def image_inference(self, image):
        samples = nested_tensor_from_tensor_list(image)
        features_raw = self.backbone(samples.tensors)

        return features_raw, samples.mask


        


def dab_resnet_fpn(type="resnet50", num_classes=91,
                   trainable_backbone_layers=None,
                   **kwargs):

    trainable_backbone_layers = _validate_trainable_layers(kwargs["pretrained"], trainable_backbone_layers, 5, 3)

    returned_layers = [1, 2, 3, 4]

    backbone = resnet_fpn_backbone(type,
                                   returned_layers=returned_layers,
                                   trainable_layers=trainable_backbone_layers,
                                   extra_blocks=None,
                                   **kwargs)

    model = DeformableDAB(backbone, num_classes, **kwargs)
    return model

def dab_swin_fpn(type="swin_b", num_classes=91,
                   trainable_backbone_layers=None,
                   **kwargs):

    trainable_backbone_layers = _validate_trainable_layers(kwargs["pretrained"], trainable_backbone_layers, 5, 3)

    returned_layers = [1, 2, 3, 4]

    backbone = swin_fpn_backbone(type,
                                   returned_layers=returned_layers,
                                   trainable_layers=trainable_backbone_layers,
                                   extra_blocks=None,
                                   **kwargs)

    model = DeformableDAB(backbone, num_classes, **kwargs)
    return model

def dab_timm_fpn(type="resnet50", num_classes=91,
                   trainable_backbone_layers=None,
                   **kwargs):

    trainable_backbone_layers = _validate_trainable_layers(kwargs["pretrained"], trainable_backbone_layers, 5, 3)

    returned_layers = [1, 2, 3, 4]

    backbone = timm_generic_fpn_backbone(
        type,
        returned_layers=returned_layers,
        trainable_layers=trainable_backbone_layers,
        extra_blocks=None,
        **kwargs
    )

    model = DeformableDAB(backbone, num_classes, **kwargs)
    return model

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, threshold=0.5, segmentation=False):
        super().__init__()
        self.threshold = threshold
        self.segmentation = segmentation
        #TODO, is there a need for number of queries argument in order to replace 100s below or is it for coco

    @torch.no_grad()
    def forward(self, outputs, num_queries=100, select_topk=True):
        """ Perform the computation
        Parameters:
            original_sizes:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        target_sizes = outputs['target_sizes']
        out_queries = outputs['quer_feat']
        channel_dim = out_queries.shape[2]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        # TODO, Why 100 objects, is it only for coco??
        # Interesting observation is that topk is selected from query x number of classes
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_queries, dim=1)
        topk_boxes = topk_indexes // out_logits.shape[2]
        if select_topk:
            labels = topk_indexes % out_logits.shape[2]
            scores = topk_values
        else:
            label_score_tuple = prob.max(-1)
            labels = label_score_tuple[1]
            scores = label_score_tuple[0]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if select_topk:
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        target_boxes = boxes * scale_fct[:, None, :]

        if select_topk:
            queries = torch.gather(out_queries, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, channel_dim))
        else:
            queries = out_queries

        # TODO, check this whether this is reasonable or not
        # max_h, max_w = target_sizes.max(0)[0].tolist()

        if self.segmentation:
            out_mask = outputs["pred_masks"]
            if select_topk:
                B, R, H, W = out_mask.shape
                out_mask = out_mask.view(B, R, H * W)
                out_mask = torch.gather(out_mask, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, H * W))
                outputs_masks = out_mask.view(B, num_queries, H, W)
            else:
                outputs_masks = out_mask
            outputs_masks = F.interpolate(outputs_masks, size=(img_h, img_w), mode="bilinear", align_corners=False)
            # TODO, why move this to cpu according to SOLQ paper
            outputs_masks = (outputs_masks.sigmoid() > self.threshold)

        if "orig_target_sizes" in outputs:
            original_target_sizes = outputs['orig_target_sizes']
            img_h, img_w = original_target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            scale_fct = scale_fct.to(boxes.device)
            original_boxes = boxes * scale_fct[:, None, :]
            results = [{'scores': s, 'labels': l, 'boxes': b, 'correctly_resized_boxes': ob, 'quer_feat': q}
                       for s, l, b, ob, q in zip(scores, labels, original_boxes, target_boxes, queries)]

            if self.segmentation:
                for result, output_mask, orig_target_size in zip(results, outputs_masks, original_target_sizes):
                    result.update({"correctly_resized_masks": output_mask.unsqueeze(1)})
                    original_mask = F.interpolate(result["correctly_resized_masks"].float(),
                                                  size=tuple(orig_target_size.int().tolist()), mode="nearest").byte()
                    result.update({"masks": original_mask})
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b, 'quer_feat': q}
                       for s, l, b, q in zip(scores, labels, target_boxes, queries)]
            if self.segmentation:
                for result, output_mask in zip(results, outputs_masks):
                    result.update({"masks": output_mask.unsqueeze(1)})

        return results


