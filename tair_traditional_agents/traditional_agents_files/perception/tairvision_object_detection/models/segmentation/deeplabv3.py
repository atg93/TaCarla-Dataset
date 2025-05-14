"""
The related repositories are
Nvidia Semantic Segmentation - https://github.com/NVIDIA/semantic-segmentation
DeeplabV3Plus - https://github.com/VainF/DeepLabV3Plus-Pytorch
Panoptic Deeplab - https://github.com/bowenc0221/panoptic-deeplab
"""

import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from .multi_level_losses import AdaptiveModule,EmbeddingModule, SelfTraining
from ._utils import _SimpleSegmentationModel, convert_to_separable_conv, preprocess_temporal_info

__all__ = ["DeepLabV3", "PanopticDeepLabHeadPlusGeneric"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabV3_mc(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    def forward_backbone(self, image):
        features = self.backbone(image)
        result = OrderedDict()

        return features, result

    def forward_mc(self, image, features, result):
        self._get_result_from_classifiers(features, result, self.classifier_keys, self.classifier_list, image)

        if self.aux_classifier_list is not None:
            self._get_result_from_classifiers(features, result, self.aux_classifier_keys, self.aux_classifier_list, image)
        return result

    pass


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, size, atrous_rates, **kwargs):
        deployment_mode = kwargs.get('deployment_mode', False)
        aspp_number_of_channels = kwargs.get('aspp_number_of_channels')
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, atrous_rates, size, deployment_mode, out_channels=aspp_number_of_channels),
            nn.Conv2d(aspp_number_of_channels, aspp_number_of_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_number_of_channels),
            nn.ReLU(),
            nn.Conv2d(aspp_number_of_channels, num_classes, 1)
        )

        self.output_key = kwargs.get('output_key', 'out')

    def forward(self, features):
        x = features[self.output_key]
        x = super(DeepLabHead, self).forward(x)
        return x

class DeepLabHeadX(nn.Sequential):
    def __init__(self, in_channels, num_classes, size, atrous_rates, **kwargs):
        deployment_mode = kwargs.get('deployment_mode', False)
        aspp_number_of_channels = kwargs.get('aspp_number_of_channels')
        super(DeepLabHeadX, self).__init__(
            ASPP(in_channels, atrous_rates, size, deployment_mode, out_channels=aspp_number_of_channels),
            nn.Conv2d(aspp_number_of_channels, aspp_number_of_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_number_of_channels),
            nn.ReLU(),
            nn.Conv2d(aspp_number_of_channels, num_classes, 1)
        )
        self.output_key = kwargs.get('output_key', 'out')
        self.lambda_st  = kwargs.get('lambda_st', 1)

    def forward(self, features):
        x = features[self.output_key]
        x = self[0].forward(x)
        x = self[1].forward(x)
        x = self[2].forward(x)
        x = self[3].forward(x)
        self.z = x
        x = self[4].forward(x)
        return x

    def calculate_adaptive_loss(self,device):
        input   = self.z
        batch_size , c1,H,W      = input.size()

        adap_module = AdaptiveModule(c1, H, W, device).to(device)

        f_inter     = adap_module(input)


        return f_inter

    def calculate_triplet_loss(self,positives, negatives, device):
        input   = self.z
        B, c1, H1, W1 = input.size()
        _, H2, W2 = positives.size()

        # TODO: Maybe After this point turn this into another function. Maybe. Just a suggestion. :)
        positives      = positives.to(torch.float)
        down_positives = F.interpolate(positives.view(B,1, H2, W2),size=(H1,W1),mode="bilinear",align_corners=False)
        negatives      = torch.from_numpy(negatives).to(device)
        down_negatives = F.interpolate(negatives.view(B,1, H2, W2),size=(H1,W1),mode="bilinear",align_corners=False)
        down_negatives = down_negatives - down_positives
        down_negatives = torch.where(down_negatives<=0,0.0,down_negatives)

        down_positives = torch.mul(down_positives, input)
        down_negatives = torch.mul(down_negatives, input)

        global_avg     = nn.AvgPool2d(H1, W1)

        down_negatives = global_avg( down_negatives).to(device)
        down_positives = global_avg( down_positives).to(device)

        embedding      = EmbeddingModule(c1).to(device)

        neg = torch.squeeze(down_negatives.float())
        pos = torch.squeeze(down_positives.float())

        down_negatives = embedding(neg)
        down_positives = embedding(pos)

        # Calculate L2 norm distance as a matrix
        PP = torch.cdist(down_positives, down_positives, p=2.0)
        NP = torch.cdist(down_negatives, down_positives, p=2.0)

        # Convert to Lower triangle matrix bcs it is symmetric
        NP = torch.tril(NP)
        PP = torch.tril(PP,diagonal=-1)

        return torch.sum(NP)+torch.sum(PP)


    def main_loss(self,out,mask,target_out,device):
        # Triplet loss and adaptive module can be called here if desired
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
        self_training = SelfTraining(0.0, 0.0).to(device)
        loss_st = self_training(target_out)
        loss_ce = ce_loss(out,mask)
        return loss_ce + self.lambda_st * loss_st


    def set_loss_function(self):
        return self.main_loss




class DeepLabHeadPlusGeneric(nn.Module):
    def __init__(self, in_channels, num_classes, size, atrous_rates,
                 number_of_channel_levels,
                 level_sizes,
                 activate_levels,
                 number_of_reduction_channel_levels,
                 aspp_channels,
                 decoder_channels,
                 head_channels,
                 kernel_size,
                 **kwargs):
        super(DeepLabHeadPlusGeneric, self).__init__()
        deployment_mode = kwargs.get('deployment_mode', False)
        self.deployment_mode = deployment_mode
        self.num_classes = num_classes

        padding = kernel_size // 2

        self.output_key = kwargs.get('output_key', 'out')

        self.project_levels = []
        self.upsample_levels = []
        self.fuse_channel_levels = []
        self.output_keys_levels = []
        self.activate_levels = activate_levels
        self.number_of_levels = len(self.activate_levels)

        number_of_channels_fusion_modules = [decoder_channels] * sum(activate_levels)
        number_of_channels_fusion_modules[-1] = aspp_channels

        for lvl_index, (lvl_bool, lvl_channel, lvl_size, lvl_reduction_channel, fuse_in_channels) in \
                enumerate(
                    zip(activate_levels, number_of_channel_levels, level_sizes,
                        number_of_reduction_channel_levels, number_of_channels_fusion_modules)
                ):

            if lvl_bool is True:
                lvl_project_module = nn.Sequential(
                    nn.Conv2d(lvl_channel, lvl_reduction_channel, (1, 1), bias=False),
                    nn.BatchNorm2d(lvl_reduction_channel),
                    nn.ReLU(inplace=True),
                )

                self.project_levels.append(lvl_project_module)
                self.add_module(f'project_levels_{lvl_index}', lvl_project_module)

                lvl_fuse_module = nn.Sequential(
                    nn.Conv2d(lvl_reduction_channel + fuse_in_channels, decoder_channels, (kernel_size, kernel_size),
                              padding=padding, bias=False),
                    nn.BatchNorm2d(decoder_channels),
                    nn.ReLU(inplace=True),
                )

                self.fuse_channel_levels.append(lvl_fuse_module)
                self.add_module(f'fuse_levels_{lvl_index}', lvl_fuse_module)

                lvl_upsample_module = \
                    nn.Upsample(size=lvl_size, mode='bilinear', align_corners=False)

                self.upsample_levels.append(lvl_upsample_module)
                self.add_module(f'upsample_levels_{lvl_index}', lvl_upsample_module)

                self.output_keys_levels.append(f"stage{lvl_index + 1}")

        self.aspp = ASPP(in_channels, atrous_rates, size, deployment_mode, out_channels=aspp_channels)

        # TODO, How can we generalize this part, other heads cannot be utilized in panoptic deeplab head
        if isinstance(num_classes, list):
            classifier = []
            for num_class in num_classes:
                classifier.append(nn.Sequential(
                    nn.Conv2d(decoder_channels, head_channels, (kernel_size, kernel_size), padding=padding, bias=False),
                    nn.BatchNorm2d(head_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_channels, num_class, (1, 1))
                ))
            self.classifier = nn.ModuleList(classifier)
        else:
            self.classifier = nn.Sequential(
                nn.Conv2d(decoder_channels, head_channels, (kernel_size, kernel_size), padding=padding, bias=False),
                nn.BatchNorm2d(head_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, num_classes, (1, 1))
            )

        self._init_weight()

    def forward(self, features):

        output_feature = self.aspp(features[self.output_key])
        for lvl_index in range(self.number_of_levels - 1, -1, -1):
            if self.activate_levels[lvl_index] is True:
                lvl_feature = self.project_levels[lvl_index](features[self.output_keys_levels[lvl_index]])
                if self.deployment_mode:
                    output_feature_upsampled = self.upsample_levels[lvl_index](output_feature)
                else:
                    size = lvl_feature.shape[-2:]
                    output_feature_upsampled = F.interpolate(output_feature, size=size, mode='bilinear',
                                                             align_corners=False)

                output_feature = self.fuse_channel_levels[lvl_index](
                    torch.cat([lvl_feature, output_feature_upsampled], dim=1))

        if isinstance(self.num_classes, list):
            return_list = []
            for i in range(len(self.num_classes)):
                return_list.append(self.classifier[i](output_feature))
            return return_list
        else:
            return self.classifier(output_feature)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, size, **kwargs):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(size=size, mode='bilinear', align_corners=False))

        self.deployment_mode = kwargs.get('deployment_mode', False)

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            if isinstance(mod, nn.Upsample) and not self.deployment_mode:
                x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            else:
                x = mod(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, size, deployment_mode, out_channels):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels, size,
                                   deployment_mode=deployment_mode))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            # TODO, Nvidia Semantic Segmentation Library choses kernel as 3x3,
            #  However Panoptic Deeplab repo also implements like this
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))  # TODO, There is not dropout in Nvidia repo

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHeadPlusGenericDepthWise(DeepLabHeadPlusGeneric):
    def __init__(self, *args, **kwargs):
        super(DeepLabHeadPlusGenericDepthWise, self).__init__(*args, **kwargs)
        self = convert_to_separable_conv(self)


class PanopticDeepLabHeadPlusGenericDepthWise(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PanopticDeepLabHeadPlusGenericDepthWise, self).__init__()
        self.semantic = DeepLabHeadPlusGenericDepthWise(*args, **kwargs, **kwargs['semantic'])
        kwargs['num_classes'] = [1, 2]
        self.instance = DeepLabHeadPlusGenericDepthWise(*args, **kwargs, **kwargs['instance'])

    def forward(self, x):
        semantic_result = self.semantic(x)
        instance_result = self.instance(x)
        result = {"semantic": semantic_result, "center": instance_result[0], "offset": instance_result[1]}

        return result


class PanopticDeepLabHeadPlusGeneric(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PanopticDeepLabHeadPlusGeneric, self).__init__()
        self.semantic = DeepLabHeadPlusGeneric(*args, **kwargs, **kwargs['semantic'])
        kwargs['num_classes'] = [1, 2]
        self.instance = DeepLabHeadPlusGeneric(*args, **kwargs, **kwargs['instance'])

    def forward(self, x):
        semantic_result = self.semantic(x)
        instance_result = self.instance(x)
        result = {"semantic": semantic_result, "center": instance_result[0], "offset": instance_result[1]}

        return result
