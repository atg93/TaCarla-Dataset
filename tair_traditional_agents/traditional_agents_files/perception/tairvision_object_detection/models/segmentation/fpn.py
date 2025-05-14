import torch.nn as nn
import torch.nn.functional as F


class FPNSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, number_of_channel_list,**kwargs):
        super().__init__()

        conv = nn.ModuleList()
        for i_level in range(len(number_of_channel_list)):
            if i_level == 0:
                conv.append(nn.ModuleList())
                conv[0].append(
                    nn.Sequential(nn.Conv2d(in_channels, number_of_channel_list[0], kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(32, number_of_channel_list[0]),
                                  nn.ReLU()
                                  )
                    )
            else:
                for i_conv in range(i_level):
                    if i_conv == 0:
                        conv.append(nn.ModuleList())
                        conv[i_level].append(
                            nn.Sequential(
                                nn.Conv2d(in_channels, number_of_channel_list[i_level], kernel_size=3, stride=1, padding=1),
                                nn.GroupNorm(32, number_of_channel_list[i_level]),
                                nn.ReLU()
                                )
                        )
                    else:
                        conv[i_level].append(
                            nn.Sequential(
                                nn.Conv2d(number_of_channel_list[i_level], number_of_channel_list[i_level], kernel_size=3, stride=1, padding=1),
                                nn.GroupNorm(32, number_of_channel_list[i_level]),
                                nn.ReLU()
                                )
                        )

        self.conv = conv
        self.conv_out = nn.Conv2d(in_channels // 2, num_classes, 1)

    def forward(self, features):
        features = list(features.values())
        sizes = [feature.shape[-2:] for feature in features]

        for i_level, feature in enumerate(features):
            if i_level == 0:
                out = self.conv[0][0](features[0])
            else:
                for i_conv in range(i_level):
                    if i_conv == 0:
                        x = features[i_level]
                    x = self.conv[i_level][i_conv](x)
                    x = F.interpolate(x, size=sizes[i_level - i_conv - 1], mode='bilinear', align_corners=False)
                out += x

        out = self.conv_out(out)

        return out

