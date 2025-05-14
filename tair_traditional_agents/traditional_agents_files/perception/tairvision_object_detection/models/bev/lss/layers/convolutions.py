import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        if scale_factor is not None:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        else:
            self.upsample = None

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        if self.upsample is not None:
            x_to_upsample = self.upsample(x_to_upsample)
        else:
            x_to_upsample = F.interpolate(x_to_upsample, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)


class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        if scale_factor is not None:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        else:
            self.upsample = None
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        if self.upsample is not None:
            x = self.upsample(x)
        else:
            x = F.interpolate(x, size=x_skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x + x_skip


class SimpleFusion(nn.Module):
    def __init__(self, in_channels_pixels, in_channels_pcloud, out_channels):
        super().__init__()

        in_channels = in_channels_pixels + in_channels_pcloud

        if in_channels == out_channels:
            self.conv = None
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.in_channels_pcloud = in_channels_pcloud

    def forward(self, x, feats_pcloud=None):

        if self.in_channels_pcloud > 0:
            x = torch.cat([x, feats_pcloud], dim=1)

        if self.conv is not None:
            x = self.conv(x)
        return x
