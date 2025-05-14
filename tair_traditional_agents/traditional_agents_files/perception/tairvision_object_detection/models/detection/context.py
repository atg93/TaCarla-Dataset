import torch
from torch import nn, Tensor

from typing import Dict, List, Tuple, Optional
from collections import OrderedDict


class RetinafaceContextModule(nn.Module):
    """
    An independent context module for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
    """

    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                # nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.relu(self.conv1(x))

        x1 = self.relu(self.conv2(x))
        x2 = self.relu(self.conv3(x1))
        x3 = self.relu(self.conv4(x2))

        return torch.cat((x1, x2, x3), dim=1)


class SshContextModule(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))
        x2 = self.relu(self.conv3(x))

        x = self.relu(self.conv4(x))
        x3 = self.relu(self.conv5(x))

        return torch.cat((x1, x2, x3), dim=1)


class PyramidboxContextModule(nn.Module):
    def __init__(self, in_channels):
        #TODO: Verify Pyramidbox Context Module, e.g. kernel size, activations, how to use multi-loss
        super().__init__()
        self.conv1 = Conv1x1(in_channels, [in_channels // 4, in_channels // 4], in_channels // 2)
        self.conv2 = Conv1x1(in_channels, [in_channels // 4, in_channels // 4], in_channels)
        self.conv3 = Conv1x1(in_channels, [in_channels // 4, in_channels // 8], in_channels // 4)
        self.conv4 = Conv1x1(in_channels, [in_channels // 4, in_channels // 8], in_channels)
        self.conv5 = Conv1x1(in_channels, [in_channels // 4, in_channels // 8], in_channels // 4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))
        x2 = self.relu(self.conv3(x))

        x = self.relu(self.conv4(x))
        x3 = self.relu(self.conv5(x))

        return torch.cat((x1, x2, x3), dim=1)


class Conv1x1(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(channels[1], out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class IndependentContextNetwork(nn.Module):
    """
    n-ple context network with independent modules.

    Args:
        in_channels (int): number of channels of the input feature
        num_feature_layers (int) : number of feature layers
        context_module (nn.Module): context module type

    """

    def __init__(self, in_channels, context_module="retinaface", num_feature_layers=5):
        super().__init__()

        if context_module == 'retinaface':
            context_module = RetinafaceContextModule
        elif context_module == 'ssh':
            context_module = SshContextModule
        elif context_module == 'pyramidbox':
            context_module = PyramidboxContextModule

        self.context_network = nn.ModuleList()
        for _ in range(num_feature_layers):
            self.context_network.append(context_module(in_channels))

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the context for a set of feature maps.
        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """

        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        results = []

        for i_feature, features in enumerate(x):
            results.append(self.context_network[i_feature](features))

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class SharedContextNetwork(nn.Module):
    """
    n-ple context network with shared modules.

    Args:
        in_channels (int): number of channels of the input feature
        context_module (nn.Module): context module type
    """

    def __init__(self, in_channels, context_module="retinaface"):
        super().__init__()

        if context_module == 'retinaface':
            context_module = RetinafaceContextModule
        elif context_module == 'ssh':
            context_module = SshContextModule
        elif context_module == 'pyramidbox':
            context_module = PyramidboxContextModule

        self.context_network = context_module(in_channels)

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the context for a set of feature maps.
        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """

        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        results = []

        for i_feature, features in enumerate(x):
            results.append(self.context_network(features))

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out
