from collections import OrderedDict

import torch
import torch.nn as nn

from tairvision.models.bev.fiery.layers.convolutions import ConvBlock
from tairvision.models.bev.common.utils.geometry import warp_features


class SpatialGRU(nn.Module):
    """A GRU cell that takes an input tensor [BxTxCxHxW] and an optional previous state and passes a
    convolutional gated recurrent unit over the data"""

    def __init__(self, input_size, hidden_size, gru_bias_init=0.0, norm='bn', activation='relu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init

        self.conv_update = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.conv_state_tilde = ConvBlock(
            input_size + hidden_size, hidden_size, kernel_size=3, bias=False, norm=norm, activation=activation
        )

    def forward(self, x, state=None, flow=None, mode='bilinear'):
        # pylint: disable=unused-argument, arguments-differ
        # Check size
        assert len(x.size()) == 5, 'Input tensor must be BxTxCxHxW.'
        b, timesteps, c, h, w = x.size()
        assert c == self.input_size, f'feature sizes must match, got input {c} for layer with size {self.input_size}'

        # recurrent layers
        rnn_output = []
        rnn_state = torch.zeros(b, self.hidden_size, h, w, device=x.device) if state is None else state
        for t in range(timesteps):
            x_t = x[:, t]
            if flow is not None:
                rnn_state = warp_features(rnn_state, flow[:, t], mode=mode)

            # propagate rnn state
            rnn_state = self.gru_cell(x_t, rnn_state)
            rnn_output.append(rnn_state)

        # reshape rnn output to batch tensor
        return torch.stack(rnn_output, dim=1)

    def gru_cell(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, gru_bias_init=0.0, norm='bn', activation='relu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init

        self.conv_update = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.conv_state_tilde = ConvBlock(
            input_size + hidden_size, hidden_size, kernel_size=3, bias=False, norm=norm, activation=activation
        )

    def forward(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde(
            torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output
