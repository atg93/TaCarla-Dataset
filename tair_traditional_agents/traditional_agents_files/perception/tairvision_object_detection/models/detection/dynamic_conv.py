import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, in_planes, ratio, K, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temprature = temprature
        assert in_planes > ratio
        hidden_planes = in_planes // ratio
        # self.net = nn.Sequential(
        #     nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
        # )

        if init_weight:
            self._initialize_weights()

    def update_temprature(self):
        if self.temprature > 1:
            self.temprature -= 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, attn=None):
        # if attn is None:
        #     attn = self.avgpool(x)  # bs,dim,1,1
        #     attn = self.net(attn).view(x.shape[0], -1)  # bs,K
        # else:
        #     attn = self.net(attn).view(attn.shape[0], -1)
        return F.softmax(attn / self.temprature, -1)


class DynamicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True, K=8,
                 temprature=30, ratio=4, init_weight=True, attn_inplanes=256):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        # self.attention = Attention(in_planes=attn_inplanes, ratio=ratio, K=K, temprature=temprature,
        #                            init_weight=init_weight)

        # self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size),
        #                            requires_grad=True)
        # if bias:
        #     self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        # else:
        #     self.bias = None

        # self.kernel_attn = nn.Sequential(nn.Linear(attn_inplanes, attn_inplanes//4, bias=False),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(attn_inplanes//4, out_planes, bias=False))

        # self.norm = nn.GroupNorm(num_groups=8, num_channels=out_planes)
        # self.act = nn.ReLU(inplace=True)

        # self.act = nn.Sigmoid()


        # if self.init_weight:
        #     self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x, query_kernel):
        bs_, in_planels, h, w = x.shape
        bs = query_kernel.shape[0]
        # softmax_att = self.attention(x, attn_kernel[:, :, 0, 0])  # bs,K
        x = x.view(1, -1, h, w)
        # weight = self.weight.view(self.K, -1)  # K,-1
        # aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
        #                                                       self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k
        # kernel_pred = self.kernel_attn(attn[:, :, 0, 0])
        # kernel_pred = F.softmax(kernel_pred/1, dim=1)

        seg_pred = F.conv2d(x, query_kernel)[0]

        # head_bs = 16
        # segs = []
        # for i in range(0, bs, head_bs):
        #     agg_weight = aggregate_weight[i*self.out_planes: (i+ head_bs)*self.out_planes]
        #     # if (self.bias is not None):
        #     #     bias = self.bias.view(self.K, -1)  # K,out_p
        #     #     aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
        #     #     output = F.conv2d(x, weight=agg_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
        #     #                       groups=self.groups * bs_, dilation=self.dilation)
        #     # else:
        #     output = F.conv2d(x, weight=agg_weight, bias=None, stride=self.stride, padding=self.padding,
        #                       groups=self.groups * bs_, dilation=self.dilation)
        #
        #     local_bs = int(agg_weight.shape[0]/32)
        #     output = output.view(local_bs, self.out_planes, h, w)
        #     output = self.act(output)
        #     output = self.norm(output)
        #
        #     # F.conv2d(cur_ins_pred_.view(1, -1, H, W), kernel_pred_[i][None, :], stride=1).view(1, -1, H, W)
        #     seg_pred = F.conv2d(output.view(1, -1, h, w), query_kernel[i: i+head_bs], groups=local_bs)[0]
        #     segs.append(seg_pred)
        #
        # segs = torch.cat(segs, 0)

        # len_x = (x ** 2).sum(dim=1).sqrt().repeat(query_kernel.shape[0], 1, 1)
        # len_q = (query_kernel ** 2).sum(dim=1).sqrt().repeat(1, *x.shape[2:])

        seg_pred = seg_pred.sigmoid()
        return seg_pred


if __name__ == '__main__':
    input = torch.randn(2, 32, 64, 64)
    m = DynamicConv(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1, bias=False)
    out = m(input)
    print(out.shape)