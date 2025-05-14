import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from torchvision import models

__all__ = [
    "DAN", "DANWithMoodAgeGender", "AffinityLoss", "PartitionLoss", "AgeLoss"
]


class DAN(nn.Module):
    def __init__(self, num_class=7, num_head=4, use_gender=False,
                 backbone_path='/workspace/shared/trained_models/dan_res18_backbone/resnet18_msceleb.pth'):
        super(DAN, self).__init__()

        resnet = models.resnet18(True)

        checkpoint = torch.load(backbone_path)
        resnet.load_state_dict(checkpoint['state_dict'], strict=True)

        self.use_gender = use_gender

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            self.heads.append(CrossAttentionHead())
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

        if self.use_gender:
            self.fc2 = nn.Linear(512, 2)
            self.bn2 = nn.BatchNorm1d(2)

    def forward(self, x):
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(self.heads[i](x))

        heads = torch.stack(heads).permute([1, 0, 2])
        if heads.size(1) > 1:
            heads = F.log_softmax(heads, dim=1)

        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)

        out2 = None
        if self.use_gender:
            out2 = self.fc2(heads.sum(dim=1))
            out2 = self.bn2(out2)

        return [out, out2], x, heads


class DANWithMoodAgeGender(nn.Module):
    def __init__(self, num_class=7, num_head=4, use_mood=False, use_gender=False, use_valence=False, use_dan=False,
                 extra_layer=False,
                 backbone_path='/workspace/shared/trained_models/dan_res18_backbone/resnet18_msceleb.pth'):
        super(DANWithMoodAgeGender, self).__init__()

        resnet = models.resnet18(True)

        checkpoint = torch.load(backbone_path)
        resnet.load_state_dict(checkpoint['state_dict'], strict=True)

        self.use_gender = use_gender
        self.use_mood = use_mood
        self.use_valence = use_valence
        self.use_dan = use_dan
        self.extra_layer = extra_layer

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        self.heads = nn.ModuleList()
        for i in range(num_head):
            self.heads.append(CrossAttentionHead())

        if self.use_mood:
            self.fc = nn.Linear(512, num_class)
            self.bn = nn.BatchNorm1d(num_class)

        if self.use_gender:

            self.fc2 = nn.Linear(512, 2)
            self.fc31 = nn.Linear(512, 12)
            self.fc32 = nn.Linear(12, 1)

            if self.use_dan:
                self.bn2 = nn.BatchNorm1d(2)

            else:

                if self.extra_layer:
                    self.fc1_7 = nn.Linear(512 * 7 * 7, 512 * 7)
                    self.fc11_7 = nn.Linear(512 * 7, 512)

                else:
                    self.fc1 = nn.Linear(512 * 7 * 7, 512)

            self.fc31 = nn.Linear(512, 12)
            self.fc32 = nn.Linear(12, 1)

        if self.use_valence:
            self.fc41 = nn.Linear(512, 1)
            torch.nn.init.xavier_normal_(self.fc41.weight, gain=1 / 30)
            # self.bn41 = nn.BatchNorm1d(1)
            self.tanh1 = nn.Tanh()

            self.fc42 = nn.Linear(512, 1)
            torch.nn.init.xavier_normal_(self.fc42.weight, gain=1 / 30)
            # self.bn42 = nn.BatchNorm1d(1)
            self.tanh2 = nn.Tanh()

    def forward(self, x):
        x = self.features(x)
        heads = []

        out_exp = None
        out_gender = None
        out_age_dist = None
        out_age = None
        out_val = None
        out_aro = None

        for i in range(self.num_head):
            heads.append(self.heads[i](x))

        heads = torch.stack(heads).permute([1, 0, 2])
        if heads.size(1) > 1:
            heads = F.log_softmax(heads, dim=1)

        if self.use_mood:
            out_exp = self.fc(heads.sum(dim=1))
            out_exp = self.bn(out_exp)

        if self.use_gender:

            if self.use_dan:

                out_gender = self.fc2(heads.sum(dim=1))
                out_gender = self.bn2(out_gender)

                out_age_dist = self.fc31(heads.sum(dim=1))
                out_age = self.fc32(out_age_dist)
            else:
                N = x.shape[0]

                if self.extra_layer:
                    y = F.relu(self.fc1_7(x.view(N, -1)))
                    y = self.fc11_7(y)
                    out_gender = self.fc2(y)
                else:
                    y = F.relu(self.fc1(x.view(N, -1)))
                    out_gender = self.fc2(y)

                out_age_dist = self.fc31(y)
                out_age = self.fc32(out_age_dist)

        if self.use_valence:
            out_val = self.fc41(heads.sum(dim=1))
            # out_val = self.bn41(out_val)
            out_val = self.tanh1(out_val)

            out_aro = self.fc42(heads.sum(dim=1))
            # out_aro = self.bn42(out_aro)
            out_aro = self.tanh2(out_aro)

        return [out_exp, out_gender, out_age_dist, out_age, out_val, out_aro], x, heads


class TurkcellAgeGender(nn.Module):
    def __init__(self, num_class=7, num_head=4, use_mood=False, use_gender=False, use_valence=False, use_dan=False,
                 extra_layer=False,
                 backbone_path='/workspace/shared/trained_models/dan_res18_backbone/resnet18_msceleb.pth'):
        super(TurkcellAgeGender, self).__init__()

        self.use_gender = use_gender
        self.use_mood = use_mood
        self.use_valence = use_valence
        self.use_dan = use_dan
        self.extra_layer = extra_layer

        self.relu = nn.ReLU()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.25),
            nn.MaxPool2d(3, stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.25),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.25),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.4),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.4),
            nn.MaxPool2d(2, stride=2)
        )

        self.block6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 13 * 13, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),

        )

        self.fc1 = nn.Linear(64, 2)

        self.softmax = nn.Softmax(dim=1)

        # self.fc2 = nn.Linear(512, 2)
        # self.bn7 = nn.BatchNorm1d(2)

        self.fc31 = nn.Linear(64, 12)
        self.fc32 = nn.Linear(12, 1)

    def forward(self, x):
        out_exp = None
        out_val = None
        out_aro = None
        heads = None

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        # x = self.conv_block5(x)
        x = self.block6(x)

        out_gender = self.fc1(x)

        out_age_dist = self.fc31(x)
        out_age = self.fc32(out_age_dist)

        return [out_exp, out_gender, out_age_dist, out_age, out_val, out_aro], x, heads


class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca


class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True)
        out = x * y

        return out


class ChannelAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()
        )

    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0), -1)
        y = self.attention(sa)
        out = sa * y

        return out


class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=7, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, target_val, pred_val, target_aro, pred_aro):
        mean_target_val = torch.mean(target_val, 0)
        mean_pred_val = torch.mean(pred_val, 0)
        var_target_val = torch.var(target_val, 0)
        var_pred_val = torch.var(pred_val, 0)
        v_pred = pred_val - mean_pred_val
        v_gt = target_val - mean_target_val
        cor_val = torch.sum(v_pred * v_gt) / (torch.sqrt(torch.sum(v_pred ** 2)) * torch.sqrt(torch.sum(v_gt ** 2)))
        sd_gt = torch.std(target_val)
        sd_pred = torch.std(pred_val)
        numerator_val = 2 * cor_val * sd_gt * sd_pred
        denominator_val = var_target_val + var_pred_val + (mean_target_val - mean_pred_val) ** 2
        ccc_val = numerator_val / denominator_val

        mean_target_aro = torch.mean(target_aro, 0)
        mean_pred_aro = torch.mean(pred_aro, 0)
        var_target_aro = torch.var(target_aro, 0)
        var_pred_aro = torch.var(pred_aro, 0)
        a_pred = pred_aro - mean_pred_aro
        a_gt = target_aro - mean_target_aro
        cor_aro = torch.sum(a_pred * a_gt) / (torch.sqrt(torch.sum(a_pred ** 2)) * torch.sqrt(torch.sum(a_gt ** 2)))
        sd_gt = torch.std(target_aro)
        sd_pred = torch.std(pred_aro)
        numerator_aro = 2 * cor_aro * sd_gt * sd_pred
        denominator_aro = var_target_aro + var_pred_aro + (mean_target_aro - mean_pred_aro) ** 2
        ccc_aro = numerator_aro / denominator_aro

        return 1 - 0.5 * (ccc_val + ccc_aro)


class PartitionLoss(nn.Module):
    def __init__(self, ):
        super(PartitionLoss, self).__init__()

    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            loss = torch.log(1 + num_head / var)
        else:
            loss = 0

        return loss


class AgeLoss(nn.Module):
    def __init__(self, ):
        super(AgeLoss, self).__init__()

    def forward(self, pred_age_dist, pred_age, target_age_dist, target_age, model, lambda_reg=1e-5):
        log_age_dist = F.log_softmax(pred_age_dist, dim=1)
        l1_loss = F.l1_loss(pred_age, target_age.unsqueeze(1))  # MAE

        kl_reg = 0
        for param in model.fc31.parameters():
            kl_reg += torch.sum(torch.abs(param))

        kl_loss = F.kl_div(log_age_dist, target_age_dist) + kl_reg * lambda_reg

        return kl_loss, l1_loss
