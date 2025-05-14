import torch
from   torch import nn
from torch.nn import functional as F

class AdaptiveModule(nn.Module):
    """
        This is the adaptive module specified in the multi level domain adaptation for lane detection paper.
        This module calculates spatial and channel-wise attention and increase the precision of the predictions.
    """
    def __init__(self, in_channels , H, W,device):
        super(AdaptiveModule, self).__init__()
        self.f_bn = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels //2, kernel_size=(1, 3), padding=(0, 1)),
                                  nn.BatchNorm2d(num_features=in_channels //2),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=in_channels //2, out_channels=in_channels//2, kernel_size=(3, 1)),
                                  nn.BatchNorm2d(num_features=in_channels//2),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=in_channels //2, out_channels=in_channels //4, kernel_size=(1, 1), padding=(1, 0)),
                                  nn.BatchNorm2d(num_features=in_channels //4),
                                  nn.ReLU())


        self.f_spa_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels //4, out_channels=in_channels //4, kernel_size=(1, 7), padding=(0, 3)),
                                     nn.Conv2d(in_channels=in_channels //4, out_channels=in_channels //4, kernel_size=(7, 1), padding=(3, 0)))

        self.f_spa_2 = nn.Sequential(nn.Conv2d(in_channels=in_channels //4, out_channels=in_channels //4, kernel_size=(1, 7), padding=(0, 3)),
                                     nn.Conv2d(in_channels=in_channels //4, out_channels=in_channels //4, kernel_size=(7, 1), padding=(3, 0)))

        self.softmax = nn.Softmax(dim=1)
        self.avgp    = nn.AvgPool2d(H, W)
        self.maxp    = nn.MaxPool2d(H, W)

        self.mlp     = nn.Sequential(nn.Linear(in_features=in_channels // 2,out_features=in_channels//4),
                                     nn.ReLU(),
                                     nn.Linear(in_features=in_channels //4,out_features=in_channels//32),
                                     nn.ReLU(),
                                     nn.Linear(in_features=in_channels //32,out_features=in_channels//4),
                                     nn.ReLU())

        self.fc      = nn.Sequential(nn.Linear(in_features=(in_channels//4)*(in_channels//4), out_features=4),
                                     nn.Sigmoid())
        self.device  = device


    def forward(self, input):
        batch_size , _ , _, _     = input.size()

        f_bn_out    = self.f_bn.forward(input)

        f_spa_1_out = self.f_spa_1.forward(f_bn_out)
        f_spa_2_out = self.f_spa_2.forward(f_bn_out)

        f_spa_out   = torch.cat((f_spa_1_out,f_spa_2_out),dim=1)

        del f_spa_1_out
        del f_spa_2_out

        f_spa_out   = self.softmax(f_spa_out)
        f_spa_out   = torch.argmax(f_spa_out, dim=1, keepdim=True)
        f_spa_out   = torch.mul(f_bn_out,f_spa_out)

        f_cha_out_1 = self.avgp(f_bn_out)
        f_cha_out_2 = self.maxp(f_bn_out)

        f_cha_out   = torch.cat((f_cha_out_1, f_cha_out_2), dim=1)

        del f_cha_out_1
        del f_cha_out_2

        f_cha_out   = self.mlp(f_cha_out[:,:,0,0])

        f_cha_out   = torch.unsqueeze(f_cha_out,2)
        f_cha_out   = torch.unsqueeze(f_cha_out,3)

        f_cha_out   = torch.mul(f_cha_out,f_bn_out)

        del f_bn_out

        f_inter     = torch.einsum("bchw,bkhw->bck",f_spa_out,f_cha_out)

        del f_cha_out
        del f_spa_out

        f_inter     = torch.flatten(f_inter, start_dim=1)
        f_inter     = self.fc(f_inter)

        return f_inter

class EmbeddingModule(nn.Module):
    """
        This is the embedding module specified in the multi level domain adaptation for lane detection paper.
        This module is for triplet loss. The implementation of triplet loss is in the uda branch.
        This module takes the negative and positive examples and creates and positional embedding.
        For further explanation please visit :https://gitlab.togg.com.tr/tam/ai/perception/tairvision/-/blob/uda/tairvision/models/segmentation/deeplabv3.py#L177
    """
    def __init__(self,in_channels):
        super(EmbeddingModule, self).__init__()
        self.fully_connected = nn.Linear(in_features=in_channels,out_features=in_channels//32)
    def forward(self,input):
        input = self.fully_connected(input)
        return F.normalize(input, p=2.0)


class SelfTraining(nn.Module):
    """
        This is the selt training module specified in the multi level domain adaptation for lane detection paper.
        It takes the predictions of the target dataset and creates pseudo-labels and
            calculates cross entropy with the predictions and pseudo labels.
    """
    def __init__(self,foreground_thr,background_thr):
        super(SelfTraining, self).__init__()
        self.foreground_thr     = foreground_thr
        self.background_thr     = background_thr

    def forward(self,target_out):
        if target_out!=None:
            ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
            pseudo_logits = target_out.detach()
            softmax       = torch.nn.Softmax(dim=1)
            normalized_pseudo_logits = softmax(pseudo_logits)
            pseudo_max    = torch.max(normalized_pseudo_logits, dim=1, keepdim=False)
            pseudo_labels = pseudo_max.indices
            pseudo_probs  = pseudo_max.values

            # TODO, Self training requires additional guidance such as thresholding
            # Multi level domain adaptation for lane detection paper
            # Daformer sample quality measurement
            """non_background_labels_ind = torch.where(pseudo_labels > 0, 1, 0)
            non_background_probs  = torch.mul(non_background_labels_ind, pseudo_probs)
            non_background_labels = torch.where(non_background_probs >= self.foreground_thr, pseudo_labels + 1, 0)

            background_labels_ind = torch.where(pseudo_labels == 0, 1, 0)
            background_probs = torch.mul(background_labels_ind, pseudo_probs)
            background_labels = torch.where(background_probs >= self.background_thr, 1, 0)

            final_labels = non_background_labels + background_labels
            final_labels = final_labels - 1
            final_labels = torch.where(final_labels == -1, 255, final_labels)


            loss_st = ce_loss(target_out, final_labels)
            """
            loss_st = ce_loss(target_out, pseudo_labels)
        else:
            loss_st = 0
        return loss_st

