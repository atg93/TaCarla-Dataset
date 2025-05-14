import torch
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss


class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss. Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        # def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
        eps = 1e-12  # 1e-05
        pos_weights = target.eq(1)
        neg_weights = (1 - target).pow(self.gamma)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(self.alpha) * pos_weights
        neg_loss = -(1 - pred + eps).log() * pred.pow(self.alpha) * neg_weights
        return (self.loss_weight * (pos_loss + neg_loss)).sum() / avg_factor


class L1Loss(nn.Module):
    """L1 loss wrapper."""

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        loss_bbox = self.loss_weight * torch.abs(pred - target)
        loss_bbox = loss_bbox * weight
        loss = loss_bbox.sum() / avg_factor
        return loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss wrapper."""

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        loss_bbox = self.loss_weight * smooth_l1_loss(pred, target, reduction="sum", beta=1.0 / 9.0)
        loss_bbox = loss_bbox * weight
        loss = loss_bbox.sum() / avg_factor
        return loss