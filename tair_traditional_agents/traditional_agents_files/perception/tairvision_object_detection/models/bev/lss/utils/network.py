import torch
import torch.nn as nn
import torchvision


def pack_sequence_dim(x, pack_n=False):
    b, s, n = x.shape[:3]
    if pack_n:
        return x.view(b * s * n, *x.shape[3:])
    else:
        return x.view(b * s, *x.shape[2:])


def unpack_sequence_dim(x, b, s, n=None):
    if n is not None:
        return x.view(b, s, n, *x.shape[1:])
    else:
        return x.view(b, s, *x.shape[1:])


def preprocess_batch(batch, device, unsqueeze=False, filtered_keys=[]):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and key not in filtered_keys:
            batch[key] = value.to(device)
            if unsqueeze:
                batch[key] = batch[key].unsqueeze(0)

def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def set_module_grad(module, requires_grad=False):
    for p in module.parameters():
        p.requires_grad = requires_grad


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = momentum


def remove_past_frames(feats, b, s, n):
    out = {}
    for key in feats.keys():
        feat = feats[key].view(b, s, n, *feats[key].shape[1:])
        feat = feat[:, -1:]
        out[key] = feat.reshape(b * 1 * n, *feat.shape[3:])

    return out


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
