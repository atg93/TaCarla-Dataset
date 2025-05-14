import torch


def get_targets_static(batch, receptive_field=1):

    # if any of the channels is non-zero then the segmentation label is 1, otherwise 0
    lanes = batch['lanes'].sum(2, keepdims=True)
    lanes = (lanes > 0).to(torch.int64)

    lines = batch['lines'].sum(2, keepdims=True)
    lines = (lines > 0).to(torch.int64)

    # get only the present frame
    t = receptive_field - 1
    lanes = lanes[:, t:t+1]
    lines = lines[:, t:t+1]

    targets_static = {'lanes': lanes,
                      'lines': lines
                      }

    return targets_static
