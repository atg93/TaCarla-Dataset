

import torch

def ext_data(extrinsics):
    ext = torch.zeros((1, 6, 4, 4), dtype=torch.float32)

    for idx, im in enumerate(extrinsics):
        ext[0, idx] = extrinsics[im]

    return ext

def intr_data(intrinsics):

    intr = torch.zeros((1, 6, 3, 3), dtype=torch.float32)

    for idx, im in enumerate(intrinsics):
        intr[0, idx] = intrinsics[im]
    return intr

def im_data(data):
    images = torch.zeros((1, 6, 3, 396, 704))

    for idx, im in enumerate(data):
        images[0, idx] = torch.transpose(torch.transpose(torch.tensor(data[im][1][:, :, :3]), 2,0), 1,2)

    return images