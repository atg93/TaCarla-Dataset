import torch
import torch.nn.functional as F

def unproject_image_to_mem(feats_from_enc, pixel_from_car, cam_from_car, sideDim, upDim, frontDim, xyz_car=None):

    B, C, H, W = torch.tensor(list(feats_from_enc.shape))

    xyz_cam = apply_4x4(cam_from_car, xyz_car)
    z = xyz_cam[:, :, 2]

    xyz_pixel = apply_4x4(pixel_from_car, xyz_car)
    normalizer = torch.unsqueeze(xyz_pixel[:, :, 2], 2)
    EPS = 1e-6
    xy_pixel = xyz_pixel[:, :, :2] / torch.clamp(normalizer, min=EPS)
    # this is B x N x 2
    # this is the (floating point) pixel coordinate of each voxel
    x, y = xy_pixel[:, :, 0], xy_pixel[:, :, 1]

    x_valid = (x > -0.5).bool() & (x < W.type(torch.float) - 0.5).bool()
    y_valid = (y > -0.5).bool() & (y < H.type(torch.float) - 0.5).bool()
    z_valid = (z > 0.0).bool()

    valid_mem = x_valid & y_valid & z_valid
    valid_mem = valid_mem.reshape(B, sideDim * upDim * frontDim).unsqueeze(2).bool()
    y_map = torch.clamp(y, min=0.0, max=H - 1).long()
    x_map = torch.clamp(x, min=0.0, max=W - 1).long()
    bevSize = sideDim * upDim * frontDim
    batch_idx = torch.arange(B).unsqueeze(1).repeat(1, bevSize)
    values = feats_from_enc[batch_idx, :, y_map, x_map]
    values = values * valid_mem
    values = torch.reshape(values, [B, upDim, sideDim, frontDim, C])#.permute(0, 4, 2, 1, 3)#.contiguous()
    return values


def get_feat_occupancy(xyz_bev, rad_feat, sideDim, upDim, frontDim):
    # xyz_bev is B x N x 3 and in mem coords
    # rad_feat is B x N x D
    # we want to fill a voxel tensor with 1's at these inds
    B, N, C = list(xyz_bev.shape)
    B2, N2, D2 = list(rad_feat.shape)


    x = xyz_bev[:, :, 0]
    y = xyz_bev[:, :, 1]
    z = xyz_bev[:, :, 2]

    x_valid = (x > -0.5).bool() & (x < float(frontDim - 0.5)).bool()
    y_valid = (y > -0.5).bool() & (y < float(sideDim - 0.5)).bool()
    z_valid = (z > -0.5).bool() & (z < float(upDim - 0.5)).bool()
    nonzero = (y != 0.0).bool()
    distnorm = (torch.norm(xyz_bev, dim=2) > 0.1).bool()
    mask = x_valid & y_valid & z_valid & nonzero & distnorm

    x = x * mask  # B, N
    y = y * mask
    z = z * mask
    # -1 in unsqueeze NOT GOOD FOR ONNX!!!
    rad_feat = rad_feat * mask.unsqueeze(2)  # B, N, D

    x = torch.round(x)
    y = torch.round(y)
    z = torch.round(z)
    x = torch.clamp(x, 0, frontDim - 1).int()
    y = torch.clamp(y, 0, sideDim - 1).int()
    z = torch.clamp(z, 0, upDim - 1).int()

    x = x.view(B * N)
    y = y.view(B * N)
    z = z.view(B * N)
    B, N, L = rad_feat.shape
    rad_feat = torch.reshape(rad_feat, [B * N, L])

    dim3 = frontDim
    dim2 = frontDim * sideDim
    dim1 = frontDim * sideDim * upDim

    base = torch.arange(0, B, dtype=torch.int32, device=xyz_bev.device) * dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B * N)

    vox_inds = base + z * dim2 + y * dim3 + x
    feat_voxels = torch.zeros((B * sideDim * upDim * frontDim, D2), device=xyz_bev.device).float()
    feat_voxels[vox_inds.long()] = rad_feat
    # zero out the singularity
    feat_voxels[base.long()] = 0.0
    feat_voxels = feat_voxels.reshape(B, upDim, sideDim, frontDim, D2).permute(0, 4, 2, 1, 3)
    # B x C x sideDim x upDim x frontDim
    return feat_voxels


def gridcloud3d(B, upDim, sideDim, frontDim, device='cuda'):
    # we want to sample for each location in the grid
    grid_z = torch.linspace(0.0, upDim - 1, upDim, device=device)
    grid_z = torch.reshape(grid_z, [1, upDim, 1, 1])
    grid_z = grid_z.repeat(B, 1, sideDim, frontDim)

    grid_y = torch.linspace(0.0, sideDim - 1, sideDim, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, sideDim, 1])
    grid_y = grid_y.repeat(B, upDim, 1, frontDim)

    grid_x = torch.linspace(0.0, frontDim - 1, frontDim, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, frontDim])
    grid_x = grid_x.repeat(B, upDim, sideDim, 1)

    B, K, L, M = grid_x.shape
    N = K * L * M
    x = torch.reshape(grid_x, [B, N])
    y = torch.reshape(grid_y, [B, N])
    z = torch.reshape(grid_z, [B, N])
    # these are B x N
    xyz_bev = torch.stack([x, y, z], dim=2)
    # this is B x N x 3
    return xyz_bev


def scale_intrinsics(K, sx, sy):
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    x0 = K[:, 0, 2]
    y0 = K[:, 1, 2]
    fx = fx * sx
    fy = fy * sy
    x0 = x0 * sx
    y0 = y0 * sy
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=torch.float32, device=fx.device)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = x0
    K[:, 1, 2] = y0
    K[:, 2, 2] = 1.0
    K[:, 3, 3] = 1.0
    return K

def pack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    B_, S = shapelist[:2]
    otherdims = shapelist[2:]
    tensor = torch.reshape(tensor, [B*S]+otherdims)
    return tensor


def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    otherdims = shapelist[1:]
    S = torch.div(BS, B, rounding_mode='floor')  # onnx conversion warns about this -> (BS//B)
    tensor = torch.reshape(tensor, [B, S]+otherdims)
    return tensor


def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:,:,:3]
    return xyz2
