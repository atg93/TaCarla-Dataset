import torch
from torch.nn import functional as F
from numpy import pi, sqrt, exp
import numpy as np

from tairvision.models.bev.cpvp.blocks.bev_pool_wrapper import bev_pool_v2
from tairvision.models.bev.cprm.utils.geometry import scale_intrinsics, apply_4x4



def voxel_pooling_v2(feats, depth, coor, view, bev_dimension):
    """Data preparation for voxel pooling.
    Args:
        coor (torch.tensor): Coordinate of points in the lidar space in
            shape (B, N, D, H, W, 3).
    Returns:
        tuple[torch.tensor]: Rank of the voxel that a point is belong to
            in shape (N_Points); Reserved index of points in the depth
            space in shape (N_Points). Reserved index of points in the
            feature space in shape (N_Points).
    """
    B, N, D, H, W, _ = coor.shape
    num_points = B * N * D * H * W
    one_batch_points = N * D * H * W
    # record the index of selected points for acceleration purpose
    ranks_depth = torch.range(0, num_points - 1, dtype=torch.int, device=coor.device)
    ranks_feat = torch.range(0, num_points // D - 1, dtype=torch.int, device=coor.device)
    ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
    ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
    # convert coordinate into the voxel space
    coor = coor.view(B, one_batch_points, 3)
    coor = torch.cat([coor, torch.ones_like(coor)[:, :, 0:1]], dim=-1)
    coor = torch.bmm(view.view(B, 1, 4, 4)[:, 0], coor.transpose(2, 1)).transpose(2, 1).long()[:, :, 0:3].contiguous()
    coor = coor.view(num_points, 3)
    batch_idx = torch.range(0, B - 1).reshape(B, 1).expand(B, one_batch_points).reshape(num_points, 1).to(coor)
    coor = torch.cat((coor, batch_idx), 1)

    # filter out points that are outside box
    kept = (coor[:, 0] >= 0) & (coor[:, 0] < bev_dimension[0]) & \
           (coor[:, 1] >= 0) & (coor[:, 1] < bev_dimension[1]) & \
           (coor[:, 2] >= 0) & (coor[:, 2] < bev_dimension[2])
    if len(kept) == 0:
        return None, None, None, None, None
    coor, ranks_depth, ranks_feat = coor[kept], ranks_depth[kept], ranks_feat[kept]
    # get tensors from the same voxel next to each other
    ranks_bev = coor[:, 3] * (bev_dimension[2] * bev_dimension[1] * bev_dimension[0])
    ranks_bev += coor[:, 2] * (bev_dimension[1] * bev_dimension[0])
    ranks_bev += coor[:, 1] * bev_dimension[0] + coor[:, 0]
    order = ranks_bev.argsort()
    ranks_bev, ranks_depth, ranks_feat = ranks_bev[order], ranks_depth[order], ranks_feat[order]

    kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].int()
    if len(interval_starts) == 0:
        return None, None, None, None, None
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
    ranks_bev = ranks_bev.int().contiguous()
    ranks_depth = ranks_depth.int().contiguous()
    ranks_feat = ranks_feat.int().contiguous()
    interval_starts = interval_starts.int().contiguous()
    interval_lengths = interval_lengths.int().contiguous()

    feats = feats.permute(0, 1, 3, 4, 2)
    depth = depth.view(B, N, D, H, W)
    bev_feat_shape = (depth.shape[0], int(bev_dimension[2]),
                      int(bev_dimension[1]), int(bev_dimension[0]),
                      feats.shape[4])  # (B, Z, Y, X, C)
    bev_feat = bev_pool_v2(depth, feats, ranks_depth,
                           ranks_feat, ranks_bev,
                           bev_feat_shape, interval_starts,
                           interval_lengths)
    bev_feat = bev_feat.view(B, bev_feat.shape[1] * bev_dimension[2], bev_dimension[1], bev_dimension[0])
    return bev_feat


class LidarTargets(torch.nn.Module):
    def __init__(self, cfg, dch):
        super().__init__()
        self.cfg = cfg
        self.dch = dch
        self.ggmx = self.get_ggmx(cfg, dch)
        self.step = 5
        self.pad = 2


    def forward(self, pcloud_list, extrinsics, intrinsics, view, bev_size, use_gaussian=True):
        return self.get_lidar_targets(pcloud_list, extrinsics, intrinsics, view, bev_size, use_gaussian=use_gaussian)


    def get_ggmx(self, cfg, dch):
        B = cfg.BATCHSIZE
        N = 6  # 6 CAMS
        OrigH, OrigW = cfg.IMAGE.FINAL_DIM
        downsample = cfg.MODEL.ENCODER.DOWNSAMPLE
        sx = 1 / downsample
        sy = 1 / downsample
        W = int(OrigW * sx)
        H = int(OrigH * sy)
        ggmx = torch.zeros([B * N * H * W, dch, dch * 2 - 1])
        gg = torch.tensor(gauss(n=dch, sigma=1.5), dtype=torch.float32)
        gg = gg.unsqueeze(0).repeat(B * N * H * W, 1)
        for i in range(dch):
            if i == 0 or i == 59:
                pass
            else:
                ggmx[:, i, i:dch + i] = gg
        return ggmx[:, :, (dch - 1) // 2:((dch - 1) // 2 + dch)]

    def get_lidar_targets(self, pcloud_list, extrinsics, intrinsics, view, bev_size, use_gaussian=True):
        cfg = self.cfg
        dch = self.dch
        ggmx = self.ggmx
        OrigH, OrigW = cfg.IMAGE.FINAL_DIM
        downsample = cfg.MODEL.ENCODER.DOWNSAMPLE
        sx = 1 / downsample
        sy = 1 / downsample
        W = int(OrigW * sx)
        H = int(OrigH * sy)
        bev_side = bev_size[0]
        lidar_coords = pcloud_list[1][:, 0, 0, :, :3]
        illuminance_image = pcloud_list[1][:, 0, 0, :, 3]
        illuminance_image = torch.repeat_interleave(illuminance_image, 6, dim=0)
        view2 = view[:, 0, 0, :, :]
        lidar_coords_view = apply_4x4(view2.inverse(), lidar_coords)

        B, S, N, _, _ = extrinsics.shape
        BSN = B * S * N
        intrinsics = intrinsics.view(BSN, 3, 3)
        extrinsics_packed = extrinsics.view(BSN, 4, 4)
        xyz_car = lidar_coords_view
        xyz_car = torch.repeat_interleave(xyz_car, 6, dim=0)
        intr_scaled = scale_intrinsics(intrinsics, sx, sy)
        extrinsic_inv = extrinsics_packed.clone()
        r_transpose = extrinsics_packed[:, :3, :3].transpose(1, 2)  # inverse of rotation matrix
        extrinsic_inv[:, :3, :3] = r_transpose
        extrinsic_inv[:, :3, 3:4] = -torch.matmul(r_transpose, extrinsics_packed[:, :3, 3:4])
        pixel_from_car = torch.matmul(intr_scaled, extrinsic_inv)
        cam_from_car = extrinsic_inv
        xyz_cam = apply_4x4(cam_from_car, xyz_car)
        z = xyz_cam[:, :, 2]
        xyz_pixel = apply_4x4(pixel_from_car, xyz_car)
        normalizer = torch.unsqueeze(xyz_pixel[:, :, 2], 2)
        EPS = 1e-6
        xy_pixel = xyz_pixel[:, :, :2] / torch.clamp(normalizer, min=EPS)
        x, y = xy_pixel[:, :, 0], xy_pixel[:, :, 1]

        x_valid = (x > -0.5).bool() & (x < W - 0.5).bool()
        y_valid = (y > -0.5).bool() & (y < H - 0.5).bool()
        z_valid = (z > 0.0).bool()

        valid_mem = x_valid & y_valid & z_valid
        valid_mem = valid_mem.reshape(BSN, 70000).unsqueeze(2).bool()
        normalizer = torch.clamp(normalizer, min=0.0, max=bev_side / 2 - 1)
        depth_lid = (normalizer * valid_mem).squeeze(2)
        valid_ilu = (illuminance_image.unsqueeze(2) * valid_mem).squeeze(2)
        valid_ilu = torch.clamp(valid_ilu, min=0.0, max=255.0)  # probably dont need this
        y_map = torch.clamp(y, min=0.0, max=H - 1).long()
        x_map = torch.clamp(x, min=0.0, max=W - 1).long()


        lidar_gt_depth = torch.zeros((BSN, H, W)).to(extrinsics)
        lidar_gt_ilu = torch.zeros((BSN, H, W)).to(extrinsics)
        btch = torch.arange(BSN).unsqueeze(1).repeat(1, 70000)
        lidar_gt_depth[btch, y_map, x_map] = depth_lid
        lidar_gt_ilu[btch, y_map, x_map] = valid_ilu

        # Normalization
        lidar_gt_ilu = (lidar_gt_ilu + 1).log()
        lidar_gt_ilu = lidar_gt_ilu / np.log(256)  # 255 seems to be the maximum value of lidar illuminance, +1 = 256
        lidar_gt_depth = lidar_gt_depth / (bev_side / 2)

        lidar_gt = torch.cat([lidar_gt_depth.unsqueeze(1), lidar_gt_ilu.unsqueeze(1)], dim=1)

        """
        # This part comes from EA_BEV Transformer which densifies sparse lidar measurements, it is not useful
        stepp = self.step
        padd = self.pad
        lidarr = lidar_gt.view(B*S, N, H, W)
        depth_tmp = F.pad(lidarr, [padd, padd, padd, padd], mode='constant', value=0)
        patches = depth_tmp.unfold(dimension=2, size=stepp, step=1)
        patches = patches.unfold(dimension=3, size=stepp, step=1)
        lidar_gt, _ = patches.reshape(B, S, N, H, W, -1).max(dim=-1)
        """

        """
        # This part is preparing lidar targets that contain a channel with 59 width a gaussian that is centered
        # around the measurement.
        lidar_gt = lidar_gt / (bev_side / 2) * dch
        lidar_gt = lidar_gt.floor().long()
        lidar_gt = lidar_gt.view(BSN * H * W)
        lidar_gt_mx = torch.zeros([BSN*H*W, dch])

        lidar_gt_mx[torch.arange(0, BSN*H*W), lidar_gt] = 1.0
        if use_gaussian:
            lidar_targets = torch.bmm(lidar_gt_mx.unsqueeze(1).float(), ggmx.float()).squeeze(1)
        else:
            lidar_gt_mx[:, 0] = 0.0  # every non-existent reading goes here, it's problematic.
            lidar_targets = lidar_gt_mx
        #lidar_targets = lidar_targets.view(BSN, H, W, dch)
        lidar_targets = lidar_targets.view(B, N, H, W, dch).permute(0, 1, 4, 2, 3).contiguous()
        
        return lidar_targets
        """

        return lidar_gt


def gauss(n=59, sigma=1):
    r = range(-int(n / 2), int(n / 2) + 1)
    h = [1 / (sigma * sqrt(2 * pi)) * exp(-float(x) ** 2 / (2 * sigma ** 2)) for x in r]
    h = np.asarray(h)
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h
