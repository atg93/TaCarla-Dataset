import torch
import torch.nn as nn
import numpy as np
from tairvision.models.bev.cprm.utils.geometry import unproject_image_to_mem, get_feat_occupancy, gridcloud3d, \
                                                        scale_intrinsics, pack_seqdim, unpack_seqdim, apply_4x4
from tairvision.models.bev.lss.utils.geometry import map_pcloud_to_bev

class ReverseMapping(torch.nn.Module):
    """ReverseMapping adapted from Simple BEV.

        Args:
            cfg: Config :)
            bev_dimension (Torch.tensor): Obtained as a tensor from calculate_birds_eye_view_parameters
    """
    def __init__(self,
                 cfg,
                 bev_dimension):
        super().__init__()
        enc_out_ch = cfg.MODEL.ENCODER.OUT_CHANNELS
        H_W = cfg.IMAGE.FINAL_DIM
        self.use_radar = cfg.USE_RADAR
        # TODO: DEAL WITH 8 HERE as upDim==8
        if self.use_radar:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(enc_out_ch * 8 + 16 * 8, enc_out_ch, kernel_size=3, padding=1, stride=1,
                          bias=False),
                nn.BatchNorm2d(enc_out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(enc_out_ch * 8, enc_out_ch, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(enc_out_ch),
                nn.ReLU(inplace=True),
            )
        bev_limits = cfg.LIFT.X_BOUND + cfg.LIFT.Y_BOUND + cfg.LIFT.Z_BOUND
        self.FrontMin, self.FrontMax, _, self.SideMin, self.SideMax, _, self.UpMin, self.UpMax, _ = bev_limits
        if torch.is_tensor(bev_dimension):
            bev_dimension = np.asarray(bev_dimension)
        self.bevFront, self.bevSide = bev_dimension[0:2]
        self.H, self.W = H_W
        self.EPS = 1e-05
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.head_channels = cfg.MODEL.ENCODER.OUT_CHANNELS

    def forward(self, feats_3d, intrinsics, extrinsics, view, feats_pcloud=None, inverse_view=None):
        B, S = extrinsics.shape[0:2]
        view = view[:, 0, 0, :, :]
        if inverse_view is not None:
            inverse_view = inverse_view[:, 0, 0, :, :]

        feats_bev_ = self.pull_image_2_bev(feats_3d, intrinsics, extrinsics, view, inverse_view=inverse_view)
        if feats_pcloud is not None:
            feats_bev_ = torch.cat([feats_bev_, feats_pcloud], dim=1)
        else:
            pass
        feats_bev = self.bev_compressor(feats_bev_)
        _, C, H, W = feats_bev.shape
        feats_bev = feats_bev.view(B, S, C, H, W)
        return feats_bev

    def pull_image_2_bev(self, feats_from_enc, intrinsics, extrinsics, view, inverse_view=None):
        """
        Main Function.
        Creates a 3D bev space, reverse morphs it into image plane
        Uses valid coords for each cam to grid sample features to fill voxels
        """
        B, S, N, _, _ = extrinsics.shape
        intrinsics = intrinsics.view(B * S * N, 3, 3)
        extrinsics = extrinsics.view(B * S, N, 4, 4)
        device = extrinsics.device
        B, N, C, H, W = feats_from_enc.shape
        feats_from_enc = feats_from_enc.view(B * N, C, H, W)
        sideDim, upDim, frontDim = self.bevSide, 8, self.bevFront
        origH, origW = self.H, self.W

        __p = lambda x: pack_seqdim(x, B)
        __u = lambda x: unpack_seqdim(x, B)
        xyz_mem = gridcloud3d(B, upDim, sideDim, frontDim, device=device)
        if inverse_view is None:
            xyz_car = apply_4x4(view.inverse(), xyz_mem)
        else:
            xyz_car = apply_4x4(inverse_view, xyz_mem)
        # feats_from_enc ENCODER OUTPUT
        _, C, Hf, Wf = feats_from_enc.shape
        sy = Hf / float(origH)
        sx = Wf / float(origW)
        intr_scaled = scale_intrinsics(intrinsics, sx, sy)

        extrinsics_packed = __p(extrinsics)
        #safe_inverse
        extrinsic_inv = extrinsics_packed.clone()
        r_transpose = extrinsics_packed[:, :3, :3].transpose(1, 2)  # inverse of rotation matrix
        extrinsic_inv[:, :3, :3] = r_transpose
        extrinsic_inv[:, :3, 3:4] = -torch.matmul(r_transpose, extrinsics_packed[:, :3, 3:4])

        xyz_car = torch.repeat_interleave(xyz_car, N, dim=0)
        feat_mems_ = unproject_image_to_mem(
            feats_from_enc,
            torch.matmul(intr_scaled, extrinsic_inv),
            extrinsic_inv, sideDim, upDim, frontDim,
            xyz_car=xyz_car)


        feat_mems = __u(feat_mems_)  # B, SN, upDim, sideDim, frontDim, C
        #self.start.record()
        mask_mems = (torch.abs(feat_mems) > 0).float()
        prod = feat_mems * mask_mems
        numer = torch.sum(prod, dim=1, keepdim=False)
        denom = self.EPS + torch.sum(mask_mems, dim=1, keepdim=False)
        feat_mem = numer / denom
        feat_bev = feat_mem.permute(0, 4, 1, 2, 3).reshape(B, self.head_channels * upDim, sideDim, frontDim)#.contiguous()
        #self.end.record()
        #torch.cuda.synchronize()
        #model_time = self.start.elapsed_time(self.end)
        #print("Elapsed time during pull (in milliseconds):", model_time)
        return feat_bev

    def pull_radar_2_bev(self, radar_data):
        sideDim, upDim, frontDim = self.bevSide, 8, self.bevFront
        B,S,W,P,N = radar_data.shape
        radar_data = radar_data.view(B*S*W, P, N)
        rad_data = radar_data  # already done in loader .permute(0, 2, 1)  # B, R, 19
        xyz_rad = rad_data[:, :, :3]
        meta_rad = rad_data[:, :, 3:]
        xyz_bev = xyz_rad  # apply_4x4(view, xyz_rad)
        rad_occ_mem0 = get_feat_occupancy(xyz_bev, meta_rad, sideDim, upDim, frontDim)
        rad_bev = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16*upDim, sideDim, frontDim)  # squish the vertical dim
        return rad_bev

