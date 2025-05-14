import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from tairvision.models.bev.cvt.blocks.encoder import BEVEmbedding, CrossAttention, CrossViewAttention


class KernelAttention(CrossAttention):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__(dim, heads, dim_head, qkv_bias, norm)

    def forward(self, q, k, v, skip=None, mask=None):
        """
        q: (b n d H W)
        k: (b n k g d)
        v: (b n k g d)
        mask: (b n k 1)
        """
        _, _, _, H, W = q.shape
        num_points = k.shape[-2]
        # Move feature dim to last for multi-head proj
        # (b, n, k, d)
        q = rearrange(q, 'b n d H W -> b n (H W) d')

        # Project with multiple heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # Group the head dim with batch dim
        q = rearrange(q, 'b n q (m d) -> (b m) n q 1 d',
                      m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b n q g (m d) -> (b m) n q g d',
                      m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b n q g (m d) -> (b m) q (n g) d',
                      m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * \
              torch.einsum('b n Q c d, b n Q K d -> b n Q c K', q, k)
        dot = rearrange(dot, 'b n Q c K -> b Q (n c K)')
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1, num_points)
            mask = rearrange(mask, 'b h n Q g -> (b h) Q (n g)')
            dot[~mask] = -10 ** 9
        att = dot.to(q).softmax(dim=-1)
        a = torch.einsum('b Q K, b Q K d -> b Q d', att, v)

        a = rearrange(a, '(b m) Q d -> b Q (m d)',
                      m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z


@torch.no_grad()
def bev2image_sampling(points, I, E, height, width):
    """
    bev points to images: each bev point -> image points
    Args:
        points: (k, 3), (x,y,z)
        I: (b, n, 3, 3)
        E: (b, n, 4, 4)
    Return:
        sampled points: (k, 6, 2)
    """
    # (k, 3) -> (k, 4)
    k = points.shape[0]
    b, n = I.shape[:2]
    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    intrin_mat = F.pad(I, (0, 1, 0, 1), value=0)
    intrin_mat[..., -1, -1] = 1.0
    # (k, 3) -> (b, n, k, 4, 1)
    points = points.view(1, 1, k, 4).repeat(b, n, 1, 1).unsqueeze(-1)
    # (b, n, 4, 4) * (k, 4)^T
    point2image = (intrin_mat @ E).view(b, n, 1, 4, 4).repeat(1, 1, k, 1, 1)
    sample_points = (point2image @ points).squeeze(-1)  # (b, n, k, 4)

    # filter points
    eps = 1e-5
    # mask: (b, n, k, 4)
    mask = (sample_points[..., 2:3] > eps)
    sample_points = sample_points[..., 0:2] / \
                    sample_points[..., 2:3].clamp(min=eps)

    sample_points[..., 0] /= width
    sample_points[..., 1] /= height

    # sample points in the image
    mask = (mask & (sample_points[..., 0:1] > 0.0)
            & (sample_points[..., 0:1] < 1.0)
            & (sample_points[..., 1:2] > 0.0)
            & (sample_points[..., 1:2] < 1.0))
    mask = torch.nan_to_num(mask)

    return sample_points, mask


class IndexBEVProjector(nn.Module):
    """GridBEVProjector, based on Grid Sampling (nearest)
    """

    def __init__(self, image_size, grid_size=(3, 3), height=-1.0):
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        grid_h, grid_w = grid_size
        y = torch.arange(grid_h) - grid_h // 2
        x = torch.arange(grid_w) - grid_w // 2
        # offsets = torch.stack(torch.meshgrid(x, y, indexing="xy")).permute(1, 2, 0)   #pytorch>=1.10
        offsets = torch.stack(torch.meshgrid(y, x)).permute(1, 2, 0)
        self.register_buffer("grid_offsets", offsets, persistent=False)
        self.bev_height = height

    def forward(self, bev_grids, images, I, E):
        """
        bev_grids: (3, H, W)
        images: (b, n, c, h, w), features
        I: (b, n, 3, 3)
        E: (b, n, 4, 4)
        im_size: (height, width)
        """
        b, n = I.shape[:2]
        # unfold feature maps
        bn, c, h, w = images.shape

        # bev_grids -> image_coords
        # (3, H, W) -> (H*W, 3), k=H*W
        bev_points = bev_grids.reshape(3, -1).transpose(0, 1)
        bev_points[:, -1] = self.bev_height

        # (b, n, k, 2), (b, n, k, 1)
        sample_points, sample_mask = bev2image_sampling(
            bev_points, I, E, self.image_size[0], self.image_size[1])
        num_grid_points = self.grid_size[0] * self.grid_size[1]
        sample_points[..., 0] *= w
        sample_points[..., 1] *= h
        sample_points = sample_points.round().long()
        grid_offsets = self.grid_offsets.view(1, 1, 1, num_grid_points, 2)

        # [b, n, k, 9, 2]
        sample_points = sample_points.unsqueeze(-2) + grid_offsets
        # restrict sample_points between 0~H-1
        sample_points[..., 0].clamp_(min=0, max=w - 1)
        sample_points[..., 1].clamp_(min=0, max=h - 1)
        # [b, n, k, 9]
        k = sample_points.shape[2]
        sample_points_inds = sample_points[..., 0] + sample_points[..., 1] * w
        # [b*n, k*9]
        sample_points_inds = sample_points_inds.view(
            b * n, k * num_grid_points)
        # [b*n*h*w, c]
        images = rearrange(images, "b c h w -> (b h w) c")
        ind_offsets = (torch.arange(b * n, device=images.device)
                       * (h * w)).view(b * n, 1)
        # b*n*k*9, 1
        sample_points_inds = (sample_points_inds + ind_offsets).view(-1)
        # [b*n*k*9, c]
        sample_feats = images[sample_points_inds].reshape(
            b, n, k, num_grid_points, c)
        # embed()
        return sample_feats, sample_mask.detach()


class UnfoldBEVProjector(nn.Module):

    def __init__(self, image_size, grid_size=(3, 3), height=-1.0):
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        self.pad_size = (grid_size[0] // 2, grid_size[1] // 2)
        self.unfold = nn.Unfold(
            kernel_size=self.grid_size,
            padding=self.pad_size
        )
        self.bev_height = height

    def forward(self, bev_grids, images, I, E):
        """
        bev_grids: (3, H, W)
        images: (b*n, c, h, w), features
        I: (b, n, 3, 3)
        E: (b, n, 4, 4)
        im_size: (height, width)
        """
        # bev_grids -> image_coords
        # (3, H, W) -> (H*W, 3), k=H*W
        bev_points = bev_grids.reshape(
            3, -1).transpose(0, 1).requires_grad_(False)
        # z: bev height
        bev_points[:, -1] = self.bev_height

        # (b, n, k, 2), (b, n, k, 1)
        sample_points, sample_mask = bev2image_sampling(
            bev_points, I, E, self.image_size[0], self.image_size[1])
        sample_points = sample_points * 2.0 - 1.0

        # embed()

        b, n = I.shape[:2]
        # unfold feature maps
        bn, c, h, w = images.shape
        # (b*n, c*p, h, w)
        unfold_images = self.unfold(images).view(bn, -1, h, w)
        # (b, n, k, 2) -> (b * n, k, 1, 2)
        k = sample_points.shape[2]
        sample_points = sample_points.reshape(b * n, k, 1, 2)

        # grid-sample -> (b*n, c, k, 1)
        # reshape -> (b, n, c', num, k)
        num_grid_points = self.grid_size[0] * self.grid_size[1]
        sample_feats = F.grid_sample(
            unfold_images, sample_points, mode='nearest').reshape(b, n, c, num_grid_points, k)
        # permute -> (b, n, k, grid_points, C)
        sample_feats = sample_feats.permute(0, 1, 4, 3, 2)
        return sample_feats, sample_mask.detach()


class GeometryKernelAttention(CrossViewAttention):
    def __init__(
            self,
            feat_height: int,
            feat_width: int,
            feat_dim: int,
            dim: int,
            bev_z: int,
            kernel_h: int,
            kernel_w: int,
            image_height: int,
            image_width: int,
            qkv_bias: bool,
            heads: int = 4,
            dim_head: int = 32,
            no_image_features: bool = False,
            skip: bool = True,
            sampling_type: str = "index",
            use_kernel_conv: bool = True,
            kernel_conv_h: int = 1,
            kernel_conv_w: int = 7
    ):
        super().__init__(feat_height, feat_width, feat_dim, dim, image_height, image_width, qkv_bias, heads, dim_head, no_image_features, skip)

        if sampling_type == "unfold":
            self.sampling = UnfoldBEVProjector(
                (image_height, image_width), grid_size=(kernel_h, kernel_w), height=bev_z)
        elif sampling_type == "index":
            self.sampling = IndexBEVProjector(
                (image_height, image_width), grid_size=(kernel_h, kernel_w), height=bev_z)
        else:
            raise NotImplementedError()

        self.feature_linear = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, dim, bias=False)
        )

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, dim, bias=False)
            )

        if use_kernel_conv:
            self.conv = nn.Conv2d(
                feat_dim, feat_dim, (kernel_conv_h, kernel_conv_w),
                padding=(kernel_conv_h // 2, kernel_conv_w // 2))
        else:
            self.conv = lambda x: x

        self.img_embed = nn.Linear(4, dim, bias=False)
        self.cross_attn = KernelAttention(dim, heads, dim_head, qkv_bias)

    def forward(
            self,
            x: torch.FloatTensor,
            bev: BEVEmbedding,
            feature: torch.FloatTensor,
            I_inv: torch.FloatTensor,
            E_inv: torch.FloatTensor,
            I_: torch.FloatTensor,
            E_: torch.FloatTensor,
            view: torch.FloatTensor = None,
            pcloud_feat: torch.FloatTensor = None,
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        b, n, _, _, _ = feature.shape

        # b n 3 h w
        pixel = self.image_plane
        _, _, _, h, w = pixel.shape

        # b n 4 1
        # translation embedding, tau_{k}
        c = E_inv[..., -1:]  # b n 4 1
        # if view is not None:
        #     c = view @ c  # b n 4 1
        # (b n) 4 1 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]
        # (b n) d 1 1
        c_embed = self.cam_embed(c_flat)

        # 1 1 3 (h w)
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        # b n 3 (h w)
        cam = I_inv @ pixel_flat
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)
        # b n 4 (h w)
        d = E_inv @ cam
        # if view is not None:
        #     d = view @ d  # b n 4 (h w)
        # (b n) 4 h w
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)

        # 2 H W
        # world = bev.grid[:2]
        world = bev.get_grid(view)[:2]
        # 1 d H W
        w_embed = self.bev_embed(world[None])
        # (b n) d H W
        bev_embed = w_embed - c_embed
        # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
        # b n d H W
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)

        # (b n) d h w
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')

        feature_flat = self.conv(feature_flat)
        # project local patches using sampling
        # concat feature and embeddings for sampling
        d_feature = feature_flat.shape[1]
        feature_embed = torch.cat([feature_flat, d_flat], dim=1)
        feature_embed, mask = self.sampling(
            bev.get_grid(view), feature_embed, I_, E_)
        # feature_embed, mask = self.sampling(
        #     bev.grid, feature_embed, I_, E_)
        # b, n, q, num_points, c
        feature_flat = feature_embed[..., :d_feature]
        d_flat = feature_embed[..., d_feature:]

        # (b n) q, num_points, 4
        d_embed = self.img_embed(d_flat)

        # d_embed: b, n, q, num_points, d
        # c_embed: (b, n), d, 1, 1
        img_embed = d_embed - c_embed.view(b, n, 1, 1, d_embed.shape[-1])
        img_embed = img_embed / (img_embed.norm(dim=-1, keepdim=True) + 1e-7)

        # g: num_grid_points
        # b, n, q, g, c
        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)
        else:
            # (b, n) d, h, w
            key_flat = img_embed

        # (b, n) d, h, w
        val_flat = self.feature_linear(feature_flat)

        # Expand + refine the BEV embedding
        # b, n, d, H, W
        query = query_pos + x[:, None]

        return self.cross_attn(query, key_flat, val_flat, mask=mask, skip=x if self.skip else None)
