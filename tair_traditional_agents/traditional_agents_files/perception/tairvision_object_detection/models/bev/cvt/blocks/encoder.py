import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    # indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)  # 2 h w
    indices = torch.stack(torch.meshgrid(ys, xs), 0)
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)  # 3 h w
    indices = indices[None]  # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [0., -sw, w / 2.],
        [-sh, 0., h * offset + h / 2.],
        [0., 0., 1.]
    ]


class BEVEmbedding(nn.Module):
    def __init__(
            self,
            dim: int,
            sigma: int,
            bev_height: int,
            bev_width: int,
            h_meters: int,
            w_meters: int,
            offset: int,
            decoder_blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        self.offset = offset
        self.h_meters = h_meters
        self.w_meters = w_meters
        self.bev_height = bev_height
        self.bev_width = bev_width

        self.h = bev_height // (2 ** len(decoder_blocks))
        self.w = bev_width // (2 ** len(decoder_blocks))

        self.learned_features = nn.Parameter(sigma * torch.randn(dim, self.h, self.w))  # d h w

        # map from bev coordinates to ego frame
        grid = generate_grid(self.h, self.w).squeeze(0)
        grid[0] = self.bev_width * grid[0]
        grid[1] = self.bev_height * grid[1]

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)  # 3 h w

    def get_prior(self):
        return self.learned_features

    def get_grid(self, view: torch.FloatTensor = None):
        # bev coordinates
        if view is None:
            return self.grid
        view = view[0].contiguous().squeeze()  # 4 4
        view = view[[0, 1, 3]]  # delete row, 3 4
        view = view[:, [0, 1, 3]]  # delete col 3 3
        V_inv = view.inverse()
        d, h, w = self.grid.shape
        grid = V_inv @ rearrange(self.grid, 'd h w -> d (h w)')
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)
        return grid


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # Project with multiple heads
        q = self.to_q(q)  # b (n H W) (heads dim_head)
        k = self.to_k(k)  # b (n h w) (heads dim_head)
        v = self.to_v(v)  # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

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


class CrossViewAttention(nn.Module):
    def __init__(
            self,
            feat_height: int,
            feat_width: int,
            feat_dim: int,
            dim: int,
            image_height: int,
            image_width: int,
            qkv_bias: bool,
            heads: int = 4,
            dim_head: int = 32,
            no_image_features: bool = False,
            skip: bool = True,
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
            self,
            x: torch.FloatTensor,
            bev: BEVEmbedding,
            feature: torch.FloatTensor,
            I_inv: torch.FloatTensor,
            E_inv: torch.FloatTensor,
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

        pixel = self.image_plane  # b n 3 h w
        _, _, _, h, w = pixel.shape
        # translation embedding, tau_{k}
        c = E_inv[..., -1:]  # b n 4 1
        # if view is not None:
        #     c = view @ c  # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]  # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)  # (b n) d 1 1

        # R_{k}^{-1} X K_{k}^{-1} X x_{i}^{(I)}
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')  # 1 1 3 (h w)
        cam = I_inv @ pixel_flat  # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)  # b n 4 (h w)
        d = E_inv @ cam  # b n 4 (h w)
        # if view is not None:
        #     d = view @ d  # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)  # (b n) 4 h w
        d_embed = self.img_embed(d_flat)  # (b n) d h w

        # Normalization for attention
        img_embed = d_embed - c_embed  # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)  # (b n) d h w

        # camera location embeddings (tau_{k}) are subtracted from map-view positional encodings (c^{n}) to make map-view queries (c^{n} - tau_{k})
        world = bev.get_grid(view)[:2]  # 2 H W
        # world = bev.grid[:2]  # 2 H W
        w_embed = self.bev_embed(world[None])  # 1 d H W
        bev_embed = w_embed - c_embed  # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)  # (b n) d H W
        # bev_embed = torch.cat((bev_embed, pcloud_feat), dim=0)
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)  # b n d H W

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')  # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)  # (b n) d h w
        else:
            key_flat = img_embed  # (b n) d h w

        val_flat = self.feature_linear(feature_flat)  # (b n) d h w

        # Expand + refine the BEV embedding
        query = query_pos + x[:, None]  # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)  # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)  # b n d h w

        return self.cross_attend(query, key, val, skip=x if self.skip else None)
