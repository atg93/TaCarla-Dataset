import torch
import torch.nn as nn


class VoxelsSumming(torch.autograd.Function):
    """Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/tools.py#L193"""

    @staticmethod
    def forward(ctx, x, geometry, ranks):
        """The features `x` and `geometry` are ranked by voxel positions."""
        # Cumulative sum of all features.
        x = x.cumsum(0)

        # Indicates the change of voxel.
        mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        mask[:-1] = ranks[1:] != ranks[:-1]

        x, geometry = x[mask], geometry[mask]
        # Calculate sum of features within a voxel.
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        ctx.save_for_backward(mask)
        ctx.mark_non_differentiable(geometry)

        return x, geometry

    @staticmethod
    def backward(ctx, grad_x, grad_geometry):
        (mask,) = ctx.saved_tensors
        # Since the operation is summing, we simply need to send gradient
        # to all elements that were part of the summation process.
        indices = torch.cumsum(mask, 0)
        indices[mask] -= 1

        output_grad = grad_x[indices]

        return output_grad, None, None


def create_frustum(image_size, downsample_rate, depth_bound):
    # Create grid in image plane
    h, w = image_size
    downsampled_h, downsampled_w = h // downsample_rate, w // downsample_rate

    # Depth grid
    depth_grid = torch.arange(*depth_bound, dtype=torch.float)
    depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
    n_depth_slices = depth_grid.shape[0]

    # x and y grids
    x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
    x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
    y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
    y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

    # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
    # containing data points in the image: left-right, top-bottom, depth
    frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
    return nn.Parameter(frustum, requires_grad=False)


def get_geometry(frustum, intrinsics, extrinsics):
    """Calculate the (x, y, z) 3D position of the features.
    """
    rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
    B, N, _ = translation.shape
    # Add batch, camera dimension, and a dummy dimension at the end
    points = frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

    # Camera to ego reference frame
    points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
    combined_transformation = rotation.matmul(torch.inverse(intrinsics))
    points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    points += translation.view(B, N, 1, 1, 1, 3)

    # The 3 dimensions in the ego reference frame are: (forward, sides, height)
    return points


def projection_to_birds_eye_view(x, geometry, view, bev_dimension):
    """ Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200"""
    # batch, n_cameras, depth, height, width, channels
    batch, n, d, h, w, c = x.shape
    output = torch.zeros((batch, c, bev_dimension[0], bev_dimension[1]),
                         dtype=torch.float, device=x.device
                         )

    # Number of 3D points
    N = n * d * h * w
    for b in range(batch):
        # flatten x
        x_b = x[b].reshape(N, c)

        # Convert positions to integer indices
        # for fiery convention, note that no x <-> y exchange
        # geometry_b = ((geometry[b] - (bev_start_position - bev_resolution / 2.0)) / bev_resolution)

        # for cvt convention, note that no x <-> y exchange
        # geometry_b = ((-geometry[b] - (bev_start_position - bev_resolution / 2.0)) / bev_resolution)
        # geometry_b = geometry_b.view(N, 3).long()

        # for any convention using view matrix, assuming x <-> y exchange
        geometry_b = geometry[b].view(N, 3)
        geometry_b = torch.cat([geometry_b, torch.ones_like(geometry_b)[:, 0:1]], dim=-1)
        # get the view matrix updated with the transformations rather than the fixed one
        geometry_b = view.view(batch,1,4,4)[b,0] @ geometry_b.T
        geometry_b = geometry_b.T.long()[:, 0:3]

        # Mask out points that are outside the considered spatial extent.
        # TODO, might be wrong, x is to the right after view transormation
        # geometry_b[:, 0] corresponds to bev_dimension[1], 
        # geometry_b[:, 1] corresponds to bev_dimension[0]
        mask = ((geometry_b[:, 0] >= 0) & (geometry_b[:, 0] < bev_dimension[1]) &
                (geometry_b[:, 1] >= 0) & (geometry_b[:, 1] < bev_dimension[0]) &
                (geometry_b[:, 2] >= 0) & (geometry_b[:, 2] < bev_dimension[2])
                )

        x_b = x_b[mask]
        geometry_b = geometry_b[mask]

        # Sort tensors so that those within the same voxel are consecutives.
        # geometry_b[:, 0] corresponds to bev_dimension[1] 
        # 100 x 100 does dot yield error due to bev_dimension[0] = bev_dimension[1]
        ranks = (geometry_b[:, 0] * (bev_dimension[0] * bev_dimension[2]) +
                 geometry_b[:, 1] * (bev_dimension[2]) +
                 geometry_b[:, 2]
                 )

        ranks_indices = ranks.argsort()
        x_b, geometry_b, ranks = x_b[ranks_indices], geometry_b[ranks_indices], ranks[ranks_indices]

        # Project to bird's-eye view by summing voxels.
        x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)

        # original code when no x <-> y exchange
        # bev_feature = torch.zeros((bev_dimension[2], bev_dimension[0], bev_dimension[1], c),
        #                           device=x_b.device)
        # bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b

        # when using view matrix, x <-> y exchange applied automatically
        bev_feature = torch.zeros((bev_dimension[2], bev_dimension[0], bev_dimension[1], c),
                                  device=x_b.device)
        bev_feature[geometry_b[:, 2], geometry_b[:, 1], geometry_b[:, 0]] = x_b

        # Put channel in second position and remove z dimension
        bev_feature = bev_feature.permute((0, 3, 1, 2))
        bev_feature = bev_feature.squeeze(0)

        output[b] = bev_feature

    return output


def create_bev_geometry(bev_dimension):
    # Create grid in ego reference plane
    x_locations = torch.arange(bev_dimension[1])
    y_locations = torch.arange(bev_dimension[0])
    z_locations = torch.arange(bev_dimension[2])

    xs_bev, ys_bev, zs_bev = torch.meshgrid(x_locations, y_locations, z_locations, indexing='xy')
    xs_bev_flatten = xs_bev.reshape(-1)
    ys_bev_flatten = ys_bev.reshape(-1)
    zs_bev_flatten = zs_bev.reshape(-1)

    bev_pixel_locations = torch.stack([xs_bev_flatten,
                                       ys_bev_flatten,
                                       zs_bev_flatten,
                                       torch.ones_like(xs_bev_flatten)], dim=-1)

    return nn.Parameter(bev_pixel_locations.to(torch.float), requires_grad=False)


def get_pixel_locations(bev_geometry, intrinsics, extrinsics, view, image_hw=(224, 480), downsample_rate=1.):

    batch_size = len(intrinsics)
    height, width = image_hw

    # Create a matrix to eliminate the 4th dim
    from_4d_to_3d = torch.tensor([[1., 0., 0., 0.],
                                  [0., 1., 0., 0.],
                                  [0., 0., 1., 0.]
                                  ])
    from_4d_to_3d = from_4d_to_3d.repeat(batch_size, 1, 1).to(intrinsics.device)

    # get the transformation matrix from bev view to cam 2d
    view_to_cam2d = intrinsics @ from_4d_to_3d @ torch.inverse(extrinsics) @ torch.inverse(view)

    # convert each point in the bev geometry to a pixel location
    locations = torch.matmul(view_to_cam2d.unsqueeze(1),
                             bev_geometry.unsqueeze(2)).squeeze(-1)

    # create a mask for invalid pixel locations
    maskz = (locations[..., 2] > 0)

    # normalize the x, y locations wrt depth
    locations = locations / locations[..., 2:3]
    locations = locations.round().long()

    # create a mask for invalid pixel locations
    maskx = ((locations[..., 0] < width) & (locations[..., 0] >= 0))
    masky = ((locations[..., 1] < height) & (locations[..., 1] >= 0))
    mask = (maskx & masky & maskz).unsqueeze(-1)

    # mask invalid locations
    locations = locations * mask

    # get locations on the feats_3d that is downscaled by the encoder
    locations = (locations / downsample_rate).floor().to(torch.int64)

    return locations, mask


def map_pixels_to_bev(x, locations, mask, bev_size=(200, 200, 8), nb_cam=6):

    batch_size, nb_bev_pixels = locations.shape[0:2]
    # get feats_bev from feats_3d and locations, and make zero the invalids
    batch_idx = torch.arange(batch_size).view(-1, 1).repeat(1, nb_bev_pixels)
    feats_bev = x[batch_idx, locations[..., 2], locations[..., 1], locations[..., 0], :]
    feats_bev = feats_bev * mask

    # handle the regions where cameras' fov intersect
    feats_bev = feats_bev.view(batch_size // nb_cam, nb_cam, *bev_size, -1).sum(1)
    mask_sum = mask.view(batch_size // nb_cam, nb_cam, *bev_size).sum(1)
    feats_bev[mask_sum > 0] /= mask_sum[mask_sum > 0].unsqueeze(-1)

    return feats_bev


def map_pcloud_to_bev(pcloud, bev_size=(200, 200, 8)):
    # pcloud is B x N x (3 + C)

    # Note the x - y axes exchange here
    locations = pcloud[..., [1, 0, 2]]
    feats = pcloud[..., 3:]

    batch_size, nb_points, nb_feats = feats.shape
    feats_bev = torch.zeros(batch_size, *bev_size, nb_feats).to(feats.device)
    bev_size = torch.tensor([*bev_size]).view(1, 1, 3).to(feats.device)

    # check here to use 0.5 or not
    mask = (locations > -0.5).all(2) & (locations < bev_size - 0.5).all(2) & \
           (locations != 0).any(2)  & (torch.norm(locations, dim=2) > 0.1)
    # mask features
    feats = feats * mask.unsqueeze(2)

    locations = locations.round()
    locations = locations.clamp(min=0).clamp(max=bev_size - 1).to(torch.int64)

    # get feats_bev from feats_3d and locations, and make zero the invalids
    batch_idx = torch.arange(batch_size).view(-1, 1).repeat(1, nb_points)
    feats_bev[batch_idx, locations[...,0], locations[...,1], locations[...,2], :] = feats

    return feats_bev
