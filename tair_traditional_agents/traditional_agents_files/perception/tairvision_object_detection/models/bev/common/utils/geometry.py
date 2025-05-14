import numpy as np
import torch
from pyquaternion import Quaternion
from shapely.geometry import MultiPolygon, LineString


def update_intrinsics(intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0):
    """
    Parameters
    ----------
        intrinsics: torch.Tensor (3, 3)
        top_crop: float
        left_crop: float
        scale_width: float
        scale_height: float
    """
    updated_intrinsics = intrinsics.clone()
    # Adjust intrinsics scale due to resizing
    updated_intrinsics[0, 0] *= scale_width
    updated_intrinsics[0, 2] *= scale_width
    updated_intrinsics[1, 1] *= scale_height
    updated_intrinsics[1, 2] *= scale_height

    # Adjust principal point due to cropping
    updated_intrinsics[0, 2] -= left_crop
    updated_intrinsics[1, 2] -= top_crop

    return updated_intrinsics


def update_intrinsics_v2(intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0, hflip=False,
                         img_size=(704, 256)):
    """
    Parameters
    ----------
        intrinsics: torch.Tensor (3, 3)
        top_crop: float
        left_crop: float
        scale_width: float
        scale_height: float
        hflip: bool
        img_size: tuple(int)
    """
    updated_intrinsics = intrinsics.clone()
    device = updated_intrinsics.device
    dtype = updated_intrinsics.dtype

    # Scaling
    sx = scale_width
    sy = scale_height
    scale_matrix = torch.tensor([[sx, 0., 0.],
                                 [0., sy, 0.],
                                 [0., 0., 1.]
                                 ], dtype=dtype, device=device)

    updated_intrinsics = scale_matrix @ updated_intrinsics

    # Translation
    tx = -left_crop
    ty = -top_crop
    trans_matrix = torch.tensor([[tx],
                                 [ty]
                                 ], dtype=dtype, device=device)

    updated_intrinsics[:2, 2:3] = updated_intrinsics[:2, 2:3] + trans_matrix

    # Reflection
    hx = -1.0 if hflip else 1.0
    hy = 1.0
    flip_matrix = torch.tensor([[hx, 0.],
                                [0., hy],
                                ], dtype=dtype, device=device)

    updated_intrinsics[:2, :2] = flip_matrix @ updated_intrinsics[:2, :2]

    cx = img_size[0] if hflip else 0.0
    cy = 0.0
    flip_shift = torch.tensor([[cx],
                               [cy]
                               ], dtype=dtype, device=device)

    updated_intrinsics[:2, 2:3] = flip_matrix @ updated_intrinsics[:2, 2:3] + flip_shift

    # Rotation
    theta = 0.
    cos = np.cos(theta * np.pi / 180)
    sin = np.sin(theta * np.pi / 180)
    rotate_matrix = torch.tensor([[cos, -sin],
                                  [sin,  cos]
                                  ], dtype=dtype, device=device)

    updated_intrinsics[:2, :2] = rotate_matrix @ updated_intrinsics[:2, :2]

    return updated_intrinsics

def update_intrinsics_v3(intrinsics, top_crop=0.0, left_crop=0.0, hflip=False,
                         img_size=(704, 256)):
    """
    Parameters
    ----------
        intrinsics: torch.Tensor (3, 3)
        top_crop: float
        left_crop: float
        scale_width: float
        scale_height: float
        hflip: bool
        img_size: tuple(int)
    """
    updated_intrinsics = intrinsics.clone()
    device = updated_intrinsics.device
    dtype = updated_intrinsics.dtype



    # Translation
    tx = -left_crop
    ty = -top_crop
    trans_matrix = torch.tensor([[tx],
                                 [ty]
                                 ], dtype=dtype, device=device)

    updated_intrinsics[:2, 2:3] = updated_intrinsics[:2, 2:3] + trans_matrix

    # Reflection
    hx = -1.0 if hflip else 1.0
    hy = 1.0
    flip_matrix = torch.tensor([[hx, 0.],
                                [0., hy],
                                ], dtype=dtype, device=device)

    updated_intrinsics[:2, :2] = flip_matrix @ updated_intrinsics[:2, :2]

    cx = img_size[0] if hflip else 0.0
    cy = 0.0
    flip_shift = torch.tensor([[cx],
                               [cy]
                               ], dtype=dtype, device=device)

    updated_intrinsics[:2, 2:3] = flip_matrix @ updated_intrinsics[:2, 2:3] + flip_shift

    # Rotation
    theta = 0.
    cos = np.cos(theta * np.pi / 180)
    sin = np.sin(theta * np.pi / 180)
    rotate_matrix = torch.tensor([[cos, -sin],
                                  [sin,  cos]
                                  ], dtype=dtype, device=device)

    updated_intrinsics[:2, :2] = rotate_matrix @ updated_intrinsics[:2, :2]

    return updated_intrinsics
def update_view(view, hflip=False, vflip=False, rotation_degree=0., bev_scale=None):
    """
    Parameters
    ----------
        view: torch.Tensor (4, 4)
        hflip: bool
        vflip: bool
        rotation_degree: float
        bev_scale: float
    """
    updated_view = view.clone()
    device = updated_view.device
    dtype = updated_view.dtype

    # Reflection
    hx = -1.0 if hflip else 1.0
    hy = -1.0 if vflip else 1.0
    flip_matrix = torch.tensor([[hx, 0.],
                                [0., hy],
                                ], dtype=dtype, device=device)

    updated_view[:2, :2] = flip_matrix @ updated_view[:2, :2]

    cos = np.cos(rotation_degree * np.pi / 180)
    sin = np.sin(rotation_degree * np.pi / 180)
    rotate_matrix = torch.tensor([[cos, -sin],
                                  [sin,  cos]
                                  ], dtype=dtype, device=device)

    updated_view[:2, :2] = rotate_matrix @ updated_view[:2, :2]

    if bev_scale is not None:
        updated_view[0:2, 0:2] *= bev_scale

    return updated_view


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height
    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                 dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


def convert_egopose_to_matrix_numpy(egopose):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = Quaternion(egopose['rotation']).rotation_matrix
    translation = np.array(egopose['translation'])
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def mat2pose_vec(matrix: torch.Tensor):
    """
    Converts a 4x4 pose matrix into a 6-dof pose vector
    Args:
        matrix (ndarray): 4x4 pose matrix
    Returns:
        vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
        rotation components (rx, ry, rz)
    """

    # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])

    # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2)
    roty = torch.atan2(matrix[..., 0, 2], cosy)

    # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
    rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

    rotation = torch.stack((rotx, roty, rotz), dim=-1)

    # Extract translation params
    translation = matrix[..., :3, 3]
    return torch.cat((translation, rotation), dim=-1)


def euler2mat(angle: torch.Tensor):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) [Bx3]
    Returns:
        Rotation matrix corresponding to the euler angles [Bx3x3]
    """
    shape = angle.shape
    angle = angle.view(-1, 3)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack([cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1).view(-1, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1).view(-1, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1).view(-1, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    rot_mat = rot_mat.view(*shape[:-1], 3, 3)
    return rot_mat


def pose_vec2mat(vec: torch.Tensor):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz [B,6]
    Returns:
        A transformation matrix [B,4,4]
    """
    translation = vec[..., :3].unsqueeze(-1)  # [...x3x1]
    rot = vec[..., 3:].contiguous()  # [...x3]
    rot_mat = euler2mat(rot)  # [...,3,3]
    transform_mat = torch.cat([rot_mat, translation], dim=-1)  # [...,3,4]
    transform_mat = torch.nn.functional.pad(transform_mat, [0, 0, 0, 1], value=0)  # [...,4,4]
    transform_mat[..., 3, 3] = 1.0
    return transform_mat


def invert_pose_matrix(x):
    """
    Parameters
    ----------
        x: [B, 4, 4] batch of pose matrices
    Returns
    -------
        out: [B, 4, 4] batch of inverse pose matrices
    """
    assert len(x.shape) == 3 and x.shape[1:] == (4, 4), 'Only works for batch of pose matrices.'

    transposed_rotation = torch.transpose(x[:, :3, :3], 1, 2)
    translation = x[:, :3, 3:]

    inverse_mat = torch.cat([transposed_rotation, -torch.bmm(transposed_rotation, translation)], dim=-1)  # [B,3,4]
    inverse_mat = torch.nn.functional.pad(inverse_mat, [0, 0, 0, 1], value=0)  # [B,4,4]
    inverse_mat[..., 3, 3] = 1.0
    return inverse_mat


def warp_features(x, flow, mode='nearest', spatial_extent=None):
    """ Applies a rotation and translation to feature map x.
        Args:
            x: (b, c, h, w) feature map
            flow: (b, 3,3) 6DoF vector (only uses the xy poriton)
            mode: use 'nearest' when dealing with categorical inputs
        Returns:
            in plane transformed feature map
        """
    if flow is None:
        return x
    b, c, h, w = x.shape
    transformation = flow.clone()[:, 0:2, [0, 1, 3]]

    # Normalise translation. Need to divide by how many meters is half of the image.
    # because translation of 1.0 correspond to translation of half of the image.
    transformation[:, 0, 2] /= spatial_extent[0]
    transformation[:, 1, 2] /= spatial_extent[1]
    # both axis are inverted and rotated
    # transformation_axis_0 -> positive value makes the image move to the left
    # transformation_axis_1 -> positive value makes the image move to the top
    #old one
    #transformation[:, :, 2] *= -1
    #transformation[:, 0, 1] *= -1
    #transformation[:, 1, 0] *= -1
    ###
    transformation[:, :, 2] *= -1
    transformation[:, 0, 1] *= -1
    transformation[:, 1, 0] *= -1
    # transformation[:, :, 2] = transformation[:,[1,0],2]

    # Note that a rotation will preserve distances only if height = width. Otherwise there's
    # resizing going on. e.g. rotation of pi/2 of a 100x200 image will make what's in the center of the image elongated.
    grid = torch.nn.functional.affine_grid(transformation, size=x.shape, align_corners=False)
    warped_x = torch.nn.functional.grid_sample(x, grid.float(), mode=mode, padding_mode='zeros', align_corners=False)

    return warped_x


def cumulative_warp_features(x, flow, mode='nearest', spatial_extent=None):
    """ Warps a sequence of feature maps by accumulating incremental 2d flow.
    x[:, -1] remains unchanged
    x[:, -2] is warped using flow[:, -1]
    x[:, -3] is warped using flow[:, -2] @ flow[:, -1]
    ...
    x[:, 0] is warped using flow[:, 0] @ ... @ flow[:, -2] @ flow[:, -1]
    Args:
        x: (b, t, c, h, w) sequence of feature maps
        flow: (b, t, 3, 3) sequence of 6 DoF pose
            from t to t+1 (only uses the xy poriton)
    """
    sequence_length = x.shape[1]
    if sequence_length == 1:
        return x

    out = [x[:, -1]]
    cum_flow = flow[:, -1]
    for t in reversed(range(sequence_length - 1)):
        out.append(warp_features(x[:, t], cum_flow, mode=mode, spatial_extent=spatial_extent))
        # @ is the equivalent of torch.bmm
        cum_flow = flow[:, t - 1] @ cum_flow

    return torch.stack(out[::-1], 1)


def cumulative_warp_features_reverse(x, flow, mode='nearest', spatial_extent=None):
    """ Warps a sequence of feature maps by accumulating incremental 2d flow.
    x[:, 0] remains unchanged
    x[:, 1] is warped using flow[:, 0].inverse()
    x[:, 2] is warped using flow[:, 0].inverse() @ flow[:, 1].inverse()
    ...
    Args:
        x: (b, t, c, h, w) sequence of feature maps
        flow: (b, t, 6) sequence of 6 DoF pose
            from t to t+1 (only uses the xy poriton)
    """

    out = [x[:, 0]]

    for i in range(1, x.shape[1]):
        if i == 1:
            cum_flow = invert_pose_matrix(flow[:, 0])
        else:
            cum_flow = cum_flow @ invert_pose_matrix(flow[:, i - 1])
        out.append(warp_features(x[:, i], cum_flow, mode, spatial_extent=spatial_extent))
    return torch.stack(out, 1)


def concat_egomotion(x, future_egomotion, receptive_field):
    future_egomotion = mat2pose_vec(future_egomotion)
    b, s, c = future_egomotion.shape
    h, w = x.shape[-2:]
    future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1).expand(b, s, c, h, w)
    # at time 0, no egomotion so feed zero vector
    future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                           future_egomotions_spatial[:, :(receptive_field - 1)]], dim=1)
    x = torch.cat([x, future_egomotions_spatial], dim=-3)

    return x


def points_lidar2view(line_instances, view):
    polygons = [np.pad(p.T, ((0, 1), (0, 0)), constant_values=0.0) for p in line_instances]  # 3 n
    polygons = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in polygons]  # 4 n
    view = view[[0, 1, 3], :] if view.shape[0] == 4 else view
    polygons = [view @ p for p in polygons]
    polygons = [p[:2].T.round().astype('int') for p in polygons]
    return polygons


def sample_polyline_points(line_positions, sample_distance=5):
    line_positions_new = []
    for line_id, line in enumerate(line_positions):
        line_string = LineString(line)
        distances = np.linspace(0, line_string.length, max(int(line_string.length / sample_distance), 1))
        sampled_points = np.array([list(line_string.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        line_positions_new.append(sampled_points)
    return line_positions_new


def polygons_view2lidar(polygons, view):
    inv_view = np.linalg.inv(view)
    polygons = [p.T for p in polygons]
    polygons = pad_polygons(polygons)
    polygons = [(inv_view[[0, 1, 3], :] @ p).T[:, :2] for p in polygons]
    return polygons


def pad_polygons(polygons):
    polygons = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in polygons]  # 3 n
    polygons = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in polygons]  # 4 n
    return polygons
