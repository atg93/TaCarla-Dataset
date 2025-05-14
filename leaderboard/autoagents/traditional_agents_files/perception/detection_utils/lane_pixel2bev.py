import torch
import numpy as np


def get_world_coordinates_from_perspective_with_flat_ground_assumption(
        H_im2v, view_matrix, target_lanes_in_culane):

    if not isinstance(H_im2v, torch.Tensor):
        H_im2v = torch.tensor(H_im2v)

    if not isinstance(view_matrix, torch.Tensor):
        view_matrix = torch.tensor(view_matrix, device=H_im2v.device)

    target_lanes_world = []
    target_lanes_view = []
    for target_lane in target_lanes_in_culane:
        if not isinstance(target_lane, torch.Tensor):
            target_lane = torch.tensor(target_lane, device=H_im2v.device, dtype=torch.float32)

        target_lane = torch.cat((
            target_lane, torch.ones((target_lane.shape[0], 1), device=target_lane.device)), dim=1)

        target_world = torch.matmul(H_im2v, target_lane.T)
        target_world = target_world / target_world[2, :]
        cond1 = torch.bitwise_and(target_world[0] > 0, target_world[0] < 100)
        cond2 = torch.bitwise_and(target_world[1] > -30, target_world[1] < 30)
        cond = torch.bitwise_and(cond1, cond2)
        target_world = target_world[:, cond]
        target_world = torch.cat((
            target_world, torch.ones((1, target_world.shape[1]), device=target_world.device)), dim=0)
        target_world[2] = 0
        target_view = torch.matmul(view_matrix, target_world)
        target_world = target_world.T
        target_view = target_view.T
        idxs = torch.argsort(target_world[:, 0])
        target_world_sorted = torch.index_select(target_world, 0, idxs)
        target_lane_world_sorted = target_world_sorted[:, :2]
        target_view_sorted = torch.index_select(target_view, 0, idxs)
        target_lane_view_sorted = (target_view_sorted[:, :2]).round().to(torch.int32)

        target_lanes_world.append(target_lane_world_sorted.cpu().numpy())
        target_lanes_view.append(target_lane_view_sorted.cpu().numpy())

    return target_lanes_world, target_lanes_view


def get_homography_matrix(intrinsic, extrinsic):
    #extrinsic[0:3,0:3] = np.linalg.inv(extrinsic[0:3,0:3])
    #extrinsic[:,3] = -extrinsic[:,3]
    #extrinsic[0:3,3] = extrinsic[0:3,0:3] @ extrinsic[0:3,3]
    extrinsic = np.linalg.inv(extrinsic)

    E_vc_34 = extrinsic[0:3, :]
    H_v2c = E_vc_34[:, [0, 1, 3]]
    H_v2im = np.matmul(intrinsic, H_v2c)
    H_im2v = np.linalg.inv(H_v2im)
    H_im2v = H_im2v.astype(np.float32)

    return H_im2v

def lane_dict_filter(batch_lane_dict):
    pv_lanes = []
    categories = []
    for category, batch_lane in batch_lane_dict.items():
        if np.size(batch_lane) == 0:
            continue

        if batch_lane.shape[0] < 3:
            continue

        pv_lanes.append(batch_lane)
        categories.append(category)

    return pv_lanes, categories