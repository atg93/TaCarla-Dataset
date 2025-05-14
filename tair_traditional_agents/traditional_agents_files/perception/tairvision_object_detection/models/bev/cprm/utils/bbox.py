import torch
import numpy as np
import cv2

from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion


def view_boxes_to_lidar_boxes_3d(input, batch, filter_classes, score_threshold=0.30, t=-1, is_eval=False):
    if isinstance(input, list):
        input = [{k: v.to('cpu').detach() for k, v in t.items()} for t in input]
    elif isinstance(input, dict):
        input = {k: v.to('cpu').detach() for k, v in input.items()}
    boxes = input['boxes_3d'].clone().cpu().numpy()
    labels = input['labels_3d'].clone().cpu().numpy()
    scores = input['scores_3d'].clone().cpu().numpy() if 'scores_3d' in input.keys() else None

    view = batch['view'][0, t, 0].clone().cpu().numpy()
    view[2] = np.asarray([0., 0., 1., 0.]).astype(np.float32)
    lidar_boxes = []

    for i, box_single in enumerate(boxes):
        center = box_single[0:3]
        size = box_single[3:6]
        yaw = box_single[6]
        #yaw = np.pi/2 - yaw
        rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        # create a box object to get corners
        name = filter_classes.classes[labels[i]+1]
        score = scores[i] if scores is not None else np.nan
        box = NuScenesBox(center, size, rotation, name=name, score=score)
        corners = box.corners().T
        corners = np.concatenate([corners, np.ones_like(corners[:, 0:1])], axis=1)
        # convert pixel location to lidar locations
        corners = np.linalg.inv(view) @ corners.T
        # get the center, size and yaw in lidar domain
        center, size, yaw = corners_to_xyxwlh(corners[0:3, :])
        rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        # create the new box in lidar domain
        lidar_box = NuScenesBox(center, size, rotation, name=name, score=score)
        if not is_eval:
            if scores is None or scores[i] > score_threshold:
                lidar_boxes.append(lidar_box)
            else:
                pass
        else:
            lidar_boxes.append(lidar_box)

    return lidar_boxes


def view_boxes_to_bitmap_3d(output, bev_size=(200, 200), score_threshold=0.40):
    output = output[0]
    try:
        if len(output['boxes_3d']) != 0:
            output = {k: torch.stack([x for i, x in enumerate(v) if output['scores_3d'][i] > score_threshold]) for k, v in
                      output.items()}
    except:
        pass
    output['boxes_3d'] = output['boxes_3d'].cpu()
    base_size = 100
    scale_factor = bev_size[0] // 100  # imageified BEV will be scale_factor*100x100 pixels
    total_size = base_size * scale_factor
    center_scaled = output['boxes_3d'][:, 0:2].type(torch.long)
    length = output['boxes_3d'][:, 3] / 2
    width = output['boxes_3d'][:, 4] / 2
    output['boxes_3d'][:, 6] = np.pi / 2 - output['boxes_3d'][:, 6]

    offset_skeleton = torch.stack([torch.stack([-length, -width]), torch.stack([-length, width]),
                                   torch.stack([length, -width]), torch.stack([length, width])])
    offset_skeleton = offset_skeleton.permute(2, 0, 1)
    sinn = torch.sin(output['boxes_3d'][:, 6])
    coss = torch.cos(output['boxes_3d'][:, 6])
    R1 = torch.stack([coss, -sinn])
    R2 = torch.stack([sinn, coss])
    R = torch.stack([R1, R2])
    R = R.permute(2, 0, 1)
    rotated_offsets = torch.einsum('bij,bjk->bik', offset_skeleton, R)
    corners = torch.stack([center_scaled + rotated_offsets[:, 0, :], center_scaled + rotated_offsets[:, 1, :],
                           center_scaled + rotated_offsets[:, 2, :], center_scaled + rotated_offsets[:, 3, :]])
    corners = np.array(corners.permute(1, 0, 2)).astype(np.int32)
    blank_sheet = np.zeros((total_size, total_size)).astype(np.uint8)
    blank_sheet = np.asarray(blank_sheet)
    clr = (235, 145, 25)
    th = 1
    for i in range(center_scaled.size(0)):
        c1 = [corners[i, 0, 0], corners[i, 0, 1]]
        c2 = [corners[i, 1, 0], corners[i, 1, 1]]
        c3 = [corners[i, 2, 0], corners[i, 2, 1]]
        c4 = [corners[i, 3, 0], corners[i, 3, 1]]
        cv2.fillPoly(blank_sheet, [np.asarray([c1, c2, c4, c3]).astype(np.int32)], 3)

    show_img = blank_sheet.astype(np.uint8)

    return show_img


def corners_to_xyxwlh(corners):
    x, y, z = corners.mean(1)

    bottom_corners = corners[:, [2, 3, 7, 6]]
    rear_side = (bottom_corners[:, 0] + bottom_corners[:, 1]) / 2
    front_side = (bottom_corners[:, 3] + bottom_corners[:, 2]) / 2
    left_side = (bottom_corners[:, 0] + bottom_corners[:, 3]) / 2
    right_side = (bottom_corners[:, 1] + bottom_corners[:, 2]) / 2

    x_diff_fr, y_diff_fr, z_diff_fr = front_side - rear_side
    x_diff_lr, y_diff_lr, z_diff_lr = right_side - left_side

    # to get the same yaw with lidar_box, we need to calculate yaw from x/y instead of y/x
    if y_diff_fr == 0 and x_diff_fr == 0:
        yaw = 0
    else:
        yaw = np.arctan2(y_diff_fr, x_diff_fr)
        # yaw = np.arctan(y_diff_fr / x_diff_fr)

    w = np.sqrt(x_diff_lr**2 + y_diff_lr**2)
    l = np.sqrt(x_diff_fr**2 + y_diff_fr**2)
    h = (corners[2, [0, 1, 5, 4]] - corners[2, [2, 3, 7, 6]]).mean()

    return [x, y, z], [w, l, h], yaw


def get_targets3d(batch, receptive_field=1, spatial_extent=None):
    gtbboxes3d = []
    labels3d = []
    t = receptive_field - 1
    for bn, btch in enumerate(batch['boxes']):
        btch = btch[0][6]
        view = batch['view'][bn, t, 0].clone().cpu().numpy()
        view[2] = np.asarray([0., 0., 1., 0.]).astype(np.float32)
        if btch.__len__() == 0:
            gtbboxes3d.append(torch.empty(0, 7).type(torch.float32))  # TODO: make this 9
            labels3d.append(torch.empty(0).to('cuda').type(torch.int32))
            continue
        gtbatch3d = []
        labelbatch3d = []

        for btc in btch:
            corners = btc.corners()
            corners = np.concatenate([corners, np.ones_like(corners[0:1, :])])
            corners = (view @ corners)[0:3, :]
            xyz, wlh, yaw = corners_to_xyxwlh(corners)
            x, y, z = xyz
            w, l, h = wlh
            gtbatch3d.append(torch.tensor([x,y,z,w,l,h,yaw]))

            # TODO: NO VELO HERE!!!!!!!!
            labelbatch3d.append(torch.tensor(btc.label).to('cuda').type(torch.int32))
        gtbboxes3d.append(torch.stack(gtbatch3d, dim=0).to('cuda'))
        labels3d.append(torch.stack(labelbatch3d, dim=0).to('cuda'))

    targets3d = []
    for i in range(len(labels3d)):
        targets3d.append({'boxes_3d': gtbboxes3d[i],
                          'labels_3d': labels3d[i],
                          })

    return targets3d, None
