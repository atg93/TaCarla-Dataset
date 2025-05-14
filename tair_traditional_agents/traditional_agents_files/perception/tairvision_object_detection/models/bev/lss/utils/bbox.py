from typing import Union, List, Tuple
import torch
import numpy as np
import cv2

from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from shapely.geometry import MultiPoint, box, polygon
from pyquaternion import Quaternion

import math


def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        if isinstance(polygon_from_2d_box, polygon.Polygon):
            img_intersection = polygon_from_2d_box.intersection(img_canvas)
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            return min_x, min_y, max_x, max_y
        else:
            return None
    else:
        return None


def box2corners(boxes, intrinsics, image_size=(480, 224)):
    bboxes = []
    cuboids = []
    categories = []
    labels = []

    for i_box, box in enumerate(boxes):
        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, intrinsics, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords, image_size)

        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            bboxes.append(np.asarray(final_coords))
            cuboids.append(np.asarray(corner_coords))
            categories.append(box.name)
            labels.append(box.label)

    return bboxes, cuboids, categories, labels


def get_targets2d(batch, receptive_field=1, image_size=(480, 224)):
    b, s, c = batch['intrinsics'].shape[0:3]
    device = batch['intrinsics'].device
    t = receptive_field - 1
    intrinsics = batch['intrinsics'].cpu().numpy()

    boxes_all = []
    labels_all = []
    areas_all = []
    valid_index = []
    for i in range(b):
        for j in range(t, t+1):
            for k in range(c):
                boxes, _, categories, labels = box2corners(batch['boxes'][i][j][k], intrinsics[i, j, k], image_size)
                if len(boxes):
                    boxes = np.stack(boxes)
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

                    boxes_all.append(torch.from_numpy(boxes).to(device))
                    labels_all.append(torch.from_numpy(np.asarray(labels)).to(device))
                    areas_all.append(torch.from_numpy(areas).to(device))
                    valid_index.append(True)
                else:
                    boxes_all.append(torch.tensor([]).to(device))
                    labels_all.append(torch.tensor([]).to(device))
                    areas_all.append(torch.tensor([]).to(device))
                    valid_index.append(False)

    targets2d = []
    for i in range(len(labels_all)):
        targets2d.append({'boxes': boxes_all[i],
                          'labels': labels_all[i],
                          'area': areas_all[i],
                          })

    return targets2d, valid_index


def filter_imgs_with_no_boxes(x, valid):
    out = {}
    for key, value in x.items():
        if isinstance(value, list):
            for val in range(len(value)):
                if key not in out:
                    out[key] = []
                out[key].append(value[val][valid])
        else:
            out[key] = value[valid]

    return out


def get_targets3d_yaw(batch, receptive_field=1, spatial_extent=None, use_view_domain=True):
    b, s, c = batch['intrinsics'].shape[0:3]
    device = batch['intrinsics'].device
    t = receptive_field - 1

    view = batch['view'].clone().cpu().numpy()
    view[:, :, :, 2] = np.asarray([0., 0., 1., 0.]).astype(np.float32)

    boxes_all = []
    labels_all = []
    others_all = []
    areas_all = []
    valid_index = []
    for i in range(b):
        for j in range(t, s):
            boxes = []
            labels = []
            others = []
            others_yaw = []
            others_z_h = []
            lidar_boxes = batch['boxes'][i][j][-1]
            for lidar_box in lidar_boxes:
                if spatial_extent is not None and (spatial_extent < np.abs(lidar_box.center[0:2])).any():
                    continue
                vel_x, vel_y, vel_z = lidar_box.velocity
                if use_view_domain:
                    # Transfer box corners to 0:200 view domain from -50:50 lidar domain
                    corners = lidar_box.corners()
                    corners = np.concatenate([corners, np.ones_like(corners[0:1, :])])
                    corners = (view[i, j, 0] @ corners)[0:3, :]
                    xyz, wlh, yaw = corners_to_xyzwlh(corners, update_yaw=np.pi/2)
                    x, y, z = xyz
                    w, l, h = wlh
                    yaw_cos = np.cos(yaw)
                    yaw_sin = np.sin(yaw)
                else:
                    x, y, z = lidar_box.center
                    w, l, h = lidar_box.wlh
                    yaw = lidar_box.orientation.yaw_pitch_roll[0]
                    yaw_cos = np.cos(yaw)
                    yaw_sin = np.sin(yaw)

                label = lidar_box.label

                boxes.append(np.asarray([x-w/2, y-l/2, x+w/2, y+l/2]))
                labels.append(label)
                others_z_h.append(np.asarray([z, h]))
                others_yaw.append(np.asarray([yaw_cos, yaw_sin]))

            if len(boxes):
                boxes = np.stack(boxes)
                labels = np.stack(labels)
                others.append(torch.tensor(np.stack(others_z_h), dtype=torch.float32, device=device))
                others.append(torch.tensor(np.stack(others_yaw), dtype=torch.float32, device=device))
                others_all.append(others)
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

                boxes_all.append(torch.tensor(boxes, dtype=torch.float32, device=device))
                labels_all.append(torch.tensor(labels, device=device))
                areas_all.append(torch.tensor(areas, dtype=torch.float32, device=device))
                valid_index.append(True)
            else:
                boxes_all.append(torch.tensor([]).to(device))
                labels_all.append(torch.tensor([]).to(device))
                others_all.append(torch.tensor([]).to(device))
                areas_all.append(torch.tensor([]).to(device))
                valid_index.append(False)

    targets3d = []
    for i in range(len(labels_all)):
        targets3d.append({'boxes': boxes_all[i],
                          'labels': labels_all[i],
                          'others': others_all[i],
                          'area': areas_all[i],
                          })

    return targets3d, valid_index


def corners_to_xyzwlh(corners, update_yaw=0):
    x, y, z = corners.mean(1)

    bottom_corners = corners[:, [2, 3, 7, 6]]
    rear_side = (bottom_corners[:, 3] + bottom_corners[:, 2]) / 2
    front_side = (bottom_corners[:, 0] + bottom_corners[:, 1]) / 2
    left_side = (bottom_corners[:, 0] + bottom_corners[:, 3]) / 2
    right_side = (bottom_corners[:, 1] + bottom_corners[:, 2]) / 2

    x_diff_fr, y_diff_fr, z_diff_fr = rear_side - front_side
    x_diff_lr, y_diff_lr, z_diff_lr = right_side - left_side

    if (x_diff_fr ** 2 + y_diff_fr ** 2) < (x_diff_lr ** 2 + y_diff_lr ** 2):
        x_buf, y_buf, z_buf = x_diff_fr, y_diff_fr, z_diff_fr
        x_diff_fr, y_diff_fr, z_diff_fr = x_diff_lr, y_diff_lr, z_diff_lr
        x_diff_lr, y_diff_lr, z_diff_lr = x_buf, y_buf, z_buf

    # To get the same yaw with lidar_box, we need to calculate yaw from x/y instead of y/x
    if y_diff_fr == 0 or x_diff_fr == 0:
        yaw = 0
    else:
        yaw = np.arctan2(y_diff_fr, x_diff_fr)
    yaw = yaw + update_yaw

    w = np.sqrt(x_diff_lr**2 + y_diff_lr**2)
    l = np.sqrt(x_diff_fr**2 + y_diff_fr**2)
    h = (corners[2, [0, 1, 5, 4]] - corners[2, [2, 3, 7, 6]]).mean()

    return [x, y, z], [w, l, h], yaw


def get_targets3d_xdyd(batch, receptive_field=1, spatial_extent=None, return_vel=False):
    b, s, c = batch['intrinsics'].shape[0:3]
    device = batch['intrinsics'].device
    t = receptive_field - 1

    view = batch['view'].clone().cpu().numpy()

    boxes_all = []
    labels_all = []
    others_all = []
    areas_all = []
    valid_index = []
    for i in range(b):
        for j in range(t, s):
            boxes = []
            labels = []
            others = []
            others_xd_yd = []
            others_xc_yc = []
            others_z_h = []
            others_orient = []
            # others_yaw = []
            others_vel = []
            lidar_boxes = batch['boxes'][i][j][-1]
            for lidar_box in lidar_boxes:
                if spatial_extent is not None and (spatial_extent < np.abs(lidar_box.center[0:2])).any():
                    continue
                corners = lidar_box.corners()
                corners = np.concatenate([corners, np.ones_like(corners[0:1, :])])
                corners = (view[i, j, 0] @ corners)[0:3, :]
                # yaw = corners_to_xyzwlh(corners[0:3, ])[2] - np.pi
                # yaw_cos, yaw_sin = np.cos(yaw / 2), np.sin(yaw / 2)
                bottom_corners = corners[0:2, [2, 3, 7, 6]]

                min_x, min_y = bottom_corners.min(1)
                max_x, max_y = bottom_corners.max(1)
                width = max_x - min_x
                length = max_y - min_y

                argmin_x, argmin_y = bottom_corners.argmin(1)
                argmax_x, argmax_y = bottom_corners.argmax(1)

                x_top = bottom_corners[0, argmin_y]
                y_left = bottom_corners[1, argmin_x]
                x_bottom = bottom_corners[0, argmax_y]
                y_right = bottom_corners[1, argmax_x]

                if argmin_y == 0:
                    oc1, oc2 = 0., 0.
                elif argmin_x == 0:
                    oc1, oc2 = 0., 1.
                elif argmax_y == 0:
                    oc1, oc2 = 1., 0.
                elif argmax_x == 0:
                    oc1, oc2 = 1., 1.
                else:
                    oc1, oc2 = 0., 0.

                xd = ((x_top - min_x) + (max_x - x_bottom))/2
                yd = ((y_right - min_y) + (max_y - y_left))/2

                xc = 0. if xd < (width / 2) else 1.
                xd = xd if xd < (width / 2) else width - xd

                yc = 0. if yd < (length / 2) else 1.
                yd = yd if yd < (length / 2) else length - yd

                z = corners[2, :].mean()
                h = (corners[2, [0, 1, 5, 4]] - corners[2, [2, 3, 7, 6]]).mean()

                vel = lidar_box.velocity[0:2] if return_vel else None

                label = lidar_box.label

                boxes.append(np.asarray([min_x, min_y, max_x, max_y]))
                labels.append(label)
                others_z_h.append(np.asarray([z, h]))
                others_xd_yd.append(np.asarray([xd, yd]))
                others_xc_yc.append(np.asarray([xc, yc]))
                others_orient.append(np.asarray([oc1, oc2]))
                # others_yaw.append(np.asarray([yaw_cos, yaw_sin]))
                if return_vel:
                    others_vel.append(np.asarray([*vel]))
            if len(boxes):
                boxes = np.stack(boxes)
                labels = np.stack(labels)
                others.append(torch.tensor(np.stack(others_z_h), dtype=torch.float32, device=device))
                others.append(torch.tensor(np.stack(others_xd_yd), dtype=torch.float32, device=device))
                others.append(torch.tensor(np.stack(others_xc_yc), dtype=torch.float32, device=device))
                others.append(torch.tensor(np.stack(others_orient), dtype=torch.float32, device=device))
                # others.append(torch.tensor(np.stack(others_yaw), dtype=torch.float32, device=device))
                if return_vel:
                    others.append(torch.tensor(np.stack(others_vel), dtype=torch.float32, device=device))
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

                boxes_all.append(torch.tensor(boxes, dtype=torch.float32, device=device))
                labels_all.append(torch.tensor(labels, device=device))
                others_all.append(others)
                areas_all.append(torch.tensor(areas, dtype=torch.float32, device=device))
                valid_index.append(True)
            else:
                boxes_all.append(torch.tensor([]).to(device))
                labels_all.append(torch.tensor([]).to(device))
                others_all.append(torch.tensor([]).to(device))
                areas_all.append(torch.tensor([]).to(device))
                valid_index.append(False)

    targets3d = []
    for i in range(len(labels_all)):
        targets3d.append({'boxes': boxes_all[i],
                          'labels': labels_all[i],
                          'others': others_all[i],
                          'area': areas_all[i],
                          })

    return targets3d, valid_index


def view_boxes_to_lidar_boxes_yaw(input, batch, filter_classes, score_threshold=0.30, t=-1, is_eval=False):
    boxes = input['boxes'].clone().cpu().numpy()

    others_dict = {}
    for i in range(len(input['others'])):
        others_dict[i] = input['others'][i].cpu().numpy()

    labels = input['labels'].clone().cpu().numpy()
    scores = input['scores'].clone().cpu().numpy() if 'scores' in input.keys() else None

    view = batch['view'][0, t, 0].clone().cpu().numpy()
    view[2] = np.asarray([0., 0., 1., 0.]).astype(np.float32)
    lidar_boxes = []
    for i in range(len(boxes)):
        label = labels[i]
        name = filter_classes.classes[labels[i]]
        score = scores[i] if scores is not None else np.nan
        if label == 0:  # Need for validation check, pass background
            continue
        x1, y1, x2, y2 = boxes[i]
        z, h = others_dict[0][i]
        yaw_cos, yaw_sin = others_dict[1][i]

        if yaw_sin == 0 or yaw_cos == 0:
            yaw = 0
        else:
            yaw = np.arctan2(yaw_sin, yaw_cos)

        center = [(x1 + x2) / 2, (y1 + y2) / 2, z]
        size = [x2 - x1, y2 - y1, h]
        rotation = Quaternion(scalar=np.cos(yaw/2), vector=[0, 0, np.sin(yaw/2)])

        box = Box(center,
                  size,
                  rotation,
                  name=name,
                  score=score,
                  label=label,
                  )
        corners = box.corners().T
        corners = np.concatenate([corners, np.ones_like(corners[:, 0:1])], axis=1)
        # Convert pixel location to lidar locations
        corners = np.linalg.inv(view) @ corners.T

        # Get the center, size and yaw in lidar domain
        center, size, yaw = corners_to_xyzwlh(corners[0:3, ], update_yaw=np.pi/2)
        rotation = Quaternion(scalar=np.cos(yaw / 2),
                              vector=[0, 0, np.sin(yaw / 2)]
                              )

        # Create the new box in lidar domain
        lidar_box = Box(center, size, rotation, label=label, score=score, name=name)
        lidar_box = filter_classes.revert_sizes(lidar_box)

        if not is_eval:
            if scores is None or scores[i] > score_threshold:
                lidar_boxes.append(lidar_box)
        else:
            lidar_boxes.append(lidar_box)

    return lidar_boxes





def rotate_points(origin, point, yaw_cos, yaw_sin):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + yaw_cos * (px - ox) - yaw_sin * (py - oy)
    qy = oy + yaw_sin * (px - ox) + yaw_cos * (py - oy)
    return qx, qy


def view_boxes_to_bitmap_yaw(input, bev_size=(200, 200), score_threshold=0.40, gt=False):
    output = np.zeros(bev_size).astype(np.uint8)
    for x in input:
        scores = x['scores'].clone().cpu().numpy() if 'scores' in x.keys() else None
        for i, box in enumerate(x['boxes']):
            x1, y1, x2, y2 = box.cpu().numpy()
            center = [(x1 + x2) / 2, (y1 + y2) / 2]
            yaw_cos, yaw_sin = x['others'][1][i].cpu().numpy().reshape(-1)
            yaw = np.arctan2(yaw_sin, yaw_cos)

            points = np.asarray([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
            corners = []
            for point in points:
                new_points = rotate_points(center, point, np.cos(yaw), np.sin(yaw))
                corners.append([new_points])
            corners = np.asarray(corners)

            if scores is None or scores[i] > score_threshold:
                cv2.fillPoly(output, [points.astype(np.int32)], 2)
                cv2.fillPoly(output, [corners.astype(np.int32)], 1)

    return output

def view_boxes_to_lidar_boxes_xdyd(input, batch, score_threshold=0.30, t=-1, is_eval=False):
    input = input['head3d'][0]#tugrul
    boxes = input['boxes'].clone().cpu().numpy()
    others_z_h = input['others'][0].clone().cpu().numpy()
    others_xd_yd = input['others'][1].clone().cpu().numpy()
    others_xc_yc = input['others'][2].clone().cpu().numpy()
    others_orient = input['others'][3].clone().cpu().numpy()
    # others_yaw = input['others'][4].clone().cpu().numpy()
    labels = input['labels'].clone().cpu().numpy()
    scores = input['scores'].clone().cpu().numpy() if 'scores' in input.keys() else None

    view = batch.squeeze(0).squeeze(0).squeeze(0) #batch['view'][0, t, 0].clone().cpu().numpy()

    lidar_boxes = []
    for i, box in enumerate(boxes):
        label = labels[i]
        #name = filter_classes.classes[labels[i]]
        score = scores[i] if scores is not None else np.nan

        if label == 0:  # Need for validation check, pass background
            continue
        z, h = others_z_h[i]
        xd, yd = others_xd_yd[i]
        xc, yc = others_xc_yc[i]
        oc1, oc2 = others_orient[i]
        # yaw_cos, yaw_sin = others_yaw[i]
        xd_yd_xc_yc = [xd, yd, xc, yc]

        z_bottom = z - h / 2
        z_top = z + h / 2

        corners, _ = get_corners_xdyd(xd_yd_xc_yc, box, margin=1.0)

        bottom_corners = np.concatenate([corners, z_bottom.reshape(1, 1).repeat(4, 0)], axis=1)
        top_corners = np.concatenate([corners, z_top.reshape(1, 1).repeat(4, 0)], axis=1)
        corners = np.stack([top_corners[0], top_corners[1], bottom_corners[0], bottom_corners[1],
                            top_corners[3], top_corners[2], bottom_corners[3], bottom_corners[2]], axis=0)
        corners = np.concatenate([corners, np.ones_like(corners[:, 0:1])], axis=1)
        # convert pixel location to lidar locations
        corners = np.linalg.inv(view) @ corners.T
        # get the center, size and yaw in lidar domain

        if (oc1 < 0.5 and oc2 < 0.5) or (oc1 >= 0.5 and oc2 >= 0.5):
            center, size, yaw = corners_to_xyzwlh(corners[0:3, ], update_yaw=np.pi)
        else:
            center, size, yaw = corners_to_xyzwlh(corners[0:3, ], update_yaw=0)

        rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])

        # create the new box in lidar domain
        lidar_box = Box(center, size, rotation, label=label, score=score, name=name)
        #lidar_box = filter_classes.revert_sizes(lidar_box)

        if not is_eval:
            if scores is None or scores[i] > score_threshold:
                lidar_boxes.append(lidar_box)
        else:
            lidar_boxes.append(lidar_box)

    return lidar_boxes


def view_boxes_to_bitmap_xdyd(input, bev_size=(200, 200), score_threshold=0.40, draw_labels=None):
    if draw_labels is None:
        draw_labels = [1, 2, 3]
    output = np.zeros(bev_size).astype(np.uint8)
    for x in input:
        scores = x['scores'].clone().cpu().numpy() if 'scores' in x.keys() else None
        labels = x['labels'].clone().cpu().numpy() if 'labels' in x.keys() else None
        #scores = x['scores'].clone().numpy() if 'scores' in x.keys() else None
        #labels = x['labels'].clone().numpy() if 'labels' in x.keys() else None
        for i, box in enumerate(x['boxes']):
            box = box.cpu().numpy()
            z, h = x['others'][0][i].cpu().numpy().reshape(-1)
            xd, yd = x['others'][1][i].cpu().numpy().reshape(-1)
            xc, yc = x['others'][2][i].cpu().numpy().reshape(-1)
            oc1, oc2 = x['others'][3][i].cpu().numpy().reshape(-1)
            #box = box.numpy()
            #z, h = x['others'][0][i].numpy().reshape(-1)
            #xd, yd = x['others'][1][i].numpy().reshape(-1)
            #xc, yc = x['others'][2][i].numpy().reshape(-1)
            #oc1, oc2 = x['others'][3][i].numpy().reshape(-1)
            xd_yd_xc_yc = [xd, yd, xc, yc]

            if (oc1 < 0.5 and oc2 < 0.5) or (oc1 >= 0.5 and oc2 >= 0.5):
                color = 1
            else:
                color = 1

            corners, points = get_corners_xdyd(xd_yd_xc_yc, box, margin=1.0)



            if (scores is None or scores[i] > score_threshold) and (labels is None or labels[i] in draw_labels):
                #cv2.fillPoly(output, [points.astype(np.int32)], 3)

                cv2.fillPoly(output, [corners.astype(np.int32)], color)

    return output


def calculate_bbox_properties(bbox):
    # bbox is assumed to be a NumPy array of shape (4, 2)
    # Each row corresponds to a vertex: [top-left, top-right, bottom-right, bottom-left]

    # Calculating width and height
    width = np.linalg.norm(bbox[0] - bbox[1])
    height = np.linalg.norm(bbox[1] - bbox[2])

    # Calculating orientation
    # Orientation is the angle between the top edge and the x-axis
    top_edge = bbox[1] - bbox[0]
    angle_rad = np.arctan2(top_edge[1], top_edge[0])

    # Converting radians to degrees
    angle_deg = np.degrees(angle_rad)

    return width, height, angle_deg

def get_center_size_angle(det_output, view, bev_size=(200, 200), score_threshold=0.40, draw_labels=1, ego_per_meter=4, lss_per_meter=2, name='vehicle', t=-1, is_eval=False):
    input = det_output['head3d']  # tugrul
    plant_boxes = []
    box_list = []
    orientation_angle_list = []
    ego_front_mid_corners_loc = np.array([95,89])
    ego_back_mid_corners_loc = np.array([95,110])
    ego_orientation_angle = calculate_orientation_angle(ego_front_mid_corners_loc, ego_back_mid_corners_loc)
    if draw_labels is None:
        draw_labels = [1, 2, 3]
    output = np.zeros(bev_size).astype(np.uint8)
    ego_center_x, ego_center_y = (np.array(bev_size) / 2) - 0.5
    ratio = ego_per_meter / lss_per_meter
    for x in input:
        scores = x['scores'].clone().cpu().numpy() if 'scores' in x.keys() else None
        labels = x['labels'].clone().cpu().numpy() if 'labels' in x.keys() else None
        # scores = x['scores'].clone().numpy() if 'scores' in x.keys() else None
        # labels = x['labels'].clone().numpy() if 'labels' in x.keys() else None
        for i, box in enumerate(x['boxes']):
            label = labels[i]
            score = scores[i] if scores is not None else np.nan

            box = box.cpu().numpy()
            z, h = x['others'][0][i].cpu().numpy().reshape(-1)
            xd, yd = x['others'][1][i].cpu().numpy().reshape(-1)
            xc, yc = x['others'][2][i].cpu().numpy().reshape(-1)
            oc1, oc2 = x['others'][3][i].cpu().numpy().reshape(-1)

            # box = box.numpy()
            # z, h = x['others'][0][i].numpy().reshape(-1)
            # xd, yd = x['others'][1][i].numpy().reshape(-1)
            # xc, yc = x['others'][2][i].numpy().reshape(-1)
            # oc1, oc2 = x['others'][3][i].numpy().reshape(-1)
            xd_yd_xc_yc = [xd, yd, xc, yc]
            oc = [oc1, oc2]

            z_bottom = z - h / 2
            z_top = z + h / 2

            corners, points = get_corners_xdyd(xd_yd_xc_yc, box, margin=1.0)
            new_corners, _, front_corners = new_get_corners_xdyd(xd_yd_xc_yc, box, oc, margin=1.0)
            back_corners = (1 - front_corners).astype(np.bool)
            front_mid_corners_loc = new_corners[front_corners].mean(0)
            back_mid_corners_loc = new_corners[back_corners].mean(0)

            orientation_angle = 360-(calculate_orientation_angle(front_mid_corners_loc, back_mid_corners_loc)- ego_orientation_angle)%360
            corners = (corners - ego_center_x) * ratio

            center_x, center_y = corners.mean(0)[0] / ego_per_meter, corners.mean(0)[1] / ego_per_meter
            width, length, angle_deg = calculate_bbox_properties(corners)

            # new_center_x, new_center_y = (center_x-ego_center_x)*ratio+ego_center_x, (center_y-ego_center_y)*ratio+ego_center_y

            # center_x, center_y, width, length = new_center_x, new_center_y, width*ratio, length*ratio

            center_z, height = 0, 0

            if score > score_threshold:
                corners += ego_center_x
                box_list.append(corners)
                plant_boxes.append(np.array([center_x, center_y, center_z, angle_deg, width, length, height]))
                orientation_angle_list.append(orientation_angle)

    return plant_boxes, box_list, orientation_angle_list


def lidar_boxes_to_cam_boxes(lidar_boxes, cam_to_lidar_matrix):
    boxes = lidar_boxes.copy()
    if isinstance(cam_to_lidar_matrix, torch.Tensor):
        cam_to_lidar_matrix = cam_to_lidar_matrix.numpy()

    translation = cam_to_lidar_matrix[0:3, 3]
    rotation = Quaternion._from_matrix(cam_to_lidar_matrix, rtol=1e-05, atol=1e-06)

    cam_boxes = []
    for box in boxes:
        box.translate(-translation)
        box.rotate(rotation.inverse)
        cam_boxes.append(box)

    return cam_boxes


def filter_scenes_with_no_lidar_boxes(x, valid):
    out = {}
    for key, value in x.items():
        if isinstance(value, list):
            other_regressions = []
            for v in value:
                other_regressions.append(v[valid])
            out[key] = other_regressions
        else:
            out[key] = value[valid]

    return out


def get_corners_xdyd(xd_yd, box, margin=0.):
    xd, yd, xc, yc = xd_yd
    x1, y1, x2, y2 = box
    xd = xd if xc < 0.5 else (x2 - x1) - xd
    yd = yd if yc < 0.5 else (y2 - y1) - yd

    if xd < margin and yd > (y2 - y1 - margin):
        yd = xd
    elif yd < margin and xd > (x2 - x1 - margin):
        xd = yd

    corners = np.stack([[x1 + xd, y1], [x2, y1 + yd], [x2 - xd, y2], [x1, y2 - yd]], axis=0)
    points = np.asarray([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])

    return corners, points

def new_get_corners_xdyd(xd_yd, box, oc, margin=0.):
    xd, yd, xc, yc = xd_yd
    x1, y1, x2, y2 = box
    xd = xd if xc < 0.5 else (x2 - x1) - xd
    yd = yd if yc < 0.5 else (y2 - y1) - yd
    oc1, oc2 = oc

    if xd < margin and yd > (y2 - y1 - margin):
        yd = xd
    elif yd < margin and xd > (x2 - x1 - margin):
        xd = yd

    corners = np.stack([[x1 + xd, y1], [x2, y1 + yd], [x2 - xd, y2], [x1, y2 - yd]], axis=0)
    points = np.asarray([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])

    if oc1 < 0.5 and oc2 < 0.5:
        front_corners = np.asarray([True, False, False, True])
    elif oc1 < 0.5 and oc2 >= 0.5:
        front_corners = np.asarray([False, False, True, True])
    elif oc1 >= 0.5 and oc2 < 0.5:
        front_corners = np.asarray([False, True, True, False])
    elif oc1 >= 0.5 and oc2 >= 0.5:
        front_corners = np.asarray([True, True, False, False])
    else:
        front_corners = np.asarray([True, False, False, True])

    return corners, points, front_corners

def calculate_orientation_angle(front_mid_corners_loc, back_mid_corners_loc):
    # Calculate the orientation angle
    dx = front_mid_corners_loc[1] - back_mid_corners_loc[1]
    dy = front_mid_corners_loc[0] - back_mid_corners_loc[0]
    angle = math.atan2(dy, dx)

    # Convert the angle from radians to degrees if necessary
    angle_degrees = math.degrees(angle)

    return angle_degrees
