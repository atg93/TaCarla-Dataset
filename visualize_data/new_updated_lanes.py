import os
import gzip
import json
import numpy as np
import copy
import cv2


def plot_bounding_box_center(center, width=4, height=8):
    mask = np.zeros((200, 200)).astype(np.uint8)
    # Calculate the top-left corner from the center, width, and height
    top_left = (int(center[1] - width / 2), int(center[0] - height / 2))
    bottom_right = (int(center[1] + width / 2), int(center[0] + height / 2))

    # Draw the rectangle (bounding box)
    cv2.rectangle(mask, bottom_right, top_left, (255), 2)  # Blue box

    return mask

def draw_vehicle(center, orientation_rad, velocity, box_size, bbox, arrow_thick=2):
    """
    Draw a vehicle's bounding box with orientation and velocity on an image using OpenCV.

    Args:
    - image: The image to draw on (numpy array).
    - center (tuple): The (x, y) center of the bounding box.
    - orientation_deg (float): The orientation of the vehicle in degrees.
    - velocity (float): The velocity of the vehicle.
    - box_size (tuple): The size of the bounding box (width, height).
    """
    mask_vehicle = np.zeros((200, 200)).astype(np.uint8)
    mask_arrow = np.zeros((200, 200)).astype(np.uint8)

    # Convert center to integer coordinates for drawing
    width, height = bbox[0] * 3, bbox[1] * 3
    top_left = (int(center[1] - width / 2), int(center[0] - height / 2))
    bottom_right = (int(center[1] + width / 2), int(center[0] + height / 2))

    # Draw the rectangle (bounding box)
    # cv2.rectangle(mask_vehicle, bottom_right, top_left, (255), 2)  # Red box

    # Combine the points into a numpy array with the correct shape
    top_right = (top_left[0], bottom_right[1])
    bottom_left = (bottom_right[0], top_left[1])
    pts = np.array([bottom_right, top_right, top_left, bottom_left], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    # Fill the polygon on the image
    cv2.fillPoly(mask_vehicle, [pts], color=(255))
    # cv2.imwrite('mask_vehicle.png', mask_vehicle)

    center = (int(center[1]), int(center[0]))

    # Calculate the end point of the orientation line (arrow)
    line_length = 10  # box_size[1] // 2
    line_length = min(max(velocity * line_length, line_length), line_length * 2)
    end_point = (int(center[0] - line_length * np.cos(orientation_rad + (np.pi / 2))),
                 int(center[1] - line_length * np.sin(orientation_rad + (np.pi / 2))))

    # Draw the orientation arrow
    cv2.arrowedLine(mask_arrow, center, end_point, (255), arrow_thick)  # Blue arrow

    # Display velocity
    # font = cv2.FONT_HERSHEY_SIMPLEX
    """cv2.putText(image, f'Vel: {velocity} m/s', (center[0], int(center[1] - box_size[1] // 2 - 10)),
                font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)  # Green text"""

    return mask_vehicle.astype(np.bool), mask_arrow.astype(np.bool)

def draw_label_raw(label_raw, name, lane_enable=True):
    print("#" * 100)
    print("DRAW_LABEL_RAW")
    image = np.zeros((200, 200, 3)).astype(np.uint8)
    mask_vehicle_image = np.zeros((200, 200)).astype(np.uint8)
    special_vehicle_image = np.zeros((200, 200)).astype(np.uint8)
    bike_and_cons_vehicle_image = np.zeros((200, 200)).astype(np.uint8)
    tl_image = np.zeros((200, 200)).astype(np.uint8)
    lane_guidance_mask = np.zeros((200, 200)).astype(np.uint8)
    ego_mask = np.zeros((200, 200)).astype(np.uint8)
    # ego_front_mask = np.zeros((200, 200)).astype(np.uint8)
    tl_light_stop = False
    for index, sample in enumerate(label_raw):
        center = np.array(sample['position'])
        if name == 'detection':
            center = np.array([center[0], center[1]])
        center *= (-1)
        center = center * 4 + 100

        if sample['class'] == 'Route':
            sample['speed'] = 0

        if sample['class'] == 'Car':

            bbox = np.array(sample["extent"])

            mask_vehicle, mask_arrow = draw_vehicle(center, sample['yaw'], sample['speed'],
                                                         (bbox[0], bbox[1]), bbox)

            if index == 0:
                ego_mask = copy.deepcopy(mask_vehicle)
                ego_front_mask, _ = draw_vehicle(np.array([86, 100]), sample['yaw'],
                                                      sample['speed'],
                                                      (bbox[0] / 2, bbox[1] / 2), bbox / 2)

                image[mask_vehicle] = (255, 255, 255)
                image[mask_arrow] = (255, 0, 0)
                tl_ego_image = copy.deepcopy(mask_vehicle)

            else:
                image[mask_vehicle] = (0, 0, 255)
                image[mask_arrow] = (255, 0, 0)
                _, mask_arrow = draw_vehicle(center, sample['yaw'], sample['speed'],
                                                  (bbox[0], bbox[1]), bbox)
                mask_vehicle_image[mask_vehicle] = (255)
                mask_vehicle_image[mask_arrow] = (255)

        elif sample['class'] == 'Police' or sample['class'] == 'Firetruck' or sample['class'] == 'Crossbike' or \
                sample['class'] == 'Construction' or sample['class'] == 'Ambulance' or sample[
            'class'] == 'Walker' or sample['class'] == 'Route':

            bbox = np.array(sample["extent"])

            if sample['class'] == 'Crossbike':
                bbox = [2.0, 2.0, 2.0]

            mask_vehicle, mask_arrow = draw_vehicle(center, sample['yaw'], sample['speed'],
                                                         (bbox[0], bbox[1]), bbox)

            if index == 0:
                image[mask_vehicle] = (255, 255, 255)
                image[mask_arrow] = (0, 255, 0)
                tl_ego_image = copy.deepcopy(mask_vehicle)
            else:
                image[mask_vehicle] = (0, 255, 0)
                image[mask_arrow] = (0, 0, 255)
                mask_vehicle_image[mask_vehicle] = (255)
                special_vehicle_image[mask_vehicle] = (255)
                special_vehicle_image[mask_arrow] = (255)
                if sample['class'] == 'Crossbike' or sample['class'] == 'Construction' or sample[
                    'class'] == 'Walker':
                    bike_and_cons_vehicle_image[mask_vehicle] = (255)
                    bike_and_cons_vehicle_image[mask_arrow] = (255)

                _, mask_arrow = draw_vehicle(center, sample['yaw'], sample['speed'],
                                                  (bbox[0], bbox[1]), bbox)
                mask_vehicle_image[mask_arrow] = (255)

        elif sample['class'] == 'Radar':
            bbox = np.array(sample["extent"])

            mask_vehicle, mask_arrow = draw_vehicle(center, sample['yaw'], sample['speed'],
                                                         (bbox[0], bbox[1]), bbox)

            image[mask_vehicle] = (255, 0, 0)

        elif sample['class'] == 'lane_guidance':
            bbox = np.array(sample["extent"])
            position_center = np.array(sample['position'])
            position_center = position_center * 4 + 100

            mask_lane, mask_arrow_lane = draw_vehicle(position_center, sample['yaw'], 0.0,
                                                           (bbox[0], bbox[1]), bbox)

            lane_guidance_mask += mask_lane
            image[mask_lane] = (255, 255, 255)

        elif sample['class'] == "Lane" and lane_enable:
            bbox = np.array(sample["extent"])
            # position_center = np.array(sample['position'])
            # position_center = position_center * 4 + 100

            mask_lane, mask_arrow_lane = draw_vehicle(center, sample['yaw'], 0.0,
                                                           (bbox[0], bbox[1]), bbox)

            lane_guidance_mask += mask_lane
            image[mask_lane] = (255, 255, 0)


        elif sample['class'] == 'tl_bev_pixel' or sample['class'] == 'Stop_sign':
            bbox = np.array(sample["extent"])

            mask = plot_bounding_box_center(center, bbox[0], bbox[1]).astype(np.bool)

            if sample['state'] == 2:
                image[mask] = (0, 255, 0)
                if sample['class'] == 'tl_bev_pixel':
                    tl_image = copy.deepcopy(mask)
            elif sample['state'] == 1:
                image[mask] = (0, 255, 255)
                if sample['class'] == 'tl_bev_pixel':
                    tl_image = copy.deepcopy(mask)
            elif sample['state'] == 0:
                image[mask] = (0, 0, 255)

        elif sample['class'] == "Lane_guidance_wp":
            bbox = np.array(sample["extent"])

            mask = plot_bounding_box_center(center, bbox[0], bbox[1]).astype(np.bool)

            image[mask] = (255, 255, 255)

        if sample['class'] == 'tl_bev_pixel':

            bbox = np.array(sample["extent"])

            mask = plot_bounding_box_center(center, bbox[0], bbox[1]).astype(np.bool)
            if np.linalg.norm(np.array(sample['tl_bev_pixel_coordinate']).mean(0)) < 25:
                if sample['state'] == 1:
                    tl_light_stop = True

                    image[mask] = (0, 255, 255)
                    if sample['class'] == 'tl_bev_pixel':
                        tl_image = copy.deepcopy(mask)
                elif sample['state'] == 0:
                    tl_light_stop = True

                    image[mask] = (0, 0, 255)


    return image

path = "/workspace/tg22/updated_lanes/Updated_lanes_Town12_radar_AccidentTwoWays_3__routeNone_04_27_03_07_51/"
old_path = "/workspace/tg22/updated_lanes/Town12_radar_AccidentTwoWays_3__routeNone_04_27_03_07_51/boxes/"


assert os.path.exists(path)

boxes_list = os.listdir(path+"boxes/")

for boxes in boxes_list:
    number = boxes.split('.')[0]
    box_path = path + "boxes/" + boxes
    with gzip.open(box_path, 'rt', encoding='utf-8') as f:
        content = json.load(f)
    current_image = draw_label_raw(content, name="detection")
    cv2.imwrite("current_image.png",current_image)

    with gzip.open(old_path + boxes, 'rt', encoding='utf-8') as f:
        other_content = json.load(f)
    old_image = draw_label_raw(other_content, name="detection", lane_enable=True)
    cv2.imwrite("old_image.png", old_image)

    other_image = draw_label_raw(other_content, name="detection",lane_enable=False)
    new_image = current_image + other_image
    cv2.imwrite("new_image.png", new_image)
    asd = 0
asd = 0