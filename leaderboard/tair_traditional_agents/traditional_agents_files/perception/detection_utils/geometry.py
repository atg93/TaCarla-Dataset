import numpy as np
import cv2
from pyquaternion import Quaternion
import math

def pad_images(input_image, padding_size=5, ones=False):
    h, w, c = input_image.shape

    padding_func = np.ones if ones else np.zeros
    vertical_padding = padding_func((padding_size, w, c)).astype(np.uint8) * 255
    horizontal_padding = padding_func((h + padding_size * 2, padding_size, c)).astype(np.uint8) * 255

    output_image = np.concatenate([input_image, vertical_padding], axis=0)
    output_image = np.concatenate([vertical_padding, output_image], axis=0)
    output_image = np.concatenate([output_image, horizontal_padding], axis=1)
    output_image = np.concatenate([horizontal_padding, output_image], axis=1)

    return output_image


# function to convert scale and zoom lss outputs to use them in roach
def scale_and_zoom(lss_output):
    output = []
    for image in lss_output:
        image = image[0]
        # Threshold the grayscale values to get binary image (1s and 0s)
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        zoomed_part = image[center_y - 50:center_y + 50, center_x - 50:center_x + 50]
        # Resize the zoomed part back to shape (200, 200)
        zoomed_image = cv2.resize(zoomed_part, (200, 200))
        output.append(zoomed_image.astype(np.uint8))

    return output

def zoom_lane(lane_bev):
    center_x, center_y = lane_bev.shape[1] // 2, lane_bev.shape[0] // 2
    zoomed_part = lane_bev[center_y - 50:center_y + 50, center_x - 50:center_x + 50]
    # Resize the zoomed part back to shape (200, 200)
    zoomed_lane_bev = cv2.resize(zoomed_part, (200, 200))
    zoomed_lane_bev = np.array(zoomed_lane_bev, dtype=bool)
    return zoomed_lane_bev


def get_pose_matrix(pose, use_flat=False, return_quaternion = False):
    rotation = Quaternion(pose['rotation'])
    if use_flat:
        yaw = rotation.yaw_pitch_roll[0]
        rotation = Quaternion(scalar=np.cos(yaw / 2),
                              vector=[0, 0, np.sin(yaw / 2)]
                              )
    rotation_matrix = rotation.rotation_matrix
    translation = np.array(pose['translation'])

    pose_matrix = np.vstack([
        np.hstack((rotation_matrix, translation[:, None])),
        np.array([0, 0, 0, 1])
    ])

    return pose_matrix


def euler_to_quaternion(roll, pitch, yaw):
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)

    qw = math.cos(roll_rad / 2) * math.cos(pitch_rad / 2) * math.cos(yaw_rad / 2) + \
         math.sin(roll_rad / 2) * math.sin(pitch_rad / 2) * math.sin(yaw_rad / 2)

    qx = math.sin(roll_rad / 2) * math.cos(pitch_rad / 2) * math.cos(yaw_rad / 2) - \
         math.cos(roll_rad / 2) * math.sin(pitch_rad / 2) * math.sin(yaw_rad / 2)

    qy = math.cos(roll_rad / 2) * math.sin(pitch_rad / 2) * math.cos(yaw_rad / 2) + \
         math.sin(roll_rad / 2) * math.cos(pitch_rad / 2) * math.sin(yaw_rad / 2)

    qz = math.cos(roll_rad / 2) * math.cos(pitch_rad / 2) * math.sin(yaw_rad / 2) - \
         math.sin(roll_rad / 2) * math.sin(pitch_rad / 2) * math.cos(yaw_rad / 2)

    return qw, qx, qy, qz