import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import os

import cv2
import numpy as np
import json
new_camera = {'type': 'sensor.camera.rgb', 'x': 0.70079118954, 'y': 0.0159456324149, 'z': 1.51095763913,
                      'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'fov': 70, 'id': 'front'}
from PIL import Image
import carla

# --- STEP 0: MOCK DATA -- Replace with real values from CARLA logs or JSON ---
image_path = "camera_image.png"
image_width, image_height, fov = 800, 600, 90  # example FOV
extent = {'x': 1.2, 'y': 0.6, 'z': 1.0}  # half-dimensions of the vehicle bbox
bbox_location = {'x': 0.0, 'y': 0.0, 'z': 1.0}  # relative to vehicle center

# Example vehicle world transform
vehicle_transform = {
    'location': {'x': 10.0, 'y': 5.0, 'z': 0.0},
    'rotation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0}
}

# Example camera world transform
camera_transform = {
    'location': {'x': 8.0, 'y': 5.0, 'z': 2.0},
    'rotation': {'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0}
}

# --- STEP 1: Generate 3D BBox corners in vehicle local space ---
def get_3d_bbox_points(extent):
    x, y, z = extent['x']/2, extent['y']/2, extent['z']/2
    corners = np.array([
        [ x,  y, -z],
        [ x, -y, -z],
        [-x, -y, -z],
        [-x,  y, -z],
        [ x,  y,  z],
        [ x, -y,  z],
        [-x, -y,  z],
        [-x,  y,  z]
    ])
    return corners.T  # shape (3, 8)

# --- STEP 2: Vehicle → World ---
def transform_vehicle_to_world(corners, vehicle_transform, bbox_location):
    veh_pos = np.array([vehicle_transform['location'][k] for k in ['x', 'y', 'z']])
    bbox_offset = np.array([bbox_location[k] for k in ['x', 'y', 'z']])
    total_translation = veh_pos + bbox_offset
    rot = R.from_euler('xyz', [np.deg2rad(vehicle_transform['rotation'][k]) for k in ['roll', 'pitch', 'yaw']])
    return rot.apply(corners.T) + total_translation  # shape (8, 3)

# --- STEP 3: World → Camera ---
def transform_world_to_camera(world_pts, camera_transform):
    cam_pos = np.array([camera_transform['location'][k] for k in ['x', 'y', 'z']])
    rot = R.from_euler('xyz', [np.deg2rad(camera_transform['rotation'][k]) for k in ['roll', 'pitch', 'yaw']])
    R_world2cam = rot.as_matrix().T
    T_world2cam = -R_world2cam @ cam_pos
    return (R_world2cam @ world_pts.T + T_world2cam.reshape(3, 1))  # shape (3, 8)

# --- STEP 4: Camera → Image (project with intrinsics) ---
def get_intrinsic_matrix(w, h, fov):
    fx = fy = w / (2 * np.tan(fov * np.pi / 360))
    cx, cy = w / 2, h / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def project_to_image(pts_camera, K):
    pts_2d = K @ pts_camera
    pts_2d = pts_2d[:2] / pts_2d[2]
    return pts_2d.T  # shape (8, 2)

# --- STEP 5: Draw 3D box ---
def draw_3d_box(image, pts_2d):
    pts_2d = pts_2d.astype(int)
    for i, j in [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]:
        pt1, pt2 = tuple(pts_2d[i]), tuple(pts_2d[j])
        cv2.line(image, pt1, pt2, (0,255,0), 2)
    return image



if __name__ == '__main__':
    import gzip
    import json

    data_path = "/media/hdd/carla_data_collection/leaderboard_plant_pdm_Town13/"

    folder_list = os.listdir(data_path)

    for main_index, folder in enumerate(folder_list):
        box_path = data_path + folder + "/boxes/"
        detection_path = data_path + folder + "/detection/rgb_camera/front"
        box_list = os.listdir(box_path)
        if main_index in [0, 1, 2, 3]:
            continue
        for sample_index, box in enumerate(box_list):
            current_box = box_path + box
            with gzip.open(box_path+str(sample_index).zfill(4)+".json.gz", 'rt', encoding='utf-8') as gz_file:
                content = json.load(gz_file)

            if sample_index != 0:
                image = Image.open(detection_path+"/front_"+str(sample_index)+"_.jpg")
                image = np.array(image)
                #for object in content:
                #    if object["name"] == "Car" and
                # --- MAIN PIPELINE ---
                bbox_corners = get_3d_bbox_points(extent)
                world_corners = transform_vehicle_to_world(bbox_corners, vehicle_transform, bbox_location)
                camera_corners = transform_world_to_camera(world_corners, camera_transform)
                K = get_intrinsic_matrix(image_width, image_height, fov)
                image_corners = project_to_image(camera_corners, K)

                # Load image and draw
                image = cv2.imread(image_path)
                image = draw_3d_box(image, image_corners)
                cv2.imwrite("3D_Bounding_Box.png", image)
                asd = 0


