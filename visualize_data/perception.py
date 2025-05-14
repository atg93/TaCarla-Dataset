import os

import cv2
import numpy as np
import json
new_camera = {'type': 'sensor.camera.rgb', 'x': 0.70079118954, 'y': 0.0159456324149, 'z': 1.51095763913,
                      'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'fov': 70, 'id': 'front'}
from PIL import Image
import carla

def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def get_related_content(content_1, current_npc_location):
    for sample in content_1:
        if sample["class"] == "Car" and (np.array([current_npc_location.x,current_npc_location.y,current_npc_location.z])  == sample["world_location"]).all():
            return sample
    return None

def new_visualize(image_1, content_1, fov):
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

    image_h_1, image_w_1, _ = image_1.shape
    K_1 = build_projection_matrix(image_w_1, image_h_1, fov)
    K_b_1 = build_projection_matrix(image_w_1, image_h_1, fov, is_behind_camera=True)
    img_1 = np.reshape(np.copy(image_1), (image_h_1, image_w_1, 3))

    ego = content_1[0]
    ego_location = carla.Location(x=ego["world_location"][0], y=ego["world_location"][1], z=ego["world_location"][2])
    ego_rotation = carla.Rotation(pitch=ego["vehicle_rotation"][0], roll=ego["vehicle_rotation"][1], yaw=ego["vehicle_rotation"][2])
    ego_transform = carla.Transform(location=ego_location, rotation=ego_rotation)

    camera_relative_transform = carla.Transform(
        carla.Location(x=new_camera["x"], y=new_camera["y"], z=new_camera["z"]),  # Relative position
        carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0)  # Relative rotation
    )

    # Step 3: Convert relative location to world using ego transform
    camera_world_location = ego_transform.transform(camera_relative_transform.location)

    # Step 4 (optional): Camera rotation in world = ego + camera relative
    camera_world_rotation = carla.Rotation(
        pitch=ego_transform.rotation.pitch + camera_relative_transform.rotation.pitch,
        yaw=ego_transform.rotation.yaw + camera_relative_transform.rotation.yaw,
        roll=ego_transform.rotation.roll + camera_relative_transform.rotation.roll
    )

    # Step 5: Combine into camera world transform
    camera_transform_1 = carla.Transform(camera_world_location, camera_world_rotation)

    # Get the camera matrix
    world_2_camera_1 = np.array(camera_transform_1.get_inverse_matrix())

    # Get the camera matrix
    #world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    #world_2_camera_1 = world_2_camera
    for index, npc_1 in enumerate(content_1):


        # Filter out the ego vehicle
        if (npc_1["class"]=="Car" or npc_1["class"]=='Walker' or npc_1["class"]=='Crossbike') and index!=0:
            #current_npc_location = npc.get_location()

            #npc_1 = get_related_content(content_1, current_npc_location)
            if not npc_1:
                continue
            bb_location_1 = carla.Location(x=npc_1['bounding_box_location'][0], y=npc_1['bounding_box_location'][1], z=npc_1['bounding_box_location'][2])
            bb_rotation_1 = carla.Rotation(pitch=npc_1['bounding_box_rotation'][0],roll=npc_1['bounding_box_rotation'][1],yaw=npc_1['bounding_box_rotation'][2])
            bb_1 = carla.BoundingBox()
            bb_1.rotation = bb_rotation_1
            bb_1.extent.x = npc_1["extent"][1]/2
            bb_1.extent.y = npc_1["extent"][0]/2
            bb_1.extent.z = npc_1["extent"][0]/2
            bb_1.location = bb_location_1
            npc_location = carla.Location(x=npc_1['world_location'][0], y=npc_1['world_location'][1], z=npc_1['world_location'][2])
            npc_transform = carla.Transform(location=npc_location,rotation=carla.Rotation(pitch=npc_1["vehicle_rotation"][0],roll=npc_1["vehicle_rotation"][1],yaw=npc_1["vehicle_rotation"][2]))
            #bb.location = carla.Location(x=0, y=0, z=0)
            # Filter for the vehicles within 50m
            dist = npc_location.distance(ego_location)
            if dist < 50:

            # Calculate the dot product between the forward vector
            # of the vehicle and the vector between the vehicle
            # and the other vehicle. We threshold this dot product
            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                #forward_vec = vehicle.get_transform().get_forward_vector()
                forward_vec_1 = ego_transform.get_forward_vector()
                #ray = npc.get_transform().location - vehicle.get_transform().location
                ray_1 = npc_transform.location - ego_transform.location

                if forward_vec_1.dot(ray_1) > 0:
                    #verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    verts_0 = [v for v in bb_1.get_world_vertices(npc_transform)]
                    for edge in edges:
                        #p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                        #p2 = get_image_point(verts[edge[1]],  K, world_2_camera)

                        p1_1 = get_image_point(verts_0[edge[0]], K_1, world_2_camera_1)
                        p2_1 = get_image_point(verts_0[edge[1]],  K_1, world_2_camera_1)

                        #p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                        #p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                        p1_in_canvas_1 = point_in_canvas(p1_1, image_h_1, image_w_1)
                        p2_in_canvas_1 = point_in_canvas(p2_1, image_h_1, image_w_1)

                        if not p1_in_canvas_1 and not p2_in_canvas_1:
                            continue
                        #vehicle.get_location().x + 0.70079118954, vehicle.get_location().y + 0.0159456324149, vehicle.get_location().z + 1.51095763913
                        #ray0_0 = verts[edge[0]] - camera.get_transform().location
                        #ray1_0 = verts[edge[1]] - camera.get_transform().location
                        #cam_forward_vec_0 = camera.get_transform().get_forward_vector()



                        # Get world location of the camera
                        camera_world_location = camera_transform_1.location

                        ray0 = verts_0[edge[0]] - camera_world_location
                        ray1 = verts_0[edge[1]] - camera_world_location
                        cam_forward_vec = camera_transform_1.get_forward_vector() #camera.get_transform().get_forward_vector()

                        # One of the vertex is behind the camera
                        if not (cam_forward_vec.dot(ray0) > 0):
                            p1_1 = get_image_point(verts_0[edge[0]], K_b_1, world_2_camera_1)
                        if not (cam_forward_vec.dot(ray1) > 0):
                            p2_1 = get_image_point(verts_0[edge[1]], K_b_1, world_2_camera_1)

                        #cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)
                        cv2.line(img_1, (int(p1_1[0]),int(p1_1[1])), (int(p2_1[0]),int(p2_1[1])), (255,0,0, 255), 1)

    #cv2.imwrite('ImageWindowName.png',img)
    cv2.imwrite('ImageWindowName_1.png',img_1)
    asd = 0


import sys
sys.path.append(
    '/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/tairvision_center_line/')

sys.path.append(
    '/cta/users/tgorgulu/git/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/perception/tairvision_center_line/')
import torch
from tairvision.models.bev.common.utils.geometry import calculate_birds_eye_view_parameters
from tairvision.datasets.nuscenes import get_view_matrix

def calculate_view_matrix(X_BOUND, Y_BOUND, Z_BOUND):


    bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
        X_BOUND, Y_BOUND, Z_BOUND
    )

    bev_resolution, bev_start_position, bev_dimension = (
        bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
    )

    lidar_to_view = get_view_matrix(
        bev_dimension,
        bev_resolution,
        bev_start_position=bev_start_position,
        ego_pos='center'
    )

    view = lidar_to_view[None, None, None]
    view_tensor = torch.tensor(view, dtype=torch.float32)

    return view_tensor


def map2bev(tl_outputs):
    centerlines = tl_outputs['lane_centerline_list_pred']  # compass
    bev_image = np.zeros((586,1034)).astype(np.uint8)

    if len(tl_outputs['pred_list'][0]) != 0:
        tl_outputs['pred_list'], tl_outputs['lane_centerline_list_pred'], tl_outputs[
            'score_list_lcte'], np.argmax(tl_outputs['score_list_lcte'][0])
    color = carla.Color(r=0, g=0, b=255, a=0)
    thickness = 0.5
    life_time = 0.2
    stop_label = False

    tl_lights_masks = np.zeros([_width, _width], dtype=np.uint8)
    mid_point = int(_width / 2)
    tl_lights_masks = draw_from_center(tl_lights_masks, np.array([mid_point, mid_point]), 2, 2)
    tl_bbox = []
    box_size = (0,0,0)
    for tl_index, cl in enumerate(centerlines[0]):
        line_thickness = 2
        stop_masks = np.zeros([_width, _width], dtype=np.uint8)
        line_color, tl_state, score_list_lcte, pred_list = get_tl_color(tl_outputs, tl_index)


        if tl_state != -1:
            tl_box_mid_point = np.array([cl['points'][:, 0].mean(), cl['points'][:, 1].mean(), cl['points'][:, 2].mean()])
            tl_bbox = tl_box_mid_point
            new_tl_box_mid_point = tl_box_mid_point + mid_point
            box_size = abs((cl['points'][0] - cl['points'][-1]).astype(np.int)) + 2
            tl_lights_masks = draw_from_center(tl_lights_masks, new_tl_box_mid_point,box_size[0],box_size[1])


        tl_lights_masks = crop_array_center(tl_lights_masks, real_height,
                                                 real_width)  # cv2.resize(tl_lights_masks, (self.real_width, self.real_height))#


        stop_masks = crop_array_center(stop_masks, real_height,
                                            real_width)  # cv2.resize(tl_lights_masks, (self.real_width, self.real_height))#


    return tl_lights_masks, tl_state, score_list_lcte, pred_list, tl_bbox, box_size

import math
asd = 0


def visualize_tl(img, _intrinsic, _cam_to_lidar):
    frame_dict = {"CAM_FRONT": img}
    intrinsic_dict = {"CAM_FRONT": _intrinsic["front"]}
    cam_to_lidar_dict = {"CAM_FRONT": _cam_to_lidar["front"]}

    outputs, image = tl_inference.predict(frame_dict, intrinsic_dict, cam_to_lidar_dict)

    tl_bev_image, tl_state, score_list_lcte, pred_list, tl_bbox, box_size = map2bev(outputs['metrics'])

    def find_intrinsics(self, width, height, fov, x, y, z, roll, pitch, yaw, use_nu = True):
        if use_nu:
            roll = -math.radians(roll)
            y = -y
            pitch = math.radians(pitch)
            yaw = -math.radians(yaw)
            nu_matrix = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

        else:
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)
        fx = (width / 2.0) / math.tan(math.radians(fov / 2.0))
        cx = width / 2.0
        cy = height / 2.0

        ## Intrinsic matrix K
        K = np.array([[fx, 0, cx],
                      [0, fx, cy],
                      [0, 0, 1]])

        # Translation vector T
        T = np.array([x, y, z])

        # Rotation matrices for roll, pitch, and yaw
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]])

        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                       [0, 1, 0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])

        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]])

        # Combine the rotation matrices

        R = np.dot(Rz, np.dot(Ry, Rx))
        if use_nu:
            R_nu = R @ nu_matrix
        else:
            R_nu = R

        # Extrinsic matrix RT
        RT = np.zeros((4, 4))
        RT[:3, :3] = R_nu
        RT[:3, 3] = T
        RT[3, 3] = 1

        return torch.tensor(K), torch.tensor(RT)

def visualize_two_d_light(image, content_1):
    for index, npc_1 in enumerate(content_1):


        # Filter out the ego vehicle
        if npc_1["class"]=="two_d_light_p":
            for edge in npc_1["two_d_light_p"]:
                edge_flatten = np.array(edge).flatten()
                if npc_1["state"] == 0:
                    color = (0, 0, 255, 255)
                elif npc_1["state"] == 1:
                    color = (0, 255, 255, 255)
                elif npc_1["state"] == 2:
                    color = (0, 255, 0, 255)
                cv2.line(image, (int(edge_flatten[0]), int(edge_flatten[1])), (int(edge_flatten[2]), int(edge_flatten[3])), color, 1)
    cv2.imwrite("tl_image.png",image)
    asd = 0


if __name__ == '__main__':
    import gzip
    import json

    from tairvision.models.bev.lss_mask2former.inference import OpenLaneV2InferenceInterface

    data_path = "/media/hdd/carla_data_collection/leaderboard_plant_pdm_Town13/"

    folder_list = os.listdir(data_path)

    _width = 1845
    real_height = 200  # 586
    real_width = 200  # 1034

    X_BOUND = [-54.0, 54.0, 0.0542]  # Forward
    Y_BOUND = [-54.0, 54.0, 0.0542]  # Sides
    Z_BOUND = [-10.0, 10.0, 5.0]
    project_view = calculate_view_matrix(X_BOUND, Y_BOUND, Z_BOUND)

    tl_inference = OpenLaneV2InferenceInterface(visualize=False)
    #_intrinsic =
    #_cam_to_lidar =



    for main_index, folder in enumerate(folder_list):
        if folder.split("_")[2] != "ParkingCrossingPedestrian":
            continue

        box_path = data_path + folder + "/boxes/"
        detection_path = data_path + folder + "/detection/rgb_camera/front"
        box_list = os.listdir(box_path)
        lidar_path = data_path + folder + "/detection/lidar/"

        if main_index in [0, 1, 2, 4]:
            continue
        for sample_index, box in enumerate(box_list):
            current_box = box_path + box
            with gzip.open(box_path+str(sample_index).zfill(4)+".json.gz", 'rt', encoding='utf-8') as gz_file:
                content = json.load(gz_file)

            """lidar_sample_path = lidar_path + "lidar_" + str(sample_index) +".laz"
            assert os.path.exists(lidar_sample_path)

            import laspy

            # Path to your .laz file
            laz_path = lidar_sample_path

            # Read the file
            las = laspy.read(laz_path)
            las["float_intensity"]"""
            if sample_index != 0:
                image = Image.open(detection_path+"/front_"+str(sample_index)+"_.jpg")
                image = np.array(image)
                new_visualize(image, content, fov=70)
                visualize_two_d_light(image, content)
                asd = 0 #"two_d_light_p"
