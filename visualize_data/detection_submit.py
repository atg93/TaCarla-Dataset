import copy
import os
from PIL import Image
import carla
import cv2
import numpy as np
import json
"""new_camera = {'type': 'sensor.camera.rgb', 'x': 0.70079118954, 'y': 0.0159456324149, 'z': 1.51095763913,
                      'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'fov': 70, 'id': 'front'}"""

new_camera_dict = {}
w = 1600  # 704
h = 900  # 396
# Add cameras
new_camera = {'type': 'sensor.camera.rgb', 'x': 0.70079118954, 'y': 0.0159456324149, 'z': 1.51095763913,
              'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
              'width': w, 'height': h, 'fov': 70, 'id': 'front'}
new_camera_dict.update({new_camera["id"]:new_camera})

new_camera = {'type': 'sensor.camera.rgb', 'x': 0.5508477543, 'y': 0.493404796419, 'z': 1.49574800619,
              'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
              'width': w, 'height': h, 'fov': 70, 'id': 'front_right'}
new_camera_dict.update({new_camera["id"]:new_camera})

new_camera = {'type': 'sensor.camera.rgb', 'x': 0.52387798135, 'y': -0.494631336551, 'z': 1.50932822144,
              'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
              'width': w, 'height': h, 'fov': 70, 'id': 'front_left'}
new_camera_dict.update({new_camera["id"]:new_camera})

new_camera = {'type': 'sensor.camera.rgb', 'x': -1.5283260309358, 'y': 0.00345136761476, 'z': 1.57910346144,
              'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
              'width': w, 'height': h, 'fov': 110, 'id': 'back'}
new_camera_dict.update({new_camera["id"]:new_camera})

new_camera = {'type': 'sensor.camera.rgb', 'x': -0.53569100218, 'y': -0.484795032713, 'z': 1.59097014818,
              'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
              'width': w, 'height': h, 'fov': 70, 'id': 'back_left'}
new_camera_dict.update({new_camera["id"]:new_camera})

new_camera = {'type': 'sensor.camera.rgb', 'x': -0.5148780988, 'y': 0.480568219723, 'z': 1.56239545128,
              'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
              'width': w, 'height': h, 'fov': 70, 'id': 'back_right'}
new_camera_dict.update({new_camera["id"]:new_camera})

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

def new_visualize(image_1, content_1, camera_name):
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
    fov = new_camera_dict[camera_name]["fov"]
    image_h_1, image_w_1, _ = image_1.shape
    K_1 = build_projection_matrix(image_w_1, image_h_1, fov)
    K_b_1 = build_projection_matrix(image_w_1, image_h_1, fov, is_behind_camera=False)
    img_1 = np.reshape(np.copy(image_1), (image_h_1, image_w_1, 3))

    ego = content_1[0]
    ego_location = carla.Location(x=ego["world_location"][0], y=ego["world_location"][1], z=ego["world_location"][2])
    ego_rotation = carla.Rotation(pitch=ego["vehicle_rotation"][0], roll=ego["vehicle_rotation"][1], yaw=ego["vehicle_rotation"][2])
    ego_transform = carla.Transform(location=ego_location, rotation=ego_rotation)

    camera_relative_transform = carla.Transform(
        carla.Location(x=new_camera_dict[camera_name]["x"], y=new_camera_dict[camera_name]["y"], z=new_camera_dict[camera_name]["z"]),  # Relative position
        carla.Rotation(roll=new_camera_dict[camera_name]["roll"], pitch=new_camera_dict[camera_name]["pitch"], yaw=new_camera_dict[camera_name]["yaw"])  # Relative rotation
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

                if ("front" in camera_name.split('_') and forward_vec_1.dot(ray_1) > 0) or ("back" in camera_name.split('_') and forward_vec_1.dot(ray_1) < 0):
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
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR)
    cv2.imwrite(camera_name+ ".png",img_1)
    asd = 0

def draw_lidar(lidar,lidar_range=200):
    image_size = 400

    # Convert to 2D by ignoring z (for BEV image)
    points_2d = (lidar[:,:2] * (image_size/lidar_range)).astype(int) * 2

    # Create an empty image - 100x100 pixels
    image = np.zeros((image_size, image_size))

    # Scaling factor to fit points in the image dimensions
    new_points_2d = points_2d + 200

    # Populate the image with LiDAR points
    for point in new_points_2d:
        try:
            image[point[1], point[0]] = 1  # Increment to simulate intensity (simplistic approach)
        except:
            pass  # Increment to simulate intensity (simplistic approach)

    # Normalize image to have values between 0 and 255
    #image -= image.min()
    #image = (image / image.max()) * 255
    image[image>0] = 255
    cv2.imwrite('lidar.png', image)



def draw_tl(front_image, content):
    #front_image = copy.deepcopy(front_image)
    #front_image = cv2.cvtColor(front_image, cv2.COLOR_RGB2BGR)
    for boxes in content:
        if 'two_d_light_p' == boxes['class']:
            for edge in boxes['two_d_light_p']:
                p1_0, p1_1, p2_0, p2_1 = np.array(edge).flatten()
                cv2.line(front_image, (p1_0, p1_1), (p2_0, p2_1), (255, 0, 0, 255), 1)

    #front_image = cv2.cvtColor(front_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("tl_image.png",front_image)
    return front_image
    asd = 0

def inverse_get_relative_transform(relative_pos, ego_matrix):
    """
    return the relative transform from ego_pose to vehicle pose
    """
    rot = np.eye(3)
    relative_pos = rot @ (relative_pos + np.array([1.3, 0.0, 2.5]))

    # transform from right handed system
    relative_pos[1] = - relative_pos[1]

    ###
    rot = ego_matrix[:3, :3].T
    rot = np.linalg.inv(rot)
    relative_pos = rot @ relative_pos

    position = relative_pos + ego_matrix[:3, 3]

    return position
def draw_lane(content_1, image_1, camera_name, meas_content):
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
    fov = new_camera_dict[camera_name]["fov"]
    image_h_1, image_w_1, _ = image_1.shape
    K_1 = build_projection_matrix(image_w_1, image_h_1, fov)
    K_b_1 = build_projection_matrix(image_w_1, image_h_1, fov, is_behind_camera=False)
    img_1 = np.reshape(np.copy(image_1), (image_h_1, image_w_1, 3))

    ego = content_1[0]
    ego_location = carla.Location(x=ego["world_location"][0], y=ego["world_location"][1], z=ego["world_location"][2])
    ego_rotation = carla.Rotation(pitch=ego["vehicle_rotation"][0], roll=ego["vehicle_rotation"][1], yaw=ego["vehicle_rotation"][2])
    ego_transform = carla.Transform(location=ego_location, rotation=ego_rotation)

    camera_relative_transform = carla.Transform(
        carla.Location(x=new_camera_dict[camera_name]["x"], y=new_camera_dict[camera_name]["y"], z=new_camera_dict[camera_name]["z"]),  # Relative position
        carla.Rotation(roll=new_camera_dict[camera_name]["roll"], pitch=new_camera_dict[camera_name]["pitch"], yaw=new_camera_dict[camera_name]["yaw"])  # Relative rotation
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
    for sample in content_1:
        if sample["class"] == "Lane":
            world_coordinate = inverse_get_relative_transform(sample["position"], np.array(meas_content['ego_matrix']))


            world_coordinate = carla.Location(x=world_coordinate[0],y=world_coordinate[1],z=world_coordinate[2])
            world_rotation = ego_rotation

            bb_1 = carla.BoundingBox()
            bb_1.rotation = ego_rotation
            bb_1.extent.x = 1.0
            bb_1.extent.y = 1.0
            bb_1.extent.z = 1.0
            bb_1.location = world_coordinate

            wp_transform = carla.Transform(location=world_coordinate,rotation=world_rotation)
            dist = world_coordinate.distance(ego_location)
            if dist < 50:

                # Calculate the dot product between the forward vector
                # of the vehicle and the vector between the vehicle
                # and the other vehicle. We threshold this dot product
                # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                # forward_vec = vehicle.get_transform().get_forward_vector()
                forward_vec_1 = ego_transform.get_forward_vector()
                # ray = npc.get_transform().location - vehicle.get_transform().location
                ray_1 = wp_transform.location - ego_transform.location

                if ("front" in camera_name.split('_') and forward_vec_1.dot(ray_1) > 0) or (
                        "back" in camera_name.split('_') and forward_vec_1.dot(ray_1) < 0):
                    # verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    verts_0 = [v for v in bb_1.get_world_vertices(wp_transform)]
                    for edge in edges:
                        # p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                        # p2 = get_image_point(verts[edge[1]],  K, world_2_camera)

                        p1_1 = get_image_point(verts_0[edge[0]], K_1, world_2_camera_1)
                        p2_1 = get_image_point(verts_0[edge[1]], K_1, world_2_camera_1)

                        # p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                        # p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                        p1_in_canvas_1 = point_in_canvas(p1_1, image_h_1, image_w_1)
                        p2_in_canvas_1 = point_in_canvas(p2_1, image_h_1, image_w_1)

                        if not p1_in_canvas_1 and not p2_in_canvas_1:
                            continue
                        # vehicle.get_location().x + 0.70079118954, vehicle.get_location().y + 0.0159456324149, vehicle.get_location().z + 1.51095763913
                        # ray0_0 = verts[edge[0]] - camera.get_transform().location
                        # ray1_0 = verts[edge[1]] - camera.get_transform().location
                        # cam_forward_vec_0 = camera.get_transform().get_forward_vector()

                        # Get world location of the camera
                        camera_world_location = camera_transform_1.location

                        ray0 = verts_0[edge[0]] - camera_world_location
                        ray1 = verts_0[edge[1]] - camera_world_location
                        cam_forward_vec = camera_transform_1.get_forward_vector()  # camera.get_transform().get_forward_vector()

                        # One of the vertex is behind the camera
                        if not (cam_forward_vec.dot(ray0) > 0):
                            p1_1 = get_image_point(verts_0[edge[0]], K_b_1, world_2_camera_1)
                        if not (cam_forward_vec.dot(ray1) > 0):
                            p2_1 = get_image_point(verts_0[edge[1]], K_b_1, world_2_camera_1)

                        # cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)
                        cv2.line(img_1, (int(p1_1[0]), int(p1_1[1])), (int(p2_1[0]), int(p2_1[1])), (255, 0, 0, 255), 1)

    cv2.imwrite("Lane.png",img_1)
    asd = 0




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
    #cv2.imwrite('mask_vehicle.png', mask_vehicle)

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

def plot_bounding_box_center(center, width=4, height=8):
    mask = np.zeros((200, 200)).astype(np.uint8)
    # Calculate the top-left corner from the center, width, and height
    top_left = (int(center[1] - width / 2), int(center[0] - height / 2))
    bottom_right = (int(center[1] + width / 2), int(center[0] + height / 2))

    # Draw the rectangle (bounding box)
    cv2.rectangle(mask, bottom_right, top_left, (255), 2)  # Blue box

    return mask
    
def draw_label_raw(label_raw, name):
    image = np.zeros((200, 200, 3)).astype(np.uint8)
    mask_vehicle_image = np.zeros((200, 200)).astype(np.uint8)
    special_vehicle_image = np.zeros((200, 200)).astype(np.uint8)
    bike_and_cons_vehicle_image = np.zeros((200, 200)).astype(np.uint8)
    tl_image = np.zeros((200, 200)).astype(np.uint8)
    lane_guidance_mask = np.zeros((200, 200)).astype(np.uint8)
    ego_mask = np.zeros((200, 200)).astype(np.uint8)
    #ego_front_mask = np.zeros((200, 200)).astype(np.uint8)
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
            'class'] == 'Walker':# or sample['class'] == 'Route':

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

        elif sample['class'] == "Lane":
            bbox = np.array(sample["extent"])
            #position_center = np.array(sample['position'])
            #position_center = position_center * 4 + 100

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


    cv2.imwrite("label_image.png", image)
    return image

if __name__ == '__main__':
    import gzip
    import json

    #data_path = "/media/hdd/carla_data_collection/leaderboard_plant_pdm_debug/"
    data_path = "//workspace/tg22/detection_debug/original_data/"

    folder_list = os.listdir(data_path)
    #camera_name = "front"
    camera_name_list = ["back", "back_left", "back_right", "front", "front_left", "front_right"]
    for main_index, folder in enumerate(folder_list):

        box_path = data_path + '/' + folder + "/boxes/"
        meas_path = data_path + '/' + folder + "/measurements/"
        box_list = os.listdir(box_path)

        for sample_index, box in enumerate(box_list):
            if sample_index == 33:
                asd = 0
            current_box = box_path + box
            with gzip.open(box_path+str(sample_index).zfill(4)+".json.gz", 'rt', encoding='utf-8') as gz_file:
                content = json.load(gz_file)

            with open(meas_path+str(sample_index).zfill(4)+".json", 'r') as gz_file:
                meas_content = json.load(gz_file)

            draw_label_raw(content,"detection")

            lidar_path = data_path + '/' + folder + "/detection/lidar/"

            lidar_sample_path = lidar_path + "lidar_" + str(sample_index) + ".laz"
            assert os.path.exists(lidar_sample_path)

            import laspy

            # Path to your .laz file
            laz_path = lidar_sample_path

            # Read the file
            las = laspy.read(laz_path)
            draw_lidar(las.xyz)


            for camera_name in camera_name_list:
                detection_path = data_path + '/' + folder + "/detection/rgb_camera/" + camera_name



                # if folder.split("_")[2] != "ParkingCrossingPedestrian":
                #    continue


                #las["float_intensity"]
                if sample_index != 0:
                    #image = Image.open(detection_path+"/front_"+str(sample_index)+"_.jpg")
                    image = Image.open(detection_path+"/" + camera_name + "_"+str(sample_index)+"_.jpg")
                    image = np.array(image)
                    if camera_name == "front":
                        image = draw_tl(image, content)
                        #draw_lane(content, image, camera_name, meas_content)

                    new_visualize(image, content, camera_name)
                    asd = 0

