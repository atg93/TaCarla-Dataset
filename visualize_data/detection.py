import os

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

asd = 0

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




def mix_visualize(fov, image_1, content_1):
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
        if npc_1["class"]=="Car" and index!=0:
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
                        cam_forward_vec = carla.Vector3D(x=0.999625, y=0.021301, z=0.017200) #camera.get_transform().get_forward_vector()

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


def visualize(camera, image, world, vehicle, fov, camera_name):
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
    image_h, image_w, _ = image[1].shape
    K = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)
    img = np.reshape(np.copy(image[1]), (image_h, image_w, 4))

    # Get the camera matrix
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    for npc in world.get_actors().filter('*vehicle*'):

        # Filter out the ego vehicle
        if npc.id != vehicle.id:

            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(vehicle.get_transform().location)

            # Filter for the vehicles within 50m
            if dist < 50:

                # Calculate the dot product between the forward vector
                # of the vehicle and the vector between the vehicle
                # and the other vehicle. We threshold this dot product
                # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location

                if ("front" in camera_name.split('_') and forward_vec.dot(ray) > 0) or ("back" in camera_name.split('_') and forward_vec.dot(ray) < 0) :
                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    for edge in edges:
                        p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                        p2 = get_image_point(verts[edge[1]], K, world_2_camera)

                        p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                        p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                        if not p1_in_canvas and not p2_in_canvas:
                            continue

                        ray0 = verts[edge[0]] - camera.get_transform().location
                        ray1 = verts[edge[1]] - camera.get_transform().location
                        cam_forward_vec = camera.get_transform().get_forward_vector()

                        # One of the vertex is behind the camera
                        if not (cam_forward_vec.dot(ray0) > 0):
                            p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
                        if not (cam_forward_vec.dot(ray1) > 0):
                            p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)

                        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0, 255), 1)

    cv2.imwrite(camera_name + '.png',img)

def new_visualize_0(image, content, fov):
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
    image_h, image_w, _ = image.shape
    K = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)
    img = np.reshape(np.copy(image), (image_h, image_w, 3))

    # Get the camera matrix
    world_2_camera = np.array([[ 9.99437809e-01,  2.15849075e-02,  2.56553143e-02,
        -6.55074316e+03],
       [-2.15932559e-02,  9.99766886e-01,  4.84535667e-05,
        -4.11561182e+03],
       [-2.56482866e-02, -6.02408138e-04,  9.99670863e-01,
        -3.78358459e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])
    ego = content[0]
    ego_location = carla.Location(x=ego["world_location"][0], y=ego["world_location"][1], z=ego["world_location"][2])
    ego_rotation = carla.Rotation(pitch=ego["vehicle_rotation"][0], roll=ego["vehicle_rotation"][1], yaw=ego["vehicle_rotation"][2])
    ego_transform = carla.Transform(location=ego_location, rotation=ego_rotation)
    for index, npc in enumerate(content):


        # Filter out the ego vehicle
        if npc["class"]=="Car" and index!=0:
            bb_location = carla.Location(x=npc['bounding_box_location'][0], y=npc['bounding_box_location'][1], z=npc['bounding_box_location'][2])
            bb_rotation = carla.Rotation(pitch=npc['bounding_box_rotation'][0],roll=npc['bounding_box_rotation'][1],yaw=npc['bounding_box_rotation'][2])
            bb = carla.BoundingBox()
            bb.rotation = bb_rotation
            bb.extent.x = npc["extent"][1] / 2
            bb.extent.y = npc["extent"][0] / 2
            bb.extent.z = npc["extent"][0] / 2
            npc_location = carla.Location(x=npc['world_location'][0], y=npc['world_location'][1], z=npc['world_location'][2])
            npc_transform = carla.Transform(location=npc_location,rotation=carla.Rotation(pitch=npc["vehicle_rotation"][0],roll=npc["vehicle_rotation"][1],yaw=npc["vehicle_rotation"][2]))
            dist = npc_location.distance(ego_location)

            bb.location = carla.Location(x=0, y=0, z=0)
            # Filter for the vehicles within 50m
            if dist < 50:

            # Calculate the dot product between the forward vector
            # of the vehicle and the vector between the vehicle
            # and the other vehicle. We threshold this dot product
            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                forward_vec = ego_transform.get_forward_vector()
                ray = npc_transform.location - ego_transform.location

                if forward_vec.dot(ray) > 0:
                    verts = [v for v in bb.get_world_vertices(npc_transform)]
                    for edge in edges:
                        p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                        p2 = get_image_point(verts[edge[1]],  K, world_2_camera)

                        p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                        p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                        if not p1_in_canvas and not p2_in_canvas:
                            continue

                        new_camera_location = carla.Location(x=ego_location.x + 0.70079118954,y=ego_location.y + 0.0159456324149,z=ego_location.z + 1.51095763913)
                        ray0 = verts[edge[0]] - new_camera_location
                        ray1 = verts[edge[1]] - new_camera_location
                        cam_forward_vec = carla.Vector3D(x=0.999625, y=0.021301, z=0.017200) #camera.get_transform().get_forward_vector()

                        # One of the vertex is behind the camera
                        if not (cam_forward_vec.dot(ray0) > 0):
                            p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
                        if not (cam_forward_vec.dot(ray1) > 0):
                            p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)

                        cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)

    cv2.imwrite('ImageWindowName.png',img)
    print("current_path: ",os.getcwd())
    asd = 0


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
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR)
    cv2.imwrite(camera_name+ ".png",img_1)
    asd = 0




if __name__ == '__main__':
    import gzip
    import json

    #data_path = "/media/hdd/carla_data_collection/leaderboard_plant_pdm_debug/"
    data_path = "//workspace/tg22/detection_debug/original_data/"

    folder_list = os.listdir(data_path)
    #camera_name = "front"
    camera_name_list = ["front", "front_left", "front_right", "back", "back_left", "back_right"]
    for main_index, folder in enumerate(folder_list):

        box_path = data_path + '/' + folder + "/boxes/"
        box_list = os.listdir(box_path)

        for sample_index, box in enumerate(box_list):
            current_box = box_path + box
            with gzip.open(box_path+str(sample_index).zfill(4)+".json.gz", 'rt', encoding='utf-8') as gz_file:
                content = json.load(gz_file)
            lidar_path = data_path + '/' + folder + "/detection/lidar/"

            for camera_name in camera_name_list:
                detection_path = data_path + '/' + folder + "/detection/rgb_camera/" + camera_name

                # if folder.split("_")[2] != "ParkingCrossingPedestrian":
                #    continue

                """lidar_sample_path = lidar_path + "lidar_" + str(sample_index) +".laz"
                assert os.path.exists(lidar_sample_path)
    
                import laspy
    
                # Path to your .laz file
                laz_path = lidar_sample_path
    
                # Read the file
                las = laspy.read(laz_path)
                las["float_intensity"]"""
                if sample_index != 0:
                    #image = Image.open(detection_path+"/front_"+str(sample_index)+"_.jpg")
                    image = Image.open(detection_path+"/" + camera_name + "_"+str(sample_index)+"_.jpg")
                    image = np.array(image)
                    new_visualize(image, content, camera_name)
                    asd = 0

