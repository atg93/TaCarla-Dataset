import torch
from torch.utils.data._utils.collate import default_collate
import os
import collections
import cv2
import json
import pickle
import numpy as np
from PIL import Image
from tairvision.datasets.nuscenes import collate, get_view_matrix, translate_polygons
from tairvision.models.bev.common.utils.geometry import calculate_birds_eye_view_parameters
from math import factorial


class OpenLaneV2(torch.utils.data.Dataset):
    def __init__(self, split, collection, data_dict_main, cfg, transforms, data_videoid_to_split, transforms2d=None):
        self.transforms = transforms
        self.transforms2d = transforms2d
        self.root_path = cfg.DATASET.DATAROOT
        self.data_videoid_to_split = data_videoid_to_split
        self.collection = collection
        self.split = split
        self.cfg = cfg
        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES
        self.thickness = cfg.DATASET.CENTERLINE_THICKNESS
        
        self.data_dict_split = data_dict_main[split]
        self.data_samples = self.create_data_sample_list()

        # Bird's-eye view parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )

        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        # Taken from nuscenes part
        self.lidar_to_view = get_view_matrix(
            self.bev_dimension, self.bev_resolution, bev_start_position=self.bev_start_position
        )

        n_control = cfg.DATASET.BEZIER_CONTROL_POINTS
        self.bezier_fixedpoint = CustomParameterizeLane(method = "bezier_Endpointfixed", method_para = {"n_control": n_control})

    def __getitem__(self, index):
        self.transforms.set_random_variables()

        data_temporal_list = self.data_samples[index]
        data = {}
        
        keys_test = ['images', 'intrinsics', 'cams_to_lidar', 'lidar_to_world', 'view', "future_egomotion", 'front_view_images', "data_identifier_list"]

        keys = ['centerlines_list', 'center_line_segmentation', 'center_line_instance', "z_info", "center_line_instance_ordered", 
                'targets2d', "lcte_list", "lclc_list", "ordered_attribute_list", "bezier_list"]
        
        for key in keys_test:
            data[key] = []

        if self.split != "test":
            for key in keys:
                data[key] = []

        for index, (video_id, frame_id) in enumerate(data_temporal_list):
            split = self.data_videoid_to_split[video_id]
            identifier = (split, video_id, frame_id)
            frame = self.collection.get_frame_via_identifier(identifier)
            
            images, intrinsics, extrinsics, pose, view, front_view_image = self.get_input_data(frame)
            
            if self.split != "test":
                centerlines_list, center_line_segmentation, center_line_instance, z_info, \
                center_line_instance_ordered, ordered_attribute_list = self.get_centerline_annotations(frame, view)
                
                front_view_image, target2d = self.get_boxes2d_annotations(frame, front_view_image)
                lclc, lcte = self.get_topology_matrices(frame)

                gt_lc = np.vstack(centerlines_list)
                bezier_curves = self.bezier_fixedpoint({"gt_lc": gt_lc})["gt_lc"]
                bezier_curves = torch.tensor(bezier_curves, dtype=torch.float32)
            else:
                front_view_image, _ = self.transforms2d(front_view_image, None)

            future_egomotion = self.get_future_egomotion(index, view, data_temporal_list)
            


            data['images'].append(images)
            data['intrinsics'].append(intrinsics)
            data['cams_to_lidar'].append(extrinsics)
            data['lidar_to_world'].append(pose)
            data['view'].append(view)
            data["future_egomotion"].append(future_egomotion)
            data['front_view_images'].append(front_view_image)
            data["data_identifier_list"].append(identifier)

            if self.split != "test":
                data['centerlines_list'].append(centerlines_list)
                data['center_line_segmentation'].append(center_line_segmentation)
                data['center_line_instance'].append(center_line_instance)
                data['z_info'].append(z_info)
                data['targets2d'].append(target2d)
                data["lclc_list"].append(lclc)
                data["lcte_list"].append(lcte)
                data["center_line_instance_ordered"].append(center_line_instance_ordered)
                data["ordered_attribute_list"].append(ordered_attribute_list)
                data["bezier_list"].append(bezier_curves)
        
        for key, value in data.items():
            if key not in ["centerlines_list", "targets2d", "front_view_images", "data_identifier_list", "lcte_list", "lclc_list", "ordered_attribute_list", "bezier_list"]:
                data[key] = torch.cat(value, dim=0)

        return data

    def __len__(self):
        return len(self.data_samples)
    
    def create_data_sample_list(self):
        data_samples = []
        for video_id, frame_id_list in self.data_dict_split.items():
            for frame_index_in_sequence, _ in enumerate(frame_id_list):
                data_temporal_sample_list = []
                for i in range(self.sequence_length):
                    if frame_index_in_sequence + i < len(frame_id_list):
                        frame_id_w_ext = frame_id_list[frame_index_in_sequence + i]
                        frame_id = frame_id_w_ext[:-5]
                        data_temporal_sample_list.append((video_id, frame_id))
                    else:
                        frame_id_w_ext = frame_id_list[frame_index_in_sequence]
                        frame_id = frame_id_w_ext[:-5]
                        data_temporal_sample_list.append((video_id, frame_id))
                data_samples.append(data_temporal_sample_list)
        return data_samples
    
    def get_future_egomotion(self, index, view, data_temporal_list):
        # Identity
        sh, sw, _ = 1 / self.bev_resolution
        future_egomotion = np.eye(4, dtype=np.float32)
        view_rot_only = np.eye(4, dtype=np.float32)
        # TODO, what if data augmentation is used? Is it correct? 
        view_rot_only[0, 0:2] = view[0, 0, 0, 0:2] / sw
        view_rot_only[1, 0:2] = view[0, 0, 1, 0:2] / sh

        if index < len(data_temporal_list) - 1:
            frame_identifier_video_id_t1, frame_identifier_frame_id_t1 = data_temporal_list[index + 1]
            frame_identifier_video_id_t0, frame_identifier_frame_id_t0 = data_temporal_list[index]

            split = self.data_videoid_to_split[frame_identifier_video_id_t0]
            frame_identifier_t0 = (split, frame_identifier_video_id_t0, frame_identifier_frame_id_t0)
            frame_identifier_t1 = (split, frame_identifier_video_id_t1, frame_identifier_frame_id_t1)

            frame_t1 = self.collection.get_frame_via_identifier(frame_identifier_t1)
            frame_t0 = self.collection.get_frame_via_identifier(frame_identifier_t0)

            ego_pose_dict = frame_t0.get_pose()
            rotation = ego_pose_dict['rotation']
            translation = ego_pose_dict['translation']
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = translation
            ego_to_world_t0 = pose
            
            ego_pose_dict = frame_t1.get_pose()
            rotation = ego_pose_dict['rotation']
            translation = ego_pose_dict['translation']
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = translation
            ego_to_world_t1 = pose

            future_egomotion = np.linalg.inv(ego_to_world_t1) @ ego_to_world_t0
            future_egomotion = view_rot_only @ future_egomotion @ np.linalg.inv(view_rot_only)

            future_egomotion[3, :3] = 0.0
            future_egomotion[3, 3] = 1.0

        future_egomotion = torch.Tensor(future_egomotion).float()
        return future_egomotion.unsqueeze(0)
    
    def get_input_data(self, frame):
        """
        Parameters
        ----------
            rec: nuscenes identifier for a given timestamp
        Returns
        -------
            images: torch.Tensor<float> (1, N, 3, H, W)
            intrinsics: torch.Tensor<float> (1, N, 3, 3)
            extrinsics: torch.Tensor(1, N, 4, 4)
            pose: torch.Tensor(1, 4, 4)
        """
        images = []
        intrinsics = []
        extrinsics = []

        ego_pose_dict = frame.get_pose()
        rotation = ego_pose_dict['rotation']
        translation = ego_pose_dict['translation']
        #TODO, from ego to world or from world to ego?
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation

        camera_list = frame.get_camera_list()
        front_view_image = None
        for camera in camera_list:
            image_numpy = frame.get_rgb_image(camera)
            image = Image.fromarray(image_numpy)

            if camera == camera_list[0]:
                front_view_image = image

            #TODO, how to consider distortion in intrinsic?
            intrinsic_dict = frame.get_intrinsic(camera)
            intrinsic = intrinsic_dict["K"]
            #TODO, from camera to ego or from ego to camera? The output is expected to be cam to ego
            extrinsic_dict = frame.get_extrinsic(camera)
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = extrinsic_dict["rotation"]
            extrinsic[:3, 3] = extrinsic_dict["translation"]
            cam_to_lidar = extrinsic

            # Apply transforms
            img, intrinsic, cam_to_lidar = self.transforms(image, intrinsic, cam_to_lidar, camera.upper())

            images.append(img.unsqueeze(0).unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(cam_to_lidar.unsqueeze(0).unsqueeze(0))

        images, intrinsics, extrinsics = (
            torch.cat(images, dim=1),
            torch.cat(intrinsics, dim=1),
            torch.cat(extrinsics, dim=1)
        )
        view = self.transforms.update_view(self.lidar_to_view).unsqueeze(0).unsqueeze(0)
        pose = torch.tensor(pose, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        return images, intrinsics, extrinsics, pose, view, front_view_image
    

    def get_centerline_annotations(self, frame, view):
        COLOR_DICT = {
        0:  (0, 0, 0),
        1:  (255, 0, 0),
        2:  (0, 255, 0),
        3:  (0, 0, 255),
        4:  (255, 255, 0),
        }
        centerlines_info = frame.get_annotations_lane_centerlines()
        centerline_list = []
        h, w = self.bev_dimension[:2]
        center_line_segmentation = np.zeros((h, w), dtype=np.uint8)
        center_line_segmentation_debug = np.zeros((h, w, 3), dtype=np.uint8)
        center_line_instance_ordered = np.zeros((h, w), dtype=np.uint8)
        center_line_instance_one = np.zeros((h, w), dtype=np.uint8)
        center_line_instance_two = np.zeros((h, w), dtype=np.uint8)
        center_line_instance_three = np.zeros((h, w), dtype=np.uint8)
        center_line_instance_four = np.zeros((h, w), dtype=np.uint8)

        center_line_instance_debug = np.zeros((h, w, 3), dtype=np.uint8)
        center_line_z_info = np.zeros((h, w), dtype=np.float32)
        ordered_attribute_list = []
        for index, centerline_info in enumerate(centerlines_info):
            #TODO, is this coordiantes in IMU reference system? X forward, Y left, Z up
            centerline_points_3d = centerline_info['points']

            point_set = centerline_points_3d
            x_monotonicity_check_1 = np.sum(point_set[:, 0][:-1] - point_set[:, 0][1:] < 0)
            x_monotonicity_check_2 = np.sum(point_set[:, 0][:-1] - point_set[:, 0][1:] > 0)
            y_monotonicity_check_1 = np.sum(point_set[:, 1][:-1] - point_set[:, 1][1:] < 0)
            y_monotonicity_check_2 = np.sum(point_set[:, 1][:-1] - point_set[:, 1][1:] > 0)
            x_monotonicity_check = np.min([x_monotonicity_check_1, x_monotonicity_check_2])
            y_monotonicity_check = np.min([y_monotonicity_check_1, y_monotonicity_check_2])
            if x_monotonicity_check < y_monotonicity_check:
                if point_set[:, 0][0] - point_set[:, 0][-1] <= 0:
                    attribute = 1
                else:
                    attribute = 2
            elif x_monotonicity_check > y_monotonicity_check:
                if point_set[:, 1][0] - point_set[:, 1][-1] <= 0:
                    attribute = 3
                else:
                    attribute = 4
            else:
                x_movement = np.abs(point_set[:, 0][0] - point_set[:, 0][-1])
                y_movement = np.abs(point_set[:, 1][0] - point_set[:, 1][-1])
                if x_movement > y_movement:
                    if point_set[:, 0][0] - point_set[:, 0][-1] <= 0:
                        attribute = 1
                    else:
                        attribute = 2
                else:
                    if point_set[:, 1][0] - point_set[:, 1][-1] <= 0:
                        attribute = 3
                    else:
                        attribute = 4

            # attribute = int(centerline_info["is_intersection_or_connector"]) + 1
            centerline_points_4d_dummy = np.zeros((centerline_points_3d.shape[0], 4), dtype=np.float32)
            centerline_points_4d_dummy[:, 0] = centerline_points_3d[:, 0]
            centerline_points_4d_dummy[:, 1] = centerline_points_3d[:, 1]
            centerline_points_4d_dummy[:, 2] = centerline_points_3d[:, 2]
            centerline_points_4d_dummy[:, 3] = 1
            
            projected_centerlines = view[0, 0].numpy() @ centerline_points_4d_dummy.T
            centerline_list.append(projected_centerlines[:3, :].T[None])
            centerline_points_2d = projected_centerlines[:2, :].T
            
            # centerlines_list.append(centerline_points_2d)
            centerline_points_2d = centerline_points_2d.round().astype(np.int32).reshape(-1, 1, 2)

            centerline_points_2d[:, 0, 1][centerline_points_2d[:, 0, 1] > h - 1] = h - 1
            centerline_points_2d[:, 0, 0][centerline_points_2d[:, 0, 0] > w - 1] = w - 1

            for point_num in range(centerline_points_3d[:, 2].shape[0]):
                cv2.circle(center_line_z_info, centerline_points_2d[point_num][0],
                radius=self.thickness, thickness=-1,
                color=centerline_points_3d[point_num, 2].astype(np.float64))
            
            # center_line_z_info[centerline_points_2d[:, 0, 1], centerline_points_2d[:, 0, 0]] = centerline_points_3d[:, 2]
            cv2.polylines(center_line_segmentation, [centerline_points_2d], False, attribute, thickness=self.thickness)
            cv2.polylines(center_line_instance_ordered, [centerline_points_2d], False, index + 1, thickness=self.thickness)
            ordered_attribute_list.append(attribute)

            if attribute == 1:
                cv2.polylines(center_line_instance_one, [centerline_points_2d], False, index + 1, thickness=self.thickness)
            elif attribute == 2:
                cv2.polylines(center_line_instance_two, [centerline_points_2d], False, index + 1, thickness=self.thickness)
            elif attribute == 3:
                cv2.polylines(center_line_instance_three, [centerline_points_2d], False, index + 1, thickness=self.thickness)
            elif attribute == 4:
                cv2.polylines(center_line_instance_four, [centerline_points_2d], False, index + 1, thickness=self.thickness)
            # Uncommand the below line for visualization purpose
            # cv2.polylines(center_line_segmentation_debug, [centerline_points_2d], False, COLOR_DICT[attribute], thickness=self.thickness)
            # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            # cv2.polylines(center_line_instance_debug, [centerline_points_2d], False, color, thickness=self.thickness)
        # Uncommand the below line for visualization purpose
        # cv2.imshow("center_line_instance_debug", center_line_instance_debug)
        # cv2.imshow("center_line_segmentation_debug", center_line_segmentation_debug)
        # cv2.waitKey(0)

        center_line_segmentation = torch.tensor(center_line_segmentation)[None, None]

        # center_line_instance = torch.tensor(center_line_instance)[None, None]
        center_line_instance_one = torch.tensor(center_line_instance_one)[None, None]
        center_line_instance_two = torch.tensor(center_line_instance_two)[None, None]
        center_line_instance_three = torch.tensor(center_line_instance_three)[None, None]
        center_line_instance_four = torch.tensor(center_line_instance_four)[None, None]

        center_line_instance_ordered = torch.tensor(center_line_instance_ordered)[None, None]

        center_line_instance = torch.cat([
            center_line_instance_one, 
            center_line_instance_two, 
            center_line_instance_three, 
            center_line_instance_four], dim=1
        )
            
        center_line_z_info = torch.tensor(center_line_z_info)[None, None]
        return centerline_list, center_line_segmentation, center_line_instance, center_line_z_info, center_line_instance_ordered, ordered_attribute_list
    

    def get_topology_matrices(self, frame):
        lclc = frame.get_annotations_topology_lclc()
        lcte = frame.get_annotations_topology_lcte()
        lclc = torch.tensor(lclc).long()
        lcte = torch.tensor(lcte).long()
        return lclc, lcte
    
    def get_boxes2d_annotations(self, frame, front_view_image):
        traffic_elements_infos = frame.get_annotations_traffic_elements()
        boxes2d_list = []
        attributes_list = []

        for te_info in traffic_elements_infos:
            box2d = te_info['points']
            if np.sum(box2d[0] >= box2d[1]) > 0:
                continue
            box2d = box2d.reshape(4)
            attribute = te_info['attribute']
            boxes2d_list.append(box2d)
            attributes_list.append(attribute)

        target2d = {}
        h, w = self.cfg.FRONT_CENTER_IMAGE.ORIGINAL_HEIGHT, self.cfg.FRONT_CENTER_IMAGE.ORIGINAL_WIDTH
        target2d["orig_target_sizes"] = torch.tensor([h, w])
        if len(boxes2d_list) > 0:
            boxes = np.array(boxes2d_list)
            labels = np.array(attributes_list)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            target2d["boxes"] = boxes
            target2d["correctly_resized_boxes"] = boxes.copy()
            target2d["labels"] = labels
            target2d["areas"] = areas
            front_view_image, target2d = self.transforms2d(front_view_image, target2d)
        else:
            target2d["boxes"] = np.empty([0, 4], dtype=np.float32)
            target2d["correctly_resized_boxes"] = np.empty([0, 4], dtype=np.float32)
            target2d["labels"] = np.empty([0], dtype=np.int64)
            target2d["areas"] = np.empty([0], dtype=np.float32)
            front_view_image, _ = self.transforms2d(front_view_image, None)
            
        for key in target2d.keys():
            if not isinstance(target2d[key], torch.Tensor):
                target2d[key] = torch.tensor(target2d[key])

        return front_view_image, target2d


def prepare_dataloaders(cfg, transforms_train, transforms_val, transforms2d_train=None, transforms2d_val=None, return_testloader = False, trainval=False):
    version = cfg.DATASET.VERSION
    
    root_path = cfg.DATASET.DATAROOT
    version = cfg.DATASET.VERSION
    data_dict_main = io.json_load(f'{root_path}/{version}.json')
    data_videoid_to_split = {}
    for split, values in data_dict_main.items():
        for video_id in values.keys():
            data_videoid_to_split[video_id] = split

    # collection = Collection(root_path, root_path, f"{version}_20")
    collection_val = Collection(root_path, root_path, f"{version}_val")
    collection_train = Collection(root_path, root_path, f"{version}_train")

    if trainval:
        collection_train.keys = collection_train.keys + collection_val.keys
        collection_train.frames.update(collection_val.frames)
        data_dict_main["train"].update(data_dict_main["val"])
        
    # collection_val = io.pickle_load(f'{root_path}/{version}_val.pkl')
    # collection_train = io.pickle_load(f'{root_path}/{version}_train.pkl')
    nworkers = cfg.N_WORKERS

    traindata = OpenLaneV2(
        split="train", collection=collection_train, data_dict_main=data_dict_main,
        cfg=cfg, transforms=transforms_train, transforms2d=transforms2d_train, 
        data_videoid_to_split=data_videoid_to_split
    )

    valdata = OpenLaneV2(
        split="val", collection=collection_val, data_dict_main=data_dict_main,
        cfg=cfg, transforms=transforms_val, transforms2d=transforms2d_val, 
        data_videoid_to_split=data_videoid_to_split
    )

    sample_idx = range(0, len(traindata), cfg.DATASET.SAMPLING_RATIO)
    traindata_subset = torch.utils.data.Subset(traindata, sample_idx)
    trainloader = torch.utils.data.DataLoader(
        traindata_subset, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True,
        collate_fn=collate
    )

    sample_idx = range(0, len(valdata), cfg.DATASET.SAMPLING_RATIO)
    valdata_subset = torch.utils.data.Subset(valdata, sample_idx)
    valloader = torch.utils.data.DataLoader(
        valdata_subset, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False,
        collate_fn=collate)
    

    if return_testloader:
        collection_test = Collection(root_path, root_path, f"{version}_test")
        testdata = OpenLaneV2(
            split="test", collection=collection_test, data_dict_main=data_dict_main,
            cfg=cfg, transforms=transforms_val, transforms2d=transforms2d_val, 
            data_videoid_to_split=data_videoid_to_split
        )
        testloader = torch.utils.data.DataLoader(
        testdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False,
        collate_fn=collate)

        return trainloader, valloader, testloader
    
    else:
        return trainloader, valloader


    
#TODO, Shared class suggestion for this class
class IO:
    r"""
    Wrapping io in openlanev2,
    can be modified for different file systems.

    """

    def __init__(self) -> None:
        pass

    def os_listdir(self, path: str) -> list:
        r"""
        Parameters
        ----------
        path : str

        Returns
        -------
        list

        """
        return os.listdir(path)

    def cv2_imread(self, path: str) -> np.ndarray:
        r"""
        Parameters
        ----------
        path : str

        Returns
        -------
        np.ndarray

        """
        return cv2.imread(path)

    def json_load(self, path: str) -> dict:
        r"""
        Parameters
        ----------
        path : str

        Returns
        -------
        dict

        """
        with open(path, 'r') as f:
            result = json.load(f)
        return result

    def pickle_dump(self, path: str, obj: object) -> None:
        r"""
        Parameters
        ----------
        path : str
        obj : object

        """
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def pickle_load(self, path: str) -> object:
        r"""
        Parameters
        ----------
        path : str

        Returns
        -------
        object

        """
        with open(path, 'rb') as f:
            result = pickle.load(f)
        return result
    

io = IO() #The original code implemented this as a singleton class


class Frame:
    r"""
    A data structure containing meta data of a frame.

    """
    def __init__(self, root_path : str, meta : dict) -> None:
        r"""
        Parameters
        ----------
        root_path : str
        meta : dict
            Meta data of a frame.

        """
        self.root_path = root_path
        self.meta = meta

    def get_camera_list(self) -> list:
        r"""
        Retuens a list of camera names.

        Returns
        -------
        list
            A list of str.

        """
        return list(self.meta['sensor'].keys())

    def get_pose(self) -> dict:
        r"""
        Retuens the pose of ego vehicle.

        Returns
        -------
        dict
            {'rotation': [3, 3], 'translation': [3, ]}.

        """
        return self.meta['pose']

    def get_image_path(self, camera : str) -> str:
        r"""
        Retuens the image path given a camera.

        Parameters
        ----------
        camera : str

        Returns
        -------
        str
            Image path.

        """
        return f'{self.root_path}/{self.meta["sensor"][camera]["image_path"]}'

    def get_rgb_image(self, camera : str) -> np.ndarray:
        r"""
        Retuens the RGB image given a camera.

        Parameters
        ----------
        camera : str

        Returns
        -------
        np.ndarray
            RGB Image.

        """
        image_path = self.get_image_path(camera)
        return cv2.cvtColor(io.cv2_imread(image_path), cv2.COLOR_BGR2RGB)

    def get_intrinsic(self, camera : str) -> dict:
        r"""
        Retuens the intrinsic given a camera.

        Parameters
        ----------
        camera : str

        Returns
        -------
        dict
            {'K': [3, 3], 'distortion': [3, ]}.

        """
        return self.meta['sensor'][camera]['intrinsic']

    def get_extrinsic(self, camera : str) -> dict:
        r"""
        Retuens the extrinsic given a camera.

        Parameters
        ----------
        camera : str

        Returns
        -------
        dict
            {'rotation': [3, 3], 'translation': [3, ]}.

        """
        return self.meta['sensor'][camera]['extrinsic']

    def get_annotations(self) -> dict:
        r"""
        Retuens annotations of the current frame.

        Returns
        -------
        dict
            {'lane_centerline': list, 'traffic_element': list, 'topology_lclc': list, 'topology_lcte': list}.

        """
        if 'annotation' not in self.meta:
            return None
        else:
            return self.meta['annotation']

    def get_annotations_lane_centerlines(self) -> list:
        r"""
        Retuens lane centerline annotations of the current frame.

        Returns
        -------
        list
            [{'id': int, 'points': [n, 3]}].
        """
        result = self.get_annotations()
        return result['lane_centerline'] if result is not None else result

    def get_annotations_traffic_elements(self) -> list:
        r"""
        Retuens traffic element annotations of the current frame.

        Returns
        -------
        list
            [{'id': int, 'category': int, 'attribute': int, 'points': [2, 2]}].

        """
        result = self.get_annotations()
        return result['traffic_element'] if result is not None else result

    def get_annotations_topology_lclc(self) -> list:
        r"""
        Retuens the adjacent matrix of topology_lclc.

        Returns
        -------
        list
            [#lane_centerline, #lane_centerline].

        """
        result = self.get_annotations()
        return result['topology_lclc'] if result is not None else result

    def get_annotations_topology_lcte(self) -> list:
        r"""
        Retuens the adjacent matrix of topology_lcte.

        Returns
        -------
        list
            [#lane_centerline, #traffic_element].
        
        """
        result = self.get_annotations()
        return result['topology_lcte'] if result is not None else result


class Collection:
    r"""
    A collection of frames.
    
    """
    def __init__(self, data_root : str, meta_root : str, collection : str) -> None:
        r"""
        Parameters
        ----------
        data_root : str
        meta_root : str
        collection : str
            Name of collection.

        """
        try:
            meta = io.pickle_load(f'{meta_root}/{collection}.pkl')
        except FileNotFoundError:
            raise FileNotFoundError('Please run the preprocessing first to generate pickle file of the collection.')

        self.frames = {k: Frame(data_root, v) for k, v in meta.items()}
        self.keys = list(self.frames.keys())

    def get_frame_via_identifier(self, identifier : tuple) -> Frame:
        r"""
        Returns a frame with the given identifier (split, segment_id, timestamp).

        Parameters
        ----------
        identifier : tuple
            (split, segment_id, timestamp).

        Returns
        -------
        Frame
            A frame identified by the identifier.

        """
        return self.frames[identifier]

    def get_frame_via_index(self, index : int) -> (tuple, Frame):
        r"""
        Returns a frame with the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        (tuple, Frame)
            The identifier of the frame and the frame.

        """
        return self.keys[index], self.frames[self.keys[index]]
    

def interp_arc(points, t=1000):
    r'''
    Linearly interpolate equally-spaced points along a polyline, either in 2d or 3d.
    Parameters
    ----------
    points : List
        List of shape (N,2) or (N,3), representing 2d or 3d-coordinates.
    t : array_like
        Number of points that will be uniformly interpolated and returned.
    Returns
    -------
    array_like  
        Numpy array of shape (N,2) or (N,3)
    Notes
    -----
    Adapted from https://github.com/johnwlambert/argoverse2-api/blob/main/src/av2/geometry/interpolate.py#L120
    '''
    
    # filter consecutive points with same coordinate
    temp = []
    for point in points:
        point = point.tolist()
        if temp == [] or point != temp[-1]:
            temp.append(point)
    if len(temp) <= 1:
        return None
    points = np.array(temp, dtype=points.dtype)

    assert points.ndim == 2

    # the number of points on the curve itself
    n, _ = points.shape

    # equally spaced in arclength -- the number of points that will be uniformly interpolated
    eq_spaced_points = np.linspace(0, 1, t)

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen = np.linalg.norm(np.diff(points, axis=0), axis=1)  # type: ignore
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength

    cumarc = np.zeros(len(chordlen) + 1)
    cumarc[1:] = np.cumsum(chordlen)

    # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins = np.digitize(eq_spaced_points, bins=cumarc).astype(int)  # type: ignore

    # #catch any problems at the ends
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1  # type: ignore
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = points[tbins - 1, :]
    # broadcast to scale each row of `points` by a different row of s
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp = anchors + offsets

    return points_interp


class CustomParameterizeLane:

    def __init__(self, method, method_para):
        method_list = ['bezier', 'polygon', 'bezier_Direction_attribute', 'bezier_Endpointfixed']
        self.method = method
        if not self.method in method_list:
            raise Exception("Not implemented!")
        self.method_para = method_para

    def __call__(self, results):
        centerlines = results['gt_lc']
        para_centerlines = getattr(self, self.method)(centerlines, **self.method_para)
        results['gt_lc'] = para_centerlines
        return results

    def comb(self, n, k):
        return factorial(n) // (factorial(k) * factorial(n - k))

    def fit_bezier(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        conts = np.linalg.lstsq(A, points, rcond=None)
        return conts

    def fit_bezier_Endpointfixed(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        A_BE = A[1:-1, 1:-1]
        _points = points[1:-1]
        _points = _points - A[1:-1, 0].reshape(-1, 1) @ points[0].reshape(1, -1) - A[1:-1, -1].reshape(-1, 1) @ points[-1].reshape(1, -1)

        conts = np.linalg.lstsq(A_BE, _points, rcond=None)

        control_points = np.zeros((n_control, points.shape[1]))
        control_points[0] = points[0]
        control_points[-1] = points[-1]
        control_points[1:-1] = conts[0]

        return control_points

    def bezier(self, input_data, n_control=2):

        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            res = self.fit_bezier(points, n_control)[0]
            start_res = res[0]
            end_res = res[-1]
            first_diff = (np.sum(np.square(start_res - points[0]))) + (np.sum(np.square(end_res - points[-1])))
            second_diff = (np.sum(np.square(start_res - points[-1]))) + (np.sum(np.square(end_res - points[0])))

            if first_diff <= second_diff:
                fin_res = res
            else:
                fin_res = np.zeros_like(res)
                for m in range(len(res)):
                    fin_res[len(res) - m - 1] = res[m]

            fin_res = np.clip(fin_res, 0, 1)
            coeffs_list.append(np.reshape(np.float32(fin_res), (-1)))

        return np.array(coeffs_list)

    def bezier_Direction_attribute(self, input_data, n_control=3):
        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            centerline[:, 1] = centerline[:, 1]
            centerline[:, 0] = centerline[:, 0]
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            res = self.fit_bezier(points, n_control)[0]
            fin_res = np.clip(res, 0, 1)
            start_res = res[0]
            end_res = res[-1]
            first_diff = (np.sum(np.square(start_res - points[0]))) + (np.sum(np.square(end_res - points[-1])))
            second_diff = (np.sum(np.square(start_res - points[-1]))) + (np.sum(np.square(end_res - points[0])))
            if first_diff <= second_diff:
                da = 0
            else:
                da = 1
            fin_res = np.append(fin_res, da)
            coeffs_list.append(np.reshape(np.float32(fin_res), (-1)))
        return np.array(coeffs_list)

    def bezier_Endpointfixed(self, input_data, n_control=2):
        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            res = self.fit_bezier_Endpointfixed(centerline, n_control)
            coeffs = res.flatten()
            coeffs_list.append(coeffs)
        return np.array(coeffs_list, dtype=np.float32)

    def polygon(self, input_data, key_rep='Bounding Box'):
        keypoints = []
        for idx, centerline in enumerate(input_data):
            centerline[:, 1] = centerline[:, 1]
            centerline[:, 0] = centerline[:, 0]
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            if key_rep not in ['Bounding Box', 'SME', 'Extreme Points']:
                raise Exception(f"{key_rep} not existed!")
            elif key_rep == 'Bounding Box':
                res = np.array(
                    [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()]).reshape((2, 2))
                keypoints.append(np.reshape(np.float32(res), (-1)))
            elif key_rep == 'SME':
                res = np.array([points[0], points[-1], points[int(len(points) / 2)]])
                keypoints.append(np.reshape(np.float32(res), (-1)))
            else:
                min_x = np.min([points[:, 0] for p in points])
                ind_left = np.where(points[:, 0] == min_x)
                max_x = np.max([points[:, 0] for p in points])
                ind_right = np.where(points[:, 0] == max_x)
                max_y = np.max([points[:, 1] for p in points])
                ind_top = np.where(points[:, 1] == max_y)
                min_y = np.min([points[:, 1] for p in points])
                ind_botton = np.where(points[:, 1] == min_y)
                res = np.array(
                    [points[ind_left[0][0]], points[ind_right[0][0]], points[ind_top[0][0]], points[ind_botton[0][0]]])
                keypoints.append(np.reshape(np.float32(res), (-1)))
        return np.array(keypoints)
