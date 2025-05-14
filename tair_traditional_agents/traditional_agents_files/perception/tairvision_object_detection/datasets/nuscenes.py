import os
import collections
from PIL import Image
import itertools

import numpy as np
import cv2
import torch
from torch.utils.data._utils.collate import default_collate

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud


from tairvision.models.bev.common.utils.geometry import calculate_birds_eye_view_parameters, sample_polyline_points,\
                                                        pad_polygons, polygons_view2lidar

from shapely.geometry import MultiPolygon, LineString
from shapely import affinity, ops
# from functools import lru_cache


STATIC = ['lane', 'road_segment', 'drivable_area']  # Road_block ped_crossing walkway stop_line carpark_area traffic_light
DIVIDER = ['road_divider', 'lane_divider']
DYNAMIC = ['car', 'truck', 'bus', 'trailer', 'construction', 'pedestrian', 'motorcycle', 'bicycle']


class FuturePredictionDataset(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, cfg, transforms, filter_classes, filter_classes_segm=None):
        self.nusc = nusc
        self.is_train = is_train
        self.cfg = cfg
        self.interpolation = cv2.LINE_8

        self.dataroot = self.nusc.dataroot

        self.mode = 'train' if self.is_train else 'val'

        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()
        self.indices = self.get_indices()
        self.filter_classes = filter_classes
        self.filter_classes_segm = filter_classes_segm if filter_classes_segm is not None else filter_classes
        self.is_balanced_dataset = (self.is_train and self.cfg.DATASET.VERSION != "mini" and
                                    len(self.filter_classes.classes) > 2 and self.cfg.DATASET.BALANCE_DATASET)
        if self.is_balanced_dataset:
            self.balanced_indices = self._get_sample_indices()

        self.maps = self.get_maps()

        # Bird's-eye view parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )
        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        self.lidar_to_view = get_view_matrix(self.bev_dimension, self.bev_resolution,
                                             bev_start_position=self.bev_start_position,
                                             ego_pos=cfg.DATASET.EGO_POSITION)

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

        self.transforms = transforms

    def get_scenes(self):
        # Filter by scene split
        split = {'v1.0-trainval': {True: 'train', False: 'val'},
                 'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
                 }
        split = split[self.nusc.version][self.is_train]
        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # Remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # Sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_indices(self):
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)

    def get_cat_ids(self, idx, cat2id):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.
            cat2id

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        rec = self.ixes[idx]
        cat_ids = []
        for annotation_token in rec['anns']:
            # Filter out all non vehicle instances
            annotation = self.nusc.get('sample_annotation', annotation_token)
            if annotation['num_lidar_pts'] + annotation['num_radar_pts'] > 0:
                category_name = annotation['category_name']
                name, label = self.filter_classes.get_class_info(category_name)
                if name is not None and label is not None:
                    cat_ids.append(cat2id[name])
        cat_ids = set(cat_ids)
        return cat_ids

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        classes = self.filter_classes.classes[1:]
        cat2id = {name: i for i, name in enumerate(classes)}

        class_sample_idxs = {cat_id: [] for cat_id in cat2id.values()}
        for idx in range(len(self.ixes)):
            sample_cat_ids = self.get_cat_ids(idx, cat2id)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {k: len(v) / duplicated_samples for k, v in class_sample_idxs.items()}

        sample_indices = []
        frac = 1.0 / len(classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds, int(len(cls_inds) * ratio)).tolist()
        return sample_indices

    def get_maps(self):
        maps = {}
        for scene in self.nusc.scene:
            scene_log = self.nusc.get('log', scene['log_token'])
            map_name = scene_log['location']
            if map_name not in maps.keys():
                maps[map_name] = NuScenesMap(dataroot=self.dataroot, map_name=map_name)

        return maps

    def get_sensor_to_world(self, sample, lidar=False):
        # Transformation egopose to world
        car_egopose = self.nusc.get('ego_pose', sample['ego_pose_token'])
        car_egopose_to_world = get_pose_matrix(car_egopose, use_flat=lidar)
        if lidar:
            return car_egopose_to_world, None

        # From egopose to sensor
        sensor_sample = self.nusc.get('calibrated_sensor', sample['calibrated_sensor_token'])
        sensor_to_car_egopose = get_pose_matrix(sensor_sample)

        sensor_to_world = car_egopose_to_world @ sensor_to_car_egopose

        intrinsic = np.asarray(sensor_sample['camera_intrinsic'])

        return sensor_to_world, intrinsic

    def get_input_data(self, rec):
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
        cameras = self.cfg.IMAGE.NAMES

        # The extrinsics we want are from the camera sensor to "flat egopose" as defined
        # https://github.com/nutonomy/nuscenes-devkit/blob/9b492f76df22943daf1dc991358d3d606314af27/python-sdk/nuscenes/nuscenes.py#L279
        # which corresponds to the position of the lidar.
        # This is because the labels are generated by projecting the 3D bounding box in this lidar's reference frame.

        # From lidar egopose to world.
        lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_to_world, _ = self.get_sensor_to_world(lidar_sample, lidar=True)

        for cam in cameras:
            camera_sample = self.nusc.get('sample_data', rec['data'][cam])

            # Transformation from camera to world
            cam_to_world, intrinsic = self.get_sensor_to_world(camera_sample)
            world_to_cam = np.linalg.inv(cam_to_world)

            # Combine all the transformation.
            # From sensor to lidar.
            lidar_to_cam = world_to_cam @ lidar_to_world
            cam_to_lidar = np.linalg.inv(lidar_to_cam)

            # Load image
            image_filename = os.path.join(self.dataroot, camera_sample['filename'])
            img = Image.open(image_filename)

            # Apply transforms
            img, intrinsic, cam_to_lidar = self.transforms(img, intrinsic, cam_to_lidar)

            images.append(img.unsqueeze(0).unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(cam_to_lidar.unsqueeze(0).unsqueeze(0))

        images, intrinsics, extrinsics = (torch.cat(images, dim=1),
                                          torch.cat(intrinsics, dim=1),
                                          torch.cat(extrinsics, dim=1)
                                          )
        pose = torch.tensor(lidar_to_world, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        view = self.transforms.update_view(self.lidar_to_view).unsqueeze(0).unsqueeze(0)

        return images, intrinsics, extrinsics, pose, view

    def get_static_objects(self, rec, pose, view, layers, patch_radius=150, thickness=1):
        scene_record = self.nusc.get('scene', rec['scene_token'])
        scene_log = self.nusc.get('log', scene_record['log_token'])
        nusc_map = self.maps[scene_log['location']]

        pose = pose[0, 0].numpy()
        view = view[0, 0].numpy()
        box_coords = tuple(np.concatenate([pose[0:2, -1] - patch_radius,
                                           pose[0:2, -1] + patch_radius])
                           )

        records_in_patch = nusc_map.get_records_in_patch(box_coords, layers, 'intersect')
        world_to_lidar = np.linalg.inv(pose)

        road_polygons = []
        divider_positions = []
        renders_polygons = []
        renders_lines = []
        h, w = self.bev_dimension[:2]
        for layer in layers:
            render_polygons = np.zeros((h, w), dtype=np.uint8)
            render_lines = np.zeros((h, w), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = nusc_map.get(layer, r)

                if layer in DIVIDER:
                    lines = nusc_map.extract_line(polygon_token['line_token'])
                    lines = [np.float32(lines.xy)]  # 2 n
                    lines_bev = translate_polygons(lines, world_to_lidar, view=view, round=False)
                    lines_lidar = polygons_view2lidar(lines_bev, view)

                    divider_positions += lines_lidar

                    lines_bev = [l.astype('int32') for l in lines_bev]
                    cv2.polylines(render_lines, lines_bev, False, 1, thickness=thickness)

                if layer in STATIC:
                    if 'polygon_tokens' in polygon_token.keys():
                        polygon_tokens = polygon_token['polygon_tokens']
                    else:
                        polygon_tokens = [polygon_token['polygon_token']]

                    for p in polygon_tokens:
                        polygon = nusc_map.extract_polygon(p)
                        polygon = MultiPolygon([polygon])

                        if layer in ['lane', 'road_segment']:
                            road_polygons.append(polygon)

                        exteriors = [np.array(poly.exterior.coords).T for poly in polygon.geoms]
                        exteriors = translate_polygons(exteriors, world_to_lidar, view)
                        cv2.fillPoly(render_polygons, exteriors, 1, self.interpolation)

                        interiors = [np.array(pi.coords).T for poly in polygon.geoms for pi in poly.interiors]
                        interiors = translate_polygons(interiors, world_to_lidar, view)
                        cv2.fillPoly(render_polygons, interiors, 0, self.interpolation)

            if layer in STATIC:
                render_polygons = torch.tensor(render_polygons).unsqueeze(0).unsqueeze(0)
                renders_polygons.append(render_polygons)
            if layer in DIVIDER:
                render_lines = torch.tensor(render_lines).unsqueeze(0).unsqueeze(0)
                renders_lines.append(render_lines)

        renders_polygons = torch.cat(renders_polygons, dim=1)
        renders_lines = torch.cat(renders_lines, dim=1)

        polygon_positions = self.road_polygons_to_boundary(road_polygons, world_to_lidar)

        # TODO: change hard-coded distance parameter to configurable parameter
        divider_positions = sample_polyline_points(divider_positions, sample_distance=3)
        divider_positions = list(itertools.chain(*[segment_polylines_in_map(e, -15, 15, -30, 30) for e in divider_positions]))
        divider_positions = [line for line in divider_positions if len(line) > 1]

        polygon_positions = sample_polyline_points(polygon_positions, sample_distance=3)
        polygon_positions = list(itertools.chain(*[segment_polylines_in_map(e, -15, 15, -30, 30) for e in polygon_positions]))
        polygon_positions = [poly for poly in polygon_positions if len(poly) > 1]

        line_instances = divider_positions + polygon_positions
        line_classes = [0] * len(divider_positions) + [1] * len(polygon_positions)

        return renders_polygons, renders_lines, line_instances, line_classes

    @staticmethod
    def road_polygons_to_boundary(road_polygons, world_to_lidar):
        road_polygons = [r for r in road_polygons if r.is_valid]
        road_polygon = ops.unary_union(road_polygons)
        if road_polygon.geom_type != 'MultiPolygon':
            road_polygon = MultiPolygon([road_polygon])

        exteriors = [np.array(poly.exterior.coords).T for poly in road_polygon.geoms]
        exteriors = translate_polygons(exteriors, world_to_lidar, round=False)

        interiors = [np.array(pi.coords).T for poly in road_polygon.geoms for pi in poly.interiors]
        interiors = translate_polygons(interiors, world_to_lidar, round=False)

        return exteriors + interiors

    def get_static_objects_on_cam(self, rec, pose, extrinsics, intrinsics, image_size, layers, patch_radius=50,
                                  thickness=3):
        cameras = self.cfg.IMAGE.NAMES

        scene_record = self.nusc.get('scene', rec['scene_token'])
        scene_log = self.nusc.get('log', scene_record['log_token'])
        nusc_map = self.maps[scene_log['location']]

        pose = pose[0, 0].numpy()
        box_coords = tuple(np.concatenate([pose[0:2, -1] - patch_radius,
                                           pose[0:2, -1] + patch_radius])
                           )

        S = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]
                      ])

        renders_lines = []
        for i in range(len(cameras)):
            cam_to_lidar = extrinsics[0, i].numpy()
            view = intrinsics[0, i].numpy() @ S

            records_in_patch = nusc_map.get_records_in_patch(box_coords, layers, 'intersect')
            world_to_lidar = np.linalg.inv(pose)
            lidar_to_cam = np.linalg.inv(cam_to_lidar)
            world_to_cam = lidar_to_cam @ world_to_lidar

            renders_lines_per_cam = []
            h, w = image_size
            for layer in layers:
                render_lines = np.zeros((h, w), dtype=np.uint8)

                for r in records_in_patch[layer]:
                    polygon_token = nusc_map.get(layer, r)

                    lines = nusc_map.extract_line(polygon_token['line_token'])
                    lines = [np.float32(lines.xy)]  # 2 n
                    lines = translate_polygons(lines, world_to_cam, view, clip_invalids=True, normalize=True)
                    cv2.polylines(render_lines, lines, False, 1, thickness=thickness)

                render_lines = torch.tensor(render_lines).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                renders_lines_per_cam.append(render_lines)

            renders_lines_per_cam = torch.cat(renders_lines_per_cam, dim=2)
            renders_lines.append(renders_lines_per_cam)

        renders_lines = torch.cat(renders_lines, dim=1)

        return renders_lines

    def get_dynamic_objects(self, rec, pose, view, inst_map):
        h, w = self.bev_dimension[:2]

        pose = pose[0, 0].numpy()
        view = view[0, 0].numpy()
        render_segm = np.zeros((h, w), dtype=np.uint8)
        render_inst = np.zeros((h, w), dtype=np.uint8)
        render_attr = np.zeros((h, w), dtype=np.uint8)
        render_zpos = np.zeros((2, h, w), dtype=np.float32)

        for annotation_token in rec['anns']:
            # Filter out all non vehicle instances
            annotation = self.nusc.get('sample_annotation', annotation_token)

            if self.filter_classes_segm(annotation, self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES):
                continue

            annotation = self.filter_classes_segm.adjust_sizes(annotation)

            if annotation['instance_token'] not in inst_map:
                inst_map[annotation['instance_token']] = len(inst_map) + 1
            inst = inst_map[annotation['instance_token']]

            attr = int(annotation['visibility_token'])

            poly_region, z = extract_box_polygons(annotation, pose, view)
            cv2.fillPoly(render_segm, poly_region, 1, self.interpolation)
            cv2.fillPoly(render_inst, poly_region, inst, self.interpolation)
            cv2.fillPoly(render_attr, poly_region, attr, self.interpolation)
            cv2.fillPoly(render_zpos[0], poly_region, z[0], self.interpolation)
            cv2.fillPoly(render_zpos[1], poly_region, z[1], self.interpolation)

        render_segm = torch.tensor(render_segm, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        render_inst = torch.tensor(render_inst, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        render_attr = torch.tensor(render_attr, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        render_zpos = torch.tensor(render_zpos, dtype=torch.float32).unsqueeze(0)

        return render_segm, render_inst, render_attr, render_zpos, inst_map

    def get_projected_boxes(self, rec):
        sensors = self.cfg.IMAGE.NAMES + ['LIDAR_TOP']
        boxes = []

        rot_range = [-0.3925, 0.3925]
        scale_ratio_range = [0.95, 1.05]

        noise_rotation = torch.tensor(np.random.uniform(rot_range[0], rot_range[1]))
        scale = np.random.uniform(scale_ratio_range[0], scale_ratio_range[1])

        for sensor in sensors:
            boxes_per_sensor = []
            sensor_sample = self.nusc.get('sample_data', rec['data'][sensor])

            # Transformation from sensor to egopose
            sensor_to_world, _ = self.get_sensor_to_world(sensor_sample, sensor == 'LIDAR_TOP')

            for annotation_token in rec['anns']:
                # Filter out all non vehicle instances
                annotation = self.nusc.get('sample_annotation', annotation_token)

                if self.filter_classes(annotation, self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES):
                    continue

                annotation = self.filter_classes.adjust_sizes(annotation)

                box = get_box_in_given_pose(annotation, sensor_to_world, self.filter_classes)
                if box is not None:
                    if self.is_train and self.cfg.DATASET.BOX_AUGMENTATION:
                        box = box_rotate_scale(box, noise_rotation, scale)
                    boxes_per_sensor.append(box)
                else:
                    continue

            boxes.append(boxes_per_sensor)

        return boxes

    def get_future_egomotion(self, rec, index, view):
        rec_t0 = rec

        # Identity
        sh, sw, _ = 1 / self.bev_resolution
        future_egomotion = np.eye(4, dtype=np.float32)
        view_rot_only = np.eye(4, dtype=np.float32)
        view_rot_only[0, 0:2] = view[0, 0, 0, 0:2] / sw
        view_rot_only[1, 0:2] = view[0, 0, 1, 0:2] / sh

        if index < len(self.ixes) - 1:
            rec_t1 = self.ixes[index + 1]

            if rec_t0['scene_token'] == rec_t1['scene_token']:
                lidar_sample_t0 = self.nusc.get('sample_data', rec_t0['data']['LIDAR_TOP'])
                lidar_to_world_t0, _ = self.get_sensor_to_world(lidar_sample_t0, lidar=True)

                lidar_sample_t1 = self.nusc.get('sample_data', rec_t1['data']['LIDAR_TOP'])
                lidar_to_world_t1, _ = self.get_sensor_to_world(lidar_sample_t1, lidar=True)

                future_egomotion = np.linalg.inv(lidar_to_world_t1) @ lidar_to_world_t0
                future_egomotion = view_rot_only @ future_egomotion @ np.linalg.inv(view_rot_only)

                future_egomotion[3, :3] = 0.0
                future_egomotion[3, 3] = 1.0

        future_egomotion = torch.Tensor(future_egomotion).float()

        return future_egomotion.unsqueeze(0)

    def get_pointcloud_data(self, pcloud_cls, rec, nsweeps=5, min_distance=2.2, use_filters=False, view=None):
        """
        Returns at most nsweeps of lidar/radar in the ego frame.
        Returned tensor is 5(x, y, z, reflectance, dt, ring_index) x N
        Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
        """

        points = np.zeros((pcloud_cls.nbr_dims(), 0), dtype=np.float32 if pcloud_cls == LidarPointCloud else np.float64)
        all_pclouds = pcloud_cls(points)
        all_times = np.zeros((1, 0))

        # From lidar egopose to world.
        # TODO: Whether to use LIDAR_TOP or RADAR_FRONT, if RADAR_FRONT do not forget to use_flat be False
        lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_to_world, _ = self.get_sensor_to_world(lidar_sample, lidar=True)
        lidar_time = 1e-6 * lidar_sample['timestamp']

        lidar_to_view = view[0, 0].numpy() if view is not None else np.eye(lidar_to_world.shape[0])
        if pcloud_cls == RadarPointCloud:
            # radar has max 700 points
            max_nb_points = 700 * nsweeps
            pclouds = ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT", "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]
            if use_filters:
                RadarPointCloud.default_filters()
            else:
                RadarPointCloud.disable_filters()
        else:
            # TODO: Check the max number of lidar points
            max_nb_points = 70000 * nsweeps
            pclouds = ['LIDAR_TOP']

        # Aggregate current and previous sweeps from all pclouds
        for pcloud in pclouds:
            pcloud_sample = self.nusc.get('sample_data', rec['data'][pcloud])
            for _ in range(nsweeps):
                # Load up the pointcloud and remove points close to the sensor.
                pcloud_filename = os.path.join(self.dataroot, pcloud_sample['filename'])
                curr_pcloud = pcloud_cls.from_file(pcloud_filename)
                curr_pcloud.remove_close(min_distance)

                pcloud_to_world, _ = self.get_sensor_to_world(pcloud_sample)
                world_to_pcloud = np.linalg.inv(pcloud_to_world)

                # Combine all the transformation.
                # From pcloud to lidar.
                lidar_to_pcloud = world_to_pcloud @ lidar_to_world
                pcloud_to_lidar = np.linalg.inv(lidar_to_pcloud)

                curr_pcloud.transform(lidar_to_view @ pcloud_to_lidar)

                # Add time vector which can be used as a temporal feature.
                time_lag = lidar_time - 1e-6 * pcloud_sample['timestamp']
                times = time_lag * np.ones((1, curr_pcloud.nbr_points()))
                all_times = np.concatenate([all_times, times], axis=1)

                # Merge with key pc.
                all_pclouds.points = np.concatenate([all_pclouds.points,
                                                     curr_pcloud.points], axis=1)

                # Abort if there are no previous sweeps.
                if pcloud_sample['prev'] == '':
                    break
                else:
                    pcloud_sample = self.nusc.get('sample_data', pcloud_sample['prev'])

        pcloud_data = np.concatenate([all_pclouds.points, all_times], axis=0)
        pcloud_data = np.pad(pcloud_data, [(0, 0), (0, max_nb_points - pcloud_data.shape[1])], mode='constant')
        pcloud_data = torch.tensor(pcloud_data.T, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        return pcloud_data

    def __len__(self):
        if self.is_balanced_dataset:
            return len(self.balanced_indices)
        else:
            return len(self.indices)

    def __getitem__(self, index):
        """
        Returns
        -------
            data: dict with the following keys:
                image: torch.Tensor<float> (T, N, 3, H, W)
                    normalized cameras images with T the sequence length, and N the number of cameras.
                intrinsics: torch.Tensor<float> (T, N, 3, 3)
                    intrinsics containing resizing and cropping parameters.
                extrinsics: torch.Tensor<float> (T, N, 4, 4)
                    6 DoF pose from world coordinates to camera coordinates.
                segmentation: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                    (H_bev, W_bev) are the pixel dimensions in bird's-eye view.
                instance: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                centerness: torch.Tensor<float> (T, 1, H_bev, W_bev)
                offset: torch.Tensor<float> (T, 2, H_bev, W_bev)
                flow: torch.Tensor<float> (T, 2, H_bev, W_bev)
                future_egomotion: torch.Tensor<float> (T, 6)
                    6 DoF egomotion t -> t+1
                sample_token: List<str> (T,)
                'z_position': list_z_position,
                'attribute': list_attribute_label,
        """
        data = {}
        keys = ['images', 'intrinsics', 'cams_to_lidar', 'lidar_to_world', 'view',
                'lanes', 'lines', 'lines_cam',
                'segmentation', 'instance', 'attribute', 'z_position',
                'boxes',
                'future_egomotion',
                'sample_token', 'scene_token',
                'radar_data', 'lidar_data',
                'line_instances', 'line_classes'
                ]
        for key in keys:
            data[key] = []

        self.transforms.set_random_variables()

        inst_map = {}

        if self.is_balanced_dataset:
            index = self.balanced_indices[index]

        # Loop over all the frames in the sequence.
        for index_t in self.indices[index]:
            rec = self.ixes[index_t]

            images, intrinsics, extrinsics, pose, view = self.get_input_data(rec)
            lanes, lines, line_instances, line_classes = self.get_static_objects(rec, pose, view, layers=STATIC+DIVIDER)
            lines_cam = self.get_static_objects_on_cam(rec, pose, extrinsics, intrinsics, images.shape[3:],
                                                       layers=DIVIDER)
            segm, inst, attr, zpos, inst_map = self.get_dynamic_objects(rec, pose, view, inst_map)
            boxes = self.get_projected_boxes(rec)

            future_egomotion = self.get_future_egomotion(rec, index_t, view)
            radar_data = self.get_pointcloud_data(RadarPointCloud, rec, view=view)
            lidar_data = self.get_pointcloud_data(LidarPointCloud, rec, view=view, nsweeps=1)

            data['images'].append(images)
            data['intrinsics'].append(intrinsics)
            data['cams_to_lidar'].append(extrinsics)
            data['lidar_to_world'].append(pose)
            data['view'].append(view)
            data['line_instances'].append(line_instances)
            data['line_classes'].append(line_classes)

            data['lanes'].append(lanes)
            data['lines'].append(lines)
            data['lines_cam'].append(lines_cam)

            data['segmentation'].append(segm)
            data['instance'].append(inst)
            data['attribute'].append(attr)
            data['z_position'].append(zpos)

            data['boxes'].append(boxes)

            data['future_egomotion'].append(future_egomotion)
            data['sample_token'].append(rec['token'])
            data['scene_token'].append(rec['scene_token'])
            data['radar_data'].append(radar_data)
            data['lidar_data'].append(lidar_data)

        for key, value in data.items():
            if key not in ['boxes', 'sample_token', 'scene_token', 'line_instances', 'line_classes']:
                data[key] = torch.cat(value, dim=0)

        return data


def prepare_dataloaders(cfg, transforms_train, transforms_val, filter_classes, filter_classes_segm=None, return_dataset=False, return_nusc=False):
    version = cfg.DATASET.VERSION
    train_on_training_data = True

    # 28130 train and 6019 val
    # dataroot = os.path.join(cfg.DATASET.DATAROOT, version)
    dataroot = cfg.DATASET.DATAROOT
    nusc = NuScenes(version='v1.0-{}'.format(version), dataroot=dataroot, verbose=True)

    traindata = FuturePredictionDataset(nusc, train_on_training_data, cfg, transforms_train, filter_classes,
                                        filter_classes_segm=filter_classes_segm)
    valdata = FuturePredictionDataset(nusc, False, cfg, transforms_val, filter_classes,
                                        filter_classes_segm=filter_classes_segm)

    # if version == 'mini':
    #     traindata.indices = traindata.indices[:10]
    #     valdata.indices = valdata.indices[:10]

    nworkers = cfg.N_WORKERS
    
    # This is to create a subset of the dataset for debugging purposes
    # When 1, the original full dataset is used
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

    if return_dataset:
        return trainloader, valloader, traindata, valdata
    else:
        return trainloader, valloader


def get_pose_matrix(pose, use_flat=False):
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


def get_box_in_given_pose(annotation, pose_matrix, filter_classes):
    translation = pose_matrix[0:3, 3]
    rotation = Quaternion._from_matrix(pose_matrix)

    name, label = filter_classes.get_class_info(annotation['category_name'])
    if name is None or label is None:
        return None

    # if annotation['num_lidar_pts'] + annotation['num_radar_pts'] <= 0:
    #    return None

    box = Box(annotation['translation'],
              annotation['size'],
              Quaternion(annotation['rotation']),
              name=name,
              label=label,
              token=annotation['token']
              )
    box.translate(-translation)
    box.rotate(rotation.inverse)

    return box


def box_rotate_scale(box, noise_rotation, scale):
    center = box.center
    size = box.wlh
    label = box.label
    name = box.name
    velocity = box.velocity

    # Rotate
    # Convert the quaternion to a rotation matrix
    rotation_matrix = box.orientation.rotation_matrix

    # Apply the rotation order of roll-pitch-yaw
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    yaw += noise_rotation
    # Create a new quaternion from the roll-pitch-yaw angles
    rotation = Quaternion(axis=[0, 0, 1], radians=yaw) * Quaternion(axis=[0, 1, 0], radians=pitch) * Quaternion(
        axis=[1, 0, 0], radians=roll)
    box = Box(center, size, rotation, label=label, name=name, velocity=velocity)

    # Scale
    box.wlh *= scale

    return box


def collate(batch):
    elem = batch[0]
    if isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        if isinstance(next(it), collections.abc.Sequence):
            return batch
        else:
            return default_collate(batch)
    else:
        return default_collate(batch)



def get_view_matrix(bev_dimension, bev_resolution, bev_start_position=None, with_z=True, ego_pos='center'):
    """

    Args:
        bev_dimension:
        bev_resolution:
        bev_start_position:
        with_z:
        ego_pos: 'center' or 'bottom'

    Returns:

    """
    h, w, d = bev_dimension
    h_res, w_res, d_res = bev_resolution
    sh = -1 / h_res
    sw = -1 / w_res
    sd = 1 / d_res

    if bev_start_position is not None:
        # If this line is uncommented, positions starts including the left boundary
        # bev_start_position = bev_start_position - bev_resolution / 2
        offset = - (bev_start_position + (bev_dimension * bev_resolution) * [0.5, 0.5, 0.5])
        if ego_pos == 'center':
            offset = - (bev_start_position + (bev_dimension * bev_resolution) * [0.5, 0.5, 0.5])
        elif ego_pos == 'bottom':
            offset = - (bev_start_position + (bev_dimension * bev_resolution) * [0., 0.5, 0.5])
    else:
        offset = torch.zeros_like(bev_dimension)

    h_off, w_off, d_off = offset
    h_off *= sh
    w_off *= sw
    d_off *= sd

    h_div = 2.
    if ego_pos == 'center':
        h_div = 2.
    if ego_pos == 'bottom':
        h_div = 1.

    V = np.float32([[0., sw, w/2. - w_off],
                    [sh, 0., h/2. - h_off],
                    [sh, 0., h/h_div - h_off],
                    [0., 0.,           1.]
                    ])

    S = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]
                  ])

    H = np.float32([[0., sw, 0., w/2. - w_off],
                    [sh, 0., 0., h/2. - h_off],
                    [sh, 0., 0., h/h_div - h_off],
                    [0., 0., sd, d/2. + d_off],
                    [0., 0., 0.,           1.]
                    ])

    view = H if with_z else V @ S

    return view


def extract_box_polygons(annotation, pose, view):
    box = Box(annotation['translation'], annotation['size'], Quaternion(annotation['rotation']))
    world_to_lidar = np.linalg.inv(pose)
    bottom_idx = [2, 3, 7, 6]
    top_idx = [0, 1, 5, 4]

    corners = [box.corners()]
    corners = translate_polygons3d(corners, world_to_lidar, view)
    corners_xy = [corner[bottom_idx, 0:2].round().astype(np.int32) for corner in corners]

    z = corners[0][:, 2].mean()
    h = (corners[0][top_idx, 2] - corners[0][bottom_idx, 2]).mean()

    return corners_xy, [z, h]


def translate_polygons(polygons, pose, view=None, clip_invalids=False, normalize=False, round=True, near_plane=0.5, far_plane=30):
    polygons = pad_polygons(polygons) # 2 n -> 4 n
    polygons = [pose @ p for p in polygons]  # 4 n

    if clip_invalids:
        depths = np.array([p[2] for p in polygons])
        invalid_polygons = np.logical_or(depths < near_plane, depths > far_plane)
        if np.all(invalid_polygons) or len(polygons) == 0:
            return []
    if view is not None:
        view = view[[0, 1, 3], :] if view.shape[0] == 4 else view
        polygons = [view @ p for p in polygons]                                         # 3 n

    if clip_invalids:
        polygons = [NuScenesMapExplorer._clip_points_behind_camera(p, near_plane) for p in polygons]
        invalid_polygons = [len(p) == 0 or p.shape[1] < 3 for p in polygons]
        if np.all(invalid_polygons):
            return []

    if normalize:
        polygons = [p/p[2] for p in polygons]
    if round:
        polygons = [p[:2].round().astype(np.int32).T for p in polygons]                 # n 2
    else:
        polygons = [p[:2].T for p in polygons]                 # n 2

    return polygons


def translate_polygons3d(polygons, pose, view, normalize=False):
    polygons = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in polygons]  # 4 n
    polygons = [pose @ p for p in polygons]                                          # 4 n
    polygons = [view @ p for p in polygons]                                          # 4 n

    if normalize:
        polygons = [p/p[3] for p in polygons]
    polygons = [p[:3].T for p in polygons]                  # n 3

    return polygons


def segment_polylines_in_map(map_positions, map_min_x, map_max_x,  map_min_y, map_max_y):
    in_map_x = check_between(map_positions[:, 1], map_min_x, map_max_x)
    in_map_y = check_between(map_positions[:, 0], map_min_y, map_max_y)
    in_map = np.logical_and(in_map_y, in_map_x)
    true_indices = np.where(in_map)[0]
    boundaries = np.where(np.diff(true_indices) != 1)[0] + 1
    segments = np.split(true_indices, boundaries)
    return [map_positions[s] for s in segments]


def check_between(array, start, end):
    if type(array) == np.ndarray:
        return np.logical_and(start <= array, array <= end)
