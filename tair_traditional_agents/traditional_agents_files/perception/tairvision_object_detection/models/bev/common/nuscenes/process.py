import PIL
import numpy as np

import torch
from tairvision_object_detection.models.bev.common.utils.geometry import update_intrinsics, update_intrinsics_v2, update_view, update_intrinsics_v3
from torchvision import transforms as T
import time

class FilterClasses(object):
    def __init__(self, filter_classes, box_resizing_coef=1):
        self.box_resizing_coef = box_resizing_coef

        if filter_classes == "car":
            self.classes = ['background', 'car']
            self.name_mapping = {
                'vehicle.car': 'car',
            }
            self.eval_config = 'detection_car'
            self.classes_to_resize = []
        elif filter_classes == "pedestrian":
            self.classes = ['background', 'pedestrian']
            self.name_mapping = {
                'human.pedestrian.adult': 'pedestrian',
                'human.pedestrian.child': 'pedestrian',
                'human.pedestrian.construction_worker': 'pedestrian',
                'human.pedestrian.police_officer': 'pedestrian',
            }
            self.eval_config = 'detection_pedestrian'
            self.classes_to_resize = ['pedestrian']

        elif filter_classes == "vehicle_pedestrian":
            self.classes = ['background', 'vehicle', 'pedestrian']
            self.name_mapping = {
                'vehicle.bicycle': 'vehicle',
                'vehicle.bus.bendy': 'vehicle',
                'vehicle.bus.rigid': 'vehicle',
                'vehicle.car': 'vehicle',
                'vehicle.construction': 'vehicle',
                'vehicle.motorcycle': 'vehicle',
                'vehicle.trailer': 'vehicle',
                'vehicle.truck': 'vehicle',
                'human.pedestrian.adult': 'pedestrian',
                'human.pedestrian.child': 'pedestrian',
                'human.pedestrian.construction_worker': 'pedestrian',
                'human.pedestrian.police_officer': 'pedestrian',
            }
            self.eval_config = 'detection_vehicle_pedestrian'
            self.classes_to_resize = ['pedestrian']

        elif filter_classes == "vehicle_cycle_pedestrian":
            self.classes = ['background', 'vehicle', 'cycle', 'pedestrian']
            self.name_mapping = {
                'vehicle.bicycle': 'cycle',
                'vehicle.bus.bendy': 'vehicle',
                'vehicle.bus.rigid': 'vehicle',
                'vehicle.car': 'vehicle',
                'vehicle.construction': 'vehicle',
                'vehicle.motorcycle': 'cycle',
                'vehicle.trailer': 'vehicle',
                'vehicle.truck': 'vehicle',
                'human.pedestrian.adult': 'pedestrian',
                'human.pedestrian.child': 'pedestrian',
                'human.pedestrian.construction_worker': 'pedestrian',
                'human.pedestrian.police_officer': 'pedestrian',
            }
            self.eval_config = 'detection_vehicle_cycle_pedestrian'
            self.classes_to_resize = ['pedestrian']

        elif filter_classes == "all":
            self.classes = ['background', 'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                            'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
            self.name_mapping = {
                'movable_object.barrier': 'barrier',
                'vehicle.bicycle': 'bicycle',
                'vehicle.bus.bendy': 'bus',
                'vehicle.bus.rigid': 'bus',
                'vehicle.car': 'car',
                'vehicle.construction': 'construction_vehicle',
                'vehicle.motorcycle': 'motorcycle',
                'human.pedestrian.adult': 'pedestrian',
                'human.pedestrian.child': 'pedestrian',
                'human.pedestrian.construction_worker': 'pedestrian',
                'human.pedestrian.police_officer': 'pedestrian',
                'movable_object.trafficcone': 'traffic_cone',
                'vehicle.trailer': 'trailer',
                'vehicle.truck': 'truck'
            }
            self.eval_config = 'detection_cvpr_2019'
            self.classes_to_resize = ['pedestrian', 'barrier', 'traffic_cone']

        elif filter_classes == "dynamic":
            self.classes = ['background', 'dynamic']
            self.name_mapping = {
                'vehicle.bicycle': 'dynamic',
                'vehicle.bus.bendy': 'dynamic',
                'vehicle.bus.rigid': 'dynamic',
                'vehicle.car': 'dynamic',
                'vehicle.construction': 'dynamic',
                'vehicle.motorcycle': 'dynamic',
                'vehicle.trailer': 'dynamic',
                'vehicle.truck': 'dynamic',
                'human.pedestrian.adult': 'dynamic',
                'human.pedestrian.child': 'dynamic',
                'human.pedestrian.construction_worker': 'dynamic',
                'human.pedestrian.police_officer': 'dynamic'
            }
            self.eval_config = 'detection_dynamic'
            self.classes_to_resize = ['pedestrian']

        elif filter_classes == "car_truck_bus":
            self.classes = ['background', 'car', 'truck', 'bus']
            self.name_mapping = {
                'vehicle.bus.bendy': 'bus',
                'vehicle.bus.rigid': 'bus',
                'vehicle.car': 'car',
                'vehicle.truck': 'truck'
            }
            self.eval_config = 'detection_car_truck_bus'
            self.classes_to_resize = []

        elif filter_classes == "car_truck_bus_pedestrian":
            self.classes = ['background', 'car', 'truck', 'bus', 'pedestrian']
            self.name_mapping = {
                'vehicle.bus.bendy': 'bus',
                'vehicle.bus.rigid': 'bus',
                'vehicle.car': 'car',
                'vehicle.truck': 'truck',
                'human.pedestrian.adult': 'pedestrian',
                'human.pedestrian.child': 'pedestrian',
                'human.pedestrian.construction_worker': 'pedestrian',
                'human.pedestrian.police_officer': 'pedestrian',
            }
            self.eval_config = 'detection_car_truck_bus_pedestrian'
            self.classes_to_resize = ['pedestrian']

        elif filter_classes == "vehicle":
            self.classes = ['background', 'vehicle']
            self.name_mapping = {
                'vehicle.bicycle': 'vehicle',
                'vehicle.bus.bendy': 'vehicle',
                'vehicle.bus.rigid': 'vehicle',
                'vehicle.car': 'vehicle',
                'vehicle.construction': 'vehicle',
                'vehicle.motorcycle': 'vehicle',
                'vehicle.trailer': 'vehicle',
                'vehicle.truck': 'vehicle',
            }
            self.eval_config = 'detection_vehicle'
            self.classes_to_resize = []

        else:
            raise ValueError("filter_classes should be one of these options: car, pedestrian, vehicle_pedestrian, "
                             "all, dynamic, car_truck_bus, car_truck_bus_pedestrian or vehicle")

        self.visibility_list = [1]

    def __call__(self, annotation, filter_invisible_classes=True):
        is_in_classes = False
        # Class filter
        for c in self.name_mapping:
            if c in annotation['category_name']:
                is_in_classes = True
        if not is_in_classes:
            return True

        # Invisible box filter
        if filter_invisible_classes and int(annotation['visibility_token']) in self.visibility_list:
            return True

        return False

    def get_class_info(self, category_name):
        if category_name in self.name_mapping:
            name = self.name_mapping[category_name]
            label = self.classes.index(name)
            return name, label
        else:
            return None, None

    def adjust_sizes(self, annotation):
        name = self.name_mapping[annotation['category_name']]
        if name in self.classes_to_resize:
            annotation = annotation.copy()
            annotation['size'] = [annotation['size'][0] * self.box_resizing_coef,
                                  annotation['size'][1] * self.box_resizing_coef,
                                  annotation['size'][2]]

        return annotation

    def revert_sizes(self, box):
        if box.name in self.classes_to_resize:
            box.wlh = np.array([box.wlh[0] / self.box_resizing_coef,
                                box.wlh[1] / self.box_resizing_coef,
                                box.wlh[2]
                                ])
        return box


class ResizeCropNormalize(object):
    def __init__(self, augmentation_parameters):
        self.resize_dims = augmentation_parameters['resize_dims']
        self.crop = augmentation_parameters['crop']
        self.left_crop, self.top_crop = self.crop[0:2]
        self.scale_width = augmentation_parameters['scale_width']
        self.scale_height = augmentation_parameters['scale_height']
        self.mean, self.std = decide_mean_std(augmentation_parameters['pretrained_option'])
        self.normalize_image = T.Compose([T.ToTensor(),
                                          T.Normalize(mean=self.mean, std=self.std),
                                          ])

    def __call__(self, img, intrinsic, extrinsic):
        # Resize and crop
        img = resize_and_crop_image(img, resize_dims=self.resize_dims, crop=self.crop)

        # Normalize image
        normalized_img = self.normalize_image(img)

        # Combine resize/cropping in the intrinsics
        intrinsic = torch.Tensor(intrinsic)
        updated_intrinsic = update_intrinsics(intrinsic, self.top_crop, self.left_crop,
                                              scale_width=self.scale_width, scale_height=self.scale_height)

        updated_extrinsic = torch.Tensor(extrinsic)


        return normalized_img, updated_intrinsic, updated_extrinsic


#TODO, consider merging this class with ResizeCropRandomFlipNormalizeCameraKeyBased in openlanev2/process.py
class ResizeCropRandomFlipNormalize(object):
    def __init__(self, augmentation_parameters, enable_random_transforms=False):
        self.resize_dims = augmentation_parameters['resize_dims']
        self.crop = augmentation_parameters['crop']
        self.left_crop, self.top_crop = self.crop[0:2]
        self.scale_width = augmentation_parameters['scale_width']
        self.scale_height = augmentation_parameters['scale_height']
        self.hflip_prob = augmentation_parameters['hflip_prob'] if enable_random_transforms else 0.
        self.vflip_prob = augmentation_parameters['vflip_prob'] if enable_random_transforms else 0.
        self.rotate_prob = augmentation_parameters['rotate_prob'] if enable_random_transforms else 0.
        self.rotation_degree_increments = augmentation_parameters['rotation_degree_increments']
        self.bev_scale_prob = augmentation_parameters['bev_scale_prob'] if enable_random_transforms else 0.
        self.bev_scale_increments = augmentation_parameters['bev_scale_increments']
        self.mean, self.std = decide_mean_std(augmentation_parameters['pretrained_option'])
        self.normalize_image = T.Compose([T.ToTensor(),
                                          T.Normalize(mean=self.mean, std=self.std),
                                          ])

        self.hflip = False
        self.vflip = False
        self.rotation_degree = 0.
        self.bev_scale = 1.0

    def __call__(self, img):
        # Resize and crop

        img = resize_and_crop_image(img, resize_dims=self.resize_dims, crop=self.crop)

        # Flip image with hflip_probability

        img = flip_horizontal(img, np.logical_xor(self.hflip, self.vflip))

        # Normalize image

        normalized_img = self.normalize_image(img)

        # Combine resize/cropping in the intrinsics
        """""""""""
        intrinsic = torch.Tensor(intrinsic)
        updated_intrinsic = update_intrinsics_v2(intrinsic, self.top_crop, self.left_crop,
                                                 scale_width=self.scale_width, scale_height=self.scale_height,
                                                 hflip=np.logical_xor(self.hflip, self.vflip),
                                                 img_size=img.size
                                                 )

        updated_extrinsic = torch.Tensor(extrinsic)
        """""
        return normalized_img

    def update_intrinsics_v3(self, intrinsic):
        updated_intrinsic = update_intrinsics_v3(intrinsic, self.top_crop, self.left_crop,
                                                 hflip=np.logical_xor(self.hflip, self.vflip),
                                                 img_size=self.resize_dims
                                                 )
        return updated_intrinsic

    def set_random_variables(self):
        # Flip image with hflip_probability
        self.hflip = True if np.random.rand() < self.hflip_prob else False
        self.vflip = True if np.random.rand() < self.vflip_prob else False

        # Rotation
        nb_rotations = 360 // self.rotation_degree_increments
        rotation_degree = np.random.randint(0, high=nb_rotations) * self.rotation_degree_increments
        self.rotation_degree = rotation_degree if np.random.rand() < self.rotate_prob else 0.

        nb_bev_scales = int(1.0 / self.bev_scale_increments) + 1
        bev_scale = 1.0 + np.random.randint(0, high=nb_bev_scales) * self.bev_scale_increments
        self.bev_scale = bev_scale if np.random.rand() < self.bev_scale_prob else 1.0

    def update_view(self, view):
        # Combine resize/cropping in the intrinsics
        view = torch.Tensor(view)
        updated_view = update_view(view, hflip=self.hflip, vflip=self.vflip, rotation_degree=self.rotation_degree,
                                   bev_scale=self.bev_scale)

        return updated_view


def get_resizing_and_cropping_parameters(cfg):
    original_height, original_width = cfg.IMAGE.ORIGINAL_HEIGHT, cfg.IMAGE.ORIGINAL_WIDTH
    final_height, final_width = cfg.IMAGE.FINAL_DIM

    resize_scale = cfg.IMAGE.RESIZE_SCALE
    resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
    resized_width, resized_height = resize_dims

    crop_h = cfg.IMAGE.TOP_CROP
    crop_w = int(max(0, (resized_width - final_width) / 2))
    # Left, top, right, bottom crops.
    crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

    if resized_width != final_width:
        print('Zero padding left and right parts of the image.')
    if crop_h + final_height != resized_height:
        print('Zero padding bottom part of the image.')

    hflip_prob = cfg.IMAGE.HFLIP_PROB
    vflip_prob = cfg.IMAGE.VFLIP_PROB
    rotate_prob = cfg.IMAGE.ROTATE_PROB
    rotation_degree_increments = cfg.IMAGE.ROTATION_DEGREE_INCREMENTS
    pretrained_option = cfg.PRETRAINED.LOAD_WEIGHTS
    bev_scale_prob = cfg.IMAGE.pop('BEV_SCALE_PROB', 0.0)
    bev_scale_increments = cfg.IMAGE.pop('BEV_SCALE_INCREMENTS', 1.0)

    return {'scale_width': resize_scale,
            'scale_height': resize_scale,
            'resize_dims': resize_dims,
            'crop': crop,
            'hflip_prob': hflip_prob,
            'vflip_prob': vflip_prob,
            'rotate_prob': rotate_prob,
            'rotation_degree_increments': rotation_degree_increments,
            'pretrained_option': pretrained_option,
            'bev_scale_prob': bev_scale_prob,
            'bev_scale_increments': bev_scale_increments,
            }


def resize_and_crop_image(img, resize_dims, crop):
    # Bilinear resizing followed by cropping
    #img = img.resize(resize_dims, resample=PIL.Image.BILINEAR)
    img = img.crop(crop)
    return img


def flip_horizontal(img, flip):
    if flip:
        img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    return img

#TODO, consider merging this class with decide_mean_std_v2 in openlanev2/process.py
def decide_mean_std(pretrained_option):
    if pretrained_option:
        # nuImages mean and std
        mean = [0.412, 0.413, 0.409]
        std = [0.204, 0.200, 0.207]
    else:
        # ImageNet mean and std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return mean, std


class PCloudListCollator(object):
    def __init__(self, nb_feats):

        if nb_feats == 0:
            print("No radar or lidar")
            self.type = "no_pcloud"
        elif nb_feats == 16:
            print("Using radar only for pointcloud")
            self.type = "radar_only"
        elif nb_feats == 18:
            print("Using both radar and lidar for pointcloud")
            self.type = "radar_lidar"
        else:
            self.type = "no_pcloud"
            ValueError("PCLOUD.N_FEATS should be either 0, 16, or 18.")

    def __call__(self, batch):

        if self.type == "radar_only":
            pcloud_list = [batch['radar_data']]
        elif self.type == "radar_lidar":
            pcloud_list = [batch['radar_data'], batch['lidar_data']]
        else:
            pcloud_list = None

        return pcloud_list



