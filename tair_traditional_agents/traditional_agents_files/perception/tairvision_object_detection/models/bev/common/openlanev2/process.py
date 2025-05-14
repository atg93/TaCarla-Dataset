import numpy as np

import torch
from tairvision.models.bev.common.utils.geometry import update_intrinsics_v2, update_view
from torchvision import transforms as T
from tairvision.models.bev.common.nuscenes.process import resize_and_crop_image, flip_horizontal


class ResizeCropRandomFlipNormalizeCameraKeyBased(object):
    def __init__(self, cfg, augmentation_parameters, augmentation_parameters_dict, enable_random_transforms=False):
        self.resize_dims = augmentation_parameters['resize_dims']
        self.crop = augmentation_parameters['crop']
        self.left_crop, self.top_crop = self.crop[0:2]
        self.augmentation_parameters_dict = augmentation_parameters_dict
        self.hflip_prob = augmentation_parameters['hflip_prob'] if enable_random_transforms else 0.
        self.vflip_prob = augmentation_parameters['vflip_prob'] if enable_random_transforms else 0.
        self.rotate_prob = augmentation_parameters['rotate_prob'] if enable_random_transforms else 0.
        self.rotation_degree_increments = augmentation_parameters['rotation_degree_increments']
        self.mean, self.std = decide_mean_std_v2(cfg)
        self.normalize_image = T.Compose([T.ToTensor(),
                                          T.Normalize(mean=self.mean, std=self.std),
                                          ])

        self.hflip = False
        self.vflip = False
        self.rotation_degree = 0.

    def __call__(self, img, intrinsic, extrinsic, camera_key):
        # Resize and crop for fixed implementation for front image in ArgoverseV2
        crop = self.augmentation_parameters_dict[camera_key]["crop"]
        resize_dims = self.augmentation_parameters_dict[camera_key]["resize_dims"]
        left_crop, top_crop = crop[0:2]

        # Old wrong implementation, open this for older weights
        # crop = self.crop
        # resize_dims = self.resize_dims
        # left_crop, top_crop = crop[0:2]

        img = resize_and_crop_image(img, resize_dims=resize_dims, crop=crop)

        # Flip image with hflip_probability
        img = flip_horizontal(img, np.logical_xor(self.hflip, self.vflip))

        # Normalize image
        normalized_img = self.normalize_image(img)

        # Combine resize/cropping in the intrinsics
        intrinsic = torch.Tensor(intrinsic)
        scale_width = self.augmentation_parameters_dict[camera_key]['scale_width']
        scale_height = self.augmentation_parameters_dict[camera_key]['scale_height']

        updated_intrinsic = update_intrinsics_v2(intrinsic, top_crop, left_crop,
                                                 scale_width=scale_width, scale_height=scale_height,
                                                 hflip=np.logical_xor(self.hflip, self.vflip)
                                                 )

        updated_extrinsic = torch.Tensor(extrinsic)

        return normalized_img, updated_intrinsic, updated_extrinsic

    def set_random_variables(self):
        # Flip image with hflip_probability
        self.hflip = True if np.random.rand() < self.hflip_prob else False
        self.vflip = True if np.random.rand() < self.vflip_prob else False

        # Rotation
        nb_rotations = 360 // self.rotation_degree_increments
        rotation_degree = np.random.randint(0, high=nb_rotations) * self.rotation_degree_increments
        self.rotation_degree = rotation_degree if np.random.rand() < self.rotate_prob else 0.

    def update_view(self, view):
        # Combine resize/cropping in the intrinsics
        view = torch.Tensor(view)
        updated_view = update_view(view, hflip=self.hflip, vflip=self.vflip, rotation_degree=self.rotation_degree)

        return updated_view

def get_resizing_and_cropping_parameters_modified(CFG_IMAGE, CFG_PRETRAINED):
    original_height, original_width = CFG_IMAGE.ORIGINAL_HEIGHT, CFG_IMAGE.ORIGINAL_WIDTH
    final_height, final_width = CFG_IMAGE.FINAL_DIM

    resize_scale_width = CFG_IMAGE.RESIZE_SCALE_WIDTH
    resize_scale_height = CFG_IMAGE.RESIZE_SCALE_HEIGHT
    resize_dims = (int(np.round(original_width * resize_scale_width)), int(np.round(original_height * resize_scale_height)))
    resized_width, resized_height = resize_dims

    crop_h = CFG_IMAGE.TOP_CROP
    crop_w = int(max(0, (resized_width - final_width) / 2))
    # Left, top, right, bottom crops.
    crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

    if resized_width != final_width:
        print('Zero padding left and right parts of the image.')
    if crop_h + final_height != resized_height:
        print('Zero padding bottom part of the image.')

    hflip_prob = CFG_IMAGE.HFLIP_PROB
    vflip_prob = CFG_IMAGE.VFLIP_PROB
    rotate_prob = CFG_IMAGE.ROTATE_PROB
    rotation_degree_increments = CFG_IMAGE.ROTATION_DEGREE_INCREMENTS
    pretrained_option = CFG_PRETRAINED.LOAD_WEIGHTS

    return {'scale_width': resize_scale_width,
            'scale_height': resize_scale_height,
            'resize_dims': resize_dims,
            'crop': crop,
            'hflip_prob': hflip_prob,
            'vflip_prob': vflip_prob,
            'rotate_prob': rotate_prob,
            'rotation_degree_increments': rotation_degree_increments,
            'pretrained_option': pretrained_option,
            }


def decide_mean_std_v2(cfg):
    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # nuImages mean and std
        mean = [0.412, 0.413, 0.409]
        std = [0.204, 0.200, 0.207]
    elif "clip" in cfg.MODEL.ENCODER.BACKBONE.VERSION: 
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        # ImageNet mean and std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return mean, std