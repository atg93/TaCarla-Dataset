import tairvision.datasets as dt
from PIL import Image
from tairvision.references.segmentation.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from typing import Dict, Any, Tuple
import numpy as np
import os
import cv2
import torch
from .utils import cat_list


def load_test_loaders(dataset_name, target_type, split):
    dataset_classes_dict = dt.__dict__
    dataset = dataset_classes_dict[dataset_name](
        root="/datasets",
        split=split,
        return_dataset_info=True,
        transforms=ToTensor(),
        target_type=target_type  # This is dummy, target is not important for this code
    )
    loader = DataLoader(dataset,
                        batch_size=1,
                        collate_fn=collate_fn)
    return dataset, loader


def extract_class_mapping_for_given_class_dict(classes, base_class_dict, blocked_classes=[]):
    class_mapping: Dict[int, int] = {}
    for cls in classes:
        for base_class_name, base_class_id in base_class_dict.items():
            if base_class_name.lower() in cls.name.lower():
                if not cls.ignore_in_eval and cls.name.lower() not in blocked_classes:
                    class_mapping.update({cls.train_id: base_class_id})
                    print(cls.name.lower())
    return class_mapping


def save_numpy_mask_as_image(outputs_array, info, target_folder):
    outputs_array = outputs_array.astype(np.uint8)
    outputs_array = outputs_array.squeeze(0)
    outputs_array = cv2.resize(outputs_array, (info[0]['image_size'][1], info[0]['image_size'][0]),
                               interpolation=cv2.INTER_NEAREST)

    outputs_pil = Image.fromarray(outputs_array)
    outputs_pil.save(os.path.join(target_folder, f"{info[0]['image_id']}.png"))


def collate_fn(batch):

    images, targets, info = list(zip(*batch))
    info = list(info)

    batched_imgs, batched_targets = collate_sub_function(images, targets)

    return batched_imgs, batched_targets, info


def collate_sub_function(images, targets):
    batched_imgs = collate_sub_function_images(images)

    if isinstance(targets, Tuple):
        targets = list(targets)

    if isinstance(targets[0], list):
        targets_list = list(zip(*targets))
        batched_targets_list = []
        for target in targets_list:
            # batched_target = cat_list(target, fill_value=255)
            batched_target = torch.stack(target, 0)
            batched_targets_list.append(batched_target)

        batched_targets = batched_targets_list
    elif isinstance(targets[0], dict):
        new_target_dict = {}
        for key in targets[0].keys():
            target_list = []
            for i in range(len(targets)):
                value = targets[i][key]
                target_list.append(value)
            if key == "semantic" or key == "foreground" or key == "panoptic":
                # batched_target = cat_list(target_list, fill_value=255)
                batched_target = torch.stack(target_list, 0)
            elif key == "center_points":
                batched_target = cat_list(target_list)
            elif "mask" in key:
                batched_target = torch.stack(target_list, 0)
            else:
                # batched_target = cat_list(target_list)
                batched_target = target_list
            new_target_dict.update({key: batched_target})

        batched_targets = new_target_dict
    elif targets[0] is None:
        batched_targets = None
    else:
        batched_targets = cat_list(targets, fill_value=255)

    return batched_imgs, batched_targets


def collate_sub_function_images(images):
    if isinstance(images, Tuple):
        images = list(images)

    # batched_imgs = cat_list(images, fill_value=0)
    if isinstance(images[0], list):
        video_batch_list = []
        for image_batches in images:
            video = torch.stack(image_batches, 0)
            video_batch_list.append(video)
        video_batch = torch.stack(video_batch_list, 0)
        video_batch: torch.Tensor = video_batch.permute(0, 2, 1, 3, 4)
        if video_batch.shape[2] == 1:
            video_batch = video_batch.squeeze(2)
        batched_imgs = video_batch
    else:
        batched_imgs = torch.stack(images, 0)

    return batched_imgs