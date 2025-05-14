import torch

import tairvision.references.detection.transforms as T
from nuimages import NuImages
from nuimages.utils.utils import name_to_index_mapping
from tairvision.datasets.nuimages import NuImagesDataset


def get_nuimages(root, image_set, transforms):
    split = "v1.0-" + image_set
    num_keypoints = 0

    nuimages = NuImages(dataroot=root, version=split, verbose=True, lazy=False)
    name_to_index = name_to_index_mapping(nuimages.category)

    t = [ConvertNuImages()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    num_classes = len(nuimages.category) + 1  # add one for background class
    dataset = NuImagesDataset(nuimages, transforms=transforms, remove_empty=True)

    return dataset, num_classes, num_keypoints


class ConvertNuImages(object):
    def __init__(self, combine_classes=True):
        self.combine_classes = combine_classes

    def __call__(self, image, target):
        labels = target['labels']
        target['labels'] = get_combined_classes(labels)

        segmentation = target['segmentation']
        target['segmentation'] = get_combined_classes(segmentation)

        return image, target


def get_combined_classes(x):
    x_combined = torch.zeros_like(x)

    mask_animal = (x == 1)                              # animal
    mask_pedestrian = (x >= 2) * (x <= 8)               # human.pedestrian.*
    mask_object = (x >= 9) * (x <= 13)                  # movable_object.*, static_object.*
    mask_cycle = (x == 14) + (x == 21)                  # vehicle.bicycle, vehicle.motorcycle
    mask_bus = (x == 15) + (x == 16) + (x == 18) + \
               (x == 22) + (x == 23)                    # vehicle.bus.*, vehicle.construction, vehicle.trailer, vehicle.truck
    mask_emergency = (x == 19) + (x == 20)              # vehicle.emergency.*
    mask_car = (x == 17)                                # vehicle.car
    mask_drivable = (x == 24)                           # vehicle.car
    mask_ego = (x == 31)                                # vehicle.ego

    x_combined[mask_animal] = 1
    x_combined[mask_pedestrian] = 2
    x_combined[mask_object] = 3
    x_combined[mask_cycle] = 4
    x_combined[mask_bus] = 5
    x_combined[mask_emergency] = 6
    x_combined[mask_car] = 7
    x_combined[mask_drivable] = 8
    x_combined[mask_ego] = 9

    return x_combined


def get_label(label_index):
    CLASSMAP = {
        1: "animal",
        2: "pedestrian",
        3: "object",
        4: "cycle",
        5: "bus",
        6: "emergency",
        7: "car",
        8: "drivable",
        9: "ego",
    }
    return CLASSMAP[label_index]


def get_label_color(label_index):
    CLASSMAP = {
        1: (255, 105, 180),
        2: (255, 69, 0),
        3: (0, 250, 154),
        4: (64, 224, 208),
        5: (255, 0, 0),
        6: (128, 0, 128),
        7: (0, 0, 0),
        8: (255, 255, 255),
        9: (0, 128, 128),
    }
    return CLASSMAP[label_index]
