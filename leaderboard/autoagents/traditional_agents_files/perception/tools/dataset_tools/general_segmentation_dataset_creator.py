import os.path

import numpy as np
from utils import to_numpy, load_test_loaders, extract_class_mapping_for_given_class_dict, save_numpy_mask_as_image

target_folder = "/media/ssd/ek21/urban_understanding_V1/"
train_target_folder = os.path.join(target_folder, "train")
val_target_folder = os.path.join(target_folder, "val")

if not os.path.isdir(train_target_folder):
    os.makedirs(train_target_folder)

if not os.path.isdir(val_target_folder):
    os.makedirs(val_target_folder)

base_class_dict = {'person': 0, 'rider': 1, 'car': 2, 'truck': 3, 'bus': 4, 'train': 5, 'motorcycle': 6, 'bicycle': 7,
                   'traffic light': 8, "traffic sign": 9, "road": 10, "sidewalk": 11, "ego vehicle": 13,
                   'vegetation': 14, 'sky': 15, 'building': 16}

blocked_classes = ["lane marking - symbol (bicycle)", "traffic sign frame", "road median",
                   "road side", "road shoulder", "car mount"]

bdd_train_set, bdd_train_loader = load_test_loaders("BDD100k", target_type="pan_seg-bitmasks-semantic-10k",
                                                    split="train")
cityscapes_train_set, cityscapes_train_loader = load_test_loaders("Cityscapes", target_type="semantic", split="train")
cityscapes_coarse_train_set, cityscapes_coarse_train_loader = load_test_loaders("Cityscapes",
                                                                                target_type="semantic-coarse",
                                                                                split="train")
mapillary_train_set, mapillary_train_loader = load_test_loaders("Mapillary", target_type="semantic", split="train")

bdd_val_set, bdd_val_loader = load_test_loaders("BDD100k", target_type="pan_seg-bitmasks-semantic-10k", split="val")
cityscapes_val_set, cityscapes_val_loader = load_test_loaders("Cityscapes", target_type="semantic", split="val")
mapillary_val_set, mapillary_val_loader = load_test_loaders("Mapillary", target_type="semantic", split="val")

datasets = [bdd_train_set, cityscapes_train_set, cityscapes_coarse_train_set, mapillary_train_set,
            bdd_val_set, cityscapes_val_set, mapillary_val_set]

loaders = [bdd_train_loader, cityscapes_train_loader, cityscapes_coarse_train_loader, mapillary_train_loader,
           bdd_val_loader, cityscapes_val_loader, mapillary_val_loader]

assert len(datasets) == len(loaders), "length of the dataset and loaders should be the same "

class_mappings = []
for dataset in datasets:
    cls_mapping = extract_class_mapping_for_given_class_dict(dataset.classes, base_class_dict, blocked_classes=blocked_classes)
    class_mappings.append(cls_mapping)

for loader, dataset, class_mapping in zip(loaders, datasets, class_mappings):
    dataset_target_type = dataset.target_type[0]
    dataset_name = dataset.root.split('/')[-1]
    dataset_split = dataset.split
    number_of_sample = dataset.__len__()

    if "train" in dataset_split:
        selected_target_folder = train_target_folder
    else:
        selected_target_folder = val_target_folder

    for count, (images, labels, info) in enumerate(loader):
        image = images[0]
        if isinstance(labels, list):
            label = labels[0]
        else:
            label = labels

        image_numpy = to_numpy(image)
        image_numpy = image_numpy.transpose([1, 2, 0])

        labels_numpy = to_numpy(label)
        target_label = 255 * np.ones_like(labels_numpy)

        for label_index, target_index in class_mapping.items():
            target_label[label == label_index] = target_index

        save_numpy_mask_as_image(outputs_array=target_label, info=info, target_folder=selected_target_folder)

        print(f"dataset: {dataset_name}, target_type: {dataset_target_type}, split: {dataset_split},"
              f" count: {count}/{number_of_sample}, image_name: {info[0]['image_id']}")
