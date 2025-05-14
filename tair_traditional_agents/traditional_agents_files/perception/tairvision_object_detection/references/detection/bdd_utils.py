import torch
import numpy as np

from PIL import Image

import tairvision.references.detection.transforms as T
from tairvision.datasets.bdd100k import BDD100k
from tairvision.datasets.bdd10k import BDD10k


def get_bdd(root, image_set, transforms, eliminate_others=True):
    num_classes_det20 = 11 if eliminate_others else 14

    SETTINGS = {
        "train": ("train", "det_20", num_classes_det20),
        "train-det": ("train", "det_20", num_classes_det20),
        "val": ("val", "det_20", num_classes_det20),
        "val-det": ("val", "det_20", num_classes_det20),
    }

    t = [ConvertBDD100k(eliminate_others)]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    split, target_type, num_classes = SETTINGS[image_set]
    num_keypoints = 0

    print("Using the split " + split + " and target type(s) " + str(target_type))
    dataset = BDD100k(root=root, split=split, target_type=target_type, transforms=transforms, use_culane=False)

    return dataset, num_classes, num_keypoints

def get_bdd_10k(root, image_set, transforms, eliminate_others=True):
    num_classes_det20 = 11 if eliminate_others else 14

    SETTINGS = {
        "train": ("train", "ins_seg-10k", num_classes_det20),
        "val": ("val", "ins_seg-10k", num_classes_det20)
    }

    t = [ConvertBDD10k(eliminate_others)]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    split, target_type, num_classes = SETTINGS[image_set]
    num_keypoints = 0

    print("Using the split " + split + " and target type(s) " + str(target_type))
    dataset = BDD10k(root=root, split=split, target_type=target_type, transforms=transforms, use_culane=False)

    return dataset, num_classes, num_keypoints

class ConvertBDD100k(object):

    CLASSMAP = {
        "pedestrian": 1,
        "rider": 2,
        "car": 3,
        "truck": 4,
        "bus": 5,
        "train": 6,
        "motorcycle": 7,
        "bicycle": 8,
        "traffic light": 9,
        "traffic sign": 10,
        "other vehicle": 11,
        "other person": 12,
        "trailer": 13,
    }

    def __init__(self, eliminate_others=True):
        self.eliminate_others = eliminate_others

    def __call__(self, image, target):
        w, h = image.size

        anno = target[0]

        boxes = []
        classes = []
        label_ids = []
        categories = []
        for obj in anno['labels']:
            label_ids.append(int(obj['id']))
            boxes.append(list(obj['box2d'].values()))
            classes.append(self.CLASSMAP[obj['category']])
            categories.append(obj['category'])

        # guard against no boxes via resizing
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=h)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        classes = np.array(classes, dtype=np.int64)

        # check if boxes has > 0 lengths, and also eliminate other classes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        if self.eliminate_others:
            keep = keep & (classes < 11)

        target = {}
        target["boxes"] = boxes[keep]
        target["labels"] = classes[keep]
        # target["label_ids"] = np.array([label_ids[idx] for idx in np.where(keep)[0]], dtype=np.int64)
        # target["categories"] = [categories[idx] for idx in np.where(keep)[0]]

        # these keys has been added for coco utils compability
        # target["image_name"] = anno['image_name']
        target["image_id"] = anno['image_id']
        target['area'] = area[keep]
        target["iscrowd"] = np.array([0 for idx in np.where(keep)[0]], dtype=np.uint8)

        target["orig_target_sizes"] = torch.tensor([h, w])

        return image, target


class ConvertBDD10k(object):

    CLASSMAP = {
        "person": 1,
        "rider": 2,
        "bicycle": 3,
        "bus": 4,
        "car": 5,
        "caravan": 6,
        "motorcycle": 7,
        "trailer": 8,
        "train": 9,
        "truck": 10,
    }

    def __init__(self, eliminate_others=True):
        self.eliminate_others = eliminate_others

    @staticmethod
    def __mask2box(mask):
        assert type(mask) == np.ndarray
        y_min = (mask != 0).nonzero()[0].min()
        y_max = (mask != 0).nonzero()[0].max()
        x_min = (mask != 0).nonzero()[1].min()
        x_max = (mask != 0).nonzero()[1].max()
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def __call__(self, image, target):
        w, h = image.size

        anno = target[0]

        boxes = []
        classes = []
        label_ids = []
        categories = []
        masks = []
        for obj in anno['labels']:
            if obj['mask'].max() == 0:
                continue
            label_ids.append(int(obj['id']))
            masks.append(obj['mask'])
            boxes.append(self.__mask2box(obj['mask']))
            classes.append(self.CLASSMAP[obj['category']])
            categories.append(obj['category'])

        masks = np.array(masks)
        # guard against no boxes via resizing
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=h)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        classes = np.array(classes, dtype=np.int64)

        # check if boxes has > 0 lengths, and also eliminate other classes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        if self.eliminate_others:
            keep = keep & (classes < 11)

        target = {}
        target["boxes"] = boxes[keep]
        # target["masks"] = torch.from_numpy(masks[keep])
        target["masks"] = masks[keep]
        target["labels"] = classes[keep]
        target["label_ids"] = np.array([label_ids[idx] for idx in np.where(keep)[0]], dtype=np.int64)
        target["categories"] = [categories[idx] for idx in np.where(keep)[0]]

        # these keys has been added for coco utils compability
        target["image_name"] = anno['image_name']
        target["image_id"] = anno['image_id']
        target['area'] = area[keep]
        target["iscrowd"] = np.array([0 for idx in np.where(keep)[0]], dtype=np.uint8)

        return image, target


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
        10: (0, 255, 255),
        11: (255, 255, 0)
    }
    return CLASSMAP[label_index]


def get_label(label_index):
    CLASSMAP = {
        1: "pedestrian",
        2: "rider",
        3: "car",
        4: "truck",
        5: "bus",
        6: "train",
        7: "motorcycle",
        8: "bicycle",
        9: "traffic light",
        10: "traffic sign",
        11: "OOD"
    }
    return CLASSMAP[label_index]


def get_bdd_subset_with_weather_attribute(train_set, test_set, attribute):
    indices_train = []
    for idx, ele in enumerate(train_set.targets):
        if ele[0]["attributes"]["weather"] == attribute:
            indices_train.append(idx)

    indices_test = []
    for idx, ele in enumerate(test_set.targets):
        if ele[0]["attributes"]["weather"] == attribute:
            indices_test.append(idx)

    train_set = torch.utils.data.Subset(train_set, indices_train)
    test_set = torch.utils.data.Subset(test_set, indices_test)

    return train_set, test_set
