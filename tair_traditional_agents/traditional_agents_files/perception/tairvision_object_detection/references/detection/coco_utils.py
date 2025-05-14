import copy
import os
import numpy as np

import torch
import torch.utils.data
import torchvision
import tairvision

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

# Compose is called from another file due to circular import problems
from tairvision.transforms.common_transforms import Compose
import tqdm


class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, target):
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            target["annotations"] = anno
            return image, target
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        target["annotations"] = anno
        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, enable_mask=True):
        self.enable_mask = enable_mask

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        # image_id = np.array([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = np.array([obj["bbox"] for obj in anno], dtype=np.float32)
        # guard against no boxes via resizing
        boxes = boxes.reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=h)

        classes = np.array([obj["category_id"] for obj in anno], dtype=np.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)
        masks = masks.numpy()

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = np.array([obj["keypoints"] for obj in anno], dtype=np.float32)

            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.enable_mask:
            if masks.shape[0] != 0:
                target["masks"] = masks

        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = np.array([obj["area"] for obj in anno], dtype=np.float32)
        iscrowd = np.array([obj["iscrowd"] for obj in anno], dtype=np.int64)
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_target_sizes"] = torch.tensor([h, w])

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(dl):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx, (img, targets) in enumerate(tqdm.tqdm(dl)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        # img, targets = dl[img_idx]
        img = img[0]
        targets = {k: t[0] for k, t in targets.items()}
        image_id = targets["image_id"].item() if torch.is_tensor(targets["image_id"]) else targets["image_id"]
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        num_objs = len(bboxes)
        if 'masks' in targets and num_objs > 0:
            masks = targets['masks']
            if type(masks) == np.ndarray:
                masks = torch.from_numpy(masks)
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if 'keypoints' in targets and num_objs > 0:
            keypoints = targets['keypoints']
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            if 'masks' in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if 'keypoints' in targets:
                ann['keypoints'] = keypoints[i]
                ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    # collate_fn = dataloader.collate_fn
    # TODO tairvision.datasets.CocoDetection or tairvision.datasets.coco.CocoDetection
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection) \
                or isinstance(dataset, tairvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection) \
            or isinstance(dataset, tairvision.datasets.CocoDetection):
        return dataset.coco
    # TODO, make number of workers generic
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=16)
    # return convert_to_coco_api(dataset)
    return convert_to_coco_api(data_loader)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_coco(root, image_set, transforms, mode='instances'):
    anno_file_template = "{}_{}2017.json"
    PATHS = {
        "train": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    t = [ConvertCocoPolysToMask()]

    if transforms is not None:
        t.append(transforms)
    transforms = Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    num_classes = 91
    num_keypoints = 0

    return dataset, num_classes, num_keypoints


def get_coco_kp(root, image_set, transforms):
    dataset = get_coco(root, image_set, transforms, mode="person_keypoints")
    num_classes = 2
    num_keypoints = 17

    return dataset, num_classes, num_keypoints


def get_label_color(label_index):
    CLASSMAP = [[220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228], [0, 60, 100], [0, 80, 100],
                [0, 0, 70], [0, 0, 192], [250, 170, 30], [100, 170, 30], [0, 0, 0], [220, 220, 0], [175, 116, 175],
                [250, 0, 30], [165, 42, 42], [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0],
                [120, 166, 157], [110, 76, 0], [174, 57, 255], [199, 100, 0], [72, 0, 118], [0, 0, 0], [255, 179, 240],
                [0, 125, 92], [0, 0, 0], [0, 0, 0], [209, 0, 151], [188, 208, 182], [0, 220, 176], [255, 99, 164],
                [92, 0, 73], [133, 129, 255], [78, 180, 255], [0, 228, 0], [174, 255, 243], [45, 89, 255],
                [134, 134, 103], [145, 148, 174], [255, 208, 186], [197, 226, 255], [0, 0, 0], [171, 134, 1],
                [109, 63, 54], [207, 138, 255], [151, 0, 95], [9, 80, 61], [84, 105, 51], [74, 65, 105],
                [166, 196, 102], [208, 195, 210], [255, 109, 65], [0, 143, 149], [179, 0, 194], [209, 99, 106],
                [5, 121, 0], [227, 255, 205], [147, 186, 208], [153, 69, 1], [3, 95, 161], [163, 255, 0],
                [119, 0, 170], [0, 0, 0], [0, 182, 199], [0, 0, 0], [0, 0, 0], [0, 165, 120], [0, 0, 0],
                [183, 130, 88], [95, 32, 0], [130, 114, 135], [110, 129, 133], [166, 74, 118], [219, 142, 185],
                [79, 210, 114], [178, 90, 62], [65, 70, 15], [127, 167, 115], [59, 105, 106], [0, 0, 0],
                [142, 108, 45], [196, 172, 0], [95, 54, 80], [128, 76, 255], [201, 57, 1], [246, 0, 122],
                [191, 162, 208], [0, 0, 0]]

    # CLASSMAP = {
    #     1: (255, 105, 180),
    #     2: (255, 255, 255),
    #     3: (0, 250, 154),
    #     4: (0, 0, 0),
    #     6: (255, 0, 0),
    #     7: (128, 0, 128),
    #     8: (64, 224, 208),
    #     10: (0, 128, 128),
    #     12: (0, 128, 128)
    # }
    other = (255, 255, 0)
    try:
        return CLASSMAP[label_index - 1]
    except:
        return other


def get_label(label_index):
    CLASSMAP = {1: 'person',
                2: 'bicycle',
                3: 'car',
                4: 'motorcycle',
                5: 'airplane',
                6: 'bus',
                7: 'train',
                8: 'truck',
                9: 'boat',
                10: 'traffic light',
                11: 'fire hydrant',
                12: 'stop sign',
                13: 'parking meter',
                14: 'bench',
                15: 'bird',
                16: 'cat',
                17: 'dog',
                18: 'horse',
                19: 'sheep',
                20: 'cow',
                21: 'elephant',
                22: 'bear',
                23: 'zebra',
                24: 'giraffe',
                25: 'backpack',
                26: 'umbrella',
                27: 'handbag',
                28: 'tie',
                29: 'suitcase',
                30: 'frisbee',
                31: 'skis',
                32: 'snowboard',
                33: 'sports ball',
                34: 'kite',
                35: 'baseball bat',
                36: 'baseball glove',
                37: 'skateboard',
                38: 'surfboard',
                39: 'tennis racket',
                40: 'bottle',
                41: 'wine glass',
                42: 'cup',
                43: 'fork',
                44: 'knife',
                45: 'spoon',
                46: 'bowl',
                47: 'banana',
                48: 'apple',
                49: 'sandwich',
                50: 'orange',
                51: 'broccoli',
                52: 'carrot',
                53: 'hot dog',
                54: 'pizza',
                55: 'donut',
                56: 'cake',
                57: 'chair',
                58: 'couch',
                59: 'potted plant',
                60: 'bed',
                61: 'dining table',
                62: 'toilet',
                63: 'tv',
                64: 'laptop',
                65: 'mouse',
                66: 'remote',
                67: 'keyboard',
                68: 'cell phone',
                69: 'microwave',
                70: 'oven',
                71: 'toaster',
                72: 'sink',
                73: 'refrigerator',
                74: 'book',
                75: 'clock',
                76: 'vase',
                77: 'scissors',
                78: 'teddy bear',
                79: 'hair drier',
                80: 'toothbrush',
                81: 'sink',
                82: 'refrigerator',
                83: 'blender',
                84: 'book',
                85: 'clock',
                86: 'vase',
                87: 'scissors',
                88: 'teddy bear',
                89: 'hair drier',
                90: 'toothbrush',
                91: 'hair brush',
                92: 'OOD'
                }
    return CLASSMAP[label_index]
