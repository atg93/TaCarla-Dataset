from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple, List, Union
import numpy as np
from .generic_data import GenericSegmentationVisionDataset
from pycocotools.coco import COCO
from collections import namedtuple


class CocoDetection(GenericSegmentationVisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    COCO_CATEGORIES = [
        {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
        {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
        {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
        {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
        {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
        {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
        {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
        {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
        {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
        {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
        {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
        {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
        {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
        {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
        {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
        {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
        {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
        {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
        {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
        {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
        {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
        {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
        {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
        {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
        {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
        {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
        {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
        {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
        {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
        {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
        {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
        {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
        {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
        {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
        {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
        {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
        {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
        {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
        {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
        {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
        {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
        {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
        {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
        {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
        {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
        {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
        {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
        {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
        {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
        {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
        {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
        {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
        {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
        {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
        {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
        {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
        {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
        {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
        {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
        {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
        {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
        {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
        {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
        {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
        {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
        {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
        {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
        {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
        {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
        {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
        {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
        {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
        {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
        {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
        {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
        {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
        {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
        {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
        {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
        {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"}
    ]

    CocoPanopticClass = namedtuple('CocoPanopticClass',
                                   ['name', 'id', 'train_id', 'has_instances', 'ignore_in_eval', 'color'])
    train_id = 0
    classes = []
    for coco_instance in COCO_CATEGORIES:
        coco_class = CocoPanopticClass(coco_instance["name"], coco_instance["id"],
                                       train_id, bool(coco_instance["isthing"]),
                                       False, coco_instance["color"])

        classes.append(coco_class)
        train_id += 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.root = os.path.join(self.root, "coco-2017")

        target_type = self.target_type[0]
        anno_file_template = "{}_{}2017.json".format(target_type, self.split)
        annFile_wo_root = os.path.join("annotations", anno_file_template)
        annFile = os.path.join(self.root, annFile_wo_root)

        self.coco = COCO(annFile)
        ids = list(sorted(self.coco.imgs.keys()))
        self.ids = ids
        if self.split == self.valid_splits[0]:
            new_ids = []
            for id in ids:
                target_list = self._load_target(id)
                if len(target_list) != 0:
                    new_ids.append(id)
            self.ids = new_ids
        self.root = os.path.join(self.root, f"{self.split}2017")

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = dict(image_id=id, annotations=target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)

    def get_number_of_classes(self) -> int:
        target_type = self.target_type[0]
        if target_type == "instances":
            return 91

    def _valid_splits(self):
        valid_splits = ["train", "val"]
        return valid_splits

    def _valid_target_types(self):
        valid_target_types = ["instances"]
        return valid_target_types


class CocoCaptions(CocoDetection):
    """`MS Coco Captions <https://cocodataset.org/#captions-2015>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Example:

        .. code:: python

            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.ToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """

    def _load_target(self, id) -> List[str]:
        return [ann["caption"] for ann in super()._load_target(id)]
