import torchvision
import numpy as np

from tairvision.references.segmentation.transforms import Compose


def get_voc(root, image_set, transforms):
    transforms = Compose([
        ConvertVOC(),
        transforms
    ])

    num_classes = 21
    num_keypoints = 0
    dataset = VOCDetection(root, image_set=image_set, transforms=transforms)

    return dataset, num_classes, num_keypoints


class VOCDetection(torchvision.datasets.VOCDetection):
    def __init__(self, root, image_set, transforms):
        super(VOCDetection, self).__init__(root, image_set=image_set)
        self._transforms = transforms
        self.ids = range(self.__len__())

    def __getitem__(self, idx):
        img, target = super(VOCDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target['annotation']['image_id'] = image_id
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertVOC(object):
    CLASSMAP = {
        "aeroplane": 1,
        "bicycle": 2,
        "bird": 3,
        "boat": 4,
        "bottle": 5,
        "bus": 6,
        "car": 7,
        "cat": 8,
        "chair": 9,
        "cow": 10,
        "diningtable": 11,
        "dog": 12,
        "horse": 13,
        "motorbike": 14,
        "person": 15,
        "pottedplant": 16,
        "sheep": 17,
        "sofa": 18,
        "train": 19,
        "tvmonitor": 20,
    }

    def __call__(self, image, target):
        w, h = image.size

        anno = target["annotation"]

        boxes = []
        classes = []
        categories = []
        for obj in anno['object']:
            boxes.append([int(obj['bndbox']['xmin']),
                          int(obj['bndbox']['ymin']),
                          int(obj['bndbox']['xmax']),
                          int(obj['bndbox']['ymax'])
                         ])
            classes.append(self.CLASSMAP[obj['name']])
            categories.append(obj['name'])


        # guard against no boxes via resizing
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=h)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        classes = np.array(classes, dtype=np.int64)

        # check if boxes has > 0 lengths, and also eliminate other classes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        target = {}
        target["boxes"] = boxes[keep]
        target["labels"] = classes[keep]
        target["categories"] = [categories[idx] for idx in np.where(keep)[0]]

        # these keys has been added for coco utils compability
        target["image_name"] = anno['filename']
        target["image_id"] = anno['image_id']
        target['area'] = area[keep]
        target["iscrowd"] = np.array([0 for idx in np.where(keep)[0]], dtype=np.uint8)

        return image, target


def get_label(label_index):
    CLASSMAP = {
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "pottedplant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tvmonitor",
    }
    return CLASSMAP[label_index]