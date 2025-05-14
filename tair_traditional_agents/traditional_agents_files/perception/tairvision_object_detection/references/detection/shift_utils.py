import os
from tairvision.transforms.common_transforms import Compose
from tairvision.datasets.shift import SHIFT
import numpy as np


def get_shift(root, image_set, transforms, mode='instances'):
    anno_file_template = "{}_{}2017.json"
    PATHS = {
        "train": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
    }

    t = [ConvertSHIFT()]

    if transforms is not None:
        t.append(transforms)
    transforms = Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = SHIFT(img_folder, ann_file, transforms=transforms)

    num_classes = 6
    num_keypoints = 0

    return dataset, num_classes, num_keypoints


class ConvertSHIFT(object):

    CLASSMAP = {
        "pedestrian": 1,
        "bicycle": 2,
        "bus": 3,
        "car": 4,
        "motorcycle": 5,
        "truck": 6,
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

        anno = target

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


if __name__ == '__main__':
    ds, _, _ = get_shift('/home/io22/shift', 'val', None)
    item = ds[0]
    print('.')