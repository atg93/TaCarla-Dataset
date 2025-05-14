from posixpath import split
import torchvision
import numpy as np

from PIL import Image

from tairvision.references.segmentation.transforms import Compose
from tairvision.datasets.bdd100k import BDD100k


class DummyConvertion(object):
    def __call__(self, image, target):
        # This convertion from PIL to numpy array and back to PIL
        # is needed to prevent a bug in RandomCrop function
        target_out = []
        for t in target:
            #if t.mode == 'RGBA':
            #    t = t.convert('P')
            t = np.array(t)
            t = Image.fromarray(t)
            target_out.append(t)
        return image, target_out


def get_bdd(root, image_set, transforms, use_culane):
    # TODO, add mask convert generic structure here also, lane-bitmasks_morph requires mask convert input
    SETTINGS = {
        "train": ("train", "lane-simplified", 2),
        "train-lane": ("train", "lane-simplified", 2),
        "train-drivable": ("train", "drivable-colormaps", 3),
        "train-dual": ("train", ["lane-simplified", "drivable-colormaps"], [2,3]),
        "train-triple": ("train", ["lane-simplified", "drivable-colormaps", "sem_seg-masks"], [2, 3, 19]),
        "val": ("val", "lane-simplified", 2),
        "val-lane": ("val", "lane-simplified", 2),
        "val-drivable": ("val", "drivable-colormaps", 3),
        "val-dual": ("val", ["lane-simplified", "drivable-colormaps"], [2,3]),
        "val-triple": ("val", ["lane-simplified", "drivable-colormaps", "sem_seg-masks"], [2, 3, 19]),
    }
    
    transforms = Compose([
        DummyConvertion(),
        transforms
    ])

    split, target_type, num_classes = SETTINGS[image_set]

    print("Using the split " + split + " and target type(s) " + str(target_type))
    dataset = BDD100k(root=root, split=split, target_type=target_type, transforms=transforms, use_culane=use_culane)

    return dataset, num_classes
