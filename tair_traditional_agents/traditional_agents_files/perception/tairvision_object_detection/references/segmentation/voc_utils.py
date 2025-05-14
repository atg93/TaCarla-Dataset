import torchvision
import numpy as np
from PIL import Image

from tairvision.references.segmentation.transforms import Compose


class DummyConvertion(object):
    def __call__(self, image, target):
        # This convertion from PIL to numpy array and back to PIL
        # is needed to prevent a bug in RandomCrop function
        target = np.array(target)
        target = Image.fromarray(target)
        return image, target


def get_voc(root, image_set, transforms):
    
    transforms = Compose([
        DummyConvertion(),
        transforms
    ])

    num_classes = 21
    dataset = torchvision.datasets.VOCSegmentation(root, image_set=image_set, transforms=transforms)

    return dataset, num_classes
