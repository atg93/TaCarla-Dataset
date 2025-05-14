from .cityscapes import Cityscapes
from .culane import Culane
from .bdd100k import BDD100k
from .bdd10k import BDD10k
from .widerface import WIDERFace
from .mapillary import Mapillary, Mapillary12
from .coco_panoptic import CocoPanoptic
from .forddata import Forddata
try:
    from .coco import CocoDetection
except:
    print("coco detection are not important, probably due to missing pycocotools")
try:
    from .coco_panoptic import CocoPanoptic
except:
    print("coco panoptic are not important, probably due to missing pycocotools")
from .generic_data import GenericVisionDataset
from .widerface import WIDERFace
from .lfw import LFWPeople, LFWPairs
from .celeba import CelebA
from .affectnet import AffectNetMulti, UTKFace
from .gtsrb import GTSRB
from .urban_understanding import UrbanUnderstanding
from .nuscenes_monocular import NuScenesMonoDataset
from .carlane import Carlane
from .openlanev1_2d import Openlane
from .tusimple import TuSimple
from .curvelanes import CurveLanes
from .llamas import Llamas
from .once_lanes import Once3DLanes
from .klane import Klane
from .openlane_v2 import OpenLaneV2
# TODO find a solution to commented datasets for jetson deployment
