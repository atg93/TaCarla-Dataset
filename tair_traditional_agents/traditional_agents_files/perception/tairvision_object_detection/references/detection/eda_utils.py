import numpy as np
import os
import torch
import torchvision
import torch.utils.data as data
import tairvision.references.detection.transforms as T
from tairvision.references.detection.coco_utils import ConvertCocoPolysToMask
from tairvision.transforms.common_transforms import Compose


class EdaDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(EdaDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(EdaDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_eda(root, image_set, transforms):
    anno_file_train = "eda_coco"
    anno_file_val = "eda_coco_test"
    PATHS = {
        "train": ("train", os.path.join("annotations", anno_file_train)),
        "val": ("validation", os.path.join("annotations", anno_file_val))
    }

    t = [ConvertEda()]

    if transforms is not None:
        t.append(transforms)
    transforms = Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = EdaDetection(img_folder, ann_file, transforms=transforms)

    num_classes = 46
    num_keypoints = 0

    return dataset, num_classes, num_keypoints

class ConvertEda:

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        #image_id = np.array([image_id])
        anno = target["annotations"]

        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        boxes = np.array([obj["bbox"] for obj in anno], dtype=np.float32)
        # guard against no boxes via resizing
        boxes = boxes.reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=h)


        classes = np.array([obj["category_id"] for obj in anno], dtype=np.int64)
        #for obj in anno:
        #    if obj['category_id'] == 502:
        #        classes.append([obj['category_id']])
        #classes = np.array(classes, dtype=np.int64)


        #segmentations = [obj["segmentation"] for obj in anno]
        #masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = np.array([obj["keypoints"] for obj in anno], dtype=np.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        boxes = boxes[keep]
        classes = classes[keep]
        #masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        #if self.choose_classes:
        #    keep = keep & (classes == 502)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        #target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints


        # for conversion to coco api

        area = np.array([obj["area"] for obj in anno], dtype=np.float32)
        iscrowd = np.array([obj["iscrowd"] for obj in anno], dtype=np.int64)
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        #target_out["iscrowd"] = np.array([0 for idx in np.where(keep)[0]], dtype=np.uint8)

        return image, target

def get_label_color_eda(label_index):
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
        11: (0, 250, 154),
        12: (64, 224, 208),
        13: (255, 0, 0),
        14: (128, 0, 128),
        15: (0, 0, 0),
        16: (255, 255, 255),
        17: (0, 128, 128),
        18: (0, 255, 255),
    }
    return CLASSMAP[label_index]

def get_label_eda(label_index):
    CLASSMAP = {
                1: 'girilmez',
                2: 'tasit_trafigine_kapali',
                3: 'duz veya sola',
                4: 'duz veya saga',
                5: 'yalnizca sola',
                6: '20 hiz limit sonu',
                7: '30 limit',
                8: '20 limit',
                9: 'yalnizca saga',
                10: 'saga donulmez',
                11: 'sola donulmez',
                12: 'dur',
                13: 'park yapilmaz',
                14: 'park',
                15: 'durak',
                16: 'kirmizi isik',
                17: 'sari isik',
                18: 'yesil_isik'

                }
    return CLASSMAP[label_index]

def get_label_eda_gtsrb(label_index):
    CLASSMAP = {
                1: 'girilmez',
                2: 'tasit_trafigine_kapali',
                3: 'duz veya sola',
                4: 'duz veya saga',
                5: 'yalnizca sola',
                6: '20 hiz limit sonu',
                7: '30 limit',
                8: '20 limit',
                9: 'yalnizca saga',
                10: 'saga donulmez',
                11: 'sola donulmez',
                12: 'dur',
                13: 'park yapilmaz',
                14: 'park',
                15: 'durak',
                16: 'kirmizi isik',
                17: 'sari isik',
                18: 'yesil_isik',
                19: '50 limit',
                20: 'ondeki_tasiti_gecmek_yasaktir',
                21: 'kamyonlar_icin_gecmek_yasaktir',
                22: 'ana_tali_yol_kavsagi',
                23: 'yol_ver',
                24: 'kamyon_giremez',
                25: 'dikkat',
                26: 'sola_tehlikeli_viraj',
                27: 'saga_tehlikeli_viraj',
                28: 'sola_tehlikeli_devamli_virajlar',
                29: 'kasisli_yol',
                30: 'kaygan_yol',
                31: 'sagdan_daralan_kaplama',
                32: 'yolda_calisma',
                33: 'isikli_isaret_cihazi',
                34: 'okul_gecidi',
                35: 'gizli_buzlanma',
                36: 'vahsi_hayvan',
                37: 'yasaklamalarin_sonu',
                38: 'saga_mecburi',
                39: 'sola_mecburi',
                40: 'ileri_mecburi',
                41: 'sagdan_gidiniz',
                42: 'soldan_gidiniz',
                43: 'ada_etrafindan_donunuz',
                44: 'gecme_yasagi_sonu',
                45: 'kamyon_icin_gecme_yasagi_sonu',
                46: 'other',
                47: ' '
                }
    return CLASSMAP[label_index]