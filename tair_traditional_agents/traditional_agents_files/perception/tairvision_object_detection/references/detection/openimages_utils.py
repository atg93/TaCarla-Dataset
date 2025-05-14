import numpy as np
import os
import torch
import torchvision
import torch.utils.data as data
import tairvision.references.detection.transforms as T


class OpenImages(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(OpenImages, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.choose_classes()

    def __getitem__(self, idx):
        img, tar = super(OpenImages, self).__getitem__(idx)
        target = []
        for p in tar:
            if p['category_id'] == 1:
                target.append(p)

        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def choose_classes(self):

        new_id = []
        for idx in self.ids:
            tar = self._load_target(idx)
            target = []
            for p in tar:
                if p['category_id'] == 502:
                    p['category_id'] = 1
                    target.append(p)
                else:
                    p['category_id'] = -1

            if len(target) != 0:
                new_id.append(idx)
        self.ids = new_id
        # target = dict(image_id=image_id, annotations=target)


def get_openimages(root, image_set, transforms):
    anno_file_train = "openimages_train_bbox.json"
    anno_file_val = "openimages_validation_bbox.json"
    PATHS = {
        "train": ("train", os.path.join("annotations", anno_file_train)),
        "val": ("validation", os.path.join("annotations", anno_file_val))
    }

    t = [ConvertOpenImages()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = OpenImages(img_folder, ann_file, transforms=transforms)

    #if image_set == "train":
    #    dataset = _coco_remove_images_without_annotations(dataset)

    num_classes = 2
    num_keypoints = 0

    return dataset, num_classes, num_keypoints


class ConvertOpenImages(object):

    CLASSMAP = {
        "Human face": 502
    }

    #def __init__(self, choose_classes=True):
    #    self.choose_classes = choose_classes

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


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 0

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
