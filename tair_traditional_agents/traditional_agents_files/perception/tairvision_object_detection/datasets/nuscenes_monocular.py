import copy

import numpy as np

import os
from typing import Any, Tuple
from abc import ABC

from torchvision.datasets.vision import VisionDataset
from pycocotools.coco import COCO
from PIL import Image


class NuScenesMonoDataset(VisionDataset, ABC):
    r"""Monocular 3D detection on NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        data_root (str): Path of dataset root.
        eval_version (str, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        version (str, optional): Dataset version. Defaults to 'v1.0-trainval'.
    """
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }

    def __init__(self,
                 version='v1.0-trainval',
                 split="train",
                 **kwargs):
        super().__init__(**kwargs)
        self.data_root = self.root
        self.version = version
        self.split = split

        anno_file_template = "nuscenes_infos_{}_mono3d.coco.json".format(self.split)
        annFile = os.path.join(self.root, anno_file_template)
        self.ann_file = annFile
        self.img_prefix = self.root

        self.coco = COCO(annFile)

        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{annFile}' are not unique!"
        self.data_infos = data_infos

        # check if needed or not
        valid_inds = self._filter_imgs()
        self.data_infos = [self.data_infos[i] for i in valid_inds]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img_info = self.data_infos[index]
        ann_info = self.get_ann_info(index)

        target = dict(img_info=img_info, ann_info=ann_info)
        target['img_info']['cam2img'] = target['img_info']['cam_intrinsic']
        img_path = img_info['file_name']
        image = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.catToImgs[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        self.valid_inds = valid_inds
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotation.

        Args:
            img_info (list[dict]): Image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, labels,
                gt_bboxes_3d, gt_labels_3d, attr_labels, centers2d,
                depths, bboxes_ignore, masks, seg_map
        """
        gt_bboxes = []
        gt_labels = []
        attr_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                attr_labels.append(ann['attribute_id'])
                gt_masks_ann.append(ann.get('segmentation', None))
                # 3D annotations in camera coordinates
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(1, -1)
                velo_cam3d = np.array(ann['velo_cam3d']).reshape(1, 2)
                nan_mask = np.isnan(velo_cam3d[:, 0])
                velo_cam3d[nan_mask] = [0.0, 0.0]
                bbox_cam3d = np.concatenate([bbox_cam3d, velo_cam3d], axis=-1)
                gt_bboxes_cam3d.append(bbox_cam3d.squeeze())
                # 2.5D annotations in camera coordinates
                center2d = ann['center2d'][:2]
                depth = ann['center2d'][2]
                centers2d.append(center2d)
                depths.append(depth)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            attr_labels = np.array(attr_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            attr_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size),
                                       dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)

        dst = np.array([0.5, 1.0, 0.5], dtype=np.float32)
        src = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        gt_bboxes_cam3d[:, :3] += gt_bboxes_cam3d[:, 3:6] * (dst - src)

        # gt_bboxes_cam3d = CameraInstance3DBoxes(
        #     gt_bboxes_cam3d,
        #     box_dim=gt_bboxes_cam3d.shape[-1],
        #     origin=(0.5, 0.5, 0.5))
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_bboxes_3d=gt_bboxes_cam3d,
            gt_labels_3d=gt_labels_3d,
            attr_labels=attr_labels,
            centers2d=centers2d,
            depths=depths,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def get_attr_name(self, attr_idx, label_name):
        """Get attribute from predicted index.

        This is a workaround to predict attribute when the predicted velocity
        is not reliable. We map the predicted attribute index to the one
        in the attribute set. If it is consistent with the category, we will
        keep it. Otherwise, we will use the default attribute.

        Args:
            attr_idx (int): Attribute index.
            label_name (str): Predicted category name.

        Returns:
            str: Predicted attribute name.
        """
        # TODO: Simplify the variable name
        AttrMapping_rev2 = [
            'cycle.with_rider', 'cycle.without_rider', 'pedestrian.moving',
            'pedestrian.standing', 'pedestrian.sitting_lying_down',
            'vehicle.moving', 'vehicle.parked', 'vehicle.stopped', 'None'
        ]
        if label_name == 'car' or label_name == 'bus' \
                or label_name == 'truck' or label_name == 'trailer' \
                or label_name == 'construction_vehicle':
            if AttrMapping_rev2[attr_idx] == 'vehicle.moving' or \
                    AttrMapping_rev2[attr_idx] == 'vehicle.parked' or \
                    AttrMapping_rev2[attr_idx] == 'vehicle.stopped':
                return AttrMapping_rev2[attr_idx]
            else:
                return NuScenesMonoDataset.DefaultAttribute[label_name]
        elif label_name == 'pedestrian':
            if AttrMapping_rev2[attr_idx] == 'pedestrian.moving' or \
                    AttrMapping_rev2[attr_idx] == 'pedestrian.standing' or \
                    AttrMapping_rev2[attr_idx] == \
                    'pedestrian.sitting_lying_down':
                return AttrMapping_rev2[attr_idx]
            else:
                return NuScenesMonoDataset.DefaultAttribute[label_name]
        elif label_name == 'bicycle' or label_name == 'motorcycle':
            if AttrMapping_rev2[attr_idx] == 'cycle.with_rider' or \
                    AttrMapping_rev2[attr_idx] == 'cycle.without_rider':
                return AttrMapping_rev2[attr_idx]
            else:
                return NuScenesMonoDataset.DefaultAttribute[label_name]
        else:
            return NuScenesMonoDataset.DefaultAttribute[label_name]

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def __len__(self) -> int:
        return len(self.img_ids)
