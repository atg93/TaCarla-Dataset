from .generic_data import GenericVisionDataset
import os.path as osp
from collections import namedtuple, OrderedDict
from PIL import Image
import numpy as np
import json
from pycocotools import mask as coco_mask
import torch
import copy


class SHIFT(GenericVisionDataset):
    def __init__(self, root, ann_file, transforms):
        super(SHIFT, self).__init__(root=root, transforms=transforms, target_type='ins_seg')
        self.ann_file =ann_file

        self.anno_dict = self.create_ins_dict_from_json()


    def create_ins_dict_from_json(self):
        print("Caching json file into dictionary")
        image_id = 0
        with open(self.ann_file, 'r') as file:
            annos = json.load(file)

        anno_dict = OrderedDict()
        classes = set()
        for anno in annos['frames']:
            if len(anno['labels']) == 0:
                continue
            if not osp.exists(osp.join(self.root, anno['videoName'], anno['name'])):
                continue
            if image_id % 5 == 0:
                anno_dict[osp.join(anno['videoName'], anno['name'])] = {
                    'labels'    : anno['labels'] if 'labels' in anno.keys() else [],
                                           'image_name': anno['name'],
                                           'video_name': anno['videoName'],
                                           'image_id'  : image_id
                                           }
            classes.update(set([a['category'] for a in anno['labels']]))
            if 'img_info' in anno:
                anno_dict[anno['name']]['img_info'] = anno['img_info']
            image_id += 1

        return anno_dict

    def _valid_target_types(self):
        valid_target_types = ("ins_seg", "ins_seg")
        return valid_target_types

    def __len__(self):
        return len(self.anno_dict)

    def __getitem__(self, item):
        img_name, targets = copy.deepcopy(list(self.anno_dict.items())[item])
        img_path = osp.join(self.root, img_name)
        image = Image.open(img_path).convert('RGB')

        for label in targets['labels']:
            mask = coco_mask.decode(label['rle'])
            label['mask'] = mask

        if self.transforms is not None:
            image, target = self.transforms(image, targets)

        return image, target
