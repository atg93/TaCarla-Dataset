import os
from PIL import Image
from collections import namedtuple
import numpy as np
from .generic_data import GenericSegmentationVisionDataset
import os.path as osp
from tairvision.references.segmentation.lane_utils import lane_with_radius_settings, simplistic_target
import json


class TuSimple(GenericSegmentationVisionDataset):
    TuSimpleClass = namedtuple('CulaneClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        TuSimpleClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        TuSimpleClass('Line 1', 1, 1, 'line', 0, True, False, (180, 200, 30)),
        TuSimpleClass('Line 2', 2, 2, 'line', 0, True, False, (200, 100, 1)),
        TuSimpleClass('Line 3', 3, 3, 'line', 0, True, False, (1, 100, 200)),
        TuSimpleClass('Line 4', 4, 4, 'line', 0, True, False, (30, 200, 180)),
    ]

    SPLIT_FILES = {
        'trainval': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
        'trainvaltest': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json', 'test_label.json'],
        'train': ['label_data_0313.json', 'label_data_0601.json'],
        'val': ['label_data_0531.json'],
        'test': ['test_label.json'],
    }

    image_height = 720
    image_width = 1280
    def __init__(self,
                 lane_width_radius=None,
                 lane_width_radius_for_metric=None,
                 lane_width_radius_for_uncertain=None,
                 lane_width_radius_for_binary=None,
                 **kwargs) -> None:
        super(TuSimple, self).__init__(**kwargs)

        self.root = os.path.join(self.root, "tusimple")

        self.data_dir_path = self.root
        self.valid_modes = self.valid_splits
        self.images = []
        self.data_info = []
        self.exist_list = []

        anno_files = self.SPLIT_FILES[self.split]

        self.createIndex(anno_files)

        lane_width_radius_dict = lane_with_radius_settings(lane_width_radius=lane_width_radius,
                                                           lane_width_radius_for_metric=lane_width_radius_for_metric,
                                                           lane_width_radius_for_uncertain=lane_width_radius_for_uncertain,
                                                           lane_width_radius_for_binary=lane_width_radius_for_binary,
                                                           transforms=self.transforms,
                                                           image_height=self.image_height, image_width=self.image_width)

        self.lane_width_radius = lane_width_radius_dict['lane_width_radius']
        self.lane_width_radius_for_metric = lane_width_radius_dict['lane_width_radius_for_metric']
        self.lane_width_radius_for_uncertain = lane_width_radius_dict['lane_width_radius_for_uncertain']
        self.lane_width_radius_for_metric_for_resized = lane_width_radius_dict['lane_width_radius_for_metric_for_resized']
        self.lane_width_radius_binary = lane_width_radius_dict['lane_width_radius_for_binary']

    def createIndex(self, anno_files):
        max_lanes = 0
        for anno_file in anno_files:
            anno_file = osp.join(self.root, anno_file)
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                mask_path = data['raw_file'].replace('clips', 'seg_label')[:-3] + 'png'
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.data_info.append({
                    'img_path': osp.join(self.root, data['raw_file']),
                    'img_name': data['raw_file'],
                    'mask_path': osp.join(self.root, mask_path),
                    'lanes': lanes,
                })


    def __getitem__(self, idx):
        image_name = self.data_info[idx]["img_path"]
        image = Image.open(image_name).convert('RGB')

        target = Image.open(self.data_info[idx]["mask_path"]).convert('L')
        if self.target_type[0] == "semantic":
            pass
        elif self.target_type[0] == "lane_trainer":
            lanes = self.data_info[idx]["lanes"]
            target_array = np.array(target)
            categories = []

            target_array_zeros = np.zeros_like(np.array(image))
            for lane in lanes:
                x_pos = [point[0] for point in lane]
                y_pos = [point[1] for point in lane]

                selected_indices = target_array[y_pos, x_pos]
                index = np.bincount(selected_indices).argmax()
                categories.append(index)

            lanes = [np.array(lane) for lane in lanes]
            target_orig = simplistic_target(target_array_zeros.copy(), lanes=lanes,
                                                       lane_categories=categories,
                                                       lane_width_radius=self.lane_width_radius,
                                                       return_reduced_mask=True)
            target_validation = simplistic_target(target_array_zeros.copy(), lanes=lanes,
                                                       lane_categories=categories,
                                                       lane_width_radius=self.lane_width_radius_for_metric,
                                                       return_reduced_mask=True)
            target_orig = Image.fromarray(target_orig)
            target_validation = Image.fromarray(target_validation)
            target = [target_validation, target_orig]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.target_type[0] == "lane_trainer":
            target_main_mask = target[1]
            target_validation_mask = target[0]
            target = {"mask": target_main_mask, "validation_mask": target_validation_mask}

        return image, target

    def __len__(self):
        return len(self.data_info)

    def _valid_splits(self):
        valid_splits = ["train", "val", "test", "trainval", "trainvaltest"]
        return valid_splits

    def _valid_target_types(self):
        valid_target_types = ["semantic", "lane_trainer"]
        return valid_target_types