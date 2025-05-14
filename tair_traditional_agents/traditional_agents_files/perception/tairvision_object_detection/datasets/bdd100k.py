import json
import os
import numpy as np
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, NamedTuple
from PIL import Image
from .generic_data import GenericSegmentationVisionDataset


class BDD100k(GenericSegmentationVisionDataset):
    BDD100kClass = namedtuple('BDD100kClass', ['name', 'id', 'train_id',
                                               'lane_category_id',
                                               'lane_style_id',
                                               'lane_direction_id',
                                               'ignore_in_eval', 'color', 'has_instances'])

    lane_category_name_dict = {
        0: "crosswalk",
        1: "double other",
        2: "double white",
        3: "double yellow",
        4: "road curb",
        5: "single other",
        6: "single white",
        7: "single yellow",
        -1: "background"
    }

    lane_directions_name_dict = {
        0: "parallel",
        1: "vertical",
        -1: "background"
    }

    lane_styles_name_dict = {
        0: "solid",
        1: "dashed",
        -1: "background"
    }
    # color_chart = []
    # number_of_classes = 29
    # for class_id in range(number_of_classes):
    #     # get different colors for the edges
    #     rgb = matplotlib.colors.hsv_to_rgb([
    #         class_id / number_of_classes, 1.0, 1.0
    #     ])
    #     rgb = rgb * 255
    #     color_chart.append(rgb)
    # color_chart_numpy = np.concatenate(color_chart).reshape(-1, 3)
    # color_chart_numpy = color_chart_numpy.astype(np.uint8)

    lane_mask_classes = [
        BDD100kClass('Background', 255, 0, -1, -1, -1, False, (0, 0, 0), False),
        BDD100kClass('crosswalk vertical dashed', 48, 1, 0, 1, 1, False, (255,   0,   0), False),
        BDD100kClass('road curb', 4, 2, 4, 0, 0, False, (255, 153,   0), False),
        BDD100kClass('single white', 6, 3, 6, 0, 0, False, (255, 255,   255), False),
        BDD100kClass('single white dashed', 22, 4, 6, 1, 0, False, (51, 255,   0), False),
        BDD100kClass('single white vertical', 38, 5, 6, 0, 1, False, (0, 255, 102), False),
        BDD100kClass('double white', 2, 6, 2, 0, 0, False, (0, 255, 255), False),
        BDD100kClass('double white dashed', 18, 7, 2, 1, 0, False, (0, 102, 255), False),
        BDD100kClass('double yellow', 3, 8, 3, 0, 0, False, (50,   0, 255), False),
        BDD100kClass('single yellow', 7, 9, 7, 0, 0, False, (204,   0, 255), False),
        BDD100kClass('double yellow dashed', 19, 10, 3, 1, 0, False, (255,   0, 152), False),
        BDD100kClass('double other', 1, 11, 1, 0, 0, False, (200,   52, 0), False),
        BDD100kClass('single other', 5, 12, 5, 0, 0, False, (200, 105,   0), False),
        BDD100kClass('single other dashed', 21, 13, 5, 1, 0, False, (200, 158,   0), False),
        BDD100kClass('single yellow dashed', 23, 14, 7, 1, 0, False, (200, 211,   0), False),
        BDD100kClass('double white vertical', 34, 15, 2, 0, 1, False, (246, 200,   0), False),
        BDD100kClass('crosswalk vertical', 0, 16, 0, 1, 0, False, (193, 200,   0), False),
        BDD100kClass('single yellow vertical', 39, 17, 7, 0, 1, False, (140, 200,   0), False),
        BDD100kClass('road curb vertical', 36, 18, 4, 0, 1, False, (82, 200,   0), False),
        BDD100kClass('single white vertical dashed', 54, 19, 6, 1, 1, False, (35, 200,   0), False),
        BDD100kClass('road curb dashed', 20, 20, 4, 1, 0, False, (0, 200,  17), False),
        BDD100kClass('crosswalk vertical', 32, 21, 0, 0, 1, False, (0, 200,  70), False),
        BDD100kClass('road curb dashed vertical', 52, 22, 4, 1, 1, False, (0, 200,  123), False),
        BDD100kClass('double other dashed', 17, 23, 1, 1, 0, False, (0, 200,  175), False),
        BDD100kClass('crosswalk dashed', 16, 24, 0, 1, 0, False, (0, 200,  228), False),
        BDD100kClass('single other vertical', 37, 25, 5, 0, 1, False, (0, 228, 200), False),
        BDD100kClass('double yellow vertical', 35, 26, 3, 0, 1, False, (0, 170, 200), False),
        BDD100kClass('double yellow vertical dashed', 51, 27, 3, 1, 1, False, (0, 123, 200), False),
        BDD100kClass('single yellow vertical dashed', 55, 28, 3, 1, 1, False, (0, 70, 200), False),
        BDD100kClass('double white vertical dashed', 50, 29, 2, 1, 1, False, (0, 17, 200), False),
    ]

    # lane_mask_ids = []
    # for lane_mask_class in lane_mask_classes:
    #     lane_mask_ids.append(lane_mask_class[1])

    simplified_lane_classes = [
        BDD100kClass('Background', 0, 0, 'background', 0, True, False, (0, 0, 0), False),
        BDD100kClass('line', 1, 1, 'line', 0, True, False, (255, 255, 255), False),
    ]

    drivable_area_classes = [
        BDD100kClass('background', 2, 2, -1, -1, -1, False, (1, 1, 1), False),
        BDD100kClass('Drivable Area', 0, 0, -1, -1, -1, False, (255, 1, 1), False),
        BDD100kClass('Alternative Area', 1, 1, -1, -1, -1, False, (1, 1, 255), False),
    ]


    detection_classes = [
        BDD100kClass('pedestrian', 1, 0, -1, -1, -1, False, (220, 20, 60), True),
        BDD100kClass('rider', 2, 1, -1, -1, -1, False, (255, 0, 0), True),
        BDD100kClass('car', 3, 2, -1, -1, -1, False, (0, 0, 142), True),
        BDD100kClass('truck', 4, 3, -1, -1, -1, False, (0, 0, 70), True),
        BDD100kClass('bus', 5, 4, -1, -1, -1, False, (0, 60, 100), True),
        BDD100kClass('train', 6, 5, -1, -1, -1, False, (0, 80, 100), True),
        BDD100kClass('motorcycle', 7, 6, -1, -1, -1, False, (0, 0, 230), True),
        BDD100kClass('bicycle', 8, 7, -1, -1, -1, False, (119, 11, 32), True),
        BDD100kClass('traffic light', 9, 8, -1, -1, -1, False, (250, 170, 30), True),
        BDD100kClass('traffic sign', 10, 9, -1, -1, -1, False, (192, 192, 192), True),
    ]

    culane_classes = [
        BDD100kClass('road', 0, 0, -1, -1, -1, False, (0, 0, 0), False),
        BDD100kClass('Line 1', 1, 1, -1, -1, -1, False, (180, 200, 30), False),
        BDD100kClass('Line 2', 2, 2, -1, -1, -1, False, (200, 100, 1), False),
        BDD100kClass('Line 3', 3, 3, -1, -1, -1, False, (1, 100, 200), False),
        BDD100kClass('Line 4', 4, 4, -1, -1, -1, False, (30, 200, 180), False),
    ]

    single_ids_kernel = np.ones((16, 16), np.uint8)
    double_ids_kernel = np.ones((40, 40), np.uint8)
    only_strengten_kernel = np.ones((10, 10), np.uint8)

    def __init__(
            self,
            use_culane: Optional[Any] = False,
            prediction_mode: Optional[Any] = False,
            number_of_temporal_frames=None,
            temporal_stride=None,
            **kwargs
    ) -> None:
        super(BDD100k, self).__init__(**kwargs)

        self.prediction_mode = prediction_mode
        self.root = os.path.join(self.root, "bdd100k")

        self.targets_dir = os.path.join(self.root, 'labels')
        self.images = []
        self.targets = []
        self.use_culane = use_culane
        self.number_of_temporal_frames = number_of_temporal_frames
        self.temporal_stride = temporal_stride

        if not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders '
                               'for the specified "split" is inside the "root" directory')

        self.image_id = 0
        if "det_20" in self.target_type:
            self.anno_dict = self.create_dict_from_json(self.split)

        if "pan_seg-coco_panoptic-10k" in self.target_type:
            annotation_json = os.path.join(self.targets_dir, "pan_seg", 'coco_panoptic', 'coco.json')
            with open(annotation_json, 'r') as file:
                annotations_raw = json.load(file)

            self.categories = annotations_raw["categories"]
            self.panoptic_annotations = {}
            for annotation in annotations_raw['annotations']:
                self.panoptic_annotations[annotation["file_name"].split("/")[-1]] = annotation['segments_info']

        if self.split == "trainval":
            self.create_index("train")
            self.create_index("val")
        else:
            self.create_index(self.split)

    def create_index(self, split):
        images_dir = os.path.join(self.root, 'images/100k', split)
        video_frames_dir = os.path.join(self.root, 'video_frames', split)

        if self.number_of_temporal_frames is None:
            list_of_files = os.listdir(images_dir)

        else:
            list_of_files = os.listdir(video_frames_dir)

        list_of_files.sort()
        for file_name in list_of_files:
            target_types = []
            # Detection annotation file has missing entries,
            # so this condition checks if image has a valid entry
            if 'det_20' in self.target_type and not file_name in self.anno_dict.keys():
                continue
            for t in self.target_type:
                if t == 'det_20':
                    target_types.append(self.anno_dict[file_name])
                else:
                    targets_dir = os.path.join(self.targets_dir, t.split('-')[0], t.split('-')[1], split)
                    target_name = '{}.{}'.format(file_name.split('.')[0],
                                                    self._get_target_suffix(t))

                    target_types.append(os.path.join(targets_dir, target_name))
            if self.number_of_temporal_frames is None:
                self.images.append(os.path.join(images_dir, file_name))
            else:
                image_list = []
                image_filename_list = os.listdir(os.path.join(video_frames_dir, file_name))
                image_filename_list.sort()
                start_index = 0
                end_index = self.number_of_temporal_frames * self.temporal_stride
                if self.prediction_mode:
                    start_index += self.temporal_stride
                    end_index += self.temporal_stride

                for image_file in image_filename_list[start_index: end_index: self.temporal_stride]:
                    image_list.append(os.path.join(video_frames_dir, file_name, image_file))

                image_list = image_list[::-1]
                self.images.append(image_list)
            self.targets.append(target_types)

    def create_dict_from_json(self, split):
        print("Caching json file into dictionary")
        image_id = 0
        detection_targets_dir = os.path.join(self.targets_dir, "det_20", 'det_' + split + '.json')
        with open(detection_targets_dir, 'r') as file:
            annos = json.load(file)

        anno_dict = {}
        for anno in annos:
            anno_dict[anno['name']] = {'attributes': anno['attributes'],
                                       'timestamp' : anno['timestamp'],
                                       'labels'    : anno['labels'] if 'labels' in anno.keys() else [],
                                       'image_name': anno['name'],
                                       'image_id'  : image_id
                                       }
            image_id += 1

        return anno_dict

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        if not isinstance(self.images[index], List):
            image = Image.open(self.images[index]).convert('RGB')
        else:
            image = []
            for single_image in self.images[index]:
                image.append(Image.open(single_image).convert('RGB'))

        if not isinstance(self.images[index], List):
            info_dict = {"image_id": self.images[index].split('/')[-1][:-4],
                         "image_size": np.asarray(image).shape}
        else:
            # TODO, image id for the temporal dataloader
            pass

        targets: Any = []
        for i, target_type in enumerate(self.target_type):
            if target_type == 'det_20':
                target = self.targets[index][i]
            else:

                target = Image.open(self.targets[index][i]).convert('P')
                #target = Image.open(self.targets[index][i])
                target = np.array(target)

                if target_type == "lane-bitmasks":
                    for label in self.only_strengten:
                        self._apply_morphology_operation(target, self.only_strengten_kernel, label,
                                                         operation="dilation", null_label=255)

                    for label in self.single_ids:
                        self._apply_morphology_operation(target, self.single_ids_kernel, label,
                                                         operation="closing", null_label=255)

                    for label in self.double_ids:
                        self._apply_morphology_operation(target, self.double_ids_kernel, label,
                                                         operation="closing", null_label=255)

                if target_type == "lane-simplified":
                    target[target > 0] = 1

                elif self.use_culane and index >= self.len_without_culane:
                    target[:] = 255

                if target_type == "lane-masks" or \
                        target_type == "lane-bitmasks" or \
                        target_type == "lane-bitmasks_morph":

                    target = self._convert_mask_labels(target, target_type)

                target = Image.fromarray(target)
            targets.append(target)

        #target = tuple(targets)

        if self.transforms is not None:
            image, target = self.transforms(image, targets)

        if self.return_dataset_info is True:
            return image, target, info_dict
        else:
            return image, target

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, target_type: str) -> str:
        return 'png'

    def _valid_target_types(self):
        valid_target_types = ("lane-masks", "lane-colormaps", "lane-simplified", "drivable-masks", "drivable-colormaps",
                              "det_20", "lane-bitmasks", "lane-bitmasks_morph", "lane-bitmasks_eliminated",
                              "sem_seg-masks", "lane-deeplabv3_resnet18_culane_640x368")
        return valid_target_types

    def _valid_splits(self):
        valid_splits = ["train", "test", "val", "trainval"]
        return valid_splits

    def _determine_classes(self):
        classes_dict = {}
        if not isinstance(self.mask_convert, List):
            self.mask_convert = [self.mask_convert] * len(self.target_type)

        # assert len(self.mask_convert) == len(self.target_type), \
        #     "mask convert and target type lists does not have equal sizes"

        for mask_convert, target_type in zip(self.mask_convert, self.target_type):
            if target_type == "lane-masks":
                # TODO fix this part if needed
                classes = self.lane_mask_classes
            elif target_type == "drivable-masks" or target_type == "drivable-colormaps":
                classes = self.drivable_area_classes
            elif target_type == "lane-simplified":
                classes = self.simplified_lane_classes
            elif target_type == "sem_seg-masks"  or target_type == "sem_seg-masks-10k":
                classes = self.segmentation_classes
            elif target_type == "lane-bitmasks" or \
                    target_type == "lane-bitmasks_morph":
                self._create_lane_class_id_list()
                if mask_convert:
                    self._modify_training_classes_for_solid_dashed()
                else:
                    self._modify_training_classes()
                classes = self.lane_mask_classes
            elif target_type == "det_20":
                classes = self.detection_classes  # TODO, correct this if needed
            elif target_type == "lane-deeplabv3_resnet18_culane_640x368":
                classes = self.culane_classes
            else:
                raise ValueError("Not supported for the time being")
            classes_dict.update({target_type: classes})
        if len(self.target_type) == 1:
            return list(classes_dict.values())[0]
        else:
            return classes_dict

    def _modify_training_classes(self):
        train_id_to_assign = 0
        for cls_index, bdd_cls in enumerate(self.lane_mask_classes):
            if bdd_cls.lane_direction_id == 1 or bdd_cls.lane_category_id == 0 \
                    or bdd_cls.lane_category_id == 4 or bdd_cls.lane_category_id == 5 \
                    or bdd_cls.lane_category_id == 1:
                bdd_cls = bdd_cls._replace(train_id=255)
                bdd_cls = bdd_cls._replace(ignore_in_eval=True)
            else:
                bdd_cls = bdd_cls._replace(train_id=train_id_to_assign)
                train_id_to_assign += 1
            self.lane_mask_classes[cls_index] = bdd_cls

    def _modify_training_classes_for_solid_dashed(self):
        solid = self.BDD100kClass('solid', -1, 1, -1, 0, 0, False, (255, 0, 0), False)
        dashed = self.BDD100kClass('dashed', -1, 2, -1, 1, 0, False, (0, 0, 255), False)
        self.lane_mask_classes.insert(1, solid)
        self.lane_mask_classes.insert(2, dashed)
        for cls_index, bdd_cls in enumerate(self.lane_mask_classes):
            if bdd_cls.lane_direction_id == 1 or bdd_cls.lane_category_id == 0 \
                    or bdd_cls.lane_category_id == 4:
                bdd_cls = bdd_cls._replace(train_id=255)
                bdd_cls = bdd_cls._replace(ignore_in_eval=True)
            else:
                if cls_index == 0 or cls_index == 1 or cls_index == 2:
                    continue
                else:
                    bdd_cls = bdd_cls._replace(train_id=bdd_cls.lane_style_id + 1)
                    bdd_cls = bdd_cls._replace(ignore_in_eval=True)
            self.lane_mask_classes[cls_index] = bdd_cls

    def _create_lane_class_id_list(self):
        double_ids = []
        single_ids = []
        only_strengten = []
        for lane_mask_class in self.lane_mask_classes:
            if lane_mask_class.id == 0:
                continue
            if 'double' in self.lane_category_name_dict[lane_mask_class.lane_category_id]:
                double_ids.append(lane_mask_class.id)
                only_strengten.append(lane_mask_class.id)
        for lane_mask_class in self.lane_mask_classes:
            if lane_mask_class.id == 0:
                continue
            if 'single' in self.lane_category_name_dict[lane_mask_class.lane_category_id]:
                single_ids.append(lane_mask_class.id)
                only_strengten.append(lane_mask_class.id)

        for lane_mask_class in self.lane_mask_classes:
            if lane_mask_class.id == 0:
                continue
            if 'curb' in self.lane_category_name_dict[lane_mask_class.lane_category_id] or \
                    'crosswalk' in self.lane_category_name_dict[lane_mask_class.lane_category_id]:
                only_strengten.append(lane_mask_class.id)

        self.double_ids = double_ids
        self.single_ids = single_ids
        self.only_strengten = only_strengten

    def get_number_of_bdd_classes(self):
        number_of_class = 0
        for bdd_cls in self.lane_mask_classes:
            if bdd_cls.ignore_in_eval is False:
                number_of_class += 1
        return number_of_class

