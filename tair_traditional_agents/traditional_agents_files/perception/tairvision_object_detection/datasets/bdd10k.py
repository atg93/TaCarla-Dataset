import json
import os
import numpy as np
from collections import namedtuple, OrderedDict
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, NamedTuple
from PIL import Image
from .generic_data import GenericSegmentationVisionDataset
import cv2

from tairvision.references.detection.coco_utils import convert_coco_poly_to_mask

class BDD10k(GenericSegmentationVisionDataset):
    BDD100kClass = namedtuple('BDD100kClass', ['name', 'id', 'train_id',
                                               'lane_category_id',
                                               'lane_style_id',
                                               'lane_direction_id',
                                               'ignore_in_eval', 'color', 'has_instances'])

    segmentation_classes = [
        BDD100kClass('road', 0, 0, -1, -1, -1, False, (128, 64, 128), False),
        BDD100kClass('sidewalk', 1, 1, -1, -1, -1, False, (244, 35, 232), False),
        BDD100kClass('building', 2, 2, -1, -1, -1, False, (70, 70, 70), False),
        BDD100kClass('wall', 3, 3, -1, -1, -1, False, (102, 102, 156), False),
        BDD100kClass('fence', 4, 4, -1, -1, -1, False, (190, 153, 153), False),
        BDD100kClass('pole', 5, 5, -1, -1, -1, False, (153, 153, 153), False),
        BDD100kClass('traffic light', 6, 6, -1, -1, -1, False, (250, 170, 30), True),
        BDD100kClass('traffic sign', 7, 7, -1, -1, -1, False, (220, 220, 0), True),
        BDD100kClass('vegetation', 8, 8, -1, -1, -1, False, (107, 142, 35), False),
        BDD100kClass('terrain', 9, 9, -1, -1, -1, False, (152, 251, 152), False),
        BDD100kClass('sky', 10, 10, -1, -1, -1, False, (70, 130, 180), False),
        BDD100kClass('person', 11, 11, -1, -1, -1, False, (220, 20, 60), True),
        BDD100kClass('rider', 12, 12, -1, -1, -1, False, (255, 0, 0), True),
        BDD100kClass('car', 13, 13, -1, -1, -1, False, (0, 0, 142), True),
        BDD100kClass('truck', 14, 14, -1, -1, -1, False, (0, 0, 70), True),
        BDD100kClass('bus', 15, 15, -1, -1, -1, False, (0, 60, 100), True),
        BDD100kClass('train', 16, 16, -1, -1, -1, False, (0, 80, 100), True),
        BDD100kClass('motorcycle', 17, 17, -1, -1, -1, False, (0, 0, 230), True),
        BDD100kClass('bicycle', 18, 18, -1, -1, -1, False, (119, 11, 32), True),
    ]

    panoptic_segmentation_classes = [
        BDD100kClass('unlabeled', 0, 0, -1, -1, -1, True, (0, 0, 0), False),
        BDD100kClass('dynamic', 1, 1, -1, -1, -1, False, (111, 74, 0), False),
        BDD100kClass('ego vehicle', 2, 2, -1, -1, -1, False, (120, 10, 10), False),
        BDD100kClass('ground', 3, 3, -1, -1, -1, False, (81, 0, 81), False),
        BDD100kClass('static', 4, 4, -1, -1, -1, False, (111, 111, 0), False),
        BDD100kClass('parking', 5, 5, -1, -1, -1, False, (250, 170, 160), False),
        BDD100kClass('rail track', 6, 6, -1, -1, -1, False, (230, 150, 140), False),
        BDD100kClass('road', 7, 7, -1, -1, -1, False, (128, 64, 128), False),
        BDD100kClass('sidewalk', 8, 8, -1, -1, -1, False, (244, 35, 232), False),
        BDD100kClass('bridge', 9, 9, -1, -1, -1, False, (150, 100, 100), False),
        BDD100kClass('building', 10, 10, -1, -1, -1, False, (70, 70, 70), False),
        BDD100kClass('fence', 11, 11, -1, -1, -1, False, (190, 153, 153), False),
        BDD100kClass('garage', 12, 12, -1, -1, -1, False, (150, 150, 150), False),  # TODO decide whether to include
        BDD100kClass('guard rail', 13, 13, -1, -1, -1, False, (180, 165, 180), False),
        BDD100kClass('tunnel', 14, 14, -1, -1, -1, False, (150, 120, 90), False),
        BDD100kClass('wall', 15, 15, -1, -1, -1, False, (102, 102, 156), False),
        BDD100kClass('banner', 16, 16, -1, -1, -1, False, (255, 255, 128), False),
        BDD100kClass('billboard', 17, 17, -1, -1, -1, False, (250, 174, 30), False),  # TODO billboard class is not found on mapillary, what is the corresponding class, I think it is signage on mapillary
        BDD100kClass('lane divider', 18, 18, -1, -1, -1, False, (128, 128, 128), False), #  TODO do we need to include this class in evaluation
        BDD100kClass('parking sign', 19, 19, -1, -1, -1, False, (81, 0, 81), False),  #  TODO
        BDD100kClass('pole', 20, 20, -1, -1, -1, False, (153, 153, 153), False),
        BDD100kClass('pole group', 21, 21, -1, -1, -1, False, (153, 153, 153), False),  #  TODO
        BDD100kClass('street light', 22, 22, -1, -1, -1, False, (210, 170, 100), False),
        BDD100kClass('traffic cone', 23, 23, -1, -1, -1, False, (210, 60, 60), False), #  TODO bu ne
        BDD100kClass('traffic device', 24, 24, -1, -1, -1, False, (81, 0, 81), False),  #  TODO
        BDD100kClass('traffic light', 25, 25, -1, -1, -1, False, (250, 170, 30), False),
        BDD100kClass('traffic sign', 26, 26, -1, -1, -1, False, (192, 192, 192), False),
        BDD100kClass('traffic sign frame', 27, 27, -1, -1, -1, False, (128, 128, 128), False),  #  TODO
        BDD100kClass('terrain', 28, 28, -1, -1, -1, False, (152, 251, 152), False),
        BDD100kClass('vegetation', 29, 29, -1, -1, -1, False, (107, 142, 35), False),
        BDD100kClass('sky', 30, 30, -1, -1, -1, False, (70, 130, 180), False),
        BDD100kClass('person', 31, 31, -1, -1, -1, False, (220, 20, 60), True),
        BDD100kClass('rider', 32, 32, -1, -1, -1, False, (255, 0, 0), True),
        BDD100kClass('bicycle', 33, 33, -1, -1, -1, False, (119, 11, 32), True),
        BDD100kClass('bus', 34, 34, -1, -1, -1, False, (0, 60, 100), True),
        BDD100kClass('car', 35, 35, -1, -1, -1, False, (0, 0, 142), True),
        BDD100kClass('caravan', 36, 36, -1, -1, -1, False, (0, 0, 90), True),
        BDD100kClass('motorcycle', 37, 37, -1, -1, -1, False, (0, 0, 230), True),
        BDD100kClass('trailer', 38, 38, -1, -1, -1, False, (0, 0, 110), True),
        BDD100kClass('train', 39, 39, -1, -1, -1, False, (0, 80, 100), True),
        BDD100kClass('truck', 40, 40, -1, -1, -1, False, (0, 0, 70), True),

    ]

    segmentation_merged_classes = [
        BDD100kClass('background', -1, 1, -1, -1, -1, False, (220, 20, 60), False),
        BDD100kClass('lane', -1, 1, -1, -1, -1, False, (220, 20, 60), False),
        BDD100kClass('person', 31, 2, -1, -1, -1, False, (220, 20, 60), True),
        BDD100kClass('rider', 32, 3, -1, -1, -1, False, (255, 0, 0), True),
        BDD100kClass('car', 35, 4, -1, -1, -1, False, (0, 0, 142), True),
        BDD100kClass('truck', 40, 5, -1, -1, -1, False, (0, 0, 70), True),
        BDD100kClass('bus', 34, 6, -1, -1, -1, False, (0, 60, 100), True),
        BDD100kClass('train', 39, 7, -1, -1, -1, False, (0, 80, 100), True),
        BDD100kClass('motorcycle', 37, 8, -1, -1, -1, False, (0, 0, 230), True),
        BDD100kClass('bicycle', 33, 9, -1, -1, -1, False, (119, 11, 32), True),
        BDD100kClass('traffic light', 25, 10, -1, -1, -1, False, (250, 170, 30), True),
        BDD100kClass('traffic sign', 26, 11, -1, -1, -1, False, (220, 220, 0), True),
        BDD100kClass('road', 7, 12, -1, -1, -1, False, (128, 64, 128), False),
        BDD100kClass('sidewalk', 8, 13, -1, -1, -1, False, (244, 35, 232), False),
        BDD100kClass('ego vehicle', 2, 14, -1, -1, -1, False, (120, 10, 10), False),
        BDD100kClass('vegetation', 29, 15, -1, -1, -1, False, (107, 142, 35), False),
        BDD100kClass('sky', 30, 16, -1, -1, -1, False, (70, 130, 180), False),
        BDD100kClass('building', 10, 17, -1, -1, -1, False, (70, 70, 70), False),
        BDD100kClass('pole', 20, 18, -1, -1, -1, False, (153, 153, 153), False),
        BDD100kClass('terrain', 21, 19, -1, -1, -1, False, (152, 251, 152), False),
        BDD100kClass('fence', 22, 20, -1, -1, -1, False, (190, 153, 153), False),
    ]

    def __init__(
            self,
            use_culane: Optional[Any] = False,
            prediction_mode: Optional[Any] = False,
            number_of_temporal_frames=None,
            temporal_stride=None,
            **kwargs
    ) -> None:
        super(BDD10k, self).__init__(**kwargs)

        if use_culane:
            self.culane_root = os.path.join(self.root, "culane")
            self.culane_list = os.path.join(self.culane_root, "list", "{}_gt.txt".format(self.split))

        self.prediction_mode = prediction_mode
        self.root = os.path.join(self.root, "bdd100k")
        self.images_dir = os.path.join(self.root, 'images/10k', self.split)
        self.video_frames_dir = os.path.join(self.root, 'video_frames', self.split)
        # self.images_sub_dir = os.path.join(self.root, 'images/10k', self.split)
        self.list_to_delete = ["3d581db5-2564fb7e.jpg", "52e3fd10-c205dec2.jpg", "78ac84ba-07bd30c2.jpg",
                          "781756b0-61e0a182.jpg", "9342e334-33d167eb.jpg", "80a9e37d-e4548ac1.jpg"]
        # if "10k" in self.target_type[0]:
        #     # ff1e4d6d-f4d85cfd.png, ff3d3536-04986e25.png, ff3da814-c3463a43.png, fee92217-63b3f87f.png 10k train set problem
        #     # 3d581db5-2564fb7e.jpg, 52e3fd10-c205dec2.jpg, 78ac84ba-07bd30c2.jpg, 781756b0-61e0a182.jpg 10k train set problem
        #
        #     # ff55861e-a06b953c.png, ff7b98c7-3cb964ac.png 10k val set problem
        #     # 9342e334-33d167eb.jpg, 80a9e37d-e4548ac1.jpg 10k val set problem
        #     self.images_dir = os.path.join(self.root, 'images/10k', self.split)
        self.targets_dir = os.path.join(self.root, 'labels')
        self.images = []
        self.targets = []
        self.use_culane = use_culane
        self.number_of_temporal_frames = number_of_temporal_frames
        self.temporal_stride = temporal_stride

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders '
                               'for the specified "split" is inside the "root" directory')

        self.image_id = 0
        if "ins_seg-10k" in self.target_type:
            self.detection_targets_dir = os.path.join(self.targets_dir, "ins_seg", "polygons",
                                                      'ins_seg_' + self.split + '.json')
            # self.detection_targets_dir = 'ins_seg_train.json'
            self.anno_dict = self.create_ins_dict_from_json()


        if "pan_seg-coco_panoptic-10k" in self.target_type:
            annotation_json = os.path.join(self.targets_dir, "pan_seg", 'coco_panoptic', 'coco.json')
            with open(annotation_json, 'r') as file:
                annotations_raw = json.load(file)

            self.categories = annotations_raw["categories"]
            self.panoptic_annotations = {}
            for annotation in annotations_raw['annotations']:
                self.panoptic_annotations[annotation["file_name"].split("/")[-1]] = annotation['segments_info']

        self.create_index()

    def create_index(self):
        # images_sub_dir_filenames = os.listdir(self.images_sub_dir)
        if self.number_of_temporal_frames is None:
            list_of_files = os.listdir(self.images_dir)
            # if "10k" in self.target_type[0]:
            for file in self.list_to_delete:
                if file in list_of_files:
                    list_of_files.remove(file)
        else:
            list_of_files = os.listdir(self.video_frames_dir)

        list_of_files.sort()
        for file_name in list_of_files:
            target_types = []
            # Detection annotation file has missing entries,
            # so this condition checks if image has a valid entry
            if 'ins_seg-10k' in self.target_type and not file_name in self.anno_dict.keys():
                continue

            if 'ins_seg-10k' in self.target_type:
                if len(self.anno_dict[file_name]['labels']) == 0:
                    continue

            for t in self.target_type:
                if t == 'ins_seg-10k':
                    target_types.append(self.anno_dict[file_name])
                else:
                    targets_dir = os.path.join(self.targets_dir, t.split('-')[0], t.split('-')[1], self.split)
                    target_name = '{}.{}'.format(file_name.split('.')[0],
                                                    self._get_target_suffix(t))
                    if t == 'sem_seg-masks':
                        if file_name in list_of_files:
                            target_types.append(os.path.join(targets_dir, target_name))
                        else:
                            target_types.append(None)
                    else:
                        target_types.append(os.path.join(targets_dir, target_name))

            if self.number_of_temporal_frames is None:
                self.images.append(os.path.join(self.images_dir, file_name))
            else:
                image_list = []
                image_filename_list = os.listdir(os.path.join(self.video_frames_dir, file_name))
                image_filename_list.sort()
                start_index = 0
                end_index = self.number_of_temporal_frames * self.temporal_stride
                if self.prediction_mode:
                    start_index += self.temporal_stride
                    end_index += self.temporal_stride

                for image_file in image_filename_list[start_index: end_index: self.temporal_stride]:
                    image_list.append(os.path.join(self.video_frames_dir, file_name, image_file))

                image_list = image_list[::-1]
                self.images.append(image_list)
            self.targets.append(target_types)

        self.len_without_culane = len(self.images)


    def create_ins_dict_from_json(self):
        print("Caching json file into dictionary")
        image_id = 0
        with open(self.detection_targets_dir, 'r') as file:
            annos = json.load(file)

        anno_dict = OrderedDict()

        for anno in annos:
            anno_dict[anno['name']] = {'labels'    : anno['labels'] if 'labels' in anno.keys() else [],
                                       'image_name': anno['name'],
                                       'image_id'  : image_id
                                       }
            if 'img_info' in anno:
                anno_dict[anno['name']]['img_info'] = anno['img_info']
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
            elif target_type == 'ins_seg-10k':
                target = self.targets[index][i]
                segmentations = []
                for labels in target['labels']:
                    vert_list = [np.array(p['vertices']).flatten().tolist() for p in labels['poly2d']]
                    segmentations.append(vert_list)

                masks = convert_coco_poly_to_mask(segmentations, image.height, image.width)
                assert len(masks) == len(segmentations)
                for labels, mask in zip(target['labels'], masks):
                    labels['mask'] = mask.numpy()
            else:
                if target_type == 'sem_seg-masks':
                    target_filename = self.targets[index][i]
                    if target_filename:
                        target = Image.open(self.targets[index][i]).convert('P')
                    else:
                        if isinstance(image, List):
                            target = 255 * np.ones(np.array(image[0]).shape[:2], dtype=np.uint8)
                        else:
                            target = 255 * np.ones(np.array(image).shape[:2], dtype=np.uint8)
                elif 'pan_seg' in target_type:
                    target = Image.open(self.targets[index][i])
                    if "coco" in target_type:
                        target_key = self.targets[index][i].split('/')[-1]
                        target_info = self.panoptic_annotations[target_key]
                        info_dict.update({"target_info": target_info})
                        target = np.array(target)
                        target = self.rgb2id(target)
                        target = Image.fromarray(target)
                    else:
                        target = np.array(target)[:, :, 0] - 1
                else:
                    target = Image.open(self.targets[index][i]).convert('P')

                target = np.array(target)

                if target_type == "sem_seg-deeplabv3_resnet50_mapillary_1856x1024_apex" or \
                        target_type == "pan_seg-bitmasks-semantic-10k":
                    target = self._convert_mask_labels(target, target_type)

                target = Image.fromarray(target)
                # save_name = self.targets[index][i].replace("bitmasks", "bitmasks_morph")
                # target.save(save_name)
            targets.append(target)

        #target = tuple(targets)

        if self.transforms is not None:
            image, target = self.transforms(image, targets)

        if "coco" in self.target_type[0]:
            target = self.maskformer_panoptic_format_converter(target_info, target)

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
        valid_target_types = ("sem_seg-masks", "sem_seg-masks-10k", "pan_seg-bitmasks-semantic-10k",
                              "pan_seg-coco_panoptic-10k", "ins_seg-10k", "sem_seg-deeplabv3_resnet50_mapillary_1856x1024_apex")
        return valid_target_types

    def _valid_splits(self):
        valid_splits = ["train", "test", "val"]
        return valid_splits

    def _determine_classes(self):
        classes_dict = {}
        if not isinstance(self.mask_convert, List):
            self.mask_convert = [self.mask_convert] * len(self.target_type)

        # assert len(self.mask_convert) == len(self.target_type), \
        #     "mask convert and target type lists does not have equal sizes"

        for mask_convert, target_type in zip(self.mask_convert, self.target_type):

            if target_type == "sem_seg-masks"  or target_type == "sem_seg-masks-10k":
                classes = self.segmentation_classes
            elif target_type == "ins_seg-10k":
                classes = self.panoptic_segmentation_classes[-10:]  # TODO, correct this if needed
            elif target_type == "lane-deeplabv3_resnet18_culane_640x368":
                classes = self.culane_classes
            elif target_type == "sem_seg-deeplabv3_resnet50_mapillary_1856x1024_apex":
                targets_dir = os.path.join("/datasets/bdd100k/labels",
                                           target_type.split('-')[0],
                                           target_type.split('-')[1],
                                           "class_settings.json")
                with open(os.path.join(targets_dir), 'r') as f:
                    mapillary_class = json.load(f)
                MapillaryClass = namedtuple('MapillaryClass', ['name', 'id', 'train_id', 'category1', 'category2',
                                                               'has_instances', 'ignore_in_eval', 'color'])
                classes = []
                for single_class in mapillary_class:
                    classes.append(MapillaryClass(**single_class))
            elif target_type == "pan_seg-bitmasks-semantic-10k" or target_type == "pan_seg-coco_panoptic-10k":
                classes = self.panoptic_segmentation_classes
                # classes = self.modify_training_classes(classes)
            else:
                raise ValueError("Not supported for the time being")
            classes_dict.update({target_type: classes})
        if len(self.target_type) == 1:
            return list(classes_dict.values())[0]
        else:
            return classes_dict



