from abc import ABC
import os
from .generic_data import GenericSegmentationVisionDataset
import json
import torch
from collections import namedtuple
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
from tairvision.utils import PanopticDeepLabTargetGenerator
from tairvision.ops.boxes import masks_to_boxes, box_xyxy_to_cxcywh


class Mapillary(GenericSegmentationVisionDataset, ABC):
    MapillaryClass = namedtuple('MapillaryClass', ['name', 'id', 'train_id', 'category1', 'category2',
                                                   'has_instances', 'ignore_in_eval', 'color'])
    version = "v2.0"
    panoptic_json_file_name = "panoptic_2020.json"

    def __init__(self, small_instance_area=4096, small_instance_weight=3, **kwargs) -> None:
        self.config_file = f'config_{self.version}.json'
        kwargs['root'] = os.path.join(kwargs['root'], "mapillary")
        super(Mapillary, self).__init__(**kwargs)
        split_dict = {'train': 'training',
                      'val': 'validation',
                      'test': 'testing'}

        image_path = f'{split_dict[self.split]}/images/'
        label_path = f'{split_dict[self.split]}/{self.version}/labels/'
        instance_path = f'{split_dict[self.split]}/{self.version}/instances/'
        panoptic_path = f'{split_dict[self.split]}/{self.version}/panoptic/'

        self.image_path_format: str = image_path + "{}.jpg"
        self.label_path_format: str = label_path + "{}.png"
        self.instance_path_format: str = instance_path + "{}.png"
        self.panoptic_path_format: str = panoptic_path + "{}.png"

        if 'instance' in self.target_type[0]:

            problematic_ids = ["zM3_5Do7DOHTg8E7znzzZA", "pAJn1J0Shqn4jqYIxDWvfQ", "bZX73mF1IMDgsRIvbO6pfA",
                               "5bzCVDcbUOkQzXuNeFY7Pg", "aTqMVsUK7RUtmm2aFUZrCQ", "uuAbGKUAdR4udDeUuBVrHA",
                               "lhcO4Mis5abhkD2onf3ljw", "PPlh1X-ZiacbCL_KthPA8Q", "PeFGnKwBCQlcqrkubVz7Tg",
                               "cccN4AnQOeRCzYjGbRCRuA", "ag_jPVdDqk3pgs71bk1ZrQ", "8ki0ngSC8PN9qBtA7urAFw",
                               "c33c0LCGoXl9P6mFprE1ww", "gs46ImCkvqW4kI3jt4rZSA", "un7a-2cX8xTnx7CWAqOR9w",
                               "9ITSm456nenlvnGoiekdOA", "dLDQVvjx-pum56rk5ikbmQ", "6lil341DJG1jpldI6bIufQ",
                               "3FQqX7wPSghdZVX6KYZDOw", "UlL5aLls4YQTQ_ZyCvdJhg", "dle7znBkQpEGyes5XwzwkA",
                               "1Jql_9CBH7fSKpyo95aiJg", "6CgEv4XYpi-pManpHabzaw", "R79QIHNA6E6Sf9b-6qk55A",
                               "hSGB2Clh3dsbOvjkYavt1w", "Rbf4pVIrRbgwBqshccJt8w", "XI0q6hOfgWLvi7PqkKYVsw",
                               "dWWrC5-aMlmiyfesLzz4tQ", "GsyBd_8RDgMt0vmZv061-g", "4rEOHXD8dnJwWUudFrw2wQ",
                               "BsfvYZP-oz-Ig6Qa4K-fLA", "4heOdBWknMrgW-tu8b1PTw", "CLygNVtHpXXGvoK8KJZyww",
                               "KnWRc4spbyohCvDraEEHAA", "VExMu3C02RhOmmJiOux5qQ", "ztaR_O637iVPndNgjR7nZA",
                               "I6wgOd062i8cWeRhj6bC3g", "40ye4e3WlMSUTklWTEp8dg"
                               ]

        else:
            problematic_ids = []

        image_path_full_directory = os.listdir(os.path.join(self.root, image_path))
        self.image_id_list = [image[:-4] for image in image_path_full_directory if image[:-4] not in problematic_ids]
        self.image_id_list.sort()

        # For debug purposes
        # self.image_id_list = [self.image_id_list[3]]

        panoptic_json_file = f"{split_dict[self.split]}/{self.version}/panoptic/{self.panoptic_json_file_name}"
        panoptic_json_path = os.path.join(self.root, panoptic_json_file)

        self.panoptic_json_file = panoptic_json_path
        self.panoptic_folder_path = os.path.join(self.root, panoptic_path)

        self.panoptic = None
        if "panoptic" in self.target_type[0]:
            with open(panoptic_json_path) as panoptic_file:
                panoptic = json.load(panoptic_file)
            self.panoptic = panoptic
            self.panoptic_annotations: Dict = {}
            for annotation in panoptic["annotations"]:
                for segment_info in annotation['segments_info']:
                    segment_info["category_id"] -= 1
                self.panoptic_annotations[annotation["image_id"]] = annotation['segments_info']
            if "panoptic-deeplab" == self.target_type[0]:
                # TODO sigma is set to 10, default is shown as 8 but detectron2 config seems to using 10
                self.panoptic_target_generator = PanopticDeepLabTargetGenerator(
                    ignore_label=self.panoptic_ignore_label,
                    thing_list=self.thing_list,
                    sigma=10,
                    ignore_stuff_in_offset=True,
                    small_instance_area=small_instance_area,
                    small_instance_weight=small_instance_weight,
                    ignore_crowd_in_semantic=False
                )

            self.categories = panoptic["categories"]
        self.reversed_class_mapping = {key: value + 1 for (key, value) in self.reversed_class_mapping.items()}

    def __getitem__(self, index: int) -> Union[Tuple[Any, Any], Tuple[Any, Any, Any]]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image_id = self.image_id_list[index]
        image_path = os.path.join(self.root, self.image_path_format.format(image_id))
        image = Image.open(image_path)

        image_numpy = np.array(image)
        h, w = image_numpy.shape[0], image_numpy.shape[1]

        info_dict = {"image_id": image_id,
                     "file_name": image_id + '.png',
                     "image_size": np.asarray(image).shape}

        targets: List = []
        target_info = None
        for i, t in enumerate(self.target_type):
            target = self.load_target(target_type=t, image_id=image_id)
            targets.append(target)

        target = targets if len(targets) > 1 else targets[0]
        if isinstance(target, Tuple):
            target_info = target[1]
            info_dict.update({"target_info": target_info})
            target = target[0]

        if "panoptic" in self.target_type[0]:
            target = self.rgb2id(target)

        target = Image.fromarray(target)

        if self.target_type[0] == "panoptic-deeplab":
            if self.transforms is not None:
                image, target = self.transforms(image, target)

            target, target_info = self.panoptic_ground_truth_conversion_to_constant_standard(
                panoptic=target,
                segment_infos=target_info
            )

            target = self.panoptic_target_generator(panoptic=np.array(target), segments=target_info)
            for key, value in target.items():
                if key == "semantic" or key == "foreground" or key == "mask":
                    value = Image.fromarray(value.astype(np.uint8))
                else:
                    pass
                target[key] = value
            _, target = self.ToTensor(None, target)

        else:
            target_original = target
            image_original = image
            target = None
            count = 0
            while target is None:
                if self.transforms is not None:
                    image, target = self.transforms(image_original, target_original)

                # TODO, is there a much better way to do this???
                target = self.maskformer_panoptic_format_converter(target_info, target)
                count += 1
                if count == 100:
                    raise ValueError(f"problematic sample for instance dataloader: {image_id}")

            if 'boxes' in self.target_type[0] and target is not None:
                h, w = image.shape[1], image.shape[2]
                target["orig_target_sizes"] = torch.tensor([h, w])
                boxes = masks_to_boxes(target["masks"])
                target["boxes"] = boxes
                target["image_id"] = image_id
                target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                target["iscrowd"] = np.zeros([np.shape(boxes)[0]], dtype=np.uint8)

                if "normalized" in self.target_type[0]:

                    target["original_boxes"] = target["boxes"]
                    boxes = box_xyxy_to_cxcywh(target["boxes"])
                    after_h, after_w = image.shape[-2:]
                    boxes = boxes / torch.tensor([after_w, after_h, after_w, after_h], dtype=torch.float32)
                    target["boxes"] = boxes

        if "panoptic" not in self.target_type[0]:
            target = self._convert_mask_labels(target)

        if self.return_dataset_info is True:
            return image, target, info_dict
        else:
            return image, target

    def _valid_splits(self):
        valid_splits = ["train", "test", "val"]
        return valid_splits

    def _valid_target_types(self):
        valid_target_types = ["instance", "panoptic", "semantic", "panoptic-deeplab", "panoptic_with_boxes",
                              "instance_with_boxes_from_panoptic_normalized", "instance_with_boxes_from_panoptic"]
        return valid_target_types

    def _get_target_suffix(self, target_type: str) -> str:
        if target_type == 'instance':
            return self.instance_path_format
        elif target_type == 'semantic':
            return self.label_path_format
        elif 'panoptic' in target_type:
            return self.panoptic_path_format

    def _determine_classes(self):
        config_fn = os.path.join(self.root, self.config_file)
        with open(config_fn) as config_file:
            config = json.load(config_file)

        if "instance" in self.target_type[0]:
            stuff_disabled = True
        else:
            stuff_disabled = False

        classes = []
        train_index = 0
        for index, label in enumerate(config['labels']):
            name = label['readable']
            color = label['color']
            has_instance = label['instances']

            ignore_in_eval = not label['evaluate']
            if stuff_disabled:
                ignore_in_eval = stuff_disabled and not has_instance

            category_list = label['name'].split('--')
            sub_category1 = category_list[0]
            if len(category_list) == 3:
                sub_category2 = category_list[1]
            else:
                sub_category2 = None

            if ignore_in_eval:
                train_index_to_save = 255
            else:
                train_index_to_save = train_index
                train_index += 1

            classes.append(self.MapillaryClass(name, index, train_index_to_save, sub_category1, sub_category2,
                                               has_instance, ignore_in_eval, color))
        return classes

    def load_target(self, target_type, image_id):
        suffix = self._get_target_suffix(target_type=target_type)
        target_path = os.path.join(self.root, suffix)
        target = Image.open(target_path.format(image_id))
        if target_type == "semantic":
            label_array = np.array(target)
            return label_array
        elif target_type == "instance":
            instance_array = np.array(target, dtype=np.uint16)
            # now we split the instance_array into labels and instance ids
            instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
            instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)
            return instance_label_array
        elif "panoptic" in target_type:
            panoptic_segment_info = self.get_panoptic_segment_info(image_id)
            panoptic = np.array(target)

            return panoptic, panoptic_segment_info

    def get_panoptic_segment_info(self, image_id):
        segments = self.panoptic_annotations[image_id]
        return segments

    def __len__(self) -> int:
        return len(self.image_id_list)


class Mapillary12(Mapillary):
    version = "v1.2"
    panoptic_json_file_name = "panoptic_2018.json"

    def __init__(self, **kwargs):
        # kwargs['root'] = os.path.join(kwargs['root'], "mapillary")
        super(Mapillary12, self).__init__(**kwargs)
