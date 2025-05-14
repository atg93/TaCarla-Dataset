from abc import ABC
import cv2
import torch
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, NamedTuple
import numpy as np
from collections import namedtuple
from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from tairvision.transforms.common_transforms import ToTensor
from PIL import Image


class GenericVisionDataset(VisionDataset, ABC):
    def __init__(self,
                 target_type: Union[List[str], str],
                 split: str = "train",
                 **kwargs
                 ):
        super(GenericVisionDataset, self).__init__(**kwargs)
        self.target_type = target_type
        self.split = split

        valid_splits = self._valid_splits()
        self.valid_splits = valid_splits
        msg_split = "Unknown value '{}' for argument split. Valid values are {{{}}}."
        msg_split = msg_split.format(split, iterable_to_str(valid_splits))
        verify_str_arg(split, "split", valid_splits, msg_split)

        if not isinstance(target_type, list):
            self.target_type = [target_type]

        valid_target_types = self._valid_target_types()
        self.valid_target_types = valid_target_types
        msg_target_type = "Unknown value '{}' for argument target_split. Valid values are {{{}}}."
        msg_target_type = msg_target_type.format(target_type, iterable_to_str(valid_target_types))
        [verify_str_arg(value, "target_type", valid_target_types, msg_target_type) for value in self.target_type]
        #verify_str_arg(self.target_type, "target_type", valid_target_types, msg_target_type)

    def get_number_of_classes(self) -> int:
        return 0

    def get_color_palette(self) -> np.ndarray:
        pass

    def _valid_target_types(self):
        return []

    def _valid_splits(self):
        valid_splits = ["train", "val", "test"]
        return valid_splits

    def _determine_classes(self):
        return self.classes


class GenericSegmentationVisionDataset(GenericVisionDataset, ABC):
    def __init__(self, mask_convert: Optional[Any] = False,
                 return_dataset_info=False,
                 **kwargs):
        super(GenericSegmentationVisionDataset, self).__init__(**kwargs)
        self.mask_convert = mask_convert
        self.return_dataset_info = return_dataset_info
        self.classes = self._determine_classes()
        self.class_mapping = self._create_mapping_for_mask_label_conversion()
        self.reversed_class_mapping = self._create_reverse_class_mappping()
        self.thing_list = self.get_thing_id_list()
        self.panoptic_ignore_label = 0
        self.ToTensor = ToTensor()
        self.label_divisor = 1000

    def get_color_palette(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if isinstance(self.get_number_of_classes(), List):
            palette_color_array_dict = {}
            for target_type, class_element in self.classes.items():
                palette_color_array = np.zeros([256, 3])
                for cls in class_element:
                    if cls.ignore_in_eval is False:
                        palette_color_array[cls.train_id] = cls.color
                palette_color_array = palette_color_array.astype(np.uint8)
                palette_color_array_dict.update({target_type: palette_color_array})
            return palette_color_array_dict
        else:
            palette_color_array = np.zeros([256, 3])
            for cls in self.classes:
                if cls.ignore_in_eval is False:
                    palette_color_array[cls.train_id] = cls.color
            palette_color_array = palette_color_array.astype(np.uint8)
            return palette_color_array

    def get_number_of_classes(self) -> Union[int, List[int]]:

        if isinstance(self.classes, dict):
            number_of_classes_list = []
            for _, class_instance in self.classes.items():
                number_of_class = 0
                for cls in class_instance:
                    if cls.ignore_in_eval is False:
                        number_of_class += 1
                number_of_classes_list.append(number_of_class)
            number_of_class = number_of_classes_list
        else:
            number_of_class = 0
            for cls in self.classes:
                if cls.ignore_in_eval is False:
                    number_of_class += 1
        return number_of_class

    def _create_reverse_class_mappping(self):
        reversed_class_mapping = {}
        if self.classes is None:
            return None
        if isinstance(self.get_number_of_classes(), List):
            reversed_class_mapping_dict = {}
            for target_type, class_element in self.classes.items():
                reversed_class_mapping = {}
                for cls in class_element:
                    reversed_class_mapping.update({cls.train_id: cls.id})
                reversed_class_mapping_dict.update({target_type: reversed_class_mapping})
            return reversed_class_mapping_dict
        else:
            for cls in self.classes:
                reversed_class_mapping.update({cls.train_id: cls.id})
            return reversed_class_mapping

    def get_thing_id_list(self):
        thing_id_list = []
        if self.classes is None:
            return
        if isinstance(self.get_number_of_classes(), List):
            pass
            # TODO, correct this
        else:
            for cls in self.classes:
                if cls.has_instances is True:
                    if cls.train_id == 255:
                        continue
                    thing_id_list.append(cls.train_id)
        return thing_id_list

    def maskformer_panoptic_format_converter(self, target_info, target, return_extra_info=False):
        classes = []
        masks = []
        area = []
        iscrowd = []
        if isinstance(target, list):
            pan_seg_gt = target[0]
        else:
            pan_seg_gt = target
        for segment_info in target_info:
            class_id = segment_info["category_id"]
            # Coco detection side eliminates iscrowd labels, is it correct also here?
            # My initial mask2former trainings also implemented with iscrowd elimination
            if not segment_info["iscrowd"]:
                if self.class_mapping[class_id] == 255:
                    continue
                created_mask = pan_seg_gt == segment_info["id"]
                # TODO, how to handle this part, does this line corrupts mapillary or cityscapes trainings
                if torch.sum(created_mask != 0) == 0:
                    continue
                classes.append(torch.tensor([self.class_mapping[class_id]], dtype=torch.int64))
                masks.append(created_mask)
                # area.append(segment_info["area"])
                # iscrowd.append(segment_info["iscrowd"])
        if len(masks) == 0:
            return None
        else:
            return {"masks": torch.stack(masks, 0), "labels": torch.cat(classes, 0)}

    def panoptic_ground_truth_conversion_to_constant_standard(self, panoptic, segment_infos):
        stuff_memory_list = {}
        class_id_tracker = {}
        for segment_info in segment_infos:
            class_id = segment_info["category_id"]
            pred_class = self.class_mapping[class_id]
            isthing = pred_class in self.thing_list
            id = segment_info["id"]

            if not isthing:
                if int(pred_class) in stuff_memory_list.keys():
                    current_segment_id = stuff_memory_list[int(pred_class)]
                else:
                    current_segment_id = self.label_divisor * pred_class
                    if current_segment_id == 0:
                        # In order to handle the confusion with void
                        current_segment_id = 1
                    stuff_memory_list[int(pred_class)] = current_segment_id
            else:
                if pred_class in class_id_tracker:
                    new_ins_id = class_id_tracker[pred_class]
                else:
                    class_id_tracker[pred_class] = 1
                    new_ins_id = 1
                class_id_tracker[pred_class] += 1
                current_segment_id = self.label_divisor * pred_class + new_ins_id

            panoptic[panoptic == id] = current_segment_id
            segment_info["id"] = current_segment_id
            segment_info["category_id"] = pred_class
        return panoptic, segment_infos

    def _create_mapping_for_mask_label_conversion(self):
        class_mapping = {}
        if self.classes is None:
            return class_mapping
        if isinstance(self.get_number_of_classes(), List):
            class_mapping_dict = {}
            for target_type, class_element in self.classes.items():
                class_mapping = {}
                for cls in class_element:
                    class_mapping.update({cls.id: cls.train_id})
                class_mapping_dict.update({target_type: class_mapping})
            return class_mapping_dict
        else:
            for cls in self.classes:
                class_mapping.update({cls.id: cls.train_id})
            return class_mapping

    @staticmethod
    def modify_training_classes(classes):
        train_id_to_assign = 0
        for cls_index, cls in enumerate(classes):
            if cls.ignore_in_eval:
                cls = cls._replace(train_id=255)
            else:
                cls = cls._replace(train_id=train_id_to_assign)
                train_id_to_assign += 1
            classes[cls_index] = cls
        return classes

    def _convert_mask_labels(self, target, target_type: Optional[str] = None):
        if isinstance(target, np.ndarray):
            copied_target = target.copy()
        elif isinstance(target, torch.Tensor):
            copied_target = target.clone()
        else:
            raise ValueError("Unsupported type for the convert mask labels")

        if isinstance(self.get_number_of_classes(), List):
            assert target_type is not None, "target type should exist"
            class_mapping = self.class_mapping[target_type]
            for key, value in class_mapping.items():
                copied_target[target == key] = value
        else:
            for key, value in self.class_mapping.items():
                copied_target[target == key] = value

        return copied_target

    def get_class_names(self):
        class_name_list = []
        if isinstance(self.get_number_of_classes(), List):
            class_name_dict = {}
            for target_type, class_element in self.classes.items():
                class_name_list = []
                for cls in class_element:
                    if cls.ignore_in_eval is False:
                        class_name_list.append(cls.name)
                class_name_dict.update({target_type: class_name_list})
            return class_name_dict
        else:
            for cls in self.classes:
                if cls.ignore_in_eval is False:
                    class_name_list.append(cls.name)
            return class_name_list

    @staticmethod
    def _apply_morphology_operation(target: np.ndarray, kernel, label: int,
                                    operation: str, null_label: int):
        assert operation in ["dilation", "closing", "opening", "erosion"],\
            "Only dilation or closing operation are supported"
        if label == 0:
            return
        if label == 255:
            return
        if np.sum(target == label) == 0:
            return

        target_temp = np.zeros_like(target)
        target_temp[target == label] = label
        target[target == label] = null_label
        if operation == "closing":
            target_temp = cv2.morphologyEx(target_temp, cv2.MORPH_CLOSE, kernel)
        elif operation == "opening":
            target_temp = cv2.morphologyEx(target_temp, cv2.MORPH_OPEN, kernel)
        elif operation == "dilation":
            target_temp = cv2.dilate(target_temp, kernel, iterations=1)
        elif operation == "erosion":
            target_temp = cv2.erode(target_temp, kernel, iterations=1)
        target[target_temp == label] = label

    @staticmethod
    def rgb2id(color):
        """Converts the color to panoptic label.
        Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.
        Args:
            color: Ndarray or a tuple, color encoded image.
        Returns:
            Panoptic label.
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


class GenericObjectVisionDataset(GenericVisionDataset, ABC):
    def __init__(self, mask_convert,
                 **kwargs):
        super(GenericObjectVisionDataset, self).__init__(**kwargs)
