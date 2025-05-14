import cv2
import numpy as np
import torch
from tairvision.models.segmentation import generic_model
from typing import Tuple, List, Optional, Union
import os
from .inference_sub_utils import post_process_outputs_single, visualize_single, to_numpy
from .inference_dataset_info import get_color_palette_and_number_of_classes, get_thing_list_and_class_names
from tairvision.utils import panoptic_inference_mask2former, panoptic_post_process_single_mask, \
    put_thing_names_on_image, panoptic_inference_deeplab

try:
    from apex import amp
except:
    print("apex is not imported")

from tairvision_master.utils.deployment_utils import initialize_tensorrt_model, tensorrt_inference
from tairvision_master.utils.deployment_utils import read_raw_file, snpe_net_run
from .inference_main import Inference


class InferenceSegmentation(Inference):
    def __init__(self, yaml_name: str, **kwargs):
        super(InferenceSegmentation, self).__init__(yaml_name, **kwargs)

        ## self.output_names = ["out", "aux"]
        self.auxiliary_loss: bool = self.training_dictionary.get('auxiliary_loss')
        self.segmentation_mask_visualization_weights = 0.8
        self.segmentation_merged_mask_visualization_weights = 0.5
        self.color_palette, self.number_of_classes = self.get_number_of_classes_and_color_palette()
        self.output_names = self._create_output_names()
        self.model_path = self.path_for_model_load()
        if "second_yaml" in kwargs and kwargs["second_yaml"]:
            self.second_auxiliary_loss: bool = self.second_training_dictionary.get('auxiliary_loss')
            self.second_output_names = self._create_output_names()

        self.load_model()
        self.erosion_kernel = np.ones((7, 7), np.uint8)

    def _create_output_names(self):
        output_names = []
        output_names.append("out")
        if self.auxiliary_loss:
            output_names.append("aux")

        if isinstance(self.dataset_target_type, List):
            for out_index, _ in enumerate(self.dataset_target_type):
                if out_index == 0:
                    continue
                else:
                    output_key = f"out_{out_index + 1}"
                    auxiliary_key = f"aux_{out_index + 1}"
                output_names.append(output_key)
                if self.auxiliary_loss:
                    output_names.append(auxiliary_key)

        return output_names

    def get_number_of_classes_and_color_palette(self):
        return get_color_palette_and_number_of_classes(self.dataset_name, self.dataset_target_type)

    def post_process_outputs(self, outputs):
        output_mask_list = []
        if not isinstance(self.dataset_target_type, List):
            dataset_target_type_in_funct = [self.dataset_target_type]
        else:
            dataset_target_type_in_funct = self.dataset_target_type

        for output_index, _ in enumerate(dataset_target_type_in_funct):
            if output_index == 0:
                output_key = "out"
            else:
                output_key = f"out_{output_index + 1}"
            output_prob = outputs[output_key]
            output_mask = post_process_outputs_single(output_prob)
            output_mask_list.append(output_mask)
        if self.is_second_yaml:
            second_output_mask_list = []
            if not isinstance(self.second_dataset_target_type, List):
                second_dataset_target_type_in_funct = [self.second_dataset_target_type]
            else:
                second_dataset_target_type_in_funct = self.second_dataset_target_type

            for second_output_index, _ in enumerate(second_dataset_target_type_in_funct):
                if second_output_index == 0:
                    second_output_key = "out"
                else:
                    second_output_key = f"out_{second_output_index + 1}"
                second_output_prob = self.second_outputs[second_output_key]
                second_output_mask = post_process_outputs_single(second_output_prob)
                second_output_mask_list.append(second_output_mask)
            self.second_outputs = second_output_mask_list

        return output_mask_list

    def visualize(self, outputs: List[Union[np.ndarray, torch.Tensor]], frame_resized):
        output_frame_list = []
        if not isinstance(self.color_palette, List):
            color_palette_in_func = [self.color_palette]
        else:
            color_palette_in_func = self.color_palette

        frame_merged = frame_resized.copy()
        for out, color_palette in zip(outputs[::-1], color_palette_in_func[::-1]):
            output_frame = visualize_single(out,
                                            frame_resized.copy(),
                                            color_palette,
                                            self.segmentation_mask_visualization_weights)

            visualize_single(out,
                             frame_merged,
                             color_palette,
                             self.segmentation_merged_mask_visualization_weights)

            output_frame_list.append(output_frame)
        if self.is_second_yaml:
            # TODO, if second yaml requires different color palette, handle here
            second_outputs = self.second_outputs
            second_output_frame_list = []
            second_frame_merged = frame_resized.copy()
            for second_out, color_palette in zip(second_outputs[::-1], color_palette_in_func[::-1]):
                second_output_frame = visualize_single(second_out,
                                                frame_resized.copy(),
                                                color_palette,
                                                self.segmentation_mask_visualization_weights)

                visualize_single(second_out,
                                 second_frame_merged,
                                 color_palette,
                                 self.segmentation_merged_mask_visualization_weights)

                second_output_frame_list.append(second_output_frame)
            self.second_output_frame_list = second_output_frame_list
            self.second_frame_merged      = second_frame_merged
        return output_frame_list, frame_merged


class InferencePanoptic(InferenceSegmentation):
    def __init__(self, yaml_name: str, **kwargs):
        super(InferencePanoptic, self).__init__(yaml_name, **kwargs)

        self.class_names, self.thing_list = get_thing_list_and_class_names(self.dataset_name)
        self.ignore_label = 0
        self.label_divisor = 1000

        self.segmentation_mask_visualization_weights = 0.95

        self.post_processing_arguments = self.training_dictionary.get('post_processing_arguments')

    def visualize(self, outputs: List[Union[np.ndarray, torch.Tensor]], frame_resized):
        output_frame_list = []
        semantic = to_numpy(outputs["semantic"])
        panoptic = to_numpy(outputs["panoptic"])
        center_points = to_numpy(outputs["center_points"])
        semantic[panoptic == self.ignore_label] = self.number_of_classes

        center_points = center_points.astype(np.int32)
        point_list = center_points[0].tolist()

        output_frame = visualize_single(
            semantic,
            frame_resized.copy(),
            self.color_palette,
            self.segmentation_mask_visualization_weights
        )

        put_thing_names_on_image(point_list, semantic, output_frame, self.class_names, self.thing_list)

        output_frame_list.append(output_frame)
        frame_merged = output_frame

        return output_frame_list, frame_merged


class InferenceSegmentationTorch(InferenceSegmentation):
    def __init__(self, yaml_name: str, **kwargs):
        super(InferenceSegmentationTorch, self).__init__(yaml_name, **kwargs)

    def load_model(self):
        model = generic_model(
            pretrained=False,
            num_classes=self.number_of_classes,
            aux_loss=self.auxiliary_loss,
            size=self.size,
            **self.model_config
        )

        weights = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(weights['model_state_dict'])
        model.to(self.device)
        model.eval()

        if self.mixed_precision_training:
            amp.initialize(model, opt_level="O3", keep_batchnorm_fp32=True)

        self.model = model

        if self.is_second_yaml:
            model2 = generic_model(
                pretrained=False,
                num_classes=self.number_of_classes,
                aux_loss=self.second_auxiliary_loss,
                size=self.second_size,
                **self.second_model_config
            )

            weights = torch.load(self.second_model_path, map_location="cpu")
            model2.load_state_dict(weights['model_state_dict'])
            model2.to(self.second_device)
            model2.eval()

            if self.second_mixed_precision_training:
                amp.initialize(model2, opt_level="O3", keep_batchnorm_fp32=True)

            self.model2 = model2

    def path_for_model_load(self):
        torch_file = os.path.join(self.weights_main_path, self.yaml_name, "final.pth")
        if self.is_second_yaml:
            self.second_model_path  = os.path.join(self.second_weights_main_path, self.second_yaml_name, "final.pth")
        return torch_file

    def model_feedforward(self, image):
        image_tensor = torch.tensor(image)
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            if self.is_second_yaml:
                self.second_outputs = self.model2(image_tensor)
        return outputs


class InferenceMask2formerTorch(InferencePanoptic, InferenceSegmentationTorch):
    def __init__(self, yaml_name: str, **kwargs):
        super(InferenceMask2formerTorch, self).__init__(yaml_name, **kwargs)

    def post_process_outputs(self, outputs):
        if "out" in outputs:
            outputs = outputs["out"]

        outputs = panoptic_inference_mask2former(
            panoptic_mask_function=panoptic_post_process_single_mask,
            outputs=outputs, number_of_classes=self.number_of_classes,
            ignore_label=self.ignore_label, thing_list=self.thing_list, label_divisor=self.label_divisor,
            **self.post_processing_arguments
        )

        return outputs


class InferenceDeeplabTorch(InferencePanoptic, InferenceSegmentationTorch):
    def __init__(self, yaml_name: str, **kwargs):
        super(InferenceDeeplabTorch, self).__init__(yaml_name, **kwargs)

    def post_process_outputs(self, outputs):
        outputs = outputs["out"]

        outputs = panoptic_inference_deeplab(
            panoptic_mask_function=panoptic_post_process_single_mask,
            outputs=outputs, thing_list=self.thing_list, ignore_label=self.ignore_label,
            label_divisor=self.label_divisor,
            **self.post_processing_arguments
        )

        return outputs


class InferenceSegmentationTensorRT(InferenceSegmentation):
    def __init__(self, yaml_name: str, tensorrt_file: Optional[str] = "model_op12.trt", **kwargs):
        self.tensorrt_file = tensorrt_file
        self.context = None
        self.bindings = None
        self.device_input = None
        self.device_tensorrt_outs = None
        self.stream = None
        self.host_tensorrt_outs = None
        super(InferenceSegmentationTensorRT, self).__init__(yaml_name, **kwargs)

    def load_model(self):
        input_dimension = [1, 3] + self.size
        empty_input = np.empty(input_dimension, dtype=np.float32)
        self.context, self.bindings, self.device_input, self.device_tensorrt_outs, self.stream = \
            initialize_tensorrt_model(tensorrt_file=self.model_path,
                                      image=empty_input,
                                      output_names=self.output_names,
                                      number_of_classes=self.number_of_classes)

    def path_for_model_load(self):
        tensorrt_file = os.path.join(self.weights_main_path, self.yaml_name, self.tensorrt_file)
        return tensorrt_file

    def model_feedforward(self, image):
        tensorrt_inference(device_input=self.device_input,
                           device_tensorrt_outs=self.device_tensorrt_outs,
                           image=image,
                           stream=self.stream,
                           context=self.context,
                           bindings=self.bindings)

        return self.device_tensorrt_outs


class InferenceSegmentationSNPE(InferenceSegmentation):
    def __init__(self, yaml_name: str, dlc_file: Optional[str] = "model.dlc", **kwargs):
        self.dlc_file = dlc_file
        folder_location = os.path.join(self.weights_main_path, yaml_name)
        save_input_image_file_location = os.path.join(folder_location, "image_files")
        self.save_input_image_file = os.path.join(save_input_image_file_location, "demo_0.raw")
        self.raw_list_file = os.path.join(folder_location, "raw_list_for_python.txt")
        self.output_file_location = os.path.join(folder_location, "output")
        super(InferenceSegmentationSNPE, self).__init__(yaml_name, **kwargs)

    def path_for_model_load(self):
        dlc_file = os.path.join(self.weights_main_path, self.yaml_name, self.dlc_file)
        return dlc_file

    def model_feedforward(self, image):
        snpe_net_run(input_list_file=self.raw_list_file,
                     container_file=self.model_path,
                     output_dir=self.output_file_location)

        raw_result_out = read_raw_file(model_dir=self.output_file_location,
                                       size=self.size,
                                       number_of_classes=self.number_of_classes,
                                       output_names=self.output_names)
        return raw_result_out

    def preprocess_image(self, sample_image):
        sample_image_resized = cv2.resize(sample_image, self.cv2_dim, interpolation=cv2.INTER_NEAREST)
        sample_image_rgb = cv2.cvtColor(sample_image_resized, cv2.COLOR_BGR2RGB)
        sample_image_rgb = sample_image_rgb / 255
        sample_image_rgb = np.ascontiguousarray(sample_image_rgb, dtype=np.float32)
        sample_image_rgb.tofile(self.save_input_image_file)
        return None, sample_image_resized
