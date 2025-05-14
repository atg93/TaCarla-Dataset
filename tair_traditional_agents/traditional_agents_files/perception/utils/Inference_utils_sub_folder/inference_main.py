import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional, Union
import yaml
import os
import time
from .inference_sub_utils import preprocess_image
from .inference_sub_utils import CARLA

try:
    import skcuda.misc as sk
except:
    print("skcuda is not imported")


class Inference:
    def __init__(self, yaml_file: str, video_file: str = None,
                 save_video: bool = False, frame_interval: int = 5, cropped: bool = False,
                 preserve_ratio: bool = False, **kwargs):
        super(Inference, self).__init__()
        self.yaml_name = yaml_file.split('/')[-1][:-4]
        with open(yaml_file) as f:
            training_dictionary = yaml.load(f, Loader=yaml.Loader)
            self.training_dictionary = training_dictionary
        self.is_second_yaml = False
        if "second_yaml" in kwargs and kwargs["second_yaml"]:
            self.is_second_yaml = True
            self.second_yaml_name  = kwargs["second_yaml"].split('/')[-1][:-4]
            with open(kwargs["second_yaml"]) as f2:
                second_training_dictionary      = yaml.load(f2, Loader=yaml.Loader)
                self.second_training_dictionary = second_training_dictionary
                self.second_size: List[int]     = second_training_dictionary.get('demo_size')
                self.second_cv2_dim: Tuple[int, int] = (self.second_size[1], self.second_size[0])
                self.second_device: str = second_training_dictionary.get('torch_device')
                self.second_model_config = self.second_training_dictionary.get('model_config')
                self.second_mixed_precision_training: bool = second_training_dictionary.get('mixed_precision_training')
                self.second_dataset_name: str = second_training_dictionary.get('dataset_name')
                self.second_dataset_kwargs: dict = training_dictionary.get('dataset_kwargs')
                self.second_load_coco: Optional[bool] = self.second_training_dictionary.get('load_coco', False)

                self.second_mean    = self.second_training_dictionary.get('mean', [0, 0, 0])
                self.second_std     = self.second_training_dictionary.get('std', [1, 1, 1])

                second_workspace_folder    = self.second_training_dictionary.get('workspace_folder', "/workspace")

                self.second_weights_main_path = f"{second_workspace_folder}/{os.environ['USER']}/tairvision/weights"
                self.second_dataset_target_type = second_training_dictionary["dataset_kwargs"]["target_type"]

        self.size: List[int] = training_dictionary.get('demo_size')
        self.cv2_dim: Tuple[int, int] = (self.size[1], self.size[0])
        self.device: str = training_dictionary.get('torch_device')
        self.model_config = self.training_dictionary.get('model_config')
        self.mixed_precision_training: bool = training_dictionary.get('mixed_precision_training')
        self.dataset_name: str = training_dictionary.get('dataset_name')
        self.dataset_kwargs: dict = training_dictionary.get('dataset_kwargs')
        self.load_coco: Optional[bool] = self.training_dictionary.get('load_coco', False)

        self.mean = self.training_dictionary.get('mean', [0, 0, 0])
        self.std = self.training_dictionary.get('std', [1, 1, 1])
        self.preserve_ratio = preserve_ratio

        self.frame_interval = frame_interval
        self.cropped = cropped
        workspace_folder = training_dictionary.get('workspace_folder', "/workspace")

        self.weights_main_path = f"{workspace_folder}/{os.environ['USER']}/tairvision/weights"
        self.dataset_target_type = training_dictionary["dataset_kwargs"]["target_type"]
        self.save_video: bool = save_video
        self.save_video_enabled = False
        self.video_file = video_file

        # self.start = torch.cuda.Event(enable_timing=True)
        # self.end = torch.cuda.Event(enable_timing=True)

        # self.start = drv.Event()
        # self.end = drv.Event()
        self.pre_time_list = []
        self.port = 2000
        # TODO, fix the save video for list of output frames
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.out = cv2.VideoWriter(f"{self.yaml_name}_{video_file.split('/')[-1].split('.')[0]}.avi", fourcc, 10.0,
                                       self.cv2_dim)
        try:
            sk.init()
        except:
            print("skcuda is not imported !")

    def preprocess_image(self, sample_image, preserve_ratio=False):
        sample_image_rgb, sample_image_resized = preprocess_image(sample_image=sample_image, cv2_dim=self.cv2_dim,
                                                                  mean=self.mean, std=self.std,
                                                                  preserve_ratio=preserve_ratio)
        self.size = list(sample_image_rgb.shape[2:])
        return sample_image_rgb, sample_image_resized

    def load_model(self):
        pass

    def model_feedforward(self, image):
        pass

    def model_feedforward_main(self, image):
        outputs = self.model_feedforward(image)
        outputs_viz, outputs_culane  = self.post_process_outputs(outputs)
        return outputs_viz, outputs_culane

    def post_process_outputs(self, outputs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        pass

    def path_for_model_load(self):
        return ""

    def post_process_in_main_loop(self, frame):
        pass

    def take_inference(self, frame, preserve_ratio=False):
        start_time = time.time()
        sample_image_rgb, frame_resized = self.preprocess_image(frame, preserve_ratio=preserve_ratio)
        end_time = time.time()
        inference_time = end_time - start_time
        self.pre_time_list.append(inference_time)
        print(f'Mean pre-processing time: {np.mean(self.pre_time_list) * 1000: 0.4f}')
        outputsviz, outputsculane = self.model_feedforward_main(image=sample_image_rgb)
        self.post_process_in_main_loop(outputsviz)
        return outputsviz, frame_resized, outputsculane

    def main_loop(self):
        if self.video_file == "camera":
            cap = cv2.VideoCapture(0)
        elif self.video_file == "carla" or self.video_file == "carla1":
            cap = CARLA(port=self.port)
        elif self.video_file == "carla2":
            cap = CARLA(scenario=2, port=self.port)
        elif self.video_file == "carla3":
            cap = CARLA(scenario=3, port=self.port)
        else:
            cap = cv2.VideoCapture(self.video_file)
        timelist = []
        timelist_torch = []
        timelist_trt = []

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if ret is False:
                break

            if not frame_num % self.frame_interval == 0:
                frame_num += 1
                continue

            if self.cropped:
                frame = frame[:, 160:-160, :]

            # TODO, time measurement for torch and tensorrt separately, or directly normal python timer??
            start_time = time.time()
            # self.start.record()
            outputs, frame_resized = self.take_inference(frame, preserve_ratio=self.preserve_ratio)
            end_time = time.time()
            # self.end.record()
            # torch.cuda.synchronize()
            # self.end.synchronize()

            # torch_measured = self.start.elapsed_time(self.end)
            # trt_measured = self.start.time_till(self.end)
            output_frames, frame_merged = self.visualize(outputs, frame_resized)
            inference_time = end_time - start_time

            # color_seg = cv2.resize(color_seg, frame.shape[:-1][::-1], interpolation=cv2.INTER_AREA)

            # print(f'Inference time: {inference_time: 0.4f}')
            if frame_num != 0:
                timelist.append(inference_time)
                print(f'Mean inference time: {np.mean(timelist) * 1000: 0.4f} for frame number {frame_num}')

            # if frame_num != 0:
            #     timelist_torch.append(torch_measured)
            #     print(f'Mean inference time: {np.mean(timelist_torch): 0.4f} for frame number {frame_num}')

            # if frame_num != 0:
            #     timelist_trt.append(trt_measured)
            #     print(f'Mean inference time: {np.mean(timelist_trt): 0.4f} for frame number {frame_num}')

            frame_num += 1
            # frame_merged = cv2.resize(frame_merged,
            #                           (frame.shape[1], frame.shape[0]))

            # cv2.imshow("merged", frame_merged)
            cv2.imshow("original", frame)
            if self.save_video and self.save_video_enabled:
                self.out.write(frame_merged)

            for index, frame_sub in enumerate(output_frames):
                # frame_sub = cv2.resize(frame_sub,
                #                        (frame.shape[1], frame.shape[0]))
                cv2.imshow(f"out_{index}", frame_sub)
                if self.is_second_yaml:
                    frame_sub2 = self.second_output_frame_list[index]
                    frame_sub2 = cv2.resize(frame_sub2,
                                           (frame.shape[1], frame.shape[0]))
                    cv2.imshow(f"out_{index}_second", frame_sub2)
            #     # if self.save_video:
            #     #     self.out.write(output_frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            if key == ord('p'):
                while True:
                    key = cv2.waitKey(1)
                    if key == ord('p'):
                        break

            if key == ord('s'):
                if self.save_video_enabled is False:
                    self.save_video_enabled = True
                else:
                    self.out.release()

        print(f'Mean inference time: {np.mean(timelist): 0.4f}')
