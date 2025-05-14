from Initialize import load_trainer_model
import os
import torch
import argparse
import subprocess
import numpy as np
from utils import tensorrt_inference, initialize_tensorrt_model
from trainers import Trainer
from typing import Tuple, List, Optional, Union, Dict
from Initialize.initialize import INITIALIZE
from utils import prob2lines_CULane
import torch.nn.functional as F
from utils import MetricLogger, collate_sub_function


class InitializeCuLaneEval(INITIALIZE):
    def __init__(self,
                 yaml_location: str,
                 **kwargs: Union[bool, str, int]) -> None:
        super(InitializeCuLaneEval, self).__init__(yaml_location=yaml_location, **kwargs)
        self.trainer_class = TrainerCuLaneEval
        self.train_dict["dataset_kwargs"]["split"] = "test"
        self.train_dict["verbose_frequency"] = 100
        self.train_dict["dataset_kwargs"]["target_type"] = "semantic_lane_from_txt"
        self.train_dict["dataset_kwargs"]["return_image_name"] = True

class TrainerCuLaneEval(Trainer):
    def __init__(
            self,
            training_dictionary: Dict[str, Union[str, int, float, List[int], List[float]]],
            yaml_name: str,
    ) -> None:
        super(TrainerCuLaneEval, self).__init__(yaml_name=yaml_name,
                                             training_dictionary=training_dictionary)
    def evaluate_culane_performance(self, mode: str):
        """
        :param mode: The available modes are torch and tensorrt
        :return:
        """

        context, bindings, device_input, device_tensorrt_outs, stream, host_tensorrt_outs = None, None, None, None, None, None

        if "tensorrt" in mode:
            if not os.path.isfile(self.onnx_file_12) and not self.quantize:
                dummy_image = torch.randn([1, 3, self.size[0], self.size[1]])
                self.model.eval()
                dummy_image = dummy_image.to(self.torch_device)
                with torch.no_grad():
                    outputs = self.model(dummy_image)

                self.self_save_onnx_model_12(inputs=dummy_image,
                                             outputs=outputs)

            if not os.path.isfile(self.onnx_file_13) and not self.quantize:
                dummy_image = torch.randn([1, 3, self.size[0], self.size[1]])
                self.model.eval()
                dummy_image = dummy_image.to(self.torch_device)
                with torch.no_grad():
                    outputs = self.model(dummy_image)

                self.self_save_onnx_model_13(inputs=dummy_image,
                                             outputs=outputs)

            if mode == "tensorrt_13":
                tensorrt_file_for_evaluation = self.tensorrt_file_13
                if not os.path.isfile(tensorrt_file_for_evaluation):
                    self.onnx13_to_tensorrt()

            elif mode == "tensorrt_fp16_13":
                tensorrt_file_for_evaluation = self.tensorrt_file_fp16_13
                if not os.path.isfile(tensorrt_file_for_evaluation):
                    self.onnx13_to_tensorrt_fp16()

            elif mode == "tensorrt_int8_13":
                tensorrt_file_for_evaluation = self.tensorrt_file_int8_13
                if not os.path.isfile(tensorrt_file_for_evaluation):
                    self.onnx13_to_tensorrt_int8()

            elif mode == "tensorrt_ptq_pc_hist_13":
                tensorrt_file_for_evaluation = self.tensorrt_file_ptq_pc_hist_13
                if not os.path.isfile(tensorrt_file_for_evaluation):
                    self.onnx13_ptq_pc_hist_to_tensorrt()

            elif mode == "tensorrt_qat_pc_hist_13":
                tensorrt_file_for_evaluation = self.tensorrt_file_qat_pc_hist_13
                if not os.path.isfile(tensorrt_file_for_evaluation):
                    self.onnx13_qat_pc_hist_to_tensorrt()
            else:
                raise ValueError("Not a Supported Format")

            dummy_image = torch.randn([1, 3, self.size[0], self.size[1]])
            context, bindings, device_input, device_tensorrt_outs, stream, host_tensorrt_outs = \
                initialize_tensorrt_model(tensorrt_file=tensorrt_file_for_evaluation,
                                          image=dummy_image,
                                          output_names=self.output_names,
                                          number_of_classes=self.number_of_classes)
        elif "torch" in mode:
            self.model.eval()

        out_path = os.path.join(self.evaluate_result_directory, "coord_output")
        evaluation_path = os.path.join(self.evaluate_result_directory, "evaluate")
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        if not os.path.exists(evaluation_path):
            os.mkdir(evaluation_path)

        header = ""
        self.metric_logger = MetricLogger(delimiter="  ")
        # self._initialize_metric_loggers()
        for batch_idx, (image, target) in self.metric_logger.log_every(self.val_loader, self.verbose_frequency, header):
            # for batch_idx, (image, image_name) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            # image_name = data_dict['img_name']
            # image = data_dict['img']
            image_name = target["image_name"]
            with torch.no_grad():
                if "tensorrt" in mode:
                    tensorrt_inference(device_input=device_input,
                                       device_tensorrt_outs=device_tensorrt_outs,
                                       image=image,
                                       stream=stream,
                                       context=context,
                                       bindings=bindings,
                                       host_tensorrt_outs=host_tensorrt_outs)

                    output = host_tensorrt_outs["out"]

                elif mode == "torch":
                    image = image.to(self.device)
                    outputs = self.model(image)
                    output = outputs["out"]

                if isinstance(output, np.ndarray):
                    output = torch.from_numpy(output)

                output_probabilities = F.softmax(output, dim=1)
                output_probabilities = self.to_numpy(output_probabilities)
                exist = []
                for i in range(1, 5):
                    # TODO, Is there a much better way for thıs?
                    if np.sum(output_probabilities[:, i, ...] > 0.5):
                        exist.append(1)
                    else:
                        exist.append(0)

                lane_coords = prob2lines_CULane(output_probabilities[0], exist, resize_shape=(590, 1640),
                                                y_px_gap=20, pts=18)

                path_tree = self.split_path(image_name[0])
                save_dir, save_name = path_tree[-3:-1], path_tree[-1]
                save_dir = os.path.join(out_path, *save_dir)
                save_name = save_name[:-3] + "lines.txt"
                save_name = os.path.join(save_dir, save_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                with open(save_name, "w") as f:
                    for lane_coord in lane_coords:
                        for (x, y) in lane_coord:
                            print("{} {}".format(x, y), end=" ", file=f)
                        print(file=f)

        evaluate_script_file = "utils/lane_evaluation/CULane/Run.sh"
        with open(evaluate_script_file, 'rb') as f:
            lines = f.readlines()

        root_directory = os.getcwd()
        lines[0] = f"result_root={self.outside_save_folder}/\n".encode('ascii')
        lines[1] = f"repo_root={root_directory}/\n".encode('ascii')
        lines[2] = f"data_dir={self.dataset_root_folder}/culane/\n".encode('ascii')

        with open(evaluate_script_file, 'wb') as f:
            f.writelines(lines)

        try:
            subprocess.run(f'sh utils/lane_evaluation/CULane/Run.sh {self.yaml_name}',
                           check=True,
                           shell=True)
        except Exception as e:
            print(e)
            print("Culane evaluation code might not be compiled. Follow the readme file for proper installation...")

        result_txt_file_name = f"{self.yaml_name}_iou0.5_split.txt"
        result_txt_file = os.path.join(evaluation_path, result_txt_file_name)
        with open(result_txt_file) as f:
            lines = f.readlines()

        for line in lines:
            print(line)

    def collate_fn(self, batch):
        images, targets = list(zip(*batch))
        return images[0][None, ...], targets[0]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CuLane Evaluations')
    # parser.add_argument('yaml_name', metavar='FILE', help='path to the yaml')

    yaml_name = "settings/Deeplab/deeplabv3_resnet18_culane_640x368.yml"

    trainer = InitializeCuLaneEval(yaml_name, evaluation_mode=True, deployment_mode=True, evaluation_batch_size=1,
                                 evaluation_torch_device="cuda:0", evaluation_load_mode="final").create()

    trainer.evaluate_culane_performance(mode="torch")