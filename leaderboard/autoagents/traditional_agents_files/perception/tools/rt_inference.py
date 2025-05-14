import os
import torch
import argparse
import numpy as np

try:
    from Initialize import load_trainer_model
except:
    print("initialize is not exported")

from utils.common_keys import *
from utils import onnx_to_tensorrt
from utils.deployment_utils import to_numpy


def convert_torch_to_tensorrt(yaml_location, opset_version, precision):
    # Set deployment mode to True for segmentation models, for SNPE compatibility
    trainer = load_trainer_model(yaml_location, evaluation_mode=True, deployment_mode=False,
                                 evaluation_torch_device="cuda:0")

    image, label = next(iter(trainer.val_loader))
    trainer.model.eval()

    image, _ = trainer.to_device(image)

    with torch.no_grad():
        outputs = trainer.model(image)

    onnx_file = trainer.file_dict[f"{ONNX_FILE}_{opset_version}"]
    tensorrt_file = trainer.file_dict[f"{TENSORRT_FILE}_{precision}_{opset_version}"]

    if not os.path.isfile(onnx_file):
        trainer.self_save_onnx_model(inputs=image,
                                     outputs=outputs,
                                     opset_version=opset_version)

    if not os.path.isfile(tensorrt_file):
        trainer.self_onnx_to_tensorrt(opset_version=opset_version,
                                      precision=precision)

    torch_post_outs = trainer.postprocess_outputs(outputs)

    device_tensorrt_outs  = trainer.self_check_tensorrt_model(image=image,
                                                              tensorrt_file=tensorrt_file,
                                                              outputs=outputs)

    device_tensorrt_post_outs  = trainer.postprocess_outputs(device_tensorrt_outs)


    trainer.visualize(image=image,
                      ground_truth=label,
                      outputs=torch_post_outs,
                      optional_outputs=device_tensorrt_post_outs,
                      show=True,
                      wandb_log_name=tensorrt_file)

    if trainer.wandb_object:
        trainer.wandb_object.finish()

    try:
        print("== Checking post process output(torch vs device) ==")
        [np.testing.assert_allclose(to_numpy(torch_post_outs),
                                    device_tensorrt_outs, rtol=1e-02, atol=1e-02)
         for x in device_tensorrt_outs]
        print("== Done ==")
    except:
        "POST PROCESSING NOT TESTED !"


def convert_only_torch_to_tensorrt(yaml_name, opset_version, precision):
    weights_main_path = os.path.join(f"/workspace/{os.environ['USER']}/tairvision/weights", yaml_name)
    onnx_file = os.path.join(weights_main_path, f"{MODEL}_{opset_version}.{ONNX_EXTENSION}")
    tensorrt_file = os.path.join(weights_main_path, f"{MODEL}_{precision}_{opset_version}.{TENSORRT_EXTENSION}")
    if not os.path.isfile(tensorrt_file):
        onnx_to_tensorrt(onnx_file=onnx_file,
                         tensorrt_file=tensorrt_file,
                         precision=precision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Torch to TensorRT correction')
    parser.add_argument('yaml_name', metavar='FILE', help='path to the yaml')
    parser.add_argument('--opset_version', type=int,
                        help='opset version of the onnx file', default=13, required=False)
    parser.add_argument('--precision', help='opset version of the onnx file', default=FP32, required=False)
    parser.add_argument('--only_tensorrt', dest='only_tensorrt',
                        help='In order not to implement whole pipeline, onnx conversion and visualization excluded',
                        action='store_true', default=False)
    args = parser.parse_args()
    if args.only_tensorrt:
        yaml_name = args.yaml_name.split('/')[-1][:-4]
        convert_only_torch_to_tensorrt(yaml_name, args.opset_version, args.precision)
    else:
        convert_torch_to_tensorrt(args.yaml_name, args.opset_version, args.precision)
