import torch

from Initialize import load_trainer_model
from utils.common_keys import *
import time
import numpy as np
from utils.deployment_utils import tensorrt_inference, to_numpy

torch_start_post = torch.cuda.Event(enable_timing=True)
torch_end_post = torch.cuda.Event(enable_timing=True)

torch_start_main = torch.cuda.Event(enable_timing=True)
torch_end_main = torch.cuda.Event(enable_timing=True)

device_start_post = torch.cuda.Event(enable_timing=True)
device_end_post = torch.cuda.Event(enable_timing=True)

device_start_main = torch.cuda.Event(enable_timing=True)
device_end_main = torch.cuda.Event(enable_timing=True)


yaml_location = "settings/deeplabv3_resnet50_mapillary_640x368_apex.yml"

trainer = load_trainer_model(yaml_location, evaluation_mode=True, deployment_mode=False,
                             evaluation_torch_device="cuda:0",mixed_precision_training=True, opt_level="O3")
trainer.model.eval()

image, _ = next(iter(trainer.val_loader))
context_device, bindings_device, device_input_device, device_tensorrt_outs_device, stream_device = \
    trainer.self_initialize_tensorrt_model(opset_version=13, precision="fp32", sample_image=image)

number_of_feedforward = 100
count = 0

torch_time_list = []
torch_time_list_post = []

device_time_list_post = []
device_time_list = []


trainer.get_model_complexity_info()
for count, (host_images, labels) in enumerate(trainer.val_loader):
    # ----------------------------------------------------------------------------------------
    torch_start_main.record()#---- main start --------
    images, _ = trainer.to_device(host_images)

    with torch.no_grad():
        head_outputs = trainer.model(images)

    torch_start_post.record()#---- post process start --------
    torch_post_outs = trainer.postprocess_outputs(head_outputs)

    if isinstance(torch_post_outs, list):
        for index, out in enumerate(torch_post_outs):
            torch_post_outs[index]= to_numpy( out)
    else:
        torch_post_outs = to_numpy(torch_post_outs)
    torch_end_post.record()#---- post process end --------
    torch.cuda.synchronize()
    torch_measured_post = torch_start_post.elapsed_time(torch_end_post)
    torch_measured_main = torch_start_main.elapsed_time(torch_end_post)#---- main end --------
    #----------------------------------------------------------------------------------------
    device_start_main.record()  # ---- main start --------
    tensorrt_inference(device_input=device_input_device,
                       device_tensorrt_outs=device_tensorrt_outs_device,
                       image=host_images,
                       stream=stream_device,
                       context=context_device,
                       bindings=bindings_device)
    device_start_post.record()#---- post process start --------
    device_tensorrt_post_outs = trainer.postprocess_outputs(device_tensorrt_outs_device)
    device_end_post.record()#---- post process end --------
    torch.cuda.synchronize()
    device_measured_post = device_start_post.elapsed_time(device_end_post)
    device_measured_main = device_start_main.elapsed_time(device_end_post)#---- main end --------


    if count != 0:
        torch_time_list.append(torch_measured_main / 1000)
        device_time_list.append(device_measured_main / 1000)
        torch_time_list_post.append(torch_measured_post / 1000)
        device_time_list_post.append(device_measured_post / 1000)
    count += 1

    if count == number_of_feedforward:
        break

print(f"The mean time for torch is {1000 * np.mean(torch_time_list): 0.3f} ms")
print(f"The mean time is tensorrt(device) is {1000 * np.mean(device_time_list): 0.3f} ms")

print(f"The mean post processing time of torch is {1000 * np.mean(torch_time_list_post): 0.3f} ms")
print(f"The mean post processing time of tensorrt(device) is {1000 * np.mean(device_time_list_post): 0.3f} ms")
