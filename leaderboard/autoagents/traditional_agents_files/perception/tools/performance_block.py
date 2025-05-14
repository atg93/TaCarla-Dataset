import torch

from Initialize import load_trainer_model
import time
import numpy as np
from utils.deployment_utils import tensorrt_inference, to_numpy

start_cpu_to_gpu = torch.cuda.Event(enable_timing=True)
end_cpu_to_gpu = torch.cuda.Event(enable_timing=True)

start_backbone = torch.cuda.Event(enable_timing=True)
end_backbone = torch.cuda.Event(enable_timing=True)

start_classifier = torch.cuda.Event(enable_timing=True)
end_classifier = torch.cuda.Event(enable_timing=True)

start_post_process = torch.cuda.Event(enable_timing=True)
end_post_process = torch.cuda.Event(enable_timing=True)

start_gpu_to_cpu = torch.cuda.Event(enable_timing=True)
end_gpu_to_cpu = torch.cuda.Event(enable_timing=True)

yaml_location = "settings/mask2former_resnet50_mapillary12_1024x1024.yml"
trainer = load_trainer_model(yaml_location, evaluation_mode=True, deployment_mode=False, evaluation_batch_size=1,
                             evaluation_torch_device="cuda:0", evaluation_load_mode="final")

trainer.model.eval()

number_of_feedforward = 100
count = 0

cpu_to_gpu_list = []
backbone_list = []
classifier_list = []
post_process_list = []
gpu_to_cpu_list = []

# trainer.get_model_complexity_info()
for count, (images, labels) in enumerate(trainer.val_loader):
    start_cpu_to_gpu.record()
    images, _ = trainer.to_device(images)
    # torch_start = time.time()
    end_cpu_to_gpu.record()
    torch.cuda.synchronize()
    cpu_to_gpu_time = start_cpu_to_gpu.elapsed_time(end_cpu_to_gpu)
    cpu_to_gpu_list.append(cpu_to_gpu_time)

    with torch.no_grad():
        start_backbone.record()
        features = trainer.model.backbone(images)
        end_backbone.record()
        torch.cuda.synchronize()
        backbone_time = start_backbone.elapsed_time(end_backbone)
        backbone_list.append(backbone_time)

        start_classifier.record()
        head_outputs = trainer.model.classifier(features)
        end_classifier.record()
        torch.cuda.synchronize()
        classifier_time = start_classifier.elapsed_time(end_classifier)
        classifier_list.append(classifier_time)

    start_post_process.record()
    outputs = trainer.postprocess_outputs({"out": head_outputs})
    end_post_process.record()
    torch.cuda.synchronize()
    post_process_time = start_post_process.elapsed_time(end_post_process)
    post_process_list.append(post_process_time)
    count += 1

    start_gpu_to_cpu.record()
    if isinstance(outputs, dict):
        trainer.to_numpy(outputs["panoptic"])
        trainer.to_numpy(outputs["semantic"])
    else:
        trainer.to_numpy(outputs)
    end_gpu_to_cpu.record()
    torch.cuda.synchronize()
    gpu_to_cpu_time = start_gpu_to_cpu.elapsed_time(end_gpu_to_cpu)
    gpu_to_cpu_list.append(gpu_to_cpu_time)

    if count == number_of_feedforward:
        break

print(f"CPU to GPU time {np.mean(cpu_to_gpu_list[1:]): 0.3f} ms")
print(f"Backbone time {np.mean(backbone_list[1:]): 0.3f} ms")
print(f"Classifier time {np.mean(classifier_list[1:]): 0.3f} ms")
print(f"Post Process time {np.mean(post_process_list[1:]): 0.3f} ms")
print(f"GPU to CPU time {np.mean(gpu_to_cpu_list[1:]): 0.3f} ms")
