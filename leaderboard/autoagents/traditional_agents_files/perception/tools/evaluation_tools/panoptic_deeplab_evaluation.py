import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import torch

from Initialize import load_trainer_model
from utils.panoptic_evaluation.utils import id2rgb, save_json
from utils.panoptic_evaluation import pq_compute
from math import floor
from torch.nn import functional as F
from tairvision.utils import retry_if_cuda_oom

# TODO, merge this evaluation script and mask2former script into single file, too much common code
def bilinear_interpolation(head_outputs_in_func, size_in_func):
    head_outputs_in_func["out"]["semantic"] = F.interpolate(
        head_outputs_in_func["out"]["semantic"],
        size=size_in_func,
        mode='bilinear',
        align_corners=False).detach()

    head_outputs_in_func["out"]["center"] = F.interpolate(
        head_outputs_in_func["out"]["center"],
        size=size_in_func,
        mode='bilinear',
        align_corners=False).detach()

    output_width = head_outputs_in_func["out"]["offset"].shape[3]
    desired_width = size_in_func[1]
    scale = (desired_width - 1) // (output_width - 1)
    head_outputs_in_func["out"]["offset"] = F.interpolate(
        head_outputs_in_func["out"]["offset"],
        size=size_in_func,
        mode='bilinear',
        align_corners=False).detach() * scale


yaml_name = "settings/panoptic_deeplabv3_resnet50_mapillary12_1856x1024.yml"

trainer = load_trainer_model(yaml_name, evaluation_mode=True, deployment_mode=False, evaluation_batch_size=1,
                             evaluation_torch_device="cuda:0", evaluation_load_mode="final",
                             return_dataset_info=True)

trainer.model.eval()
count = 0
predictions = []
image_size_list = []
for count, (images, labels, info) in tqdm(enumerate(trainer.val_loader), total=len(trainer.val_loader)):
    images, targets = trainer.to_device(images, labels)

    head_outputs = trainer.model(images)

    retry_if_cuda_oom(bilinear_interpolation)(
        head_outputs,
        size_in_func=(info[0]['image_size'][0], info[0]['image_size'][1])
    )

    head_outputs["out"]["semantic"] = head_outputs["out"]["semantic"].to("cuda:1")
    head_outputs["out"]["center"] = head_outputs["out"]["center"].to("cuda:1")
    head_outputs["out"]["offset"] = head_outputs["out"]["offset"].to("cuda:1")

    outputs = retry_if_cuda_oom(trainer.postprocess_outputs)(head_outputs)

    panoptic_pred = trainer.to_numpy(outputs['panoptic'][0])

    unique_ids = np.unique(panoptic_pred)
    if "segments_info" in outputs.keys():
        segment_info_list = outputs["segments_info"]
        for segment in segment_info_list:
            segment["category_id"] = segment["category_id"]
    else:
        segment_info_list = []
        for id in unique_ids:
            if id == trainer.ignore_label:
                continue
            determined_semantic_class_id = floor(id / 1000)
            segment_info_list.append({'id': int(id), 'category_id': int(determined_semantic_class_id)})

    predictions.append(
        {
                'image_id': info[0]["image_id"],
                'file_name': info[0]["file_name"],
                'segments_info': segment_info_list,
        }
    )

    panoptic_rgb = id2rgb(panoptic_pred)
    panoptic_pil = Image.fromarray(panoptic_rgb.astype(dtype=np.uint8))
    with open('%s/%s.png' % (trainer.evaluate_result_directory, info[0]["file_name"].split('.')[0]), mode='wb') as f:
        panoptic_pil.save(f, 'PNG')
    count += 1
    # if count == 5:
    #     break

print("inference step is completed, now beginning evaluation...")
save_json({"annotations": predictions}, f"{trainer.evaluate_result_directory}/pred_file.json")
pq_compute(gt_json_file=trainer.eval_set.panoptic_json_file,
           pred_json_file=f"{trainer.evaluate_result_directory}/pred_file.json",
           gt_folder=trainer.eval_set.panoptic_folder_path,
           pred_folder=trainer.evaluate_result_directory
           )


# print(np.mean(image_size_list, 0))
# [2.4885660e+03 3.4352715e+03 3.0000000e+00]
# 1483 short for 2048 long
# If 1536, then 2120 which is still less than 2177 which is the Nvidia paper
