import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from Initialize import load_trainer_model
from utils.panoptic_evaluation.utils import id2rgb, save_json
from utils.panoptic_evaluation import pq_compute
from utils.panoptic_evaluation import pq_compute_single_image, PQStat
from math import floor


def create_segment_info(outputs, info):
    panoptic = trainer.to_numpy(outputs['panoptic'][0])
    unique_ids = np.unique(panoptic)
    if "segments_info" in outputs.keys():
        segment_info_list = outputs["segments_info"]
        for segment in segment_info_list:
            segment["category_id"] = segment["category_id"]
            segment["area"] = np.sum(panoptic == segment['id'])
    else:
        segment_info_list = []
        for id in unique_ids:
            if id == trainer.ignore_label:
                continue
            determined_semantic_class_id = floor(id / 1000)
            area = np.sum(panoptic == id)
            segment_info_list.append({'id': int(id), 'category_id': int(determined_semantic_class_id), "area": area})

    segment_info = {
                'image_id': info[0]["image_id"],
                'file_name': info[0]["file_name"],
                'segments_info': segment_info_list,
        }

    return panoptic, segment_info


yaml_name = "settings/mask2former_resnet50_cityscapes_1024x512.yml"

trainer = load_trainer_model(yaml_name, evaluation_mode=True, deployment_mode=False, evaluation_batch_size=1,
                             evaluation_torch_device="cuda", evaluation_load_mode="final",
                             return_dataset_info=True)

trainer.model.eval()
count = 0
predictions = []
image_size_list = []
pq_stat = PQStat()
categories = {el['id']: el for el in trainer.categories}
for count, (images, labels, info) in tqdm(enumerate(trainer.val_loader), total=len(trainer.val_loader)):
    images, targets = trainer.to_device(images, labels)

    head_outputs = trainer.model(images)

    outputs = trainer.postprocess_outputs(head_outputs)
    ground_truths = trainer.post_process_ground_truths(labels)

    pan_pred, pred_ann = create_segment_info(outputs, info)
    pan_gt, gt_ann = create_segment_info(ground_truths, info)

    pq_stat_single = pq_compute_single_image(pan_gt, pan_pred, gt_ann, pred_ann, categories, trainer.ignore_label)
    pq_stat += pq_stat_single

metrics = [("All", None), ("Things", True), ("Stuff", False)]
results = {}
for name, isthing in metrics:
    results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
    if name == 'All':
        results['per_class'] = per_class_results
print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
print("-" * (10 + 7 * 4))

for name, _isthing in metrics:
    print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
        name,
        100 * results[name]['pq'],
        100 * results[name]['sq'],
        100 * results[name]['rq'],
        results[name]['n'])
    )
