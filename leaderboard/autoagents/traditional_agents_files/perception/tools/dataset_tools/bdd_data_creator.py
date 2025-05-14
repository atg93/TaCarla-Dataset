import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import json
from Initialize import load_trainer_model
from tairvision.datasets.bdd100k import BDD100k
from torch.utils.data.dataloader import DataLoader
from utils import save_numpy_mask_as_image

split_type = "val"

yaml_name = "settings/deeplabv3_resnet50_mapillary12_1856x1024_apex.yml"
print(f"yaml_name: {yaml_name}, split_type: {split_type}")

trainer = load_trainer_model(yaml_name, evaluation_mode=True, deployment_mode=True, evaluation_batch_size=1,
                             evaluation_torch_device="cuda:1", evaluation_load_mode="best_mIoU",
                             return_dataset_info=True)

bdd_set = BDD100k(
    root=trainer.dataset_root_folder,
    split=split_type,
    return_dataset_info=True,
    transforms=trainer.common_eval_transformation,
    target_type="drivable-masks"  # This is dummy, target is not important for this code
)

class_mapping = {}
for cls in trainer.train_set.classes:
    class_mapping.update({cls.train_id: cls.id})

loader = DataLoader(bdd_set,
                    batch_size=1,
                    collate_fn=trainer.collate_fn)

target_folder = os.path.join("/media/hdd/bdd100k", yaml_name.split('/')[-1][:-4], split_type)
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

class_info_list = []
for item in trainer.train_set.classes:
    class_info_list.append(item._asdict())

with open(os.path.join("/media/hdd/bdd100k", "class_settings.json"), 'w') as f:
    json.dump(class_info_list, f)

trainer.model.eval()
count = 0

for count, (images, _, info) in tqdm(enumerate(loader)):
    images, _ = trainer.to_device(images)

    head_outputs = trainer.model(images)

    outputs = trainer.postprocess_outputs(head_outputs)
    outputs_array = trainer.to_numpy(outputs)

    # conversion part
    original_outputs_array = outputs_array.copy()
    for output_id, target_id in class_mapping.items():
        outputs_array[original_outputs_array == output_id] = target_id

    save_numpy_mask_as_image(outputs_array=outputs_array, info=info, target_folder=target_folder)

    # visualize_target = np.expand_dims(outputs_array, 0)
    # trainer.visualize(image=images,
    #                   ground_truth=None,
    #                   outputs=outputs,
    #                   show=True,
    #                   wandb_log_name="deneme",
    #                   save_images_to_local=False,
    #                   image_name=f"sample_{count}")

    # count += 1
    # if count == 4:
    #     break
