import os
from Initialize import load_trainer_model

image_count = 10
yaml_name = "settings/Deeplab/deeplabv3_resnet18_culane_640x368.yml"
trainer = load_trainer_model(yaml_name, evaluation_mode=True,
                             evaluation_wandb_enabled=False, deployment_mode=True,
                             evaluation_batch_size=1)

model = trainer.model
model.eval()

image_list = []
label_list = []

for count, (image, label) in enumerate(trainer.val_loader):
    image_list.append(image)
    label_list.append(label)
    if count == image_count:
        break

trainer.prepare_onnx_for_dlc_conversion(image=image_list)

if not os.path.isfile(trainer.dlc_file):
    trainer.self_snpe_onnx_to_dlc()

if not os.path.isfile(trainer.dlc_quantized_file):
    trainer.self_snpe_dlc_to_quantized_dlc()

trainer.self_snpe_net_run_quantized()

for image_idx in range(image_count):
    dlc_outputs = trainer.self_read_raw_file(result_idx=image_idx)
    dlc_outputs = trainer.postprocess_outputs(dlc_outputs)

    image = image_list[image_idx]
    label = label_list[image_idx]
    trainer.visualize(image=image,
                      ground_truth=label,
                      optional_outputs=dlc_outputs,
                      show=False,
                      wandb_log_name="dlc_quantized_check_correct")

if trainer.wandb_object:
    trainer.wandb_object.finish()
