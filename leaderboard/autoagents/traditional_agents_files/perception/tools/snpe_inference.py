import os
from Initialize import load_trainer_model

yaml_name = "settings/Deeplab/deeplabv3_resnet18_culane_640x368.yml"
trainer = load_trainer_model(yaml_name, evaluation_mode=True,
                             evaluation_wandb_enabled=False, deployment_mode=True,
                             evaluation_batch_size=1)

model = trainer.model
model.eval()

image, label = next(iter(trainer.val_loader))
trainer.prepare_onnx_for_dlc_conversion(image=image)

if not os.path.isfile(trainer.dlc_file):
    trainer.self_snpe_onnx_to_dlc()

trainer.self_snpe_net_run()

dlc_outputs = trainer.self_read_raw_file()

dlc_outputs = trainer.postprocess_outputs(dlc_outputs)

trainer.visualize(image=image,
                  ground_truth=label,
                  optional_outputs=dlc_outputs,
                  show=False,
                  wandb_log_name="dlc_check")

if trainer.wandb_object:
    trainer.wandb_object.finish()
