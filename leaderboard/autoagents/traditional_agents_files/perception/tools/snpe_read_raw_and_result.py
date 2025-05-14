import os
import glob
from Initialize import load_trainer_model

yaml_name = "settings/deeplabv3_resnet18_culane_640x368.yml"
trainer = load_trainer_model(yaml_name, evaluation_mode=True,
                             evaluation_wandb_enabled=False, deployment_mode=True,
                             evaluation_batch_size=1)


model = trainer.model
model.eval()

result_format = os.path.join(trainer.snpe_result_output_folder, "Result_*")
sample_count = len(glob.glob(result_format))

for result_idx in range(sample_count):
    dlc_outputs = trainer.self_read_raw_file(result_idx=result_idx)
    dlc_outputs = trainer.postprocess_outputs(dlc_outputs)
    dlc_image = trainer.self_read_raw_image(sample_idx=result_idx)
    trainer.visualize(image=dlc_image,
                      optional_outputs=dlc_outputs,
                      show=False,
                      wandb_log_name="snpe_result_check")

if trainer.wandb_object:
    trainer.wandb_object.finish()
