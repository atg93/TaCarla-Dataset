from Initialize import load_trainer_model
from tairvision.utils import retry_if_cuda_oom
import argparse


def visualize(args):
    trainer = load_trainer_model(args.yaml_name, evaluation_mode=args.evaluation_mode, deployment_mode=False,
                                 evaluation_torch_device="cuda:0", evaluation_load_mode="final")
    if args.activate_train_loader:
        loader = trainer.train_loader
    else:
        loader = trainer.val_loader

    trainer.model.eval()
    for count, (images, labels) in enumerate(loader):
        images, targets = trainer.to_device(images, labels)

        head_outputs = trainer.model(images)
        # trainer.visualize_heatmaps(head_outputs)
        outputs = retry_if_cuda_oom(trainer.postprocess_outputs)(head_outputs)

        trainer.visualize(image=images,
                          ground_truth=labels,
                          outputs=outputs,
                          show=True,
                          wandb_log_name="deneme",
                          save_images_to_local=False,
                          image_name=f"sample_{count}")

        print(count)
        if count == args.count-1:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualization for segmentation and panoptic segmentation models')
    parser.add_argument('yaml_name', metavar='FILE', help='path to the yaml')
    parser.add_argument('--count', type=int, default=2, help='Number of samples to visualize')
    parser.add_argument('--evaluation_mode_disabled', dest="evaluation_mode", action='store_false', help='disable evaluation mode')
    parser.add_argument('--activate_train_loader', dest="activate_train_loader", action='store_true', help='activate train loader')
    arguments = parser.parse_args()
    visualize(arguments)


