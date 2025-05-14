import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tairvision import retry_if_cuda_oom

def mc_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def mc_predictions(trainer,
                   forward_passes,
                   model,
                   n_classes):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size = trainer.size
    h = size[0]
    w = size[1]

    with torch.no_grad():

        softmax = nn.Softmax(dim=1)
        model.eval()
        mc_dropout(model)

        for step, (image, label) in enumerate(trainer.val_loader):

            image = image.to(device)
            predictions = np.empty((0, n_classes, h, w))
            features, result = model.forward_backbone(image)

            for i in range(forward_passes):

                output = model.forward_mc(image, features, result)
                output_vis = retry_if_cuda_oom(trainer.postprocess_outputs)(output)
                output = output['out']
                output = softmax(output)  # shape (n_samples, n_classes)
                predictions = np.vstack((predictions, output.cpu().numpy()))

            predictions = torch.from_numpy(predictions).to(device)

            var = torch.var(predictions, dim=0)
            mean_var = torch.mean(var, dim=0)
            mean_var = mean_var.cpu().numpy()  # shape (n_samples, n_classes)

            trainer.visualize(image=image,
                              ground_truth=label,
                              outputs=output_vis,
                              show=True,
                              wandb_log_name="deneme",
                              save_images_to_local=False,
                              image_name=f"sample_{step}")

            plt.imshow(mean_var, cmap='gray_r')
            plt.show()


def uncertainty_aux(trainer,
                   model,
                   n_classes,
                   n_samples):
    h = 368
    w = 640

    with torch.no_grad():

        dropout_predictions = np.empty((0, n_samples, n_classes, h, w))
        softmax = nn.Softmax(dim=1)
        kl_distance = nn.KLDivLoss( reduction = 'none')
        log_sm = torch.nn.LogSoftmax(dim=1)
        sm = torch.nn.Softmax(dim=1)

        model.eval()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for step, (image, label) in enumerate(trainer.val_loader):

            image = image.to(device)
            predictions = np.empty((0, n_classes, h, w))
            output = model(image)
            output_vis = retry_if_cuda_oom(trainer.postprocess_outputs)(output)

            out = output['out']
            aux = output['aux']

            predictions = np.vstack((predictions, out.cpu().numpy()))
            predictions = np.vstack((predictions, aux.cpu().numpy()))
            variance = torch.sum(kl_distance(log_sm(out), sm(aux)), dim=1)
            exp_variance = torch.exp(-variance)

            predictions = torch.from_numpy(predictions).to(device)
            var = torch.var(predictions, dim=0)
            mean_var = torch.mean(var, dim=0)

            trainer.visualize(image=image,
                              ground_truth=label,
                              outputs=output_vis,
                              show=True,
                              wandb_log_name="deneme",
                              save_images_to_local=False,
                              image_name=f"sample_{step}")

            mean_var = mean_var.cpu().numpy()
            mean_variance = torch.mean(exp_variance, dim=0)
            mean_variance = mean_variance.cpu().numpy()
            plt.imshow(mean_var, cmap='gray')
            plt.show()
