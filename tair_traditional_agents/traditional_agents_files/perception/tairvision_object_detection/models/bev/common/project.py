from argparse import ArgumentParser
from tqdm import tqdm

import cv2
from tairvision_object_detection.datasets.nuscenes import prepare_dataloaders
from tairvision_object_detection.models.bev.common.nuscenes.process import (ResizeCropRandomFlipNormalize,
                                                           get_resizing_and_cropping_parameters, FilterClasses)
from tairvision_object_detection.models.bev.lss.training.config import get_cf_from_yaml
from tairvision_object_detection.models.bev.common.utils.geometry import calculate_birds_eye_view_parameters
from tairvision_object_detection.models.bev.common.utils.visualization import image2bev
from tairvision_object_detection.models.bev.lss.utils.network import preprocess_batch


def project_from_image_to_bev(config_file, dataroot, version):
    cfg = get_cf_from_yaml(config_file)

    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1

    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.VERSION = version

    # Bird's-eye view parameters
    bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
        cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
    )

    augmentation_parameters = get_resizing_and_cropping_parameters(cfg)
    transforms_val = ResizeCropRandomFlipNormalize(augmentation_parameters, enable_random_transforms=False)

    filter_classes = FilterClasses(cfg.DATASET.FILTER_CLASSES)
    _, valloader = prepare_dataloaders(cfg, None, transforms_val, filter_classes)

    for i, batch in enumerate(tqdm(valloader)):
        device = 'cuda:0'
        preprocess_batch(batch, device)
        bev_image = image2bev(batch, cfg, bev_dimension)
        cv2.imshow("bev", cv2.cvtColor(bev_image[0], cv2.COLOR_RGB2BGR))

        ch = cv2.waitKey(1)
        if ch == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser(description='Image to BEV projection')
    parser.add_argument('--dataroot', default='/datasets/nu/nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')
    parser.add_argument('--config-file', default=None, type=str, help='path to checkpoint')

    args = parser.parse_args()

    project_from_image_to_bev(args.config_file, args.dataroot, args.version)
