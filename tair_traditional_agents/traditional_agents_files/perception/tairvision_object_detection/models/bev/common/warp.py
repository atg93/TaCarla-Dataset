from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import cv2
from tairvision_object_detection.datasets.nuscenes import prepare_dataloaders
from tairvision_object_detection.models.bev.lss.training.process import (ResizeCropRandomFlipNormalize,
                                                        get_resizing_and_cropping_parameters, FilterClasses)
from tairvision_object_detection.models.bev.lss.training.config import get_cf_from_yaml
from tairvision_object_detection.models.bev.common.utils.geometry import (calculate_birds_eye_view_parameters, cumulative_warp_features,
                                                         cumulative_warp_features_reverse)
from tairvision_object_detection.models.bev.lss.utils.static import get_targets_static
from tairvision_object_detection.models.bev.lss.utils.visualization import get_bitmap_with_road


def warp_segmentation_images(config_file, dataroot, version):
    cfg = get_cf_from_yaml(config_file)

    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1

    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.VERSION = version

    # Bird's-eye view parameters
    bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
        cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
    )
    bev_size = bev_dimension.numpy()[0:2]
    bev_res = bev_resolution.numpy()[0:2]
    spatial_extent = (cfg.LIFT.X_BOUND[1], cfg.LIFT.Y_BOUND[1])

    augmentation_parameters = get_resizing_and_cropping_parameters(cfg)
    transforms_val = ResizeCropRandomFlipNormalize(augmentation_parameters, enable_random_transforms=True)

    filter_classes = FilterClasses(cfg.DATASET.FILTER_CLASSES)
    _, valloader = prepare_dataloaders(cfg, None, transforms_val, filter_classes)

    cv2.setNumThreads(4)

    pause = False
    for i, batch in enumerate(tqdm(valloader)):
        receptive_field = cfg.TIME_RECEPTIVE_FIELD
        t = receptive_field - 1  # Index of the present frame

        lanes_gt = get_targets_static(batch, receptive_field)['lanes'][0, 0].cpu().numpy()
        lines_gt = get_targets_static(batch, receptive_field)['lines'][0, 0].cpu().numpy()

        segm_gts_concat = []
        segm_gts = batch['segmentation'][0, :receptive_field, 0].clone().cpu().numpy()
        for j, segm_gt in enumerate(segm_gts):
            segm_gt = get_bitmap_with_road(segm_gt[None, ...], lanes_gt, lines_gt, bev_size=bev_size)
            segm_gts_concat.append(segm_gt)

        segm_gts_concat = np.concatenate(segm_gts_concat, axis=1)
        segm_gts_concat = cv2.resize(segm_gts_concat, dsize=(len(segm_gts)*400, 400), interpolation=cv2.INTER_CUBIC)

        segm_gts_no_warp = get_bitmap_with_road_history(segm_gts, lanes_gt, lines_gt, bev_size=bev_size)
        segm_gts_no_warp = cv2.resize(segm_gts_no_warp, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)

        segm_gts_warp = cumulative_warp_features(batch['segmentation'][0:1, :receptive_field].clone().cpu().float(),
                                                 batch['future_egomotion'][0:1, :receptive_field].clone().cpu(),
                                                 mode='bilinear',
                                                 spatial_extent=spatial_extent)

        segm_gts_warp = get_bitmap_with_road_history(segm_gts_warp[0, :, 0].long().numpy(), lanes_gt, lines_gt,
                                                     bev_size=bev_size)
        segm_gts_warp = cv2.resize(segm_gts_warp, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)

        segm_gts_rev_warp = cumulative_warp_features_reverse(batch['segmentation'][0:1, receptive_field:].clone().cpu().float(),
                                                             batch['future_egomotion'][0:1, receptive_field:].clone().cpu(),
                                                             mode='bilinear',
                                                             spatial_extent=spatial_extent)

        segm_gts_rev_warp = get_bitmap_with_road_history(segm_gts_rev_warp[0, :, 0].long().numpy(), lanes_gt, lines_gt,
                                                         bev_size=bev_size)
        segm_gts_rev_warp = cv2.resize(segm_gts_rev_warp, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)

        cv2.imshow("Concat", cv2.cvtColor(segm_gts_concat, cv2.COLOR_RGB2BGR))
        cv2.imshow("No warp", cv2.cvtColor(segm_gts_no_warp, cv2.COLOR_RGB2BGR))
        cv2.imshow("Warp", cv2.cvtColor(segm_gts_warp, cv2.COLOR_RGB2BGR))
        cv2.imshow("Reverse Warp", cv2.cvtColor(segm_gts_rev_warp, cv2.COLOR_RGB2BGR))

        ch = cv2.waitKey(1)
        if ch == 27:
            break
        elif ch == 32:
            pause = not pause

        while pause:
            ch = cv2.waitKey(1)
            if ch == 32:
                pause = not pause

    cv2.destroyAllWindows()


def get_bitmap_with_road_history(xs, lanes, lines, bev_size=(200, 200)):
    output = np.zeros((*bev_size, 3)).astype(np.uint8)

    lane_mask = (lanes > 0).any(0)
    line_mask = (lines > 0).any(0)

    output[lane_mask] = [75, 75, 75]
    output[line_mask] = [255, 255, 255]

    for i, x in enumerate(xs):
        x_mask = x > 0
        alpha = ((i + 1)/len(xs))
        output[x_mask] = [np.int(np.round(255*alpha)), np.int(np.round(172*alpha)), np.int(np.round(28*alpha))]

    output[95:105, 97:103] = [52, 152, 219]

    return output


if __name__ == '__main__':
    parser = ArgumentParser(description='Image Warping Demonstration')
    parser.add_argument('--dataroot', default='/datasets/nu/nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')
    parser.add_argument('--config-file', default=None, type=str, help='path to checkpoint')

    args = parser.parse_args()

    warp_segmentation_images(args.config_file, args.dataroot, args.version)
