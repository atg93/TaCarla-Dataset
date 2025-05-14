import datetime
import os
import time

import torch
import torch.utils.data

import cv2
import numpy as np

import tairvision
from torchvision.transforms import ToPILImage

import tairvision.references.detection.presets as presets
from tairvision.utils import draw_bounding_boxes

from tairvision.references.detection.coco_utils import get_label, get_label_color
from tairvision.references.detection.bdd_utils import get_label as get_label_bdd
from tairvision.references.detection.utils import get_dataset
from tairvision.references.detection.config import get_arguments
from tairvision.references.detection.visualization_utils import vis_det_bboxes

import copy


EPS = 1e-2

def main(args):

    device = torch.device(args.device)

    dataset_test, num_classes, collate_fn, num_keypoints = get_dataset(args.data_path, args.dataset, "val-ins",
                                                                       presets.DetectionPresetEval(None))

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   sampler=test_sampler,
                                                   num_workers=0,
                                                   collate_fn=collate_fn)


    print('Press "Esc", "q" or "Q" to exit.')
    for images, targets in data_loader_test:
        pre_img = cv2.cvtColor(images[0].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        images = list(img.to(device) for img in images)

        pre_img = (pre_img * 255).astype('uint8')

        boxes_gt = torch.tensor(targets[0]['boxes'])
        masks_gt = torch.tensor(targets[0]['masks'])
        labels_gt = targets[0]['labels']


        colors_gt = np.array([get_label_color(label) for label in labels_gt])

        gt_img = vis_det_bboxes(copy.deepcopy(pre_img), boxes_gt, masks_gt, color=colors_gt)

        gt_img = cv2.copyMakeBorder(gt_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        pre_img = cv2.copyMakeBorder(pre_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT)

        vis = np.concatenate((pre_img, gt_img), axis=1)

        cv2.imshow('abc', vis)
        cv2.waitKey()

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break


    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main(args)
