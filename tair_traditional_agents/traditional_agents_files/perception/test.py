import os
import time

import torch
import torch.utils.data

import cv2
import numpy as np

import leaderboard.autoagents.traditional_agents_files.perception.tairvision

import leaderboard.autoagents.traditional_agents_files.perception.tairvision.references.detection.presets as presets

from leaderboard.autoagents.traditional_agents_files.perception.tairvision.references.detection.coco_utils import get_label, get_label_color
from leaderboard.autoagents.traditional_agents_files.perception.tairvision.references.detection.utils import get_dataset
from leaderboard.autoagents.traditional_agents_files.perception.tairvision.references.detection.config import get_arguments
import copy


EPS = 1e-2

def draw_boxes(img,x_min,y_min,x_max,y_max,color):
    if color == 3:
        color = (0, 0, 255)
    elif color == 2:
        color = (0, 255, 255)
    elif color == 1:
        color = (0, 255, 0)
    else:
        color = (255, 255, 255)

    cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), color, 1)
    cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), color, 1)
    cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), color, 1)
    cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), color, 1)

    return img

def main(args):

    device = torch.device(args.device)

    dataset_test, num_classes, collate_fn, num_keypoints = get_dataset(args.data_path, args.dataset, "val", None)
    transform = presets.DetectionPresetEval(None, min_size=args.transform_min_size, max_size=args.transform_max_size)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   sampler=test_sampler,
                                                   num_workers=0,
                                                   collate_fn=collate_fn)

    print("Creating model")
    kwargs = {
        "type": args.backbone,
        "trainable_backbone_layers": args.trainable_backbone_layers,
        "pretrained": args.pretrained,
        "pyramid_type": args.pyramid_type,
        "repeats": args.bifpn_repeats,
        "fusion_type": args.fusion_type,
        "depthwise": args.use_depthwise,
        "use_P2": args.use_P2,
        "no_extra_blocks": args.no_extra_blocks,
        "extra_before": args.extra_before,
        "context_module": args.context_module,
        "loss_weights": args.loss_weights,
        "nms_thresh": args.nms_thresh,
        "post_nms_topk": args.post_nms_topk,
        "min_size": args.transform_min_size,
        "max_size": args.transform_max_size
    }

    if "fcos" in args.model:
        kwargs["fpn_strides"] = args.fpn_strides
        kwargs["sois"] = args.sois
        kwargs["thresh_with_ctr"] = args.thresh_with_ctr
        kwargs["use_deformable"] = args.use_deformable
    else:
        kwargs["anchor_sizes"] = args.anchor_sizes
        if "rcnn" in args.model:
            if args.rpn_score_thresh is not None:
                kwargs["rpn_score_thresh"] = args.rpn_score_thresh

    if args.model_kwargs:
        kwargs.update(args.model_kwargs)

    model = tairvision.models.detection.__dict__[args.model](num_classes=num_classes,
                                                             num_keypoints=num_keypoints,
                                                             **kwargs)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print('Press "Esc", "q" or "Q" to exit.')
    for index, data in enumerate(data_loader_test):
        images, targets = data
        t1 = time.time()
        with torch.no_grad():
            output = model([images[0].to(device)])
        t2 = time.time()

        pre_img = np.array(images[0].reshape(396,704,3).numpy())

        boxes = output[0]['boxes'].cpu()
        scores = output[0]['scores'].cpu()
        # labels = output[0]['labels'].cpu().numpy()
        labels = output[0]['labels'].clamp_(1).cpu().numpy()
        #speed = output[0]['speed'].clamp_(1).cpu().numpy()-1
        masks = output[0]['masks'].cpu() if 'masks' in output[0].keys() else None

        threshold_mask = scores > float(args.score_thres)
        boxes = boxes[threshold_mask.numpy()]
        labels = labels[threshold_mask.numpy()]
        if masks is not None:
            masks = masks[threshold_mask.numpy()]

        boxes_gt = torch.tensor(targets[0]['boxes'])
        masks_gt = targets[0]['masks'] if 'masks' in targets[0].keys() else None
        labels_gt = targets[0]['labels']
        #speed_gt = targets[0]['speed']
        print("*"*50,"labels:",labels, "labels_gt:")

        colors = np.array([get_label_color(label) for label in labels])
        colors_gt = np.array([get_label_color(label) for label in labels_gt])

        if masks is not None:
            res_masks = masks[:, 0, :, :]
        else:
            res_masks = masks
            masks_gt = None

        if len(boxes) !=0:
            res_img = copy.deepcopy(pre_img)
            for bb in boxes:
                res_img = draw_boxes(res_img, bb[0], bb[1], bb[2], bb[3],labels[0])
        else:
            res_img = copy.deepcopy(pre_img)

        if len(boxes_gt) !=0:
            gt_img = copy.deepcopy(pre_img)
            for bb in boxes_gt:
                gt_img = draw_boxes(gt_img, bb[0], bb[1], bb[2], bb[3],labels_gt[0])
        else:
            gt_img = copy.deepcopy(pre_img)


        vis = np.concatenate((res_img,gt_img), axis=1)
        cv2.imwrite("fcos_resim/vis"+str(index)+".png",vis)
        print(os.getcwd())


        t3 = time.time()

        # time_preprocess = t1 - t0
        time_model = t2 - t1
        time_cv2 = t3 - t2

        # print("Preprocess: " + str(time_preprocess))
        print("Model: " + str(time_model))
        print("CV2: " + str(time_cv2))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main(args)
