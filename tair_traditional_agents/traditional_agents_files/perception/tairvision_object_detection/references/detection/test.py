import time

import torch
import torch.utils.data

import cv2
import numpy as np

import tairvision

import tairvision.references.detection.presets as presets

from tairvision.references.detection.coco_utils import get_label, get_label_color
from tairvision.references.detection.bdd_utils import get_label as get_label_bdd
from tairvision.references.detection.voc_utils import get_label as get_label_voc
from tairvision.references.detection.nuimages_utils import get_label as get_label_nuimg
from tairvision.references.detection.utils import get_dataset
from tairvision.references.detection.config import get_arguments
from tairvision.references.detection.visualization_utils import vis_det_bboxes
import tairvision.references.detection.transforms as T
from PIL import Image
import copy


EPS = 1e-2


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
    for images, targets in data_loader_test:
        pre_img = T.PILToTensor()(images[0])
        pre_img_tensor = T.ConvertImageDtype(torch.float)(pre_img[0])
        pre_img = np.array(pre_img_tensor[0]).transpose(1, 2, 0)
        pre_img = (pre_img * 255).astype('uint8')

        image = Image.fromarray(pre_img)
        image, original_image_sizes = transform(image, None)
        image = image.unsqueeze(0).to(device)
        t1 = time.time()
        with torch.no_grad():
            output = model(image, original_image_sizes)
        t2 = time.time()
        boxes = output[0]['boxes'].cpu()
        scores = output[0]['scores'].cpu()
        # labels = output[0]['labels'].cpu().numpy()
        labels = output[0]['labels'].clamp_(1).cpu().numpy()
        masks = output[0]['masks'].cpu() if 'masks' in output[0].keys() else None

        threshold_mask = scores > float(args.score_thres)
        boxes = boxes[threshold_mask.numpy()]
        labels = labels[threshold_mask.numpy()]
        if masks is not None:
            masks = masks[threshold_mask.numpy()]

        boxes_gt = torch.tensor(targets[0]['boxes'])
        masks_gt = torch.tensor(targets[0]['masks'])
        labels_gt = targets[0]['labels']

        if args.dataset == "bdd":
            label_names = [*map(get_label_bdd, [*labels])]
            label_names_gt = [*map(get_label_bdd, [*labels_gt])]
        elif args.dataset == "voc":
            label_names = [*map(get_label_voc, [*labels])]
            label_names_gt = [*map(get_label_voc, [*labels_gt])]
        elif args.dataset == "nuimages":
            label_names = [*map(get_label_nuimg, [*labels])]
            label_names_gt = [*map(get_label_nuimg, [*labels_gt.numpy()])]
        else:
            label_names = [*map(get_label, [*labels])]
            label_names_gt = [*map(get_label, [*labels_gt])]

        colors = np.array([get_label_color(label) for label in labels])
        colors_gt = np.array([get_label_color(label) for label in labels_gt])

        if masks is not None:
            res_masks = masks[:, 0, :, :]
        else:
            res_masks = masks
            masks_gt = None

        res_img = vis_det_bboxes(copy.deepcopy(pre_img), boxes, res_masks, color=colors)
        gt_img = vis_det_bboxes(copy.deepcopy(pre_img), boxes_gt, masks_gt, color=colors_gt)

        res_img = cv2.copyMakeBorder(res_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        gt_img = cv2.copyMakeBorder(gt_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        pre_img = cv2.copyMakeBorder(pre_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT)

        vis = np.concatenate((pre_img, res_img, gt_img), axis=1)

        cv2.imshow('Test', vis)
        cv2.waitKey()

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

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
