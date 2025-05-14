import time

import torch
import torch.utils.data

import cv2
import numpy as np
from copy import deepcopy

import tairvision
from torchvision.transforms import ToPILImage

import tairvision.references.detection.presets as presets
from tairvision.utils import draw_bounding_boxes

from tairvision.references.detection.coco_utils import get_label as get_label_coco
from tairvision.references.detection.coco_utils import get_label_color as get_label_color_coco
from tairvision.references.detection.bdd_utils import get_label as get_label_bdd
from tairvision.references.detection.bdd_utils import get_label_color as get_label_color_bdd
from tairvision.references.detection.nuimages_utils import get_label as get_label_nuimg
from tairvision.references.detection.nuimages_utils import get_label_color as get_label_color_nuimg
from tairvision.references.detection.utils import get_dataset
from tairvision.references.detection.config import get_arguments
from tairvision.models.tracking.track_adapter import TAIRTrackAdapter

from PIL import Image

from apex import amp


def main(args):
    device = torch.device(args.device)

    dataset_test, num_classes, collate_fn, num_keypoints = get_dataset(args.data_path, args.dataset, "val",
                                                                       presets.DetectionPresetEval(None))

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

    model = tairvision.models.detection.__dict__[args.model](num_classes=num_classes,
                                                             num_keypoints=num_keypoints,
                                                             **kwargs)
    model.to(device)

    if args.use_track:
        track = TAIRTrackAdapter(args, match_thresh=0.9, iou_thresh=0.6, score_thres=0.5, classes=[1, 2, 3, 4, 6, 7, 8])

    # amp.initialize(model, opt_level="O3", keep_batchnorm_fp32=True)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    transform = presets.DetectionPresetEval(None)

    cap = cv2.VideoCapture(args.video_path)

    print('Press "Esc", "q" or "Q" to exit.')
    frame_count = 0
    ret_val = True
    print_freq = 1
    print_count = 0
    time_preprocess, time_model, time_track, time_cv2 = [], [], [], []
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(args.save_path, fourcc, 30.0, (1280, 1440))
    while ret_val:
        ret_val, image = cap.read()
        im0_size = image.shape[:2]
        images_to_show = deepcopy(image)
        frame_count += 1
        if frame_count % print_freq == 0:
            t0 = time.time()
            image = Image.fromarray(image)
            image, original_image_sizes = transform(image, None)
            image = image.unsqueeze(0).to(device)
            t1 = time.time()
            with torch.no_grad():
                output = model(image, original_image_sizes)
            t2 = time.time()

            if 'image_shape' in output[0].keys():
                popped_image_shape = output[0].pop('image_shape')
            masks = output[0]['masks'].cpu() if 'masks' in output[0].keys() else None
            if args.use_track:
                outputs_track = track.update(output[0], im0_size, im0_size)
                other_keys = ['masks'] if masks is not None else []
                boxes, ids, scores, labels, others = track.postprocess_xyxy(outputs_track, other_keys=other_keys)
                if masks is not None:
                    masks = others['masks']
                    masks = torch.reshape(masks, (-1, images_to_show.shape[0], images_to_show.shape[1]))
            else:
                boxes = output[0]['boxes'].cpu()
                scores = output[0]['scores'].cpu()
                labels = output[0]['labels'].cpu().numpy()

                if 'dist_scores' in output[0].keys():
                    dist_scores = output[0]['dist_scores'].cpu().squeeze(1)

                    labels[(dist_scores < 0.5) * (scores > 0.30)] = num_classes + 1

                    boxes = boxes[(scores > float(args.score_thres)) + ((dist_scores < 0.5) * (scores > 0.3))]
                    labels = labels[(scores > float(args.score_thres)) + ((dist_scores < 0.5) * (scores > 0.3))]
                else:
                    boxes = boxes[scores > float(args.score_thres)]
                    labels = labels[scores > float(args.score_thres)]

            t3 = time.time()

            if args.dataset == "bdd":
                labels_str = [*map(get_label_bdd, [*labels])]
                label_colors = [*map(get_label_color_bdd, [*labels])]
            elif args.dataset == "coco":
                labels_str = [*map(get_label_coco, [*labels])]
                label_colors = [*map(get_label_color_coco, [*labels])]
            elif args.dataset == "nuimages":
                labels_str = [*map(get_label_nuimg, [*labels])]
                label_colors = [*map(get_label_color_nuimg, [*labels])]
            else:
                labels_str = [*map(get_label_coco, [*labels])]
                label_colors = [*map(get_label_color_coco, [*labels])]

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break

            images_to_show = torch.from_numpy(images_to_show).permute(2, 0, 1)

            if masks is not None:
                if not args.use_track:
                    masks = masks[scores > float(args.score_thres)]
                if len(masks) > 0:
                    masks_max = masks.max(0)[0]
                    masks_max = torch.reshape(masks_max, (1, 720, 1280))
                    masks_max_mask = (masks_max > 0.10).squeeze(0)

                    image_segm = 0.8 * torch.tensor([[0], [100], [0]]) * masks_max[:, masks_max_mask] + \
                                 (1 - 0.8) * images_to_show[:, masks_max_mask]
                    images_to_show[:, masks_max_mask] = image_segm.to(torch.uint8)

            output_to_show = draw_bounding_boxes(images_to_show, boxes, labels_str, font_size=12, colors=label_colors)
            output_to_show = np.array(ToPILImage()(output_to_show))

            cv2.imshow("Seperate", output_to_show)
            t4 = time.time()
            out.write(output_to_show)

            time_preprocess.append(t1 - t0)
            time_model.append(t2 - t1)
            time_track.append(t3 - t2)
            time_cv2.append(t4 - t3)

            if print_count == 100:
                print("Preprocess: " + str(np.asarray(time_preprocess).mean()))
                print("Model: " + str(np.asarray(time_model).mean()))
                print("Track/Filter: " + str(np.asarray(time_track).mean()))
                print("CV2: " + str(np.asarray(time_cv2).mean()))
                print_count = 0
                time_preprocess, time_model, time_cv2 = [], [], []
            else:
                print_count += 1

    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main(args)
