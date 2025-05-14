import tairvision
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Optional, Union, Dict
import torch
import torch.utils.data
import tairvision
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as F

from tairvision.utils import draw_bounding_boxes
from tairvision.references.detection.bdd_utils import get_label as get_label_bdd
from tairvision.references.detection.bdd_utils import get_label_color as get_label_color_bdd

from utils import InferenceSegmentationTorch, InferenceSegmentationTensorRT, InferenceSegmentationFourHead

import tairvision.references.detection.presets as presets


def main(args):
    if args.cropped:
        ratio = 960 / 720
    else:
        ratio = 1280 / 720

    kwargs = {
        'type': 'regnet_y_8gf',
        'trainable_backbone_layers': None,
        'pretrained': False,
        'pyramid_type': 'bifpn',
        'repeats': 3,
        'fusion_type': 'fastnormed',
        'depthwise': False,
        'use_P2': False,
        'no_extra_blocks': False,
        'extra_before': False,
        'context_module': None,
        'loss_weights': [1.0, 1.0, 1.0, 0.25],
        'nms_thresh': 0.6,
        'post_nms_topk': 100,
        'min_size': 800,
        'max_size': 1333,
        'fpn_strides': [8, 16, 32, 64, 128],
        'sois': [-1, 64, 128, 256, 512, 100000000],
        'thresh_with_ctr': False,
        'use_deformable': False
    }

    model = tairvision.models.detection.__dict__['fcos_regnet_fpn'](num_classes=11,
                                                                    num_keypoints=0,
                                                                    **kwargs)

    model.to("cuda")

    checkpoint = torch.load(args.detection_weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(f"/workspace/demo.avi", fourcc, 20.0, (int(ratio * args.crop_size * 4),
                                                                 args.crop_size * 2))
    # dataset_test, num_classes, collate_fn, num_keypoints = get_dataset(args.data_path, args.dataset, "val",
    #                                                                    presets.DetectionPresetEval(None))

    transform = presets.DetectionPresetEval(None)

    inference_instance = InferenceSegmentationFourHead(args.segmentation_yaml)
    inference_mapillary_instance = InferenceSegmentationTorch(
        "settings/deeplabv3_resnet50_mapillary12_1856x1024_apex.yml")

    cap = cv2.VideoCapture(args.video_path)
    frame_num = 0
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, image = cap.read()
        if ret_val is False:
            break
        if not frame_num % 10 == 0:
            frame_num += 1
            continue

        if args.cropped:
            input = image[:, 160:-160, :]
        else:
            input = image

        outputs, frame_resized = inference_instance.take_inference(input, preserve_ratio=True)
        outputs_mapillary, frame_resized_mapillary = inference_mapillary_instance.take_inference(input, preserve_ratio=True)
        output_frames, frame_merged = inference_instance.visualize(outputs, input)
        _, frame_merged = inference_mapillary_instance.visualize(outputs_mapillary, frame_merged)
        output_frames_mapillary, _ = inference_mapillary_instance.visualize(outputs_mapillary, input)


        image = Image.fromarray(input)
        image, _ = transform(image, None)
        image = image.unsqueeze(0).to("cuda")
        with torch.no_grad():
            output = model(image)
        boxes = output[0]['boxes'].cpu()
        scores = output[0]['scores'].cpu()
        labels = output[0]['labels'].cpu().numpy()

        boxes = boxes[scores > float(args.score_thres)]
        labels = labels[scores > float(args.score_thres)]
        labels_str = [*map(get_label_bdd, [*labels])]  # if args.dataset == 'bdd' else [*map(get_label_coco, [*labels])]
        label_colors = [*map(get_label_color_bdd,
                             [*labels])]  # if args.dataset == 'bdd' else [*map(get_label_color_coco, [*labels])]

        images_to_show = (image.cpu().clone() * 255).to(dtype=torch.uint8).squeeze(0)
        output_to_show = draw_bounding_boxes(images_to_show, boxes, labels_str, font_size=12, colors=label_colors, width=4)

        frame_merged = Image.fromarray(frame_merged)
        frame_merged, _ = transform(frame_merged, None)
        frame_merged = (frame_merged * 255).to(dtype=torch.uint8).squeeze(0)
        frame_merged = draw_bounding_boxes(frame_merged, boxes, labels_str, font_size=12, colors=label_colors, width=4)
        frame_merged = np.array(ToPILImage()(frame_merged))

        detection_show = np.array(ToPILImage()(output_to_show))

        detection_frame = cv2.resize(detection_show,
                                     (int(ratio * args.crop_size), args.crop_size))

        mapillary_detection_frame = cv2.resize(output_frames_mapillary[0],
                                     (int(ratio * args.crop_size), args.crop_size))

        resized_output_frames = []
        for output_frame in output_frames:
            output_frame = cv2.resize(output_frame,
                                      (int(ratio * args.crop_size), args.crop_size))
            resized_output_frames.append(output_frame)

        frame_merged_resized = cv2.resize(frame_merged,
                                          (int(ratio * args.crop_size) * 2,
                                           args.crop_size * 2))

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        if key == ord('p'):
            while True:
                key = cv2.waitKey(1)
                if key == ord('p'):
                    break
        if key == ord('s'):
            args.save_video = True

        # cv2.imshow("joint", frame_resized)
        vis1 = np.concatenate((resized_output_frames[0], detection_frame), axis=0)
        vis2 = np.concatenate((resized_output_frames[2], mapillary_detection_frame), axis=0)

        vis3 = np.concatenate((vis1, vis2), axis=1)

        vis4 = np.concatenate((vis3, frame_merged_resized), axis=1)
        cv2.imshow("joint", vis4)
        # cv2.imshow("joint2", output_image_ver2)
        # cv2.imshow("DEMO", output_to_show_merged)
        if args.save_video:
            out.write(vis4)
        # cv2.imshow("general segmentation", output_image2)
        # cv2.imshow("drivable area", output_image3)
        # cv2.imshow("lane type", output_image4)
        # cv2.imshow("bdd general segmentation", output_image5)
        frame_num += 1

    cv2.destroyAllWindows()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--video-path', default='/datasets/forddata/V1.mp4', help='dataset path')
    parser.add_argument('--segmentation_yaml', help='the segmentation yaml in order to load the model and weights')
    parser.add_argument('--detection_weights', default='', help='resume from checkpoint')
    parser.add_argument('--save_video', dest='save_video', action='store_true', help='in order to save video',
                        required=False)
    parser.add_argument('--score-thres', default='0.6', help='score threshold for ignoring bboxes')
    parser.add_argument('--crop-size', default=360, type=int)
    parser.add_argument('--cropped', dest='cropped', action='store_true', help='crop',
                        required=False)
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    for i in range(10):
        main(args)
