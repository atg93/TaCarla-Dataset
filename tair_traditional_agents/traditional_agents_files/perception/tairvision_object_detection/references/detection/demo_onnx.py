import time

import torch
import torch.utils.data

import cv2
import numpy as np

import tairvision.models.detection as detection
from torchvision.transforms import ToPILImage

import tairvision.references.detection.presets as presets
from tairvision.utils import draw_bounding_boxes
from tairvision.references.detection.bdd_utils import get_label as get_label_bdd
from tairvision.references.detection.coco_utils import get_label as get_label
from tairvision.references.detection.config import get_args_parser

from PIL import Image

import onnxruntime as ort


def main(args):

    device = torch.device(args.device)

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

    model = detection.__dict__[args.model](num_classes=91,
                                           num_keypoints=0,
                                           **kwargs)
    model.to(device)
    model.eval()

    transform = presets.DetectionPresetEval(None)

    cap = cv2.VideoCapture(args.video_path)

    ort_sess = ort.InferenceSession(args.resume, providers=['CUDAExecutionProvider'])

    def to_numpy(input_tensor: torch.Tensor) -> np.ndarray:
        """
        Converts torch tensor to numpy array by proper operations
        :param input_tensor:
        :return:
        """
        input_numpy = input_tensor.detach().cpu().numpy()
        return input_numpy

    print('Press "Esc", "q" or "Q" to exit.')
    frame_count = 0
    ret_val = True
    print_freq = 1
    first_time = True
    print_count = 0
    time_preprocess, time_model, time_postprocess, time_cv2 = [], [], [], []
    while ret_val:
        ret_val, image = cap.read()
        frame_count += 1
        t0 = time.time()
        if frame_count % print_freq == 0:
            image = Image.fromarray(image)
            image, _ = transform(image, None)
            image = image.unsqueeze(0).to(device)

            if first_time:
                # get the original image sizes
                original_image_sizes = []
                for img in image:
                    val = img.shape[-2:]
                    assert len(val) == 2
                    original_image_sizes.append((val[0], val[1]))

            image_transformed, target = model.transform(image, None)

            if first_time:
                features = model.backbone(image_transformed.tensors)
                features = list(features.values())
                locations, strides, _ = model.compute_locations(features)
                locations = locations
                strides = strides
                first_time = False
            ort_inputs = dict(
                (ort_sess.get_inputs()[i].name,
                 to_numpy(input_flatten)
                 )
                for i, input_flatten in enumerate([image_transformed.tensors])
            )
            t1 = time.time()
            ort_outs = ort_sess.run(None, ort_inputs)
            ort_outs = {'cls_logits': torch.tensor(ort_outs[0]).to(device),
                        'centerness': torch.tensor(ort_outs[1]).to(device),
                        'bbox_regression': torch.tensor(ort_outs[2]).to(device),
            }
            t2 = time.time()
            output = model.predict_proposals(ort_outs, locations, strides, top_feats=None)
            output = model.transform.postprocess(output, image_transformed.image_sizes, original_image_sizes)
            t3 = time.time()

            boxes = output[0]['boxes'].cpu()
            scores = output[0]['scores'].cpu()
            labels = output[0]['labels'].cpu().numpy()
            
            boxes = boxes[scores > float(args.score_thres)]
            labels = labels[scores > float(args.score_thres)]
            if args.use_bdd_labels:
                labels = [*map(get_label_bdd, [*labels])]
            else:
                labels = [*map(get_label, [*labels])]

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break

            images_to_show = (image.cpu().clone()*255).to(dtype=torch.uint8).squeeze(0)

            output_to_show = draw_bounding_boxes(images_to_show, boxes, labels, font_size=12)
            output_to_show = np.array(ToPILImage()(output_to_show))
            
            cv2.imshow("Onnx Model", output_to_show)

            t4 = time.time()

            time_preprocess.append(t1 - t0)
            time_model.append(t2 - t1)
            time_postprocess.append(t3 - t2)
            time_cv2.append(t4 - t3)

            if print_count == 100:
                print("Preprocess: " + str(np.asarray(time_preprocess).mean()))
                print("Model: " + str(np.asarray(time_model).mean()))
                print("Postprocess: " + str(np.asarray(time_postprocess).mean()))
                print("CV2: " + str(np.asarray(time_cv2).mean()))
                print_count = 0
                time_preprocess, time_model, time_postprocess, time_cv2 = [], [], [], []
            else:
                print_count += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
