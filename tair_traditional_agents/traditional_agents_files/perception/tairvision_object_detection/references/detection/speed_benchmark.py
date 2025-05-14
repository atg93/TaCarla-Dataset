import torch
import torch.utils.data

import cv2
import time
import numpy as np

import tairvision
from torchvision.transforms import ToPILImage

import tairvision.references.detection.presets as presets
from tairvision.references.detection.utils import get_dataset

from PIL import Image
from tairvision.references.detection.config import get_arguments



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

    if args.model_kwargs:
        kwargs.update(args.model_kwargs)

    model = tairvision.models.detection.__dict__[args.model](num_classes=num_classes,
                                                             num_keypoints=num_keypoints,
                                                             **kwargs)
    model.to(device)

    #amp.initialize(model, opt_level="O3", keep_batchnorm_fp32=True)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # model.transform.min_size = (args.transform_min_size,)
    model.transform.min_size = (480,)

    transform = presets.DetectionPresetEval(None)

    cap = cv2.VideoCapture(args.video_path)

    model_times = []
    cv2.setNumThreads(1)
    ret_val = True
    counter = 0
    while ret_val and counter < 300:
        ret_val, image = cap.read()
        t0 = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image, _ = transform(image, None)
        image = image.unsqueeze(0).to(device)
        t1 = time.time()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            output = model(image)
        end.record()

        torch.cuda.synchronize()
        model_time = start.elapsed_time(end)
        print(model_time)
        model_times.append(model_time)
        counter += 1

    print(np.array(model_times).mean())

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main(args)
