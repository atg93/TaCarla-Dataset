import torch
import torch.utils.data
import cv2
import tairvision
from torchvision.transforms import ToPILImage
import tairvision.references.detection.presets as presets
from tairvision.utils import draw_bounding_boxes
import torchvision.transforms as T
from tairvision.references.detection.widerface_utils import warp_and_crop_face, get_reference_facial_points, get_label
from tairvision.references.detection.config import get_arguments
from PIL import Image
from tairvision.models.analysis.dan import DANWithMoodAgeGender as DAN
from tairvision.references.analysis.demo_face import face_analysis_multi
import numpy as np


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

    model = tairvision.models.detection.__dict__[args.model](num_classes=2,
                                                             num_keypoints=5,
                                                             **kwargs)

    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    transform = presets.DetectionPresetEval(None)

    # affectnet labels
    mood_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

    if args.mood_model is not None:
        model_mood = DAN(num_head=4, num_class=7, use_gender=True)
        model_mood.load_state_dict(torch.load(args.mood_model, map_location=device)['model'], strict=True)
        model_mood.eval()
        model_mood.to(device)

        transform_mood = T.Compose([T.Lambda(lambda image: image.convert('RGB')),
                                    T.Resize((224, 224)),
                                    T.ToTensor(),
                                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ])

    while True:

        image = Image.open('/home/ig21/git/HIC/frames/013/media_dashboard_camera/visible_spectrum.png')
        image, _ = transform(image, None)
        image = image[:3,:,:]

        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)

        boxes = output[0]['boxes'].cpu()
        scores = output[0]['scores'].cpu()
        keypoints = output[0]['keypoints'].cpu()

        min_len = min(len(boxes), len(keypoints))
        scores = scores[0:min_len]
        boxes = boxes[scores > float(args.score_thres),]
        keypoints = keypoints[scores > float(args.score_thres),]
        keypoints = keypoints.numpy()

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        images_to_show = (image.cpu().clone() * 255).to(dtype=torch.uint8).squeeze(0)

        labels = None
        if len(boxes) > 0 and args.mood_model is not None:

            labels = face_analysis_multi(images_to_show, boxes, keypoints,
                                         model_mood, transform_mood, mood_labels,
                                         max_faces=None,
                                         device=device)

            for i in range(len(labels), len(keypoints)):
                labels.append('.')

        output_to_show = draw_bounding_boxes(images_to_show,
                                             boxes,
                                             keypoints=keypoints,
                                             labels=labels,
                                             fill=True,
                                             font='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                                             font_size=16)
        output_to_show = np.array(ToPILImage()(output_to_show))
        output_to_show = cv2.cvtColor(output_to_show, cv2.COLOR_RGB2BGR)

        cv2.imshow("FaceDetection", output_to_show)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main(args)
