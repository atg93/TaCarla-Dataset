import datetime
import os
import pickle
import time

import torch
import torch.utils.data

import cv2
import numpy as np

import tairvision
from torchvision.transforms import ToPILImage

import tairvision.references.detection.presets as presets
from tairvision.models.tracking.tair_track import TAIRTracker
from tairvision.references.similarity.model import EmbeddingNet
from tairvision.utils import draw_bounding_boxes

import torchvision.transforms as T

from tairvision.references.detection.widerface_utils import warp_and_crop_face, get_reference_facial_points, get_label
from tairvision.references.detection.config import get_arguments

from PIL import Image
from tairvision.models.analysis.dan import DANWithAgeGender as DAN
from torch.nn import functional as F


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

    track = TAIRTracker(args)
    track.create_tracker()

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    transform = presets.DetectionPresetEval(None)

    # affectnet labels
    mood_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
    # raf-db labels
    # mood_labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']

    model_sim = EmbeddingNet()
    model_sim.load_state_dict(torch.load(args.sim_model_dir))
    model_sim.eval()
    model_sim.to(device)

    transform_sim = T.Compose([T.Lambda(lambda image: image.convert('RGB')),
                        T.Resize((224, 224)),
                        T.ToTensor(),
                        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        ])

    if args.mood_model is not None:
        model_mood = DAN(num_head=4, use_gender=True)
        model_mood.load_state_dict(torch.load(args.mood_model, map_location=device)['model'], strict=True)
        model_mood.eval()
        model_mood.to(device)

        transform_mood = T.Compose([T.Lambda(lambda image: image.convert('RGB')),
                                    T.Resize((224, 224)),
                                    T.ToTensor(),
                                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ])

    with open(args.face_db_dir, 'rb') as handle:
        face_db = pickle.load(handle)

    emb_base = []
    id_base = []
    for k, v in enumerate(face_db.values()):
        emb_base.append(v)
        id_base.append(np.ones(v.shape[0])*k)

    if len(emb_base) > 0:
        emb_base = np.concatenate(emb_base, axis=0)
        id_base = np.concatenate(id_base, axis=0)

    emb_list = []

    if args.video_path is not None:
        camera = cv2.VideoCapture(args.video_path)
    else:
        camera = cv2.VideoCapture(args.camera_id)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print('Press "Esc", "q" or "Q" to exit.')
    prev_label = 'neutral'
    label_count = 0
    if args.record:
        out = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (1280, 720))
    image_list = []
    while True:
        ret_val, image = camera.read()
        image1 = image
        if args.video_path is None:
            image = image[:, ::-1, :]
        h, w = image.shape[0:2]
        if args.video_crop_height is not None:
            h_out = args.video_crop_height
            w_out = (args.video_crop_height // 9) * 16
            h_start = (h - h_out) // 2
            h_end = h_start + h_out
            w_start = (w - w_out) // 2
            w_end = w_start + w_out
            image = image[h_start:h_end, w_start:w_end, :]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image, _ = transform(image, None)
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)

        boxes = output[0]['boxes'].cpu()
        scores = output[0]['scores'].cpu()
        keypoints = output[0]['keypoints'].cpu()
        labels = np.ones((len(boxes),))

        out_image = []
        out_im = dict()
        out_im['boxes'] = output[0]['boxes'].cpu()
        out_im['scores'] = output[0]['scores'].cpu()
        out_im['labels'] = torch.tensor(labels)
        out_image.append(out_im)

        # outputs_track, scores_track = track.update(out_image, image, image1, keypoints)
        # boxes, ids, keypoints, scores = track.postprocess_face(outputs_track, scores)

        boxes = torch.Tensor(boxes)

        #scores = torch.Tensor(scores)
        #min_len = min(len(boxes[0]), len(keypoints))
        #scores = scores[0:min_len]
        #boxes = boxes[scores[0] > float(args.score_thres),]
        #keypoints = keypoints[scores[0] > float(args.score_thres),]



        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        images_to_show = (image.cpu().clone() * 255).to(dtype=torch.uint8).squeeze(0)
        output_to_show = draw_bounding_boxes(images_to_show, boxes, keypoints=keypoints)
        output_to_show = np.array(ToPILImage()(output_to_show))

        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        output_size = (224,224)
        reference_5pts = get_reference_facial_points(output_size, inner_padding_factor, outer_padding, default_square)

        labels = None

        cv2.imshow("FaceDetection", output_to_show)

        if len(boxes) > 0:

            # labels = face_analysis_multi(images_to_show, boxes, keypoints,
            #                              model_mood, transform_mood, mood_labels,
            #                              max_faces=None,
            #                              device=device,
            #                              ids=ids)

            images_to_show2 = images_to_show.permute(1, 2, 0).numpy()
            images_to_show2 = warp_and_crop_face(images_to_show2, keypoints[0], reference_pts=reference_5pts, crop_size=output_size)
            cv2.imshow("FaceDetection2", cv2.resize(images_to_show2, dsize=(224,224)))

            image_to_model_sim = Image.fromarray(images_to_show2)
            image_to_model_sim = transform_sim(image_to_model_sim)
            with torch.no_grad():
                similarity_embedding = model_sim(image_to_model_sim.unsqueeze(0).to(device))

            emb_list.append(similarity_embedding[0].cpu().numpy())

            if args.register:
                print('Recording ' + str(len(emb_list)) + '/1000')
                if len(emb_list) > 1000:
                    #import pdb; pdb.set_trace()
                    emb_list = np.stack(emb_list)
                    new_id = 'id' + str(int(list(face_db.keys())[-1].rstrip('d')[-1])+1) if len(emb_base) > 0 else 'id0'
                    face_db[new_id] = emb_list
                    with open(args.face_db_dir, 'wb') as handle:
                        pickle.dump(face_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    break

#            for i in range(len(labels), len(keypoints)):
#                labels.append('.')
        #images_to_show2 = torch.tensor(images_to_show2)
        output_to_show = draw_bounding_boxes(images_to_show2,
                                             boxes,
                                             keypoints=keypoints,
                                             labels=labels,
                                             fill=True,
                                             font='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                                             font_size=16)
#        output_to_show = np.array(ToPILImage()(output_to_show))
#        output_to_show = cv2.cvtColor(output_to_show, cv2.COLOR_RGB2BGR)

#        cv2.imshow("FaceDetection", output_to_show)
#        image_list.append(output_to_show)

    if args.record:
        for image in image_list:
            out.write(image)

    cv2.destroyAllWindows()
    if args.record:
        out.release()



def face_analysis_multi(images_to_show, boxes, keypoints, model_mood, transform_mood,
                        mood_labels=['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt'],
                        max_faces=3,
                        device='cuda',
                        ids=None):
    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (96, 112)
    # get the reference 5 landmarks position in the crop settings
    reference_5pts = None  # get_reference_facial_points(output_size, inner_padding_factor, outer_padding, default_square)

    images_to_show = images_to_show.permute(1, 2, 0).numpy()

    labels = []
    if max_faces is None:
        max_faces = len(keypoints)
    for i in range(min(max_faces, len(keypoints))):
        images_to_show2 = warp_and_crop_face(images_to_show, keypoints[i], reference_pts=reference_5pts,
                                             crop_size=output_size, align_type='cv2_affine')
        id = ids[i]
        image_to_model_mood = Image.fromarray(images_to_show2)
        image_to_model_mood = transform_mood(image_to_model_mood)
        with torch.no_grad():
            out, _, _ = model_mood(image_to_model_mood.unsqueeze(0).to(device))

        pred_exp, pred_gender, pred_age_dist, pred_age = out
        pred_exp_ratio = F.softmax(pred_exp, 1)
        pred_exp_ratio_sorted = np.sort(pred_exp_ratio.cpu())[0]
        pred_exp_idx_sorted = np.argsort(pred_exp_ratio.cpu())[0]
        _, pred_gender = torch.max(pred_gender, 1)
        index_1 = pred_exp_idx_sorted[-1]
        index_2 = pred_exp_idx_sorted[-2]
        if pred_exp_ratio_sorted[-1] > 0.40:
            label = mood_labels[index_1]

            if pred_exp_ratio_sorted[-2] > 0.30:
                label = label + ' - ' + mood_labels[index_2]

        else:
            label = '---------'
        gender = 'Male' if pred_gender == 0 else 'Female'

        if pred_age < 18:
            pred_age_print = '18-'
        elif pred_age < 24:
            pred_age_print = '18-30'
        elif pred_age < 36:
            pred_age_print = '30-42'
        elif pred_age < 48:
            pred_age_print = '42-54'
        else:
            pred_age_print = '54+'

        # labels.append(pred_exp_ratio + '%' + label + ' ' + gender + ' Age ' + pred_age_print)
        labels.append('Expression: ' + label +
                      '\nGender: ' + gender +
                      '\nAge: ' + pred_age_print +
                      '\nID: ' + str(id))

    return labels
def get_track_args(args):
    # tracker related args
    args.result_root = None
    args.frame_dir = None
    args.result_filename = None
    args.detector_img_size = [1280, 720]
    args.tracker_model = 'bytetrack'
    args.use_byte = True
    args.score_thres = 0.6
    args.classes = [1]
    args.detector_mode = 'face'
    args.iou_thresh = 0.5
    args.show_image = 0.5
    args.output_format = 'video'
    args.out_root = '/workspace/ik22/face_demo'
    args.track_only = True
    args.mot20 = True
    args.match_thresh = 0.9
    args.track_buffer = 30

    return args

if __name__ == "__main__":
    args = get_arguments()
    args = get_track_args(args)

    print(args)

    main(args)