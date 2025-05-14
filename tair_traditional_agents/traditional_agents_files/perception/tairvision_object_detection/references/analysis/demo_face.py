import torch
import torch.utils.data
import cv2
import numpy as np
import tairvision
from torchvision.transforms import ToPILImage
import tairvision.references.detection.presets as presets
from tairvision.utils import draw_bounding_boxes
import torchvision.transforms as T
from tairvision.references.detection.widerface_utils import warp_and_crop_face, get_reference_facial_points, get_label
from tairvision.references.analysis.config import get_arguments_analysis
from PIL import Image
from tairvision.models.analysis.dan import DANWithMoodAgeGender as DAN
from torch.nn import functional as F
from tairvision.models.tracking.track_adapter import TAIRTrackAdapter
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tairvision.references.detection.transforms import UnNormalize



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
    num_classes = 2
    model.to(device)

#    if args.use_track:
    track = TAIRTrackAdapter(args, match_thresh=0.9, iou_thresh=0.6, score_thres=0.6,
                                 classes=[1, 2])

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    transform = presets.DetectionPresetEval(None)

    # affectnet labels
    mood_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

    if args.mood_model is not None:
        model_mood = DAN(num_class=7, use_mood=True,
                         use_gender=False,
                         use_valence=True, use_dan=False,
                         extra_layer=False)
        model_mood.load_state_dict(torch.load(args.mood_model, map_location=device)['model'], strict=True)
        model_mood.eval()
        model_mood.to(device)

        transform_mood = T.Compose([T.Lambda(lambda image: image.convert('RGB')),
                                    T.Resize((224, 224)),
                                    T.ToTensor(),
                                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ])

    if args.gender_model is not None:
        model_gender = DAN(use_mood=False,
                           use_gender=True,
                           use_valence=False, use_dan=False,
                           extra_layer=False)
        model_gender.load_state_dict(torch.load(args.gender_model, map_location=device)['model'], strict=True)
        model_gender.eval()
        model_gender.to(device)

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
        out2 = cv2.VideoWriter(args.record_hmap, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (2000, 720))
    image_list = []
    hmap_list = []
    normalizer_dict = {}
    while True:
        ret_val, image = camera.read()
        im0_size = image.shape[:2]
        if image is None:
            break
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

        im_unnormal = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image, None)[0]

        with torch.no_grad():
            output = model(im_unnormal)

#        if args.use_track:
        outputs_track = track.update(output[0], image.shape[2:], im0_size)
        boxes, ids, scores, labels, others = track.postprocess_xyxy(outputs_track, other_keys=['keypoints'])
        keypoints = others['keypoints'].cpu()
        keypoints = keypoints.numpy()
        ids = ids.tolist()

        if len(ids) != 0:
            # forget
            keys_to_pop = []
            for key in normalizer_dict.keys():
                if key not in ids:
                    keys_to_pop.append(key)

            for key in keys_to_pop:
                normalizer_dict.pop(key)

            # new record
            for id in ids:
                if id not in normalizer_dict.keys():
                    normalizer_dict[id] = Normalizer()

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        images_to_show = (im_unnormal.cpu().clone() * 255).to(dtype=torch.uint8).squeeze(0)

        labels = None
        labels_list_of_dicts = []

        if len(boxes) > 0 and args.mood_model is not None:

            labels, labels_list_of_dicts = face_analysis_multi(images_to_show, boxes, keypoints,
                                                               model_mood, model_gender, transform_mood, mood_labels,
                                                               max_faces=None,
                                                               device=device,
                                                               ids=ids, normalizer=normalizer_dict)

            for i in range(len(labels), len(keypoints)):
                labels.append('.')

        output_to_show = draw_bounding_boxes(images_to_show,
                                             boxes,
                                             keypoints=keypoints,
                                             labels=labels,
                                             fill=True,
                                             #font='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                                             font_size=16)
        output_to_show = np.array(ToPILImage()(output_to_show))
        output_to_show = cv2.cvtColor(output_to_show, cv2.COLOR_RGB2BGR)

        cv2.imshow("FaceDetection", output_to_show)
        image_list.append(output_to_show)

        if args.heatmap:
            if len(labels_list_of_dicts) > 0:
                cv2_emotion_plot_img = plot_2d_emotion(labels_list_of_dicts, to_cv2=True, heatmap=True)
                if cv2_emotion_plot_img is not None:
                    cv2.imshow("plot", cv2_emotion_plot_img)
                    hmap_list.append(cv2_emotion_plot_img)

    if args.record:
        for image in image_list:
            out.write(image)
        #for hmap in hmap_list:
        #    out2.write(hmap)
        for i, image in enumerate(image_list):
            hmap = hmap_list[i]
            hmap_resize = cv2.resize(hmap, (720,720))
            image_cat = np.concatenate((image, hmap_resize), axis=1)
            out2.write(image_cat)

    cv2.destroyAllWindows()
    if args.record:
        out.release()
        out2.release()

def face_analysis_multi(images_to_show, boxes, keypoints, model_mood, model_gender, transform_mood,
                        mood_labels=['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt'],
                        max_faces=3,
                        device='cuda',
                        ids=None,
                        normalizer=None):
    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (96, 112)
    # get the reference 5 landmarks position in the crop settings
    reference_5pts = None  # get_reference_facial_points(output_size, inner_padding_factor, outer_padding, default_square)

    images_to_show = images_to_show.permute(1, 2, 0).numpy()

    labels = []
    labels_list_of_dicts = []
    if max_faces is None:
        max_faces = len(keypoints)
    for i in range(min(max_faces, len(keypoints))):
        images_to_show2 = warp_and_crop_face(images_to_show, keypoints[i], reference_pts=reference_5pts,
                                             crop_size=output_size, align_type='cv2_affine')
        id = ids[i]
        image_to_model_mood = Image.fromarray(images_to_show2)
        image_to_model_mood = transform_mood(image_to_model_mood)
        with torch.no_grad():
            out_mood, _, _ = model_mood(image_to_model_mood.unsqueeze(0).to(device))
            out_gender, _, _ = model_gender(image_to_model_mood.unsqueeze(0).to(device))

        pred_exp, _pred_gender, _pred_age_dist, _pred_age, pred_val, pred_aro = out_mood
        _pred_exp, pred_gender, pred_age_dist, pred_age, _pred_val, _pred_aro = out_gender

        age, gender, label = predict(pred_age, pred_gender, pred_exp, mood_labels)

        pred_valence = pred_val.cpu()
        pred_arousal = pred_aro.cpu()

        pred_val_aro = torch.cat([pred_valence, pred_arousal], dim=-1)
        normalizer[id].update_stats(pred_val_aro)

        means, stds = normalizer[id].get_stats()

        labels.append('Expression: ' + label +
                      '\nGender: ' + gender +
                      '\nAge: ' + age +
                      '\nID: ' + str(int(id)))

        labels_dict = {'exp': label,
                       'means': means,
                       'stds': stds,
                       'id': id
                       }
        labels_list_of_dicts.append(labels_dict)

    return labels, labels_list_of_dicts

def predict(pred_age, pred_gender, pred_exp, mood_labels):

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
    elif pred_age < 30:
        pred_age_print = '18-30'
    elif pred_age < 42:
        pred_age_print = '30-42'
    elif pred_age < 54:
        pred_age_print = '42-54'
    else:
        pred_age_print = '54+'
    return pred_age_print, gender, label

class Normalizer(object):
    def __init__(self):
        self.N = 0.0

    def update_stats(self, x):
        for foo in x:
            self.N += 1
            if self.N == 1:
                self.oldMean = foo
                self.Mean = foo
                self.Var = torch.zeros_like(foo)
            # elif self.N == 100:
            #    self.N = 1
            #    self.oldMean = foo
            #    self.Mean = foo
            #    self.Var = torch.zeros_like(foo)
            else:
                # self.Mean = self.oldMean + (foo - self.oldMean) / self.N
                self.Mean = self.oldMean + (foo - self.oldMean) / 10
                self.Var = self.Var + (foo - self.oldMean) * (foo - self.Mean)
                self.oldMean = self.Mean

    def get_stats(self):
        if self.N < 2:
            return torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])
        else:
            return self.Mean, np.sqrt(self.Var / (self.N - 1))


def plot_2d_emotion(lists_of_dict, to_cv2=False, heatmap=True):
    nb_plots = len(lists_of_dict)
    fig, ax = plt.subplots(figsize=(4 * nb_plots, 4), ncols=nb_plots, subplot_kw={'aspect': 'equal'})

    for i_dict, dict in enumerate(lists_of_dict):
        id = lists_of_dict[i_dict]['id']
        means = lists_of_dict[i_dict]['means'].cpu().numpy()
        stds = lists_of_dict[i_dict]['stds'].cpu().numpy()

        ax_i = ax[i_dict] if nb_plots > 1 else ax

        ax_i.clear()

        ax_i.set_xlim([-1, 1])
        ax_i.set_ylim([-1, 1])
        ax_i.set_xlabel('Valence')
        ax_i.set_ylabel('Arousal')
        ax_i.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_i.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax_i.set_title('id: ' + str(id))

        if heatmap:
            xi, yi, zi = get_heatmap(means, stds)
            ax_i.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='gist_heat')
        ax_i.scatter(means[0], means[1])

    img = None
    if to_cv2:
        fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def get_heatmap(means, stds):
    # create data
    xi, yi = np.mgrid[-1:1:0.1, -1:1:0.1]
    pos = np.dstack((xi, yi))

    means = means.reshape(-1)
    cov = np.eye(len(means)) * stds

    dist = multivariate_normal(means, cov)
    zi = dist.pdf(pos)

    return xi, yi, zi

if __name__ == "__main__":
    args = get_arguments_analysis()
    print(args)

    main(args)
