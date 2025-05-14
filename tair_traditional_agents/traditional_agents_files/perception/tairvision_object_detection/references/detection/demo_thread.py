from datetime import datetime
import os
import time

import torch
import torch.utils.data
from tairvision.references.detection.utils import CARLA

import cv2
import numpy as np

import tairvision
from torchvision.transforms import ToPILImage

import tairvision.references.detection.presets as presets
from tairvision.utils import draw_bounding_boxes

from tairvision.references.detection.coco_utils import get_label as get_label_coco
from tairvision.references.detection.coco_utils import get_label_color as get_label_color_coco
from tairvision.references.detection.bdd_utils import get_label as get_label_bdd
from tairvision.references.detection.bdd_utils import get_label_color as get_label_color_bdd
from tairvision.references.detection.utils import get_dataset
from tairvision.references.detection.config import get_arguments
from tairvision.references.detection.visualization_utils import vis_det_bboxes

from PIL import Image

import threading
from  queue import Queue


from apex import amp

class FrameReader(threading.Thread):
    def __init__(self, q, stop_event, video_path, transform, device):
        super(FrameReader, self).__init__()

        self.q = q
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        self.transform = transform
        self.device = device

        self.stop = stop_event
        self.frame_count = 0
        self.print_freq = 2

    def run(self):
        cv2.setNumThreads(1)
        ret_val = True
        while ret_val and not self.stop.is_set():
            ret_val, image = self.cap.read()
            self.frame_count += 1
            if self.frame_count % self.print_freq == 0:
                t0 = time.time()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image, _ = self.transform(image, None)
                image = image.unsqueeze(0).to(self.device)

                self.q.put(image)
                time.sleep(0.1)

        # self.q.put('finish')
        # self.stop.set()
        self.cap.release()
        print('Read thread has been finished')
        self.stop.clear()
        self.stop.set()

class OutputVisualizer(threading.Thread):
    def __init__(self, q, vis_q, stop_event):
        super(OutputVisualizer, self).__init__()
        self.q = q
        self.vis_q = vis_q
        self.stop_event = stop_event

    def run(self):
        cv2.setNumThreads(1)
        # output_video = None
        first = True
        while True and not self.stop_event.is_set():
            if not self.q.empty():
                entry = self.q.get()
                # if entry == 'finish':
                #     self.vis_q.put(entry)
                #     break

                if first:
                    time.sleep(3)
                    first = False

                output = entry['output']
                image = entry['image']

                boxes = output[0]['boxes'].cpu()
                scores = output[0]['scores'].cpu()
                labels = output[0]['labels'].cpu().numpy()

                masks = output[0]['masks'][:, 0, :, :].cpu() if 'masks' in output[0].keys() else None
                if masks is not None:
                    masks = masks[scores > float(args.score_thres)]

                masks = (masks > 0.1).float()

                boxes = boxes[scores > float(args.score_thres)]
                labels = labels[scores > float(args.score_thres)]
                if args.dataset == "bdd":
                    labels_str = [*map(get_label_bdd, [*labels])]
                    label_colors = [*map(get_label_color_bdd, [*labels])]
                elif args.dataset == "coco":
                    labels_str = [*map(get_label_coco, [*labels])]
                    label_colors = [*map(get_label_color_coco, [*labels])]
                else:
                    labels_str = [*map(get_label_coco, [*labels])]
                    label_colors = [*map(get_label_color_coco, [*labels])]

                images_to_show = (image.cpu().clone()).squeeze(0)

                images_to_show = cv2.cvtColor(images_to_show.cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                output_to_show = vis_det_bboxes(images_to_show, boxes, masks, color=label_colors)
                output_to_show = output_to_show * 255
                output_to_show = np.array(ToPILImage()(output_to_show.astype('uint8')))

                self.vis_q.put(output_to_show)
            else:
                time.sleep(0.1)

        print('Output Visualizer has been finished')
        self.stop_event.clear()
        self.stop_event.set()

class OutputDisplayer(threading.Thread):
    def __init__(self, vis_q, stop_event, record=None):
        super(OutputDisplayer, self).__init__()
        self.vis_q = vis_q
        self.stop_event = stop_event
        self.record = record

    def run(self):
        output_video = None
        while True and not self.stop_event.is_set():
            if not self.vis_q.empty():
                output_to_show = self.vis_q.get()

                # if output_to_show == 'finish':
                #     break

                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord('q') or ch == ord('Q'):
                    self.stop_event.set()
                    break

                if self.record is not None:
                    if output_video is None:
                        filepath = os.path.join(self.record, 'out-{}.avi'.format(datetime.now().strftime("%H-%M_%d-%m-%Y")))
                        output_video = cv2.VideoWriter(filepath,
                                                       cv2.VideoWriter_fourcc(*'MJPG'),
                                                       15, [output_to_show.shape[1], output_to_show.shape[0]])

                    output_video.write(output_to_show)

                cv2.imshow("Seperate", output_to_show)
            else:
                time.sleep(0.01)

        if output_video is not None:
            output_video.release()
        cv2.destroyAllWindows()


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

    model.transform.min_size = (args.transform_min_size,)

    transform = presets.DetectionPresetEval(None)

    stop_event = threading.Event()
    q = Queue(25)
    if args.video_path == 'carla':
        read_thread = CARLA(q, transform=transform, device=device)
        read_thread.start()
    else:
        # read_thread = threading.Thread(target=read_frames, args=[q, args.video_path, transform, device])
        read_thread = FrameReader(q, stop_event, args.video_path, transform, device)
        read_thread.start()


    out_q = Queue(100)
    vis_q = Queue(100)

    vis_thread = OutputVisualizer(out_q, vis_q, stop_event)
    vis_thread.start()

    show_thread = OutputDisplayer(vis_q, stop_event, args.record)
    show_thread.start()

    # print('{}  {}'.format(q.qsize(), out_q.qsize()))

    print('Press "Esc", "q" or "Q" to exit.')
    print_count = 0
    time_preprocess, time_model, time_cv2 = [], [], []
    while True and not stop_event.is_set():
        if not q.empty():
            image = q.get()

            if image == 'finish':
                out_q.put('finish')
                break

            t1 = time.time()
            with torch.no_grad():
                output = model(image)
            t2 = time.time()

            out_q.put(dict(image=image, output=output))
            print('-'*20)
            print('Input queue: {}'.format(q.qsize()))
            print('Output queue: {}'.format(out_q.qsize()))
            print('Show queue: {}'.format(vis_q.qsize()))
            # time_preprocess.append(t1 - t0)
            time_model.append(t2 - t1)
            # time_cv2.append(t3 - t2)
            #
            if print_count == 100:
                # print("Preprocess: " + str(np.asarray(time_preprocess).mean()))
                print("Model: " + str(np.asarray(time_model).mean()))

                # print("CV2: " + str(np.asarray(time_cv2).mean()))
                print_count = 0
                time_preprocess, time_model, time_cv2 = [], [], []
            else:
                print_count += 1
    print('main thread has been stopped while loop')
    stop_event.clear()
    stop_event.set()

    read_thread.join()
    print('Read thread has been joined')
    vis_thread.join()
    print('Visualization thread has been joined')
    show_thread.join()
    print('Display thread has been joined')

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main(args)
