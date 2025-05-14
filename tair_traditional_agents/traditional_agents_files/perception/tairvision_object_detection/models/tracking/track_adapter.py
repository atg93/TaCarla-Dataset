from tairvision_object_detection.models.tracking.utils import filter_boxes_2D
from tairvision_object_detection.models.tracking.bytetrack.byte_tracker import BYTETracker
import torch
import numpy as np
from typing import List


class TAIRTrackAdapter(object):
    def __init__(self, args, match_thresh=0.9, iou_thresh=0.5, score_thres=0.5, classes: List[int]=None):
        args.mot20 = False
        args.match_thresh = match_thresh
        args.track_buffer = 30
        args.iou_thresh = iou_thresh
        args.classes = classes
        args.score_thres = score_thres
        self.tracker = 'bytetrack'
        self.score_thres = float(score_thres)
        self.iou_threshold = iou_thresh
        self.track_model = BYTETracker(args)
        self.classes = args.classes
        self.track_only = False

    def update(self, output, img_size, im0_size):
        # img_size: HxW of tranformed image (input of detection/segmentation model)
        # im0_size: HxW of original image
        out_im = dict()
        out_keys = output.keys()
        assert 'boxes' in out_keys, 'boxes should be in detection output'
        assert 'scores' in out_keys, 'scores should be in detection output'
        assert 'labels' in out_keys, 'labels should be in detection output'
        for k in out_keys:
            out_im[k] = output[k].cpu()

        if 'masks' not in out_keys:
            out_im['masks'] = None
        if 'keypoints' not in out_keys:
            out_im['keypoints'] = None

        out_im = filter_boxes_2D(out_im, self.classes)

        outputs = self.track_model.update(out_im, im0_size, img_size)
        return outputs

    def postprocess(self, outputs, scores):
        online_tlwhs, online_ids, cls = [], [], []
        for j, (output, conf) in enumerate(zip(outputs, scores)):
            # ByteTrack gives tlwh
            tlwhs = output.tlwh
            tid = output.track_id
            online_tlwhs.append(tlwhs)
            online_ids.append(tid)

        return online_tlwhs, online_ids

    def postprocess_xyxy(self, outputs, other_keys=[]): # conf = score -- output.score
        online_xyxy, online_ids, online_scores, online_labels = [], [], [], []
        online_others = dict()
        for k in other_keys:
            online_others[k] = []

        for j, (output) in enumerate(outputs):
            tlwhs = output.tlwh
            tid = output.track_id
            xyxy = np.array([tlwhs[0], tlwhs[1], tlwhs[0]+tlwhs[2], tlwhs[1]+tlwhs[3]])
            labels = output.label

            for k in other_keys:
                online_others[k].append(output.others[k])

            online_xyxy.append(xyxy)
            online_ids.append(tid)
            online_scores.append(output.score)
            online_labels.append(labels)

        online_xyxy = torch.Tensor(online_xyxy)
        online_ids = torch.Tensor(online_ids)
        online_scores = torch.Tensor(online_scores)
        online_labels = np.array(online_labels)
        for k in other_keys:
            if len(outputs) > 0:
                online_others[k] = torch.stack(online_others[k])
            else:
                online_others[k] = torch.Tensor([])
        return online_xyxy, online_ids, online_scores, online_labels, online_others

