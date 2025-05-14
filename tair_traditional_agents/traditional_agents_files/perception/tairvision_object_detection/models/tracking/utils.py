import torch
import numpy as np
import cv2
import os
import yaml
import tairvision


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)




def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    if boxes.shape[0] == 0:
        return boxes
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def xyxy2xywh(x):
    if x.shape[0] == 0:
        return x
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2tlwh(bbox_xywh):
    if isinstance(bbox_xywh, np.ndarray):
        bbox_tlwh = bbox_xywh.copy()
    elif isinstance(bbox_xywh, torch.Tensor):
        bbox_tlwh = bbox_xywh.clone()
    bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
    bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
    return bbox_tlwh


def xyxy2tlwh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    t = x1
    l = y1
    w = int(x2 - x1)
    h = int(y2 - y1)
    return t, l, w, h


def tlwh2xyxy(box):
    tmp = box
    tmp[:, 2] = box[:, 0] + box[:, 2]
    tmp[:, 3] = box[:, 1] + box[:, 3]
    return tmp


def filter_boxes_ocsort(output, cls=None):
    boxes = output[0]['boxes'].cpu()
    scores = output[0]['scores'].cpu()
    labels = output[0]['labels'].cpu().numpy()

    if cls is not None:
        cls_mask = [i for i in range(labels.shape[0]) if labels[i] in cls]
        boxes = boxes[cls_mask]
        labels = labels[cls_mask]
        scores = scores[cls_mask]

    return boxes, labels, scores


def filter_boxes_2D(output, cls=None):
    out_im = dict()
    out_keys = output.keys()
    labels = output['labels'].cpu().numpy()
    for k in out_keys:
        if output[k] is not None:
            out_im[k] = output[k].cpu()

    if cls is not None:
        cls_mask = [i for i in range(labels.shape[0]) if labels[i] in cls]
        for k in out_keys:
            if output[k] is not None:
                out_im[k] = out_im[k][cls_mask]

    return out_im

def sort_trackIds(online_tlwhs, online_ids, cls, tracker):
    sorted_idx = sorted(range(len(online_ids)), key=lambda k: online_ids[k])
    online_tlwhs = [online_tlwhs[i] for i in sorted_idx]
    online_ids = [online_ids[i] for i in sorted_idx]

    return online_tlwhs, online_ids, cls


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


# RELATED TO PLOTS
def plot_tracking(image, tlwhs, obj_ids, cls_names, scores=None, frame_id=0, fps=0., ids2=None):
    #x1, y1, w, h = 110, 60, 250, 883
    #constant_box = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        if cls_names is not None:
            cl_name = cls_names[i]
            id_text = '{}:{}'.format(cl_name, int(obj_id))
        else:
            id_text = str(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
        # CONSTANT BOX PART
        #cv2.rectangle(im, constant_box[0:2], constant_box[2:4], color=color, thickness=line_thickness*3)
        #cv2.putText(im, id_text, (constant_box[0], constant_box[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
        #            thickness=text_thickness)
    return im


def plot_tracking2(image, tlwhs, obj_ids, tlwhs_g, obj_ids_g, cls_names, scores=None, frame_id=0, fps=0., ids2=None):
    #x1, y1, w, h = 110, 60, 250, 883
    #constant_box = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        if cls_names is not None:
            cl_name = cls_names[i]
            id_text = '{}:{}'.format(cl_name, int(obj_id))
        else:
            id_text = str(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)

        # GT part
        for i, tlwh in enumerate(tlwhs_g):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = 1 #int(obj_ids_g[i])
            if cls_names is not None:
                cl_name = cls_names[i]
                id_text = '{}:{}'.format(cl_name, int(obj_id))
            else:
                id_text = str(int(obj_id))
            if ids2 is not None:
                id_text = id_text + ', {}'.format(int(ids2[i]))
            _line_thickness = 1 if obj_id <= 0 else line_thickness
            color = get_color(abs(obj_id))
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=(0, 0, 0), thickness=line_thickness)
            #cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
            #            thickness=text_thickness)

    return im


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=int(track_id), x1=round(x1, 1), y1=round(y1, 1),
                                          x2=round(x2, 1), y2=round(y2, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=int(track_id), x1=x1, y1=y1, w=w, h=h)
                f.write(line)
    print(filename)


def write_results_new(filename, results):
    #save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    res = []
    for frame_id, tlwhs, track_ids in results:
        for tlwh, track_id in zip(tlwhs, track_ids):
            if track_id < 0:
                continue
            x1, y1, w, h = tlwh
            line = [frame_id, track_id, x1, y1, w, h,-1,-1,-1,-1]
            res.append(line)
    np.savetxt(filename, res, delimiter=",", fmt='%i')
    print("NEW", filename)


def get_detector(opt):
    print("Creating model")
    kwargs = {
        "type": opt.backbone,
        "trainable_backbone_layers": opt.trainable_backbone_layers,
        "pretrained": opt.pretrained,
        "pyramid_type": opt.pyramid_type,
        "repeats": opt.bifpn_repeats,
        "fusion_type": opt.fusion_type,
        "depthwise": opt.use_depthwise,
        "use_P2": opt.use_P2,
        "no_extra_blocks": opt.no_extra_blocks,
        "extra_before": opt.extra_before,
        "context_module": opt.context_module,
        "loss_weights": opt.loss_weights,
        "nms_thresh": opt.nms_thresh,
        "post_nms_topk": opt.post_nms_topk,
        "min_size": opt.transform_min_size,
        "max_size": opt.transform_max_size
    }

    if "fcos" in opt.model:
        kwargs["fpn_strides"] = opt.fpn_strides
        kwargs["sois"] = opt.sois
        kwargs["thresh_with_ctr"] = opt.thresh_with_ctr
        kwargs["use_deformable"] = opt.use_deformable
    else:
        kwargs["anchor_sizes"] = opt.anchor_sizes
        if "rcnn" in opt.model:
            if opt.rpn_score_thresh is not None:
                kwargs["rpn_score_thresh"] = opt.rpn_score_thresh

    num_classes = 11
    num_keypoints = 0
    if opt.detector_mode == 'coco':
        num_classes = 91
        num_keypoints = 0
    elif opt.detector_mode == 'face':
        num_classes = 2
        num_keypoints = 5

    model = tairvision_object_detection.models.detection.__dict__[opt.model](num_classes=num_classes,
                                                            num_keypoints=num_keypoints,
                                                            **kwargs)
    return model
