from tairvision_object_detection.models.tracking.utils import get_detector
from .utils import  filter_boxes_ocsort, plot_tracking, mkdir_if_missing
from tairvision_object_detection.references.tracking.video_dataset import LoadImages, LoadVideo, LoadWebcam
from tairvision_object_detection.models.tracking.ocsort.ocsort import OCSort
from tairvision_object_detection.models.tracking.bytetrack.byte_tracker import BYTETracker
import torch
import cv2
import numpy as np
from os import path as osp
import os


class TAIRTracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracker = args.tracker_model
        self.device = torch.device(args.device)
        self.score_thres = args.score_thres
        self.iou_threshold = args.iou_thresh
        self.use_byte = args.use_byte
        # default tracker model is ByteTrack
        self.track_model = BYTETracker(args)
        self.classes = args.classes
        self.show_image = args.show_image
        self.frame_dir = args.frame_dir
        self.output_format = args.output_format
        self.out_root = args.out_root
        mkdir_if_missing(self.out_root)
        self.detector_img_size = args.detector_img_size

        if not args.face_demo:
            # Get detector model and put it to device
            self.detector_model = get_detector(args)
            self.detector_model.to(self.device)
            checkpoint = torch.load(args.resume, map_location='cpu')
            # load and eval model for prediction
            self.detector_model.load_state_dict(checkpoint['model'])
            self.detector_model.eval()

        self.args = args
        # Track only eval parameters
        self.track_only = args.track_only


    def create_tracker(self):
        if self.tracker == 'ocsort':
            self.track_model = OCSort(det_thresh=self.score_thres,
                                      iou_threshold=self.iou_threshold,
                                      use_byte=self.use_byte)

        elif self.tracker == 'bytetrack':
            self.track_model = BYTETracker(self.args)

    def get_data_loader(self, path, img_input=True):
        if self.args.webcam:
            return LoadWebcam(self.args.camera_id, img_size=self.detector_img_size)
        elif img_input:
            return LoadImages(path=path, img_size=self.detector_img_size)
        else:
            return LoadVideo(path=path, img_size=self.detector_img_size)

    def inference(self, img, gt_file, frame_idx):
        if self.track_only:
            output_img = []
            gt = np.loadtxt(gt_file, delimiter=',').astype(int)
            gt = gt[gt[:, 7] == 1]
            inds = gt[:, 0] == frame_idx + 1
            frame_seqs = gt[inds, :]
            img_xyxy = []
            for i in range(frame_seqs.shape[0]):
                xyxy_ = [frame_seqs[i,2], frame_seqs[i,3],
                         frame_seqs[i,2]+frame_seqs[i,4],
                         frame_seqs[i,3]+frame_seqs[i,5]]
                img_xyxy.append(xyxy_)
            scores = np.ones((len(img_xyxy),))
            bbox = np.array(img_xyxy).astype('float64')
            labels = np.ones((len(img_xyxy),))

            im = dict()
            im['boxes'] = torch.tensor(bbox)
            im['scores'] = torch.tensor(scores)
            im['labels'] = torch.tensor(labels)
            output_img.append(im)

        else:
            img = torch.from_numpy(img)
            image = img.to(self.device).unsqueeze(0)
            # get the predictions from model (bbox, score, label)
            with torch.no_grad():
                output_img = self.detector_model(image)

        return output_img

    def update(self, output_img, img, im0s, keypoints=None):
        if self.tracker == 'ocsort':
            outputs, scores = self._update_ocsort(output_img, img, im0s)
        elif self.tracker == 'bytetrack':
            if keypoints is not None:
                return self._update_bytetrack(output_img, img, im0s, keypoints)
            else:
                outputs, scores = self._update_bytetrack(output_img, img, im0s)

        return outputs, scores

    def _update_ocsort(self, output_img, img, im0s):
        boxes, labels, scores = filter_boxes_ocsort(output_img, self.classes)  # boxes -> xyxy format
        im_shape = [im0s.shape[0], im0s.shape[1]] if self.track_only else img.shape[1:]
        outputs = self.track_model.update(boxes, scores, [im0s.shape[0], im0s.shape[1]], im_shape)
        return outputs, scores

    def _update_bytetrack(self, output_img, img, im0s, keypoints=None):
        boxes, labels, scores = filter_boxes_ocsort(output_img, self.classes)
        im_shape = [im0s.shape[0], im0s.shape[1]] if self.track_only else img.shape[1:]
        outputs = self.track_model.update_(boxes, scores, [im0s.shape[0], im0s.shape[1]], im_shape, keypoints)
        return outputs, scores

    def postprocess(self, outputs, scores):
        online_tlwhs, online_ids, cls = [], [], []
        for j, (output, conf) in enumerate(zip(outputs, scores)):
            # ocsort gives xyxy format, plot_tracking accepts (tlwh)
            if self.tracker == 'bytetrack':
                tlwhs = output.tlwh
                tid = output.track_id
            else:
                tlwhs = np.array([output[0], output[1], output[2] - output[0], output[3] - output[1]])
                tid = output[4]
            online_tlwhs.append(tlwhs)
            online_ids.append(tid)

        return online_tlwhs, online_ids

    def postprocess_face(self, outputs, scores):
        online_xyxy, online_ids, online_keypoints, online_scores = [], [], [], []
        for j, (output, conf) in enumerate(zip(outputs, scores)):
            # ocsort gives xyxy format, plot_tracking accepts (tlwh)
            tlwhs = output.tlwh
            tid = output.track_id
            kp = output.keypoints
            xyxy = np.array([tlwhs[0], tlwhs[1], tlwhs[0]+tlwhs[2], tlwhs[1]+tlwhs[3]])

            online_xyxy.append(xyxy)
            online_ids.append(tid)
            online_keypoints.append(kp)
            online_scores.append(conf)

        return online_xyxy, online_ids, online_keypoints, online_scores

    def write_results(self, filename, results, data_type):
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

    def write_results_no_score(self, filename, results):
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

    def write_results_new(self, filename, results):
        # save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
        res = []
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = [frame_id, track_id, x1, y1, w, h, -1, -1, -1, -1]
                res.append(line)
        np.savetxt(filename, res, delimiter=",", fmt='%i')

    def write_video(self, vid_name, frame_out):
        if self.output_format == 'video':
            video_out_path = osp.join(self.out_root, self.tracker, 'video_out', vid_name) # vid_name=seq
            mkdir_if_missing(video_out_path)
            input_video_name = vid_name + '.mp4'
            output_video_path = osp.join(video_out_path, input_video_name)
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(frame_out, output_video_path)
            os.system(cmd_str)

    def plot_track(self, im0s, online_tlwhs, online_ids, cls, frame_id, fps, frame_out):
        if self.show_image or self.frame_dir is not None:
            if self.tracker == 'ocsort':
                cls = None
            online_im = plot_tracking(im0s, online_tlwhs, online_ids, cls, frame_id=frame_id, fps=fps)
        if self.show_image:
            cv2.imshow('online_im', online_im)
            ch = cv2.waitKey(1)
        mkdir_if_missing(frame_out)
        if self.frame_dir is not None:
            cv2.imwrite(osp.join(frame_out, '{:05d}.jpg'.format(frame_id)), online_im)
