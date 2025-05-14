import os
import numpy as np
import torch
import os.path as osp
from tairvision.models.tracking.tair_track import TAIRTracker
from _utils import time_sync, mkdir_if_missing
from tairvision.references.tracking.config import get_arguments
from evaluator import Evaluator


def prepare_directories(demo_path, output_format, tracker):
    result_root = osp.join(demo_path, tracker)
    frame_dir = None if output_format == 'text' else osp.join(result_root, 'frame')
    result_filename = osp.join(result_root, 'results.txt')
    mkdir_if_missing(result_root)
    mkdir_if_missing(frame_dir)
    return result_root, frame_dir, result_filename

detection_time = []
tracker_time = []
seqs = ['MOT17-13-FRCNN',
        'MOT17-04-FRCNN',
        'MOT17-09-FRCNN',
        'MOT17-11-FRCNN',
        'MOT17-10-FRCNN',
        'MOT17-02-FRCNN',
        'MOT17-05-FRCNN']


def detect(opt):
    opt.detector_img_size = [1920, 1080]

    n_seq = len(seqs)
    track = TAIRTracker(opt)
    for vid_idx, seq in enumerate(seqs):
        print('{}/{}'.format(vid_idx+1, n_seq))
        track.create_tracker()
        img_path = os.path.join(opt.data_root, seq, 'img1')
        gt_file = osp.join(opt.data_root, seq, 'gt/gt.txt')
        results = []

        dataloader = track.get_data_loader(img_path)
        frame_out = osp.join(opt.frame_dir, seq)
        for frame_idx, (path, img, im0s) in enumerate(dataloader):
            if (frame_idx + 1) % 100 == 0:
                print('{} frames are processed'.format(frame_idx + 1))
            t0 = time_sync()  # start time

            output_img = track.inference(img, gt_file, frame_idx)
            t1 = time_sync()  # detection model

            outputs, scores = track.update(output_img, img, im0s)
            online_tlwhs, online_ids = track.postprocess(outputs, scores)
            results.append((frame_idx + 1, online_tlwhs, online_ids))
            t2 = time_sync()  # tracker

            fps = 1 / (t2 - t0) # detection + tracking speed
            track.plot_track(im0s, online_tlwhs, online_ids, None, (frame_idx+1), fps, frame_out)
            t3 = time_sync()  # plotting

            detection_time.append(t1 - t0)
            tracker_time.append(t2 - t1)

        res = os.path.join(opt.result_root, 'track_results')
        mkdir_if_missing(res)
        # write results to txt file with required format
        track.write_results_new(filename=osp.join(res, '{}.txt'.format(seq)), results=results)
        # convert frames to video output if needed
        # track.write_video(seq, frame_out)

    # End of prediction, start evaluation
    print('Prediction is done')
    print('Detection mean time: {}, Tracker mean time: {}'.format(np.mean(detection_time), np.mean(tracker_time)))

    evaluator = Evaluator(opt.data_root, opt.out_root, opt.tracker_model)
    evaluator.evaluate()


if __name__ == '__main__':
    opt = get_arguments()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    result_root, frame_dir, result_filename = prepare_directories(opt.demo_path, opt.output_format, opt.tracker_model)
    opt.result_root = result_root
    opt.frame_dir = frame_dir
    opt.result_filename = result_filename
    with torch.no_grad():
        detect(opt)
