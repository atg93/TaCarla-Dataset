import numpy as np
import torch
from tairvision.models.tracking.tair_track import TAIRTracker
from _utils import time_sync,prepare_directories
from tairvision.references.tracking.config import get_arguments


detection_time = []
tracker_time = []


def detect(opt):
    opt.detector_img_size = [1280, 720]
    track = TAIRTracker(opt)
    track.create_tracker()
    results = []
    dataloader = track.get_data_loader(opt.video_path, img_input=False)

    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataloader):
        if (frame_idx + 1) % 100 == 0:
            print('{} frames are processed'.format(frame_idx + 1))
        t0 = time_sync()  # start time
        output_img = track.inference(img, None, frame_idx)
        t1 = time_sync()  # detection model

        outputs, scores = track.update(output_img, img, im0s)
        online_tlwhs, online_ids = track.postprocess(outputs, scores)
        results.append((frame_idx + 1, online_tlwhs, online_ids))
        t2 = time_sync()  # tracker

        fps = 1 / (t2 - t0) # detection + tracking speed
        track.plot_track(im0s, online_tlwhs, online_ids, None, (frame_idx+1), fps, opt.frame_dir)
        t3 = time_sync()  # plotting

        detection_time.append(t1 - t0)
        tracker_time.append(t2 - t1)


    # write results to txt file with required format
    track.write_results_new(filename=opt.result_filename, results=results)
    # convert frames to video output if needed
    vid_name = opt.video_path.split('/')[-1].split('.')[0]
    track.write_video(vid_name, opt.frame_dir)

    # End of prediction, start evaluation
    print('Prediction is done')
    print('Detection mean time: {}, Tracker mean time: {}'.format(np.mean(detection_time), np.mean(tracker_time)))


if __name__ == '__main__':
    opt = get_arguments()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    result_root, frame_dir, result_filename = prepare_directories(opt.demo_path, opt.output_format,
                                                                  opt.tracker_model, opt.video_path)
    opt.result_root = result_root
    opt.frame_dir = frame_dir
    opt.result_filename = result_filename
    with torch.no_grad():
        detect(opt)
