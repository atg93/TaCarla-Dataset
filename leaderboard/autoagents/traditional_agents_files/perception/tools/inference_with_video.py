import argparse
import torch
from utils.Inference_utils_sub_folder import *

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Torch inference for lane detection')
parser.add_argument('yaml_file', metavar='FILE', help='path to yaml')
parser.add_argument('video_file', metavar='FILE', help='path to the video')
parser.add_argument('--mask2former', dest='mask2former', action='store_true', help='panoptic mask2former model', required=False)
parser.add_argument('--deeplab', dest='deeplab', action='store_true', help='panoptic deeplab model', required=False)
parser.add_argument('--trt', dest='trt', action='store_true', help='inference in tensorrt', required=False)
parser.add_argument('--trt_filename', dest='trt_filename', metavar='FILE', help='detection mode', required=False)
parser.add_argument('--four_head', dest='four_head', action='store_true', help='special four head post process',
                    required=False)
parser.add_argument('--lane_fit', dest='lane_fit', action='store_true', help='special four head post process',
                    required=False)
parser.add_argument('--lane', dest='lane', action='store_true', help='point based formulation for the creation of mask',
                    required=False)
parser.add_argument('-d', '--detection', dest='detection', action='store_true', help='detection mode', required=False)
parser.add_argument('-dt', '--detection_track', dest='detection_track', action='store_true',
                    help='detection track mode', required=False)
parser.add_argument('--save_video', dest='save_video', action='store_true', help='in order to save video',
                    required=False)
parser.add_argument('--frame_interval', default=5, type=int,
                    help='Number of frames to skip')
parser.add_argument('--cropped', dest='cropped', action='store_true', help='crop the frame for the ford data', required=False)
parser.add_argument('--preserve_ration', dest='preserve_ratio', action='store_true',
                    help='Preserve the ratio while clipping', required=False)

parser.add_argument('--second_yaml', metavar='FILE', help='path to yaml', required=False)

args = parser.parse_args()

if args.detection:
    inference_instance = InferenceDetectionTorch(
        args.yaml_file, video_file=args.video_file,
        save_video=args.save_video, frame_interval=args.frame_interval, cropped=args.cropped,
        preserve_ratio=args.preserve_ratio
    )
elif args.detection_track:
    inference_instance = InferenceTrackedDetectionTorch(
        args.yaml_file, video_file=args.video_file,
        save_video=args.save_video, frame_interval=args.frame_interval, cropped=args.cropped,
        preserve_ratio=args.preserve_ratio
    )

elif args.four_head:
    inference_instance = InferenceSegmentationFourHead(
        args.yaml_file, video_file=args.video_file,
        save_video=args.save_video, frame_interval=args.frame_interval, cropped=args.cropped,
        preserve_ratio=args.preserve_ratio
    )

elif args.lane_fit:
    inference_instance = InferenceSegmentationLaneFit(
        args.yaml_file, video_file=args.video_file,
        save_video=args.save_video, frame_interval=args.frame_interval, cropped=args.cropped,
        preserve_ratio=args.preserve_ratio
    )

elif args.mask2former:
    inference_instance = InferenceMask2formerTorch(
        args.yaml_file, video_file=args.video_file,
        save_video=args.save_video, frame_interval=args.frame_interval, cropped=args.cropped,
        preserve_ratio=args.preserve_ratio
    )

elif args.deeplab:
    inference_instance = InferenceDeeplabTorch(
        args.yaml_file, video_file=args.video_file,
        save_video=args.save_video, frame_interval=args.frame_interval, cropped=args.cropped,
        preserve_ratio=args.preserve_ratio
    )
elif args.lane:
    inference_instance = InferenceSegmentationLane(
        args.yaml_file, video_file=args.video_file,
        save_video=args.save_video, frame_interval=args.frame_interval, cropped=args.cropped,
        preserve_ratio=args.preserve_ratio
    )

else:
    if args.trt:
        inference_instance = InferenceSegmentationTensorRT(
            args.yaml_file, video_file=args.video_file,
            save_video=args.save_video, tensorrt_file=args.trt_filename,
            frame_interval=args.frame_interval, cropped=args.cropped,
            preserve_ratio=args.preserve_ratio
        )
    else:
        inference_instance = InferenceSegmentationTorch(
            args.yaml_file, video_file=args.video_file,
            save_video=args.save_video, frame_interval=args.frame_interval, cropped=args.cropped,
            preserve_ratio=args.preserve_ratio,second_yaml=args.second_yaml
        )


for i in range(10):
    inference_instance.main_loop()
