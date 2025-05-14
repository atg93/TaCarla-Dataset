import argparse
import yaml
import sys

# Get names and colors
NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker_model', type=str, default='bytetrack')
    parser.add_argument('--track_only', default=False, action="store_true")
    parser.add_argument('--show_image', action='store_true', help='display tracking video results')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--output-format', type=str, default='video', help='video or text')
    parser.add_argument('--demo-path', type=str, default='/track_demo', help='video or text')
    parser.add_argument("--webcam", dest="webcam", default=False, action="store_true", help="use webcam in tracking.")
    # eval paths
    parser.add_argument('--data-root', type=str, default='/datasets/track_data/mot_data/MOT17/train',
                        help='dataset root path')
    parser.add_argument('--out-root', type=str, default='/workspace/track_demo',
                        help='output directory')
    parser.add_argument('--detector-mode', type=str, default='bdd',
                        help='mode for detector labels (bdd/coco/face)')


    # GET ARGS from tairvision.references.detection.config
    # dataset arguments
    parser.add_argument('--data-path', default='/datasets/coco-2017',
                        help='dataset')
    parser.add_argument('--dataset', default='coco',
                        help='dataset')
    parser.add_argument('--data-augmentation', default="hflip",
                        help='data augmentation policy (default: hflip)')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)

    # Transform arguments
    parser.add_argument('--transform-min-size', default=800, type=int,
                        help='Min size for transform')
    parser.add_argument('--transform-max-size', default=1333, type=int,
                        help='Max size for transform')

    # model arguments
    parser.add_argument('--model', default='fcos_regnet_fpn',
                        help='model')

    # processor arguments
    parser.add_argument('--device', default='cuda',
                        help='device')
    parser.add_argument('--workers', '-j', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # backbone arguments
    parser.add_argument('--backbone', default='regnet_y_1_6gf',
                        help='backbone type')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo",
                        action="store_true")

    # fpn arguments
    parser.add_argument("--pyramid-type", default="bifpn", type=str,
                        help="Type of feature pyramid network")
    parser.add_argument('--bifpn-repeats', default=3, type=int,
                        help='Times to repeat bifpn module')
    parser.add_argument("--fusion-type", default="fastnormed", type=str,
                        help="Type of feature feature fusion in BiFPN blocks")
    parser.add_argument("--use-depthwise", dest="use_depthwise",
                        help="Use depthwise convolutions in BiFPN blocks", action="store_true")
    parser.add_argument("--use-P2", dest="use_P2", action="store_true",
                        help="Use P2 feature layer of FPN")

    # extra block arguments
    parser.add_argument("--no-extra-blocks", dest="no_extra_blocks", action="store_true",
                        help="No extra blocks after FPN layers")
    parser.add_argument("--extra-before", dest="extra_before", action="store_true",
                        help="Putting extra blocks to the backbone")

    # common head arguments
    parser.add_argument('--post-nms-topk', default=100, type=int,
                        help='Top-k after applying NMS')
    parser.add_argument('--nms-thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--loss-weights', default=[1.0, 1.0, 1.0, 0.25], nargs='+', type=float,
                        help='Loss weightings')

    # fcos arguments
    INF = 100000000
    parser.add_argument('--fpn-strides', default=[8, 16, 32, 64, 128], nargs='+', type=int,
                        help='FPN strides for feature levels')
    parser.add_argument('--sois', default=[-1, 64, 128, 256, 512, INF], nargs='+', type=int,
                        help='Sizes of interest')
    parser.add_argument("--thresh-with-ctr", dest="thresh_with_ctr", action="store_true",
                        help="Thresholding score with centerness")
    parser.add_argument("--use-deformable", dest="use_deformable", help="Using DCNv2 for FCOS Heads",
                        action="store_true")

    # retinanet-fasterrcnn arguments
    parser.add_argument('--anchor-sizes', default=[32, 64, 128, 256, 512], nargs='+', type=int,
                        help='Anchor size to be used by the anchor generator')
    parser.add_argument('--rpn-score-thresh', default=None, type=float,
                        help='rpn score threshold for faster-rcnn')

    # context module arguments
    parser.add_argument("--context-module", default=None, type=str,
                        help="Type of context module after feature pyramid")

    # training arguments
    parser.add_argument('--batch-size', '-b', default=16, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=13, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--output-dir', default=None,
                        help='path where to save')
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--coco-pretrained', default=None,
                        help='resume from COCO pretrained backbone + neck')

    # optimizer arguments
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    # lr schedular arguments
    parser.add_argument('--lr-scheduler', default="multisteplr", help='the lr scheduler (default: multisteplr)')
    parser.add_argument('--lr-step-size', default=8, type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')

    # evaluation arguments
    parser.add_argument("--test-only", dest="test_only", action="store_true",
                        help="Only evaluate the model")
    parser.add_argument('--maxDets', default=[10, 100, 1000], nargs='+', type=int,
                        help='Max detection for coco-like eval')

    # test-demo related arguments
    parser.add_argument('--video-path', default=None,
                        help='dataset')
    parser.add_argument('--score-thres', default=0.5, type=float,
                        help='score threshold for ignoring bboxes')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='camera device id')
    parser.add_argument("--record", default=None,
                        help="Video output record directory")
    parser.add_argument("--use-bdd-labels", action="store_true",
                        help="Use BDD100k label names")
    parser.add_argument("--mood-model", default=None,
                        help="Directory for mood-gender-age model")
    parser.add_argument('--video-crop-height', type=int, default=None,
                        help='Image height after video cropping')
    parser.add_argument("--face-demo", dest="face_demo", default=False, action="store_true", help="")

    # distributed training parameters
    parser.add_argument("--sync-bn", dest="sync_bn", action="store_true",
                        help="Use sync batch norm")
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    # OC-SORT arguments
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="the iou threshold in Sort for matching")
    parser.add_argument("--use_byte", dest="use_byte", default=False, action="store_true", help="use byte in tracking.")

    # other arguments
    parser.add_argument('--print-freq', default=20, type=int,
                        help='print frequency')
    parser.add_argument('--cfg', default=None,
                        help='Config file for experiment')

    # ByteTrack arguments
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")



    return parser


def get_arguments():
    # override order goes as follow: defaults < config < command line
    parser = get_args_parser()
    # get default arguments from config.py
    args = parser.parse_args([])
    # get arguments from command line
    args_command, _ = parser._parse_known_args(sys.argv[1:], argparse.Namespace())
    # get the arguments given in the command line, and keep them
    difference = {}
    for key, value in vars(args_command).items():
        difference[key] = value

    difference['names'] = NAMES

    # update args wrt config file if given in the command line
    if args_command.cfg is not None:
        with open(args_command.cfg) as file:
            config = yaml.safe_load(file)
        for key, value in config.items():
            if args.__contains__(key):
                args.__setattr__(key, value)
            else:
                raise ValueError
    # finally, update args from the difference
    for key, value in difference.items():
        args.__setattr__(key, value)

    return args
