import argparse
import yaml
import os
import copy
import sys


def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    # dataset arguments
    parser.add_argument('--data-path', default='/datasets/coco-2017',
                        help='dataset')
    parser.add_argument('--dataset', default='coco',
                        help='dataset')
    parser.add_argument('--data-augmentation', default="hflip",
                        help='data augmentation policy (default: hflip)')
    parser.add_argument('--eval-data-augmentation', default=None,
                        help='evaluation data augmentation')

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
    parser.add_argument("--cls-loss", default=None, type=str, dest="cls_loss",
                        help="Classification loss type for FCOS, Focal Loss or AP Loss")
    parser.add_argument("--ap-delta", default=0.5, type=float, dest="ap_delta", help="delta value for AP Loss")

    # maskfcos arguments
    parser.add_argument('--roi-output-size', default=14, type=int, help='Output size for roi align')
    parser.add_argument('--roi-sampling-ratio', default=2, type=int, help='Output sampling ratio for roi align')

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
    parser.add_argument('--exclude-pretrained-params', default=['(?!backbone)'], nargs='+',
                        help='list of parameter names to exclude during loading')

    # optimizer arguments
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--lr-backbone', default=None, type=float,
                        help='initial backbone learning rate, None')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--amp', default=False)
    parser.add_argument('--optimizer-type', default="sgd", type=str,
                        help='type of the optimizer, sgd is the default value')
    parser.add_argument('--clip-grad-norm', type=float, help='clip grad norm')

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
    parser.add_argument('--calc-val-loss', dest="calc_val_loss", action="store_true",
                        help='Enable to calculate validation loss during eval')

    # test-demo related arguments
    parser.add_argument('--video-path', default=None,
                        help='dataset')
    parser.add_argument('--save-path', default=None,
                        help='path to save video output')
    parser.add_argument('--score-thres', default=0.6,
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

    # distributed training parameters
    parser.add_argument("--sync-bn", dest="sync_bn", action="store_true",
                        help="Use sync batch norm")
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    # other arguments
    parser.add_argument('--print-freq', default=20, type=int,
                        help='print frequency')
    parser.add_argument('--cfg', default=None,
                        help='Config file for experiment')
    parser.add_argument('--wandb-id', dest="wandb_id", default=None,
                        help='Wandb id for experiment')
    parser.add_argument('--model-kwargs', default=None,
                        help='Extra model settings')
    parser.add_argument('--debug', dest="debug", action="store_true",
                        help='debug mode')

    # additional tracking argument
    parser.add_argument("--use-track", dest="use_track", default=False, action="store_true",
                        help="use track for detection.")

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

    # update args wrt config file if given in the command line
    if args_command.cfg is not None:
        with open(args_command.cfg) as file:
            config = yaml.safe_load(file)
        for key, value in config.items():
            if args.__contains__(key):
                args.__setattr__(key, value)
            else:
                print(key)
                raise ValueError
    # finally, update args from the difference
    for key, value in difference.items():
        args.__setattr__(key, value)

    return args


def add_resume(args):
    if args.resume is None:
        resume_path = os.path.join(args.output_dir, 'checkpoint.pth')
        args.__setattr__('resume', resume_path)


def store_yaml(args):
    config_path = os.path.join(args.output_dir, 'config.yaml')
    args_dict = copy.copy(vars(args))
    args_dict.pop('cfg', None)
    args_dict.pop('resume', None)
    args_dict.pop('output_dir', None)
    args_dict.pop('mood_model', None)
    args_dict.pop('wandb_id', None)
    args_dict.pop('debug', None)
    with open(config_path, 'w+') as file:
        yaml.dump(args_dict, file, default_flow_style=False)
