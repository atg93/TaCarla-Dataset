import argparse
import yaml
import os
import copy
import sys

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='PyTorch Mood-Age-Gender Analysis Training', add_help=add_help)
    # Detection args
    parser.add_argument('--backbone', default='regnet_y_1_6gf',
                        help='backbone type')
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo",
                        action="store_true")
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument("--pyramid-type", default="bifpn", type=str,
                        help="Type of feature pyramid network")
    parser.add_argument('--bifpn-repeats', default=3, type=int,
                        help='Times to repeat bifpn module')
    parser.add_argument("--fusion-type", default="fastnormed", type=str,
                        help="Type of feature feature fusion in BiFPN blocks")
    parser.add_argument("--use-depthwise", dest="use_depthwise",
                        help="Use depthwise convolutions in BiFPN blocks", action="store_true")
    parser.add_argument("--use-deformable", dest="use_deformable", help="Using DCNv2 for FCOS Heads",
                        action="store_true")
    parser.add_argument("--use-P2", dest="use_P2", action="store_true",
                        help="Use P2 feature layer of FPN")
    parser.add_argument("--no-extra-blocks", dest="no_extra_blocks", action="store_true",
                        help="No extra blocks after FPN layers")
    parser.add_argument("--extra-before", dest="extra_before", action="store_true",
                        help="Putting extra blocks to the backbone")
    parser.add_argument('--post-nms-topk', default=100, type=int,
                        help='Top-k after applying NMS')
    parser.add_argument('--nms-thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--loss-weights', default=[1.0, 1.0, 1.0, 0.25], nargs='+', type=float,
                        help='Loss weightings')
    parser.add_argument("--context-module", default=None, type=str,
                        help="Type of context module after feature pyramid")
    parser.add_argument('--transform-min-size', default=800, type=int,
                        help='Min size for GeneralizedRCNN Transform')
    parser.add_argument('--transform-max-size', default=1333, type=int,
                        help='Max size for GeneralizedRCNN Transform')
    parser.add_argument('--fpn-strides', default=[8, 16, 32, 64, 128], nargs='+', type=int,
                        help='FPN strides for feature levels')
    INF = 100000000
    parser.add_argument('--sois', default=[-1, 64, 128, 256, 512, INF], nargs='+', type=int,
                        help='Sizes of interest')
    parser.add_argument("--thresh-with-ctr", dest="thresh_with_ctr", action="store_true",
                        help="Thresholding score with centerness")
    # model arguments
    parser.add_argument('--model', default='fcos_regnet_fpn',
                        help='model')
    parser.add_argument('--anchor-sizes', default=[32, 64, 128, 256, 512], nargs='+', type=int,
                        help='Anchor size to be used by the anchor generator')
    parser.add_argument('--rpn-score-thresh', default=None, type=float,
                        help='rpn score threshold for faster-rcnn')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--camera-id', default=0, type=int)
    parser.add_argument('--data-augmentation', default="hflip",
                        help='data augmentation policy (default: hflip)')
    parser.add_argument('--dataset', default='coco',
                        help='dataset')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--record", default=None,
                        help="Video output record directory")
    parser.add_argument('--video-path', default=None,
                        help='dataset')
    parser.add_argument('--video-crop-height', type=int, default=None,
                        help='Image height after video cropping')

    # additional tracking argument
    parser.add_argument("--use-track", dest="use_track", default=False, action="store_true", help="use track for detection.")

    # traffic sign classification argument
    parser.add_argument('--cls-score-thresh', dest="cls_score_thres", default=0.75, type=float,
                        help='classification score threshold for traffic sign')
    # Analysis args
    parser.add_argument('--data-path', type=str, default='/datasets', help='AffectNet dataset path.')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--batch-size', '-b', type=int, default=256, help='Batch size.')
    parser.add_argument('--workers', '-j', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num-head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--num-class', type=int, default=8, help='Number of class.')
    parser.add_argument("--use-gender", help="Use auto-generated gender labels", action="store_true")
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument('--loss-coeffs', default=[1., 1., 1.], nargs='+', type=float,
                        help='Weighting between mood, gender, age losses')
    parser.add_argument('--label-path', default='/datasets',
                        help='path where the label csv is')
    parser.add_argument('--backbone-path',
                        default='/workspace/shared/trained_models/dan_res18_backbone/resnet18_msceleb.pth',
                        help='path where the pretrained backbone is')
    parser.add_argument("--utkface-finetune", dest="utkface_finetune", help="UTKFace Finetuning", action="store_true")
    parser.add_argument("--use-valence", help="Use AffectNet valence and arousal labels", action="store_true")
    parser.add_argument("--face-analysis-multi", help="Use mood, valence, arousal and auto-generated gender labels", action="store_true")
    parser.add_argument("--use-dan", help="Use attention module with age and gender", action="store_true")
    parser.add_argument('--cfg', default=None, help='Config file for experiment')
    parser.add_argument("--use-mood", help="Use AffectNet facial expression labels", action="store_true")
    parser.add_argument("--score-thres", help="Score threshold", default=0.6)
    parser.add_argument("--gender-model", help="gender model", default='/workspace/ig21/gender-utk-new/checkpoint.pth')
    parser.add_argument("--mood-model", help="mood model", default='/workspace/ig21/valence-arousal-new/checkpoint.pth')
    parser.add_argument('--model-kwargs', default=None,
                        help='Extra model settings')
    parser.add_argument("--heatmap", action="store_true", help="Plot heatmap")


    return parser

def get_arguments_analysis():
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
                print('key: ',key)
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
    # args_dict.pop('mood_model', None)
    with open(config_path, 'w+') as file:
        yaml.dump(args_dict, file, default_flow_style=False)


