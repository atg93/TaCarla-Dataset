import argparse
import os
from fvcore.common.config import CfgNode as _CfgNode


def convert_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary."""
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, _CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                'Key {} with value {} is not a valid type; valid types: {}'.format(
                    '.'.join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


class CfgNode(_CfgNode):
    """Remove once https://github.com/rbgirshick/yacs/issues/19 is merged."""

    def convert_to_dict(self):
        return convert_to_dict(self)


CN = CfgNode

_C = CN()
_C.LOG_DIR = '/workspace/' + os.environ['USER'] + '/exps'
_C.TAG = 'default'

_C.GPUS = [0, 1, 2, 3]  # GPUs to use
_C.PRECISION = 16  # 16bit or 32bit
_C.BATCHSIZE = 16
_C.EPOCHS = 20

_C.N_WORKERS = 16
_C.VIS_INTERVAL_TRAIN = 1000
_C.VIS_INTERVAL_VAL = 100
_C.LOGGING_INTERVAL = 100

_C.PRETRAINED = CN()
_C.PRETRAINED.LOAD_WEIGHTS = True  # If True, mean and std from nuImages Dataset else ImageNet
_C.PRETRAINED.PATH = \
    '/workspace/pa22/experiments/nuimages/nuimages_fcos_regy800mf_bifpn_multiscale_adamwclipping_out_128/checkpoint.pth'

_C.DATASET = CN()
_C.DATASET.DATAROOT = '/datasets/nu/nuscenes/'
_C.DATASET.VERSION = 'trainval'
_C.DATASET.NAME = 'nuscenes'
_C.DATASET.IGNORE_INDEX = 255  # Ignore index when creating flow/offset labels
_C.DATASET.FILTER_INVISIBLE_VEHICLES = True  # Filter vehicles that are not visible from the cameras
_C.DATASET.FILTER_CLASSES = "vehicle"  # Options: "car", "vehicle_pedestrian", "all", and "vehicle"(default)
_C.DATASET.FILTER_CLASSES_SEGM = "vehicle"  # Options: "car", "vehicle_pedestrian", "all", and "vehicle"(default)
_C.DATASET.BALANCE_DATASET = False
_C.DATASET.BOX_RESIZING_COEF = 1
_C.DATASET.BOX_RESIZING_COEF_SEGM = 1
_C.DATASET.BOX_AUGMENTATION = False
_C.DATASET.EGO_POSITION = 'center'
# SAMPLING_RATIO is to make the subset of the dataset smaller. 1 means no subset and the whole dataset is used.
# When 2 is used, every second sample is used. 3 means every third sample is used and so on.
_C.DATASET.SAMPLING_RATIO = 1 


_C.TIME_RECEPTIVE_FIELD = 1  # How many frames of temporal context (1 for single timeframe)
_C.N_FUTURE_FRAMES = 0  # How many time steps into the future to predict
_C.BEV_SCALE = False # whether scaling BEV or not
_C.BEV_SCALED_SIZE = (120, 120)

_C.IMAGE = CN()
_C.IMAGE.FINAL_DIM = (224, 480)
_C.IMAGE.RESIZE_SCALE = 0.3
_C.IMAGE.TOP_CROP = 46
_C.IMAGE.ORIGINAL_HEIGHT = 900  # Original input RGB camera height
_C.IMAGE.ORIGINAL_WIDTH = 1600  # Original input RGB camera width
_C.IMAGE.NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
_C.IMAGE.HFLIP_PROB = 0.5
_C.IMAGE.VFLIP_PROB = 0.5
_C.IMAGE.ROTATE_PROB = 0.5
_C.IMAGE.ROTATION_DEGREE_INCREMENTS = 45

_C.PCLOUD = CN()
_C.PCLOUD.N_FEATS = 0

_C.LIFT = CN()  # Image to BEV lifting
_C.LIFT.X_BOUND = [-50.0, 50.0, 0.5]  # Forward
_C.LIFT.Y_BOUND = [-50.0, 50.0, 0.5]  # Sides
_C.LIFT.Z_BOUND = [-10.0, 10.0, 20.0] # Height
_C.LIFT.D_BOUND = [1.0, 50.0, 1.0]

_C.MODEL = CN()

_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.DOWNSAMPLE = 8
_C.MODEL.ENCODER.OUT_CHANNELS = 64
_C.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION = True
_C.MODEL.ENCODER.BACKBONE = CN()
_C.MODEL.ENCODER.BACKBONE.TYPE = 'regnet'
_C.MODEL.ENCODER.BACKBONE.VERSION = '_y_800mf'
_C.MODEL.ENCODER.BACKBONE.LAYERS = ['0', '1', '2', 'p6', 'p7']
_C.MODEL.ENCODER.BACKBONE.PYRAMID_CHANNELS = 256
_C.MODEL.ENCODER.BACKBONE.PYRAMID_TYPE = 'bifpn'
_C.MODEL.ENCODER.BACKBONE_LAYERS_TO_BEV = ['0', '1', '2', 'p6', 'p7']
_C.MODEL.ENCODER.BACKBONE.BIFPN_REPEAT = 3
_C.MODEL.TEMPORAL_MODEL = CN()
_C.MODEL.TEMPORAL_MODEL.START_OUT_CHANNELS = 64
_C.MODEL.TEMPORAL_MODEL.EXTRA_IN_CHANNELS = 0
_C.MODEL.TEMPORAL_MODEL.INBETWEEN_LAYERS = 0
_C.MODEL.TEMPORAL_MODEL.PYRAMID_POOLING = True
_C.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE = True

_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.BACKBONE = CN()
_C.MODEL.DECODER.BACKBONE.TYPE = 'resnet'
_C.MODEL.DECODER.BACKBONE.VERSION = 18
_C.MODEL.DECODER.BACKBONE.LAYERS = ['0', '1', '2', '3']
_C.MODEL.DECODER.BACKBONE.PYRAMID_CHANNELS = 128
_C.MODEL.DECODER.BACKBONE.PYRAMID_TYPE = None

_C.MODEL.USE_HEADDYNAMIC = True
_C.MODEL.HEADDYNAMIC = CN()
_C.MODEL.HEADDYNAMIC.SEMANTIC_SEG = CN()
_C.MODEL.HEADDYNAMIC.SEMANTIC_SEG.WEIGHTS = [1.0, 2.0]  # Per class cross entropy weights (bg, dynamic)
_C.MODEL.HEADDYNAMIC.SEMANTIC_SEG.USE_TOP_K = True  # Backprop only top-k hardest pixels
_C.MODEL.HEADDYNAMIC.SEMANTIC_SEG.TOP_K_RATIO = 0.25
_C.MODEL.HEADDYNAMIC.INSTANCE_SEG = CN()

_C.MODEL.USE_HEADSTATIC = False
_C.MODEL.HEADSTATIC = CN()
_C.MODEL.HEADSTATIC.LANES = CN()
_C.MODEL.HEADSTATIC.LANES.WEIGHTS = [1.0, 1.0]  # Per class cross entropy weights (bg, lanes)
_C.MODEL.HEADSTATIC.LANES.USE_TOP_K = False  # Backprop only top-k hardest pixels
_C.MODEL.HEADSTATIC.LANES.TOP_K_RATIO = 0.25
_C.MODEL.HEADSTATIC.LINES = CN()
_C.MODEL.HEADSTATIC.LINES.WEIGHTS = [0.1, 1.0]  # Per class cross entropy weights (bg, lines)
_C.MODEL.HEADSTATIC.LINES.USE_TOP_K = False  # Backprop only top-k hardest pixels
_C.MODEL.HEADSTATIC.LINES.TOP_K_RATIO = 0.05

_C.MODEL.USE_HEAD2D = True
_C.MODEL.HEAD2D = CN()
_C.MODEL.HEAD2D.NUM_CLASSES = 2

_C.MODEL.USE_HEAD3D = True
_C.MODEL.HEAD3D = CN()
_C.MODEL.HEAD3D.NUM_CLASSES = 2
_C.MODEL.HEAD3D.TARGET_TYPE = "xdyd"
_C.MODEL.HEAD3D.REGRESSION_CHANNELS = [2, 2, 2, 2]
_C.MODEL.HEAD3D.REGRESSION_FUNCTIONS = ["relu", "relu", "none", "none"]

_C.MODEL.BN_MOMENTUM = 0.05

_C.FUTURE_DISCOUNT = 0.95

_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR = 3e-4
_C.OPTIMIZER.WEIGHT_DECAY = 1e-7
_C.GRAD_NORM_CLIP = 5


def get_parser():
    parser = argparse.ArgumentParser(description='Lift Splat training')
    # TODO: remove below?
    parser.add_argument('--config-file', default='', metavar='FILE', help='path to config file')
    parser.add_argument('--enable-wandb', dest='enable_wandb', help='Activate WandB logging',
                        action='store_true', default=False)
    parser.add_argument('--enable-tensorboard', dest='enable_tensorboard', help='Activate Tensorboard logging',
                        action='store_true', default=False)
    parser.add_argument('--enable-mlflow', dest='enable_mlflow', help='Activate MlFlow logging',
                        action='store_true', default=False)
    parser.add_argument('--disable-distributed-training', dest='disable_distributed_training', help='Disable distributed training',
                        action='store_true', default=False)
    parser.add_argument('--wandb-entity-name', type=str, help='Wandb Entity Name')
    parser.add_argument('--wandb-project-name', type=str, help='Wandb Project Name')
    parser.add_argument('--find-unused-parameters', dest='find_unused_parameters', help='Activate find unused parameters option',
                        action='store_true', default=False)
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint in order to resume training')
    parser.add_argument('--fresh-start-for-checkpoint', dest='fresh_start_for_checkpoint',
                        help='Do not load trainer settings just only the model',
                        action='store_true', default=False)
    parser.add_argument('--num-nodes', default=1, type=int, help='Number of nodes for distributed training')
    parser.add_argument('--init-method', type=str, default="env://", help='init method for distributed training')
    parser.add_argument(
        'opts', help='Modify config options using the command-line', default=None, nargs=argparse.REMAINDER,
    )
    return parser


def get_cfg(args=None, cfg_dict=None):
    """ First get default config. Then merge cfg_dict. Then merge according to args. """

    cfg = _C.clone()

    if cfg_dict is not None:
        cfg.merge_from_other_cfg(CfgNode(cfg_dict))

    if args is not None:
        if args.config_file:
            cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    return cfg


def get_cf_from_yaml(yaml_path):

    cfg = _C.clone()
    cfg.merge_from_file(yaml_path)
    return cfg
