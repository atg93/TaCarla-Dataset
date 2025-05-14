import argparse
from tairvision.models.bev.lss.training.config import _C, CfgNode

CN = CfgNode

_C.LOG_DIR = '/workspace/ct22/experiments/lssbevdet4d'

_C.PRETRAINED.PATH = '/workspace/pa22/experiments/nuimages/nuimages_fcos_regy800mf_bifpn_multiscale_adamwclipping_out_128/checkpoint.pth'
# _C.PRETRAINED.PATH = '/workspace/pa22/experiments/nuimages/nuimages_fcos_regy8gf_bifpn_multiscale_adamwclipping_out_128/checkpoint.pth'

_C.USE_RADAR = True
_C.USE_LIDAR_HEAD = False
_C.USE_LIDAR_DECODER = False

_C.MODEL.ENCODER.USE_BEVPOOLV2 = True


_C.MODEL.CPHEAD = CN()
_C.MODEL.CPHEAD.USE_VELOCITY = False
_C.MODEL.CPHEAD.MIN_RADIUS = [4, 12, 10, 1, 0.85, 0.175]  # Radii for car, truck, bus, barrier, cycle, pedestrian
_C.MODEL.CPHEAD.NMS_TYPE = ['rotate',  'rotate', 'rotate', 'circle', 'rotate', 'rotate']
_C.MODEL.CPHEAD.NMS_THR = [0.2, 0.2, 0.2, 0.2, 0.2, 0.45]
_C.MODEL.CPHEAD.NMS_RESCALE_FACTOR = [1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]]
# car, truck, constveh, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone
"""
EXAMPLE CONFIG FOR VEHICLE + PEDESTRIAN TWO CLASS CASE
_C.MODEL.CPHEAD.MIN_RAD = [4, 0.175]  #THE VEHICLE RADIUS HERE IS SUBJECT TO EXPERIMENT
_C.MODEL.CPHEAD.NMS_TYPE = ['rotate',  'rotate']
_C.MODEL.CPHEAD.NMS_THR = [0.2, 0.45]
_C.MODEL.CPHEAD.NMS_RESCALE_FACTOR = [1.0, 4.5]
"""

_C.OPTIMIZER.LR_STEPS = [18, 22]


def get_parser():
    parser = argparse.ArgumentParser(description='CPVP training')
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
    parser.add_argument('--wandb-entity-name', type=str, default="tair", help='Wandb Entiy Name')
    parser.add_argument('--wandb-project-name', type=str, default="cpvp_checkpoints", help='Wandb Entiy Name')
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
