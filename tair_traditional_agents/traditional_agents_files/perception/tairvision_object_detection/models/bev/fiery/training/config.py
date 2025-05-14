import argparse
from tairvision.models.bev.lss.training.config import _C, CfgNode


CN = CfgNode

_C.TIME_RECEPTIVE_FIELD = 3  # how many frames of temporal context (1 for single timeframe)
_C.N_FUTURE_FRAMES = 4  # how many time steps into the future to predict
_C.BEV_MODE = "rev_map" # it can be rev_map (reverse mapping) or lss
_C.USE_RADAR = True
_C.USE_GRID_SAMPLE = False

_C.PRETRAINED = CN()
_C.PRETRAINED.MODE = 'LSS'
_C.PRETRAINED.LOAD_WEIGHTS = True
_C.PRETRAINED.PATH = '/workspace/ok21/exps/2023_04_16_at_2227_lss_static_800mf_res18bifpn_nostat_pretrained_revmap_05050545_704256_radar/default/version_0/checkpoints/epoch=19-step=17579.ckpt'

_C.MODEL.FUTURE_PREDICTOR = "fiery" # Future predictor model (fiery or beverse)
_C.MODEL.DISTRIBUTION = CN()
_C.MODEL.DISTRIBUTION.LATENT_DIM = 32
_C.MODEL.DISTRIBUTION.MIN_LOG_SIGMA = -5.0
_C.MODEL.DISTRIBUTION.MAX_LOG_SIGMA = 5.0

_C.MODEL.FUTURE_PRED = CN()
_C.MODEL.FUTURE_PRED.N_GRU_BLOCKS = 3
_C.MODEL.FUTURE_PRED.N_RES_LAYERS = 3

_C.INSTANCE_SEG = CN()

_C.INSTANCE_FLOW = CN()
_C.INSTANCE_FLOW.ENABLED = True

_C.PROBABILISTIC = CN()
_C.PROBABILISTIC.ENABLED = True  # learn a distribution over futures
_C.PROBABILISTIC.WEIGHT = 100.0
_C.PROBABILISTIC.FUTURE_DIM = 6  # number of dimension added (future flow, future centerness, offset, seg)


def get_parser():
    parser = argparse.ArgumentParser(description='Fiery training')
    # TODO: remove below?
    parser.add_argument('--config-file', default='', metavar='FILE', help='path to config file')
    parser.add_argument('--enable-wandb', dest='enable_wandb', help='Activate WandB logging',
                        action='store_true', default=False)
    parser.add_argument('--enable-tensorboard', dest='enable_tensorboard', help='Activate Tensorboard logging',
                        action='store_true', default=False)
    parser.add_argument('--enable-mlflow', dest='enable_mlflow', help='Activate MlFlow logging',
                        action='store_true', default=False)
    parser.add_argument('--disable-distributed-training', dest='disable_distributed_training',
                        help='Disable distributed training',
                        action='store_true', default=False)
    parser.add_argument('--wandb-entity-name', type=str, default="esat", help='Wandb Entiy Name')
    parser.add_argument('--wandb-project-name', type=str, default="lss_trial", help='Wandb Entiy Name')
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
