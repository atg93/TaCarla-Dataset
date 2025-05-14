from tairvision.models.bev.lss.training.config import _C, CfgNode

CN = CfgNode

_C.LOG_DIR = '/workspace/gc21/cvt/cvt_gkt/'
_C.BATCHSIZE = 3
_C.EPOCHS = 30

# _C.PRETRAINED.PATH = '/workspace/pa22/experiments/nuimages/nuimages_fcos_regy8gf_bifpn_multiscale_adamwclipping_out_128/checkpoint.pth'
                    # '/workspace/pa22/experiments/nuimages/nuimages_fcos_regy800mf_bifpn_multiscale_adamwclipping_out_128/checkpoint.pth'
                    # '/workspace/gc21/cvt/nuimages_fcos_efficientnet_b4_bifpn_multiscale_adamwclippingNEW_50ep/checkpoint.pth'

_C.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION = False

_C.MODEL.ENCODER.BACKBONE.LAYERS_TO_BEV = ['0', '1', '2', '3', 'p6', 'p7']
_C.MODEL.ENCODER.BACKBONE.LAYERS_TO_CVA = ['2', '0']
_C.MODEL.ENCODER.BACKBONE.DOWNSAMPLE = [16, 4]

_C.MODEL.ENCODER.BACKBONE.CROSS_VIEW = CN()
_C.MODEL.ENCODER.BACKBONE.CROSS_VIEW.HEADS = 4
_C.MODEL.ENCODER.BACKBONE.CROSS_VIEW.DIM_HEAD = 32
_C.MODEL.ENCODER.BACKBONE.CROSS_VIEW.QKV_BIAS = True
_C.MODEL.ENCODER.BACKBONE.CROSS_VIEW.SKIP = True
_C.MODEL.ENCODER.BACKBONE.CROSS_VIEW.NO_IMG_FEATS = False
_C.MODEL.ENCODER.BACKBONE.CROSS_VIEW.MID = [2, 2]

_C.MODEL.ENCODER.BACKBONE.BEV_EMBEDDING = CN()
_C.MODEL.ENCODER.BACKBONE.BEV_EMBEDDING.SIGMA = 1.0
_C.MODEL.ENCODER.BACKBONE.BEV_EMBEDDING.H_METERS = 100.0
_C.MODEL.ENCODER.BACKBONE.BEV_EMBEDDING.W_METERS = 100.0
_C.MODEL.ENCODER.BACKBONE.BEV_EMBEDDING.OFFSET = 0.0
_C.MODEL.ENCODER.BACKBONE.BEV_EMBEDDING.DECODER_BLOCKS = [128, 128, 64]

_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR = 4e-3
_C.OPTIMIZER.WEIGHT_DECAY = 1e-7
_C.GRAD_NORM_CLIP = 5

_C.SCHEDULER = CN()
_C.SCHEDULER.DIV_FACTOR = 10.0
_C.SCHEDULER.PCT_START = 0.3
_C.SCHEDULER.FINAL_DIV_FACTOR = 10
_C.SCHEDULER.CYCLE_MOMENTUM = False


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
