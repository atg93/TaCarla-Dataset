from tairvision.models.bev.cvt.training.config import _C, CfgNode

CN = CfgNode

_C.LOG_DIR = '/workspace/gc21/cvt/cvt_gkt/'

_C.MODEL.ENCODER.BACKBONE.CROSS_VIEW.BEV_Z = 1.0
_C.MODEL.ENCODER.BACKBONE.CROSS_VIEW.KERNEL_H = 7
_C.MODEL.ENCODER.BACKBONE.CROSS_VIEW.KERNEL_W = 1


def get_cfg(args=None, cfg_dict=None):
    """ First get default config. Then merge cfg_dict. Then merge according to args. """

    cfg = _C.clone()
    cfg.register_deprecated_key("MODEL.DYNAMIC_HEAD")
    cfg.register_deprecated_key("MODEL.DYNAMIC_TRANSFORMER_DECODER")
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
