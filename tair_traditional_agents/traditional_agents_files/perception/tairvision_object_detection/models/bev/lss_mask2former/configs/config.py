import argparse
from tairvision.models.bev.lss.training.config import _C, CfgNode, get_parser, get_cfg
from tairvision.models.bev.lss_mask2former.configs.pv_dabdetr import get_dab_head2d_cfg_defaults
from tairvision.models.bev.lss_mask2former.configs.mask2former import get_bev_mask2former_cfg_defaults
from tairvision.models.bev.lss_mask2former.configs.polynomial_lr_scheduler import get_polynomial_lr_scheduler_cfg_defaults
from tairvision.models.bev.lss_mask2former.configs.adamw_transformer_optimizer import get_adamw_transformer_optimizer_cfg_defaults


CN = CfgNode

_C.MODEL.DYNAMIC_TRANSFORMER_DECODER = get_bev_mask2former_cfg_defaults()
_C.MODEL.HEAD2D = get_dab_head2d_cfg_defaults()

_C.CALLBACKS = CN()
_C.CALLBACKS.SCHEDULER_CALLBACK = get_polynomial_lr_scheduler_cfg_defaults()

_C.OPTIMIZER = get_adamw_transformer_optimizer_cfg_defaults()
_C.GRAD_NORM_CLIP = 0.01

