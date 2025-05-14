from tairvision.models.bev.lss.training.config import CfgNode as CN


def get_adamw_transformer_optimizer_cfg_defaults():
    OPTIMIZER = CN()
    OPTIMIZER.NAME = 'AdamW'
    OPTIMIZER.CONFIG = CN()
    OPTIMIZER.CONFIG.LR = 3e-4
    OPTIMIZER.CONFIG.WEIGHT_DECAY = 2e-4
    OPTIMIZER.PARAMS = CN()
    OPTIMIZER.PARAMS.NAMES = ["encoder.backbone"]
    OPTIMIZER.PARAMS.CONFIGS = [{"lr": 3e-5}]

    return OPTIMIZER
