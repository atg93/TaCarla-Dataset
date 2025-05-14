from tairvision.models.bev.lss.training.config import CfgNode as CN


def get_polynomial_lr_scheduler_cfg_defaults():
    SCHEDULER_CALLBACK = CN()
    SCHEDULER_CALLBACK.NAME = 'PolynomialLRScheduler'
    SCHEDULER_CALLBACK.CONFIG = CN()
    SCHEDULER_CALLBACK.CONFIG.LAMBDA_LR_SCHEDULER_POLYNOMIAL_RATE = 0.9
    SCHEDULER_CALLBACK.CONFIG.WARMUP_ITERATIONS = 1000

    return SCHEDULER_CALLBACK