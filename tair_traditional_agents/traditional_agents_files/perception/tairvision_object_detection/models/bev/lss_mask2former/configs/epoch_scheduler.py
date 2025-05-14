import argparse
from tairvision.models.bev.lss.training.config import CfgNode

CN = CfgNode

SCHEDULER_CALLBACK = CN()
SCHEDULER_CALLBACK.NAME = 'EpochScheduler'
SCHEDULER_CALLBACK.CONFIG = CN()
SCHEDULER_CALLBACK.CONFIG.SCHEDULER_NAME = 'MultiStepLR'
SCHEDULER_CALLBACK.CONFIG.MILESTONES = [2, 3]