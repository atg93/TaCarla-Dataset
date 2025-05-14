from torch.optim import AdamW
import itertools
import torch


class AdamWClipping(AdamW):
    def __init__(self,
                 max_norm,
                 norm_type=2.0,
                 **kwargs) -> None:
        super(AdamWClipping, self).__init__(**kwargs)
        self.max_norm = max_norm
        self.norm_type = norm_type

    def step(self, closure=None):
        # TODO, this part should be modified for amp mixed precision training
        all_params = itertools.chain(*[x["params"] for x in self.param_groups])
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.max_norm, norm_type=self.norm_type)
        super().step(closure=closure)
