from pytorch_lightning.callbacks import Callback
import tairvision.nn.optim as opt
from typing import Any, Tuple, List, Optional, Union, Dict


class EpochScheduler(Callback):
    def __init__(self, scheduler_name: str, **kwargs):
        super().__init__()
        lr_schedulers_dict = opt.lr_scheduler.__dict__

        self.scheduler_name = scheduler_name
        self.scheduler_config = kwargs
        self.lr_schedulers_dict = lr_schedulers_dict
        self.scheduler = None

    def on_fit_start(self, *args, **kwargs):
        # The first argument is trainer, the second argument is pl_module
        scheduler = self.lr_schedulers_dict[self.scheduler_name](optimizer=args[0].optimizers[0], **self.scheduler_config)
        self.scheduler = scheduler
    def on_train_epoch_end(self, *args, **kwargs):
        # The first argument is trainer, the second argument is pl_module
        self.scheduler.step()

    def on_train_epoch_start(self, *args, **kwargs):
        #logging part
        number_of_param_groups = len(args[0].optimizers[0].param_groups)
        param_keys = args[1].param_keys
        for i in range(number_of_param_groups):
            args[1].log(f"training/lr_{param_keys[i]}", args[0].optimizers[0].param_groups[i]["lr"])

    @staticmethod
    def _callable_dict(library_dict: Dict) -> List:
        callable_class_list = sorted(name for name in library_dict
                                     if not name.startswith("__")
                                     and callable(library_dict[name]))
        return callable_class_list


class PolynomialLRScheduler(Callback):
    def __init__(self, lambda_lr_scheduler_polynomial_rate, warmup_iterations):
        super().__init__()
        self.lambda_lr_scheduler_polynomial_rate = lambda_lr_scheduler_polynomial_rate
        self.warmup_iterations = warmup_iterations

        lr_schedulers_dict = opt.lr_scheduler.__dict__
        self.lr_schedulers_dict = lr_schedulers_dict
        self.warmup_scheduler = None
        self.scheduler = None
        self.state_dict_loaded = None

    def on_fit_start(self, *args, **kwargs):
        # The first argument is trainer, the second argument is pl_module
        total_step = args[0].estimated_stepping_batches
        optimizer = args[0].optimizers[0]
        self.scheduler = self.lr_schedulers_dict["LambdaLR"](
            optimizer,
            lambda x: (1 - x / (total_step - self.warmup_iterations)) ** self.lambda_lr_scheduler_polynomial_rate)

        self.warmup_scheduler = self.lr_schedulers_dict["LambdaLR"](optimizer, lambda x: (x / self.warmup_iterations))
        if self.state_dict_loaded is not None:
            self.load_state_dict_self(self.state_dict_loaded)

    def on_train_batch_start(self, *args, **kwargs):
        number_of_param_groups = len(args[0].optimizers[0].param_groups)
        param_keys = args[1].param_keys
        for i in range(number_of_param_groups):
            args[1].log(f"training/lr_{param_keys[i]}", args[0].optimizers[0].param_groups[i]["lr"])
    def on_train_batch_end(self, *args, **kwargs):
        # The first argument is trainer, the second argument is pl_module
        if args[0].global_step < self.warmup_iterations:
            self.warmup_scheduler.step()
        else:
            self.scheduler.step()

    def load_state_dict_self(self, state_dict):
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.warmup_scheduler.load_state_dict(state_dict["warmup_scheduler"])

    def on_load_checkpoint(self, *args, **kwargs):
        self.state_dict_loaded = args[2]["callbacks"]["PolynomialLRScheduler"] 

    def state_dict(self) -> Dict[str, Any]:
        return {"scheduler": self.scheduler.state_dict(), "warmup_scheduler": self.warmup_scheduler.state_dict()}