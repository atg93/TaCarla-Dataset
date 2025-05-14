from pytorch_lightning.strategies.ddp import DDPStrategy, log, reset_seed, rank_zero_only, _init_dist_connection


class MyDDPStrategy(DDPStrategy):
    def __init__(self, init_method, **kwargs):
        super().__init__(**kwargs)
        self.init_method = init_method

    def setup_distributed(self) -> None:
        log.detail(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()
        self.set_world_ranks()
        rank_zero_only.rank = self.global_rank
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout, init_method=self.init_method)
