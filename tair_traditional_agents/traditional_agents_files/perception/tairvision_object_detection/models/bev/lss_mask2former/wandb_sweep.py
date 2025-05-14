import wandb
import warnings
from tairvision.models.bev.lss_mask2former.evaluate_openlaneV2 import EvaluationInterfaceMask2FormerOpenLaneV2, get_parser


checkpoint_path = "/home/ek21/git/tairvision/checkpoints/epoch=2-step=1686.ckpt"
dataset_sampling_ratio = 20

sweep_configuration = {
    'method': 'random',
    'name': f'ratio_{dataset_sampling_ratio}_chechpoint_{checkpoint_path}',
    'metric': {
        'goal': 'maximize',
        'name': 'OLS'
        },
    'parameters': {
        # 'overlap_threshold': {'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        # 'threshold1': {'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}, 
        # 'threshold2': {'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}, 
        # 'overlap_threshold2': {'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        "lclc_sinkhorn_threshold": {"values": [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20]}, 
        "lcte_sinkhorn_threshold": {"values": [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20]}
     },
}

def eval_wrapper(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        
        parser = get_parser()
        args = parser.parse_args()
        args_dict = vars(args)
        args_dict["checkpoint_path"] = checkpoint_path
        args_dict["lclc_sinkhorn_threshold"] = config.lclc_sinkhorn_threshold
        args_dict["lcte_sinkhorn_threshold"] = config.lcte_sinkhorn_threshold
        args_dict["dataset_sampling_ratio"] = dataset_sampling_ratio

        evaluation_interface = EvaluationInterfaceMask2FormerOpenLaneV2(**args_dict)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results_openlanev2 = evaluation_interface.eval()
            new_result_dict = {}
            for key, value in results_openlanev2["OpenLane-V2 Score"].items():
                if key == "score":
                    key = "OLS"
                new_result_dict[key] = value
        
            new_result_dict["f1_score"] = results_openlanev2['F-Score for 3D Lane']["score"]

        wandb.log(new_result_dict)

sweep_id = wandb.sweep(sweep_configuration, project="openlaneV2_sweep", entity="tair-lane")
wandb.agent(sweep_id, eval_wrapper)
