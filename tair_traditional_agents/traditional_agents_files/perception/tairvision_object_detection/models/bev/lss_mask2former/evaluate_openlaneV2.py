from tairvision.models.bev.lss.evaluate import EvaluationInterface
from argparse import ArgumentParser
import torch
import warnings
from iso3166 import countries
from tqdm import tqdm
from tairvision.datasets.openlane_v2 import io
from tairvision.models.bev.lss.utils.network import preprocess_batch, move_to
from tairvision.models.bev.lss.training.metrics import IntersectionOverUnion, PanopticMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tairvision.models.bev.lss_mask2former.train_openlanev2 import TrainingInterfaceMask2formerOpenLaneV2
from tairvision.models.bev.lss_mask2former.utils_sub.evaluate_openlaneV2_functions import calculate_and_print_results, predict_2d, update_result_dictionary, \
update_other_metrics, predict_relations, predict_relations_sinkhorn_fixed, predict_centerlines_bezier, predict_centerlines, \
predict_centerlines_bezier_mask_fusion, test_gt_performance, apply_chamfer_nms
from tairvision.models.bev.lss_dab.lss_transformer_utils.postprocess import post_process_softmax_lane


class EvaluationInterfaceMask2FormerOpenLaneV2(EvaluationInterface):
    def __init__(self, *args, **kwargs):
        super(EvaluationInterfaceMask2FormerOpenLaneV2, self).__init__(*args, **kwargs)

        self.filtered_lc_implementation = False
        self.filtered_te_implementaion = False

    @staticmethod
    def _import_training_module():
        from tairvision.models.bev.lss_mask2former.training.trainer_openlaneV2 import TrainingModuleMask2FormerOpenLaneV2
        return TrainingModuleMask2FormerOpenLaneV2
    
    def load_cfg_settings(self, cfg):
        super().load_cfg_settings(cfg)
        cfg.BATCHSIZE = self.args_dict["batchsize"]
        cfg.DATASET.SAMPLING_RATIO = self.args_dict["dataset_sampling_ratio"]
        h, w = self.module.cfg.FRONT_CENTER_IMAGE.ORIGINAL_HEIGHT, self.module.cfg.FRONT_CENTER_IMAGE.ORIGINAL_WIDTH
        self.orig_target_sizes = torch.tensor([h, w]).to(self.device)
        self.batch_size = cfg.BATCHSIZE
        return cfg
    
    def get_loaders(self, cfg):
        args = self.args_dict
        _, valloader, testloader, _ = TrainingInterfaceMask2formerOpenLaneV2.get_loaders(cfg, return_testloader=True)
        if args["split"] == "test":
            assert args["test"], "Test set can only be used for evaluation in test mode"
            assert cfg.DATASET.SAMPLING_RATIO == 1, "subset slide should be 1 for test mode"
            loader = testloader
            print("Test set is used for evaluation")
            submission_mode = "test"
        elif args["split"] == "val":
            loader = valloader
            print("Validation set is used for evaluation")
            submission_mode = "val"
        else:
            raise ValueError("Split should be either test or val")
        
        self.submission_mode = submission_mode
        return loader, None
    
    def _init_metrics(self):
        self._init_result_dict()
        self._set_eval_frames()
        self._sef_eval_ranges()
        self._init_dynamic_metrics()
        self._init_det2d_metrics()

    def _sef_eval_ranges(self):
        #TODO, make this generic
        self.evaluation_ranges = {
            '30x30': [70, 130, 22, 82],
            '100x52': [0, 200, 0, 104]
        }

    def _init_result_dict(self):
        results_dict_openlanev2 = {
        'method': 'evalai_check',
        'e-mail': 'esat.kalfaoglu@metu.edu.tr',
        'institution / company': 'Metu / Togg',
        "country / region": countries.get("TR").name,
        "authors": ["muhammet esat kalfaoglu", "halil ibrahim ozturk", "ozsel kilinc", "alptekin temizel"], 
        'results': {},
        }
        self.results_dict_openlanev2 = results_dict_openlanev2

    def _init_det2d_metrics(self):
        self.metric_ap = None
        if self.module.model.head2d is not None:
            metric_ap = MeanAveragePrecision()
            self.metric_ap = metric_ap


    def _init_dynamic_metrics(self):
        panoptic_metrics = {}
        iou_metrics = {}
        n_classes = 5
        for key in self.evaluation_ranges.keys():
            panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).to(self.device)
            iou_metrics[key] = IntersectionOverUnion(n_classes).to(self.device)

        self.panoptic_metrics = panoptic_metrics
        self.iou_metrics = iou_metrics

    def predict_2d(self, output, pred_list_batch):
        if self.module.model.head2d is not None:
            predict_2d(
                self.module, output, pred_list_batch, self.batch_size, self.orig_target_sizes,
                filtered_te_implementaion=self.filtered_te_implementaion, 
                threshold_2d=self.args_dict["threshold_2d"], 
                enable_relationship_queries=self.args_dict["enable_relationship_queries"], 
                enable_topk_selection=self.args_dict["topk_te_selection"], 
                num_topk=self.args_dict["topk_te_element_queries"]
            )
    
    def predict_centerlines(self, output, batch, lane_centerline_list_pred_batch):
        kwargs = {
            "object_mask_threshold": self.module.model.centerline_head.object_mask_threshold,
            "overlap_threshold": self.args_dict["overlap_threshold"], 
            "number_of_classes": self.module.model.centerline_head.number_of_classes,
            "threshold1": self.args_dict["threshold1"],
            "threshold2": self.args_dict["threshold2"],
            "overlap_threshold2": self.args_dict["overlap_threshold2"]
            }
        
        if self.args_dict["enable_after_centerline_pred"]:
            post_out_second_stage = post_process_softmax_lane(
                output["centerline_after_relation"], 
                **kwargs
            )
            output["post_out_lc"] = post_out_second_stage
        else:
            post_out_first_stage = post_process_softmax_lane(
                output, 
                **kwargs
            )
            output["post_out_lc"] = post_out_first_stage
        

        if self.args_dict["bezier"]:
            predict_centerlines_bezier(output, batch, lane_centerline_list_pred_batch)
        elif self.args_dict["bezier_fusion"]:
            predict_centerlines_bezier_mask_fusion(output, batch, lane_centerline_list_pred_batch)
        else:
            predict_centerlines(output, batch, lane_centerline_list_pred_batch)

    def predict_relations(self, output, score_list_lclc_batch, score_list_lcte_batch):
        if self.module.model.optimal_transport_lclc is not None:
            predict_relations(
                self.module, output, score_list_lclc_batch, score_list_lcte_batch,
                lclc_sinkhorn_threshold = self.args_dict["lclc_sinkhorn_threshold"], 
                lcte_sinkhorn_threshold = self.args_dict["lcte_sinkhorn_threshold"], 
                enable_after_centerline_pred = self.args_dict["enable_after_centerline_pred"], 
                enable_relationship_queries = self.args_dict["enable_relationship_queries"], 
                enable_distance_prior = self.args_dict["enable_distance_prior"]
            )

    def eval(self):
        result_key_list = []
        for loader_index, batch in enumerate(tqdm(self.valloader)):
            result_key_list = [tuple(identifier[0]) for identifier in batch['data_identifier_list']]
            preprocess_batch(batch, self.device, filtered_keys=
                ["centerlines_list", "targets2d", "front_view_images", "data_identifier_list", "lcte_list", "lclc_list", "ordered_attribute_list", "bezier_list"]
            )


            if self.args_dict["split"] != "test":
                batch["bezier_list"] = move_to(batch["bezier_list"], device=self.device)
                batch["front_view_images"] = move_to(batch["front_view_images"], device=self.device)

            # targets = self.module.prepare_targets(batch)

            pred_list_batch = []
            lane_centerline_list_pred_batch = []
            score_list_lclc_batch = []
            score_list_lcte_batch = []
            
            with torch.no_grad():
                output = self.module.forward(batch)

            self.predict_2d(output=output, pred_list_batch=pred_list_batch)
            self.predict_centerlines(output=output, batch=batch, lane_centerline_list_pred_batch=lane_centerline_list_pred_batch)
            
            if self.args_dict["enable_nms"]:
                apply_chamfer_nms(output, self.batch_size, self.args_dict["chamfer_nms_distance"], lane_centerline_list_pred_batch)

            self.predict_relations(output=output, 
                score_list_lclc_batch=score_list_lclc_batch, 
                score_list_lcte_batch=score_list_lcte_batch
                )
            
            update_result_dictionary(self.results_dict_openlanev2, result_key_list, lane_centerline_list_pred_batch,
                                pred_list_batch, score_list_lclc_batch, score_list_lcte_batch)
            
            if not self.args_dict["test"]:
                update_other_metrics(self.module, output, batch, self.metric_ap, self.panoptic_metrics, self.iou_metrics, self.device, self.evaluation_ranges)

        if self.args_dict["test"]:
            io.pickle_dump(f'results_{self.cfg.TAG.lower()}_{self.module.cfg.DATASET.VERSION}.pkl', self.results_dict_openlanev2)
            
        if not self.args_dict["test"]:
            results_openlanev2 = calculate_and_print_results(
                self.module, self.results_dict_openlanev2, self.metric_ap, self.panoptic_metrics, self.iou_metrics, self.evaluation_ranges
                )

            return results_openlanev2
        else:
            return None



def get_parser():
    parser = ArgumentParser(description='LSS evaluation')
    parser.add_argument('--checkpoint', default='./lss.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='/datasets/OpenLane-V2', type=str, help='path to the dataset')
    parser.add_argument('--version', default='data_dict_subset_A', type=str, choices=['data_dict_subset_A', 'data_dict_sample'],
                        help='dataset version')
    parser.add_argument('--resultdir', default=None, type=str, help='path to result directory')
    parser.add_argument('--batchsize', type=int, help='batch size for evaluation', default=4, required=False)  
    parser.add_argument('--split', default='val', type=str, choices=['test', 'val'])
    parser.add_argument('--test', dest='test', help='Test for the evalai submission',
                        action='store_true', default=False)
    parser.add_argument('--visualize', dest='visualize', help='activate visualization',
                        action='store_true', default=False)
    parser.add_argument('--bezier', dest='bezier', help='enable bezier curve fitting',
                        action='store_true', default=False)
    parser.add_argument('--bezier-fusion', dest='bezier_fusion', help='enable bezier curve fitting with fusion',
                        action='store_true', default=False)
    parser.add_argument('--enable-after-centerline-pred', 
                        dest='enable_after_centerline_pred', 
                        help='considers the centerline prediction in the second stage after relationship part',
                        action='store_true', default=False)
    parser.add_argument('--enable-relationship-queries', dest='enable_relationship_queries', help='enable the queries from the Joint Relation Improver block',
                        action='store_true', default=False)
    parser.add_argument('--topk-te-selection', dest='topk_te_selection', help='enable the topk selection for TE as dab-detr post-process manner',
                        action='store_true', default=False)
    parser.add_argument('--enable-nms', dest='enable_nms', help='enamble nms filtering',
                        action='store_true', default=False)
    parser.add_argument('--enable-distance-prior', dest='enable_distance_prior', 
                        help='Increase the relation score of lclc when the distance is low',
                        action='store_true', default=False)
    parser.add_argument('--dataset-sampling-ratio', type=int,
                        help='dataset sampling ratio in order to take fast inference', default=1, required=False)
    parser.add_argument('--overlap-threshold', type=float,
                        help='overlap threshold 1 for post-processing mask2former', default=0.0, required=False)
    parser.add_argument('--overlap-threshold2', type=float,
                        help='overlap threshold 2 for post-processing mask2former', default=0.0, required=False)
    parser.add_argument('--threshold1', type=float, help='First threshold', default=0.3, required=False)
    parser.add_argument('--threshold2', type=float, help='Second threshold', default=0.2, required=False)
    parser.add_argument('--threshold-2d', type=float, help='2D detection threshold', default=0.00, required=False)
    parser.add_argument('--lclc-sinkhorn-threshold', type=float, 
                        help='lclc relationship threshold', default=0.07, required=False)
    parser.add_argument('--lcte-sinkhorn-threshold', type=float, 
                        help='lclc relationship threshold', default=0.01, required=False)    
    parser.add_argument('--topk-te-element-queries', type=int, 
                        help='topk selection among the prediction of 2D detector', default=100, required=False)    
    parser.add_argument('--chamfer-nms-distance', type=int, 
                        help='nms distance for the possible nms implementation', default=3, required=False)     
    return parser       


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["checkpoint_path"] = args_dict.pop("checkpoint")

    evaluation_interface = EvaluationInterfaceMask2FormerOpenLaneV2(**args_dict)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        args_dict = vars(args)
        results_openlanev2 = evaluation_interface.eval()
        new_result_dict = {}
        for key, value in results_openlanev2["OpenLane-V2 Score"].items():
            if key == "score":
                key = "OLS"
            new_result_dict[key] = value
        
        new_result_dict["f1_score"] = results_openlanev2['F-Score for 3D Lane']["score"]
