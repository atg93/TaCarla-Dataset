from argparse import ArgumentParser
from tairvision_object_detection.models.bev.cprm.utils.tensorrt_inference import InferenceClass

# Tips for this to work on Orin:
# 1) comment out tairvision/ops/boxes.py line 40
# 2) copy line 89 dummy return tensor to line 93 nms function, that nms fnc doesn't work on Orin
# 3) change tairvision/models/bev/cprm/blocks/centerpoint_head.py line from ['rotate'] to ['circle'] to enable cpu nms
# 3.5) this might be needed for all 'rotate' nmses in the future if multi class trainings are going to be converted
# 4) make sure inserted sys.path is valid for tairbackend
def infer_model(engine_file_path, checkpoint_path, dataroot, version, plot_it, time_it):
    CPRM_Inference = InferenceClass(engine_file_path, checkpoint_path, dataroot, version)
    CPRM_Inference.infer(measure_time=time_it, plot_results=plot_it)

if __name__ == '__main__':
    parser = ArgumentParser(description='CPRM Inference')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='/datasets/nu/nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')
    parser.add_argument('--config-file', default=None, type=str, help='path to checkpoint')
    parser.add_argument('--engine-file', default=None, type=str, help='path to engine')
    parser.add_argument('--plot-it', action='store_true', help='plot results')
    parser.add_argument('--time-it', action='store_true', help='measure runtimes')

    args = parser.parse_args()
    infer_model(args.engine_file, args.checkpoint, args.dataroot, args.version, args.plot_it, args.time_it)

