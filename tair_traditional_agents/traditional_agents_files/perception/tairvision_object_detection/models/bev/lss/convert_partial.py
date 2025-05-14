import copy
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import onnx
import onnxruntime

from tairvision_object_detection.models.bev.lss.training.trainer import TrainingModule
from tairvision_object_detection.models.bev.lss.utils.network import preprocess_batch
from tairvision_object_detection.datasets.nuscenes import prepare_dataloaders
from tairvision_object_detection.models.bev.common.nuscenes.process import (ResizeCropRandomFlipNormalize, FilterClasses,
                                                           get_resizing_and_cropping_parameters)
import time
import numpy as np
import cv2
from tairvision_object_detection.models.bev.lss.visualize import plot_preds_and_gts, LayoutControl

def convert_to_onnx(checkpoint_path, dataroot, version, onnxfile, convert=False):
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cpu')
    trainer.to(device)
    model = trainer.model
    model = model.to(device)

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1

    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.VERSION = version

    augmentation_parameters = get_resizing_and_cropping_parameters(cfg)
    transforms_val = ResizeCropRandomFlipNormalize(augmentation_parameters, enable_random_transforms=False)
    filter_classes = FilterClasses(cfg.DATASET.FILTER_CLASSES)

    _, valloader = prepare_dataloaders(cfg, None, transforms_val, filter_classes)

    batch = next(iter(valloader))

    preprocess_batch(batch, device)
    image = batch['images']
    intrinsics = batch['intrinsics']
    extrinsics = batch['cams_to_lidar']
    view = batch['view']
    future_egomotion = batch['future_egomotion']

    model_wrapper = LiftSplatConvertionWrapper(model)
    with torch.no_grad():
        feats_3d = model_wrapper.lift(image)
        feats_bev = model_wrapper.splat(feats_3d, intrinsics, extrinsics, view, future_egomotion)
        output = model_wrapper.shoot(feats_bev)

    # TODO: Remove this later. This line is just needed to run old models after merge
    output['head3d']['other_regressions'] = torch.cat([output['head3d']['other_regressions'][0],
                                                       output['head3d']['other_regressions'][1],
                                                       output['head3d']['other_regressions'][2]], dim=-1)

    input_names = [['image'], ['feats_bev']]

    output_names, output_shapes = [], []
    output_names.append(['feats_3d'])
    output_shapes.append([feats_3d.shape])

    output_names2, output_shapes2 = [], []
    get_output_names_shapes(output, output_names2, output_shapes2)

    output_names.append(output_names2)
    output_shapes.append(output_shapes2)

    onnxfile = onnxfile.split('.')[0]
    if convert:
        print('Conversion started')
        model.forward = model_wrapper.lift
        torch.onnx.export(trainer.model,
                          args=image,
                          f=onnxfile + '1.onnx', verbose=True, opset_version=14,
                          input_names=input_names[0],
                          output_names=output_names[0]
                          )

        model.forward = model_wrapper.shoot
        torch.onnx.export(trainer.model,
                          args=feats_bev,
                          f=onnxfile + '2.onnx', verbose=True, opset_version=14,
                          input_names=input_names[1],
                          output_names=output_names[1]
                          )

        print('Conversion done')
    else:
        print('Conversion bypassed')

    return output_names, output_shapes


def check_onnx_model(onnxfile):

    print("Loading Onnx file")
    onnx_model = onnx.load(onnxfile)
    print("Checking Onnx file")
    onnx.checker.check_model(onnx_model, full_check=True)
    print("Checking done")


def run_onnx_model_with_onnx_runtime(checkpoint_path, dataroot, version, onnxfile, output_names, output_shapes, device='cuda:0'):
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device_type, device_id = device.split(':')

    trainer.to(torch.device(device))
    model = trainer.model

    cfg = model.cfg
    cfg.GPUS = "[" + device_id + "]"
    cfg.BATCHSIZE = 1
    cfg.N_WORKERS = 0

    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.VERSION = version

    augmentation_parameters = get_resizing_and_cropping_parameters(cfg)
    transforms_val = ResizeCropRandomFlipNormalize(augmentation_parameters, enable_random_transforms=False)
    filter_classes = FilterClasses(cfg.DATASET.FILTER_CLASSES)

    _, valloader = prepare_dataloaders(cfg, None, transforms_val, filter_classes)

    onnxfile = onnxfile.split('.')[0]

    if device_type == 'cpu':
        ort_session1 = onnxruntime.InferenceSession(onnxfile + '1.onnx', providers=['CPUExecutionProvider'])
        ort_session2 = onnxruntime.InferenceSession(onnxfile + '2.onnx', providers=['CPUExecutionProvider'])
    else:
        ort_session1 = onnxruntime.InferenceSession(onnxfile + '1.onnx', providers=[('CUDAExecutionProvider',
                                                                         {'device_id': int(device_id),
                                                                          'arena_extend_strategy': 'kNextPowerOfTwo',
                                                                          'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                                                                          'cudnn_conv_algo_search': 'EXHAUSTIVE',
                                                                          'do_copy_in_default_stream': True,
                                                                          }),
                                                                        'CPUExecutionProvider'
                                                                        ])
        ort_session2 = onnxruntime.InferenceSession(onnxfile + '2.onnx', providers=[('CUDAExecutionProvider',
                                                                         {'device_id': int(device_id),
                                                                          'arena_extend_strategy': 'kNextPowerOfTwo',
                                                                          'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                                                                          'cudnn_conv_algo_search': 'EXHAUSTIVE',
                                                                          'do_copy_in_default_stream': True,
                                                                          }),
                                                                        'CPUExecutionProvider'
                                                                        ])

    ort_session1.disable_fallback()
    ort_session2.disable_fallback()

    binding1 = ort_session1.io_binding()
    binding2 = ort_session2.io_binding()

    layout_control = LayoutControl()

    model_wrapper = LiftSplatConvertionWrapper(model)
    for i, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['images']
        intrinsics = batch['intrinsics']
        intrinsics_inv = torch.inverse(intrinsics).contiguous()
        extrinsics = batch['cams_to_lidar']
        view = batch['view']
        future_egomotion = batch['future_egomotion']

        # Forward pass
        if i == 0:
            with torch.no_grad():
                foo = model_wrapper.model(image, intrinsics, extrinsics, view, future_egomotion)
                # if trainer.model.head2d is not None:
                #     foo['head2d'] = trainer.model.head2d.get_detections(foo['head2d'])
                if model_wrapper.model.head3d is not None:
                    foo['head3d'] = model_wrapper.model.head3d.get_detections(foo['head3d'])

        input_tensors1 = [image]
        input_names1 = [i.name for i in ort_session1.get_inputs()]

        bind_inputs(binding1, input_tensors1, input_names1, device, np_dtype=np.float32)
        output_1 = bind_outputs(binding1, output_names[0], output_shapes[0], device, torch_dtype=torch.float32,
                              np_dtype=np.float32, return_dict=True)



        start = time.time()
        ort_session1.run_with_iobinding(binding1)

        with torch.no_grad():
            feats_bev = model_wrapper.splat(output_1['feats_3d'], intrinsics, extrinsics, view, future_egomotion)


        input_tensors2 = [feats_bev]
        input_names2 = [i.name for i in ort_session2.get_inputs()]
        bind_inputs(binding2, input_tensors2, input_names2, device, np_dtype=np.float32)
        output = bind_outputs(binding2, output_names[1], output_shapes[1], device, torch_dtype=torch.float32,
                              np_dtype=np.float32, return_dict=True)

        ort_session2.run_with_iobinding(binding2)
        stop = time.time()
        print(stop - start)

        with torch.no_grad():
            # if trainer.model.head2d is not None:
            #     output['head2d'] = trainer.model.head2d.get_detections(output['head2d'])
            if model_wrapper.model.head3d is not None:
                # TODO: Remove this later. This line is just needed to run old models after merge
                output['head3d']['other_regressions'] = [output['head3d']['other_regressions'][:,:,[3,2]], output['head3d']['other_regressions'][:,:,[1,0]], output['head3d']['other_regressions'][:,:,4]]
                output['head3d'] = model_wrapper.model.head3d.get_detections(output['head3d'])

        plot_preds_and_gts(cfg, batch, output, filter_classes, layout_control.get_show())

        ch = cv2.waitKey(1)
        if layout_control(ch):
            break

        #print(stop - start)

    print('wait')

    cv2.destroyAllWindows()


def convert_onnx_output_to_torch(onnx_outputs, names, device=None):
    output = {}
    for i, onnx_output in enumerate(onnx_outputs):
        key = names[i]
        if 'head2d' in key or 'head3d' in key:
            head_key, branch_key = key.split('.')
            if head_key not in output.keys():
                output[head_key] = {}
            if device is not None:
                output[head_key][branch_key] = torch.tensor(onnx_output, device=device)
            else:
                output[head_key][branch_key] = onnx_output
        else:
            if device is not None:
                output[key] = torch.tensor(onnx_output, device=device)
            else:
                output[key] = onnx_output
    return output


def get_output_names_shapes(output, output_names, output_shapes, parent_name=None):
    if isinstance(output, dict):
        for k, v in output.items():
            if isinstance(v, dict):
                get_output_names_shapes(v, output_names, output_shapes, parent_name=k)
            elif isinstance(v, list):
                get_output_names_shapes(v, output_names, output_shapes, parent_name=k)
            else:
                if parent_name is not None:
                    output_names.append(parent_name + '.' + k)
                else:
                    output_names.append(k)
                output_shapes.append(v.shape)

    elif isinstance(output, list):
        for i, v in enumerate(output):
            if parent_name is not None:
                output_names.append(parent_name + '.' + str(i))
            else:
                output_names.append(str(i))
            output_shapes.append(v.shape)

    else:
        TypeError('Unsupported output type')


def bind_inputs(binding, inputs, input_names, device, np_dtype=np.float32):
    device_type, device_id = device.split(':')
    for i_input, input_name in enumerate(input_names):
        tensor = inputs[i_input]
        binding.bind_input(name=input_name, device_type=device_type, device_id=int(device_id), element_type=np_dtype,
                           shape=tuple(tensor.shape), buffer_ptr=tensor.data_ptr())


def bind_outputs(binding, output_names, output_shapes, device, torch_dtype=torch.float32, np_dtype=np.float32, return_dict=False):
    tensor_list = []
    device_type, device_id = device.split(':')
    for i_output, output_name in enumerate(output_names):
        tensor = torch.empty(output_shapes[i_output], dtype=torch_dtype, device=device).contiguous()
        binding.bind_output(name=output_name, device_type=device_type, device_id=int(device_id), element_type=np_dtype,
                            shape=tuple(tensor.shape), buffer_ptr=tensor.data_ptr())

        tensor_list.append(tensor)

    if return_dict:
        tensor_dict = convert_onnx_output_to_torch(tensor_list, output_names, device=None)
        return tensor_dict
    else:
        return tensor_list


class LiftSplatConvertionWrapper(object):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.model.head2d = None

    def lift(self, image):

        # Only process features from the past and present
        image = self.filter_input(image)

        # Getting features from 2d backbone and lifting to 3d
        feats_3d, feats_2d = self.model.get_features(image)

        return feats_3d

    def splat(self, feats_3d, intrinsics, extrinsics, view, future_egomotion):

        intrinsics = self.filter_input(intrinsics)
        extrinsics = self.filter_input(extrinsics)
        view = self.filter_input(view)
        future_egomotion = self.filter_input(future_egomotion)

        feats_bev = self.model.calculate_bev_features(feats_3d, intrinsics, extrinsics, view)

        return feats_bev

    def shoot(self, feats_bev):

        # Temporal model
        states = self.model.temporal_model(feats_bev)

        # Predict bird's-eye view outputs
        feats_dec = self.model.decoder(states)

        # Get outputs for available heads using decoder features and 2d features
        output = self.model.get_head_outputs(feats_dec, None)

        return output

    def filter_input(self, x):

        x = x[:, :self.model.receptive_field].contiguous()

        return x


if __name__ == '__main__':
    parser = ArgumentParser(description='LSS to Onnx Conversion')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='/datasets/nu/nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')
    parser.add_argument('--onnxfile', default='/workspace/ok21/exps/onnx/lss_new.onnx', type=str, help='path to the onnx file')
    parser.add_argument('--convert', action="store_true", help='perform Onnx conversion')
    parser.add_argument('--device', default='cuda:0', type=str, help='device to run onnx model')

    args = parser.parse_args()

    output_names, output_shapes = convert_to_onnx(args.checkpoint, args.dataroot, args.version, args.onnxfile, args.convert)

    # check_onnx_model(args.onnxfile)
    run_onnx_model_with_onnx_runtime(args.checkpoint, args.dataroot, args.version, args.onnxfile, output_names, output_shapes, args.device)


