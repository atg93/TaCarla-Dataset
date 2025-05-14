from argparse import ArgumentParser
from tqdm import tqdm
import torch
import onnx
import onnxruntime

from tairvision_object_detection.models.bev.cprm.training.trainer import TrainingModule
from tairvision_object_detection.models.bev.lss.utils.network import preprocess_batch
from tairvision_object_detection.datasets.nuscenes import prepare_dataloaders
from tairvision_object_detection.models.bev.common.nuscenes.process import (ResizeCropRandomFlipNormalize, FilterClasses,
                                                           get_resizing_and_cropping_parameters)
import time
import numpy as np
import cv2
from tairvision_object_detection.models.bev.cprm.visualize import plot_preds_and_gts, LayoutControl


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
    radar_data = batch['radar_data']

    with torch.no_grad():
        output = model(image, intrinsics, extrinsics, view, future_egomotion, radar_data=radar_data, inverse_view=view.inverse())

    input_names = ['image', 'intrinsics', 'extrinsics', 'view', 'future_egomotion', 'radar_data', 'inverse_view']
    output_names, output_shapes = [], []
    get_output_names_shapes(output, output_names, output_shapes)
    if convert:
        print('Conversion started')
        torch.onnx.export(trainer.model,
                          args=(image, intrinsics, extrinsics, view, future_egomotion, radar_data, view.inverse()),
                          f=onnxfile, verbose=True, opset_version=14,
                          input_names=input_names,
                          output_names=output_names
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
        elif 'cphead' in key:
            head_key, _, branch_key = key.split('.')
            if head_key not in output.keys():
                output[head_key] = []
            if len(output[head_key]) == 0:
                if device is not None:
                    output[head_key].append({branch_key: torch.tensor(onnx_output, device=device)})
                else:
                    output[head_key].append({branch_key: onnx_output})
            else:
                if device is not None:
                    output[head_key][0][branch_key] = torch.tensor(onnx_output, device=device)
                else:
                    output[head_key][0][branch_key] = onnx_output


        else:
            if device is not None:
                output[key] = torch.tensor(onnx_output, device=device)
            else:
                output[key] = onnx_output
    return output

def run_onnx_model_with_onnx_runtime(checkpoint_path, dataroot, version, onnxfile, output_names, output_shapes,
                                     device='cuda:0'):
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

    if device_type == 'cpu':
        ort_session = onnxruntime.InferenceSession(onnxfile, providers=['CPUExecutionProvider'])
    else:
        ort_session = onnxruntime.InferenceSession(onnxfile, providers=[('CUDAExecutionProvider',
                                                                         {'device_id': int(device_id),
                                                                          'arena_extend_strategy': 'kNextPowerOfTwo',
                                                                          'gpu_mem_limit': 22 * 1024 * 1024 * 1024,
                                                                          'cudnn_conv_algo_search': 'EXHAUSTIVE',
                                                                          'do_copy_in_default_stream': True,
                                                                          }),
                                                                        'CPUExecutionProvider'
                                                                        ])

    ort_session.disable_fallback()
    binding = ort_session.io_binding()
    layout_control = LayoutControl()
    for i, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['images']
        intrinsics = batch['intrinsics']
        extrinsics = batch['cams_to_lidar']
        view = batch['view']
        future_egomotion = batch['future_egomotion']
        radar_data = batch['radar_data']



        # Forward pass
        if i == 0:
            with torch.no_grad():
                foo = trainer.model(image, intrinsics, extrinsics, view, future_egomotion, radar_data=radar_data, inverse_view=view.inverse())
                if device_type == "cpu":
                    foo['cphead'][0]['reg'] = foo['cphead'][0]['reg'].to('cuda')
                    foo['cphead'][0]['height'] = foo['cphead'][0]['height'].to('cuda')
                    foo['cphead'][0]['dim'] = foo['cphead'][0]['dim'].to('cuda')
                    foo['cphead'][0]['rot'] = foo['cphead'][0]['rot'].to('cuda')
                    foo['cphead'][0]['heatmap'] = foo['cphead'][0]['heatmap'].to('cuda')
                if trainer.model.head2d is not None:
                    foo['head2d'] = trainer.model.head2d.get_detections(foo['head2d'])
                if trainer.model.head3d is not None:
                    foo['head3d'] = trainer.model.head3d.get_detections(foo['head3d'])
                if trainer.model.cphead is not None:
                    foo['cphead'] = trainer.model.cphead.get_detections(foo['cphead'])
        input_tensors = [image, intrinsics, extrinsics, radar_data, view.inverse().contiguous()]
        # input_names = ['image', 'intrinsics', 'extrinsics', 'view']
        input_names = [i.name for i in ort_session.get_inputs()]

        bind_inputs(binding, input_tensors, input_names, device, np_dtype=np.float32)
        output = bind_outputs(binding, output_names, output_shapes, device, torch_dtype=torch.float32,
                              np_dtype=np.float32, return_dict=True)

        start = time.time()
        ort_session.run_with_iobinding(binding)
        stop = time.time()
        print(stop - start)

        with torch.no_grad():
            if device_type == "cpu":
                output['cphead'][0]['reg'] = output['cphead'][0]['reg'].to('cuda')
                output['cphead'][0]['height'] = output['cphead'][0]['height'].to('cuda')
                output['cphead'][0]['dim'] = output['cphead'][0]['dim'].to('cuda')
                output['cphead'][0]['rot'] = output['cphead'][0]['rot'].to('cuda')
                output['cphead'][0]['heatmap'] = output['cphead'][0]['heatmap'].to('cuda')
            if trainer.model.head2d is not None:
                output['head2d'] = trainer.model.head2d.get_detections(output['head2d'])
            if trainer.model.head3d is not None:
                output['head3d'] = trainer.model.head3d.get_detections(output['head3d'])
            if trainer.model.cphead is not None:
                output['cphead'] = trainer.model.cphead.get_detections(output['cphead'])

        plot_preds_and_gts(cfg, batch, output, filter_classes, layout_control.get_show())

        ch = cv2.waitKey(1)
        if layout_control(ch):
            break

        # print(stop - start)

    print('wait')

    cv2.destroyAllWindows()

def get_output_names_shapes(output, output_names, output_shapes, parent_name=None):
    if isinstance(output, dict):
        for k, v in output.items():
            if parent_name:
                par_nam = parent_name + '.' + k
            else:
                par_nam = k
            if isinstance(v, dict):
                get_output_names_shapes(v, output_names, output_shapes, parent_name=par_nam)
            elif isinstance(v, list):
                get_output_names_shapes(v, output_names, output_shapes, parent_name=par_nam)
            else:
                if parent_name is not None:
                    output_names.append(par_nam)
                else:
                    output_names.append(k)
                output_shapes.append(v.shape)

    elif isinstance(output, list):
        for i, v in enumerate(output):
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if parent_name:
                        par_nam = parent_name + '.' + str(i) + '.' + kk
                    else:
                        par_nam = str(i) + '.' + kk
                    output_names.append(par_nam)
                    output_shapes.append(vv.shape)
            else:
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


def bind_outputs(binding, output_names, output_shapes, device, torch_dtype=torch.float32, np_dtype=np.float32,
                 return_dict=False):
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



def create_inverse_matrix(cfg):
    # xyz is B x N x 3, in mem coordinates
    # transforms mem coordinates into ref coordinates
    XMIN, XMAX, _ = cfg.LIFT.X_BOUND
    ZMIN, ZMAX, _ = cfg.LIFT.Y_BOUND
    YMIN, YMAX, _ = cfg.LIFT.Z_BOUND
    xb = cfg.LIFT.X_BOUND
    yb = cfg.LIFT.Y_BOUND
    zb = cfg.LIFT.Z_BOUND
    X = (xb[1] - xb[0]) / xb[2]
    Z = (yb[1] - yb[0]) / yb[2]  # Y turns into Z in reverse map jargon.
    Y = 8  #TODO: THIS CAN CHANGE LATER BEWARE  # Y turns into Z in reverse map jargon.
    device = 'cpu'
    B = 1
    vox_size_X = (XMAX - XMIN) / float(X)
    vox_size_Y = (YMAX - YMIN) / float(Y)
    vox_size_Z = (ZMAX - ZMIN) / float(Z)

    # translation
    # (this makes the left edge of the leftmost voxel correspond to XMIN)
    center_T_ref = torch.eye(4, device=torch.device(device)).view(1, 4, 4).repeat([B, 1, 1])
    center_T_ref[:, 0, 3] = -XMIN - vox_size_X / 2.0
    center_T_ref[:, 1, 3] = -YMIN - vox_size_Y / 2.0
    center_T_ref[:, 2, 3] = -ZMIN - vox_size_Z / 2.0

    # scaling
    # (this makes the right edge of the rightmost voxel correspond to XMAX)
    mem_T_center = torch.eye(4, device=torch.device(device)).view(1, 4, 4).repeat([B, 1, 1])
    mem_T_center[:, 0, 0] = 1. / vox_size_X
    mem_T_center[:, 1, 1] = 1. / vox_size_Y
    mem_T_center[:, 2, 2] = 1. / vox_size_Z
    mem_T_ref = torch.matmul(mem_T_center, center_T_ref)

    # note safe_inverse is inapplicable here,
    # since the transform is nonrigid
    ref_T_mem = mem_T_ref.inverse()
    # ENTER MAGICAL FLIP, ALOHOMORA
    temp = ref_T_mem.clone()
    ref_T_mem[0, 1, :] = -temp[0, 1, :]
    ref_T_mem[0, 2, :] = -temp[0, 2, :]
    return ref_T_mem


if __name__ == '__main__':
    parser = ArgumentParser(description='LSS to Onnx Conversion')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='/datasets/nu/nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')
    parser.add_argument('--onnxfile', default='/workspace/ct22/experiments/onnx/lss_new.onnx', type=str,
                        help='path to the onnx file')
    parser.add_argument('--convert', action="store_true", help='perform Onnx conversion')
    parser.add_argument('--device', default='cuda:0', type=str, help='device to run onnx model')

    args = parser.parse_args()

    output_names, output_shapes = convert_to_onnx(args.checkpoint, args.dataroot, args.version, args.onnxfile,
                                                  args.convert)

    check_onnx_model(args.onnxfile)
    run_onnx_model_with_onnx_runtime(args.checkpoint, args.dataroot, args.version, args.onnxfile, output_names,
                                     output_shapes, args.device)