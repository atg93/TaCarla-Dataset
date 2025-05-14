import time
import torch
import cv2

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from tqdm import tqdm

from tairvision.models.bev.cprm.training.trainer import TrainingModule
from tairvision.models.bev.lss.utils.network import preprocess_batch
from tairvision.models.bev.cprm.utils.visualization import plot_preds_and_gts, LayoutControl
from tairvision.datasets.nuscenes import prepare_dataloaders
from tairvision.models.bev.common.nuscenes.process import (ResizeCropRandomFlipNormalize, FilterClasses,
                                                           get_resizing_and_cropping_parameters)


class InferenceClass(object):
    def _load_plugins(self):
        trt.init_libnvinfer_plugins(self.TRT_LOGGER, '')

    def _load_engine(self):
        print("Reading engine from file {}".format(self.engine_file_path))
        f = open(self.engine_file_path, "rb")
        runtime = trt.Runtime(self.TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def _create_context(self):
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            name = str(binding)
            size = trt.volume(shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.output_shapes.append(shape)
                self.output_names.append(name)
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        context = self.engine.create_execution_context()
        return context

    def _load_model(self):
        trainer = TrainingModule.load_from_checkpoint(self.checkpoint_path, strict=True)
        print(f'Loaded weights from \n {self.checkpoint_path}')
        trainer.eval()
        device = torch.device('cpu')
        trainer.to(device)
        return trainer

    def _get_dataloader(self):
        model = self.trainer.model
        self.cfg = model.cfg
        self.cfg.GPUS = "[0]"
        self.cfg.BATCHSIZE = 1

        self.cfg.DATASET.DATAROOT = self.dataroot
        self.cfg.DATASET.VERSION = self.version

        augmentation_parameters = get_resizing_and_cropping_parameters(self.cfg)
        transforms_val = ResizeCropRandomFlipNormalize(augmentation_parameters, enable_random_transforms=False)
        self.filter_classes = FilterClasses(self.cfg.DATASET.FILTER_CLASSES)
        _, valloader = prepare_dataloaders(self.cfg, None, transforms_val, self.filter_classes)
        return valloader

    def __init__(self, engine_file_path, checkpoint_path, dataroot, version):
        """Initialize model, data loader, TensorRT plugins, engine and context."""
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.output_shapes = []
        self.output_names = []
        self.bindings = []
        self.engine_file_path = engine_file_path
        self.checkpoint_path = checkpoint_path
        self.dataroot = dataroot
        self.version = version
        self.device = 'cpu'  # This must be the case o/w torch.cuda and pycuda operations collide.
        self.TRT_LOGGER = trt.Logger(trt.Logger.INFO)  # Alternatively trt.Logger.VERBOSE

        self._load_plugins()
        self.engine = self._load_engine()
        self.stream = cuda.Stream()
        self.context = self._create_context()
        self.trainer = self._load_model()
        self.valloader = self._get_dataloader()

    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs

    def get_data(self, batch):
        """override this fnc according to the data"""
        preprocess_batch(batch, self.device)
        image = batch['images'].cpu()
        intrinsics = batch['intrinsics'].cpu()
        extrinsics = batch['cams_to_lidar'].cpu()
        view = batch['view'].cpu()
        future_egomotion = batch['future_egomotion'].cpu()
        radar_data = batch['radar_data'].cpu()
        return image, intrinsics, extrinsics, view, future_egomotion, radar_data

    def infer(self, measure_time=True, plot_results=True):
        """Override this fnc according to the model and data
        Reads data from loader, passes into device, executes the model, collects outputs"""
        layout_control = LayoutControl()
        for i, batch in enumerate(tqdm(self.valloader)):
            image, intrinsics, extrinsics, view, future_egomotion, radar_data = self.get_data(batch)

            # Forward pass to build internal model parameters for postprocessing.
            if i == 0:
                with torch.no_grad():
                    foo = self.trainer.model(image, intrinsics, extrinsics, view, future_egomotion,
                                             radar_data=radar_data, inverse_view=view.inverse())
                    foo['head2d'] = self.trainer.model.head2d.get_detections(foo['head2d'])

            if measure_time:
                # using torch.cuda.Event to measure time is not a good idea since torch.cuda and
                # pycuda (Tensorrt) operations may collide.
                start = time.time()

            self.host_inputs[0] = np.ascontiguousarray(image)
            self.host_inputs[1] = np.ascontiguousarray(intrinsics)
            self.host_inputs[2] = np.ascontiguousarray(extrinsics)
            self.host_inputs[3] = np.ascontiguousarray(radar_data)
            self.host_inputs[4] = np.ascontiguousarray(view.inverse())

            # send data to device
            for ins in range(len(self.host_inputs)):
                cuda.memcpy_htod_async(self.cuda_inputs[ins], self.host_inputs[ins], self.stream)

            # run the model
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            # collect outputs from device
            for outs in range(len(self.host_outputs)):
                cuda.memcpy_dtoh_async(self.host_outputs[outs], self.cuda_outputs[outs], self.stream)

            self.stream.synchronize()

            if measure_time:
                stop = time.time()
                print("Elapsed time during model (in seconds):", stop - start)

                start = time.time()
            output_dict = self.postprocess_output()
            if measure_time:
                stop = time.time()
                print("Elapsed time during postprocessing (in seconds):", stop - start)

            if plot_results:
                plot_preds_and_gts(self.cfg, batch, output_dict, self.filter_classes, layout_control.get_show())
                ch = cv2.waitKey(1)
                if layout_control(ch):
                    break
                if ch == 32:
                    freeze = True
                while (freeze):
                    ch = cv2.waitKey(1)
                    if ch == 32:
                        freeze = False
        if plot_results:
            cv2.destroyAllWindows()

    def postprocess_output(self):
        """Override this fnc according to converted model and postprocessing steps
        Reshapes one dimensional device outputs into intended N-dimensional shapes
        Passes desired outputs thru model postprocessing steps (nms etc)

        This part of the code is directly compatible with convert.py in cprm folder."""
        with torch.no_grad():
            unique_name_list = []
            output_dict = {}
            for ix, name in enumerate(self.output_names):
                split_names = name.split('.')
                if len(split_names) == 2:  # for example head2d.centerness
                    if split_names[0] not in unique_name_list:
                        unique_name_list.append(split_names[0])
                        output_dict[split_names[0]] = {}
                        output_dict[split_names[0]][split_names[1]] = torch.reshape(
                            torch.tensor(self.host_outputs[ix], device=self.device), tuple(self.output_shapes[ix]))
                    else:
                        output_dict[split_names[0]][split_names[1]] = torch.reshape(
                            torch.tensor(self.host_outputs[ix], device=self.device), tuple(self.output_shapes[ix]))
                elif len(split_names) == 3:  # for example cphead.0.reg
                    if split_names[0] not in unique_name_list:
                        unique_name_list.append(split_names[0])
                        output_dict[split_names[0]] = [{}]
                        output_dict[split_names[0]][0][split_names[2]] = torch.reshape(
                            torch.tensor(self.host_outputs[ix], device=self.device), tuple(self.output_shapes[ix]))
                    else:
                        if int(split_names[1]) > len(output_dict[split_names[0]]) + 1:
                            output_dict[split_names[0]].append({})
                        output_dict[split_names[0]][int(split_names[1])][split_names[2]] = torch.reshape(
                            torch.tensor(self.host_outputs[ix], device=self.device), tuple(self.output_shapes[ix]))
                else:
                    output_dict[name] = torch.reshape(torch.tensor(self.host_outputs[ix], device=self.device),
                                                      tuple(self.output_shapes[ix]))
            output_dict['head2d'] = self.trainer.model.head2d.get_detections(output_dict['head2d'])
            output_dict['cphead'] = self.trainer.model.cphead.get_detections(output_dict['cphead'])
        return output_dict
