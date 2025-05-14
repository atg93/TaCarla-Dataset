import time

import numpy as np
import cv2
import matplotlib
import pycuda.gpuarray
import torch

try:
    import carla
except:
    print("carla is not imported")

import random
import queue
from typing import Tuple, List, Optional, Union, Dict
from sklearn.metrics import mean_squared_error
try:
    import skcuda.misc as sk
except:
    print("skcuda is not imported")


class PolynomialRegression(object):
    def __init__(self, degree=3, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))


def to_numpy(input_tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Converts torch tensor to numpy array by proper operations
    :param input_tensor:
    :return:
    """
    if isinstance(input_tensor, torch.Tensor):
        input_numpy = input_tensor.detach().cpu().numpy()
    elif isinstance(input_tensor, np.ndarray):
        input_numpy = input_tensor
    else:
        raise ValueError("Not a tensor or numpy array")
    return input_numpy


def create_color_space(number_of_colors, repeat_number, add_zero=True):
    color_chart = []
    if add_zero:
        rgb = matplotlib.colors.hsv_to_rgb([
            0, 0, 0
        ])
        rgb = rgb * 255
        color_chart.append(rgb)
    for i in range(repeat_number):
        for class_id in range(number_of_colors):
            # get different colors for the edges
            rgb = matplotlib.colors.hsv_to_rgb([
                class_id / number_of_colors, 1.0, 1.0
            ])
            rgb = rgb * 255
            color_chart.append(rgb)
    return color_chart


def preprocess_image(sample_image, cv2_dim, mean, std, preserve_ratio=False):
    if preserve_ratio:
        ratio = sample_image.shape[1] / sample_image.shape[0]
        width = int(cv2_dim[1] * ratio)
        cv2_dim = (width, cv2_dim[1])
    # else:
    #     ratio = cv2_dim[0] / cv2_dim[1]
    #     padding = int((sample_image.shape[1] * ratio - sample_image.shape[0]) / 2)
    #     sample_image = cv2.copyMakeBorder(
    #         sample_image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, None, value=(125, 125, 125)
    #     )

    sample_image_resized = cv2.resize(sample_image, cv2_dim, interpolation=cv2.INTER_NEAREST)
    sample_image_rgb = cv2.cvtColor(sample_image_resized, cv2.COLOR_BGR2RGB)
    sample_image_rgb = sample_image_rgb / 255
    sample_image_rgb = (sample_image_rgb - mean) / std
    sample_image_rgb = sample_image_rgb.transpose(2, 0, 1)
    sample_image_rgb = np.ascontiguousarray(sample_image_rgb, dtype=np.float32)
    sample_image_rgb = np.expand_dims(sample_image_rgb, 0)
    return sample_image_rgb, sample_image_resized


class CARLA:
    def __init__(self, scenario=1, port=2000, town="Town04"):
        client = carla.Client("127.0.0.1", port)
        # client = carla.Client("10.29.40.161", 2000)

        if port == 2002:
            tm = client.get_trafficmanager(8002)
            world = client.load_world(town)
        else:
            tm = client.get_trafficmanager()
            world = client.load_world(town)

        tm_port = tm.get_port()
        # world = client.get_world()
        # world = client.load_world("Town05_Opt")


        bp_lib = world.get_blueprint_library()

        # Get the map spawn points
        spawn_points = world.get_map().get_spawn_points()

        # spawn vehicle
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

        # spawn camera
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_init_trans = carla.Transform(carla.Location(z=3))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        vehicle.set_autopilot(True, tm_port)

        # create npc cars and set autopilot
        for i in range(50):
            vehicle_bp = random.choice(bp_lib.filter('vehicle'))
            npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            if npc:
                npc.set_autopilot(True, tm_port)

        # # Set up the simulator in synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = 0.05
        # weather = carla.WeatherParameters(
        #     cloudiness=80.0,
        #     precipitation=70.0,
        #     sun_altitude_angle=70.0)

        if scenario == 1:
            world.set_weather(carla.WeatherParameters.HardRainSunset)
        elif scenario == 2:
            world.set_weather(carla.WeatherParameters.ClearSunset)
        elif scenario == 3:
            world.set_weather(carla.WeatherParameters.ClearNight)

        print(world.get_weather())
        world.apply_settings(settings)

        # Create a queue to store and retrieve the sensor data
        self.image_queue = queue.Queue()
        self.camera = camera
        world.tick()
        camera.listen(self.image_queue.put)
        time.sleep(1)
        self.world = world

    def read(self):
        self.world.tick()
        image = self.image_queue.get()

        frame = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        frame = frame[..., :3]
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame


def post_process_outputs_single(output_prob):
    if isinstance(output_prob, Dict):
        output_prob = output_prob["out"]

    if isinstance(output_prob, torch.Tensor):
        output_mask = torch.argmax(output_prob, 1)
        output_mask = to_numpy(output_mask)
    elif isinstance(output_prob, np.ndarray):
        output_mask = np.argmax(output_prob, 1)
    elif isinstance(output_prob, pycuda.gpuarray.GPUArray):
        batch_size, num_of_class, H, W = output_prob.shape
        output_prob        = output_prob.reshape(num_of_class, H * W)
        device_output_mask = sk.argmax(a_gpu=output_prob, axis=0)
        output_mask        = device_output_mask.get().reshape(1, H, W)

    else:
        raise ValueError("only Tensor or numpy array are supported")

    return output_mask


def visualize_single(output: Union[np.ndarray, torch.Tensor], frame_resized, color_palette,
                     segmentation_mask_visualization_weights=0.7):
    output = output[0].astype(np.uint8)
    # for label in [1, 2, 3, 4]:
    #     apply_morphology_operation(target=output, kernel=self.erosion_kernel, label=label,
    #                                operation="erosion", null_label=0)
    color_seg = color_palette[output]

    color_seg = color_seg[..., ::-1]
    color_seg = color_seg.astype(np.uint8)
    if output.shape[0] != frame_resized.shape[0]:
        color_seg = cv2.resize(color_seg,
                               (frame_resized.shape[1], frame_resized.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    frame_resized[color_seg != 0] = segmentation_mask_visualization_weights * color_seg[color_seg != 0] + \
                                    (1 - segmentation_mask_visualization_weights) * frame_resized[
                                        color_seg != 0]
    # frame_resized = cv2.resize(frame_resized, (1280, 720), interpolation=cv2.INTER_NEAREST)
    return frame_resized


def compute_overlap(box_array1: np.ndarray, box_array2: np.ndarray):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    box_array1: (N, 4) ndarray of float
    box_array2: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    area_box_array2 = (box_array2[:, 2] - box_array2[:, 0]) * (box_array2[:, 3] - box_array2[:, 1])
    area_box_array1 = np.expand_dims((box_array1[:, 2] - box_array1[:, 0]) * (box_array1[:, 3] - box_array1[:, 1]),
                                     axis=1)

    iw = np.minimum(np.expand_dims(box_array1[:, 2], axis=1), box_array2[:, 2]) - np.maximum(
        np.expand_dims(box_array1[:, 0], 1), box_array2[:, 0])
    ih = np.minimum(np.expand_dims(box_array1[:, 3], axis=1), box_array2[:, 3]) - np.maximum(
        np.expand_dims(box_array1[:, 1], 1), box_array2[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.minimum(area_box_array1, area_box_array2)

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def apply_morphology_operation(target: np.ndarray, kernel, label: int,
                               operation: str, null_label: int):
    assert operation in ["dilation", "closing", "opening", "erosion"], \
        "Only dilation or closing operation are supported"
    if label == 0:
        return
    if label == 255:
        return
    if np.sum(target == label) == 0:
        return

    target_temp = np.zeros_like(target)
    target_temp[target == label] = label
    target[target == label] = null_label
    if operation == "closing":
        target_temp = cv2.morphologyEx(target_temp, cv2.MORPH_CLOSE, kernel)
    elif operation == "opening":
        target_temp = cv2.morphologyEx(target_temp, cv2.MORPH_OPEN, kernel)
    elif operation == "dilation":
        target_temp = cv2.dilate(target_temp, kernel, iterations=1)
    elif operation == "erosion":
        target_temp = cv2.erode(target_temp, kernel, iterations=1)
    target[target_temp == label] = label


def draw_keypoints(outputs, image):
    # pairs of edges for 17 of the keypoints detected ...
    # ... these show which points to be connected to which point ...
    # ... we can omit any of the connecting points if we want, basically ...
    # ... we can easily connect less than or equal to 17 pairs of points ...
    # ... for keypoint RCNN, not  mandatory to join all 17 keypoint pairs
    edges = [
        (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
        (12, 14), (14, 16), (5, 6)
    ]
    # the `outputs` is list which in-turn contains the dictionaries
    for i in range(len(outputs[0]['keypoints'])):
        keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()

        # proceed to draw the lines if the confidence score is above 0.9
        if outputs[0]['scores'][i] > 0.9:
            keypoints = keypoints[:, :].reshape(-1, 3)
            for p in range(keypoints.shape[0]):
                # draw the keypoints
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                           3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

                # uncomment the following lines if you want to put keypoint number
                # cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            for ie, e in enumerate(edges):
                # get different colors for the edges
                rgb = matplotlib.colors.hsv_to_rgb([
                    ie / float(len(edges)), 1.0, 1.0
                ])
                rgb = rgb * 255
                # join the keypoint pairs to draw the skeletal structure
                cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                         (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                         tuple(rgb), 2, lineType=cv2.LINE_AA)
        else:
            continue

    return image
