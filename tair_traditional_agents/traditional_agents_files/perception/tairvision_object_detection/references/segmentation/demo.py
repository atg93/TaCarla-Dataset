import tairvision
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Optional, Union, Dict
import torch
import torch.utils.data
import tairvision
from torchvision.transforms import ToPILImage

import tairvision.references.segmentation.presets as presets
from tairvision.references.segmentation.transforms import UnNormalize
from tairvision.utils import draw_segmentation_masks

lane_color_palette = np.array([[0, 0, 0],
                               [180, 220, 30],
                               [200, 100, 1],
                               [1, 100, 200],
                               [30, 200, 180]])

bdd_lane_color_palette = np.array([[0, 0, 0],
                                   [255, 1, 1],
                                   [1, 255, 1]])

bdd_drivable_color_palette = np.array([[1, 255, 1],
                                       [1, 1, 255],
                                       [0, 0, 0]])

bdd_segmentation_color_palette = np.array([[128, 64, 128],
                                           [244, 35, 232],
                                           [70, 70, 70],
                                           [102, 102, 156],
                                           [190, 153, 153],
                                           [153, 153, 153],
                                           [250, 170, 30],
                                           [220, 220, 1],
                                           [107, 142, 35],
                                           [152, 251, 152],
                                           [70, 130, 180],
                                           [220, 20, 60],
                                           [255, 1, 1],
                                           [1, 1, 142],
                                           [1, 1, 70],
                                           [1, 60, 100],
                                           [1, 80, 100],
                                           [1, 1, 230],
                                           [119, 11, 32],
                                           ])

mapillary_color_palette = np.array([[165, 42, 42],
                                    [0, 192, 0],
                                    [250, 170, 32],
                                    [196, 196, 196],
                                    [190, 153, 153],
                                    [180, 165, 180],
                                    [90, 120, 150],
                                    [250, 170, 33],
                                    [250, 170, 34],
                                    [128, 128, 128],
                                    [250, 170, 35],
                                    [102, 102, 156],
                                    [128, 64, 255],
                                    [140, 140, 200],
                                    [170, 170, 170],
                                    [250, 170, 36],
                                    [250, 170, 160],
                                    [250, 170, 37],
                                    [96, 96, 96],
                                    [230, 150, 140],
                                    [128, 64, 128],
                                    [110, 110, 110],
                                    [110, 110, 110],
                                    [244, 35, 232],
                                    [128, 196, 128],
                                    [150, 100, 100],
                                    [70, 70, 70],
                                    [150, 150, 150],
                                    [150, 120, 90],
                                    [220, 20, 60],
                                    [255, 0, 0],
                                    [255, 0, 100],
                                    [255, 0, 200],
                                    [255, 255, 255],
                                    [255, 255, 255],
                                    [250, 170, 29],
                                    [250, 170, 26],
                                    [250, 170, 25],
                                    [250, 170, 24],
                                    [250, 170, 22],
                                    [250, 170, 21],
                                    [250, 170, 20],
                                    [255, 255, 255],
                                    [250, 170, 19],
                                    [250, 170, 18],
                                    [250, 170, 12],
                                    [250, 170, 11],
                                    [255, 255, 255],
                                    [255, 255, 255],
                                    [250, 170, 16],
                                    [250, 170, 15],
                                    [250, 170, 15],
                                    [255, 255, 255],
                                    [255, 255, 255],
                                    [255, 255, 255],
                                    [255, 255, 255],
                                    [64, 170, 64],
                                    [230, 160, 50],
                                    [70, 130, 180],
                                    [190, 255, 255],
                                    [152, 251, 152],
                                    [107, 142, 35],
                                    [0, 170, 30],
                                    [255, 255, 128],
                                    [250, 0, 30],
                                    [100, 140, 180],
                                    [220, 128, 128],
                                    [222, 40, 40],
                                    [100, 170, 30],
                                    [40, 40, 40],
                                    [33, 33, 33],
                                    [100, 128, 160],
                                    [20, 20, 255],
                                    [142, 0, 0],
                                    [70, 100, 150],
                                    [250, 171, 30],
                                    [250, 173, 30],
                                    [250, 174, 30],
                                    [250, 175, 30],
                                    [250, 176, 30],
                                    [210, 170, 100],
                                    [153, 153, 153],
                                    [128, 128, 128],
                                    [0, 0, 80],
                                    [210, 60, 60],
                                    [250, 170, 30],
                                    [250, 170, 30],
                                    [250, 170, 30],
                                    [250, 170, 30],
                                    [250, 170, 30],
                                    [250, 170, 30],
                                    [192, 192, 192],
                                    [192, 192, 192],
                                    [220, 220, 0],
                                    [220, 220, 0],
                                    [0, 0, 196],
                                    [192, 192, 192],
                                    [220, 220, 0],
                                    [140, 140, 20],
                                    [119, 11, 32],
                                    [150, 0, 255],
                                    [0, 60, 100],
                                    [0, 0, 142],
                                    [0, 0, 90],
                                    [0, 0, 230],
                                    [0, 80, 100],
                                    [128, 64, 64],
                                    [0, 0, 110],
                                    [0, 0, 70],
                                    [0, 0, 192],
                                    [170, 170, 170],
                                    [32, 32, 32],
                                    [111, 74, 0],
                                    [120, 10, 10],
                                    [81, 0, 81],
                                    [111, 111, 0]])


def to_numpy(input_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts torch tensor to numpy array by proper operations
    :param input_tensor:
    :return:
    """
    input_numpy = input_tensor.detach().cpu().numpy()
    return input_numpy


def post_process_outputs(output_prob):
    if isinstance(output_prob, torch.Tensor):
        output_mask = torch.argmax(output_prob, 1)
    elif isinstance(output_prob, np.ndarray):
        output_mask = np.argmax(output_prob, 1)
    else:
        raise ValueError("only Tensor or numpy array are supported")

    return output_mask


def visualize(output: Union[np.ndarray, torch.Tensor], frame_resized, color_palette=None,
              segmentation_mask_visualization_weights=0.7):
    if isinstance(output, torch.Tensor):
        output = to_numpy(output)
    output = output[0].astype(np.uint8)
    # for label in [1, 2, 3, 4]:
    #     apply_morphology_operation(target=output, kernel=self.erosion_kernel, label=label,
    #                                operation="erosion", null_label=0)

    color_seg = color_palette[output]

    color_seg = color_seg[..., ::-1]
    color_seg = color_seg.astype(np.uint8)
    frame_resized[color_seg != 0] = segmentation_mask_visualization_weights * color_seg[color_seg != 0] + \
                                    (1 - segmentation_mask_visualization_weights) * frame_resized[
                                        color_seg != 0]
    # frame_resized = cv2.resize(frame_resized, (1280, 720), interpolation=cv2.INTER_NEAREST)
    return frame_resized


def preprocess_image(sample_image, cv2_dim):
    sample_image_resized = cv2.resize(sample_image, cv2_dim, interpolation=cv2.INTER_NEAREST)
    sample_image_rgb = cv2.cvtColor(sample_image_resized, cv2.COLOR_BGR2RGB)
    sample_image_rgb = sample_image_rgb / 255
    sample_image_rgb = sample_image_rgb.transpose(2, 0, 1)
    sample_image_rgb = np.ascontiguousarray(sample_image_rgb, dtype=np.float32)
    sample_image_rgb = np.expand_dims(sample_image_rgb, 0)
    return sample_image_rgb, sample_image_resized


def get_transform(train):
    base_size = 520
    crop_size = 480

    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)


def main(args):
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f"/workspace/demo.avi", fourcc, 20.0, (int(960 / 720 * args.base_size * 2),
                                                                     args.base_size * 2))

    device = torch.device(args.device)

    model = tairvision.models.segmentation.__dict__[args.lane_model](num_classes=5,
                                                                     aux_loss=True,
                                                                     pretrained=args.pretrained)
    model.to(device)

    checkpoint = torch.load(args.lane_weights, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model2 = tairvision.models.segmentation.__dict__[args.segmentation_model](num_classes=116,
                                                                              aux_loss=True,
                                                                              pretrained=args.pretrained)
    model2.to(device)

    checkpoint = torch.load(args.segmentation_weights, map_location='cpu')
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2.eval()

    model3 = tairvision.models.segmentation.__dict__[args.bdd_triple_model](num_classes=[3, 3, 19],
                                                                            aux_loss=True,
                                                                            pretrained=args.pretrained)
    model3.to(device)

    checkpoint = torch.load(args.bdd_triple_weights, map_location='cpu')
    model3.load_state_dict(checkpoint['model_state_dict'])
    model3.eval()

    transform = get_transform(train=False)
    cap = cv2.VideoCapture(args.video_path)
    frame_num = 0
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, image = cap.read()
        if ret_val is False:
            break
        if not frame_num % 20 == 0:
            frame_num += 1
            continue

        sample_image_rgb, frame_resized = preprocess_image(image[:, 160:-160, :], (int(960 / 720 * args.base_size), args.base_size))
        image = torch.tensor(sample_image_rgb).to(device)

        with torch.no_grad():
            output = model(image)
            output2 = model2(image)
            output3 = model3(image)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        if key == ord('p'):
            while True:
                key = cv2.waitKey(1)
                if key == ord('p'):
                    break

        output_mask = post_process_outputs(output2["out"])
        output_image2 = visualize(output_mask, frame_resized.copy(), mapillary_color_palette,
                                  segmentation_mask_visualization_weights=0.5)

        output_mask = post_process_outputs(output3["out"])
        output_image3 = visualize(output_mask, frame_resized.copy(), bdd_drivable_color_palette)

        output_image3_ver2 = visualize(output_mask, output_image2.copy(), bdd_drivable_color_palette,
                                       segmentation_mask_visualization_weights=0.9)

        output_mask = post_process_outputs(output["out"])
        output_image = visualize(output_mask, frame_resized.copy(), lane_color_palette)

        output_image_ver2 = visualize(output_mask, output_image3_ver2.copy(), lane_color_palette,
                                      segmentation_mask_visualization_weights=0.9)

        output_mask = post_process_outputs(output3["out_2"])
        output_image4 = visualize(output_mask, frame_resized.copy(), bdd_lane_color_palette)

        output_mask = post_process_outputs(output3["out_3"])
        output_image5 = visualize(output_mask, frame_resized.copy(), bdd_segmentation_color_palette)

        vis1 = np.concatenate((output_image, output_image2), axis=0)
        vis2 = np.concatenate((output_image3, output_image4), axis=0)

        vis3 = np.concatenate((vis1, vis2), axis=1)

        cv2.imshow("joint", vis3)
        cv2.imshow("joint2", output_image_ver2)
        if args.save_video:
            out.write(vis3)
        # cv2.imshow("general segmentation", output_image2)
        # cv2.imshow("drivable area", output_image3)
        # cv2.imshow("lane type", output_image4)
        # cv2.imshow("bdd general segmentation", output_image5)
        frame_num += 1

    cv2.destroyAllWindows()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--video-path', default='/datasets/forddata/V1.mp4', help='dataset path')
    parser.add_argument('--lane_model', default='ocrnet_hrnet18', help='model')
    parser.add_argument('--segmentation_model', default='deeplabv3_resnet18', help='model')
    parser.add_argument('--bdd_triple_model', default='deeplabv3_resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--lane-weights', default='', help='resume from checkpoint')
    parser.add_argument('--segmentation-weights', default='', help='resume from checkpoint')
    parser.add_argument('--bdd_triple_weights', default='', help='resume from checkpoint')
    parser.add_argument('--save_video', dest='save_video', action='store_true', help='in order to save video',
                        required=False)
    parser.add_argument('--base-size', default=540, type=int)
    parser.add_argument('--crop-size', default=480, type=int)
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
