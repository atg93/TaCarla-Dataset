import argparse
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
from l2cs_utils import draw_gaze
from PIL import Image
import tairvision
from tairvision.models.analysis import L2CS
from tairvision.references.detection.config import get_arguments
import tairvision.references.detection.presets as presets

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam', dest='cam_id', help='Camera device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args



def getArch(arch, bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
    return model


if __name__ == '__main__':
    args = get_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = {
        "type": args.backbone,
        "trainable_backbone_layers": args.trainable_backbone_layers,
        "pretrained": args.pretrained,
        "pyramid_type": args.pyramid_type,
        "repeats": args.bifpn_repeats,
        "fusion_type": args.fusion_type,
        "depthwise": args.use_depthwise,
        "use_P2": args.use_P2,
        "no_extra_blocks": args.no_extra_blocks,
        "extra_before": args.extra_before,
        "context_module": args.context_module,
        "loss_weights": args.loss_weights,
        "nms_thresh": args.nms_thresh,
        "post_nms_topk": args.post_nms_topk,
        "min_size": args.transform_min_size,
        "max_size": args.transform_max_size
    }


    detector = tairvision.models.detection.__dict__[args.model](num_classes=2,
                                                             num_keypoints=5,
                                                             **kwargs)
    detector.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    detector.load_state_dict(checkpoint['model'])
    detector.eval()

    transform = presets.DetectionPresetEval(None)


    cudnn.enabled = True
#    arch = arguments.arch
    batch_size = 1
    cam = 0

    snapshot_path = ("/home/ig21/git-local/L2CS-Net/epoch50/_epoch_50.pkl")

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.to(device)
    model.eval()

    softmax = nn.Softmax(dim=1)
    #detector = RetinaFace()
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    x = 0

    cap = cv2.VideoCapture(cam)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            success, image1 = cap.read()

            start_fps = time.time()
            if args.video_path is None:
                image = image1[:, ::-1, :]
            h, w = image.shape[0:2]

            image1 = cv2.flip(image1,1)
            if args.video_crop_height is not None:
                h_out = args.video_crop_height
                w_out = (args.video_crop_height // 9) * 16
                h_start = (h - h_out) // 2
                h_end = h_start + h_out
                w_start = (w - w_out) // 2
                w_end = w_start + w_out
                image = image[h_start:h_end, w_start:w_end, :]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image, _ = transform(image, None)
            image = image.unsqueeze(0).to(device)
            faces = detector(image)

            boxes = faces[0]['boxes'].cpu()
            scores = faces[0]['scores'].cpu()
            keypoints = faces[0]['keypoints'].cpu()

            #box = boxes[0]
            #score = scores[0]

            if faces is not None:
                for i_box in range(len(boxes)):
                    box = boxes[i_box]
                    score = scores[i_box]
                    if score < .4:
                        continue
                    x_min = int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min = int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max = int(box[2])
                    y_max = int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    # x_min = max(0,x_min-int(0.2*bbox_height))
                    # y_min = max(0,y_min-int(0.2*bbox_width))
                    # x_max = x_max+int(0.2*bbox_height)
                    # y_max = y_max+int(0.2*bbox_width)
                    # bbox_width = x_max - x_min
                    # bbox_height = y_max - y_min

                    # Crop image
                    img = image.cpu()
                    img = img.numpy().swapaxes(0,2).swapaxes(1,3).swapaxes(2,3).squeeze()

                    img = img[y_min:y_max, x_min:x_max]

                    img = cv2.resize(img, (224, 224))
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray((img * 255).astype(np.uint8))
                    img = transformations(im_pil)
                    img = Variable(img).to(device)
                    img = img.unsqueeze(0)

                    # gaze prediction
                    gaze_pitch, gaze_yaw = model(img)

                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)

                    # Get continuous predictions in degrees.
                    pitch_predicted_degree = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted_degree = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180

                    pitch_predicted = pitch_predicted_degree.cpu().detach().numpy() * np.pi / 180.0
                    yaw_predicted = yaw_predicted_degree.cpu().detach().numpy() * np.pi / 180.0

                    draw_gaze(x_min, y_min, bbox_width, bbox_height, image1, (pitch_predicted, yaw_predicted),
                              color=(0, 0, 255))
                    cv2.rectangle(image1, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            myFPS = 1.0 / (time.time() - start_fps)
#            cv2.putText(image1, 'PITCH: {:.3f}'.format(pitch_predicted_degree) + ' YAW: {:.3f}'.format(yaw_predicted_degree), (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2,
#                        cv2.LINE_AA)

            cv2.imshow("Demo", image1)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            success, image1 = cap.read()
