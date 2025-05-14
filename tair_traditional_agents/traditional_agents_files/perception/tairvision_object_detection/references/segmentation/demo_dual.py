import numpy as np
import cv2
from PIL import Image
import time

import torch
import torch.utils.data
from torchvision.transforms import ToPILImage

import tairvision.references.segmentation.presets as presets
import tairvision
from tairvision.references.segmentation.transforms import UnNormalize
from tairvision.utils import draw_segmentation_masks

def get_transform(train, base_size, crop_size):
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)

def main(args):

    device = torch.device(args.device)

    model = tairvision.models.segmentation.__dict__[args.model](num_classes=[2,3],
                                                                aux_loss=True,
                                                                pretrained=args.pretrained,
                                                                size=[args.base_size, int(1280/720 * args.base_size)])
    model.to(device)

    checkpoint = torch.load(args.trained_model, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    transform = get_transform(train=False, base_size=args.base_size, crop_size=args.crop_size)


    cap = cv2.VideoCapture(args.video_path)
    #out = cv2.VideoWriter('/home/ok21/output_3.avi',cv2.VideoWriter_fourcc(*'MJPG'), 40.0, (924,520))

    print('Press "Esc", "q" or "Q" to exit.')
    frame_count = 0
    while True:
        ret_val, image = cap.read()
        if not ret_val:
            break
        frame_count += 1
        if frame_count % 20 == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image, _ = transform(image, [image])
            image = image.unsqueeze(0).to(device)
            start = time.time()
            with torch.no_grad():
                output = model(image)
            stop = time.time()
            #print(stop - start)
            output1 = output['out']
            output2 = output['out_2']

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break

            images_to_show = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image, None)[0]
            images_to_show = (images_to_show.cpu().clone()*255).to(dtype=torch.uint8).squeeze(0)

            output1 = output1.clone().detach().cpu().softmax(1).round().to(torch.bool)
            output2 = output2.clone().detach().cpu().softmax(1).round().to(torch.bool)

            output_to_show = torch.cat((output2, output1[:,1:2]), dim=1)
            output_to_show = draw_segmentation_masks(images_to_show, output_to_show[0], colors=['green', 'blue', 'red', 'white'], alpha=0.2)
            output_to_show = np.array(ToPILImage()(output_to_show))
            output_to_show = cv2.cvtColor(output_to_show, cv2.COLOR_RGB2BGR)

            #out.write(output_to_show)
            
            cv2.imshow("Dual", output_to_show)

    cap.release()
    #out.release()
    cv2.destroyAllWindows()

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--video-path', default='/datasets/forddata/V1.mp4', help='video location')
    parser.add_argument('--model', default='deeplabv3_resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--base-size', default=520, type=int)
    parser.add_argument('--crop-size', default=480, type=int)
    parser.add_argument('--trained-model', default='', help='resume from checkpoint')
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
