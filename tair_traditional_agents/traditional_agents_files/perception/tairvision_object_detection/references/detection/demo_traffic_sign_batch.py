import time
import torch
import torch.utils.data
import cv2
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch import nn
import tairvision
from torchvision.transforms import ToPILImage
from torchvision import transforms
import tairvision.references.detection.presets as presets
from tairvision.references.detection.eda_utils import get_label_eda, get_label_eda_gtsrb
from tairvision.utils import draw_bounding_boxes
from tairvision.references.detection.coco_utils import get_label_color as get_label_color_coco
from tairvision.references.detection.utils import get_dataset
from tairvision.references.detection.config import get_arguments
from PIL import Image

def main(args):
    device = torch.device(args.device)

    dataset_test, num_classes, collate_fn, num_keypoints = get_dataset(args.data_path, args.dataset, "val",
                                                                       presets.DetectionPresetEval(None))

    print("Creating model")
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

    if "fcos" in args.model:
        kwargs["fpn_strides"] = args.fpn_strides
        kwargs["sois"] = args.sois
        kwargs["thresh_with_ctr"] = args.thresh_with_ctr
        kwargs["use_deformable"] = args.use_deformable
    else:
        kwargs["anchor_sizes"] = args.anchor_sizes
        if "rcnn" in args.model:
            if args.rpn_score_thresh is not None:
                kwargs["rpn_score_thresh"] = args.rpn_score_thresh

    def imshow(img):
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )
        img = inv_normalize(img)
        # img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3337, 0.3064, 0.3171],
            std=[0.2672, 0.2564, 0.2629]
        )
    ])

    model = tairvision.models.detection.__dict__[args.model](num_classes=num_classes,
                                                             num_keypoints=num_keypoints,
                                                             **kwargs)

    sign_model = torchvision.models.resnet18()
    num_ftrs = sign_model.fc.in_features
    sign_model.fc = nn.Linear(num_ftrs, 46)
    sign_model.load_state_dict(torch.load("/home/ig21/eda_cls/models/3epoch_res18_GTSRB_EDA_224_othersclass.pth"))

    model.to(device)

    sign_model.to(device)
    sign_model.eval()

    # amp.initialize(model, opt_level="O3", keep_batchnorm_fp32=True)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    model.transform.min_size = (args.transform_min_size,)

    transform = presets.DetectionPresetEval(None)

    cap = cv2.VideoCapture(args.video_path)

    print('Press "Esc", "q" or "Q" to exit.')
    frame_count = 0
    ret_val = True
    print_freq = 1
    print_count = 0
    counter = 0
    total_time = 0
    time_preprocess, time_model, time_cv2 = [], [], []
    while ret_val:
        ret_val, image1 = cap.read()
        frame_count += 1
        if frame_count % print_freq == 0:
            t0 = time.time()
            image = Image.fromarray(image1)
            image, _ = transform(image, None)
            image = image.unsqueeze(0).to(device)
            t1 = time.time()
            with torch.no_grad():
                output = model(image)
            t2 = time.time()
            boxes = output[0]['boxes'].cpu()
            scores = output[0]['scores'].cpu()
            labels = output[0]['labels'].cpu().numpy()
            masks = output[0]['masks'].cpu() if 'masks' in output[0].keys() else None

            boxes = boxes[scores > float(args.score_thres)]
            labels = labels[scores > float(args.score_thres)]

            boxes1 = boxes[labels == 9]
            boxes2 = boxes[labels == 10]
            boxes = torch.cat((boxes1, boxes2), 0)

            labels1 = labels[labels == 9]
            labels2 = labels[labels == 10]
            labels_traffic = np.concatenate([labels1, labels2])

            predicted_labels = []
            predicted_scores = []
            img_list = []

            if boxes is not None:
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

                    img = image1[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)

                    img = tf(img)
                    img_list.append(img)

                    # imshow(img_tensor)

                if len(img_list) > 0:

                    tic = time.perf_counter()
                    imgs_tensor = torch.stack(img_list)

                    imgs_tensor = imgs_tensor.to(device)
                    outs = sign_model(imgs_tensor)
                    _, predicted = torch.max(outs.data, 1)
                    toc = time.perf_counter()
                    new_time = (toc - tic)
                    counter = counter + 1
                    total_time = total_time + new_time

                    avg_time = total_time / counter
                    print("time: ", new_time)

                    # percentage = torch.nn.functional.softmax(outs, dim=1)[0] * 100
                    percentage = torch.nn.functional.softmax(outs, dim=1) * 100
                    scrs = torch.amax(percentage, 1)

                    for i in range(predicted.shape[0]):
                        pred = predicted[i].item() + 1

                        scr = round(scrs.data[i].item(), 2)

                        if scr > args.cls_score_thres * 100:
                            predicted_labels.append(pred)
                            predicted_scores.append(scr)
                        else:
                            predicted_labels.append(47)
                            predicted_scores.append(0)

                # toc = time.perf_counter()

                labels_str = [*map(get_label_eda_gtsrb, [*predicted_labels])]

                at = np.full(fill_value=' - ', shape=len(labels_str), dtype=object)  # optional third list
                result = np.array(labels_str, dtype=str) + at + np.array(predicted_scores, dtype=str)

                label_colors = [*map(get_label_color_coco, [*predicted_labels])]

                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord('q') or ch == ord('Q'):
                    break

                images_to_show = (image.cpu().clone() * 255).to(dtype=torch.uint8).squeeze(0)

                if masks is not None:
                    masks = masks[scores > float(args.score_thres)]
                    if len(masks) > 0:
                        masks_max = masks.max(0)[0]
                        masks_max_mask = (masks_max > 0.10).squeeze(0)

                        image_segm = 0.8 * torch.tensor([[0], [100], [0]]) * masks_max[:, masks_max_mask] + \
                                     (1 - 0.8) * images_to_show[:, masks_max_mask]
                        images_to_show[:, masks_max_mask] = image_segm.to(torch.uint8)

                output_to_show = draw_bounding_boxes(images_to_show, boxes, result, font_size=40)
                output_to_show = np.array(ToPILImage()(output_to_show))

            cv2.imshow("Seperate", output_to_show)
            t3 = time.time()

            time_preprocess.append(t1 - t0)
            time_model.append(t2 - t1)
            time_cv2.append(t3 - t2)

            if print_count == 100:
                print("Preprocess: " + str(np.asarray(time_preprocess).mean()))
                print("Model: " + str(np.asarray(time_model).mean()))
                print("CV2: " + str(np.asarray(time_cv2).mean()))
                print_count = 0
                time_preprocess, time_model, time_cv2 = [], [], []
            else:
                print_count += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main(args)
