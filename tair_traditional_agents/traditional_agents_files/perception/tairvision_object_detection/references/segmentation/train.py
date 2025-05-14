import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision

from tairvision.references.segmentation.coco_utils import get_coco
import tairvision.references.segmentation.presets as presets
import tairvision.references.segmentation.utils as utils

from tairvision.references.segmentation.voc_utils import get_voc
from tairvision.references.segmentation.bdd_utils import get_bdd
from tairvision.references.segmentation.transforms import UnNormalize

from torch.utils.tensorboard import SummaryWriter
from tairvision.utils import draw_segmentation_masks


def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "voc": (dir_path, get_voc, utils.CollateFn(False)),
        "voc_aug": (dir_path, sbd, utils.CollateFn(False)),
        "coco": (dir_path, get_coco, utils.CollateFn(False)),
        "bdd": (dir_path, get_bdd, utils.CollateFn(True))
    }
    p, ds_fn, collate_fn = paths[name]

    ds, num_classes = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes, collate_fn


def get_transform(train):
    base_size = 520
    crop_size = 480

    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes, criterion, epoch, writer):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    loss_total = 0
    batch_idx = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 10, header):
            if isinstance(target, list):
                target = target[0]
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            output = output['out']
            loss_total += loss.item()
            batch_idx += 1

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()
        
        origin_imgs = []
        images_to_show = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image, target)[0]
        images_to_show = (images_to_show.cpu().clone()*255).to(dtype=torch.uint8)
        origin_imgs.append(images_to_show)
        grid = torchvision.utils.make_grid(origin_imgs[0])
        
        writer.add_image('images_input', grid, epoch) 
        
        output_to_show = output.clone().detach().cpu().softmax(1).round().to(torch.uint8)
        output_to_show_list = []
        for i_sample in range(len(images_to_show)):
            output_to_show_list.append(draw_segmentation_masks(images_to_show[i_sample], output_to_show[i_sample]))
        output_to_show_list = torch.stack(output_to_show_list, dim=0)
        origin_imgs.append(output_to_show_list)

        grid = torchvision.utils.make_grid(origin_imgs[1])
        writer.add_image('images_overlay', grid, epoch) 
            
        writer.add_scalar("Loss/val", loss_total/batch_idx, epoch)
        writer.flush()

    return confmat


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    batch_idx = 0
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        if isinstance(target, list):
            target = target[0]
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        
        batch_idx += 1
        iter_idx = epoch * len(data_loader) + batch_idx
        writer.add_scalar("Loss/train", loss.item(), iter_idx)
        writer.add_scalar("Lr", optimizer.param_groups[0]["lr"], iter_idx)
        writer.flush()


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.dataset_split_suffix == '':
        train_split = 'train'
        val_split = 'val'
    else: 
        train_split = 'train-' + args.dataset_split_suffix
        val_split = 'val-' + args.dataset_split_suffix

    dataset, num_classes, collate_fn = get_dataset(args.data_path, args.dataset, train_split, get_transform(train=True))
    dataset_test, _, _ = get_dataset(args.data_path, args.dataset, val_split, get_transform(train=False))

    writer = SummaryWriter(os.path.join(args.output_dir, 'runs'))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=16,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=collate_fn)

    model = torchvision.models.segmentation.__dict__[args.model](num_classes=num_classes,
                                                                 aux_loss=args.aux_loss,
                                                                 pretrained=args.pretrained)
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, criterion=criterion, epoch=-1, writer=writer)
        print(confmat)
        return

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, writer)
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, criterion=criterion, epoch=epoch, writer=writer)
        print(confmat)
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args
        }
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    writer.close()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset path')
    parser.add_argument('--dataset', default='coco', help='dataset name')
    parser.add_argument('--dataset-split-suffix', default='', help='dataset split suffix')
    parser.add_argument('--model', default='fcn_resnet101', help='model')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
