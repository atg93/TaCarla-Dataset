r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time
import re

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import tairvision
import tairvision.references.detection.utils as utils
from tairvision.nn.optim.gradient_clipping import AdamWClipping
from tairvision.references.detection.utils import get_dataset, initialize_wandb
from tairvision.references.detection.config import get_arguments, store_yaml
from tairvision.references.detection.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from tairvision.references.detection.engine import train_one_epoch, evaluate
from tairvision.references.detection.presets import get_transform
from tairvision.ops import misc as misc_nn_ops


def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    if args.output_dir:
        utils.mkdir(args.output_dir)
        store_yaml(args)

    utils.init_distributed_mode(args)

    wandb_object = None
    try:
        import wandb
        if not args.debug:
            wandb_object = initialize_wandb(args)
    except:
        print("Install Wandb and start experiment tracking ASAP!")

    device = torch.device(args.device)

    if args.debug:
        args.workers = 0
        print("Debug mode is set, workers=0, wandb is not initialized.")

    # Data loading code
    print("Loading data")

    if not args.test_only:
        train_split = "train"
    else:
        train_split = "val"

    dataset, num_classes, collate_fn, num_keypoints = get_dataset(args.data_path, args.dataset, train_split,
                                                                  get_transform(is_train=True,
                                                                                data_aug=args.data_augmentation,
                                                                                min_size=args.transform_min_size,
                                                                                max_size=args.transform_max_size))
    dataset_test, _, _, _ = get_dataset(args.data_path, args.dataset, "val",
                                        get_transform(is_train=False, data_aug=args.eval_data_augmentation,
                                                      min_size=args.transform_min_size,
                                                      max_size=args.transform_max_size))

    # subset_indices = list(range(0, len(dataset), 4))
    # dataset = torch.utils.data.Subset(dataset, subset_indices)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=collate_fn)

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
        "bifpn_norm_layer": misc_nn_ops.FrozenBatchNorm2d if args.coco_pretrained else None
    }

    if "fcos" in args.model or "tood" in args.model:
        kwargs["fpn_strides"] = args.fpn_strides
        kwargs["sois"] = args.sois
        kwargs["thresh_with_ctr"] = args.thresh_with_ctr
        kwargs["use_deformable"] = args.use_deformable
        kwargs["cls_loss"] = args.cls_loss
        if kwargs["cls_loss"] == "ap_loss":
            kwargs["ap_delta"]: args.ap_delta
        if "mask" in args.model:
            kwargs["roi_output_size"] = args.roi_output_size
            kwargs["roi_sampling_ratio"] = args.roi_sampling_ratio
    else:
        kwargs["anchor_sizes"] = args.anchor_sizes
        if "rcnn" in args.model:
            if args.rpn_score_thresh is not None:
                kwargs["rpn_score_thresh"] = args.rpn_score_thresh

    if args.model_kwargs:
        kwargs.update(args.model_kwargs)

    model = tairvision.models.detection.__dict__[args.model](num_classes=num_classes,
                                                             num_keypoints=num_keypoints,
                                                             **kwargs)

    if args.coco_pretrained:
        checkpoint = torch.load(args.coco_pretrained, map_location='cpu')
        exclude_keys = args.exclude_pretrained_params
        model_dict = model.state_dict()

        if len(exclude_keys) != 0:
            pretrained_dict = {k: v for k, v in checkpoint['model'].items() if not re.match("|".join(exclude_keys), k)}
        else:
            pretrained_dict = checkpoint['model']

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)  # TODO: find the unused params
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.lr_backbone is not None:
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if
                        "backbone.body" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if
                           "backbone.body" in n and p.requires_grad],
                "lr": args.lr_backbone,
            }
        ]
    else:
        param_dicts = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(params=param_dicts, lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    elif args.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(params=param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_type == "adamwclipping":
        optimizer = AdamWClipping(params=param_dicts, lr=args.lr, weight_decay=args.weight_decay, max_norm=0.01)
    else:
        raise RuntimeError("Invalid optimizer type '{}'. Only SGD, AdamW, and AdamWClipping"
                           "are supported.".format(args.optimizer_type))

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                           "are supported.".format(args.lr_scheduler))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if "model" in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['model'])
        elif "model_state_dict" in checkpoint:
            print("alternative weight is loading...")
            model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("Convenient weight is not found!!!")

        if not args.test_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        evaluate(model, data_loader_test, device=device, maxDets=args.maxDets, wandb_object=None,
                 calc_val_loss=args.calc_val_loss)
        return

    writer = SummaryWriter(os.path.join(args.output_dir, 'runs'))

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, writer,
                        scaler=scaler, wandb_object=wandb_object, clip_grad_norm=args.clip_grad_norm)

        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            # utils.save_on_master(
            #    checkpoint,
            #    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device, maxDets=args.maxDets, wandb_object=wandb_object,
                 calc_val_loss=args.calc_val_loss)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    writer.close()
    print('Training time {}'.format(total_time_str))

    # evaluate(model, data_loader_test, device=device, maxDets=args.maxDets, wandb_object=wandb_object)
    if wandb_object:
        wandb_object.finish()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
