import os
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from tairvision.datasets import LFWPeople
from tairvision.datasets.affectnet import AffectNet, UTKFace, AffectNetMulti
from tairvision.models.analysis.dan import AffinityLoss, PartitionLoss, AgeLoss, CCCLoss, DANWithMoodAgeGender, TurkcellAgeGender
from tairvision.references.analysis.utils import ImbalancedDatasetSampler, get_dist
from tairvision.references.analysis.config import get_arguments_analysis, store_yaml
from tairvision.references.analysis.utils import initialize_wandb
from tairvision.references.detection import utils as tairvision_utils

from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_funcs, args, writer, wandb_object=None):
    model.train()

    running_loss = 0.0
    correct_sum_exp = 0
    sample_cnt_exp = 0
    batch_idx = 0
    c_exp, c_gender, c_age = args.loss_coeffs

    criterion_cls, criterion_af, criterion_pt, criterion_age, criterion_val_aro = loss_funcs

    for data in data_loader:
        batch_idx += 1
        optimizer.zero_grad()

        if args.use_mood:
            if args.use_valence:
                if args.use_gender:
                    imgs, target_exp, target_gender, target_age, target_val, target_aro = data
                    target_gender = target_gender.to(torch.int64).to(device)
                    target_age = target_age.to(device)
                    target_age_dist = get_dist(target_age, K=10).to(device)
                else:
                    imgs, target_exp, target_val, target_aro = data
                target_val = target_val.to(device)
                target_aro = target_aro.to(device)
            else:
                imgs, target_exp = data
            target_exp = target_exp.to(device)
        else:
            if args.use_gender:
                imgs, target_gender, target_age = data
                target_gender = target_gender.to(torch.int64).to(device)
                target_age = target_age.to(device)
                target_age_dist = get_dist(target_age, K=10).to(device)

        imgs = imgs.to(device)

        out, feat, heads = model(imgs)
        pred_exp, pred_gender, pred_age_dist, pred_age, pred_val, pred_aro = out

        if args.use_mood:
            loss_cls = criterion_cls(pred_exp, target_exp)
            loss_pt = criterion_pt(heads)
            loss_af = criterion_af(feat, target_exp)
            if args.use_valence:
                pred_val = torch.squeeze(pred_val)
                pred_aro = torch.squeeze(pred_aro)
                loss_val_aro = criterion_val_aro(target_val.float(), pred_val.float(), target_aro.float(),
                                                 pred_aro.float())
                if args.use_gender:
                    # loss_af = criterion_af(feat, target_exp + args.num_class * target_gender)
                    loss_cls_gender = criterion_cls(pred_gender, target_gender)
                    loss_dist_age, loss_age = criterion_age(pred_age_dist, pred_age, target_age_dist, target_age, model)
                    loss_exp = c_exp * (loss_af + loss_pt + loss_cls + loss_val_aro)
                    loss = loss_exp + c_gender * loss_cls_gender + \
                           c_age * (loss_dist_age + loss_age)

                else:
                    loss_exp = c_exp * (loss_cls + loss_pt + loss_af + loss_val_aro)
                    loss = loss_exp
            else:
                loss_exp = c_exp * (loss_cls + loss_pt + loss_af)
                loss = loss_exp
        else:
            if args.use_gender:
                loss_cls_gender = criterion_cls(pred_gender, target_gender)
                loss_dist_age, loss_age = criterion_age(pred_age_dist, pred_age, target_age_dist, target_age, model)
                if args.use_dan:
                    loss_af = criterion_af(feat, target_gender)
                    loss_pt = criterion_pt(heads)
                    loss = c_gender * (loss_cls_gender + loss_af + loss_pt) + \
                           c_age * (loss_dist_age + loss_age)
                else:
                    loss = c_gender * loss_cls_gender + \
                           c_age * (loss_dist_age + loss_age)


        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if pred_exp is not None:
            # running_loss += loss.item()
            _, predicts_exp = torch.max(pred_exp, 1)
            correct_num_exp = torch.eq(predicts_exp, target_exp).sum()
            correct_sum_exp += correct_num_exp
            sample_cnt_exp += pred_exp.size(0)

        iter_idx = epoch * len(data_loader) + batch_idx

        writer.add_scalar("Loss/train_all", loss.item(), iter_idx)

        if pred_exp is not None:
            writer.add_scalar("Loss/train_cls", loss_cls.item(), iter_idx)
            writer.add_scalar("Loss/train_af", loss_af.item(), iter_idx)
            writer.add_scalar("Loss/train_pt", loss_pt.item(), iter_idx)
            writer.add_scalar("Loss/train_all", loss.item(), iter_idx)

        if pred_val is not None:
            writer.add_scalar("Loss/train_val_aro", loss_val_aro.item(), iter_idx)

        if pred_gender is not None:
            writer.add_scalar("Loss/train_cls_gender", loss_cls_gender.item(), iter_idx)
            writer.add_scalar("Loss/train_dist_age", loss_dist_age.item(), iter_idx)
            writer.add_scalar("Loss/train_age", loss_age.item(), iter_idx)

            if pred_exp is not None or not args.use_dan:
                continue
            writer.add_scalar("Loss/train_af", loss_af.item(), iter_idx)
            writer.add_scalar("Loss/train_pt", loss_pt.item(), iter_idx)

        writer.add_scalar("Lr", optimizer.param_groups[0]["lr"], iter_idx)
        writer.flush()

    running_loss = running_loss / batch_idx
    if pred_exp is not None:
        acc_exp = correct_sum_exp.float() / sample_cnt_exp  # float(data_loader.__len__())
        tqdm.write('[Epoch %d] Training accuracy for Mood: %.4f. Loss: %.3f. LR %.6f' % (
            epoch, acc_exp, running_loss, optimizer.param_groups[0]['lr']))
    if pred_val is not None:
        tqdm.write("[Epoch %d] Training Loss for Valence and Arousal:%.4f." % (epoch, loss_val_aro))
    if pred_gender is not None:
        tqdm.write('[Epoch %d] Training Loss for Age: %.3f. LR %.6f' % (
            epoch, loss_age, optimizer.param_groups[0]['lr']))
        tqdm.write('[Epoch %d] Training Loss for Gender: %.3f. LR %.6f' % (
            epoch, loss_cls_gender, optimizer.param_groups[0]['lr']))

    if wandb_object:
        wandb_object.log({"Training_Loss": running_loss}, step=iter_idx)
        wandb_object.log({"Lr": optimizer.param_groups[0]["lr"]}, step=iter_idx)
        if pred_exp is not None:
            wandb_object.log({"Exp_Loss": loss_exp}, step=iter_idx)
        if pred_gender is not None:
            wandb_object.log({"Gender_Loss_cls": loss_cls_gender}, step=iter_idx)
            wandb_object.log({"Age_Loss": loss_age}, step=iter_idx)
            wandb_object.log({"Age_Loss_dist": loss_dist_age}, step=iter_idx)
        if pred_val is not None:
            wandb_object.log({"Val_Aro_Loss": loss_val_aro}, step=iter_idx)


def evaluate(model, data_loader, device, epoch, loss_funcs, args, writer, wandb_object=None):
    model.eval()

    running_loss = 0.0
    running_mae_age = 0.0
    iter_cnt = 0
    bingo_cnt_exp = 0
    sample_cnt_exp = 0
    bingo_cnt_gender = 0
    sample_cnt_gender = 0
    running_loss_val_aro = 0.0

    c_exp, c_gender, c_age = args.loss_coeffs

    criterion_cls, criterion_af, criterion_pt, criterion_age, criterion_val_aro = loss_funcs

    for data in data_loader:
        if args.use_mood:
            if args.use_valence:
                if args.use_gender:
                    imgs, target_exp, target_gender, target_age, target_val, target_aro = data
                    target_gender = target_gender.to(torch.int64).to(device)
                    target_age = target_age.to(device)
                    target_age_dist = get_dist(target_age, K=10).to(device)
                else:
                    imgs, target_exp, target_val, target_aro = data
                target_val = target_val.to(device)
                target_aro = target_aro.to(device)
            else:
                imgs, target_exp = data
            target_exp = target_exp.to(device)
        else:
            if args.use_gender:
                imgs, target_gender, target_age = data
                target_gender = target_gender.to(torch.int64).to(device)
                target_age = target_age.to(device)
                target_age_dist = get_dist(target_age, K=10).to(device)

        imgs = imgs.to(device)


        with torch.no_grad():
            out, feat, heads = model(imgs)
        pred_exp, pred_gender, pred_age_dist, pred_age, pred_val, pred_aro = out

        if args.use_mood:
            loss_cls = criterion_cls(pred_exp, target_exp)
            loss_pt = criterion_pt(heads)
            loss_af = criterion_af(feat, target_exp)
            if args.use_valence:
                pred_val = torch.squeeze(pred_val)
                pred_aro = torch.squeeze(pred_aro)
                loss_val_aro = criterion_val_aro(target_val.float(), pred_val.float(), target_aro.float(),
                                                 pred_aro.float())
                if args.use_gender:
                    # loss_af = criterion_af(feat, target_exp + args.num_class * target_gender)
                    loss_cls_gender = criterion_cls(pred_gender, target_gender)
                    loss_dist_age, loss_age = criterion_age(pred_age_dist, pred_age, target_age_dist, target_age, model)
                    loss_exp = c_exp * (loss_cls + loss_pt + loss_af + loss_val_aro)
                    loss = loss_exp + c_gender * loss_cls_gender + \
                           c_age * (loss_dist_age + loss_age)
                else:
                    loss_exp = c_exp * (loss_cls + loss_pt + loss_af)
                    loss = loss_exp + loss_val_aro
            else:
                loss_exp = c_exp * (loss_cls + loss_pt + loss_af)
                loss = loss_exp
        else:
            if args.use_gender:
                loss_cls_gender = criterion_cls(pred_gender, target_gender)
                loss_dist_age, loss_age = criterion_age(pred_age_dist, pred_age, target_age_dist, target_age, model)
                if args.use_dan:
                    loss_af = criterion_af(feat, target_gender)
                    loss_pt = criterion_pt(heads)
                    loss = c_exp * (loss_af + loss_pt) + \
                           c_gender * loss_cls_gender + \
                           c_age * (loss_dist_age + loss_age)
                else:
                    loss = c_gender * loss_cls_gender + \
                           c_age * (loss_dist_age + loss_age)


        running_loss += loss.item()
        iter_cnt += 1
        if pred_exp is not None:
            _, predicts_exp = torch.max(pred_exp, 1)
            correct_num_exp = torch.eq(predicts_exp, target_exp)
            bingo_cnt_exp += correct_num_exp.sum().cpu()
            sample_cnt_exp += pred_exp.size(0)

        if pred_gender is not None:
            _, predicts_gender = torch.max(pred_gender, 1)
            correct_num_gender = torch.eq(predicts_gender, target_gender)
            bingo_cnt_gender += correct_num_gender.sum().cpu()
            sample_cnt_gender += pred_gender.size(0)
            running_mae_age += loss_age

        if pred_val is not None:
            running_loss_val_aro += loss_val_aro

    running_loss = running_loss / iter_cnt

    tqdm.write("[Epoch %d] Validation Loss:%.3f" % (epoch, running_loss))

    if pred_exp is not None:
        acc_exp = bingo_cnt_exp.float() / float(sample_cnt_exp)
        acc_exp = np.around(acc_exp.numpy(), 4)
        tqdm.write("[Epoch %d] Validation accuracy for Mood:%.4f." % (epoch, acc_exp))

        if pred_val is not None:
            running_loss_val_aro = running_loss_val_aro / iter_cnt
            tqdm.write("[Epoch %d] Validation Loss for Valence and Arousal:%.4f." % (epoch, running_loss_val_aro))

            if pred_gender is not None:
                acc_gender = bingo_cnt_gender.float() / float(sample_cnt_gender)
                acc_gender = np.around(acc_gender.numpy(), 4)
                tqdm.write("[Epoch %d] Validation accuracy for gender:%.4f." % (epoch, acc_gender))
                running_mae_age = running_mae_age / iter_cnt
                tqdm.write("[Epoch %d] Validation MAE for age:%.4f." % (epoch, running_mae_age))
            else:
                acc_gender = None
                running_mae_age = None
        else:
            running_loss_val_aro = None
            acc_gender = None
            running_mae_age = None
    else:
        if pred_gender is not None:
            acc_gender = bingo_cnt_gender.float() / float(sample_cnt_gender)
            acc_gender = np.around(acc_gender.numpy(), 4)
            tqdm.write("[Epoch %d] Validation accuracy for gender:%.4f." % (epoch, acc_gender))
            running_mae_age = running_mae_age / iter_cnt
            tqdm.write("[Epoch %d] Validation MAE for age:%.4f." % (epoch, running_mae_age))
            acc_exp = None
            running_loss_val_aro = None

    iter_idx = (epoch + 1) * len(data_loader)
    writer.add_scalar("Val/train_all", loss.item(), iter_idx)

    if pred_exp is not None:
        writer.add_scalar("Val/acc_expression", acc_exp, iter_idx)

    if pred_gender is not None:
        writer.add_scalar("Val/acc_gender", acc_gender, iter_idx)
        writer.add_scalar("Val/mae_age", running_mae_age, iter_idx)
    if pred_val is not None:
        writer.add_scalar("Val/loss_val_aro", running_loss_val_aro, iter_idx)
    writer.flush()

    if wandb_object:
        wandb_object.log({"Validation_Loss": running_loss})
        if pred_exp is not None:
            wandb_object.log({"Validation_Accuracy_Exp": acc_exp}, step=iter_idx)
        if pred_gender is not None:
            wandb_object.log({"Validation_Accuracy_Gender": acc_gender})
            wandb_object.log({"Validation_MAE_Age": running_mae_age})
        if pred_val is not None:
            wandb_object.log({"Val_Aro_Loss": running_loss_val_aro}, step=iter_idx)

    return acc_exp, acc_gender, running_mae_age, running_loss_val_aro


def main(args):
    if args.output_dir:
        tairvision_utils.mkdir(args.output_dir)
        writer = SummaryWriter(os.path.join(args.output_dir, 'runs'))
        store_yaml(args)

    print(args)

    wandb_object = None
    try:
        import wandb
        if not args.test_only:
            wandb_object = initialize_wandb(args)
    except:
        print("Install Wandb and start experiment tracking ASAP!")

    device = torch.device(args.device)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DANWithMoodAgeGender(num_class=args.num_class, num_head=args.num_head, use_mood=args.use_mood,
                                 use_gender=args.use_gender,
                                 use_valence=args.use_valence, use_dan=args.use_dan, backbone_path=args.backbone_path,
                                 extra_layer=args.extra_layer)

    # model = TurkcellAgeGender(num_class=args.num_class, num_head=args.num_head, use_mood=args.use_mood,
    #                           use_gender=args.use_gender,
    #                           use_valence=args.use_valence, use_dan=args.use_dan, backbone_path=args.backbone_path,
    #                           extra_layer=args.extra_layer)

    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)), ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
    ])

    # Data loading code
    print("Loading data")

    if args.dataset == 'affectnet':
        train_dataset = AffectNetMulti(os.path.join(args.data_path, 'affectnet'), split='train',
                                       transform=data_transforms,
                                       num_class=args.num_class,
                                       use_mood=args.use_mood, use_gender=args.use_gender, use_valence=args.use_valence,
                                       label_path=os.path.join(args.label_path,
                                                               'affectnet/affectnet_mood_gender_valence.csv'))

    elif args.dataset == 'utkface':
        train_dataset = UTKFace(os.path.join(args.data_path, 'UTKFace'), split='train',
                                transform=data_transforms,
                                label_path=os.path.join(args.label_path, 'UTKFace/utkface.csv'))

    elif args.dataset == 'lfw':
        train_dataset = LFWPeople(args.dataset_dir, split='train', image_set='original', transform=data_transforms)

        targets = train_dataset.targets

    print('Whole train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               sampler=ImbalancedDatasetSampler(train_dataset, use_mood=args.use_mood,
                                                                                use_gender=args.use_gender,
                                                                                use_valence=args.use_valence),
                                               shuffle=False,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.dataset == 'affectnet':
        val_dataset = AffectNetMulti(os.path.join(args.data_path, 'affectnet'), split='val',
                                     transform=data_transforms_val,
                                     num_class=args.num_class,
                                     use_mood=args.use_mood, use_gender=args.use_gender, use_valence=args.use_valence,
                                     label_path=os.path.join(args.label_path,
                                                             'affectnet/affectnet_mood_gender_valence.csv'))

    elif args.dataset == 'utkface':
        val_dataset = UTKFace(os.path.join(args.data_path, 'UTKFace'), split='val',
                              transform=data_transforms_val,
                              label_path=os.path.join(args.label_path, 'UTKFace/utkface.csv'))

    elif args.dataset == 'lfw':

        val_dataset = LFWPeople(args.dataset_dir, split='test', image_set='original',
                                transform=data_transforms_val)

    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    if args.use_mood:
        criterion_pt = PartitionLoss()
        criterion_cls = torch.nn.CrossEntropyLoss().to(device)
        criterion_af = AffinityLoss(device, num_class=args.num_class)
        if args.use_valence:
            criterion_val_aro = CCCLoss()
            if args.use_gender:
                criterion_age = AgeLoss()
            else:
                criterion_age = None
        else:
            criterion_val_aro = None
            criterion_age = None
    else:
        if args.use_gender:
            criterion_cls = torch.nn.CrossEntropyLoss().to(device)
            criterion_age = AgeLoss()
            criterion_val_aro = None
            if args.use_dan:
                criterion_af = AffinityLoss(device, num_class=2)
                criterion_pt = PartitionLoss()
            else:
                criterion_af = None
                criterion_pt = None

    loss_funcs = [criterion_cls, criterion_af, criterion_pt, criterion_age, criterion_val_aro]

    if criterion_af is not None:
        params = list(model.parameters()) + list(criterion_af.parameters())
    else:
        params = list(model.parameters())
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.utkface_finetune:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.utkface_finetune:
        model.bn.track_running_stats = False

    if args.test_only:
        acc_exp, acc_gender, mae_age, ccc_val_aro = evaluate(model, val_loader, device, 1, loss_funcs, args, writer)

        if wandb_object:
            wandb_object.finish()

        if args.use_mood:
            tqdm.write("acc_exp:" + str(acc_exp))
        if args.use_valence:
            tqdm.write("ccc_val_aro:" + str(ccc_val_aro))
        if args.use_gender:
            tqdm.write("acc_gender:" + str(acc_gender))
            tqdm.write("mae_age:" + str(mae_age))
        return

    best_acc_exp = 0
    best_acc_gender = 0
    best_mae_age = 100
    best_ccc_val_aro = 100
    for epoch in tqdm(range(args.start_epoch, args.epochs + 1)):
        train_one_epoch(model, optimizer, train_loader, device, epoch, loss_funcs, args, writer,
                        wandb_object=wandb_object)

        acc_exp, acc_gender, mae_age, ccc_val_aro = evaluate(model, val_loader, device, epoch, loss_funcs, args, writer,
                                                             wandb_object=wandb_object)

        lr_scheduler.step()

        if args.output_dir:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch
            }
            tairvision_utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            tairvision_utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))
        if acc_exp is not None:
            best_acc_exp = max(acc_exp, best_acc_exp)
        if acc_gender is not None:
            best_acc_gender = max(acc_gender, best_acc_gender)
        if mae_age is not None:
            best_mae_age = min(mae_age, best_mae_age)
        if ccc_val_aro is not None:
            best_ccc_val_aro = min(ccc_val_aro, best_ccc_val_aro)

    if args.use_mood:
        tqdm.write("best_acc_exp:" + str(best_acc_exp))
    if args.use_valence:
        tqdm.write("best_ccc_val_aro:" + str(best_ccc_val_aro))
    if args.use_gender:
        tqdm.write("best_acc_gender:" + str(best_acc_gender))
        tqdm.write("best_mae_age:" + str(best_mae_age))

    writer.close()

    if wandb_object:
        if args.use_mood:
            wandb_object.log({"best_acc_exp": best_acc_exp})
        if args.use_gender:
            wandb_object.log({"best_acc_gender": best_acc_gender})
            wandb_object.log({"best_mae_age": best_mae_age})
        if args.use_valence:
            wandb_object.log({"best_ccc_val_aro": best_ccc_val_aro})

            wandb_object.finish()


if __name__ == "__main__":
    # args = get_args_parser().parse_args()
    args = get_arguments_analysis()
    print(args)
    main(args)
