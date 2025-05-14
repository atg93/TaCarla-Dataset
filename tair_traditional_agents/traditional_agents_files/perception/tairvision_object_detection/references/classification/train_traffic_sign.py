import torchvision
import sys
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import glob
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision.transforms as TF
import torch
from datasets.traffic_merged_dataset import edaDataset
import argparse

parser = argparse.ArgumentParser(description='Train Code for Traffic Sign Classification')
parser.add_argument('--out-path', dest='out_path', default='/home/ig21/eda_cls/out.pth')
args = parser.parse_args()

def train_one_epoch(epoch_index, tb_writer, model, optimizer, criterion, train_loader, device):
    running_loss = 0.
    last_loss = 0.
    train_accu = []
    total1 = 0
    correct1 = 0
    accu1 = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _1, predicted1 = torch.max(outputs.data, 1)
        total1 += labels.size(0)
        correct1 += (predicted1 == labels).sum().item()
        accu1 = 100. * correct1 / total1
        train_accu.append(accu1)

        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    print('Train Correct:', correct1)
    print('Train Total:', total1)
    print('Train Accuracy:', "%.2f" % accu1)

    return last_loss

def main(args):

    EPOCHS = 5
    batch_size = 4
    shuffle = False
    pin_memory = False
    num_workers = 2
    epoch_number = 0
    best_vloss = 1_000_000.
    device = torch.device('cuda:3')

    path = '/home/ig21/eda_cls'
    trainpath = '/home/ig21/GTSRB_EDA/TRAIN_others'
    testpath = '/home/ig21/GTSRB_EDA/TEST_others'

    mean_nums = [0.3337, 0.3064, 0.3171]
    std_nums = [0.2672, 0.2564, 0.2629]

    transforms_test = TF.Compose([
        TF.Resize((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean_nums, std_nums)
    ])

    random_transforms = [
        TF.RandomApply([], p=0),
        TF.ColorJitter(brightness=(0.875, 1.125)),
        TF.ColorJitter(saturation=(0.5, 1.5)),
        TF.ColorJitter(contrast=(0.5, 1.5)),
        TF.ColorJitter(hue=(-0.05, 0.05)),
        TF.RandomRotation(15)
    ]

    transforms = TF.Compose([
        TF.RandomChoice(random_transforms),
        TF.Resize((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean_nums, std_nums)
    ])

    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 46)

    model.to(device)

    train_image_paths = []

    train_paths = glob.glob(trainpath + "/*.png")
    test_paths = glob.glob(testpath + "/*.png")

    classes = []
    for i in range(len(train_paths)):
        label = train_paths[i].split('/')[-1]
        label = label.split('_')[0]
        label = int(label)
        classes.append(label)

        train_image_paths.append(train_paths[i])

    test_image_paths = []

    classes_test = []
    for i in range(len(test_paths)):
        label_test = test_paths[i].split('/')[-1]
        label_test = label_test.split('_')[0]
        label_test = int(label_test)
        classes_test.append(label_test)

        test_image_paths.append(test_paths[i])

    train_dataset = edaDataset(train_image_paths, transform=transforms)
    val_dataset = edaDataset(test_image_paths, transform=transforms_test)

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)
    validation_loader = DataLoader(dataset=val_dataset, shuffle=shuffle, batch_size=1, num_workers=num_workers,
                                   pin_memory=pin_memory)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))


    for epoch in range(EPOCHS):
        vtotal = 0
        vcorrect = 0
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train()
        avg_loss = train_one_epoch(epoch_number, writer, model, optimizer, criterion, train_loader, device)

        model.eval()

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)

            with torch.no_grad():
                voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss

            v_, vpredicted = torch.max(voutputs.data, 1)
            vtotal += vlabels.size(0)
            vcorrect += (vpredicted == vlabels).sum().item()
            accu2 = 100. * vcorrect / vtotal

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('Val Correct:', vcorrect)
        print('Val Total:', vtotal)
        acc2 = 100 * (vcorrect / vtotal)
        print('Val Accuracy:', "%.2f" % acc2)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), args.out_path)

        epoch_number += 1

if __name__ == "__main__":
    main(args)










