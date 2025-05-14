import os
import glob
import csv

from PIL import Image
import numpy as np
import pandas as pd

import torch.utils.data as data


class AffectNet(data.Dataset):
    def __init__(self, data_path, split, transform=None, num_class=8,
                 label_path='/datasets/affectnet/affectnet_mood_gender_valence.csv'):
        self.split = split
        self.transform = transform
        self.data_path = data_path

        if os.path.exists(label_path):
            df = pd.read_csv(label_path)
        else:
            df = self.get_df()
            df.to_csv(label_path)

        self.data = df[df['split'] == split]
        label = self.data.loc[:, 'label'].values
        self.data = self.data[label < num_class]

        self.file_paths = self.data.loc[:, 'img_path'].values
#        self.label = self.data.loc[:, 'label'].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {split} samples: {self.sample_counts}')

    def get_df(self):
        path = os.path.join(self.data_path, self.split + '_set/')
        data = []

        for anno in glob.glob(path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(path, f'images/{idx}.jpg')
            label = int(np.load(anno))
            data.append([img_path, label])

        return pd.DataFrame(data=data, columns=['img_path', 'label'])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class AffectNetWithMoodValenceArousal(data.Dataset):
    def __init__(self, data_path, split, transform=None, num_class=8,
                 label_path='/datasets/affectnet/affectnet_mood_gender_valence.csv'):
        self.split = split
        self.transform = transform
        self.data_path = data_path
        self.label_path = label_path

        # if os.path.exists(label_path):
        df = pd.read_csv(label_path)
        # else:
        # df = self.get_df()
        # df.to_csv(label_path)



        self.data = df[df['split'] == split]
        label = self.data.loc[:, 'label'].values
        self.data= self.data[label < num_class]

        self.file_paths = self.data.loc[:, 'img_path'].values
        self.label = self.data.loc[:, 'label'].values
        self.valence = self.data.loc[:, 'valence'].values
        self.arousal = self.data.loc[:, 'arousal'].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {split} samples: {self.sample_counts}')

    def get_df(self):
        path = os.path.join(self.data_path, self.split + '_set/')
        data = []

        for anno in glob.glob(path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(path, f'images/{idx}.jpg')
            label = int(np.load(anno))

            if os.path.isfile(path + 'annotations/' + str(idx) + '_val.npy'):
                valence = np.load(path + 'annotations/' + str(idx) + '_val.npy')
                arousal = np.load(path + 'annotations/' + str(idx) + '_aro.npy')
                val = valence.item()
                aro = arousal.item()
            else:
                val = 5
                aro = 5
            data.append([img_path, label, val, aro])

        return pd.DataFrame(data=data, columns=['img_path', 'label', 'valence', 'arousal'])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]
        valence = self.valence[idx]
        arousal = self.arousal[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, valence, arousal


class AffectNetWithMood(data.Dataset):
    def __init__(self, data_path, split, transform=None, num_class=8,
                 label_path='/datasets/affectnet/affectnet_mood_gender_valence.csv'):
        self.split = split
        self.transform = transform
        self.data_path = data_path
        self.label_path = label_path

        # if os.path.exists(label_path):
        df = pd.read_csv(label_path)
        # else:
        # df = self.get_df()
        # df.to_csv(label_path)



        self.data = df[df['split'] == split]
        label = self.data.loc[:, 'label'].values
        self.data= self.data[label < num_class]

        self.file_paths = self.data.loc[:, 'img_path'].values
        self.label = self.data.loc[:, 'label'].values
        self.valence = self.data.loc[:, 'valence'].values
        self.arousal = self.data.loc[:, 'arousal'].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {split} samples: {self.sample_counts}')

    def get_df(self):
        path = os.path.join(self.data_path, self.split + '_set/')
        data = []

        for anno in glob.glob(path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(path, f'images/{idx}.jpg')
            label = int(np.load(anno))

            if os.path.isfile(path + 'annotations/' + str(idx) + '_val.npy'):
                valence = np.load(path + 'annotations/' + str(idx) + '_val.npy')
                arousal = np.load(path + 'annotations/' + str(idx) + '_aro.npy')
                val = valence.item()
                aro = arousal.item()
            else:
                val = 5
                aro = 5
            data.append([img_path, label, val, aro])

        return pd.DataFrame(data=data, columns=['img_path', 'label', 'valence', 'arousal'])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]
        valence = self.valence[idx]
        arousal = self.arousal[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class AffectNetWithValenceArousal(data.Dataset):
    def __init__(self, data_path, split, transform=None, num_class=8,
                 label_path='/datasets/affectnet/affectnet_mood_gender_valence.csv'):
        self.split = split
        self.transform = transform
        self.data_path = data_path
        self.label_path = label_path

        # if os.path.exists(label_path):
        df = pd.read_csv(label_path)
        # else:
        # df = self.get_df()
        # df.to_csv(label_path)



        self.data = df[df['split'] == split]
        label = self.data.loc[:, 'label'].values
        self.data= self.data[label < num_class]

        self.file_paths = self.data.loc[:, 'img_path'].values
        self.label = self.data.loc[:, 'label'].values
        self.valence = self.data.loc[:, 'valence'].values
        self.arousal = self.data.loc[:, 'arousal'].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {split} samples: {self.sample_counts}')

    def get_df(self):
        path = os.path.join(self.data_path, self.split + '_set/')
        data = []

        for anno in glob.glob(path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(path, f'images/{idx}.jpg')
            label = int(np.load(anno))

            if os.path.isfile(path + 'annotations/' + str(idx) + '_val.npy'):
                valence = np.load(path + 'annotations/' + str(idx) + '_val.npy')
                arousal = np.load(path + 'annotations/' + str(idx) + '_aro.npy')
                val = valence.item()
                aro = arousal.item()
            else:
                val = 5
                aro = 5
            data.append([img_path, label, val, aro])

        return pd.DataFrame(data=data, columns=['img_path','valence', 'arousal'])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]
        valence = self.valence[idx]
        arousal = self.arousal[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, valence, arousal



class AffectNetMulti(data.Dataset):
    def __init__(self, data_path, split, transform=None, num_class=7, use_mood=False, use_gender=False, use_valence=False,
                 label_path='/datasets/affectnet/affectnet_mood_gender_valence.csv'):
        self.split = split
        self.transform = transform
        self.data_path = data_path
        self.label_path = label_path

        self.use_mood = use_mood
        self.use_gender = use_gender
        self.use_valence = use_valence

        # if os.path.exists(label_path):
        df = pd.read_csv(label_path)
        # else:
        # df = self.get_df()
        # df.to_csv(label_path)

        self.data = df[df['split'] == split]
        label = self.data.loc[:, 'label'].values
        self.data= self.data[label < num_class]

        self.file_paths = self.data.loc[:, 'img_path'].values
        self.label = self.data.loc[:, 'label'].values
        self.valence = self.data.loc[:, 'valence'].values
        self.arousal = self.data.loc[:, 'arousal'].values
        self.gender = self.data.loc[:, 'gender'].values == 'Female'
        self.age = self.data.loc[:, 'age'].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {split} samples: {self.sample_counts}')

    def get_df(self):
        path = os.path.join(self.data_path, self.split + '_set/')
        data = []

        for anno in glob.glob(path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(path, f'images/{idx}.jpg')
            label = int(np.load(anno))

            if os.path.isfile(path + 'annotations/' + str(idx) + '_val.npy'):
                valence = np.load(path + 'annotations/' + str(idx) + '_val.npy')
                arousal = np.load(path + 'annotations/' + str(idx) + '_aro.npy')

                val = valence.item()
                aro = arousal.item()
            else:
                val = 5
                aro = 5
            data.append([img_path, label, val, aro])

        return pd.DataFrame(data=data, columns=['img_path', 'label', 'gender', 'age', 'valence', 'arousal'])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]
        valence = self.valence[idx]
        arousal = self.arousal[idx]
        gender = self.gender[idx]
        age = self.age[idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.use_mood:
            if self.use_valence:
                if self.use_gender:
                    return image, label, gender, age, valence, arousal
                else:
                    return image, label, valence, arousal
            else:
                return image, label
        else:
            if self.use_gender:
                return image, gender, age


#class AffectNetWithAgeGender(data.Dataset):
#    def __init__(self, data_path, split, transform=None, num_class=8,
#                 label_path='/datasets/affectnet/affectnet_mood_gender_valence.csv'):
#        self.split = split
#        self.transform = transform
#        self.data_path = data_path
#
#        if os.path.exists(label_path):
#            df = pd.read_csv(label_path)
#        else:
#            df = self.get_df()
#            df.to_csv(label_path)
#
#        self.data = df[df['split'] == split]
#        label = self.data.loc[:, 'label'].values
#        self.data= self.data[label < num_class]
#
#        self.file_paths = self.data.loc[:, 'img_path'].values
#        self.label = self.data.loc[:, 'label'].values
#        self.gender = self.data.loc[:, 'gender'].values == 'Female'
#        self.age = self.data.loc[:, 'age'].values
#
#        _, self.sample_counts = np.unique(self.label, return_counts=True)
#        # print(f' distribution of {split} samples: {self.sample_counts}')
#
#    def get_df(self):
#        path = os.path.join(self.data_path, self.split + '_set/')
#        data = []
#
#        for anno in glob.glob(path + 'annotations/*_exp.npy'):
#            idx = os.path.basename(anno).split('_')[0]
#            img_path = os.path.join(path, f'images/{idx}.jpg')
#            label = int(np.load(anno))
#            data.append([img_path, label])
#
#        return pd.DataFrame(data=data, columns=['img_path', 'label'])
#
#    def __len__(self):
#        return len(self.file_paths)
#
#    def __getitem__(self, idx):
#        path = self.file_paths[idx]
#        image = Image.open(path).convert('RGB')
#        label = self.label[idx]
#        gender = self.gender[idx]
#        age = self.age[idx]
#
#        if self.transform is not None:
#            image = self.transform(image)
#
#        return image, label, gender, age


class UTKFace(data.Dataset):
    def __init__(self, data_path, split, transform=None,
                 label_path='/datasets/UTKFace/utkface.csv'):
        self.split = split
        self.transform = transform
        self.data_path = data_path

        if os.path.exists(label_path):
            df = pd.read_csv(label_path)
        else:
            label_path = label_path.rstrip('utkface.csv')
            create_csv_from_utk(save_path=label_path)
            df = pd.read_csv(os.path.join(label_path, 'utkface.csv'))

        self.data = df[df['split'] == split]

        self.file_paths = self.data.loc[:, 'img_path'].values
        self.label = self.data.loc[:, 'label'].values
        self.gender = self.data.loc[:, 'gender'].values == 'Female'
        self.age = self.data.loc[:, 'age'].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {split} samples: {self.sample_counts}')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]
        gender = self.gender[idx]
        age = self.age[idx]

        if self.transform is not None:
            image = self.transform(image)

#        return image, label, gender, age
        return image, gender, age


def create_csv_from_utk(dataset_path='/datasets/utkface', save_path='/datasets/utkface'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, 'utkface.csv')
    data_path = os.path.join(dataset_path, 'images')
    csv_file = open(save_path, 'w', encoding='UTF8', newline='')

    writer = csv.writer(csv_file)
    header = ['', 'split', 'img_path', 'label', 'gender', 'age']
    writer.writerow(header)

    dataset_len = len(os.listdir(data_path))

    for i_file, file_name in enumerate(os.listdir(data_path)):
        attribure_list = file_name.split('_')
        if len(attribure_list) == 4:
            age, gender, race, _ = file_name.split('_')
        else:
            continue
        split = 'train' if i_file < dataset_len * 0.7 else 'val'
        img_path = os.path.join(data_path, file_name)
        label = 0  # neutral for all face
        gender = 'Male' if gender == '0' else 'Female'
        age = int(age)
        age = age if age < 110 else 109
        row = [i_file, split, img_path, label, gender, age]
        writer.writerow(row)

    csv_file.close()
