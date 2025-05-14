import os
from shutil import copy

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd

from tairvision.datasets.affectnet import AffectNet, UTKFace, AffectNetWithValenceArousal, \
    AffectNetWithMoodValenceArousal, AffectNetWithMood, AffectNetMulti





class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None, use_mood=False, use_gender=False,
                 use_valence=False):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        if use_gender:
            df["gender"], df["age"] = self._get_genders(dataset)
        if use_valence:
            df["valence"], df["arousal"] = self._get_valence(dataset)
        # if use_mood:
        #     df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        weights = 1.0 / label_to_count[df["label"]]
        self.weights = torch.DoubleTensor(weights.to_list())

        # self.weights = self.weights.clamp(min=1e-5)

    def _get_labels(self, dataset):
        if isinstance(dataset, AffectNet):
            return dataset.data.label.to_list()
        elif isinstance(dataset, AffectNetMulti):
            return dataset.data.label.to_list()
        elif isinstance(dataset, UTKFace):
            return dataset.data.label.to_list()
        else:
            raise NotImplementedError

    def _get_genders(self, dataset):
        if isinstance(dataset, UTKFace):
            return dataset.data.gender.to_list(), dataset.data.age.to_list()
        if isinstance(dataset, AffectNetMulti):
            return dataset.data.gender.to_list(), dataset.data.age.to_list()

    def _get_valence(self, dataset):
        if isinstance(dataset, AffectNetMulti):
            return dataset.data.valence.to_list(), dataset.data.arousal.to_list()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def convert_affectnet(cache_dir='./.cache'):
    df = pd.read_csv(os.path.join(cache_dir, 'affectnet.csv'))

    for i in range(8):
        for j in ['train_set', 'val_set']:
            os.makedirs(os.path.join(cache_dir, f'affectnet/{j}/{i}'), exist_ok=True)

    for i, row in df.iterrows():
        p = row['split']
        l = row['label']
        copy(row['img_path'], os.path.join(cache_dir, f'affectnet/{p}_set/{l}'))

    print('convert done.')


def get_dist(age_array, K=10):
    batch_size = age_array.shape[0]
    nb_bins = 120 // K

    lower_bound = (age_array // K) * K
    higher_bound = ((age_array // K) + 1) * K

    lower_coeff = (higher_bound - age_array) / K
    higher_coeff = (age_array - lower_bound) / K

    age_dist = torch.zeros(batch_size, nb_bins)

    for ind in range(batch_size):
        age_dist[ind, lower_bound[ind] // K] = lower_coeff[ind]
        age_dist[ind, higher_bound[ind] // K] = higher_coeff[ind]

    return age_dist


def initialize_wandb(args):
    import wandb

    yaml_location: str = args.cfg
    yaml_name = yaml_location.split('/')[-2]

    config_dict = vars(args)

    # if args.wandb_id:
    #     experiment_id = args.wandb_id
    # else:
    import datetime

    # name = str(args.output_dir)

    current_time = datetime.datetime.now()
    experiment_id = "{:4d}{:02d}{:02d}{:02d}{:02d}".format(current_time.year, current_time.month, current_time.day,
                                                           current_time.hour, current_time.minute)

    wandb_settings_dict = {'project': "face-analysis",
                           'entity': 'tair-face',
                           'resume': 'allow',
                           'name': yaml_name,
                           'id': experiment_id,
                           'config': config_dict}

    # if args.distributed:
    #     if args.rank == 0:
    #         wandb_object = wandb.init(**wandb_settings_dict)
    # else:
    wandb_object = wandb.init(**wandb_settings_dict)

    return wandb_object
