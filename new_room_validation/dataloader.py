import os.path
import warnings
from math import comb
from random import sample

import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import efficiency_dict
from utils import normal_room_list

warnings.filterwarnings("ignore")


class AC_sparse_separate_dataset(Dataset):
    def __init__(self, mode='trn', test=False, trn_ratio=0.8, group_size=48, cla=False, total_number=100000,
                 verbose=False, data_path='./data/', room_ratio=0.8):
        if mode not in ['trn', 'val', 'all']:
            raise NotImplementedError("mode must be either 'trn' or 'val'")

        self.cla = cla
        self.verbose = verbose
        self.total_number = total_number
        self.group_size = group_size
        self.room_ratio = room_ratio
        self.data_path = data_path
        self.data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
        self.data_without0 = self.data[self.data.AC > 0]

        self.training_tensor_list = []
        self.validation_tensor_list = []

        if not os.path.exists(data_path):
            os.mkdir(data_path)
        if mode == 'trn':
            if os.path.exists('{}trn_{}_{}_{}.npy'.format(data_path, total_number, trn_ratio, group_size)):
                self.tensor_list = np.load('{}trn_{}_{}_{}.npy'.format(data_path, total_number, trn_ratio, group_size),
                                           allow_pickle=True).tolist()
            else:
                self.room_length = {r: len(self.data_without0[self.data_without0.Location == r]) for r in
                                    self.data_without0['Location'].unique()}
                self.total_rooms = [i for i in self.data_without0['Location'].unique().tolist() if
                                    i in efficiency_dict.keys() and self.room_length[i] >= 5 * group_size]
                self.available_room_length = {r: self.room_length[r] for r in self.total_rooms}
                self.room_maximum_number = {r: comb(self.room_length[r], group_size) for r in self.total_rooms}
                self.training_rooms = sample(self.total_rooms, int(room_ratio * len(self.total_rooms)))
                self.validation_rooms = [i for i in self.total_rooms if i not in self.training_rooms]
                self.trn_sampling_number = {
                    r: self.room_length[r] / sum(
                        [self.available_room_length[rr] for rr in self.training_rooms]) * trn_ratio * total_number for r
                    in self.training_rooms}
                self.val_sampling_number = {
                    r: self.room_length[r] / sum([self.available_room_length[rr] for rr in self.validation_rooms]) * (
                            1 - trn_ratio) * total_number for r in self.validation_rooms}

                while not (sum([comb(self.room_length[r], group_size) for r in
                                self.training_rooms]) >= total_number * trn_ratio and sum(
                    [comb(self.room_length[r], group_size) for r in self.validation_rooms]) >= total_number * (
                                   1 - trn_ratio) and all(
                    self.trn_sampling_number[r] < self.room_maximum_number[r] for r in self.training_rooms) and all(
                    self.val_sampling_number[r] < self.room_maximum_number[r] for r in self.validation_rooms)):
                    self.training_rooms = sample(self.total_rooms, int(room_ratio * len(self.total_rooms)))
                    self.validation_rooms = [i for i in self.total_rooms if i not in self.training_rooms]
                    self.trn_sampling_number = {
                        r: self.room_length[r] / sum(
                            [self.available_room_length[rr] for rr in self.training_rooms]) * trn_ratio * total_number
                        for r in self.training_rooms}
                    self.val_sampling_number = {
                        r: self.room_length[r] / sum(
                            [self.available_room_length[rr] for rr in self.validation_rooms]) * (
                                   1 - trn_ratio) * total_number for r in self.validation_rooms}
                print(self.trn_sampling_number)
                print(self.val_sampling_number)
                for r in tqdm(self.training_rooms, desc=f"Building dataset from training rooms: "):
                    self.data_room = self.data_without0[self.data_without0.Location == r]
                    self.data_room['index'] = self.data_room.index
                    trn_index_list = []
                    while len(trn_index_list) < self.trn_sampling_number[r]:
                        sampled_data = self.data_room.sample(n=group_size).sort_values(by=['index'])
                        if sampled_data['index'].tolist() in trn_index_list:
                            continue
                        if self.cla:
                            self.training_tensor_list.append((torch.tensor(sampled_data.drop(
                                ['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time', 'Location',
                                 'index'], axis=1).reset_index(drop=True).to_numpy(dtype=float), dtype=torch.float),
                                                              int(r in normal_room_list)))
                        else:
                            self.training_tensor_list.append((torch.tensor(sampled_data.drop(
                                ['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time', 'Location',
                                 'index'], axis=1).reset_index(drop=True).to_numpy(dtype=float), dtype=torch.float),
                                                              float(efficiency_dict[r])))
                        trn_index_list.append(sampled_data['index'].tolist())

                for r in tqdm(self.validation_rooms, desc=f"Building dataset from validation rooms: "):
                    self.data_room = self.data_without0[self.data_without0.Location == r]
                    self.data_room['index'] = self.data_room.index
                    val_index_list = []
                    while len(val_index_list) < self.val_sampling_number[r]:
                        sampled_data = self.data_room.sample(n=group_size).sort_values(by=['index'])
                        if sampled_data['index'].tolist() in val_index_list:
                            continue
                        if self.cla:
                            self.validation_tensor_list.append((torch.tensor(sampled_data.drop(
                                ['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time', 'Location',
                                 'index'], axis=1).reset_index(drop=True).to_numpy(dtype=float), dtype=torch.float),
                                                                int(r in normal_room_list)))
                        else:
                            self.validation_tensor_list.append((torch.tensor(sampled_data.drop(
                                ['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time', 'Location',
                                 'index'], axis=1).reset_index(drop=True).to_numpy(dtype=float), dtype=torch.float),
                                                                float(efficiency_dict[r])))
                        val_index_list.append(sampled_data['index'].tolist())

                self.training_tensor_list = shuffle(self.training_tensor_list, random_state=621)
                self.validation_tensor_list = shuffle(self.validation_tensor_list, random_state=621)

                np.save('{}val_{}_{}_{}.npy'.format(data_path, total_number, trn_ratio, group_size),
                        self.validation_tensor_list)
                np.save('{}trn_{}_{}_{}.npy'.format(data_path, total_number, trn_ratio, group_size),
                        self.training_tensor_list)

                self.tensor_list = self.training_tensor_list

        elif mode == 'val':
            print("Loading validation set")
            self.tensor_list = list(
                np.load('{}val_{}_{}_{}.npy'.format(data_path, total_number, trn_ratio, group_size),
                        allow_pickle=True).tolist())
            print("Validation set Loaded!\n\n")

        if test:
            self.tensor_list = self.tensor_list[:100]

    def __getitem__(self, item):
        return self.tensor_list[item][0], self.tensor_list[item][1]

    def __len__(self):
        return len(self.tensor_list)
