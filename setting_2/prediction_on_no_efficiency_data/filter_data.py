import os
from collections import Counter
from math import comb
from random import sample

import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from tqdm import tqdm

efficiency_dict = dict(np.load('../../data/efficiency_dict.npy', allow_pickle=True).item())
all_room_list = list(efficiency_dict.keys())
data = pd.read_csv('../../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
data = data[data.AC > 0]

data_non = data[~data.Location.isin(all_room_list)]
print(len(data_non))
print(Counter(data_non['Location']))


class AC_sparse_separate_dataset(Dataset):
    def __init__(self, mode='trn', test=False, trn_ratio=1, group_size=192, cla=False, total_number=2000,
                 verbose=False, data_path='./', room_ratio=1):
        if mode not in ['trn', 'val', 'all']:
            raise NotImplementedError("mode must be either 'trn' or 'val'")

        self.cla = cla
        self.verbose = verbose
        self.total_number = total_number
        self.group_size = group_size
        self.room_ratio = room_ratio
        self.data_path = data_path
        self.data = data_non
        self.data_without0 = self.data[self.data.AC > 0]
        self.rooms = self.data_without0['Location'].unique().tolist()
        self.training_tensor_list = []

        if mode == 'trn':
            if os.path.exists('{}trn_{}_{}_{}.npy'.format(data_path, total_number, trn_ratio, group_size)):
                self.tensor_list = np.load('{}trn_{}_{}_{}.npy'.format(data_path, total_number, trn_ratio, group_size),
                                           allow_pickle=True).tolist()
            else:
                self.rooms = [r for r in self.rooms if
                              len(self.data_without0[self.data_without0.Location == r]) >= 2 * group_size]
                room_length = {r: len(self.data_without0[self.data_without0.Location == r]) for r in
                               self.rooms}

                self.trn_sampling_number = {
                    r: room_length[r] / sum(list(room_length.values())) * self.total_number * trn_ratio for r in
                    self.rooms}
                while any(
                        self.trn_sampling_number[r] >= comb(int(trn_ratio * room_length[r]), group_size) for r in
                        self.trn_sampling_number.keys()):

                    room_to_be_poped = [r for r in self.trn_sampling_number.keys() if
                                        self.trn_sampling_number[r] >= comb(int(trn_ratio * room_length[r]),
                                                                            group_size)]
                    for r in room_to_be_poped:
                        room_length.pop(r)
                        self.rooms.remove(r)
                    self.trn_sampling_number = {
                        r: room_length[r] / sum(list(room_length.values())) * self.total_number * trn_ratio for r
                        in self.rooms
                    }

                print(self.trn_sampling_number)

                for r in tqdm(self.rooms, desc=f"Building dataset for the first time: "):
                    self.data_room = self.data_without0[self.data_without0.Location == r]
                    self.data_room['index'] = self.data_room.index
                    trn_index_list = []

                    self.train_data_room = self.data_room.sort_values(by=['index'])

                    while len(trn_index_list) < self.trn_sampling_number[r]:
                        sampled_data = self.train_data_room.sample(n=group_size).sort_values(by=['index'])
                        if sampled_data['index'].tolist() in trn_index_list:
                            continue

                        self.training_tensor_list.append((torch.tensor(sampled_data.drop(
                            ['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time', 'Location',
                             'index'], axis=1).reset_index(drop=True).to_numpy(dtype=float), dtype=torch.float), r))

                        trn_index_list.append(sampled_data['index'].tolist())

                self.training_tensor_list = shuffle(self.training_tensor_list, random_state=621)

                self.training_tensor_list = sample(self.training_tensor_list, int(total_number * trn_ratio))

                np.save('{}trn_{}_{}_{}.npy'.format(data_path, total_number, trn_ratio, group_size),
                        self.training_tensor_list)

                self.tensor_list = self.training_tensor_list

        if test:
            self.tensor_list = self.tensor_list[:100]

    def __getitem__(self, item):
        return self.tensor_list[item][0], self.tensor_list[item][1]

    def __len__(self):
        return len(self.tensor_list)


ds = AC_sparse_separate_dataset()
