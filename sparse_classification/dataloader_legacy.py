import os.path
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append('../')
from sparse_classification.utils import efficiency_dict
from sparse_classification.utils import normal_room_list

warnings.filterwarnings("ignore")


class AC_Sparse_Dataset(Dataset):
    def __init__(self, mode='trn', test=False, trn_ratio=0.8, group_size=200, cla=False, total_number=400000,
                 verbose=False):
        if mode not in ['trn', 'val', 'all']:
            raise NotImplementedError("mode must be either 'trn' or 'val'")

        self.cla = cla
        self.verbose = verbose
        self.total_number = total_number
        self.group_size = group_size
        self.data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
        self.data_without0 = self.data[self.data.AC > 0]
        self.rooms = self.data_without0['Location'].unique()

        self.tensor_list = []

        if mode == 'trn':
            if os.path.exists('./trn_{}_{}_{}.npy'.format(total_number, trn_ratio, group_size)):
                self.tensor_list = np.load('./trn_{}_{}_{}.npy'.format(total_number, trn_ratio, group_size),
                                           allow_pickle=True).tolist()
            else:
                room_length = {r: len(self.data_without0[self.data_without0.Location == r]) for r in self.rooms}
                self.sampling_number = {r: room_length[r] / sum(list(room_length.values())) * self.total_number for r in
                                        self.rooms}

                for r in tqdm(self.rooms, desc=f"Building {mode} dataset: "):
                    self.data_room = self.data_without0[self.data_without0.Location == r]
                    self.data_room['index'] = self.data_room.index
                    self.room_tensor_list = []
                    if r not in efficiency_dict.keys():
                        if verbose:
                            print(f"Room {r}'s efficiency is not recorded")
                    elif len(self.data_room) < group_size:
                        if verbose:
                            print(f"Room {r}'s data is less than {self.group_size}")
                    else:
                        while len(self.room_tensor_list) < self.sampling_number[r]:
                            sampled_data = self.data_room.sample(n=self.group_size).sort_values(by=['index'])
                            if self.cla:
                                self.tensor_list.append((torch.tensor(sampled_data.drop(
                                    ['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time', 'Location',
                                     'index'], axis=1).reset_index(drop=True).to_numpy(dtype=float), dtype=torch.float),
                                                         int(r in normal_room_list)))
                            else:
                                self.tensor_list.append((torch.tensor(sampled_data.drop(
                                    ['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time', 'Location'],
                                    axis=1).reset_index(drop=True).to_numpy(dtype=float), dtype=torch.float),
                                                         float(efficiency_dict[r])))
                            self.room_tensor_list.append(1)
                self.tensor_list = shuffle(self.tensor_list, random_state=621)
                # seaborn.histplot(x=[i[1] for i in self.tensor_list])
                # plt.savefig(f'./dis_{self.total_number}.png')
                # plt.clf()
                if test:
                    self.tensor_list = self.tensor_list[:100]
                np.save('./val_{}_{}_{}.npy'.format(total_number, trn_ratio, group_size),
                        self.tensor_list[int(len(self.tensor_list) * trn_ratio):])
                self.tensor_list = self.tensor_list[:int(len(self.tensor_list) * trn_ratio)]
                np.save('./trn_{}_{}_{}.npy'.format(total_number, trn_ratio, group_size), self.tensor_list)
        elif mode == 'val':
            print("Loading validation set")
            self.tensor_list = list(
                np.load('./val_{}_{}_{}.npy'.format(total_number, trn_ratio, group_size), allow_pickle=True).tolist())
            print("Validation set Loaded!\n\n")
            if test:
                self.tensor_list = self.tensor_list[:100]

    def __getitem__(self, item):
        return self.tensor_list[item][0], self.tensor_list[item][1]

    def __len__(self):
        return len(self.tensor_list)
