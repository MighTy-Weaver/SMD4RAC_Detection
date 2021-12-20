import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import normal_room_list


class AC_Triplet_Dataset(Dataset):
    def __init__(self, mode='trn', sample_frac=0.5, csv_path='./triplet.csv'):
        """
        The Triplet Dataset for Contrastive Learning Training Process. getitem returns a triplet (anchor, pos, neg)
        :param mode: 'trn' or 'val'
        :param sample_frac: the fraction used to sample from all triplets for training/validation
        :param csv_path: The path to the triplet csv file
        """
        if mode not in ['trn', 'val']:
            raise NotImplementedError("mode can only be 'trn' or 'val'!")
        self.mode = mode

        self.triplet_csv = pd.read_csv(csv_path, index_col=None).sample(frac=sample_frac).reset_index(drop=True)
        print("Triplet DATALOADER: finished loading triplet.csv, total {} triples.".format(len(self.triplet_csv)))
        self.data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
        self.X = self.data.drop(['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time', 'AC'], axis=1)
        self.X['Date'] = self.data['Time'].apply(lambda x: x.split(' ')[0])
        print("Triplet DATALOADER: finished loading original data")

    def __getitem__(self, index):
        if isinstance(index, int):
            anchor_date = self.triplet_csv.loc[index, 'anchor']
            anchor_room = self.triplet_csv.loc[index, 'anchor_room']
            pos_date = self.triplet_csv.loc[index, 'pos']
            pos_room = self.triplet_csv.loc[index, 'pos_room']
            neg_date = self.triplet_csv.loc[index, 'neg']
            neg_room = self.triplet_csv.loc[index, 'neg_room']
            return self.X[(self.X.Date == anchor_date) & (self.X.Location == anchor_room)] \
                       .drop(['Location', 'Date'], axis=1).to_numpy(dtype=float), \
                   self.X[(self.X.Date == pos_date) & (self.X.Location == pos_room)] \
                       .drop(['Location', 'Date'], axis=1).to_numpy(dtype=float), \
                   self.X[(self.X.Date == neg_date) & (self.X.Location == neg_room)] \
                       .drop(['Location', 'Date'], axis=1).to_numpy(dtype=float)
        elif isinstance(index, slice):
            anchor_date = list(self.triplet_csv.loc[index, 'anchor'])
            anchor_room = list(self.triplet_csv.loc[index, 'anchor_room'])
            pos_date = list(self.triplet_csv.loc[index, 'pos'])
            pos_room = list(self.triplet_csv.loc[index, 'pos_room'])
            neg_date = list(self.triplet_csv.loc[index, 'neg'])
            neg_room = list(self.triplet_csv.loc[index, 'neg_room'])
            return self.X[(self.X.Date.isin(anchor_date)) & (self.X.Location.isin(anchor_room))] \
                       .drop(['Location', 'Date'], axis=1).to_numpy(dtype=float), \
                   self.X[(self.X.Date.isin(pos_date)) & (self.X.Location.isin(pos_room))] \
                       .drop(['Location', 'Date'], axis=1).to_numpy(dtype=float), \
                   self.X[(self.X.Date.isin(neg_date)) & (self.X.Location.isin(neg_room))] \
                       .drop(['Location', 'Date'], axis=1).to_numpy(dtype=float)
        else:
            raise TypeError('Should call dataloader object with int or slice!')

    def __len__(self):
        return len(self.triplet_csv)


class AC_Normal_Dataset(Dataset):
    def __init__(self, mode='trn'):
        if mode not in ['trn', 'val']:
            raise NotImplementedError("mode must be either 'trn' or 'val'")
        self.room_date_dict = dict(np.load('../data/room_date_dict.npy', allow_pickle=True).item())
        self.room_date_list = []
        for room, value in self.room_date_dict.items():
            self.room_date_list.extend([(room, d) for d in value])

        if mode == 'trn':
            self.room_date_list = self.room_date_list[:int(len(self.room_date_list) * 0.9)]
        else:
            self.room_date_list = self.room_date_list[int(len(self.room_date_list) * 0.9):]

        self.data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
        self.X = self.data.drop(['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time', 'AC'], axis=1)
        self.X['Date'] = self.data['Time'].apply(lambda x: x.split(' ')[0])
        print("Normal DATALOADER: finished loading original data")

    def __getitem__(self, index):
        if isinstance(index, int):
            room = self.room_date_list[index][0]
            date = self.room_date_list[index][1]
            return torch.tensor(self.X[(self.X.Date == date) & (self.X.Location == room)] \
                                .drop(['Location', 'Date'], axis=1).to_numpy(dtype=float), dtype=torch.float32), int(
                room in normal_room_list)

    def __len__(self):
        return len(self.room_date_list)
