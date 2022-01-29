import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset

from utils.utils import efficiency_dict


class AC_Normal_Dataset(Dataset):
    def __init__(self, mode='trn', test=False, trn_ratio=0.9):
        if mode not in ['trn', 'val']:
            raise NotImplementedError("mode must be either 'trn' or 'val'")
        self.room_date_dict = dict(np.load('../data/room_date_dict.npy', allow_pickle=True).item())
        self.room_date_list = []
        for room, value in self.room_date_dict.items():
            self.room_date_list.extend([(room, d) for d in value])

        self.room_date_list = shuffle(self.room_date_list, random_state=621)

        if test:
            self.room_date_list = self.room_date_list[:500]

        if mode == 'trn':
            self.room_date_list = self.room_date_list[:int(len(self.room_date_list) * trn_ratio)]
        else:
            self.room_date_list = self.room_date_list[int(len(self.room_date_list) * trn_ratio):]

        self.data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
        self.X = self.data.drop(['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time', 'AC'], axis=1)
        self.X['Date'] = self.data['Time'].apply(lambda x: x.split(' ')[0])
        print("Normal Regression DATALOADER: finished loading original data, total {}".format(len(self.room_date_list)))

    def __getitem__(self, index):
        if isinstance(index, int):
            room = self.room_date_list[index][0]
            date = self.room_date_list[index][1]
            return torch.tensor(self.X[(self.X.Date == date) & (self.X.Location == room)] \
                                .drop(['Location', 'Date'], axis=1).to_numpy(dtype=float), dtype=torch.float32), \
                   efficiency_dict[room]

    def __len__(self):
        return len(self.room_date_list)
