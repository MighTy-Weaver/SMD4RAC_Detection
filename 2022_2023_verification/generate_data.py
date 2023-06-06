import os
import sys
import warnings
from math import comb
from random import sample

import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.append('../')
from setting_1.utils import efficiency_dict

replaced_room_dict = {
    324: '2022/4/28',
    619: '2022/4/28',
    916: '2022/4/28',
    911: '2022/6/15',
    632: '2022/6/24',
    1004: '2022/6/24',
    301: '2022/7/14',
    1014: '2022/7/14',
    304: '2022/8/23',
    633: '2022/8/23',
    805: '2022/8/23',
    909: '2022/8/23',
    1006: '2022/8/23',
    1010: '2022/8/23',
    306: '2022/9/14',
    635: '2022/9/23'
}

replaced_room_category_dict = {
    324: 'Poor',
    619: 'Estimated Normal',
    916: 'Estimated Poor',
    911: 'Normal',
    632: 'Poor',
    1004: 'Estimated Normal',
    301: 'Poor',
    1014: 'Normal',
    304: 'Poor',
    633: 'Poor',
    805: 'Normal',
    909: 'Normal',
    1006: 'Poor',
    1010: 'Poor',
    306: 'Poor',
    635: 'Estimated Poor'
}

total_data = pd.read_csv('./FINAL_total_csv.csv', index_col=None)
print('Before dropping 0:', len(total_data))
total_data = total_data[total_data.AC > 0.01].reset_index(drop=True)
print('After dropping 0:', len(total_data))

total_rooms = [i for i in total_data['Location'].unique() if
               i in efficiency_dict.keys() or int(i) in replaced_room_dict.keys()]

print('Total rooms:', len(total_rooms), total_rooms)


class AC_sparse_separate_dataset(Dataset):
    def __init__(self, data, data_path, mode='trn', test=False, trn_ratio=1, group_size=24, cla=False, total_number=20,
                 verbose=False, room_ratio=1):
        if mode not in ['trn', 'val', 'all']:
            raise NotImplementedError("mode must be either 'trn' or 'val'")

        self.cla = cla
        self.verbose = verbose
        self.total_number = total_number
        self.group_size = group_size
        self.room_ratio = room_ratio
        self.data = data
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
                room_length = {r: len(self.data_without0[self.data_without0.Location == r]) for r in self.rooms}

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

                        self.training_tensor_list.append((torch.tensor(
                            sampled_data.drop(['Time', 'Location', 'index'], axis=1).reset_index(drop=True).to_numpy(
                                dtype=float), dtype=torch.float), r))

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


if __name__ == '__main__':
    for r in tqdm(total_rooms):
        data = pd.read_csv('./room_data_csv/{}.csv'.format(r), index_col=None)
        data = data[data.AC >= 0.01]
        data = data.dropna().reset_index(drop=True)
        data['Time'] = data['Time'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
        # Weekday,Location,Total,AC,Lighting,Socket,WaterHeater,Time,Temperature,Irradiance,Precipitation,Humidity,Prev_one,Prev_three,Prev_five,Prev_one_on,Prev_two_on,Next_one_on,Next_two_on
        if r in replaced_room_dict:
            replace_date = pd.to_datetime(replaced_room_dict[r], format='%Y/%m/%d')
            data = data[data['Time'] < replace_date].sort_values(by=['Time']).reset_index(drop=True)
        data = data[
            ['Location', 'AC', 'Time', 'temperature', 'irradiance', 'precipitation', 'humidity', 'Prev_one',
             'Prev_three', 'Prev_five', 'Prev_one_on', 'Prev_two_on', 'Next_one_on', 'Next_two_on']]
        print(len(data))
        print(data.head(20))
        try:
            AC_sparse_separate_dataset(data=data, data_path='./room_data_csv/{}'.format(r), total_number=20)
        except ValueError:
            print("Room {} has less than 100 rows".format(r))
            continue
