import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import normal_room_list, poor_room_list

data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
triplet_csv = pd.DataFrame(columns=['anchor', 'pos', 'neg'])
X = data.drop(['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time'], axis=1)
Y = data['AC']
X['Date'] = data['Time'].apply(lambda x: x.split(' ')[0])
dates = X['Date'].unique()
rooms = normal_room_list + poor_room_list

if os.path.exists('../data/room_date_dict.npy'):
    room_data_dict = np.load('../data/room_date_dict.npy', allow_pickle=True).item()
    print('Loading pre-generated room date summary dict, no sample applied.')
    print(room_data_dict.keys(), list(room_data_dict.values())[0])
else:
    room_data_dict = {r: [] for r in rooms}
    room_data_dict_ac0_filtered = {r: [] for r in rooms}
    for r in tqdm(rooms, desc="filtering data samples for ALL rooms: "):
        for d in tqdm(dates):
            if len(X[(X.Date == d) & (X.Location == r)]) == 48:
                room_data_dict[r].append(d)
                if any(
                        i != 0 for i in list(X[(X.Date == d) & (X.Location == r)])
                ):
                    room_data_dict_ac0_filtered[r].append(d)
    np.save('../data/room_date_dict.npy', room_data_dict)
    np.save('../data/room_date_dict_ac0_filtered.npy', room_data_dict_ac0_filtered)
