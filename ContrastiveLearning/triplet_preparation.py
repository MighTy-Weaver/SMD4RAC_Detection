import argparse
from itertools import product
from random import sample

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import normal_room_list
from utils import poor_room_list

parser = argparse.ArgumentParser()
parser.add_argument('--frac', default=0.4, type=float, help="Number of fraction to sample from all triplets")
args = parser.parse_args()

sample_frac = args.frac

print(normal_room_list)
print(poor_room_list)

data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
triplet_csv = pd.DataFrame(columns=['anchor', 'pos', 'neg'])
X = data.drop(['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time'], axis=1)
Y = data['AC']
X['Date'] = data['Time'].apply(lambda x: x.split(' ')[0])
dates = X['Date'].unique()
rooms = normal_room_list + poor_room_list

room_data_dict = {r: [] for r in rooms}
for r in tqdm(normal_room_list + poor_room_list, desc="filtering data samples for ALL rooms: "):
    for d in tqdm(dates):
        if len(X[(X.Date == d) & (X.Location == r)]) == 48:
            room_data_dict[r].append(d)
np.save('../data/room_date_dict.npy', room_data_dict)
room_data_dict_sample = {r: sample(room_data_dict[r], int(len(room_data_dict[r]) * sample_frac)) for r in rooms}
np.save('../data/room_date_dict_sample{}.npy'.format(sample_frac), room_data_dict_sample)

for n in tqdm(normal_room_list, desc="Generating triplets for NORMAL rooms: "):
    for p in tqdm(poor_room_list):
        for n_d in tqdm(room_data_dict_sample[n]):
            pos_neg = product([i for i in room_data_dict_sample[n] if i != n_d], room_data_dict_sample[p])
            for pos, neg in tqdm(pos_neg):
                triplet_csv = triplet_csv.append({'anchor': (n, n_d), 'pos': (n, pos), 'neg': (p, neg)},
                                                 ignore_index=True)
triplet_csv.to_csv('./triplet.csv', index=False)

for p in tqdm(poor_room_list, desc="Generating triplets for POOR rooms: "):
    for n in tqdm(normal_room_list):
        for p_d in tqdm(room_data_dict_sample[p]):
            pos_neg = product([i for i in room_data_dict_sample[p] if i != p_d], room_data_dict_sample[n])
            for pos, neg in tqdm(pos_neg):
                triplet_csv = triplet_csv.append({'anchor': (p, p_d), 'pos': (p, pos), 'neg': (n, neg)},
                                                 ignore_index=True)

triplet_csv.to_csv('./triplet.csv', index=False)
