import argparse
import os.path
from itertools import permutations
from itertools import product
from random import sample

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import normal_room_list
from utils import poor_room_list

parser = argparse.ArgumentParser()
parser.add_argument('--frac', default=0.5, type=float, help="Number of fraction to sample from all triplets")
parser.add_argument('--k', default=50000, type=int, help="Number of samples for each room")
args = parser.parse_args()

sample_frac = args.frac
sample_num = args.k

print(normal_room_list)
print(poor_room_list)

data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
triplet_csv = pd.DataFrame(columns=['anchor', 'pos', 'neg'])
X = data.drop(['Weekday', 'Total', 'Lighting', 'Socket', 'WaterHeater', 'Time'], axis=1)
Y = data['AC']
X['Date'] = data['Time'].apply(lambda x: x.split(' ')[0])
dates = X['Date'].unique()
print(dates)
rooms = normal_room_list + poor_room_list
print(rooms)

if False:  # os.path.exists('../data/room_date_dict.npy'):
    room_data_dict = np.load('../data/room_date_dict.npy', allow_pickle=True).item()
    print('Loading pre-generated room date summary dict, no sample applied.')
    print(room_data_dict.keys(), list(room_data_dict.values())[0])
else:
    room_data_dict = {r: [] for r in rooms}
    for r in tqdm(rooms, desc="filtering data samples for ALL rooms: "):
        for d in tqdm(dates):
            if len(X[(X.Date == d) & (X.Location == r)]) == 48:
                room_data_dict[r].append(d)
    np.save('../data/room_date_dict.npy', room_data_dict)
    room_data_dict_sample = {r: sample(room_data_dict[r], int(len(room_data_dict[r]) * sample_frac)) for r in rooms}
    np.save('../data/room_date_dict_sample{}.npy'.format(sample_frac), room_data_dict_sample)

for n in tqdm(normal_room_list, desc="Generating triplets for NORMAL rooms: "):
    for p in tqdm(poor_room_list):
        pos_list = permutations(room_data_dict[n], 2)
        neg_list = room_data_dict[p]
        pos_neg_list = sample(list(product(pos_list, neg_list)), k=sample_num)
        n_p_df = pd.DataFrame(data={'anchor': [i[0][0] for i in pos_neg_list], 'pos': [i[0][1] for i in pos_neg_list],
                                    'neg': [i[1] for i in pos_neg_list], 'anchor_room': [n for _ in pos_neg_list],
                                    'pos_room': [n for _ in pos_neg_list], 'neg_room': [p for _ in pos_neg_list]})
        if os.path.exists('./pos_triplet.csv'):
            n_p_df.to_csv('./pos_triplet.csv', mode='a', header=False, index=False)
        else:
            n_p_df.to_csv('./pos_triplet.csv', index=False)

        if os.path.exists('./triplet.csv'):
            n_p_df.to_csv('./triplet.csv', mode='a', header=False, index=False)
        else:
            n_p_df.to_csv('./triplet.csv', index=False)

for p in tqdm(poor_room_list, desc="Generating triplets for POOR rooms: "):
    for n in tqdm(normal_room_list):
        neg_list = permutations(room_data_dict[p], 2)
        pos_list = room_data_dict[n]
        neg_pos_list = sample(list(product(neg_list, pos_list)), k=sample_num)
        p_n_df = pd.DataFrame(data={'anchor': [i[0][0] for i in neg_pos_list], 'pos': [i[0][1] for i in neg_pos_list],
                                    'neg': [i[1] for i in neg_pos_list], 'anchor_room': [p for _ in neg_pos_list],
                                    'pos_room': [p for _ in neg_pos_list], 'neg_room': [n for _ in neg_pos_list]})
        if os.path.exists('./neg_triplet.csv'):
            p_n_df.to_csv('./neg_triplet.csv', mode='a', header=False, index=False)
        else:
            p_n_df.to_csv('./neg_triplet.csv', index=False)

        if os.path.exists('./triplet.csv'):
            p_n_df.to_csv('./triplet.csv', mode='a', header=False, index=False)
        else:
            p_n_df.to_csv('./triplet.csv', index=False)
