import glob
import math

import numpy as np
import pandas as pd

from utils import replace_dict

room_data = pd.read_csv('./data/room_check_original.csv', index_col=None)
pd.set_option("display.max_columns", None)

print(room_data)
print(list(room_data))
room_data['room'] = room_data['room'].astype(int)
room_data['Occupancy'].replace({math.nan: 0, 'Y': 1})

for i in range(len(room_data)):
    if math.isnan(room_data.loc[i, 'Indoor_after']):
        room_data = room_data.drop(i, axis=0)
    try:
        if math.isnan(room_data.loc[i, 'Indoor_RH_init']):
            room_data = room_data.drop(i, axis=0)
    except TypeError:
        pass
    room_data.loc[i, 'replaced_year'] = 0000
    for key in replace_dict.keys():
        if room_data.loc[i, 'room'] in replace_dict[key]:
            room_data.loc[i, 'replaced_year'] = key

    room_data.loc[i, 'Indoor_RH_init'] = float(room_data.loc[i, 'Indoor_RH_init'].replace('%', ''))
    room_data.loc[i, 'Indoor_RH_after'] = float(room_data.loc[i, 'Indoor_RH_after'].replace('%', ''))
    room_data.loc[i, 'Env RH'] = float(room_data.loc[i, 'Env RH'].replace('%', ''))
    room_data.loc[i, 'Rated Power'] = float(room_data.loc[i, 'Rated Power'].replace('W', ''))
    if len(room_data.loc[i, 'Time before stablization'].split('min')) == 1:

        room_data.loc[i, 'Time before stablization'] = \
            int(room_data.loc[i, 'Time before stablization'].replace('s', '').split('min')[0]) * 60 + \
            int(room_data.loc[i, 'Time before stablization'].replace('s', '').split('min')[1])
    else:
        room_data.loc[
            i, 'Time before stablization'] = int(
            room_data.loc[i, 'Time before stablization'].replace('s', '').split('min')[0]) * 60

    room_data.loc[i, 'Velocity'] = room_data.loc[i, 'Velocity'].replace('m/s', '')

    room_data.loc[i, 'Indoor_T_diff'] = -round(room_data.loc[i, 'Indoor_after'] - room_data.loc[i, 'Env T'], 5)
    room_data.loc[i, 'Indoor_RH_diff'] = -round(room_data.loc[i, 'Indoor_RH_after'] -
                                                room_data.loc[i, 'Indoor_RH_init'], 5)
room_data['Indoor_RH_init'] = room_data['Indoor_RH_init'].astype('float64')
room_data['Indoor_RH_after'] = room_data['Indoor_RH_after'].astype('float64')
room_data['Env RH'] = room_data['Env RH'].astype('float64')
room_data['Stablized Tair'] = room_data['Stablized Tair'].astype('float64')
room_data['Time before stablization'] = room_data['Time before stablization'].astype('int64')
room_data['Velocity'] = room_data['Velocity'].astype('float64')
room_data['replaced_year'] = room_data['replaced_year'].astype('int64')
room_data['Efficiency'] = (1.012 * room_data['Velocity'] * room_data['Indoor_T_diff'] + 2260 * room_data['Velocity'] *
                           room_data['Indoor_RH_diff'] * 0.01) / room_data['Rated Power']
print(room_data['Efficiency'])
room_data['Efficiency'] = room_data['Efficiency'].astype('float64')
Efficiency = list(room_data['Efficiency'])

room_data.sort_values(by=['Efficiency'], inplace=True, ascending=False)
print(room_data)
efficiency_room_list = list(room_data['room'])
print(efficiency_room_list)
efficiency_dict = {}
for i in range(len(room_data)):
    efficiency_dict[room_data.loc[i, 'room']] = room_data.loc[i, 'Efficiency']
np.save('./data/efficiency_dict.npy', efficiency_dict)

high_class = [int(i.split('\\')[-1].split('.')[0]) for i in
              glob.glob(
                  '../../UROP 2 - Zhongming Lu/Inefficient-AC-detection/20210713191456_cate3_KMeans_22_33/a/*.png')] + [
                 int(j.split('\\')[-1].split('.')[0]) for j in
                 glob.glob(
                     './20210713191456_cate3_KMeans_22_33/c/*.png')]
high_class = [i for i in high_class if i in efficiency_room_list]
medium_class = [int(j.split('\\')[-1].split('.')[0]) for j in glob.glob(
    '../../UROP 2 - Zhongming Lu/Inefficient-AC-detection/20210713191456_cate3_KMeans_22_33/c/*.png')]
medium_class = [i for i in medium_class if i in efficiency_room_list]
low_class = [int(i.split('\\')[-1].split('.')[0]) for i in glob.glob(
    '../../UROP 2 - Zhongming Lu/Inefficient-AC-detection/20210713191456_cate3_KMeans_22_33/b/*.png')]
low_class = [i for i in low_class if i in efficiency_room_list]

efficiency_room_list = [i for i in efficiency_room_list if i in high_class + medium_class + low_class]
print(high_class, medium_class, low_class)
a = high_class + medium_class
print('TP', len([i for i in efficiency_room_list[:len(a)] if i in a]))
print('TN', len([i for i in efficiency_room_list[len(a):] if i in low_class]))
print('FP', len([i for i in efficiency_room_list[:len(a)] if i not in a]))
print('FN', len([i for i in efficiency_room_list[len(a):] if i not in low_class]))

TP = len([i for i in efficiency_room_list[:len(a)] if i in a])
TN = len([i for i in efficiency_room_list[len(a):] if i in low_class])
FP = len([i for i in efficiency_room_list[:len(a)] if i not in a])
FN = len([i for i in efficiency_room_list[len(a):] if i not in low_class])

print('Accuracy = {}'.format(round((TP + TN) / (len(efficiency_room_list)), 7)))
print('Recall = {}'.format(round(TP / (TP + FN), 7)))
print('Precision = {}'.format(round(TP / (TP + FP), 7)))
print('F_1 score = {}'.format(round(2 * TP / (2 * TP + FP + FN), 7)))

# plt.hist(Efficiency, bins=50, density=False)
# plt.show()
# room_data.to_csv('./data/room_check.csv', index=False)
#
# room_data = room_data.drop(['Occupancy', 'Machine Mode', 'Rated Power', 'Facing', 'recent_status', 'weather'], axis=1)
# print(list(room_data))
# print(room_data.dtypes)
# print(room_data.describe())
