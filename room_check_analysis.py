import math

import pandas as pd
from matplotlib import pyplot as plt

from utils import replace_dict

room_data = pd.read_csv('./data/room_check_original.csv', index_col=None)
pd.set_option("display.max_columns", None)

print(room_data)
print(list(room_data))

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

plt.hist(Efficiency, bins=50, density=False)
plt.show()
room_data.to_csv('./data/room_check.csv', index=False)

room_data = room_data.drop(['Occupancy', 'Machine Mode', 'Rated Power', 'Facing', 'recent_status', 'weather'], axis=1)
print(list(room_data))
print(room_data.dtypes)
print(room_data.describe())
