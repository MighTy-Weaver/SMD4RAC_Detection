import math

import pandas as pd

from utils import replace_dict

room_data = pd.read_csv('./room_check.csv', index_col=None)

print(room_data)
print(list(room_data))

for i in range(len(room_data)):
    if math.isnan(room_data.loc[i, 'Indoor_after']):
        room_data = room_data.drop(i, axis=0)
    room_data.loc[i, 'replaced_year'] = 0000
    for key in replace_dict.keys():
        if room_data.loc[i, 'room'] in replace_dict[key]:
            room_data.loc[i, 'replaced_year'] = key
    room_data.loc[i, 'Indoor_T_diff'] = -round(room_data.loc[i, 'Indoor_after'] - room_data.loc[i, 'Indoor_init'], 5)

    room_data.loc[i, 'Indoor_RH_diff'] = -round(float(room_data.loc[i, 'Indoor_RH_after'].replace('%', '')) - float(
        room_data.loc[i, 'Indoor_RH_init'].replace('%', '')), 5)

room_data['replaced_year'] = room_data['replaced_year'].astype('int64')
room_data.to_csv('./room_check.csv', index=False)
