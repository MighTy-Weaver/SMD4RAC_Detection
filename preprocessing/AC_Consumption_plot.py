from random import sample

import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = 16, 8

data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
sampled_Location = sample(data['Location'].unique().tolist(), 1)
print(sampled_Location)

data['Time'] = pd.to_datetime(data['Time'])

date_start = pd.to_datetime('2021-05-14')
date_end = pd.to_datetime('2021-05-16')
sampled_Location = [804]
for r in sampled_Location + [621]:
    print(r)
    plt.subplot(2, 1, (sampled_Location + [621]).index(r) + 1)
    data_temp = data[data.Location == r].reset_index(drop=True)
    data_temp = data_temp[(data_temp.Time >= date_start) & (data_temp.Time <= date_end)]
    plt.plot(data_temp['Time'], data_temp['AC'], label='AC', color='#002060' if r == 804 else '#002060', linewidth=5)
    # plt.title("Room: {}".format(r), fontsize=23, loc='left')
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=26)
    plt.xticks([pd.to_datetime('2021-05-14'), pd.to_datetime('2021-05-15'), pd.to_datetime('2021-05-16')], fontsize=26)
    if r == 804:
        pass
plt.ylabel("Half-hour Electricity Consumption/kWh", fontsize=26, loc='bottom')
plt.savefig('./AC_consumption.png', dpi=600, bbox_inches='tight')
