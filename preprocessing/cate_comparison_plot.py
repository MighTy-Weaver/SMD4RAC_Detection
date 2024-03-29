import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append('../')
from setting_1.utils import normal_room_list, poor_room_list
import pandas as pd
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["figure.autolayout"] = True
plt.rcParams['figure.figsize'] = 21, 14
plt.rcParams.update({'font.size': 15})

time1 = ['00:00:00', '00:30:00', '01:00:00', '01:30:00', '02:00:00', '02:30:00', '03:00:00', '03:30:00', '04:00:00',
         '04:30:00', '05:00:00', '05:30:00', '06:00:00', '06:30:00', '07:00:00', '07:30:00']
time2 = ['08:00:00', '08:30:00', '09:00:00', '09:30:00', '10:00:00', '10:30:00', '11:00:00', '11:30:00', '12:00:00',
         '12:30:00', '13:00:00', '13:30:00']
time3 = ['14:00:00', '14:30:00', '15:00:00', '15:30:00', '16:00:00', '16:30:00', '17:00:00', '17:30:00']
time4 = ['18:00:00', '18:30:00', '19:00:00', '19:30:00', '20:00:00', '20:30:00', '21:00:00', '21:30:00', '22:00:00',
         '22:30:00', '23:00:00', '23:30:00']
time_period = [time1, time2, time3, time4]
title = ["(a) 00:00AM - 08:00AM", "(b) 08:00AM - 14:00PM", "(c) 14:00PM - 18:00PM", "(d) 18:00PM - 24:00PM"]

data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
data = data[data.AC > 0]
print(list(data))
data['Hour'] = data['Time'].apply(lambda x: x.split(' ')[-1])

normal_room_list_final = normal_room_list + [619, 629, 1004]
poor_room_list_final = poor_room_list + [303, 307, 308, 311, 328, 612, 622, 624, 628, 630, 635, 806, 808, 809, 812, 819,
                                         822, 823, 829,
                                         832, 901, 903, 904, 906, 913, 916, 1001, 1002, 1003, 1012]
normal_data = data[data.Location.isin(normal_room_list_final)]
poor_data = data[data.Location.isin(poor_room_list_final)]

for i in range(4):
    data_temp = data[data.Hour.isin(time_period[i])]
    normal_data = data_temp[data_temp.Location.isin(normal_room_list_final)]
    poor_data = data_temp[data_temp.Location.isin(poor_room_list_final)]
    plt.subplot(2, 2, i + 1)
    sns.distplot(normal_data['AC'], bins=sorted(normal_data['AC'].unique()),
                 label="Normal efficiency: Mean={}kWh".format(round(np.mean(normal_data['AC']), 3)),
                 color="brown", hist_kws={"edgecolor": "black"}, kde_kws={"linewidth": "3"})
    sns.distplot(poor_data['AC'], bins=sorted(poor_data['AC'].unique()),
                 label="Low efficiency: Mean={}kWh".format(round(np.mean(poor_data['AC']), 3)),
                 color="skyblue", hist_kws={"edgecolor": "black"}, kde_kws={"linewidth": "3"})
    plt.title(title[i], fontsize=23, loc='left')
    ratio = "%.2f" % (100 * (round(np.mean(poor_data['AC']), 4) - round(np.mean(normal_data['AC']), 4)) / round(
        np.mean(poor_data['AC']), 4))
    plt.xlabel("Half-hourly AC Electricity Consumption/kWh", fontsize=21)
    plt.text(0.025, 0.78, "Potentially savable electricity: {}%".format(
        ratio), ha='left', va='center', transform=plt.gca().transAxes, fontsize=21)
    if i % 2 == 0:
        plt.ylabel("Kernel Density", fontsize=21)
    else:
        plt.ylabel("")
    plt.grid(color='lightgray', linestyle='--', linewidth=0.8)
    plt.xticks(fontsize=21)
    plt.ylim(0, 10)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], fontsize=21)
    plt.legend(fontsize=21, frameon=False, loc='upper left')
# plt.suptitle(
#     "Energy Consumption Comparison Between Different RAC Efficiency Groups\nDuring Four Time Periods in 2020/12/30 - 2021/08/15",
#     fontsize=26)
plt.savefig('./2021_efficiency_comparison.png', bbox_inches='tight', dpi=500)
