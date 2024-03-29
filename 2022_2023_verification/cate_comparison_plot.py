import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.append('../')
from setting_1.utils import efficiency_dict, normal_room_list, poor_room_list

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["figure.autolayout"] = True
plt.rcParams['figure.figsize'] = 21, 15
plt.rcParams.update({'font.size': 15})

predictions = pd.read_csv('./prediction.csv', index_col=None).sort_values(by=['pred'], ascending=True)
rooms = predictions['room'].tolist()
bottom_rooms = rooms[:int(len(rooms) * 0.5)]
top_rooms = rooms[int(len(rooms) * 0.5):]

time1 = range(0, 8)
time2 = range(8, 14)
time3 = range(14, 18)
time4 = range(18, 24)
time_period = [time1, time2, time3, time4]
title = ["(a) 00:00AM - 08:00AM", "(b) 08:00AM - 14:00PM", "(c) 14:00PM - 18:00PM", "(d) 18:00PM - 24:00PM"]

data = pd.read_csv('./FINAL_total_csv.csv', index_col=None)
data = data[data.AC > 0.01]
data['Time'] = pd.to_datetime(data['Time'])
data['Hour'] = data['Time'].apply(lambda x: int(x.hour))

# make the AC column to two decimal floats
data['AC'] = data['AC'].apply(lambda x: round(x, 2))

original_dict = efficiency_dict

normal_room_list_final = [i for i in top_rooms if i in normal_room_list]
poor_room_list_final = [i for i in bottom_rooms if i in poor_room_list]

avg_normal = np.mean(data[data.Location.isin(normal_room_list_final)]['AC'])
avg_poor = np.mean(data[data.Location.isin(poor_room_list_final)]['AC'])
print(avg_normal, avg_poor, avg_poor - avg_normal)

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
    plt.title(title[i], fontsize=28, loc='left')
    ratio = "%.2f" % (100 * (round(np.mean(poor_data['AC']), 4) - round(np.mean(normal_data['AC']), 4)) / round(
        np.mean(poor_data['AC']), 4))
    plt.xlabel("Half-hourly AC Electricity Consumption/kWh", fontsize=26)
    plt.text(0.028, 0.78, "Potentially savable electricity: {}%".format(
        ratio), ha='left', va='center', transform=plt.gca().transAxes, fontsize=23)
    if i % 2 == 0:
        plt.ylabel("Kernel Density", fontsize=26)
    else:
        plt.ylabel("")
    plt.grid(color='lightgray', linestyle='--', linewidth=0.8)
    plt.xticks(fontsize=26)
    plt.ylim(0, 10)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize=26)
    plt.legend(fontsize=23, frameon=False, loc='upper left')
# plt.suptitle(
#     "Energy Consumption Comparison Between Different RAC Efficiency Groups\nDuring Four Time Periods in 2022/01/01 - 2022/12/31",
#     fontsize=26)
plt.savefig('./22-23_efficiency_comparison.png', bbox_inches='tight', dpi=600)
plt.clf()
