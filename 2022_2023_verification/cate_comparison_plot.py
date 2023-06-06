import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append('../')
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.autolayout"] = True
plt.rcParams['figure.figsize'] = 20, 12
plt.rcParams.update({'font.size': 15})

predictions = pd.read_csv('./prediction.csv', index_col=None).sort_values(by=['pred'], ascending=True)
rooms = predictions['room'].tolist()
top_rooms = rooms[:int(len(rooms) * 0.5)]
bottom_rooms = rooms[int(len(rooms) * 0.5):]

time1 = range(0, 8)
time2 = range(8, 14)
time3 = range(14, 18)
time4 = range(18, 24)
time_period = [time1, time2, time3, time4]
title = ["00:00AM - 08:00AM", "08:00AM - 14:00PM", "14:00PM - 18:00PM", "18:00PM - 24:00PM"]

data = pd.read_csv('./FINAL_total_csv.csv', index_col=None)
data = data[data.AC > 0.01]
data['Time'] = pd.to_datetime(data['Time'])
data['Hour'] = data['Time'].apply(lambda x: int(x.hour))

normal_room_list_final = top_rooms
poor_room_list_final = bottom_rooms

for i in range(4):
    data_temp = data[data.Hour.isin(time_period[i])]
    normal_data = data_temp[data_temp.Location.isin(normal_room_list_final)].sample(frac=0.2)
    poor_data = data_temp[data_temp.Location.isin(poor_room_list_final)].sample(frac=0.2)
    plt.subplot(2, 2, i + 1)
    sns.distplot(normal_data['AC'], bins=sorted(normal_data['AC'].unique()),
                 label="Normal efficiency: Mean={}kWh".format(round(np.mean(normal_data['AC']), 3)),
                 color="brown", hist_kws={"edgecolor": "black"}, kde_kws={"linewidth": "3"})
    sns.distplot(poor_data['AC'], bins=sorted(poor_data['AC'].unique()),
                 label="Low efficiency: Mean={}kWh".format(round(np.mean(poor_data['AC']), 3)),
                 color="skyblue", hist_kws={"edgecolor": "black"}, kde_kws={"linewidth": "3"})
    plt.title(title[i], fontsize=22)
    ratio = "%.2f" % (100 * (round(np.mean(poor_data['AC']), 4) - round(np.mean(normal_data['AC']), 4)) / round(
        np.mean(normal_data['AC']), 4))
    plt.xlabel("Half-hourly AC Electricity Consumption/kWh\nPotentially avoidable electricity ratio: {}%".format(
        ratio), fontsize=22)
    plt.ylabel("Kernel Density", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], fontsize=22)
    plt.legend(fontsize=20)
plt.suptitle("Energy Consumption Comparison Between Different RAC Efficiency Groups During Four Time Periods",
             fontsize=26)
plt.savefig('./TOTAL_comparison.png', bbox_inches='tight', dpi=1000)
