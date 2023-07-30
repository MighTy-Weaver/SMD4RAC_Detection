import pandas as pd
from matplotlib import pyplot as plt
from numpy import mean

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = 14, 6

data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)

Location = data['Location'].unique().tolist()
location_number_data = [len(data[data.Location == l]) for l in Location]
location_number_nonzero_data = [len(data[(data.Location == l) & (data.AC > 0)]) for l in Location]
location_nonzero_ratio = [100 * location_number_nonzero_data[i] / location_number_data[i] for i in range(len(Location))]
# sort location_number_data in descending order
avg = mean(location_nonzero_ratio)
location_nonzero_ratio, Location = zip(*sorted(zip(location_nonzero_ratio, Location), reverse=False))
# plot a barplot containing len(Location) bars, each with the value in location_number_data
plt.bar(range(len(Location)), location_nonzero_ratio, color='tab:blue', zorder=2)
plt.yticks(fontsize=26)
plt.xticks([], fontsize=26)
plt.ylabel("Ratio of Non-zero values (%)\nAverage = {}%".format(round(avg, 2)), fontsize=26)
plt.xlabel("Rooms", fontsize=26)
# plt.hlines(avg, -1, len(Location), colors='r', linestyles='dashed', linewidth=3)
# send the grid to bottom behind the bars
plt.grid(color='lightgray', linestyle='--', linewidth=0.8, zorder=-2)
plt.savefig('./nonzero_ratio.png', dpi=600, bbox_inches='tight')
