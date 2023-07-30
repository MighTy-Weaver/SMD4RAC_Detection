import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = 9, 7
plt.rcParams["figure.autolayout"] = True

data = pd.read_csv('../data/20201230_20210815_data_compiled_half_hour.csv', index_col=None)
# bar plot data['AC'] with 50 bars
sns.histplot(data=data, x="AC", bins=100)
plt.xlabel("Half-hour AC Electricity Consumption", fontsize=23)
plt.ylabel("Count", fontsize=23)
# log scale for y
plt.yscale('log')
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.savefig('./electricity_distribution.jpg', dpi=600, bbox_inches='tight')
plt.clf()
