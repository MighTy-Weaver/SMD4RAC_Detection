import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import mean

from sparse_classification.utils import data_num_choices
from sparse_classification.utils import gs_choices

if not os.path.exists('./plot/'):
    os.mkdir('./plot/')

linewidth = 0
add_annot = False

cla = pd.read_csv('results/sparse_classification_record.csv', index_col=None)
reg = pd.read_csv('results/sparse_regression_record.csv', index_col=None)

heatmap_df_acc_max = pd.DataFrame(columns=gs_choices, index=data_num_choices)
heatmap_df_acc_mean = pd.DataFrame(columns=gs_choices, index=data_num_choices)
heatmap_df_f1_max = pd.DataFrame(columns=gs_choices, index=data_num_choices)
heatmap_df_f1_mean = pd.DataFrame(columns=gs_choices, index=data_num_choices)
heatmap_df_rmse_max = pd.DataFrame(columns=gs_choices, index=data_num_choices)
heatmap_df_rmse_mean = pd.DataFrame(columns=gs_choices, index=data_num_choices)
heatmap_df_r2_max = pd.DataFrame(columns=gs_choices, index=data_num_choices)
heatmap_df_r2_mean = pd.DataFrame(columns=gs_choices, index=data_num_choices)
for gs in gs_choices:
    for dm in data_num_choices:
        cla_temp = cla[(cla.gs == gs) & (cla.data_number == dm)].reset_index(drop=True)
        reg_temp = reg[(reg.gs == gs) & (reg.data_number == dm)].reset_index(drop=True)
        if len(cla_temp) == 0:
            heatmap_df_acc_max.loc[dm, gs] = 0.8
            heatmap_df_acc_mean.loc[dm, gs] = 0.8
            heatmap_df_f1_max.loc[dm, gs] = 0.8
            heatmap_df_f1_mean.loc[dm, gs] = 0.8
        else:
            heatmap_df_acc_max.loc[dm, gs] = max(cla_temp['best_valid_acc'])
            heatmap_df_acc_mean.loc[dm, gs] = mean(cla_temp['best_valid_acc'])
            heatmap_df_f1_max.loc[dm, gs] = max(cla_temp['best_valid_f1'])
            heatmap_df_f1_mean.loc[dm, gs] = mean(cla_temp['best_valid_f1'])
        if len(reg_temp) == 0:
            heatmap_df_rmse_max.loc[dm, gs] = 0.8
            heatmap_df_rmse_mean.loc[dm, gs] = 0.8
            heatmap_df_r2_max.loc[dm, gs] = 0.8
            heatmap_df_r2_mean.loc[dm, gs] = 0.8
        else:
            heatmap_df_rmse_max.loc[dm, gs] = max(reg_temp['best_valid_rmse'])
            heatmap_df_rmse_mean.loc[dm, gs] = mean(reg_temp['best_valid_rmse'])
            heatmap_df_r2_max.loc[dm, gs] = max(reg_temp['best_valid_r2'])
            heatmap_df_r2_mean.loc[dm, gs] = mean(reg_temp['best_valid_r2'])
    heatmap_df_acc_mean[gs] = heatmap_df_acc_mean[gs].astype(float)
    heatmap_df_acc_max[gs] = heatmap_df_acc_max[gs].astype(float)
    heatmap_df_f1_mean[gs] = heatmap_df_f1_mean[gs].astype(float)
    heatmap_df_f1_max[gs] = heatmap_df_f1_max[gs].astype(float)
    heatmap_df_rmse_mean[gs] = heatmap_df_rmse_mean[gs].astype(float)
    heatmap_df_rmse_max[gs] = heatmap_df_rmse_max[gs].astype(float)
    heatmap_df_r2_mean[gs] = heatmap_df_r2_mean[gs].astype(float)
    heatmap_df_r2_max[gs] = heatmap_df_r2_max[gs].astype(float)

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = 16, 12
plt.rcParams["figure.autolayout"] = True
plt.subplot(2, 2, 1)
sns.heatmap(heatmap_df_acc_mean.to_numpy(), annot=add_annot, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu", fmt=".2f", linewidths=linewidth)
plt.xlabel("Number of data points in one sparse data sample", fontsize=18)
plt.ylabel("Number of Training data", fontsize=18)
plt.title("Average Highest Accuracy by Five Models in Classification Task")

plt.subplot(2, 2, 2)
sns.heatmap(heatmap_df_f1_mean.to_numpy(), annot=add_annot, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu", fmt=".2f", linewidths=linewidth)
plt.xlabel("Number of data points in one sparse data sample", fontsize=18)
plt.ylabel("Number of Training data", fontsize=18)
plt.title("Average Highest $F_1$ Score by Five Models in Classification Task")

plt.subplot(2, 2, 3)
sns.heatmap(heatmap_df_rmse_mean.to_numpy(), annot=add_annot, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu_r", fmt=".2f", linewidths=linewidth)
plt.xlabel("Number of data points in one sparse data sample", fontsize=18)
plt.ylabel("Number of Training data", fontsize=18)
plt.title("Average Minimum RMSE by Five Models in Regression Task")

plt.subplot(2, 2, 4)
# heatmap_df_r2_mean = gaussian_filter(heatmap_df_r2_mean, sigma=1)
sns.heatmap(heatmap_df_r2_mean.to_numpy(), annot=add_annot, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu", fmt=".2f", linewidths=linewidth)
plt.xlabel("Number of data points in one sparse data sample", fontsize=18)
plt.ylabel("Number of Training data", fontsize=18)
plt.title("Average Highest $R^2$ Score by Five Models in Regression Task")

plt.suptitle("Metrics Statistics on Test Set in Setting I", fontsize=22)
plt.savefig('./ALL_PLOT.jpg', bbox_inches='tight', dpi=800)
plt.savefig('../demo/SettingI_all.jpg', bbox_inches='tight', dpi=400)
plt.clf()

cla.sort_values(by='model', inplace=True)
reg.sort_values(by='model', inplace=True)
cla['model'] = cla['model'].apply(lambda x: {'bilstm': 'BiLSTM', 'lstm': 'LSTM', 'transformer': 'Transformer',
                                             'lstm-transformer': 'LSTM-Transformer',
                                             'bilstm-transformer': 'BiLSTM-Transformer'}[x])
reg['model'] = reg['model'].apply(lambda x: {'bilstm': 'BiLSTM', 'lstm': 'LSTM', 'transformer': 'Transformer',
                                             'lstm-transformer': 'LSTM-Transformer',
                                             'bilstm-transformer': 'BiLSTM-Transformer'}[x])
plt.subplot(2, 2, 1)
sns.lineplot(data=cla, y='best_valid_f1', x='gs', hue='model', ci=None)
plt.xlabel("Number of data points in one sparse data sample", fontsize=18)
plt.ylabel("Maximum $f_1$ Score", fontsize=18)
plt.subplot(2, 2, 2)
sns.lineplot(data=cla, y='best_valid_f1', x='data_number', hue='model', ci=None)
plt.xlabel("Number of training data", fontsize=18)
plt.ylabel("Maximum $f_1$ Score", fontsize=18)
plt.subplot(2, 2, 3)
sns.lineplot(data=reg, y='best_valid_r2', x='gs', hue='model', ci=None)
plt.xlabel("Number of data points in one sparse data sample", fontsize=18)
plt.ylabel("Maximum $R^2$ Score", fontsize=18)
plt.subplot(2, 2, 4)
sns.lineplot(data=reg, y='best_valid_r2', x='data_number', hue='model', ci=None)
plt.xlabel("Number of training data", fontsize=18)
plt.ylabel("Maximum $R^2$ Score", fontsize=18)
plt.suptitle(
    "Comparison between five models in both tasks on test set in Setting I\nwith respect to two parameters in sparse data sampling",
    fontsize=22)
plt.savefig('./Model_Plot.png', bbox_inches='tight', dpi=800)
plt.savefig('../demo/SettingI_model.jpg', bbox_inches='tight', dpi=400)
plt.clf()
