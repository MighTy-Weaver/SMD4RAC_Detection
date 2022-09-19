import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import mean

from setting_1.utils import data_num_choices
from setting_1.utils import gs_choices

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
plt.rcParams['figure.figsize'] = 17, 12
plt.rcParams["figure.autolayout"] = True
plt.subplot(2, 2, 1)
sns.heatmap(heatmap_df_acc_mean.to_numpy(), annot=add_annot, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu", fmt=".2f", linewidths=linewidth)
plt.xlabel("Number of data points in one sparse data sample", fontsize=22)
plt.ylabel("Number of Training data", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.title("Average Highest Accuracy by Five Models in Classification Task", fontsize=20)

plt.subplot(2, 2, 2)
sns.heatmap(heatmap_df_f1_mean.to_numpy(), annot=add_annot, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu", fmt=".2f", linewidths=linewidth)
plt.xlabel("Number of data points in one sparse data sample", fontsize=22)
plt.ylabel("Number of Training data", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.title("Average Highest $F_1$ Score by Five Models in Classification Task", fontsize=20)

plt.subplot(2, 2, 3)
sns.heatmap(heatmap_df_rmse_mean.to_numpy(), annot=add_annot, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu_r", fmt=".2f", linewidths=linewidth)
plt.xlabel("Number of data points in one sparse data sample", fontsize=22)
plt.ylabel("Number of Training data", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.title("Average Minimum RMSE by Five Models in Regression Task", fontsize=20)

plt.subplot(2, 2, 4)
# heatmap_df_r2_mean = gaussian_filter(heatmap_df_r2_mean, sigma=1)
sns.heatmap(heatmap_df_r2_mean.to_numpy(), annot=add_annot, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu", fmt=".2f", linewidths=linewidth)
plt.xlabel("Number of data points in one sparse data sample", fontsize=22)
plt.ylabel("Number of Training data", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.title("Average Highest $R^2$ Score by Five Models in Regression Task", fontsize=20)

plt.suptitle("Metrics Statistics on Test Set in Setting I", fontsize=26)
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

model_dict = {'bilstm': 'BiLSTM', 'lstm': 'LSTM', 'transformer': 'Transformer',
              'lstm-transformer': 'LSTM-Transformer', 'bilstm-transformer': 'BiLSTM-Transformer'}
data_dict = {0: cla, 1: reg}
metric_dict = {0: 'best_valid_f1', 1: 'best_valid_r2'}
ylabel_dict = {0: 'Maximum $f_1$ Score', 1: 'Maximum $R^2$ Score'}
title_dict = {0: 'Classification', 1: 'Regression'}
for i in range(2):
    plt.figure(figsize=(18, 7))
    plt.subplot(1, 2, 1)
    data_draw = data_dict[i]
    gs_csv = pd.DataFrame()
    for m in model_dict.keys():
        for gs in gs_choices:
            metric_avg = np.mean(data_draw[(data_draw.gs == gs) & (data_draw.model == model_dict[m])][metric_dict[i]])
            gs_csv = gs_csv.append({
                'model': model_dict[m],
                'gs': gs,
                'metric': metric_avg
            }, ignore_index=True)
        gs_model_csv = gs_csv[gs_csv.model == model_dict[m]].reset_index(drop=True).sort_values(by=['gs'])
        plt.plot(gs_model_csv['gs'], gs_model_csv['metric'], linewidth=3)
    plt.xlabel("Number of data points in one sparse data sample", fontsize=22)
    plt.ylabel(ylabel_dict[i], fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.subplot(1, 2, 2)
    data_num_csv = pd.DataFrame()
    for m in model_dict.keys():
        for data_num in data_num_choices:
            metric_avg = np.mean(
                data_draw[(data_draw.data_number == data_num) & (data_draw.model == model_dict[m])][metric_dict[i]])
            data_num_csv = data_num_csv.append({
                'model': model_dict[m],
                'data_number': data_num,
                'metric': metric_avg
            }, ignore_index=True)
        data_num_model_csv = data_num_csv[data_num_csv.model == model_dict[m]].reset_index(drop=True).sort_values(
            by=['data_number'])
        plt.plot(data_num_model_csv['data_number'], data_num_model_csv['metric'], linewidth=3, label=model_dict[m])
    plt.xlabel("Number of total data", fontsize=22)
    plt.ylabel(ylabel_dict[i], fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=19)

    plt.suptitle("{}".format(title_dict[i]), fontsize=32)
    plt.savefig('./Model_Plot_{}.png'.format(title_dict[i]), bbox_inches='tight', dpi=700)
    plt.savefig('../demo/SettingI_model_{}.jpg'.format(title_dict[i]), bbox_inches='tight', dpi=400)
    plt.clf()
