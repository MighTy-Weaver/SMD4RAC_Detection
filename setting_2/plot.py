import os.path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import mean

from setting_1.utils import data_num_choices
from setting_1.utils import gs_choices

if not os.path.exists('./plot/'):
    os.mkdir('./plot/')
warnings.filterwarnings("ignore")
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
            heatmap_df_r2_mean.loc[dm, gs] = mean(reg_temp['best_valid_r2']) if mean(
                reg_temp['best_valid_r2']) > 0 else 0
    heatmap_df_acc_mean[gs] = heatmap_df_acc_mean[gs].astype(float)
    heatmap_df_acc_max[gs] = heatmap_df_acc_max[gs].astype(float)
    heatmap_df_f1_mean[gs] = heatmap_df_f1_mean[gs].astype(float)
    heatmap_df_f1_max[gs] = heatmap_df_f1_max[gs].astype(float)
    heatmap_df_rmse_mean[gs] = heatmap_df_rmse_mean[gs].astype(float)
    heatmap_df_rmse_max[gs] = heatmap_df_rmse_max[gs].astype(float)
    heatmap_df_r2_mean[gs] = heatmap_df_r2_mean[gs].astype(float)
    heatmap_df_r2_max[gs] = heatmap_df_r2_max[gs].astype(float)

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 15})
plt.rcParams['figure.figsize'] = 17, 12
plt.rcParams["figure.autolayout"] = True

plt.subplot(2, 2, 1)
ax1 = sns.heatmap(heatmap_df_acc_mean.to_numpy(), annot=add_annot, xticklabels=gs_choices,
                  yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu", linewidths=linewidth,
                  cbar_kws={"format": '%.2f'})
cbar = ax1.collections[0].colorbar
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=23)
plt.xlabel("Number of Sampled Points", fontsize=23)
plt.ylabel("Training Sample Size (~100)", fontsize=23)
plt.xticks(fontsize=23, rotation=30)
plt.yticks(labels=['16', '40', '80', '200', '400', '600', '800', '1200', '1600', '2400', '4000',
                   '8000', '16000'], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
           fontsize=23)
plt.title("(a) Average Accuracy in Classification", fontsize=26, pad=15, loc='left')

plt.subplot(2, 2, 2)
ax2 = sns.heatmap(heatmap_df_f1_mean.to_numpy(), annot=add_annot, xticklabels=gs_choices,
                  yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu", fmt=".2f", linewidths=linewidth,
                  cbar_kws={"format": '%.2f'})
cbar = ax2.collections[0].colorbar
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=23)
plt.xlabel("Number of Sampled Points", fontsize=23)
plt.ylabel("Training Sample Size (~100)", fontsize=23)
plt.xticks(fontsize=23, rotation=30)
plt.yticks(labels=['16', '40', '80', '200', '400', '600', '800', '1200', '1600', '2400', '4000',
                   '8000', '16000'], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
           fontsize=23)
plt.title("(b) Average $F_1$ Score in Classification", fontsize=26, pad=15, loc='left')

plt.subplot(2, 2, 3)
ax3 = sns.heatmap(heatmap_df_rmse_mean.to_numpy(), annot=add_annot, xticklabels=gs_choices,
                  yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu_r", fmt=".2f",
                  linewidths=linewidth, cbar_kws={"format": '%.2f'})
cbar = ax3.collections[0].colorbar
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=23)
plt.xlabel("Number of Sampled Points", fontsize=23)
plt.ylabel("Training Sample Size (~100)", fontsize=23)
plt.xticks(fontsize=23, rotation=30)
plt.yticks(labels=['16', '40', '80', '200', '400', '600', '800', '1200', '1600', '2400', '4000',
                   '8000', '16000'], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
           fontsize=23)
plt.title("(c) Average RMSE in Regression", fontsize=26, pad=15, loc='left')

plt.subplot(2, 2, 4)
ax4 = sns.heatmap(heatmap_df_r2_mean.to_numpy(), annot=add_annot, xticklabels=gs_choices,
                  yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu", fmt=".2f", linewidths=linewidth,
                  cbar_kws={"format": '%.2f'})
cbar = ax4.collections[0].colorbar
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=23)
plt.xlabel("Number of Sampled Points", fontsize=23)
plt.ylabel("Training Sample Size (~100)", fontsize=23)
plt.xticks(fontsize=23, rotation=30)
plt.yticks(labels=['16', '40', '80', '200', '400', '600', '800', '1200', '1600', '2400', '4000',
                   '8000', '16000'], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
           fontsize=23)
plt.title("(d) Average $R^2$ Score in Regression", fontsize=26, pad=15, loc='left')

# plt.suptitle("Metrics Statistics on Test Set in Setting II", fontsize=26)
plt.savefig('./ALL_PLOT.jpg', bbox_inches='tight', dpi=600)
plt.savefig('../demo/SettingII_all.jpg', bbox_inches='tight', dpi=600)
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
ylabel_dict = {0: 'Max $F_1$ Score', 1: 'Max $R^2$ Score'}
title_dict = {0: 'Setting II Classification', 1: 'Setting II Regression'}

# for i in range(2):
#     plt.figure(figsize=(18, 7))
#     plt.subplot(1, 2, 1)
#     data_draw = data_dict[i]
#     gs_csv = pd.DataFrame()
#     for m in model_dict.keys():
#         for gs in gs_choices:
#             metric_avg = np.mean(data_draw[(data_draw.gs == gs) & (data_draw.model == model_dict[m])][metric_dict[i]])
#             gs_csv = gs_csv.append({
#                 'model': model_dict[m],
#                 'gs': gs,
#                 'metric': metric_avg
#             }, ignore_index=True)
#         gs_model_csv = gs_csv[gs_csv.model == model_dict[m]].reset_index(drop=True).sort_values(by=['gs'])
#         plt.plot(gs_model_csv['gs'], gs_model_csv['metric'], linewidth=3)
#     plt.xlabel("Sample Size", fontsize=22)
#     plt.ylabel(ylabel_dict[i], fontsize=22)
#     plt.xticks(fontsize=22)
#     plt.yticks(fontsize=22)
#
#     plt.subplot(1, 2, 2)
#     data_num_csv = pd.DataFrame()
#     for m in model_dict.keys():
#         for data_num in data_num_choices:
#             metric_avg = np.mean(
#                 data_draw[(data_draw.data_number == data_num) & (data_draw.model == model_dict[m])][metric_dict[i]])
#             data_num_csv = data_num_csv.append({
#                 'model': model_dict[m],
#                 'data_number': data_num,
#                 'metric': metric_avg
#             }, ignore_index=True)
#         data_num_model_csv = data_num_csv[data_num_csv.model == model_dict[m]].reset_index(drop=True).sort_values(
#             by=['data_number'])
#         plt.plot(data_num_model_csv['data_number'], data_num_model_csv['metric'], linewidth=3, label=model_dict[m])
#     plt.xlabel("Data Size", fontsize=22)
#     plt.ylabel(ylabel_dict[i], fontsize=22)
#     plt.xticks(fontsize=22)
#     plt.yticks(fontsize=22)
#     plt.legend(fontsize=19)
#
#     plt.suptitle("{}".format(title_dict[i]), fontsize=32)
#     plt.savefig('./Model_Plot_{}.png'.format(title_dict[i]), bbox_inches='tight', dpi=600)
#     plt.savefig('../demo/SettingII_model_{}.jpg'.format(title_dict[i]), bbox_inches='tight', dpi=600)
#     plt.clf()

y_ticks = [[0.45, 0.54, 0.63, 0.72, 0.81, 0.90], [-2, -1.5, -1, -0.5, 0, 0.5]]
alphabet = ['(a)', '(b)', '(c)', '(d)']
plt.figure(figsize=(20, 14))

for i in range(2):
    plt.subplot(2, 2, 2 * i + 1)
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
    plt.xlabel("Number of Sampled Points", fontsize=26)
    plt.ylabel(ylabel_dict[i], fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(y_ticks[i], fontsize=26)
    plt.ylim(y_ticks[i][0], y_ticks[i][-1])
    plt.title("{} {} w.r.t. Number of Sampled Points".format(alphabet[2 * i], ylabel_dict[i]), fontsize=27, loc='left',
              fontweight='bold', pad=14)

    plt.subplot(2, 2, 2 * i + 2)
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
    plt.xlabel("Training Sample Size (~10000)", fontsize=26)
    plt.ylabel(ylabel_dict[i], fontsize=26)
    plt.xticks(ticks=[0., 250000., 500000., 750000., 1000000.,
                      1250000., 1500000., 1750000., 2000000.],
               labels=[0, 25, 50, 75, 100, 125, 150, 175, 200], fontsize=26)
    plt.yticks(y_ticks[i], fontsize=26)
    plt.ylim(y_ticks[i][0], y_ticks[i][-1])
    plt.title("{} {} w.r.t. Training Sample Size".format(alphabet[2 * i + 1], ylabel_dict[i]), fontsize=27, loc='left',
              fontweight='bold', pad=14)
    if i == 1:
        plt.legend(fontsize=24)

# plt.suptitle("{}".format(title_dict[i]), fontsize=32)
plt.savefig('./Model_Plot_both_tasks.png', bbox_inches='tight', dpi=600)
plt.savefig('../demo/SettingII_model_both_tasks.jpg', bbox_inches='tight', dpi=300)
plt.clf()
