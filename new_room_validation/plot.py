import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import mean

from sparse_classification.utils import data_num_choices
from sparse_classification.utils import gs_choices

if not os.path.exists('./plot/'):
    os.mkdir('./plot/')
plt.rcParams["font.family"] = "Times New Roman"

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

sns.heatmap(heatmap_df_acc_mean.to_numpy(), annot=True, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu")
plt.xlabel("Number of data points in one sparse data sample")
plt.ylabel("Number of Training data")
plt.title("Average Highest Accuracy Achieved on Test Set by Five Models in Classification Task (Setting I)")
plt.savefig('./plot/avg_acc.png', bbox_inches='tight', dpi=900)
plt.clf()

sns.heatmap(heatmap_df_acc_max.to_numpy(), annot=True, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu")
plt.xlabel("Number of data points in one sparse data sample")
plt.ylabel("Number of Training data")
plt.title("Maximum Highest Accuracy Achieved on Test Set by Five Models in Classification Task (Setting I)")
plt.savefig('./plot/max_acc.png', bbox_inches='tight', dpi=900)
plt.clf()

sns.heatmap(heatmap_df_f1_mean.to_numpy(), annot=True, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu")
plt.xlabel("Number of data points in one sparse data sample")
plt.ylabel("Number of Training data")
plt.title("Average Highest $F_1$ Score Achieved on Test Set by Five Models in Classification Task (Setting I)")
plt.savefig('./plot/avg_f1.png', bbox_inches='tight', dpi=900)
plt.clf()

sns.heatmap(heatmap_df_f1_max.to_numpy(), annot=True, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu")
plt.xlabel("Number of data points in one sparse data sample")
plt.ylabel("Number of Training data")
plt.title("Maximum Highest $F_1$ Score Achieved on Test Set by Five Models in Classification Task (Setting I)")
plt.savefig('./plot/max_f1.png', bbox_inches='tight', dpi=900)
plt.clf()

sns.heatmap(heatmap_df_rmse_mean.to_numpy(), annot=True, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu")
plt.xlabel("Number of data points in one sparse data sample")
plt.ylabel("Number of Training data")
plt.title("Average Minimum RMSE Achieved on Test Set by Five Models in Regression Task (Setting I)")
plt.savefig('./plot/avg_rmse.png', bbox_inches='tight', dpi=900)
plt.clf()

sns.heatmap(heatmap_df_rmse_max.to_numpy(), annot=True, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu")
plt.xlabel("Number of data points in one sparse data sample")
plt.ylabel("Number of Training data")
plt.title("Lowest Minimum RMSE Achieved on Test Set by Five Models in Regression Task (Setting I)")
plt.savefig('./plot/min_rmse.png', bbox_inches='tight', dpi=900)
plt.clf()

sns.heatmap(heatmap_df_r2_mean.to_numpy(), annot=True, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu")
plt.xlabel("Number of data points in one sparse data sample")
plt.ylabel("Number of Training data")
plt.title("Average Highest $R^2$ Score Achieved on Test Set by Five Models in Regression Task (Setting I)")
plt.savefig('./plot/avg_r2.png', bbox_inches='tight', dpi=900)
plt.clf()

sns.heatmap(heatmap_df_r2_max.to_numpy(), annot=True, xticklabels=gs_choices,
            yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu")
plt.xlabel("Number of data points in one sparse data sample")
plt.ylabel("Number of Training data")
plt.title("Maximum Highest $R^2$ Score Achieved on Test Set by Five Models in Regression Task (Setting I)")
plt.savefig('./plot/max_r2.png', bbox_inches='tight', dpi=900)
plt.clf()

model_dict = {'lstm': 'LSTM', 'bilstm': 'BiLSTM', 'transformer': 'Transformer', 'lstm-transformer': 'LSTM-Transformer',
              'bilstm-transformer': 'BiLSTM-Transformer'}
for m in cla['model'].unique():
    model_acc_df = pd.DataFrame(columns=gs_choices, index=data_num_choices, dtype=float).applymap(lambda x: 0)
    model_f1_df = pd.DataFrame(columns=gs_choices, index=data_num_choices, dtype=float).applymap(lambda x: 0)
    model_rmse_df = pd.DataFrame(columns=gs_choices, index=data_num_choices, dtype=float).applymap(lambda x: 0)
    model_r2_df = pd.DataFrame(columns=gs_choices, index=data_num_choices, dtype=float).applymap(lambda x: 0)
    cla_m = cla[cla.model == m].reset_index(drop=True)
    reg_m = reg[reg.model == m].reset_index(drop=True)
    for i in range(len(cla_m)):
        model_acc_df.loc[cla_m.loc[i, 'data_number'], cla_m.loc[i, 'gs']] = cla_m.loc[i, 'best_valid_acc']
        model_f1_df.loc[cla_m.loc[i, 'data_number'], cla_m.loc[i, 'gs']] = cla_m.loc[i, 'best_valid_f1']
    for i in range(len(reg_m)):
        model_rmse_df.loc[reg_m.loc[i, 'data_number'], reg_m.loc[i, 'gs']] = reg_m.loc[i, 'best_valid_rmse']
        model_r2_df.loc[reg_m.loc[i, 'data_number'], reg_m.loc[i, 'gs']] = reg_m.loc[i, 'best_valid_r2']

    sns.heatmap(model_acc_df.to_numpy(), annot=True, xticklabels=gs_choices,
                yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu")
    plt.xlabel("Number of data points in one sparse data sample")
    plt.ylabel("Number of Training data")
    plt.title(f"Highest Accuracy Achieved on Test Set by {model_dict[m]} in Classification Task (Setting I)")
    plt.savefig(f'./plot/{m}_acc.png', bbox_inches='tight', dpi=900)
    plt.clf()

    sns.heatmap(model_f1_df.to_numpy(), annot=True, xticklabels=gs_choices,
                yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu")
    plt.xlabel("Number of data points in one sparse data sample")
    plt.ylabel("Number of Training data")
    plt.title(f"Highest $F_1$ Score Achieved on Test Set by {model_dict[m]} in Classification Task (Setting I)")
    plt.savefig(f'./plot/{m}_f1.png', bbox_inches='tight', dpi=900)
    plt.clf()

    sns.heatmap(model_rmse_df.to_numpy(), annot=True, xticklabels=gs_choices,
                yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu")
    plt.xlabel("Number of data points in one sparse data sample")
    plt.ylabel("Number of Training data")
    plt.title(f"Minimum RMSE Achieved on Test Set by {model_dict[m]} in Classification Task (Setting I)")
    plt.savefig(f'./plot/{m}_rmse.png', bbox_inches='tight', dpi=900)
    plt.clf()

    sns.heatmap(model_acc_df.to_numpy(), annot=True, xticklabels=gs_choices,
                yticklabels=[int(i * 0.8) for i in data_num_choices], cmap="YlGnBu")
    plt.xlabel("Number of data points in one sparse data sample")
    plt.ylabel("Number of Training data")
    plt.title(f"Highest $R^2$ Score Achieved on Test Set by {model_dict[m]} in Classification Task (Setting I)")
    plt.savefig(f'./plot/{m}_r2.png', bbox_inches='tight', dpi=900)
    plt.clf()

sns.lineplot(data=cla, x='best_valid_acc', style='model', y='gs', hue='data_number')
plt.savefig(f'./plot/setting2_all.png', bbox_inches='tight', dpi=900)
plt.clf()
