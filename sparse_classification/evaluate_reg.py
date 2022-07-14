import glob
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

if not os.path.exists('./results/'):
    os.mkdir('./results/')
warnings.filterwarnings('ignore')

checkpoints = glob.glob('./reg_ckpt/*regpoint*/')

gs_choices = [6, 12, 18, 24, 48, 72, 96, 120, 144, 192]
data_num_choices = [2000, 5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000, 300000]

model_choices = ['lstm', 'bilstm', 'transformer', 'lstm-transformer', 'bilstm-transformer']

model_dict = {}
csv_record = pd.DataFrame(
    columns=['model', 'gs', 'data_number'])

for f in tqdm(checkpoints):
    info = f.split('_')
    model_version = info[1].replace("./reg_ckpt/", '')
    epoch_num = int(info[4].replace('e', ''))
    gs = int(info[7].replace('gs', ''))
    data_num = int(info[-1].replace('numdata', '').replace('/', ''))
    try:
        record = np.load(f'{f}/record.npy', allow_pickle=True).item()
        train_pred = np.load(f'{f}/best_train_pred.npy', allow_pickle=True).tolist()
        train_label = np.load(f'{f}/best_train_label.npy', allow_pickle=True).tolist()
        valid_pred = np.load(f'{f}/best_valid_pred.npy', allow_pickle=True).tolist()
        valid_label = np.load(f'{f}/best_valid_label.npy', allow_pickle=True).tolist()
        model_dict[(model_version, gs, data_num)] = [train_pred, valid_pred, train_label, valid_label]
        if len(record['trn_r2']) < epoch_num:
            print(
                "\nWARNING: model: {} gs: {} data: {} hasn't fully ran. Currently finished {}/{}".format(model_version,
                                                                                                         gs,
                                                                                                         data_num, len(
                        record['trn_r2']), epoch_num))
            # os.system("rm -rf {}".format(f))
        csv_record = csv_record.append(
            {'model': model_version, 'gs': gs, 'data_number': data_num, 'best_train_r2': max(record['trn_r2']),
             'best_valid_r2': max(record['val_r2']), 'best_train_rmse': max(record['trn_rmse']),
             'best_valid_rmse': max(record['val_rmse'])}, ignore_index=True)
    except FileNotFoundError:
        print("\nWARNING: model: {} gs: {} data: {} hasn't ran yet. Currently finished 0/{}".format(model_version, gs,
                                                                                                    data_num,
                                                                                                    epoch_num))
csv_record.sort_values(by=['best_valid_r2', 'best_valid_rmse'], ascending=False).to_csv(
    './results/sparse_regression_record.csv', index=False)
np.save('./results/sparse_regression_statistics.npy', model_dict)
