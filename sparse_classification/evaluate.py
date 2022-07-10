import glob
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

if not os.path.exists('./results/'):
    os.mkdir('./results/')
warnings.filterwarnings('ignore')

checkpoints = glob.glob('./ckpt/*checkpoint*/')

gs_choices = [6, 12, 18, 24, 48, 72, 96, 120, 144, 192]
data_num_choices = [2000, 5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000, 300000]

model_choices = ['lstm', 'bilstm', 'transformer', 'lstm-transformer', 'bilstm-transformer']

model_dict = {}
csv_record = pd.DataFrame(
    columns=['model', 'gs', 'data_number', 'best_train_acc', 'best_valid_acc', 'best_train_f1', 'best_valid_f1'])

for f in tqdm(checkpoints):
    info = f.split('_')
    model_version = info[0].replace("./", '')
    epoch_num = int(info[3].replace('e', ''))
    gs = int(info[6].replace('gs', ''))
    data_num = int(info[-1].replace('numdata', '').replace('/', ''))
    record = np.load(f'{f}/record.npy', allow_pickle=True).item()
    train_pred = np.load(f'{f}/best_train_pred.npy', allow_pickle=True).tolist()
    train_label = np.load(f'{f}/best_train_label.npy', allow_pickle=True).tolist()
    valid_pred = np.load(f'{f}/best_valid_pred.npy', allow_pickle=True).tolist()
    valid_label = np.load(f'{f}/best_valid_label.npy', allow_pickle=True).tolist()
    model_dict[(model_version, gs, data_num)] = [train_pred, valid_pred, train_label, valid_label]
    if len(record['trn_acc']) < epoch_num:
        print("\nWARNING: model: {} gs: {} data: {} hasn't fully ran".format(model_version, gs, data_num))
    csv_record = csv_record.append(
        {'model': model_version, 'gs': gs, 'data_number': data_num, 'best_train_acc': max(record['trn_acc']),
         'best_valid_acc': max(record['val_acc']), 'best_train_f1': max(record['trn_f1']),
         'best_valid_f1': max(record['val_f1'])}, ignore_index=True)
csv_record.sort_values(by=['best_valid_acc', 'best_valid_f1'], ascending=False).to_csv(
    './results/sparse_classification_record.csv', index=False)
np.save('./results/sparse_classification_statistics.npy', model_dict)
