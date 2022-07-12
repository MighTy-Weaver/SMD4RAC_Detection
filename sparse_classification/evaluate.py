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

model_dict = {}
csv_record = pd.DataFrame(
    columns=['model', 'gs', 'data_number', 'best_train_acc', 'best_valid_acc', 'best_train_f1', 'best_valid_f1'])

for f in tqdm(checkpoints):
    info = f.split('_')
    model_version = info[0].replace("./", '')
    epoch_num = int(info[3].replace('e', ''))
    gs = int(info[6].replace('gs', ''))
    data_num = int(info[-1].replace('numdata', '').replace('/', ''))
    try:
        record = np.load(f'{f}/record.npy', allow_pickle=True).item()
        train_pred = np.load(f'{f}/best_train_pred.npy', allow_pickle=True).tolist()
        train_label = np.load(f'{f}/best_train_label.npy', allow_pickle=True).tolist()
        valid_pred = np.load(f'{f}/best_valid_pred.npy', allow_pickle=True).tolist()
        valid_label = np.load(f'{f}/best_valid_label.npy', allow_pickle=True).tolist()
        model_dict[(model_version, gs, data_num)] = [train_pred, valid_pred, train_label, valid_label]
        if len(record['trn_acc']) < epoch_num:
            print(
                "\nWARNING: model: {} gs: {} data: {} hasn't fully ran. Currently finished {}/{}".format(model_version,
                                                                                                         gs,
                                                                                                         data_num, len(
                        record['trn_acc']), epoch_num))
            # os.system("rm -rf {}".format(f))
        csv_record = csv_record.append(
            {'model': model_version, 'gs': gs, 'data_number': data_num, 'best_train_acc': max(record['trn_acc']),
             'best_valid_acc': max(record['val_acc']), 'best_train_f1': max(record['trn_f1']),
             'best_valid_f1': max(record['val_f1'])}, ignore_index=True)
    except FileNotFoundError:
        print("\nWARNING: model: {} gs: {} data: {} hasn't ran yet. Currently finished 0/{}".format(model_version, gs,
                                                                                                    data_num,
                                                                                                    epoch_num))
csv_record.sort_values(by=['best_valid_acc', 'best_valid_f1'], ascending=False).to_csv(
    './results/sparse_classification_record.csv', index=False)
np.save('./results/sparse_classification_statistics.npy', model_dict)
