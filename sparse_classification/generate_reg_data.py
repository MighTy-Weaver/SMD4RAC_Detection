import os.path
import sys

from tqdm import tqdm

sys.path.append('../')

from sparse_classification.dataloader import AC_sparse_separate_dataset

gs_choices = [6, 12, 18, 24, 48, 72, 96, 120, 144, 192]
data_num_choices = [2000, 5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000, 300000]

bar = tqdm(range(len(gs_choices) * (len(data_num_choices))))

log = open('./reg_data_record.txt', 'w')

for gs in gs_choices:
    for dm in data_num_choices:
        if os.path.exists(f'./data_reg/val_{dm}_0.8_{gs}.npy'):
            print(f'./data_reg/val_{dm}_0.8_{gs}.npy')
            bar.update(1)
            print('skipping {} {}'.format(gs, dm))
            continue
        try:
            dataset = AC_sparse_separate_dataset('trn', test=False, group_size=gs, trn_ratio=0.8, cla=False,
                                                 total_number=dm, data_path='./data_reg/')
            val_dataset = AC_sparse_separate_dataset('val', test=False, group_size=gs, trn_ratio=0.8, cla=False,
                                                     total_number=dm, data_path='./data_reg/')
            trn_dataset_length = len(dataset)
            val_dataset_length = len(val_dataset)
            bar.update(1)
            log.write(f"{gs}\t{dm}\t{trn_dataset_length}\t{val_dataset_length}\n")
            print(f"{gs}\t{dm}\t{trn_dataset_length}\t{val_dataset_length}\n")
        except Exception:
            print(gs, dm, 'Failed')
            log.write(f'{gs} {dm} Failed\n')
log.close()
