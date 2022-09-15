import os.path
import sys

from tqdm import tqdm

sys.path.append('../')

from setting_1.dataloader import AC_sparse_separate_dataset
from setting_1.utils import gs_choices, data_num_choices

bar = tqdm(range(len(gs_choices) * (len(data_num_choices))))

for dm in data_num_choices:
    for gs in gs_choices:
        if os.path.exists(f'./data/val_{dm}_0.8_{gs}.npy'):
            print(f'./data/val_{dm}_0.8_{gs}.npy')
            bar.update(1)
            print('skipping {} {}'.format(gs, dm))
            continue
        try:
            dataset = AC_sparse_separate_dataset('trn', test=False, group_size=gs, trn_ratio=0.8, cla=True,
                                                 total_number=dm)
            val_dataset = AC_sparse_separate_dataset('val', test=False, group_size=gs, trn_ratio=0.8, cla=True,
                                                     total_number=dm)
            trn_dataset_length = len(dataset)
            val_dataset_length = len(val_dataset)
            bar.update(1)
            print(f"{gs}\t{dm}\t{trn_dataset_length}\t{val_dataset_length}\n")
        except Exception:
            print(gs, dm, 'Failed')
