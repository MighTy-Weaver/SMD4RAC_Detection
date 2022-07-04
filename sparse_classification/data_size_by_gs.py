import sys

from tqdm import tqdm

sys.path.append('../')
from sparse_classification.dataloader import AC_sparse_separate_dataset

gs_choices = [10, 20, 24, 25, 48, 50, 100, 150, 200]
data_num_choices = [10000, 20000, 50000, 100000, 150000, 200000, 300000, 400000]

bar = tqdm(range(len(gs_choices) * (len(data_num_choices))))

log = open('./record.txt', 'w')

for gs in gs_choices:
    for dm in data_num_choices:
        dataset = AC_sparse_separate_dataset('trn', test=False, group_size=gs, trn_ratio=0.8, cla=True, total_number=dm)
        val_dataset = AC_sparse_separate_dataset('val', test=False, group_size=gs, trn_ratio=0.8, cla=True,
                                                 total_number=dm)
        trn_dataset_length = len(dataset)
        val_dataset_length = len(val_dataset)
        bar.update(1)
        log.write(f"{gs}\t{dm}\t{trn_dataset_length}\t{val_dataset_length}\n")
        print(f"{gs}\t{dm}\t{trn_dataset_length}\t{val_dataset_length}\n")
log.close()
