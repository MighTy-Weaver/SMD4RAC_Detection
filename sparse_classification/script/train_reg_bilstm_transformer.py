import glob
import itertools
import os
import sys

sys.path.append('../')
from sparse_classification.utils import data_num_choices, gs_choices

models = ['bilstm-transformer']

data = glob.glob('./data_reg/trn*.npy')
data_settings = [(int(setting.split('_')[2]), int(setting.split('_')[4].split('.')[0])) for setting in data]

for dm, gs, m in itertools.product(data_num_choices, gs_choices, models):
    if (dm, gs) in data_settings and not os.path.exists(
            f"./reg_ckpt/{m}_regpoint_bs64_e200_lr5e-05_modesparse_gs{gs}_rat0.8_numdata{dm}/") and not os.path.exists(
        f"./reg_ckpt/{m}_regpoint_bs64_e100_lr5e-05_modesparse_gs{gs}_rat0.8_numdata{dm}/"):
        os.system(f"python regression.py --model {m} --data {dm} --gs {gs} --gpu 3")
