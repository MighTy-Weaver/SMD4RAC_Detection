import glob
import itertools
import os
import sys

sys.path.append('../')
from setting_2.utils import data_num_choices, gs_choices

models = ['transformer']

data = glob.glob('./data/trn*.npy')
data_settings = [(int(setting.split('_')[1]), int(setting.split('_')[3].split('.')[0])) for setting in data]

for dm, gs, m in itertools.product(data_num_choices, gs_choices, models):
    if (dm, gs) in data_settings and not os.path.exists(
            f"./ckpt/{m}_checkpoint_bs64_e200_lr5e-05_modesparse_gs{gs}_rat0.8_roomrat1_numdata{dm}/") and not os.path.exists(
        f"./ckpt/{m}_checkpoint_bs64_e100_lr5e-05_modesparse_gs{gs}_rat0.8_roomrat1_numdata{dm}/") and not os.path.exists(
        f"./ckpt/{m}_checkpoint_bs65_e100_lr5e-05_modesparse_gs{gs}_rat0.8_roomrat1_numdata{dm}/"):
        os.system(f"python train.py --model {m} --data {dm} --gs {gs} --gpu 1")
