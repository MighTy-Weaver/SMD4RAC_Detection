import glob
import os

data = glob.glob('./data/trn*.npy')
ckpts = glob.glob('./ckpt/*+checkpoint*/')

models = ['transformer']

for d in data:
    setting = d.split('_')
    data_num = setting[1]
    gs = setting[3].split('.')[0]
    for m in models:
        if not os.path.exists(
                "./ckpt/{}_checkpoint_bs64_e200_lr5e-05_modesparse_gs{}_rat0.8_roomrat1_numdata{}/".format(m, gs,
                                                                                                           data_num)):
            os.system("python train.py --model {} --data {} --gs {} --gpu 3".format(m, data_num, gs))
        else:
            print(m, gs, data_num, 'Already trained')
