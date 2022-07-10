import glob
import os

data = glob.glob('./data_reg/trn*.npy')

models = ['lstm-transformer']

for d in data:
    setting = d.split('_')
    data_num = setting[2]
    gs = setting[4].split('.')[0]
    for m in models:
        if not os.path.exists(
                "./reg_ckpt/{}_regpoint_bs64_e200_lr5e-05_modesparse_gs{}_rat0.8_numdata{}/".format(m, gs, data_num)):
            os.system("python regression.py --model {} --data {} --gs {} --gpu 1".format(m, data_num, gs))
        else:
            print(m, gs, data_num, 'Already trained')
