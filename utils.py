import numpy as np
from numpy import mean

replace_2016 = [714, 503, 1012, 235, 520, 735, 220, 335, 619, 817, 807, 202, 424, 801, 211, 402, 201, 326, 306, 429,
                414, 715, 311, 330]
replace_2017 = [432, 802, 227, 231, 733, 210, 315, 427, 430, 612, 613, 626, 630, 704, 914, 123, 307, 903]
replace_2018 = [219, 516, 417, 605, 816, 703, 803, 818, 915, 122, 207, 310, 320, 824, 518, 530, 913]
replace_2019 = [822, 730, 608, 617, 708, 825, 204, 216, 413, 703, 725, 810, 410, 830, 523, 618, 415, 328, 1007, 821,
                332]
replace_2020 = [808, 819, 403, 716, 303, 334, 832, 401, 622]
replace_2021 = [604, 702, 735, 217, 517, 710]

replace_dict = {2016: replace_2016, 2017: replace_2017, 2018: replace_2018, 2019: replace_2019, 2020: replace_2020,
                2021: replace_2021}

gs_choices = [6, 12, 18, 24, 48, 72, 96, 120, 144, 192]
data_num_choices = [2000, 5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 1000000, 2000000]
model_choices = ['lstm', 'bilstm', 'transformer', 'lstm-transformer', 'bilstm-transformer']

efficiency_dict = dict(np.load('./data/efficiency_dict.npy', allow_pickle=True).item())
all_room_list = list(efficiency_dict.keys())
all_room_efficiency_list = list(efficiency_dict.values())
normal_room_list = [i for i in all_room_list if efficiency_dict[i] >= mean(all_room_efficiency_list)]
poor_room_list = [i for i in all_room_list if i not in normal_room_list]
