import numpy as np
from numpy import mean

efficiency_dict = dict(np.load('../data/efficiency_dict.npy', allow_pickle=True).item())
all_room_list = list(efficiency_dict.keys())
all_room_efficiency_list = list(efficiency_dict.values())
normal_room_list = [i for i in all_room_list if efficiency_dict[i] >= mean(all_room_efficiency_list)]
poor_room_list = [i for i in all_room_list if i not in normal_room_list]
print("Efficiency list loaded, {} normal rooms and {} poor rooms".format(len(normal_room_list), len(poor_room_list)))
print("Mean efficiency is {}".format(mean(all_room_efficiency_list)))

