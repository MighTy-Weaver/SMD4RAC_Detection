import numpy as np

efficiency_dict = dict(np.load('../data/efficiency_dict.npy', allow_pickle=True).item())
all_room_list = list(efficiency_dict.keys())
normal_room_list = [i for i in all_room_list if efficiency_dict[i] >= 1]
poor_room_list = [i for i in all_room_list if i not in normal_room_list]
