import numpy as np

efficiency_dict = dict(np.load('../data/efficiency_dict.npy', allow_pickle=True).item())
all_room_list = list(efficiency_dict.keys())
