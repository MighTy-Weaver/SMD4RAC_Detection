import numpy as np
import torch
import torch.nn.functional as F
from numpy import mean

efficiency_dict = dict(np.load('../data/efficiency_dict.npy', allow_pickle=True).item())
all_room_list = list(efficiency_dict.keys())
all_room_efficiency_list = list(efficiency_dict.values())
normal_room_list = [i for i in all_room_list if efficiency_dict[i] >= mean(all_room_efficiency_list)]
poor_room_list = [i for i in all_room_list if i not in normal_room_list]
print("Efficiency list loaded, {} normal rooms and {} poor rooms".format(len(normal_room_list), len(poor_room_list)))


def cosine_similarity_loss(x, y):
    """
    This function calculates the cosine similarity loss between two tensors: 1 - cosine similarity
    :param x: Input tensor x
    :param y: Input tensor y
    :return: 1 - cosine similarity between two tensors
    """
    return 1.0 - F.cosine_similarity(x, y)


def l_infinity(x, y):
    """
    This function calculates the l_infinity loss between two tensors: max(abs(x-y))
    :param x: Input tensor x
    :param y: Input tensor y
    :return: max(abs(x-y))
    """
    return torch.max(torch.abs(x - y), dim=1).values


def get_AC_efficiency_class(room: int):
    """
    This function returns the AC efficiency class of a given room
    :param room: An integer as the room number
    :return: True/False, True -> Normal AC, False -> Poor AC
    """
    assert room in efficiency_dict.keys(), 'Room {} is not in the manually checked rooms'.format(room)
    return efficiency_dict[room] >= 1


def get_AC_efficiency(room: int):
    """
    This function returns the AC efficiency of a given room
    :param room: An integer as the room number
    :return: The AC efficiency of that room
    """
    assert room in efficiency_dict.keys(), 'Room {} is not in the manually checked rooms'.format(room)
    return efficiency_dict[room]
