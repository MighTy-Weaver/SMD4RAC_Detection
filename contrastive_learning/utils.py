import numpy as np
import torch
import torch.nn.functional as F

efficiency_dict = dict(np.load('../data/efficiency_dict.npy', allow_pickle=True).item())
all_room_list = list(efficiency_dict.keys())
normal_room_list = [i for i in all_room_list if efficiency_dict[i] >= 1]
poor_room_list = [i for i in all_room_list if i not in normal_room_list]


def cosine_similarity_loss(x, y):
    """
    This function calculates the cosine similarity loss between two tensors by 1 - cosine similarity
    :param x: Input tensor x.
    :param y: Input tensor y.
    :return: 1 - cosine similarity between two tensors
    """
    return 1.0 - F.cosine_similarity(x, y)


def l_infinity(x, y):
    """
    This function calculates the l_infinity loss between two tensors: max(abs(x-y))
    :param x: Input tensor x.
    :param y: Input tensor y.
    :return: max(abs(x-y))
    """
    return torch.max(torch.abs(x - y), dim=1).values


def get_AC_efficiency_class(room: int):
    pass


def get_AC_efficiency(room: int):
    pass
