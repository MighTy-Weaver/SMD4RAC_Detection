import torch
from torch.utils.data import DataLoader

from dataloader import AC_Normal_Dataset
from model import LSTM_encoder

dataset = AC_Normal_Dataset()
dl = DataLoader(dataset, batch_size=64, shuffle=True)
model = LSTM_encoder()

test, label = dataset[0]
a, (b, c) = model.LSTM(torch.tensor(test, dtype=torch.float32).unsqueeze(0))

print(a.shape, b.shape, c.shape)
print(test.shape)
print(len(dl))

for iter, input in enumerate(dl):
    pass
