import torch
from torch import nn
from torch.nn import Sequential


class LSTM_encoder(nn.Module):
    def __init__(self, feature_num=11, hidden_size=32, num_layers=1, bias=True, batch_first=True, bidirectional=True,
                 dropout=0.4):
        super(LSTM_encoder, self).__init__()
        self.feature_num = feature_num
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.LSTM = nn.LSTM(input_size=feature_num, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                            batch_first=batch_first, bidirectional=bidirectional, dropout=dropout)
        self.bn = nn.BatchNorm1d(self.get_output_length())

    def forward(self, x):
        out, (h0, c0) = self.LSTM(x)
        seq_avg = torch.mean(out, dim=1).squeeze()  # (bs, 2 * hidden size)
        h0_avg = torch.mean(h0, dim=0).squeeze()  # (bs, hidden size)
        c0_avg = torch.mean(c0, dim=0).squeeze()  # (bs, hidden size)
        return self.bn(torch.cat((seq_avg, h0_avg, c0_avg), dim=0))

    def get_output_length(self):
        return 3 * self.hidden_size if not self.bidirectional else 4 * self.hidden_size


class NN_regressor(nn.Module):
    def __init__(self, encoder, output_dimension=1):
        super(NN_regressor, self).__init__()
        self.encoder = encoder
        self.nn1 = Sequential(nn.Linear(in_features=self.encoder.get_output_length(), out_features=64),
                              nn.BatchNorm1d(64), nn.Dropout(0.4), nn.ReLU())
        self.nn2 = nn.Linear(in_features=64, out_features=output_dimension)

    def forward(self, x):
        out = self.encoder(x)
        out = self.nn1(out)
        out = self.nn2(out)
        return out
