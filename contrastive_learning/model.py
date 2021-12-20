import torch
from torch import nn
from torch.nn import Sequential


class LSTM_encoder(nn.Module):
    def __init__(self, feature_num=11, hidden_size=128, num_layers=16, bias=True, batch_first=True, bidirectional=True,
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
        avg = torch.mean(out, dim=1)
        return self.bn(avg)

    def get_output_length(self):
        return self.hidden_size if not self.bidirectional else 2 * self.hidden_size


class NN_classifier(nn.Module):
    def __init__(self, output_dimension, encoder):
        super(NN_classifier, self).__init__()
        self.encoder = encoder
        self.nn1 = Sequential(nn.Linear(in_features=self.encoder.get_output_length(), out_features=64),
                              nn.BatchNorm1d(64), nn.Dropout(0.4), nn.ReLU())
        self.nn2 = nn.Linear(in_features=64, out_features=output_dimension)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.encoder(x)
        out = self.nn1(out)
        out = self.nn2(out)
        out = self.softmax(out)
        return out
