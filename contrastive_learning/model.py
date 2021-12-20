from torch import nn
from torch.nn import Sequential


class LSTM_encoder(nn.Module):
    def __init__(self, feature_num=11, hidden_size=64, num_layers=16, bias=True, batch_first=True, bidirectional=True):
        super(LSTM_encoder, self).__init__()
        self.feature_num = feature_num
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.LSTM = nn.LSTM(input_size=feature_num, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                            batch_first=batch_first, bidirectional=bidirectional)
        self.dropout = nn.Dropout(0.4)
        self.bn = nn.BatchNorm1d(64)

    def forward(self, x):
        out = self.LSTM(x)
        out = self.dropout(out)
        out = self.bn(out)
        return out


class NN_classifier(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(NN_classifier, self).__init__()
        self.nn1 = Sequential(nn.Linear(in_features=input_dimension, out_features=64), nn.BatchNorm1d(64),
                              nn.Dropout(0.4), nn.ReLU())
        self.nn2 = nn.Linear(in_features=64, out_features=output_dimension)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.nn1(x)
        out = self.nn2(out)
        out = self.softmax(out)
        return out
