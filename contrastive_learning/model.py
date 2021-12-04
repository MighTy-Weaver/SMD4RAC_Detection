from torch import nn
from torch.nn import Sequential


class LSTM_encoder(nn.Module):
    def __init__(self):
        super(LSTM_encoder, self).__init__()
        pass

    def forward(self, x):
        return x


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
