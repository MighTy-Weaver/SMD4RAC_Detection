import torch
from torch import nn
from torch.nn import Sequential


class simple_LSTM_encoder(nn.Module):
    def __init__(self, feature_num=11, hidden_size=32, num_layers=4, bias=True, batch_first=True, bidirectional=True,
                 dropout=0.2):
        super(simple_LSTM_encoder, self).__init__()
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
        # print(seq_avg.shape, h0_avg.shape, c0_avg.shape)  # ,torch.cat((seq_avg, h0_avg, c0_avg), dim=1).shape)
        try:
            return self.bn(torch.cat((seq_avg, h0_avg, c0_avg), dim=1))
        except IndexError:
            return self.bn(torch.cat((seq_avg, h0_avg, c0_avg), dim=-1).unsqueeze(0))

    def get_output_length(self):
        return 3 * self.hidden_size if not self.bidirectional else 4 * self.hidden_size


class Transformer_encoder(nn.Module):
    def __init__(self, gs, feature_num=12, num_head=8, num_layers=6, LSTM_hidden_size=128, LSTM_num_layers=3,
                 LSTM_bias=True, bidirectional=True, dropout=0.2, mode='flat'):
        super(Transformer_encoder, self).__init__()
        self.feature_num = feature_num
        self.gs = gs
        self.num_head = num_head
        self.num_layers = num_layers
        self.LSTM_hidden_size = LSTM_hidden_size
        self.LSTM_num_layers = LSTM_num_layers
        self.LSTM_bias = LSTM_bias
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.mode = mode

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_num, nhead=self.num_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.LSTM = nn.LSTM(input_size=feature_num, hidden_size=LSTM_hidden_size, num_layers=LSTM_num_layers,
                            bias=LSTM_bias, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.bn = nn.BatchNorm1d(self.get_output_length())

    def forward(self, x):
        encode = self.encoder(x)
        if self.mode == 'flat':
            out = torch.flatten(encode, start_dim=1)
            return out
        out, (h0, c0) = self.LSTM(encode)
        seq_avg = torch.mean(out, dim=1).squeeze()  # (bs, 2 * hidden size)
        h0_avg = torch.mean(h0, dim=0).squeeze()  # (bs, hidden size)
        c0_avg = torch.mean(c0, dim=0).squeeze()  # (bs, hidden size)
        # print(seq_avg.shape, h0_avg.shape, c0_avg.shape)  # ,torch.cat((seq_avg, h0_avg, c0_avg), dim=1).shape)
        try:
            return self.bn(torch.cat((seq_avg, h0_avg, c0_avg), dim=1))
        except IndexError:
            return self.bn(torch.cat((seq_avg, h0_avg, c0_avg), dim=-1).unsqueeze(0))

    def get_output_length(self):
        return (
            4 * self.LSTM_hidden_size if self.mode != 'flat' else self.feature_num * self.gs) if self.bidirectional else 3 * self.LSTM_hidden_size


class NN_regressor(nn.Module):
    def __init__(self, encoder, output_dimension=1, cla=False):
        super(NN_regressor, self).__init__()
        self.encoder = encoder
        self.cla = cla
        self.nn1 = Sequential(nn.Linear(in_features=self.encoder.get_output_length(), out_features=512),
                              nn.BatchNorm1d(512), nn.Dropout(0.3), nn.ReLU())
        self.nn2 = Sequential(nn.Linear(in_features=512, out_features=128),
                              nn.BatchNorm1d(128), nn.Dropout(0.3), nn.ReLU())
        self.nn3 = Sequential(nn.Linear(in_features=128, out_features=64),
                              nn.BatchNorm1d(64), nn.Dropout(0.3), nn.ReLU())
        self.nn4 = nn.Linear(in_features=64, out_features=output_dimension)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.encoder(x)
        out = self.nn1(out)
        out = self.nn2(out)
        out = self.nn3(out)
        out = self.nn4(out)
        if self.cla:
            return self.softmax(out)
        return out
