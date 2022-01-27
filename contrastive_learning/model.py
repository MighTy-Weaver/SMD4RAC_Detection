import torch
from numba.cuda import const
from torch import nn
from torch.autograd import Variable
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


class NN(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(NN, self).__init__()
        self.nn1 = Sequential(nn.Linear(in_features=input_dimension, out_features=64),
                              nn.BatchNorm1d(64), nn.Dropout(0.4), nn.ReLU())
        self.nn2 = nn.Linear(in_features=64, out_features=output_dimension)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.nn1(x)
        out = self.nn2(out)
        out = self.softmax(out)
        return out


class Attention_BiLSTM(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embed_dim, bidirectional, dropout, use_cuda,
                 attention_size, sequence_length):
        super(Attention_BiLSTM, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.sequence_length = sequence_length
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=const.PAD)
        self.lookup_table.weight.data.uniform_(-1., 1.)

        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size *= 2
        else:
            self.layer_size = self.layer_size

        self.attention_size = attention_size
        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

        self.label = nn.Linear(hidden_size * self.layer_size, output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (sequence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        # print(output_reshape.size()) = (sequence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (sequence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (sequence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # print(exps.size()) = (batch_size, sequence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, sequence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        # print(alphas_reshape.size()) = (batch_size, sequence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return torch.sum(state * alphas_reshape, 1)

    def forward(self, input_sentences, batch_size=None):
        input = self.lookup_table(input_sentences)
        input = input.permute(1, 0, 2)

        if self.use_cuda:
            h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        attn_output = self.attention_net(lstm_output)
        return self.label(attn_output)
