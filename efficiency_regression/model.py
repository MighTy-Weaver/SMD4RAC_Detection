import torch
from numba.cuda import const
from torch import nn
from torch.autograd import Variable
from torch.nn import Sequential


class simple_LSTM_encoder(nn.Module):
    def __init__(self, feature_num=11, hidden_size=32, num_layers=4, bias=True, batch_first=True, bidirectional=True,
                 dropout=0.4):
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
    def __init__(self, feature_num=12, num_head=8, num_layers=6, LSTM_hidden_size=32, LSTM_num_layers=4, LSTM_bias=True,
                 bidirectional=True, dropout=0.2):
        super(Transformer_encoder, self).__init__()
        self.feature_num = feature_num
        self.num_head = num_head
        self.num_layers = num_layers
        self.LSTM_hidden_size = LSTM_hidden_size
        self.LSTM_num_layers = LSTM_num_layers
        self.LSTM_bias = LSTM_bias
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_num, nhead=self.num_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers == self.num_layers)
        self.LSTM = nn.LSTM(input_size=feature_num, hidden_size=LSTM_hidden_size, num_layers=LSTM_num_layers,
                            bias=LSTM_bias, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.bn = nn.BatchNorm1d(self.get_output_length())

    def forward(self, x):
        encode = self.encoder(x)
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
        return 3 * self.LSTM_hidden_size if not self.bidirectional else 4 * self.LSTM_hidden_size


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


class bilstm_attn(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embed_dim, bidirectional, dropout, use_cuda,
                 attention_size, sequence_length):
        super(bilstm_attn, self).__init__()

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
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

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
        logits = self.label(attn_output)
        return logits
