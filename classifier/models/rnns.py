from models.rnn_cells import *
import torch
import torch.nn as nn
from torch.autograd import Variable


class GeneralRNN(nn.Module):
    '''
        general rnn cell class, supports GRU and LSTM cells
    '''

    def __init__(self, input_size, hidden_size, num_layers, bias=True,
                 cell_class=GRUCell, bidirectional=False, batch_first=False, dropout=0.):
        super(GeneralRNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.cell_class = "GRU" if cell_class == GRUCell else "LSTM"
        self.dropout = nn.Dropout(dropout)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(cell_class(self.input_size,
                                             self.hidden_size,
                                             self.bias))
        for _ in range(1, self.num_layers):
            self.rnn_cell_list.append(cell_class(self.hidden_size,
                                                 self.hidden_size,
                                                 self.bias))

    def forward(self, input):
        '''
        :param input: input batch:
            shape:  if batch_first:     (batch_size, sequence_length, feature_size)
                    if not batch_first: (sequence_length, batch_size, feature_size)
        :param h_in:
        :return:
        '''

        if not self.batch_first:  # always work with batch_first
            input = torch.permute(input, (1, 0, 2))

        h0 = Variable(torch.zeros(self.num_layers, input.size(0),
                                  self.hidden_size).to(self.device))

        if self.bidirectional:
            hT = Variable(torch.zeros(self.num_layers, input.size(0),
                                      self.hidden_size).to(self.device))

        out = torch.empty(size=self.num_layers, dtype=torch.float)
        hidden = list()
        out_rev = torch.empty(size=self.num_layers, dtype=torch.float)
        hidden_backward = list()

        # create all layers
        for layer in range(self.num_layers):
            if self.cell_class == "LSTM":
                hidden.append(h0[layer, :, :], h0[layer, :, :])
                if self.bidirectional:
                    hidden_backward.append(hT[layer, :, :], hT[layer, :, :])
            else:
                hidden.append(h0[layer, :, :])
                if self.bidirectional:
                    hidden_backward.append(hT[layer, :, :])

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    parent_f = input[:, t, :]
                    parent_b = input[:, -(t + 1), :]
                else:
                    parent_f = hidden[layer - 1][0]
                    parent_b = hidden_backward[layer - 1][0]

                if self.cell_class == "LSTM":
                    hidden_layer_f = self.rnn_cell_list[layer](parent_f, (hidden[layer][0], hidden[layer][1]))
                    if self.bidirectional:
                        hidden_layer_b = self.rnn_cell_list[layer](parent_b, (hidden_backward[layer][0], hidden_backward[layer][1]))
                else:  # GRU
                    hidden_layer_f = self.rnn_cell_list[layer](parent_f, hidden[layer])
                    if self.bidirectional:
                        hidden_layer_b = self.rnn_cell_list[layer](parent_b, hidden_backward[layer])

                hidden[layer] = hidden_layer_f
                if self.bidirectional:
                    hidden_backward[layer] = hidden_layer_b

            if self.cell_class == "LSTM":
                out[layer] = hidden_layer_f[0]
                if layer != self.num_layers-1:
                    out[layer] = self.dropout(out[layer])
                if self.bidirectional:
                    out_rev[layer] = hidden_layer_b[0]
                    if layer != self.num_layers-1:
                        out_rev[layer] = self.dropout(out_rev[layer])

            else:  # gru
                out[layer] = hidden_layer_f
                if self.bidirectional:
                    out[layer] = hidden_layer_b

        if self.bidirectional:
            out = torch.cat((out, out_rev), 1)  # cat as longer sequence

        if not self.batch_first:
            out = torch.permute(out, (1, 0, 2))

        return out

class GRU(GeneralRNN):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout=0.,
                 bidirectional=False, batch_first=False):
        super().__init__(input_size, hidden_size, num_layers, bias=bias, dropout=dropout,
                 cell_class=GRUCell, bidirectional=bidirectional, batch_first=batch_first)
    def forward(self, input):
        super().forward(input)