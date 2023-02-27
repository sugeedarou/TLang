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

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.cell_class = "GRU" if cell_class == GRUCell else "LSTM"
        self.dropout = nn.Dropout(dropout)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.layer_cells_f = nn.ModuleList([cell_class(self.input_size,
                                                       self.hidden_size,
                                                       self.bias)] +
                                           [cell_class(self.hidden_size,
                                                       self.hidden_size,
                                                       self.bias)
                                            for _ in range(1, num_layers)])

        if bidirectional:
            self.layer_cells_b = nn.ModuleList([cell_class(self.input_size,
                                                           self.hidden_size,
                                                           self.bias)] +
                                               [cell_class(self.hidden_size,
                                                           self.hidden_size,
                                                           self.bias)
                                                for _ in range(1, num_layers)])

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

        hf = Variable(torch.zeros(self.num_layers, input.size(0),
                                  self.hidden_size).to(self.device))

        if self.bidirectional:
            hb = Variable(torch.zeros(self.num_layers, input.size(0),
                                      self.hidden_size).to(self.device))

        hidden_f = list()
        outs_f = torch.empty(size=(input.size(1), input.size(0), self.hidden_size),
                             dtype=torch.float).to(self.device)

        if self.bidirectional:
            hidden_b = list()
            outs_b = torch.empty(size=(input.size(1), input.size(0), self.hidden_size),
                                 dtype=torch.float).to(self.device)

        # create all layers
        for layer in range(self.num_layers):
            if self.cell_class == "LSTM":
                hidden_f.append(hf[layer, :, :], hf[layer, :, :])
                if self.bidirectional:
                    hidden_b.append(hb[layer, :, :], hb[layer, :, :])
            else:
                hidden_f.append(hf[layer, :, :])
                if self.bidirectional:
                    hidden_b.append(hb[layer, :, :])

        for layer in range(self.num_layers):
            for t in range(input.size(1)):
                if layer == 0:
                    parent_f = input[:, t, :]
                    if self.bidirectional:
                        parent_b = input[:, -(t + 1), :]
                else:
                    parent_f = hidden_f[layer - 1]
                    if self.bidirectional:
                        parent_b = hidden_b[layer - 1]

                if self.cell_class == "LSTM":
                    hidden_layer_f = self.layer_cells_f[layer](
                        parent_f, (hidden_f[layer][0], hidden_f[layer][1]))
                    if self.bidirectional:
                        hidden_layer_b = self.layer_cells_b[layer](
                            parent_b, (hidden_b[layer][0], hidden_b[layer][1]))
                else:  # GRU
                    hidden_layer_f = self.layer_cells_f[layer](
                        parent_f, hidden_f[layer])
                    if self.bidirectional:
                        hidden_layer_b = self.layer_cells_b[layer](
                            parent_b, hidden_b[layer])

                hidden_f[layer] = self.dropout(hidden_layer_f)
                if self.bidirectional:
                    hidden_b[layer] = self.dropout(hidden_layer_b)

                if self.cell_class == "LSTM":
                    outs_f[t] = hidden_layer_f[0]
                    if self.bidirectional:
                        outs_b[t] = hidden_layer_b[0]

                else:  # gru
                    outs_f[t] = hidden_layer_f
                    if self.bidirectional:
                        outs_b[t] = hidden_layer_b
        if self.bidirectional:
            # cat as longer sequence
            out = torch.cat((outs_f[-1], outs_b[-1]), 1)
        else:
            out = outs_f[-1]

        return out


class GRU(GeneralRNN):
    '''
        GeneralRNN with cell_class=GRUCell
    '''

    def __init__(self, input_size, hidden_size, num_layers, bias=True,
                 bidirectional=False, batch_first=False, dropout=0.):
        super().__init__(input_size, hidden_size, num_layers, bias=bias,
                         cell_class=GRUCell, bidirectional=bidirectional,
                         batch_first=batch_first, dropout=dropout)

    def forward(self, input):
        return super().forward(input)
