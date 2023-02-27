from models.rnn_cells import *
import torch
import torch.nn as nn
from torch.autograd import Variable


class BiGRU(nn.Module):
    '''
        Bidirectional GRU
    '''

    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout=0.):
        super(BiGRU, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.layer_cells_f = nn.ModuleList([GRUCell(self.input_size,
                                                        self.hidden_size,
                                                        self.bias)] +
                                            [GRUCell(2 * self.hidden_size,
                                                        self.hidden_size,
                                                        self.bias)
                                            for _ in range(1, num_layers)])

        self.layer_cells_b = nn.ModuleList([GRUCell(self.input_size,
                                                        self.hidden_size,
                                                        self.bias)] +
                                            [GRUCell(2 * self.hidden_size,
                                                        self.hidden_size,
                                                        self.bias)
                                            for _ in range(1, num_layers)])

    def forward(self, input):
        '''
        :param input: input batch:
            shape: (batch_size, sequence_length, feature_size)
        :param h_in:
        :return:
        '''

        batch_size = input.size(0)
        seq_length = input.size(1)

        # initialize hidden layers
        hf = torch.zeros(self.num_layers, batch_size, seq_length,
                                  self.hidden_size).to(self.device)
        hb = torch.zeros(self.num_layers, batch_size, seq_length, 
                                    self.hidden_size).to(self.device)
        
        # initialize outputs
        outs_f = torch.empty(size=(input.size(1), batch_size, self.hidden_size),
                             dtype=torch.float).to(self.device)
        outs_b = torch.empty(size=(input.size(1), batch_size, self.hidden_size),
                                dtype=torch.float).to(self.device)


        for layer in range(self.num_layers):
            print(f'layer {layer}')
            # use hidden states of forward and backward gru as next input sequence
            if layer == 0:
                parent = torch.cat((input, torch.zeros(input.shape, device=self.device)), 1)
                print(parent.shape)
            else:
                parent = torch.cat((hf[layer-1], hb[layer-1]), 1)
                print(parent.shape)

            for t in range(input.size(1)):
                hidden_layer_f = self.layer_cells_f[layer](
                    parent[:, t, :], hf[layer][t-1])
                hidden_layer_b = self.layer_cells_b[layer](
                    parent[:, -(t+1), :], hb[layer][-1])

                hf[layer][t] = self.dropout(hidden_layer_f)
                hf[layer][t] = self.dropout(hidden_layer_b)

                outs_f[t] = hidden_layer_f
                outs_b[t] = hidden_layer_b
        
        out = torch.cat((outs_f[-1], outs_b[-1]), 1)

        return out
