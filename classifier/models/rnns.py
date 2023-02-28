import torch
import torch.nn as nn
from models.rnn_cells import GRUCell

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(BiGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.forward_cells = nn.ModuleList([nn.GRUCell(self.input_size, self.hidden_size)] +
                                           [nn.GRUCell(2 * self.hidden_size, self.hidden_size) for _ in range(1, num_layers)])
        self.backward_cells = nn.ModuleList([nn.GRUCell(self.input_size, self.hidden_size)] +
                                           [nn.GRUCell(2 * self.hidden_size, self.hidden_size) for _ in range(1, num_layers)])

    def forward(self, input):
        '''
        :param input: (sequence_length, batch_size, feature_size)
        :return: (sequence_length, batch_size, 2*hidden_size)
        '''
        seq_length = input.size(0)
        batch_size = input.size(1)
        h_f = [torch.zeros(seq_length, batch_size, self.hidden_size, device=self.device)
               for _ in range(self.num_layers)]
        h_b = [torch.zeros(seq_length, batch_size, self.hidden_size, device=self.device)
               for _ in range(self.num_layers)]
        out = torch.zeros((seq_length, batch_size, 2*self.hidden_size), device=self.device)

        last_hidden = input

        for i in range(self.num_layers):
            for t in range(seq_length):
                x1 = last_hidden[t]
                x2 = last_hidden[-(t+1)]
                h_f[i][t] = self.forward_cells[i](x1, h_f[i][t-1])
                h_b[i][t] = self.backward_cells[i](x2, h_b[i][t-1])
            
            last_hidden = torch.cat((h_f[i], h_b[i].flip(0)), dim=2)

            if i == self.num_layers - 1:
                out = last_hidden
            if self.dropout > 0:
                last_hidden = nn.Dropout(p=self.dropout)(last_hidden)
        return out
