import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMCell(nn.Module):
    '''
        a single LSTM Cell, used to build lstm-chains as rnn
    '''
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        #4 times as long, because we need it for 4 gates
        self.input_lf = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.hidden_lf = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
     #   self.reset_parameters()

    #def reset_parameters(self):
    #    std = 1.0 / np.sqrt(self.hidden_size)
    #    for w in self.parameters():
    #        w.data.uniform_(-std, std)

    def forward(self, input, h_c_in=None):
        '''
        :param input:
            shape (batch_size, input_size)
        :param h_c_in: hidden state and cell state (h_in, c_in)
            shape (2, batch_size, hidden_size)
        :return: h_c_out: (hidden_state, cell_state)
            shape: (2, batch_size, hidden_size)
        '''

        if h_c_in is None:
            h_c_in = Variable(input.new_zeros(input.size(0), self.hidden_size))
            h_c_in = (h_c_in, h_c_in)

        h_in, c_in = h_c_in

        gates = self.input_lf(input) + self.hidden_lf(h_in)

        #split 4 times the sized network in the 4 actual sized ones for gate funcs
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        c_out = c_in * f_t + i_t * g_t

        h_out = o_t * torch.tanh(c_out)


        return (h_out, c_out)


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        #3 times the size as 3 times with different weights needed
        self.input_lf = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.hidden_lf = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        #self.reset_parameters()


    #def reset_parameters(self):
    #    std = 1.0 / np.sqrt(self.hidden_size)
    #    for w in self.parameters():
    #        w.data.uniform_(-std, std)

    def forward(self, input, h_in=None):
        '''
            :param input:
                shape (batch_size, input_size)
            :param h_in: hidden state
                shape (batch_size, hidden_size)
            :return: h_out: hidden_state
                shape (batch_size, hidden_size)
        '''

        if h_in is None:
            h_in = Variable(input.new_zeros(input.size(0), self.hidden_size))

        tripled_input = self.input_lf(input)
        tripled_hidden = self.hidden_lf(h_in)

        x_reset, x_upd, x_out = tripled_input.chunk(3, 1)
        h_reset, h_upd, h_out = tripled_hidden.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        candidate = torch.tanh(x_out + (reset_gate * h_out))
        h_out = update_gate * h_in + (1 - update_gate) * candidate

        return h_out
