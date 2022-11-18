import torch.nn as nn

from settings import *
from dataset import Dataset


class GRUModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.rnn = nn.GRU(input_size=Dataset.get_class_names()
                           hidden_size=128,
                           num_layers=3,
                           batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(self.rnn.hidden_size*2, self.output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, batch, _lengths, _device):
        out, _ = self.rnn(batch)
        out = self.fc(out)
        out = self.dropout(out)
        return out