import torch.nn as nn

from settings import *
from dataset import Dataset


class GRUModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.rnn = nn.GRU(input_size=Dataset.characters_count+1,
                          hidden_size=256,
                          num_layers=3,
                          bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(self.rnn.hidden_size*2, self.output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch):
        out, _ = self.rnn(batch)
        out = self.dropout(out)
        out = self.fc(out[-1])
        return out
