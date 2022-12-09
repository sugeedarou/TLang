import torch.nn as nn

from settings import *
from dataset import Dataset


class GRUModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.rnn = nn.GRU(input_size=Dataset.characters_count+1,
                          hidden_size=512,
                          num_layers=4,
                          bidirectional=True, dropout=0.5,
                          batch_first=True)
        self.fc = nn.Linear(self.rnn.hidden_size*2, self.output_size)

    def forward(self, batch):
        out, _ = self.rnn(batch)
        out = self.fc(out[:, -1])
        return out
