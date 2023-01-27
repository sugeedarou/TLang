import torch.nn as nn

from settings import *
from twitter_dataset import TwitterDataset


class GRUModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.rnn = nn.GRU(input_size=TwitterDataset.num_characters+1,
                          hidden_size=128,
                          num_layers=3,
                          bidirectional=True, dropout=0.5,
                          batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.rnn.hidden_size*2, self.output_size)

    def forward(self, batch):
        out, _ = self.rnn(batch)
        out = self.dropout(out[:, -1])
        out = self.fc(out)
        return out
