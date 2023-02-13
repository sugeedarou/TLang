import torch.nn as nn

from settings import *
from twitter_dataset import TwitterDataset


class GRUModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.rnn = nn.GRU(input_size=TwitterDataset.num_characters+1,
                          hidden_size=128, # 256
                          num_layers=2,   # 3
                          bidirectional=True, dropout=0.5,
                          batch_first=True)
        self.fc = nn.Linear(self.rnn.hidden_size*2, self.output_size)

    def forward(self, batch):
        out, _ = self.rnn(batch)
        out = self.fc(out[:, -1])
        return out
