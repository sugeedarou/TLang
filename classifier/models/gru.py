from settings import *
from twitter_dataset import TwitterDataset
from models.rnns import *

import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.rnn = nn.GRU(input_size=TwitterDataset.num_characters+1,
                          hidden_size=256, # 256
                          num_layers=3,   # 3
                          bidirectional=True, dropout=0.4,
                          batch_first=True)
        self.fc = nn.Linear(self.rnn.hidden_size*2, self.output_size)

    def forward(self, batch):
        out, _ = self.rnn(batch)
        # print(out.shape)
        # print(out[:, -1].shape)
        # exit()
        out = self.fc(out[:, -1])
        return out
