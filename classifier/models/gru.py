from settings import *
from twitter_dataset import TwitterDataset
from models.rnns import *

import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.rnn = BiGRU(input_size=TwitterDataset.num_characters+1,
                          hidden_size=128, # 256
                          num_layers=2,   # 3
                          dropout=0.4)
        
        self.fc = nn.Linear(self.rnn.hidden_size*2, self.output_size)

    def forward(self, batch):
        # out, _ = self.rnn(batch)
        # out = self.fc(out[:, -1])
        out = self.rnn(batch)
        out = self.fc(out)
        return out
