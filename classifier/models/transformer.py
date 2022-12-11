import torch.nn as nn
import math
import torch
from torch import nn, Tensor

from settings import *
from dataset import Dataset

# based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TransformerModel(nn.Module):
    def __init__(self, output_size, device):
        super().__init__()
        self.device = device
        self.output_size = output_size
        self.model_type = 'Transformer'
        d_model = Dataset.characters_count+1
        self.pos_encoder = PositionalEncoding(d_model, 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(self.rnn.hidden_size*2, self.output_size)

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, batch):
        bptt = batch.size(1)
        src = self.encoder(batch) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src_mask = generate_square_subsequent_mask(bptt).to(self.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)