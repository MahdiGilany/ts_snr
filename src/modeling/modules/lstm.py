import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

import numpy as np


class SimpleLSTM(nn.Module):
    def __init__(
        self,
        seq_len: int = 12,
        input_dim=256,
        output_dim=256,
        hidden_dim=1024,
        num_layers=2,
        dropout=0.5
        ):
        super(SimpleLSTM, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim + 1
        self.output_dim = output_dim + 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim+1, hidden_dim, num_layers, proj_size=output_dim+1, dropout=dropout, batch_first=True)

    def forward(self, X):
        assert X.shape[-1]==1, "X should be univariate for now"
        # X.shape # (batch_size, seq_len, input_dim, 1)
        return self.lstm(X.squeeze(-1))[:, -1, :].unsqueeze(-1) # before indexing (batch_size, seq_len, output_dim) -> after (batch_size, output_dim, 1)
