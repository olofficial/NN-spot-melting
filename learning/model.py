import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:len(pe[0])//2])
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])


class MeltSequenceTransformer(nn.Module):
    def __init__(self, num_nodes, coord_dim, d_model=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, d_model)
        self.project_coords = nn.Linear(coord_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, heads, 256, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, seq, coords):
        token = self.embedding(seq)
        spatial = self.project_coords(coords)
        x = self.pos_encoder(token + spatial)
        z = self.transformer(x).mean(dim=0)
        return self.output_layer(z).squeeze()
