import torch
import torch.nn as nn
import numpy as np

# Positional Encoding to learn order
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Encoder for Hourly Series
class HourlyEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        # x = x.permute(1, 0, 2)
        return self.transformer_encoder(x)

# Transformer Decoder for 5-min Series
class FiveMinDecoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers):
        # n_heads - more diverse attention patterns. Too many → smaller head size → may underfit
        # n_layers - More capacity for complex patterns. Too many → slower, risk of overfitting
        # n_heads must divide d_model (e.g., if d_model=128, valid head counts: 2, 4, 8, 16)
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

    def forward(self, x, memory):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        # x = x.permute(1, 0, 2)
        return self.transformer_decoder(x, memory)

# Multi-task output head
class MultiTaskHead(nn.Module):
    def __init__(self, d_model, output_dim_5min, output_dim_hourly):
        super().__init__()
        self.output_5min = nn.Linear(d_model, output_dim_5min)
        self.output_hourly = nn.Linear(d_model, output_dim_hourly)

    def forward(self, decoder_out):
        output_5min = self.output_5min(decoder_out)
        output_hourly = self.output_hourly(decoder_out[:, 0])
        return output_5min, output_hourly

# Full Model
class MultiResTrafficTransformer(nn.Module):
    def __init__(self, input_dim_hourly, input_dim_5min, d_model=256, n_heads=8, n_layers=4):
        super().__init__()
        self.encoder = HourlyEncoder(input_dim_hourly, d_model, n_heads, n_layers)
        self.decoder = FiveMinDecoder(input_dim_5min, d_model, n_heads, n_layers)
        self.head = MultiTaskHead(d_model, output_dim_5min=1, output_dim_hourly=1)

    def forward(self, hourly_seq, fivemin_seq):
        memory = self.encoder(hourly_seq)
        decoder_out = self.decoder(fivemin_seq, memory)
        return self.head(decoder_out)


