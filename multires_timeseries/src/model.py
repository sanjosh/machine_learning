import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding

# Start with dropout=0.2 for all of the above. Increase only if overfitting persists.
DROPOUT = 0.1


class RotarySelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, D = x.size()
        qkv = self.qkv(x)  # (B, T, 3 * D)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to (B, heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Original shape: (B, n_heads, T, head_dim)
        B, H, T, D = q.shape

        # Merge heads with batch
        q = q.permute(0, 2, 1, 3).reshape(B * H, T, D)
        k = k.permute(0, 2, 1, 3).reshape(B * H, T, D)

        # Apply rotary embeddings
        q, k = self.rotary_emb(q, k)

        # Reshape back
        q = q.view(B, T, H, D).permute(0, 2, 1, 3)
        k = k.view(B, T, H, D).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, heads, T, head_dim)

        out = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

class RotaryTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                RotarySelfAttention(d_model, n_heads),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model)
            ) for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return x.permute(1, 0, 2)  # for transformer decoder compatibility


class RotaryCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, query, key, value):
        # Shapes
        B, Tq, _ = query.size()
        Tk = key.size(1)
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(query).view(B, Tq, H, D).transpose(1, 2)
        k = self.k_proj(key).view(B, Tk, H, D).transpose(1, 2)
        v = self.v_proj(value).view(B, Tk, H, D).transpose(1, 2)

        # Apply rotary
        q_ = q.permute(0, 2, 1, 3).reshape(B * H, Tq, D)
        k_ = k.permute(0, 2, 1, 3).reshape(B * H, Tk, D)
        q_, k_ = self.rotary_emb(q_, k_)
        q = q_.view(B, Tq, H, D).permute(0, 2, 1, 3)
        k = k_.view(B, Tk, H, D).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        out = attn_output.transpose(1, 2).contiguous().view(B, Tq, self.n_heads * self.head_dim)
        return self.out_proj(out)

class RotaryTransformerDecoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": RotaryCrossAttention(d_model, n_heads),  # Self-attn on decoder input
                "cross_attn": RotaryCrossAttention(d_model, n_heads), # Cross-attn with encoder memory
                "ffn": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.ReLU(),
                    nn.Linear(4 * d_model, d_model)
                ),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
                "norm3": nn.LayerNorm(d_model)
            })
            for _ in range(n_layers)
        ])

    def forward(self, tgt, memory):
        x = self.input_proj(tgt)
        for layer in self.layers:
            # Self-attention (within decoder)
            x2 = layer["self_attn"](x, x, x)
            x = layer["norm1"](x + x2)

            # Cross-attention (decoder attends to encoder memory)
            x2 = layer["cross_attn"](x, memory.permute(1, 0, 2), memory.permute(1, 0, 2))
            x = layer["norm2"](x + x2)

            # Feedforward
            x2 = layer["ffn"](x)
            x = layer["norm3"](x + x2)

        return x.permute(1, 0, 2)  # (seq_len, batch, dim) → (batch, seq_len, dim)



class LearnedPositionalEncoding(nn.Module):
    """
    Automatically adjust based on training data.
    Let model learn nonlinear importance of positions.
    Useful when sinusoidal encoding is too rigid.
    useful when have fixed max sequence lengths (like 72 steps for 5-min data or 168 for hourly).
    """
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pos_embed(positions)

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
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=512, dropout=DROPOUT, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        # x = x.permute(1, 0, 2)
        return self.transformer_encoder(x)

# Transformer Decoder for 5-min Series
class FiveMinDecoder(nn.Module):
    # Larger decoder can overfit to transient spikes or local noise,
    def __init__(self, input_dim, d_model, n_heads, n_layers):
        # n_heads - more diverse attention patterns. Too many → smaller head size → may underfit
        # n_layers - More capacity for complex patterns. Too many → slower, risk of overfitting
        # n_heads must divide d_model (e.g., if d_model=128, valid head counts: 2, 4, 8, 16)
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=512, dropout=DROPOUT, batch_first=True)
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

    def __init__(self, input_dim_hourly, input_dim_5min, d_model,
                 n_heads_encoder, n_layers_encoder,
                 n_heads_decoder, n_layers_decoder):
        super().__init__()
        self.encoder = HourlyEncoder(input_dim_hourly, d_model, n_heads_encoder, n_layers_encoder)
        # self.encoder = RotaryTransformerEncoder(input_dim_hourly, d_model, n_heads_encoder, n_layers_encoder)
        self.decoder = FiveMinDecoder(input_dim_5min, d_model, n_heads_decoder, n_layers_decoder)
        # self.decoder = RotaryTransformerDecoder(input_dim_5min, d_model, n_heads_decoder, n_layers_decoder)
        self.head = MultiTaskHead(d_model, output_dim_5min=1, output_dim_hourly=1)

    def forward(self, hourly_seq, fivemin_seq):
        memory = self.encoder(hourly_seq)
        decoder_out = self.decoder(fivemin_seq, memory)
        return self.head(decoder_out)


