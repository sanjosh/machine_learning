import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Rotary 
Encodes both absolute and relative positions
No separate positional embeddings needed
Works better for longer sequences
Embedding stays clean; position enters via attention


pip install rotary-embedding-torch

from rotary_embedding_torch import RotaryEmbedding

rotary_emb = RotaryEmbedding(dim)
q, k = rotary_emb.rotate_queries_and_keys(q, k)
"""

# ---- Rotary Positional Embedding ----
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)  # (T, D)
        return emb[None, :, :]  # (1, T, D)

def apply_rotary_pos_emb(x, rotary_emb):
    x1, x2 = x[..., ::2], x[..., 1::2]
    sin, cos = rotary_emb[..., ::2], rotary_emb[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# ---- Rotary Self-Attention ----
class RotarySelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, _ = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        B, H, T, D = q.shape

        q = q.reshape(B * H, T, D)
        k = k.reshape(B * H, T, D)

        rotary_emb_q = self.rotary_emb(q)
        rotary_emb_k = self.rotary_emb(k)

        q = apply_rotary_pos_emb(q, rotary_emb_q)
        k = apply_rotary_pos_emb(k, rotary_emb_k)

        q = q.view(B, H, T, D)
        k = k.view(B, H, T, D)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        out = attn_output.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.out_proj(out)

# ---- Rotary Cross-Attention ----
class RotaryCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, query, key, value):
        B, Tq, _ = query.size()
        Tk = key.size(1)

        q = self.q_proj(query).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        B, H, Tq, D = q.shape
        _, _, Tk, _ = k.shape

        q = q.reshape(B * H, Tq, D)
        k = k.reshape(B * H, Tk, D)

        rotary_emb_q = self.rotary_emb(q)
        rotary_emb_k = self.rotary_emb(k)

        q = apply_rotary_pos_emb(q, rotary_emb_q)
        k = apply_rotary_pos_emb(k, rotary_emb_k)

        q = q.view(B, H, Tq, D)
        k = k.view(B, H, Tk, D)
        v = v.view(B, H, Tk, D)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        out = attn_output.transpose(1, 2).contiguous().view(B, Tq, self.n_heads * self.head_dim)
        return self.out_proj(out)

# ---- Rotary Transformer Encoder ----
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
        return x.permute(1, 0, 2)  # for decoder (S, B, D)

# ---- Rotary Transformer Decoder ----
class RotaryTransformerDecoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": RotaryCrossAttention(d_model, n_heads),
                "cross_attn": RotaryCrossAttention(d_model, n_heads),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.ReLU(),
                    nn.Linear(4 * d_model, d_model)
                ),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
                "norm3": nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])

    def forward(self, tgt, memory):
        x = self.input_proj(tgt)
        memory = memory.permute(1, 0, 2)
        for layer in self.layers:
            x2 = layer["self_attn"](x, x, x)
            x = layer["norm1"](x + x2)

            x2 = layer["cross_attn"](x, memory, memory)
            x = layer["norm2"](x + x2)

            x2 = layer["ffn"](x)
            x = layer["norm3"](x + x2)
        return x

# ---- Full MultiResTrafficTransformer ----
class MultiResTrafficTransformer(nn.Module):
    def __init__(self, input_dim_hourly, input_dim_5min, d_model, n_heads_encoder, n_layers_encoder, n_heads_decoder, n_layers_decoder):
        super().__init__()
        self.encoder = RotaryTransformerEncoder(input_dim_hourly, d_model, n_heads_encoder, n_layers_encoder)
        self.decoder = RotaryTransformerDecoder(input_dim_5min, d_model, n_heads_decoder, n_layers_decoder)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, hourly, fivemin):
        memory = self.encoder(hourly)  # (Th, B, D)
        decoded = self.decoder(fivemin, memory)  # (B, Tf, D)
        pred_5min = self.output_layer(decoded).squeeze(-1)

        # also predict from encoder memory (mean pooled)
        hourly_summary = memory.mean(dim=0)  # (B, D)
        pred_hourly = self.output_layer(hourly_summary).squeeze(-1)

        return pred_5min, pred_hourly

