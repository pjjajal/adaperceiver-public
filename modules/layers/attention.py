import torch
import torch.nn as nn
import torch.nn.attention.flex_attention as flex_attn
import torch.nn.functional as F

from .linear import MatLinear
from .rope import apply_rotary_emb
from .attn_masks import generate_alibi_bias


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.attn_drop = attn_drop

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        # output projection
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.score_mod = generate_alibi_bias(num_heads)
        self.score_mod = None

    def forward(self, x, freqs_cis=None, **kwargs):
        B, N, C = x.shape

        # Compute V
        v_head_dim = C // self.num_heads
        v = (
            self.v(x).reshape(B, N, self.num_heads, v_head_dim).permute(0, 2, 1, 3)
        )  # [B, H, N, D]
        # Compute QK
        head_dim = C // self.num_heads
        q = self.q.forward(x).reshape(B, N, self.num_heads, head_dim)
        k = self.k.forward(x).reshape(B, N, self.num_heads, head_dim)
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)  # [B, H, N, D]

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop
        )  # [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, D]
        out = self.proj(out)
        return self.proj_drop(out)


class FlexAttention(Attention):
    def forward(
        self,
        x,
        freqs_cis=None,
        mask: flex_attn.BlockMask = None,
        **kwargs,
    ):
        B, N, C = x.shape

        # Compute V
        v_head_dim = C // self.num_heads
        v = (
            self.v(x).reshape(B, N, self.num_heads, v_head_dim).permute(0, 2, 1, 3)
        )  # [B, H, N, D]
        # Compute QK
        head_dim = C // self.num_heads
        q = self.q.forward(x).reshape(B, N, self.num_heads, head_dim)
        k = self.k.forward(x).reshape(B, N, self.num_heads, head_dim)
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)  # [B, H, N, D]

        out = flex_attn.flex_attention(
            q, k, v, block_mask=mask, score_mod=self.score_mod
        )
        # [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, D]
        out = self.proj(out)
        return self.proj_drop(out)


class CrossAttention(Attention):
    def forward(self, sink, src, freq_cis_q=None, freq_cis_k=None, **kwargs):
        B, N, C = sink.shape
        B, M, D = src.shape

        # Compute V
        v_head_dim = D // self.num_heads
        v = self.v(src).reshape(B, M, self.num_heads, v_head_dim).permute(0, 2, 1, 3)

        # Compute QK
        head_dim = C // self.num_heads
        q = self.q(sink).reshape(B, N, self.num_heads, head_dim)
        k = self.k(src).reshape(B, M, self.num_heads, head_dim)
        if freq_cis_q is not None:
            q = apply_rotary_emb(q, freq_cis_q)
        if freq_cis_k is not None:
            k = apply_rotary_emb(k, freq_cis_k)

        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)  # [B, H, M, D]

        # for CA we use the sink as the Q and src as KV
        out = F.scaled_dot_product_attention(
        q, k, v, dropout_p=self.attn_drop
        )  # [B, H, N, D]
        # out = flex_attn.flex_attention(q, k, v, score_mod=self.score_mod)

        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, D]
        out = self.proj(out)

        return self.proj_drop(out)
