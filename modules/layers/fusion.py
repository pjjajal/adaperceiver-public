from typing import Optional

import torch.nn as nn

from .attention import CrossAttention


"""
NOTE: I added this categorized in fusion but it does not actually fuse anything.
It is a cross attention layer that applies a norm to its 
"""


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        qkv_bias=True,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.src_norm = norm_layer(embed_dim)
        self.snk_norm = norm_layer(embed_dim)
        self.cattn = CrossAttention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, sink, src, freq_cis_q=None, freq_cis_k=None):
        return self.cattn(
            sink=self.snk_norm(sink),
            src=self.src_norm(src),
            freq_cis_q=freq_cis_q,
            freq_cis_k=freq_cis_k,
        )
