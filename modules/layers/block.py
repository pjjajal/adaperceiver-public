from typing import Tuple, Literal, Optional

import torch
import torch.nn as nn
from .attention import Attention, FlexAttention
from .ffn import Mlp, SwiGLU
from timm.layers import DropPath, LayerScale


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = 1e-6,
        act_layer: nn.Module = nn.GELU,
        ffn_layer: Mlp | SwiGLU = Mlp,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_layer: Attention | FlexAttention = Attention,
    ):
        super().__init__()
        self.ffn_ratio = ffn_ratio
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        ffn_hidden_dim = int(dim * ffn_ratio)
        self.norm2 = norm_layer(dim)
        self.ffn = ffn_layer(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, freq_cis, mat_dim=None, block_mask=None):
        """
        Forward pass for a single transformer Block.

        Args:
            x (torch.Tensor):
                Input tensor of shape [batch_size, sequence_length, embed_dim].
            freq_cis (torch.Tensor):
                Precomputed rotary embedding frequencies (e.g., for RoPE).
            mat_dim (int, optional):
                Dimension used by the feed-forward network if this block
                supports "matryoshka" style (partial-dimension) layers.
            block_mask (optional):
                Attention mask to apply (e.g., block-structured or causal mask).

        Returns:
            torch.Tensor: The output tensor of the same shape as x after
            applying attention, a feed-forward network, residuals, and any
            layer-scale or drop-path transformations.
        """
        x = x + self.drop_path1(
            self.ls1(
                self.attn(
                    self.norm1(x),
                    freq_cis,
                    mask=block_mask,
                )
            )
        )

        if isinstance(mat_dim, (list, tuple)):
            ffn_mat_dim = [int(d * self.ffn_ratio) for d in mat_dim]
        else:
            ffn_mat_dim = int(mat_dim * self.ffn_ratio) if mat_dim else mat_dim
        x = x + self.drop_path2(self.ls2(self.ffn(self.norm2(x), mat_dim=ffn_mat_dim)))
        return x
