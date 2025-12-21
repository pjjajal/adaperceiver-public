from typing import Literal

import torch
import torch.nn as nn
import torch.nn.attention.flex_attention as flex_attn
from timm.layers import get_act_layer, get_norm_layer

from modules.layers.attention import FlexAttention
from modules.layers.attn_masks import (
    create_block_mask,
    create_causal_mask,
)
from modules.layers.block import Block
from modules.layers.linear import MatLinear
from modules.layers.ffn import Mlp
from modules.layers.latents import ProcessLatents
from modules.layers.rope import precompute_freqs_cis
from modules.networks.adaperceiver import get_attn_layer, get_ffn_layer

class AdaPerceiver(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        max_latent_tokens: int = 256,
        rope_theta: int = 10000,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ls_init_values: float = 1e-6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        head_drop: float = 0.0,
        act_layer: str = "gelu",
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        attn_layer: str = "flex",
        process_token_init: Literal["learned", "norm_randn"] = "learned",
        block_mask: Literal["causal", "block", "none"] = "block",
        mask_token_grans: list[int] = None,
        mat_dims: list[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.max_latent_tokens = max_latent_tokens
        self.rope_theta = rope_theta
        self.ffn_ratio = ffn_ratio
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.ls_init_values = ls_init_values
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.head_drop = head_drop
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.ffn_layer = ffn_layer
        self.attn_layer = attn_layer
        self.process_token_init = process_token_init
        self.block_mask_type = block_mask

        # Adaptivity Parameters
        self._orig_mask_token_grans = list(mask_token_grans) or []
        self.mask_token_grans = list(mask_token_grans) or []
        self.mat_dims = list(mat_dims) or []

        self.block_mask = None  # set the block mask to None initially

        act_layer = get_act_layer(self.act_layer) or nn.GELU
        ffn_layer = get_ffn_layer(self.ffn_layer) or Mlp
        attn_layer = get_attn_layer(self.attn_layer) or FlexAttention
        norm_layer = get_norm_layer(self.norm_layer) or nn.LayerNorm

        # training length should be max_latent_tokens
        self.train_len = self.max_latent_tokens

        self.freq_cis = precompute_freqs_cis(
            dim=self.embed_dim // self.num_heads,
            end=self.max_latent_tokens,
            theta=self.rope_theta,
        )

        self.process_latents = ProcessLatents(
            dim=self.embed_dim, token_init=self.process_token_init
        )

        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path, self.depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ffn_ratio=self.ffn_ratio,
                    qkv_bias=self.qkv_bias,
                    proj_bias=self.proj_bias,
                    attn_drop=self.attn_drop,
                    proj_drop=self.proj_drop,
                    drop_path=dpr[i],
                    init_values=self.ls_init_values,
                    act_layer=act_layer,
                    ffn_layer=ffn_layer,
                    norm_layer=norm_layer,
                    attn_layer=attn_layer,
                )
                for i in range(self.depth)
            ]
        )
        self.init_weights()

    def init_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.trunc_normal_(mod.weight, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            if isinstance(mod, MatLinear):
                nn.init.trunc_normal_(mod.weight, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.LayerNorm):
                nn.init.ones_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def compute_freq_cis(self, num_tokens, device):
        freq_cis = self.freq_cis
        if num_tokens >= self.train_len:
            freq_cis = precompute_freqs_cis(
                dim=self.embed_dim // self.num_heads,
                end=num_tokens,
                train_len=self.train_len,
                theta=self.rope_theta,
            )
        freq_cis = freq_cis.to(device)
        freq_cis = freq_cis[:num_tokens]
        return freq_cis

    def compute_block_mask(self, num_tokens, device, mask_type=None, token_grans=None):
        mask_type = mask_type or self.block_mask_type
        assert mask_type in [
            "causal",
            "block",
            "none",
        ], f"Invalid mask type: {mask_type}. Must be one of 'causal', 'block', or 'none'."
        if mask_type == "none":
            return None

        # Check if the block mask is already computed and matches the number of tokens
        # this is solely for the case where the block mask is not None
        if (
            (self.block_mask is not None)
            and (self.block_mask.seq_lengths[0] == num_tokens)
            and (token_grans == self.mask_token_grans)
        ):
            return self.block_mask

        if mask_type == "causal":
            # Same-length causal mask for all heads/batches.
            mask = create_causal_mask()
            block_mask = flex_attn.create_block_mask(
                mask,
                B=None,
                H=None,
                Q_LEN=num_tokens,
                KV_LEN=num_tokens,
                device=device,
            )
        elif mask_type == "block":
            assert (
                len(token_grans) > 0
            ), "token_grans must be provided for block masking"
            # Keep only granularity levels that fit the current sequence length.
            token_grans = [gr for gr in token_grans if gr <= num_tokens]
            # Ensure the final granularity always covers the full length.
            if not token_grans or num_tokens > token_grans[-1]:
                token_grans = list(token_grans) + [num_tokens]
            mask = create_block_mask(token_granularities=token_grans)
            block_mask = flex_attn.create_block_mask(
                mask,
                B=None,
                H=None,
                Q_LEN=num_tokens,
                KV_LEN=num_tokens,
                device=device,
            )
        self.block_mask = block_mask
        if mask_type == "block":
            self.mask_token_grans = list(token_grans)
        return block_mask
