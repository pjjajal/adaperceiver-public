from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.attention.flex_attention as flex_attn
import torch.nn.functional as F
from timm.layers import get_act_layer, get_norm_layer

from modules.layers.attention import Attention, FlexAttention
from modules.layers.attn_masks import create_block_mask, create_causal_mask
from modules.layers.block import Block
from modules.layers.linear import MatLinear
from modules.layers.ffn import Mlp, SwiGLU
from modules.layers.latents import ProcessLatents
from modules.layers.rope import precompute_freqs_cis


@dataclass
class AdaPercevierConfig:
    embed_dim: int
    num_heads: int
    depth: int
    max_latent_tokens: int = 256
    max_latent_tokens_mult: int = 1
    rope_theta: int = 10000
    ffn_ratio: float = 4.0
    qkv_bias: bool = True
    proj_bias: bool = True
    ls_init_values: float = 1e-6
    proj_drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.0
    head_drop: float = 0.0
    act_layer: str = "gelu"
    norm_layer: str = "layernorm"
    ffn_layer: str = "mlp"
    attn_layer: str = "flex"
    process_token_init: Literal["learned", "norm_randn"] = "learned"
    block_mask: Literal["causal", "block", "none"] = "block"
    mask_token_grans: list[int] = None
    in_dim: int = None  # only needed if input adapter dim is different from embed_dim
    out_dim: int = None  # only needed if output adapter dim is different from embed_dim


def get_ffn_layer(ffn_layer: str):
    if ffn_layer == "mlp":
        return Mlp
    elif ffn_layer == "swiglu":
        return SwiGLU
    else:
        raise ValueError(f"Unknown ffn layer: {ffn_layer}")


def get_attn_layer(attn_layer: str):
    if attn_layer == "full":
        return Attention
    elif attn_layer == "flex":
        return FlexAttention
    else:
        raise ValueError(f"Unknown attn layer: {attn_layer}")


class AdaPerceiver(nn.Module):
    def __init__(self, config: AdaPercevierConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.depth = config.depth
        self.max_latent_tokens = config.max_latent_tokens
        self.max_latent_tokens_mult = config.max_latent_tokens_mult
        self.rope_theta = config.rope_theta
        self.block_mask_type = config.block_mask

        # if config.mask_token_grans:
        #     assert (
        #         config.mask_token_grans[-1] == config.max_latent_tokens
        #     ), "The last element of mask_token_grans must match max_latent_tokens."
        self.mask_token_grans = list(config.mask_token_grans) or []

        self.block_mask = None  # set the block mask to None initially

        act_layer = get_act_layer(config.act_layer) or nn.GELU
        ffn_layer = get_ffn_layer(config.ffn_layer) or Mlp
        attn_layer = get_attn_layer(config.attn_layer) or FlexAttention
        norm_layer = get_norm_layer(config.norm_layer) or nn.LayerNorm

        self.freq_cis = precompute_freqs_cis(
            dim=self.embed_dim // self.num_heads,
            end=self.max_latent_tokens * self.max_latent_tokens_mult,
            theta=self.rope_theta,
        )

        self.process_latents = ProcessLatents(
            dim=self.embed_dim, token_init=config.process_token_init
        )

        dpr = [
            x.item() for x in torch.linspace(0, config.drop_path, config.depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    ffn_ratio=config.ffn_ratio,
                    qkv_bias=config.qkv_bias,
                    proj_bias=config.proj_bias,
                    attn_drop=config.attn_drop,
                    proj_drop=config.proj_drop,
                    drop_path=dpr[i],
                    init_values=config.ls_init_values,
                    act_layer=act_layer,
                    ffn_layer=ffn_layer,
                    norm_layer=norm_layer,
                    attn_layer=attn_layer,
                )
                for i in range(config.depth)
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

    def compute_freq_cis(self, num_tokens, device, freq_cis=None):
        freq_cis = freq_cis if freq_cis is not None else self.freq_cis
        freq_cis = freq_cis.to(device)
        freq_cis = freq_cis[:num_tokens]
        return freq_cis

    def compute_block_mask(self, num_tokens, device, mask_type=None, token_grans=[]):
        mask_type = mask_type or self.block_mask_type
        assert mask_type in [
            "causal",
            "block",
            "bidir",
            "none",
        ], f"Invalid mask type: {mask_type}. Must be one of 'causal', 'block', or 'none'."
        if mask_type == "none":
            return None

        # Check if the block mask is already computed and matches the number of tokens
        # this is solely for the case where the block mask is not None
        if (
            (self.block_mask is not None)
            and (self.block_mask.seq_lengths[0] == num_tokens)
            and mask_type != "bidir"
        ):
            return self.block_mask

        if mask_type == "causal":
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
            mask = create_block_mask(token_granularities=token_grans)
            block_mask = flex_attn.create_block_mask(
                mask,
                B=None,
                H=None,
                Q_LEN=num_tokens,
                KV_LEN=num_tokens,
                device=device,
            )
        elif mask_type == "bidir":
            num_tokens_tensor = torch.tensor([num_tokens], device=device)
            pass
        self.block_mask = block_mask
        return block_mask
