from typing import Literal, Tuple

import torch
import torch.nn as nn

from .ffn import Mlp, SwiGLU


class ProcessLatents(nn.Module):
    def __init__(
        self,
        dim: int,
        token_init: Literal["learned", "norm_randn"] = "learned",
    ):
        super().__init__()
        self.dim = dim
        self.token_init = token_init

        if token_init == "learned":
            self.token = nn.Parameter(torch.zeros(dim))
            nn.init.normal_(self.token, std=1e-6)
        elif token_init == "norm_randn":
            self.token = nn.Parameter(torch.randn(dim) * (dim**-0.5))

    def forward(self, shape: Tuple[int, int]) -> torch.Tensor:
        token = self.token.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, dim)
        return token.expand(*shape, -1)


class OutputLatents(nn.Module):
    def __init__(
        self,
        dim: int,
        num_tokens: int,
        token_init: Literal["learned", "norm_randn", "single_learned"] = "learned",
        pos_embed: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.token_init = token_init

        if token_init == "learned":
            self.token = nn.Parameter(torch.zeros(num_tokens, dim))
            nn.init.normal_(self.token, std=1e-6)
        elif token_init == "norm_randn":
            self.token = nn.Parameter(torch.randn(num_tokens, dim) * (dim**-0.5))
        elif token_init == "single_learned":
            self.token = nn.Parameter(torch.zeros(1, dim))
            nn.init.normal_(self.token, std=1e-6)

        # positional embedding with ViT style init.
        self.pos_embed = (
            nn.Parameter(torch.randn(num_tokens, dim) * 0.02) if pos_embed else None
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        token = self.token.unsqueeze(0)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.unsqueeze(0)
            return token.expand(batch_size, self.num_tokens, -1) + pos_embed.expand(
                batch_size, -1, -1
            )
        return token.expand(batch_size, self.num_tokens, -1)


class PatchEmbedOutputLatents(nn.Module):
    """
    This module constructs output latents from some input features.
    This is useful when we want the input tokens to match the output tokens,
    i.e., when we want shape behaviour similar to a traditional ViT.

    Effectively, we tie the output token shape to the post-embedding tensor shape.
    The MLP is added since we may not want the output tokens (the query tokens for latent readout), to
    have the same representation as the input tokens.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.proj(x)
        return x
