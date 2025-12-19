import timm
import torch
import torch.nn as nn
from timm.layers import PatchEmbed
from timm.layers.pos_embed import resample_abs_pos_embed
from einops import rearrange

from .ffn import Mlp, SwiGLU
from .fourier import FourierFeatures
from einops import rearrange


class PatchEmbedAdapter(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels: int,
        embed_dim: int,
        use_ffn: bool = False,
        norm_layer: nn.Module = None,
        ffn_layer: Mlp | SwiGLU = Mlp,
        ffn_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        proj_drop: float = 0.0,
        use_fourier: bool = False,
        fourier_bands: int = 16,
    ):
        super().__init__()
        self.use_ffn = use_ffn
        self.use_fourier = use_fourier
        self.fourier = (
            FourierFeatures(
                dims=2,  # hardcode this as 2 because patch embed is for images.
                max_freq=img_size / 2,
                num_bands=fourier_bands,
            )
            if use_fourier
            else nn.Identity()
        )

        in_channels = in_channels + self.fourier.out_dim if use_fourier else in_channels
        self.embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            strict_img_size=False,
            flatten=False,
        )

        self.num_tokens = self.embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim) * 0.02)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        if use_ffn:
            ffn_hidden_dim = int(embed_dim * ffn_ratio)
            self.ffn = ffn_layer(
                in_features=embed_dim,
                hidden_features=ffn_hidden_dim,
                out_features=embed_dim,
                act_layer=act_layer,
                drop=proj_drop,
            )

    def forward(self, x):
        x = self.fourier(x)
        x = self.embed(x)
        B, C, H, W = x.shape
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,  # this unsqueeze might be unnecessary.
            new_size=(H, W),
            # old_size=self.embed.grid_size,
            num_prefix_tokens=0,
        )
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + pos_embed
        x = self.norm(x)
        if self.use_ffn:
            x = self.ffn(x)
        return x


class OutputAdapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_ffn: bool = False,
        norm_layer: nn.Module = None,
        ffn_layer: Mlp | SwiGLU = Mlp,
        ffn_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        self.use_ffn = use_ffn
        self.norm = norm_layer(in_features) if norm_layer else nn.Identity()
        self.drop = nn.Dropout(drop)
        if use_ffn:
            ffn_hidden_dim = int(in_features * ffn_ratio)
            self.ffn = ffn_layer(
                in_features=in_features,
                hidden_features=ffn_hidden_dim,
                out_features=out_features,
                act_layer=act_layer,
            )
        else:
            self.ffn = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.drop(x)
        x = self.ffn(x)
        return x


class ConvAdapter(nn.Module):
    def __init__(
        self,
        img_size,
        model_name: str,
        embed_dim: int,
        use_ffn=False,
        norm_layer=None,
        ffn_layer=Mlp,
        ffn_ratio=4,
        act_layer=nn.GELU,
        proj_drop=0,
        use_fourier=False,
        fourier_bands=16,
    ):
        super().__init__()
        self.embed: timm.models.convnext.ConvNeXt = timm.create_model(
            model_name, pretrained=True
        )
        # Complete arbitrary assert, just to make sure I don't mess up the model type.
        assert isinstance(
            self.embed, timm.models.convnext.ConvNeXt
        ), "The embed model must be ConvNeXt-based."

        in_channels = self.embed.num_features
        reduction = self.embed.feature_info[-1]["reduction"]

        self.use_ffn = use_ffn
        self.use_fourier = use_fourier
        self.fourier = (
            FourierFeatures(
                dims=2,  # hardcode this as 2 because patch embed is for images.
                max_freq=img_size / 2,
                num_bands=fourier_bands,
            )
            if use_fourier
            else nn.Identity()
        )

        in_channels = in_channels + self.fourier.out_dim if use_fourier else in_channels
        self.proj = nn.Linear(in_channels, embed_dim)

        self.num_tokens = int((img_size // reduction) ** 2)
        self.pos_embed = nn.Parameter(torch.randn(self.num_tokens, embed_dim) * 0.02)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        if use_ffn:
            ffn_hidden_dim = int(embed_dim * ffn_ratio)
            self.ffn = ffn_layer(
                in_features=embed_dim,
                hidden_features=ffn_hidden_dim,
                out_features=embed_dim,
                act_layer=act_layer,
                drop=proj_drop,
            )

        # Freeze the convnext parameters.
        for param in self.embed.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Embed the input image.
        with torch.no_grad():
            x = self.embed.forward_features(x)

        # Add fourier features if needed.
        x = self.fourier(x)
        B, C, H, W = x.shape
        # Proj to the desired dimension.
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj(x)

        # Add position embeddings.
        pos_embed = resample_abs_pos_embed(
            self.pos_embed.unsqueeze(0),  # this unsqueeze might be unnecessary.
            new_size=(H, W),
            # old_size=self.embed.grid_size,
            num_prefix_tokens=0,
        )
        x = x + pos_embed
        # Norm and ffn
        x = self.norm(x)
        if self.use_ffn:
            x = self.ffn(x)
        return x
