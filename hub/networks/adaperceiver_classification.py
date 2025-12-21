from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.attention.flex_attention as flex_attn
from huggingface_hub import PyTorchModelHubMixin
from timm.layers import get_act_layer, get_norm_layer

from modules.layers.adapters import OutputAdapter, PatchEmbedAdapter, ConvAdapter
from modules.layers.ffn import Mlp
from modules.layers.latents import OutputLatents
from modules.layers.fusion import CrossAttentionFusion
from modules.networks.adaperceiver import get_ffn_layer
from .adaperceiver import AdaPerceiver
from modules.layers.rope import precompute_freqs_cis


# This is the AdaPerceiver model configured for classification
class ClassificationAdaPerceiver(
    AdaPerceiver,
    PyTorchModelHubMixin,
    repo_url="https://github.com/pjjajal/adaperceiver-public",
    paper_url="https://arxiv.org/abs/2511.18105",
):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
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
        # Classification-specific args
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        num_classes: int = 1000,
        use_embed_ffn: bool = False,
        use_output_ffn: bool = False,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            max_latent_tokens=max_latent_tokens,
            rope_theta=rope_theta,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ls_init_values=ls_init_values,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            head_drop=head_drop,
            act_layer=act_layer,
            norm_layer=norm_layer,
            ffn_layer=ffn_layer,
            attn_layer=attn_layer,
            process_token_init=process_token_init,
            block_mask=block_mask,
            mask_token_grans=mask_token_grans,
            mat_dims=mat_dims,
        )
        # Classification-specific initializations
        self.num_classes = num_classes
        self.use_embed_ffn = use_embed_ffn
        self.use_output_ffn = use_output_ffn
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size

        act_layer = get_act_layer(self.act_layer) or nn.GELU
        ffn_layer = get_ffn_layer(self.ffn_layer) or Mlp
        norm_layer = get_norm_layer(self.norm_layer) or nn.LayerNorm

        self.patch_embed = PatchEmbedAdapter(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            use_ffn=self.use_embed_ffn,
            ffn_layer=ffn_layer,
            ffn_ratio=self.ffn_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer if self.use_embed_ffn else None,
        )

        self.output_latents = OutputLatents(
            dim=self.embed_dim, num_tokens=1, token_init="learned"
        )
        self.output_adapter = OutputAdapter(
            in_features=self.embed_dim,
            out_features=self.num_classes,
            use_ffn=self.use_output_ffn,
            ffn_layer=ffn_layer,
            ffn_ratio=self.ffn_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer if self.use_output_ffn else None,
            drop=self.head_drop,
        )

        self.read_head = CrossAttentionFusion(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            norm_layer=norm_layer,
        )
        self.write_head = CrossAttentionFusion(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            norm_layer=norm_layer,
        )

        self.init_weights()

    def forward_intermediates(
        self,
        process_latents: torch.Tensor,
        freq_cis: torch.Tensor,
        block_mask: flex_attn.BlockMask = None,
        mat_dim=None,
    ):
        intermediates = []
        for block in self.blocks:
            process_latents = block(
                x=process_latents,
                freq_cis=freq_cis,
                mat_dim=mat_dim,
                block_mask=block_mask,
            )
            intermediates.append(process_latents)
        return process_latents, intermediates

    def forward_blocks(
        self,
        process_latents: torch.Tensor,
        freq_cis: torch.Tensor,
        block_mask: flex_attn.BlockMask = None,
        mat_dim=None,
        depth=None,
    ):
        for i, block in enumerate(self.blocks):
            process_latents = block(
                x=process_latents,
                freq_cis=freq_cis,
                mat_dim=mat_dim,
                block_mask=block_mask,
            )
            # If depth is specified, we stop at the specified depth.
            # This is adds early-exit functionality.
            if (depth is not None) and (i == depth - 1):
                break
        return process_latents

    def forward_block_conf(
        self,
        process_latents: torch.Tensor,
        output_latents: torch.Tensor,
        freq_cis: torch.Tensor,
        block_mask: flex_attn.BlockMask = None,
        mat_dim=None,
        depth_tau=None,
    ):
        for i, block in enumerate(self.blocks):
            process_latents = block(
                x=process_latents,
                freq_cis=freq_cis,
                mat_dim=mat_dim,
                block_mask=block_mask,
            )
            if depth_tau is not None:
                # Compute the confidence of the current output latents
                temp_writeout = self.write(
                    process_latents=process_latents,
                    output_latents=output_latents,
                    freq_cis=freq_cis,
                )
                temp_output = self.output_head(temp_writeout)
                probs = F.softmax(temp_output, dim=-1)
                conf, _ = torch.max(probs, dim=-1)  # (B,)
                avg_conf = torch.mean(conf).item()
                if avg_conf >= depth_tau:
                    break
        return process_latents

    def read(
        self,
        patches: torch.Tensor,
        process_latents: torch.Tensor,
        freq_cis: torch.Tensor,
    ):
        process_latents = process_latents + self.read_head(
            sink=process_latents,
            src=patches,
            freq_cis_q=freq_cis,  # apply the RoPE frequencies to the query
            freq_cis_k=None,  # no RoPE on key since they already have positional information
        )
        return process_latents

    def write(
        self,
        process_latents: torch.Tensor,
        output_latents: torch.Tensor,
        freq_cis: torch.Tensor,
    ):
        output_latents = output_latents + self.write_head(
            sink=output_latents,  # NOTE: output_latents is the sink.
            src=process_latents,
            freq_cis_q=None,  # no RoPE on query since they already have positional information
            freq_cis_k=freq_cis,  # NOTE: # apply the RoPE frequencies to the keys
        )
        return output_latents

    def output_head(self, output_latents: torch.Tensor):
        """
        This is the output head that converts the output latents to the final output.
        It is used in the forward pass to compute the final output.
        """
        return self.output_adapter(output_latents)

    def forward(
        self,
        x: torch.Tensor,
        num_tokens=None,
        mat_dim=None,
        depth=None,
        depth_tau=None,
        token_grans=None,
        **kwargs
    ):
        B = x.shape[0]
        device = x.device

        # Creates the process and output latents
        N = num_tokens if num_tokens else self.max_latent_tokens  # Number of tokens
        process_latents = self.process_latents((B, N))
        output_latents = self.output_latents(B)

        # Compute the RoPE frequencies
        freq_cis = self.compute_freq_cis(N, device)

        # Compute block mask
        block_mask = self.compute_block_mask(
            num_tokens=N,
            device=device,
            mask_type=self.block_mask_type,
            token_grans=self.mask_token_grans if token_grans is None else token_grans,
        )

        # Patch Embedding
        patches = self.patch_embed(x)

        # Read inputs into process latents
        process_latents = self.read(patches, process_latents, freq_cis)

        # This is the branch for early-exit with confidence.
        if depth_tau is not None:
            process_latents = self.forward_block_conf(
                process_latents=process_latents,
                output_latents=output_latents,
                freq_cis=freq_cis,
                block_mask=block_mask,
                mat_dim=mat_dim,
                depth_tau=depth_tau,
            )
        else:
            process_latents = self.forward_blocks(
                process_latents=process_latents,
                freq_cis=freq_cis,
                block_mask=block_mask,
                mat_dim=mat_dim,
                depth=depth,
            )

        # This is the final writeout step.
        final_writeout = self.write(
            process_latents=process_latents,
            output_latents=output_latents,
            freq_cis=freq_cis,
        )
        output = self.output_head(final_writeout)

        return output


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import ast

    cfg = OmegaConf.load("IN-1k-ft_config.yaml")
    cfg = cfg.model

    # Convert *once* to plain Python (dict/list/str/int/float/bool/None)
    cfg_py = OmegaConf.to_container(cfg, resolve=True)

    encoder_cfg = cfg_py["encoder_config"]

    # If ffn_ratio can be something like "[4, 8]" stored as a string, parse it safely:
    if isinstance(encoder_cfg.get("ffn_ratio"), str):
        encoder_cfg["ffn_ratio"] = ast.literal_eval(encoder_cfg["ffn_ratio"])

    model = ClassificationAdaPerceiver(
        embed_dim=encoder_cfg["embed_dim"],
        num_heads=encoder_cfg["num_heads"],
        depth=encoder_cfg["depth"],
        max_latent_tokens=encoder_cfg["max_latent_tokens"],
        rope_theta=encoder_cfg["rope_theta"],
        ffn_ratio=encoder_cfg["ffn_ratio"],
        qkv_bias=encoder_cfg["qkv_bias"],
        proj_bias=encoder_cfg["proj_bias"],
        ls_init_values=encoder_cfg["ls_init_values"],
        proj_drop=encoder_cfg["proj_drop"],
        attn_drop=encoder_cfg["attn_drop"],
        drop_path=encoder_cfg["drop_path"],
        head_drop=encoder_cfg["head_drop"],
        act_layer=encoder_cfg["act_layer"],
        norm_layer=encoder_cfg["norm_layer"],
        ffn_layer=encoder_cfg["ffn_layer"],
        attn_layer=encoder_cfg["attn_layer"],
        process_token_init=encoder_cfg["process_token_init"],
        block_mask=cfg_py["mask_type"],
        mask_token_grans=cfg_py["token_grans"],
        mat_dims=cfg_py["mat_dims"],
        img_size=encoder_cfg["img_size"],
        in_channels=encoder_cfg["in_channels"],
        patch_size=encoder_cfg["patch_size"],
        num_classes=encoder_cfg["num_classes"],
        use_embed_ffn=encoder_cfg["use_embed_ffn"],
        use_output_ffn=encoder_cfg["use_output_ffn"],
    )

    state_dict = torch.load("in_1k_ft_ema_token+depth+mat.pth", map_location="cpu")
    state_dict = {
        key.replace("module.", ""): val for key, val in state_dict.items()
    }  # this handles the cause
    state_dict = {key.replace("_orig_mod.", ""): val for key, val in state_dict.items()}
    state_dict["patch_embed.pos_embed"] = state_dict["patch_embed.pos_embed"].unsqueeze(
        0
    )
    state_dict.pop("n_averaged", None)
    state_dict.pop("ema_n_averaged", None)
    model.load_state_dict(state_dict, strict=True)

    model.save_pretrained("test-adaperceiver-classification-model")

    model = ClassificationAdaPerceiver.from_pretrained(
        "test-adaperceiver-classification-model"
    )
    print("Model loaded from hub successfully!")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input, num_tokens=256, mat_dim=128, depth=12)
    print("Output shape:", output.shape)
