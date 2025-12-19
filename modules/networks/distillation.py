import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.attention.flex_attention as flex_attn
from timm.layers import get_act_layer, get_norm_layer

from modules.layers.adapters import OutputAdapter, PatchEmbedAdapter, ConvAdapter
from modules.layers.ffn import Mlp
from modules.layers.latents import OutputLatents, PatchEmbedOutputLatents
from modules.layers.fusion import CrossAttentionFusion

from .adaperceiver import AdaPerceiver, AdaPercevierConfig, get_ffn_layer


@dataclass
class DistillationConfig:
    img_size: int
    in_channels: int
    patch_size: int
    use_fourier: bool
    fourier_bands: int
    num_classes: int
    output_feat_dim: int
    use_embed_ffn: bool = False
    use_output_ffn: bool = False
    adapter_type: str = "patch_embed" # or conv
    conv_adapter_model: str = "hf_hub:timm/convnextv2_nano.fcmae_ft_in22k_in1k_384"


class DistillationAdaPerceiver(AdaPerceiver):
    def __init__(
        self, config: AdaPercevierConfig, distillation_config: DistillationConfig
    ):
        super().__init__(config)
        self.distillation_config = distillation_config
        self.num_classes = distillation_config.num_classes
        self.use_embed_ffn = distillation_config.use_embed_ffn
        self.use_output_ffn = distillation_config.use_output_ffn
        self.img_size = distillation_config.img_size
        self.in_channels = distillation_config.in_channels
        self.patch_size = distillation_config.patch_size
        self.use_fourier = distillation_config.use_fourier
        self.fourier_bands = distillation_config.fourier_bands
        self.output_feat_dim = distillation_config.output_feat_dim

        act_layer = get_act_layer(config.act_layer) or nn.GELU
        ffn_layer = get_ffn_layer(config.ffn_layer) or Mlp
        norm_layer = get_norm_layer(config.norm_layer) or nn.LayerNorm

        if distillation_config.adapter_type == "patch_embed":
            self.patch_embed = PatchEmbedAdapter(
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_channels=self.in_channels,
                embed_dim=self.embed_dim,
                use_ffn=self.use_embed_ffn,
                ffn_layer=ffn_layer,
                ffn_ratio=self.config.ffn_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer if self.use_embed_ffn else None,
                use_fourier=self.distillation_config.use_fourier,
                fourier_bands=self.distillation_config.fourier_bands,
            )
        elif distillation_config.adapter_type == "conv":
            self.patch_embed = ConvAdapter(
                img_size=self.img_size,
                model_name=self.distillation_config.conv_adapter_model,
                embed_dim=self.embed_dim,
                use_ffn=self.use_embed_ffn,
                ffn_layer=ffn_layer,
                ffn_ratio=self.config.ffn_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer if self.use_embed_ffn else None,
                proj_drop=self.config.proj_drop,
                use_fourier=self.distillation_config.use_fourier,
                fourier_bands=self.distillation_config.fourier_bands,
            )

        # This is for logits distillation
        self.output_latents = OutputLatents(
            dim=self.embed_dim, num_tokens=1, token_init="learned"
        )
        self.output_adapter = OutputAdapter(
            in_features=self.embed_dim,
            out_features=self.num_classes,
            use_ffn=self.use_output_ffn,
            ffn_layer=ffn_layer,
            ffn_ratio=self.config.ffn_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer if self.use_output_ffn else None,
            drop=self.config.head_drop,
        )

        # This is used for feature distillation
        # self.output_latents_feat = OutputLatents(
        #     dim=self.embed_dim,
        #     num_tokens=self.output_feats,
        #     token_init="learned",
        # )
        self.output_latents_feat = PatchEmbedOutputLatents(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
        )
        self.output_adapter_feat = OutputAdapter(
            in_features=self.embed_dim,
            out_features=self.output_feat_dim,
            use_ffn=True, # THIS IS ALWAYS USED. Similar to AM-RADIO paper.
            ffn_layer=ffn_layer,
            ffn_ratio=self.config.ffn_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=self.config.head_drop,
        )

        self.read_head = CrossAttentionFusion(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=config.qkv_bias,
            proj_bias=config.proj_bias,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            norm_layer=norm_layer,
        )
        self.write_head = CrossAttentionFusion(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=config.qkv_bias,
            proj_bias=config.proj_bias,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            norm_layer=norm_layer,
        )
        self.write_head_feat = CrossAttentionFusion(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=config.qkv_bias,
            proj_bias=config.proj_bias,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
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
            process_latents = block.forward(
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
            process_latents = block.forward(
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

    def read(
        self,
        x: torch.Tensor,
        process_latents: torch.Tensor,
        freq_cis: torch.Tensor,
    ):
        process_latents = process_latents + self.read_head.forward(
            sink=process_latents,
            src=x,
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
        output_latents = output_latents + self.write_head.forward(
            sink=output_latents,  # NOTE: output_latents is the sink.
            src=process_latents,
            freq_cis_q=None,  # no RoPE on query since they already have positional information
            freq_cis_k=freq_cis,  # NOTE: # apply the RoPE frequencies to the keys
        )
        # # output head
        # out = self.output_adapter(output_latents)
        return output_latents

    def write_feat(
        self,
        process_latents: torch.Tensor,
        output_latents: torch.Tensor,
        freq_cis: torch.Tensor,
    ):
        output_latents = output_latents + self.write_head_feat.forward(
            sink=output_latents,  # NOTE: output_latents is the sink.
            src=process_latents,
            freq_cis_q=None,  # no RoPE on query since they already have positional information
            freq_cis_k=freq_cis,  # NOTE: # apply the RoPE frequencies to the keys
        )
        # # output head
        # out = self.output_adapter(output_latents)
        return output_latents

    def output_head(self, output_latents: torch.Tensor):
        """
        This is the output head that converts the output latents to the final output.
        It is used in the forward pass to compute the final output.
        """
        return self.output_adapter(output_latents)

    def output_feat_head(self, output_latents: torch.Tensor):
        """
        This is the output head that converts the output feature latents to the final output.
        It is used in the forward pass to compute the final output.
        """
        return self.output_adapter_feat(output_latents)

    def forward(
        self,
        x: torch.Tensor,
        num_tokens=None,
        mat_dim=None,
        depth=None,
        token_loss=False,
        layer_loss=False,
        freq_cis=None,
        **kwargs
    ):
        B = x.shape[0]
        device = x.device

        # Embed input into patches
        patches = self.patch_embed(x)

        # Creates the process and output latents
        N = num_tokens if num_tokens else self.max_latent_tokens  # Number of tokens
        process_latents = self.process_latents.forward((B, N))
        output_latents = self.output_latents.forward(B)
        # output_latents_feat = self.output_latents_feat.forward(B)
        output_latents_feat = self.output_latents_feat.forward(patches.clone())

        # Compute the RoPE frequencies
        freq_cis = self.compute_freq_cis(N, device, freq_cis)

        # Compute block mask
        block_mask = self.compute_block_mask(
            num_tokens=N,
            device=device,
            mask_type=self.block_mask_type,
            token_grans=self.mask_token_grans,
        )

        # Read inputs into process latents
        process_latents = self.read(patches, process_latents, freq_cis)

        # When using layer loss, we will return intermediates.
        # This is needed for the early-exit loss.
        if layer_loss:
            process_latents, intermediates = self.forward_intermediates(
                process_latents=process_latents,
                freq_cis=freq_cis,
                block_mask=block_mask,
                mat_dim=mat_dim,
            )
        else:
            process_latents = self.forward_blocks(
                process_latents=process_latents,
                freq_cis=freq_cis,
                block_mask=block_mask,
                mat_dim=mat_dim,
                depth=depth,
            )

        # This is the final writeout step to get Logits
        final_writeout = self.write(
            process_latents=process_latents,
            output_latents=output_latents,
            freq_cis=freq_cis,
        )
        output = self.output_head(final_writeout)

        # This is the final writeout step to get feature tokens
        final_feat_writeout = self.write_feat(
            process_latents=process_latents,
            output_latents=output_latents_feat,
            freq_cis=freq_cis,
        )
        output_feats = self.output_feat_head(final_feat_writeout)

        # TODO: The current version does not induce a PoE, to make it do so
        # we must update the output latents per step
        # Layer loss: convert each intermediate output to the output space.
        intermediate_outs = []
        intermediate_feat_outs = []
        if layer_loss:
            for intermediate in intermediates:
                # NOTE: To turn this into PoE we must do:
                # int_writeout (l=0) <== self.write(process_latents [l=0], output_latents [l=0])
                # int_writeout (l=n) <== self.write(process_latents [l=n], int_writeout [l=n-1])

                # We randomly sample some # of tokens for the intermediate loss, this will make
                # depth adaptivity robust to changing the number of tokens.
                write_out_tokens = random.choice(self.mask_token_grans)
                int_writeouts = self.write(
                    process_latents=intermediate[:, :write_out_tokens],
                    output_latents=output_latents,
                    freq_cis=freq_cis[:write_out_tokens],
                )
                intermediate_outs.append(self.output_head(int_writeouts))

                int_feat_writeouts = self.write_feat(
                    process_latents=intermediate[:, :write_out_tokens],
                    output_latents=output_latents_feat,
                    freq_cis=freq_cis[:write_out_tokens],
                )
                intermediate_feat_outs.append(self.output_feat_head(int_feat_writeouts))

        # Token loss: grab each token granularity and write them out.
        # We require token grans to be specified.
        out_list = []
        out_feat_list = []
        token_loss = bool(token_loss)
        num_tokens = bool(num_tokens)
        if token_loss and self.mask_token_grans:
            # We don't compute the output of the last token granularity
            # This is because the readout of the last token granularity is computed earlier.
            for token_gran in self.mask_token_grans[:-1]:
                freq_cis_slice = freq_cis[:token_gran]
                token_writeout = self.write(
                    process_latents=process_latents[:, :token_gran],
                    output_latents=output_latents,
                    freq_cis=freq_cis_slice,
                )
                out_list.append(self.output_head(token_writeout))
                # Feature Readout
                feat_token_writeout = self.write_feat(
                    process_latents=process_latents[:, :token_gran],
                    output_latents=output_latents_feat,
                    freq_cis=freq_cis_slice,
                )
                out_feat_list.append(self.output_feat_head(feat_token_writeout))

        # Append the final output to the list
        out_list.append(output)
        out_feat_list.append(output_feats)
        return out_list, intermediate_outs, (out_feat_list, intermediate_feat_outs)
