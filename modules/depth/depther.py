import torch
import torch.nn as nn
import torch.nn.functional as F
from .depth_head import LinearDepthHead
from einops import rearrange, einsum
from modules.networks.dense_pred import DensePredictionAdaPerceiver
from modules.layers.ffn import Mlp


class BaselineDepth(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_bins: int,
        head_drop: float = 0.0,
        cat_cls: bool = False,
        min_depth: float = 0.001,
        max_depth: float = 10.0,
        use_mlp: bool = False,
        upsample: int = 4,
    ):
        super().__init__()
        self.patch_size = backbone.patch_embed.patch_size[0]
        self.prefix_tokens = backbone.num_prefix_tokens
        self.backbone = backbone
        self.cat_cls = cat_cls
        in_channels = backbone.embed_dim * 2 if self.cat_cls else backbone.embed_dim
        self.head = LinearDepthHead(
            in_channels, num_bins, min_depth, max_depth, upsample, head_drop
        )
        self.mlp = (
            Mlp(backbone.embed_dim, backbone.embed_dim, drop=head_drop)
            if use_mlp
            else nn.Identity()
        )

        # Depth Binning Parameters
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_bins = num_bins

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Extract features
        feats = self.backbone.forward_features(img)
        # This is just for extra parameters to match AdaPerceiver structure
        feats = self.mlp(feats)
        # Grab the patch tokens
        patch_feats = feats[:, self.prefix_tokens :]
        # Concatenate pooled token if needed (CLS or equivalent)
        if self.cat_cls:
            cls_token = (
                self.backbone.pool(feats)
                .unsqueeze(1)
                .expand(-1, patch_feats.shape[1], -1)
            )
            patch_feats = torch.cat((cls_token, patch_feats), dim=-1)
        # Apply reshape to rectangle
        patch_feats = rearrange(
            patch_feats,
            "b (h w) c -> b c h w",
            h=img.shape[2] // self.patch_size,
            w=img.shape[3] // self.patch_size,
        )
        preds = self.head(patch_feats)

        if not self.training:
            preds = torch.clamp(preds, min=self.min_depth, max=self.max_depth)
        preds = F.interpolate(
            preds,
            size=img.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return preds


class AdaptiveDepth(nn.Module):
    def __init__(
        self,
        backbone: DensePredictionAdaPerceiver,
        num_bins: int,
        head_drop: float = 0.0,
        cat_cls: bool = False,
        min_depth: float = 0.001,
        max_depth: float = 10.0,
    ):
        super().__init__()
        self.patch_size = backbone.patch_size
        self.backbone = backbone
        self.cat_cls = cat_cls  # There is no cls token in AdaPerceiver but we just pool the features.
        in_channels = (
            backbone.output_feat_dim * 2 if cat_cls else backbone.output_feat_dim
        )
        self.head = LinearDepthHead(
            in_channels, num_bins, upsample=4, head_drop=head_drop
        )

        # Depth Binning Parameters
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_bins = num_bins

    def forward_head(
        self, img: torch.Tensor, feats: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        if self.cat_cls:
            pooled_feats = feats.mean(dim=1, keepdim=True)
            pooled_feats = pooled_feats.expand(-1, feats.shape[1], -1)
            feats = torch.cat((pooled_feats, feats), dim=-1)
        feats = rearrange(
            feats,
            "b (h w) c -> b c h w",
            h=H,
            w=W,
        )
        preds = self.head(feats)

        if not self.training:
            preds = torch.clamp(preds, min=self.min_depth, max=self.max_depth)
        preds = F.interpolate(
            preds,
            size=img.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return preds

    def forward(
        self,
        img: torch.Tensor,
        num_tokens=None,
        mat_dim=None,
        depth=None,
        token_loss=False,
        layer_loss=False,
        freq_cis=None,
        **kwargs
    ):
        out_list, intermediate_outs, (_, _) = self.backbone.forward(
            img,
            num_tokens=num_tokens,
            mat_dim=mat_dim,
            depth=depth,
            token_loss=token_loss,
            layer_loss=layer_loss,
            freq_cis=freq_cis,
        )

        H = img.shape[2] // self.patch_size
        W = img.shape[3] // self.patch_size

        out_list = [self.forward_head(img, out, H, W) for out in out_list]
        if layer_loss:
            intermediate_outs = [
                self.forward_head(img, out, H, W) for out in intermediate_outs
            ]
        return out_list, intermediate_outs
