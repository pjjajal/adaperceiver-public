import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import lecun_normal_
from einops import einsum


class LinearDepthHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_bins: int,
        min_depth: float = 0.001,
        max_depth: float = 10.0,
        upsample: int = 4,
        head_drop: float = 0.0,
    ):
        super().__init__()
        self.upsample = upsample
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_bins = num_bins
        self.dropout = nn.Dropout2d(head_drop) if head_drop > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(in_channels, num_bins, kernel_size=3, padding=1, stride=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                lecun_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats.contiguous()
        feats = F.interpolate(
            feats,
            size=[dim * self.upsample for dim in feats.shape[2:]],
            mode="bilinear",
            align_corners=False,
        )
        feats = self.dropout(feats)
        logits = self.conv_seg(feats)

        # This follows DINOv2 which follows AdaBin
        bins = torch.linspace(self.min_depth, self.max_depth, self.num_bins, device=feats.device)

        logits = F.relu(logits)
        eps = 0.1
        logits = logits + eps
        logits = logits / logits.sum(dim=1, keepdim=True)
        output = einsum(logits, bins, "i k m n, k -> i m n").unsqueeze(dim=1)
        return output
