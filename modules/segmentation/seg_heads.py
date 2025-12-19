import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d, lecun_normal_


class LinearSegHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, head_drop: float = 0.0):
        super().__init__()
        self.bn = LayerNorm2d(in_channels)
        self.dropout = nn.Dropout2d(head_drop) if head_drop > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                lecun_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, LayerNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        feats = feats.contiguous()
        x = self.bn(feats)
        x = self.dropout(x)
        x = self.conv_seg(x)
        x = F.interpolate(x, size=img.shape[-2:], mode="bilinear", align_corners=False)
        return x
