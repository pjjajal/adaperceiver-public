from typing import Tuple, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear import MatLinear

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = MatLinear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = MatLinear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x, mat_dim=None):
        x = self.fc1(x, mat_dim=mat_dim)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x, mat_dim=mat_dim, mat_input=True)
        x = self.drop2(x)
        return x


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features=None,
        act_layer=nn.SiLU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = MatLinear(in_features, hidden_features, bias=bias)
        self.gate = MatLinear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = MatLinear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def init_weights(self):
        if self.fc1.bias is not None:
            nn.init.ones_(self.fc1.bias)
        nn.init.normal_(self.fc1.weight, std=1e-6)

    def forward(self, x, mat_dim=None):
        x_gate = self.gate(x, mat_dim=mat_dim)
        x = self.fc1(x, mat_dim=mat_dim)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.fc2(x, mat_dim=mat_dim, mat_input=True)
        x = self.drop2(x)
        return x