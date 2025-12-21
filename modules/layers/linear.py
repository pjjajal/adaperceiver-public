import torch
import torch.nn as nn
import torch.nn.functional as F

class MatLinear(nn.Linear):
    # Use the one that matches your torch version:
    # @torch._dynamo.disable
    @torch.compiler.disable
    def create_col_mask(self, B, in_features, device, mat_dim_b):
        # mat_dim_b: (B,) long
        col_idx = torch.arange(in_features, device=device)           # (in,)
        col_mask = col_idx.unsqueeze(0) < mat_dim_b.unsqueeze(1)     # (B, in) bool
        return col_mask.unsqueeze(1)                                 # (B, 1, in)

    # @torch._dynamo.disable
    @torch.compiler.disable
    def create_row_mask(self, B, out_features, device, mat_dim_b):
        row_idx = torch.arange(out_features, device=device)          # (out,)
        row_mask = row_idx.unsqueeze(0) < mat_dim_b.unsqueeze(1)     # (B, out) bool
        return row_mask

    def _as_long_tensor(self, mat_dim, device):
        if isinstance(mat_dim, torch.Tensor):
            return mat_dim.to(device=device, dtype=torch.long)
        return torch.as_tensor(mat_dim, device=device, dtype=torch.long)

    def _forward_list(self, x, mat_dim, mat_input) -> torch.Tensor:
        """
        Per-batch matryoshka dims. x: (B, T, in)
        """
        B, T, in_features = x.shape
        out_features = self.out_features
        mat_dim_b = self._as_long_tensor(mat_dim, x.device)
        if mat_input:
            # Mask inputs per batch, then shared linear
            col_mask = self.create_col_mask(B, in_features, x.device, mat_dim_b)  # (B,1,in)
            x_masked = x * col_mask.to(x.dtype)                                    # (B,T,in)
            y = F.linear(x_masked, self.weight, self.bias)                         # (B,T,out)
            return y
        else:
            # Shared linear first, then mask outputs (and bias effect)
            y = F.linear(x, self.weight, self.bias)                                 # (B,T,out)
            row_mask = self.create_row_mask(B, out_features, x.device, mat_dim_b)   # (B,out) bool
            y = y * row_mask.unsqueeze(1).to(y.dtype)                               # (B,T,out)
            return y

    def forward(self, x, mat_dim=None, mat_input=False) -> torch.Tensor:
        """
        x: (B, T, in_features)
        mat_dim:
          - None: standard linear
          - int: global slice
          - list/tuple/tensor of length B: per-batch slice
        """
        if mat_dim is None:
            return super().forward(x)

        if isinstance(mat_dim, int):
            if self.training:
                # Zero masked portion to avoid structural changes during training
                mask = torch.ones_like(self.weight)
                if mat_input:
                    mask[:, mat_dim:] = 0
                    weight = self.weight * mask
                    bias = self.bias  # outputs unchanged
                else:
                    mask[mat_dim:] = 0
                    weight = self.weight * mask
                    bias = None if self.bias is None else self.bias * mask[:, 0]
                return F.linear(x, weight, bias)
            else:
                if mat_input:
                    weight = self.weight[:, :mat_dim]
                    bias = self.bias
                else:
                    weight = self.weight[:mat_dim]
                    bias = None if self.bias is None else self.bias[:mat_dim]
                return F.linear(x, weight, bias)

        # Per-batch case
        return self._forward_list(x, mat_dim, mat_input)