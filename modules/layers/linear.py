import torch
import torch.nn as nn
import torch.nn.functional as F


# A linear layer that supports dynamic slicing along the input or output dimension,
# useful for partial-dimension ("matryoshka") inference or training.
class _MatLinear(nn.Linear):

    @torch.compiler.disable
    def create_col_mask(self, x, mat_dim):
        # ---- Per-batch input masking: mask x, then standard shared-weight linear ----
        col_idx = torch.arange(self.in_features, device=x.device)  # (in_feat,)
        col_mask = col_idx.unsqueeze(0) < mat_dim.unsqueeze(
            1
        )  # (B, in_feat) booleans
        col_mask = col_mask.to(x.dtype).unsqueeze(
            1
        )  # (B, 1, in_feat) to broadcast over R
        return col_mask

    @torch.compiler.disable
    def create_row_mask(self, x, mat_dim):
        # ---- Per-batch output masking: mask weights/bias per batch ----
        row_idx = torch.arange(self.out_features, device=x.device)  # (out_feat,)
        row_mask = row_idx.unsqueeze(0) < mat_dim.unsqueeze(
            1
        )  # (B, out_feat) booleans
        return row_mask

    def handle_list(self, x, mat_dim=None, mat_input=False) -> torch.Tensor:
        mat_dim = torch.tensor(mat_dim, device=x.device)

        if mat_input:
            # # ---- Per-batch input masking: mask x, then standard shared-weight linear ----
            # col_idx = torch.arange(self.in_features, device=x.device)  # (in_feat,)
            # col_mask = col_idx.unsqueeze(0) < mat_dim.unsqueeze(
            #     1
            # )  # (B, in_feat) booleans
            # col_mask = col_mask.to(x.dtype).unsqueeze(
            #     1
            # )  # (B, 1, in_feat) to broadcast over R
            col_mask = self.create_col_mask(x, mat_dim)

            weight = self.weight.unsqueeze(0) * col_mask  # (B, out_feat, in_feat)

            out = (
                x @ weight.mT
            )  # (B, T, in_feat) @ (B, in_feat, out_feat) -> (B, T, out_feat)
            if self.bias is not None:
                out = out + self.bias.view(1, 1, -1)
            return out
        else:
            # # ---- Per-batch output masking: mask weights/bias per batch ----
            # row_idx = torch.arange(
            #     self.out_features, device=self.weight.device
            # )  # (out_feat,)
            # row_mask = row_idx.unsqueeze(0) < mat_dim.unsqueeze(
            #     1
            # )  # (B, out_feat) booleans
            row_mask = self.create_row_mask(x, mat_dim)
            # Build (B, out_feat, in_feat) mask to apply to weights
            wmask = row_mask.unsqueeze(-1).to(self.weight.dtype)  # (B, out_feat, 1)
            weights = self.weight.unsqueeze(0) * wmask  # (B, out_feat, in_feat)

            # Batched linear: (B, R, in_feat) · (B, in_feat, out_feat) -> (B, R, out_feat)
            y = x @ weights.mT  # (B, R, out_feat)

            # Per-batch bias mask and add
            if self.bias is not None:
                bmask = row_mask.to(self.bias.dtype)  # (B, out_feat)
                b = (self.bias.unsqueeze(0) * bmask).unsqueeze(
                    1
                )  # (B, 1, out_feat)
                y = y + b
            return y


    def forward(self, x, mat_dim=None, mat_input=False) -> torch.Tensor:
        # If no slicing dimension is specified, perform a standard linear transformation.
        if mat_dim is None:
            return super().forward(x)

        if isinstance(mat_dim, int):
            if self.training:
                # print("Warning: MatLinear is in training mode. Slicing will not be applied during training.")
                # During training, we do not slice the weight matrix.
                # Instead we just zero out the "sliced" part of the weight matrix.
                mask = torch.ones_like(
                    self.weight, dtype=self.weight.dtype, device=self.weight.device
                )
                if mat_input:
                    mask[:, mat_dim:] = 0
                else:
                    mask[mat_dim:] = 0
                weight = self.weight * mask
                bias = None
                if self.bias is not None:
                    bias = self.bias if mat_input else self.bias * mask[:, 0]
                return F.linear(x, weight, bias)
            else:
                # Slice the weight matrix based on whether we're slicing input or output.
                #  If mat_input is True, slice the weight along the input dimension.
                weight = (
                    self.weight[:, :mat_dim] if mat_input else self.weight[:mat_dim]
                )

                # Conditionally slice the bias if it exists.
                bias = None
                if self.bias is not None:
                    bias = self.bias if mat_input else self.bias[:mat_dim]

                # Perform the linear transformation with the sliced weight and bias.
                return F.linear(x, weight, bias)

        if isinstance(mat_dim, (list, tuple)):
            return self.handle_list(x, mat_dim, mat_input)
            # mat_dim = torch.tensor(mat_dim, device=x.device)

            # if mat_input:
            #     # ---- Per-batch input masking: mask x, then standard shared-weight linear ----
            #     col_idx = torch.arange(self.in_features, device=x.device)  # (in_feat,)
            #     col_mask = col_idx.unsqueeze(0) < mat_dim.unsqueeze(
            #         1
            #     )  # (B, in_feat) booleans
            #     col_mask = col_mask.to(x.dtype).unsqueeze(
            #         1
            #     )  # (B, 1, in_feat) to broadcast over R

            #     weight = self.weight.unsqueeze(0) * col_mask  # (B, out_feat, in_feat)

            #     out = (
            #         x @ weight.mT
            #     )  # (B, T, in_feat) @ (B, in_feat, out_feat) -> (B, T, out_feat)
            #     if self.bias is not None:
            #         out = out + self.bias.view(1, 1, -1)
            #     return out
            # else:
            #     # ---- Per-batch output masking: mask weights/bias per batch ----
            #     row_idx = torch.arange(
            #         self.out_features, device=self.weight.device
            #     )  # (out_feat,)
            #     row_mask = row_idx.unsqueeze(0) < mat_dim.unsqueeze(
            #         1
            #     )  # (B, out_feat) booleans
            #     # Build (B, out_feat, in_feat) mask to apply to weights
            #     wmask = row_mask.unsqueeze(-1).to(self.weight.dtype)  # (B, out_feat, 1)
            #     weights = self.weight.unsqueeze(0) * wmask  # (B, out_feat, in_feat)

            #     # Batched linear: (B, R, in_feat) · (B, in_feat, out_feat) -> (B, R, out_feat)
            #     y = x @ weights.mT  # (B, R, out_feat)

            #     # Per-batch bias mask and add
            #     if self.bias is not None:
            #         bmask = row_mask.to(self.bias.dtype)  # (B, out_feat)
            #         b = (self.bias.unsqueeze(0) * bmask).unsqueeze(
            #             1
            #         )  # (B, 1, out_feat)
            #         y = y + b
            #     return y


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