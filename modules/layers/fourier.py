import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

class FourierFeatures(nn.Module):
    def __init__(self, dims: int, max_freq: float, num_bands: int):
        super().__init__()
        self.dims = dims
        self.max_freq = max_freq
        self.num_bands = num_bands

        # Create frequency bands
        self.freq_bands = torch.linspace(1.0, max_freq, num_bands)

    @property
    def out_dim(self):
        return int(self.dims * (2 * self.num_bands + 1))

    def compute_meshgrid(self, spatial_dims: list[int], device) -> torch.Tensor:
        """Compute a meshgrid of coordinates in the range [-1, 1] for each spatial dimension."""
        coords = [
            torch.linspace(0.0, 1.0, steps=size, device=device)
            for size in spatial_dims
        ]
        mesh = torch.meshgrid(*coords) # (spatial_dims[0], spatial_dims[1], ...)
        grid = torch.stack(mesh, dim=-1) # (spatial_dims[0], spatial_dims[1], ..., len(spatial_dims))
        return grid * 2 - 1 # cast into [-1, 1]

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build Fourier positional features and concatenate them to the input feature map
        # x is shaped (B, C, *spatial_dims); e.g., for images: (batch, channels, H, W)
        B, C, *spatial_dims = x.shape
        # Ensure the tensor has the expected number of spatial dimensions (e.g., 2 for images, 3 for volumes)
        assert (
            len(spatial_dims) == self.dims
        ), "Input tensor must have the same number of 'spatial' dimensions as the FourierFeatures layer."

        self.freq_bands = self.freq_bands.to(device=x.device)
        # Create a normalized coordinate grid in [-1, 1] with shape (*spatial_dims, dims)
        grid = self.compute_meshgrid(spatial_dims, device=x.device) # (spatial_dims[0], spatial_dims[1], ..., len(spatial_dims))
        # Flatten the spatial grid to a 2D tensor of positions: (num_positions, dims)
        flattened_grid = grid.view(-1, self.dims)
        # For each position and each dimension, multiply coordinates by each frequency band -> (num_positions, dims, num_bands)
        theta = einsum(self.freq_bands, flattened_grid, "f, g dims -> g dims f") 
        # Evaluate sinusoidal bases at those frequency-scaled coordinates
        sin_basis = torch.sin(theta * torch.pi)
        # sin_basis and cos_basis each have shape (num_positions, dims, num_bands)
        cos_basis = torch.cos(theta * torch.pi)
        # Also keep the raw coordinates
        flattened_grid = flattened_grid.unsqueeze(-1)
        # Concatenate along the last dim: [sin, cos, coords] -> (num_positions, dims, 2 * num_bands + 1)
        fourier_features = torch.cat([sin_basis, cos_basis, flattened_grid], dim=-1)
        # Reshape features back to the spatial grid: (*spatial_dims, dims * (2 * num_bands + 1))
        fourier_features = fourier_features.view(*spatial_dims, -1)
        # Move the last dimension to channel-first format expected by conv-style tensors: (C_ff, *spatial_dims)
        fourier_features = rearrange(fourier_features, "... c -> c ...")
        # Broadcast the Fourier features over the batch dimension: (B, C_ff, *spatial_dims)
        fourier_features = fourier_features.unsqueeze(0).expand(B, -1, *spatial_dims)
        # Concatenate original channels with Fourier features along the channel axis
        x = torch.cat([x, fourier_features], dim=1)
        return x
