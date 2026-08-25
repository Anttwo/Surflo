# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PositionGetter:
    """Generates 2D spatial positions for patches in a grid.

    torch.compile-friendly: no dict cache, no .clone(), pure tensor ops.
    """

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        y_coords = torch.arange(height, device=device)
        x_coords = torch.arange(width, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        positions = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)
        return positions.unsqueeze(0).expand(batch_size, -1, -1)


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    torch.compile-friendly: cos/sin tables are computed inline for a fixed
    max_seq_len, avoiding dict caches and GPU→CPU syncs.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0
        max_seq_len: Pre-allocated table size; must be >= max position index + 1.
            Default 256 covers images up to ~3500 px at patch_size=14.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0, max_seq_len: int = 256):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.max_seq_len = max_seq_len

    def _compute_frequency_components(
        self, dim: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin tables for all positions up to max_seq_len.

        The tables are small (max_seq_len × dim) and recomputed each call so
        that torch.compile can trace pure tensor ops with no graph breaks.
        The compiler will constant-fold these when inputs are static.
        """
        exponents = torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        inv_freq = 1.0 / (self.base_frequency ** exponents)

        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq).to(dtype)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
            positions: Position tensor of shape (batch_size, n_tokens, 2).

        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.
        """
        feature_dim = tokens.size(-1) // 2

        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, tokens.device, tokens.dtype
        )

        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        return torch.cat((vertical_features, horizontal_features), dim=-1)
