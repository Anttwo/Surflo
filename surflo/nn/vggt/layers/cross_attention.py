# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, y: Tensor, attn_mask=None, pos=None) -> Tensor:
        B, N, C = x.shape
        _, M, _ = y.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, h, N, d)
        kv = self.kv(y).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (2, B, h, M, d)
        k, v = kv.unbind(0)  # (B, h, M, d)

        q, k = self.q_norm(q), self.k_norm(k)  # (B, h, N, d) and (B, h, M, d)

        if self.rope is not None:
            q = self.rope(q, pos)  # (B, h, N, d)
            k = self.rope(k, pos)  # (B, h, M, d)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0)  # (B, h, N, d)
        else:
            q = q * self.scale  # (B, h, N, d)
            attn = q @ k.transpose(-2, -1)  # (B, h, N, M)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)  # (B, h, N, M)
            attn = self.attn_drop(attn)  # (B, h, N, M)
            x = attn @ v  # (B, h, N, d)

        x = x.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)  # (B, N, C)
        x = self.proj_drop(x)  # (B, N, C)
        return x


class MemEffCrossAttention(CrossAttention):
    def forward(self, x: Tensor, y: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        _, M, _ = y.shape
        head_dim = C // self.num_heads
        
        q = self.q(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)  # (B, h, N, d)
        kv = self.kv(y).reshape(B, M, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)  # (2, B, h, M, d)
        k, v = kv.unbind(0)  # (B, h, M, d)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
